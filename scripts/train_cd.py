#!/usr/bin/env python
"""Train the Siamese change detector on LEVIR-CD or OSCD, on ROCm.

Master.md §8 Phase 5 asks for two runs, not one: the same architecture trained at
0.5 m on LEVIR-CD and at 10 m on OSCD, with the gap between them recorded. That
gap is the Cartosat-transfer evidence — it is the only number we will have that
speaks to how a detector trained at one resolution behaves at another, and it is
worth more to a judge than either F1 on its own. Run it twice::

    uv run python scripts/train_cd.py --dataset levircd --root data/raw --download
    uv run python scripts/train_cd.py --dataset oscd    --root data/raw --download
    uv run python scripts/train_cd.py --compare

Downloads never happen implicitly. ``--download`` is required to fetch an
archive, so importing this module, running ``--help``, or constructing a
datamodule in a test cannot pull a dataset.

ROCm notes: ``--precision bf16-mixed`` is the default and the right one on
gfx1100. fp16 overflows in the deep ResNet stages on this architecture; bf16's
exponent range is the entire reason it does not. Everything else is device-neutral
— Lightning's ``accelerator: auto`` finds the HIP device through the CUDA API.

The run writes a :class:`~satquery.training.cd.checkpoint.CheckpointBundle`, not a
bare ``state_dict``: the calibrated threshold, the normalisation and the training
resolution travel with the weights, because serving without any of the three
produces a plausible mask that is quietly wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.training.cd.checkpoint import (  # noqa: E402
    BUNDLE_SUFFIX,
    CheckpointBundle,
    Normalisation,
    default_checkpoint_path,
    load_bundle,
)
from satquery.training.cd.data import (  # noqa: E402
    DATASET_NAMES,
    DEFAULT_BATCH,
    DEFAULT_PATCH_PX,
    ChangeDataModule,
    DataError,
    channel_statistics,
)
from satquery.training.cd.model import SiameseConfig  # noqa: E402
from satquery.training.cd.module import build_task  # noqa: E402

F1_GATE = 0.88
"""Master.md §8 Phase 5's acceptance gate on the LEVIR-CD test split. The script
reports whether the run cleared it; it does not pretend a run that did not is
usable, and it does not fail the process either — a first run below the gate is
information, not an error."""


LAST_CHECKPOINT: Final[str] = "last.ckpt"
"""The rolling checkpoint an interrupted run resumes from.

Distinct from the ``val/f1`` best-so-far checkpoint, and needed alongside it: the
best one is only rewritten when the metric improves, so a crash ten epochs after
the last improvement would restart from ten epochs ago. This one is the *most
recent* state, which is what "carry on where you left off" means."""

DEFAULT_SAVE_EVERY_N_STEPS: Final[int] = 200
"""How often ``last.ckpt`` is refreshed inside an epoch.

Epoch-end saving alone is only as fine-grained as an epoch is short. LEVIR-CD's
is 55 steps, so this never fires there and epoch boundaries do the work; a larger
corpus, where one epoch is hours, gets a bound on how much a crash can cost."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=list(DATASET_NAMES), default="levircd")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch the archive if absent. Required; nothing downloads implicitly.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_PX)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=0.1,
        help="The pretrained encoder fine-tunes this much slower than the decoder.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Start from random weights instead of SSL4EO-S12. For the ablation.",
    )
    parser.add_argument(
        "--precision",
        default="bf16-mixed",
        help="bf16-mixed on ROCm. fp16 overflows in the deep ResNet stages.",
    )
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Checkpoint destination. Defaults to {default_checkpoint_path()}/.",
    )
    parser.add_argument(
        "--limit-batches",
        type=float,
        default=1.0,
        help="Fraction of each epoch to run. For a smoke test of the wiring.",
    )
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto | none | a path to a .ckpt. 'auto' continues from last.ckpt "
        "when one is present in the run directory and starts fresh when it is "
        "not, so an interrupted run restarts with the same command line.",
    )
    parser.add_argument(
        "--save-every-n-steps",
        type=int,
        default=DEFAULT_SAVE_EVERY_N_STEPS,
        help="Refresh last.ckpt this often *within* an epoch. 0 saves only at "
        "epoch end, which on a long epoch is a long way to fall back.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Do not train. Print the cross-resolution table from existing bundles.",
    )
    return parser.parse_args(argv)


def resume_checkpoint(run_dir: Path, setting: str) -> Path | None:
    """Resolve ``--resume`` to a checkpoint path, or None to start fresh.

    Raises:
        SystemExit: An explicit path was given and does not exist. A typo must
            not silently restart a three-day run from step zero.
    """
    value = setting.strip().lower()
    if value in {"", "none", "no", "false"}:
        return None
    if value in {"auto", "true", "yes"}:
        candidate = run_dir / LAST_CHECKPOINT
        return candidate if candidate.is_file() else None
    path = Path(setting)
    if not path.is_file():
        raise SystemExit(
            f"--resume {setting} does not exist. Available in {run_dir}: "
            f"{sorted(c.name for c in run_dir.glob('*.ckpt')) or 'none'}"
        )
    return path


def cross_resolution_table(directory: Path) -> str:
    """Summarise every trained bundle as the resolution study Phase 5 asks for."""
    bundles: list[CheckpointBundle] = []
    for path in sorted(directory.glob(f"*{BUNDLE_SUFFIX}")):
        try:
            bundles.append(load_bundle(path))
        except Exception as error:  # noqa: BLE001 - a bad bundle is skipped, not fatal
            print(f"  skipped {path.name}: {error}", file=sys.stderr)

    if not bundles:
        return f"No checkpoints in {directory}. Train at least one first."

    header = f"{'dataset':<12}{'gsd_m':>8}{'F1':>8}{'IoU':>8}{'thr':>7}  encoder"
    rows = [header, "-" * len(header)]
    for bundle in sorted(bundles, key=lambda b: (b.gsd_m or 0.0)):
        rows.append(
            f"{bundle.dataset:<12}"
            f"{(bundle.gsd_m or 0.0):>8.2f}"
            f"{bundle.metrics.get('f1', 0.0):>8.4f}"
            f"{bundle.metrics.get('iou', 0.0):>8.4f}"
            f"{bundle.threshold:>7.2f}  {bundle.encoder_source}"
        )

    scored = [b for b in bundles if b.gsd_m and b.metrics.get("f1")]
    if len(scored) >= 2:
        fine = min(scored, key=lambda b: b.gsd_m or 0.0)
        coarse = max(scored, key=lambda b: b.gsd_m or 0.0)
        delta = coarse.metrics["f1"] - fine.metrics["f1"]
        rows += [
            "",
            f"Cross-resolution delta ({fine.gsd_m:g} m -> {coarse.gsd_m:g} m): "
            f"{delta:+.4f} F1.",
            "This is the Cartosat-transfer evidence: it says what happens to this",
            "architecture when the ground sample distance moves, measured rather",
            "than assumed.",
        ]
    return "\n".join(rows)


def build_self_test(
    model: object, loader: object, threshold: float, size: int = 256
) -> dict[str, object] | None:
    """Record what this checkpoint answers on one real pair, for the load check.

    Serving replays this after every load. It is measured here, on the model that
    has just finished training, because that is the only moment the weights are
    known-good — and it must be a pair that genuinely differs, since the
    architecture subtracts the two epochs and identical inputs would exercise
    nothing but the decoder's biases. See ``CheckpointBundle.self_test``.
    """
    import numpy as np
    import torch

    try:
        batch = next(iter(loader))
    except (StopIteration, TypeError):  # pragma: no cover - loader-dependent
        return None

    pre = batch["image1"][0][:, :size, :size]
    post = batch["image2"][0][:, :size, :size]
    if pre.shape[-1] < size or pre.shape[-2] < size:  # pragma: no cover - tiny patches
        return None

    device = next(model.parameters()).device
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(pre.unsqueeze(0).to(device), post.unsqueeze(0).to(device))
        )[0, 0]
    return {
        "pre": pre.cpu().numpy().astype(np.uint8),
        "post": post.cpu().numpy().astype(np.uint8),
        "changed_fraction": round(float((probabilities >= threshold).float().mean()), 6),
        "tolerance": 0.02,
        "source": "first validation pair, cropped",
    }


def train(args: argparse.Namespace) -> int:
    """Run one training job and write its checkpoint bundle."""
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    pl.seed_everything(args.seed, workers=True)

    data = ChangeDataModule(
        dataset=args.dataset,
        root=args.root,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        download=args.download,
    )
    data.setup("fit")

    config = SiameseConfig(
        backbone=args.backbone,
        bands=data.profile.bands,
        pretrained=not args.no_pretrained,
    )
    task = build_task(
        config=config,
        learning_rate=args.learning_rate,
        encoder_lr_scale=args.encoder_lr_scale,
        max_epochs=args.epochs,
    )
    print(f"encoder: {task.model.encoder_report}", file=sys.stderr)

    output = args.out or default_checkpoint_path()
    output.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.dataset}_{args.backbone}"

    # Pinned, not left to the logger. Lightning's default puts checkpoints under
    # <logger>/<name>/version_N/checkpoints, and N increments on every launch —
    # so the run that resumes would write to version_1 while last.ckpt sat in
    # version_0, and "auto" would find nothing. A fixed directory per run name is
    # what makes the resume path stable across restarts.
    ckpt_dir = output / "runs" / run_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        default_root_dir=str(output / "runs"),
        logger=TensorBoardLogger(str(output / "runs"), name=run_name),
        limit_train_batches=args.limit_batches,
        limit_val_batches=args.limit_batches,
        callbacks=[
            # Selected on F1 over the changed class, never on loss: the loss is
            # minimised acceptably by predicting no change anywhere.
            #
            # save_last writes last.ckpt beside it on every save. Lightning puts
            # the optimiser and LR-scheduler states, the epoch and the global step
            # into every .ckpt, so this is a resumable position and not just
            # weights — which is the whole difference between restarting and
            # carrying on.
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                monitor="val/f1",
                mode="max",
                save_top_k=1,
                filename=run_name,
            ),
            # The rolling position, and the *only* writer of last.ckpt. Two
            # callbacks with save_last=True in one directory do not both write
            # it — Lightning renames the second to last-v1.ckpt, leaving which
            # file holds the newer state a matter of callback ordering. One
            # writer, one name, no ambiguity for --resume to get wrong.
            #
            # save_on_train_epoch_end covers the epoch boundary whatever the step
            # cadence does, so a corpus whose epoch is shorter than
            # --save-every-n-steps is still checkpointed once per epoch.
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                save_top_k=0,
                save_last=True,
                every_n_train_steps=args.save_every_n_steps or None,
                save_on_train_epoch_end=True,
            ),
            EarlyStopping(monitor="val/f1", mode="max", patience=10),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        deterministic=False,
    )

    resume_from = resume_checkpoint(ckpt_dir, args.resume)
    if resume_from is not None:
        print(f"resuming from {resume_from}", file=sys.stderr)
    else:
        print(f"starting from scratch (--resume {args.resume})", file=sys.stderr)
    trainer.fit(
        task,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
        # Lightning restores the epoch, the global step, the optimiser and the
        # schedulers from here; None means a fresh run.
        ckpt_path=str(resume_from) if resume_from else None,
    )

    # The bundle's normalisation is a record of what the *forward pass* did, not
    # of what the data looked like. ChangeDetectionTask feeds batch["image1"]
    # into the model untouched, so the answer is "nothing", and identity is the
    # only honest entry — serving replays this record verbatim, and a set of
    # statistics that training never applied is a training/serving skew that
    # shows up as a detector confidently predicting no change anywhere.
    #
    # The measured distribution is still worth keeping, so it travels as
    # provenance in the notes, where nothing will multiply by it.
    normalisation = Normalisation.identity(config.in_channels)
    try:
        mean, std = channel_statistics(data.train_dataloader())
        measured = (
            f"; channel mean {tuple(round(v, 3) for v in mean)}"
            f", std {tuple(round(v, 3) for v in std)}"
        )
    except DataError:
        measured = ""

    metrics = {
        "f1": float(task.best_f1),
        "iou": float(trainer.callback_metrics.get("val/iou", torch.tensor(0.0))),
        "precision": float(trainer.callback_metrics.get("val/precision", torch.tensor(0.0))),
        "recall": float(trainer.callback_metrics.get("val/recall", torch.tensor(0.0))),
        "f1_at_0.5": float(trainer.callback_metrics.get("val/f1_at_0.5", torch.tensor(0.0))),
    }

    bundle = CheckpointBundle(
        config=config.to_dict(),
        state_dict={k: v.cpu() for k, v in task.model.state_dict().items()},
        normalisation=normalisation,
        threshold=float(task.best_threshold),
        dataset=args.dataset,
        gsd_m=data.profile.gsd_m,
        metrics=metrics,
        self_test=build_self_test(
            task.model, data.val_dataloader(), float(task.best_threshold)
        ),
        encoder_source=task.model.encoder_report.source,
        trained_at=datetime.now(UTC).isoformat(),
        notes=(
            f"{args.epochs} epochs at {args.patch_size}px, precision {args.precision}"
            f"; trained on unnormalised loader output{measured}"
        ),
    )
    from satquery.training.cd.checkpoint import save_bundle

    path = save_bundle(bundle, output / f"{run_name}{BUNDLE_SUFFIX}")

    print(json.dumps({"checkpoint": str(path), **metrics, "threshold": bundle.threshold}, indent=2))
    if metrics["f1"] >= F1_GATE:
        print(f"\nF1 {metrics['f1']:.4f} clears the {F1_GATE} Phase 5 gate.")
    else:
        print(
            f"\nF1 {metrics['f1']:.4f} is BELOW the {F1_GATE} Phase 5 gate. The "
            f"checkpoint is written and servable, but no result table may quote "
            f"it as meeting the gate."
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Train, or print the cross-resolution comparison."""
    args = parse_args(argv)
    if args.compare:
        print(cross_resolution_table(args.out or default_checkpoint_path()))
        return 0
    try:
        return train(args)
    except ImportError as error:
        print(
            f"{error}\n\nTraining needs the optional stack: uv sync --extra cd",
            file=sys.stderr,
        )
        return 2
    except DataError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
