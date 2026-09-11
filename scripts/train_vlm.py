#!/usr/bin/env python
"""Run the Phase 7 QLoRA adaptation of Qwen3-VL-8B on ROCm.

Two commands, in this order, every time::

    uv run python scripts/train_vlm.py --sanity-check
    uv run python scripts/train_vlm.py --output-dir runs/sq-lora-v1

The sanity check truncates the corpus to 100 samples and one epoch. It is not a
formality: it walks the whole path — dataloader, processor, 4-bit forward, LoRA
backward through both towers, optimiser step, checkpoint write — in minutes, and
it is the only cheap way to find out whether bitsandbytes and ROCm agree on this
box today. The failure it is really insurance against is discovering a corpus or
label bug (DATA_ADAPTATION_PLAN §2.5) eleven hours into an overnight run.

Profile B (``--config configs/train/lora_qwen3vl4b_bf16.yaml``) is the fallback
when NF4 misbehaves on gfx1100. Nothing in this script is 8B- or 4-bit-specific.

The environment matters as much as the flags. ``HIP_VISIBLE_DEVICES=0`` comes
from the profile and is exported *before* torch is imported, because the
7800X3D's iGPU enumerates alongside the 7900 XTX and torch decides which device
is 0 at import time.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.models.loader import VRAM_BUDGET_BYTES, VramGuardError  # noqa: E402
from satquery.training.vlm.qlora import (  # noqa: E402
    DEFAULT_PROFILE_PATH,
    SANITY_SAMPLES,
    InterruptRequest,
    ProfileError,
    QLoraProfile,
    RunPlan,
    VramEstimate,
    build_peft_model,
    build_trainer,
    checkpoint_step,
    describe_resume,
    estimate_footprint,
    guard_budget,
    guard_images_present,
    install_interrupt_handler,
    latest_checkpoint,
    load_corpus,
    load_model_and_processor,
    load_profile,
    plan_run,
    resolve_target_modules,
    resume_target,
    split_targets,
    trainable_parameter_report,
    write_run_manifest,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help=f"Truncate to {SANITY_SAMPLES} samples for 1 epoch, to verify ROCm "
        "memory and gradient flow before committing a night to a run.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None, help="Override data.corpus.")
    parser.add_argument("--val", type=Path, default=None, help="Override data.val.")
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Cap the validation samples an eval pass reads (0 or 'all' for the "
        "whole file). Overrides the profile; an unbounded eval can outlast the "
        "training it reports on.",
    )
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Stop after this many optimiser steps."
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="auto | true | false | a checkpoint path. Defaults to the profile's "
        "'auto', which resumes when a checkpoint is present and starts fresh "
        "when it is not — so the same command line restarts an interrupted run.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Write a resumable checkpoint every N optimiser steps. Overrides the "
        "profile; lower it when the run has to survive being killed.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="How many checkpoints to keep on disk. The newest is always kept.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the profile, the run plan and the VRAM estimate, then stop "
        "without loading any weights.",
    )
    parser.add_argument(
        "--skip-vram-guard",
        action="store_true",
        help="Proceed even when the estimate exceeds the 22 GB budget. For "
        "deliberately exploring the ceiling, and nothing else.",
    )
    return parser.parse_args(argv)


def resolve_profile(args: argparse.Namespace) -> QLoraProfile:
    """Load the profile and apply the command line's overrides to it."""
    profile = load_profile(args.config)
    if args.corpus is not None:
        profile.data.corpus = args.corpus
    if args.val is not None:
        profile.data.val = args.val
    if args.resume is not None:
        profile.train.resume_from_checkpoint = args.resume
    if args.max_eval_samples is not None:
        profile.data.eval_samples = args.max_eval_samples or None
    if args.save_steps is not None:
        profile.train.save_steps = args.save_steps
    if args.save_total_limit is not None:
        profile.train.save_total_limit = args.save_total_limit
    if args.sanity_check:
        profile.run_name = f"{profile.run_name}-sanity"
        profile.train.num_train_epochs = 1
        profile.train.save_steps = min(profile.train.save_steps, 5)
        profile.train.eval_steps = min(profile.train.eval_steps, 5)
        profile.train.logging_steps = 1
    return profile


def output_dir_for(args: argparse.Namespace, profile: QLoraProfile) -> Path:
    """Where checkpoints, the adapter and the manifest are written."""
    if args.output_dir is not None:
        return Path(args.output_dir)
    return Path("runs") / profile.run_name


def report_plan(
    profile: QLoraProfile,
    plan: RunPlan,
    estimate: VramEstimate,
    steps: int | None = None,
    eval_samples: int = 0,
) -> None:
    """Print what this run will do before it does any of it."""
    print(f"profile      : {profile.run_name} ({profile.base_model})")
    print(f"quantisation : {'nf4 4-bit' if profile.quantization.load_in_4bit else 'bf16'}")
    print(f"plan         : {plan.describe()}")
    print(
        f"vram estimate: {estimate.total_gib:.1f} GiB of "
        f"{VRAM_BUDGET_BYTES / 1024**3:.0f} GiB budget "
        f"({estimate.adapter_params / 1e6:.1f} M trainable adapter params)"
    )
    # Against the steps this invocation will actually run, which --max-steps can
    # cut well below the plan's. Reporting the plan's count here told an operator
    # to expect 381 checkpoints from a 10-step run.
    running = steps if steps is not None else plan.steps
    print(
        f"eval         : {eval_samples} samples every {profile.train.eval_steps} steps"
    )
    print(
        f"checkpoints  : every {profile.train.save_steps} steps, keeping "
        f"{profile.train.save_total_limit} "
        f"({running // max(1, profile.train.save_steps)} scheduled over {running} steps)"
    )


def peak_memory_gib() -> float | None:
    """Peak allocated device memory this process reached, in GiB."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is present wherever this runs
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024**3


def main(argv: Sequence[str] | None = None) -> int:
    """Run the adaptation. Returns a process exit code."""
    args = parse_args(argv)
    profile = resolve_profile(args)
    profile.apply_env()  # Must precede the first torch import.
    interrupt = InterruptRequest()

    limit = SANITY_SAMPLES if args.sanity_check else None
    train_records = load_corpus(
        profile.data.corpus,
        limit=limit,
        max_views=profile.data.max_views_per_sample,
        image_root=args.image_root,
    )
    eval_records = (
        load_corpus(
            profile.data.val,
            limit=16 if args.sanity_check else profile.data.eval_samples,
            max_views=profile.data.max_views_per_sample,
            image_root=args.image_root,
        )
        if profile.data.val.is_file()
        else []
    )
    # Before the 17 GB of weights, not after: a corpus pointing at unrendered
    # views otherwise fails inside the dataloader, hours in and far from the cause.
    try:
        guard_images_present(train_records, str(profile.data.corpus))
        if eval_records:
            guard_images_present(eval_records, str(profile.data.val))
    except ProfileError as error:
        # A refusal, not a crash: the operator has a corpus to rebuild, and a
        # traceback through PIL is the least useful way to be told so.
        print(f"refusing to start: {error}", file=sys.stderr)
        return 4

    plan = plan_run(profile, len(train_records), sanity_check=args.sanity_check)
    estimate = estimate_footprint(profile)
    running_steps = args.max_steps if args.max_steps is not None else plan.steps
    report_plan(
        profile, plan, estimate, steps=running_steps, eval_samples=len(eval_records)
    )

    if not args.skip_vram_guard:
        try:
            guard_budget(estimate)
        except VramGuardError as error:
            print(f"refusing to start: {error}", file=sys.stderr)
            print(
                "turn the §6.3 knobs in order: max_views_per_sample 6->4, "
                "max_pixels 448²->392², max_seq_len 4096->3072, then Profile B.",
                file=sys.stderr,
            )
            return 2

    if args.dry_run:
        print("dry run: no weights loaded.")
        return 0

    from datasets import Dataset, Image

    def as_dataset(records: list[dict[str, Any]]) -> Dataset:
        """Build a vision dataset whose image paths decode lazily.

        The ``images`` column arrives as file paths and is cast to the ``Image``
        feature rather than loaded here: a 65 k-sample corpus at six 448 px views
        apiece is far more pixels than this process should hold, and the cast
        defers each decode to the batch that needs it. The column's presence is
        also what makes ``SFTTrainer`` treat the dataset as vision at all.
        """
        dataset = Dataset.from_list(records)
        return dataset.cast_column("images", [Image()])

    train_dataset = as_dataset(train_records)
    eval_dataset = as_dataset(eval_records) if eval_records else None

    model, processor = load_model_and_processor(profile)
    targets = resolve_target_modules(model, profile.lora)
    llm_targets, vision_targets = split_targets(targets)
    print(f"lora targets : {len(llm_targets)} llm, {len(vision_targets)} vision")
    if not vision_targets:
        print(
            "refusing to start: no vision-encoder modules matched. Adapting the "
            "language model alone cannot teach a colormapped index view.",
            file=sys.stderr,
        )
        return 3

    model = build_peft_model(model, profile)
    counts = trainable_parameter_report(model)
    print(
        f"parameters   : {counts['trainable'] / 1e6:.1f} M trainable of "
        f"{counts['total'] / 1e9:.2f} B ({counts['trainable_vision'] / 1e6:.1f} M in "
        "the vision tower)"
    )

    output_dir = output_dir_for(args, profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(
        profile=profile,
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        interrupt=interrupt,
        max_steps=args.max_steps if args.max_steps is not None else (
            plan.steps if args.sanity_check else None
        ),
        total_steps=plan.steps,
    )
    print(f"resume       : {describe_resume(output_dir, profile.train.resume_from_checkpoint)}")
    print(
        "interrupt    : Ctrl+C once checkpoints at the next optimiser step and "
        "stops cleanly; twice aborts immediately."
    )
    restore_handlers = install_interrupt_handler(interrupt)
    try:
        result = trainer.train(
            resume_from_checkpoint=resume_target(
                output_dir, profile.train.resume_from_checkpoint
            )
        )
    finally:
        restore_handlers()

    # Reached on a graceful stop as well as on a completed run: the callback
    # ends training through the trainer's own control flow, so `train()`
    # returns normally and the adapter and manifest below still get written.
    stopped_early = interrupt.requested

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    write_run_manifest(
        output_dir / "run_manifest.json",
        {
            "run_name": profile.run_name,
            "finished_at": datetime.now(UTC).isoformat(),
            "profile": profile.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "vram_estimate": estimate.model_dump(mode="json"),
            "peak_vram_gib": peak_memory_gib(),
            "targets": {"llm": llm_targets, "vision": vision_targets},
            "trainable_parameters": counts,
            "metrics": getattr(result, "metrics", {}),
            "adapter": str(adapter_dir),
            "interrupted": stopped_early,
            "interrupted_by": interrupt.signal_name,
            "last_checkpoint": str(last) if (last := latest_checkpoint(output_dir)) else None,
        },
    )
    print(f"adapter      : {adapter_dir}")
    print(f"peak vram    : {peak_memory_gib()} GiB")
    print(json.dumps(getattr(result, "metrics", {}), indent=2, default=str))

    if stopped_early:
        # The one thing an operator needs at this moment is the command that
        # picks the run back up, not a stack trace and a guess.
        resume_from = latest_checkpoint(output_dir)
        where = (
            f"checkpoint-{checkpoint_step(resume_from)}" if resume_from else "no checkpoint"
        )
        print()
        print(f"stopped on {interrupt.signal_name} at {where}. Resume with:")
        print(f"  {resume_command(args, output_dir)}")
        return 130
    return 0


def resume_command(args: argparse.Namespace, output_dir: Path) -> str:
    """The exact command line that continues this run.

    Rebuilt from the arguments this process was given rather than hardcoded, so
    a run started with a non-default corpus or profile resumes with them too —
    resuming under a different plan is how a schedule silently changes shape
    halfway through.
    """
    parts = ["uv", "run", "python", "scripts/train_vlm.py"]
    parts += ["--config", str(args.config)]
    parts += ["--output-dir", str(output_dir)]
    if args.corpus is not None:
        parts += ["--corpus", str(args.corpus)]
    if args.val is not None:
        parts += ["--val", str(args.val)]
    if args.image_root is not None:
        parts += ["--image-root", str(args.image_root)]
    if args.max_steps is not None:
        parts += ["--max-steps", str(args.max_steps)]
    if args.save_steps is not None:
        parts += ["--save-steps", str(args.save_steps)]
    if args.sanity_check:
        parts.append("--sanity-check")
    parts += ["--resume", "auto"]
    return " ".join(parts)


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
