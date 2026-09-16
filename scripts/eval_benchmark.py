#!/usr/bin/env python
"""Run the validation benchmark and write the slide tables.

ROADMAP_REMAINING_FIXES.md, Track 1. A stratified slice of the held-out
``val.jsonl`` — 100 per source by default, five sources, 500 samples — is sent
through the served model, scored per task family, and aggregated into
``runs/eval/<name>/results.{md,json,csv}`` plus the worst cases under
``confusion/``. The chosen ids are written to ``sample_ids.json`` so the
baseline run scores the identical slice.

Usage::

    make eval                                   # adapter, 100/source, runs/eval/sq-lora-v2-full
    make eval-baseline                          # base model on the same sample_ids.json
    uv run --no-sync python scripts/eval_benchmark.py --per-source 50 --name quick
    uv run --no-sync python scripts/eval_benchmark.py --backend llamacpp --name q4km
        --ids runs/eval/sq-lora-v2-full/sample_ids.json      (one line)
    uv run --no-sync python scripts/eval_benchmark.py --self-check   # no model: scorers vs refs
    uv run --no-sync python scripts/eval_benchmark.py --rescore --name sq-lora-v2-full

Three modes share one pipeline:

* **model** (default) — generation through the production backend, chosen
  by ``--backend`` / ``SATQUERY_VLM_BACKEND``; ``--adapter`` overrides
  ``SATQUERY_VLM_ADAPTER_PATH`` and ``--baseline`` clears it (stock Qwen3-VL).
* **--self-check** — every reference is scored as its own prediction. No
  weights, seconds on a laptop, and every applicable metric must come out at
  100 %: this is how a scorer change is proven not to move the goalposts.
* **--rescore** — re-read ``predictions.jsonl`` and rebuild the tables. For
  when the scorers changed and the GPU hours should not be spent again.

Exit status is 0 when the run completes, 2 when a self-check falls short of
its ceiling, 1 on a usage error.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.core.config import get_settings  # noqa: E402
from satquery.eval import report as report_module  # noqa: E402
from satquery.eval import runner as runner_module  # noqa: E402
from satquery.eval import sampler  # noqa: E402
from satquery.eval.scorers import score  # noqa: E402
from satquery.models.loader import (  # noqa: E402
    BackendConfig,
    BackendKind,
    backend_config,
    get_backend,
)
from satquery.schemas.enums import TaskType  # noqa: E402
from satquery.training.corpus_builder import CorpusSample  # noqa: E402

REPO: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_VAL: Final[Path] = REPO / "data/processed/corpus/v2-full/val.jsonl"
DEFAULT_ADAPTER: Final[Path] = REPO / "runs/sq-lora-v2-full/adapter"
DEFAULT_OUT: Final[Path] = REPO / "runs/eval"

SELF_CHECK_METRICS: Final[tuple[str, ...]] = (
    "exact_match",
    "set_f1",
    "grounding_recall",
    "grounding_iou",
    "grounding_format_ok",
    "citation_precision",
    "fact_recall",
    "bleu4",
    "rouge_l",
)
"""Metrics that must be 1.0 when the reference is scored against itself.
``uncited_rate`` must be 0.0 — checked separately."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL, help="Held-out corpus JSONL.")
    parser.add_argument("--root", type=Path, default=REPO, help="Where corpus view paths resolve.")
    parser.add_argument("--per-source", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, default=None, help="Stop after N samples (smoke runs)."
    )
    parser.add_argument(
        "--ids", type=Path, default=None, help="Pin the slice to a previous run's sample_ids.json."
    )
    parser.add_argument("--name", default=None, help="Run name; default derives from the adapter.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Parent of runs/eval/<name>.")
    parser.add_argument("--backend", choices=["auto", "hf", "llamacpp"], default="auto")
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter directory.")
    parser.add_argument(
        "--baseline", action="store_true", help="Serve the base model with no adapter."
    )
    parser.add_argument(
        "--baseline-of",
        default=None,
        help="Name of the adapter run this is the 'before' column for.",
    )
    parser.add_argument("--max-views", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run.")
    parser.add_argument(
        "--rescore", action="store_true", help="Rebuild the tables from predictions.jsonl."
    )
    parser.add_argument(
        "--self-check", action="store_true", help="Score references as predictions; no model."
    )
    parser.add_argument("--no-worst-cases", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _relative(path: Path) -> str:
    """*path* relative to the repository when it is inside it, else as given."""
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def git_sha() -> str:
    """The checked-out commit, read from ``.git`` without invoking git."""
    head = REPO / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref: "):
            return (REPO / ".git" / ref.removeprefix("ref: ")).read_text().strip()
        return ref
    except OSError:
        return "unknown"


def resolve_backend(args: argparse.Namespace) -> tuple[Any, BackendConfig]:
    """The production backend, with the adapter the flags ask for."""
    config = backend_config(get_settings())
    if args.backend != "auto":
        config.kind = BackendKind.HF if args.backend == "hf" else BackendKind.LLAMACPP
    if args.baseline:
        config.adapter_path = None
    elif args.adapter is not None:
        config.adapter_path = args.adapter
    elif config.adapter_path is None and DEFAULT_ADAPTER.is_dir():
        config.adapter_path = DEFAULT_ADAPTER
    if config.kind is BackendKind.NONE:
        raise SystemExit(
            "the VLM is disabled (SATQUERY_VLM_DISABLED / backend none); nothing to score"
        )
    return get_backend(config), config


def run_name(args: argparse.Namespace, config: BackendConfig | None) -> str:
    """``<adapter dir name>`` by default; ``base-<model>`` for a baseline."""
    if args.name:
        return str(args.name)
    if args.self_check:
        return "self-check"
    if config is None:
        return "rescore"
    if args.baseline or config.adapter_path is None:
        return "base-" + config.model_id.rsplit("/", 1)[-1].lower()
    return (
        Path(config.adapter_path).parent.name
        if Path(config.adapter_path).name == "adapter"
        else Path(config.adapter_path).name
    )


def select(args: argparse.Namespace) -> list[CorpusSample]:
    """The slice: pinned ids when given, else a fresh stratified draw."""
    if not args.val.exists():
        raise SystemExit(
            f"{args.val} does not exist — build the corpus first (scripts/rebuild_corpus_v2.sh)"
        )
    if args.ids is not None:
        ids = sampler.read_ids(args.ids)
        chosen = sampler.select_ids(sampler.load_samples(args.val), ids)
    else:
        chosen = sampler.stratified_sample(
            sampler.load_samples(args.val), args.per_source, args.seed
        )
    if args.limit is not None:
        chosen = chosen[: args.limit]
    return chosen


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark."""
    args = parse_args(argv)
    started = time.perf_counter()

    backend: Any = None
    config: BackendConfig | None = None
    if not (args.self_check or args.rescore):
        backend, config = resolve_backend(args)
    name = run_name(args, config)
    out = args.out / name
    predictions_path = out / "predictions.jsonl"
    ids_path = out / "sample_ids.json"

    if args.rescore:
        if not predictions_path.exists():
            raise SystemExit(f"{predictions_path} does not exist; nothing to rescore")
        predictions = list(runner_module.read_predictions(predictions_path).values())
        wanted = {prediction.id for prediction in predictions}
        samples = {s.id: s for s in sampler.load_samples(args.val) if s.id in wanted}
    else:
        chosen = (
            select(args)
            if not (args.resume and ids_path.exists())
            else sampler.select_ids(sampler.load_samples(args.val), sampler.read_ids(ids_path))
        )
        samples = {sample.id: sample for sample in chosen}
        out.mkdir(parents=True, exist_ok=True)
        sampler.write_ids(chosen, ids_path)
        if args.self_check:
            backend = runner_module.reference_backend(chosen)

        tty = sys.stdout.isatty()

        def progress(done: int, total: int, prediction: runner_module.Prediction) -> None:
            if args.quiet:
                return
            flag = " !" if prediction.error else ""
            line = (
                f"[{done:4d}/{total}] {prediction.source:<15} {prediction.task:<20} "
                f"{prediction.latency_ms:6d} ms{flag}"
            )
            if tty:
                print("\r" + line, end="" if done < total else "\n", flush=True)
            elif done % 25 == 0 or done == total or prediction.error:
                print(line, flush=True)

        predictions = list(
            runner_module.run(
                chosen,
                backend,
                predictions_path,
                root=args.root,
                max_views=args.max_views,
                max_new_tokens=args.max_new_tokens,
                resume=args.resume,
                progress=progress,
            )
        )

    rows = [
        report_module.ScoredSample(
            prediction=prediction,
            score=score(
                TaskType(prediction.task),
                prediction.reference,
                prediction.prediction,
                samples[prediction.id].fact_sheet if prediction.id in samples else {},
            ),
        )
        for prediction in predictions
    ]

    provenance: dict[str, Any] = {
        "name": name,
        "mode": "self-check" if args.self_check else ("rescore" if args.rescore else "model"),
        "backend": str(config.kind)
        if config is not None
        else ("reference" if args.self_check else "rescore"),
        "model_id": config.model_id if config is not None else None,
        "adapter": (
            _relative(config.adapter_path) if config is not None and config.adapter_path else None
        ),
        "baseline_of": args.baseline_of,
        "git_sha": git_sha(),
        "seed": args.seed,
        "per_source": args.per_source,
        "val": _relative(args.val),
        "max_views": args.max_views,
        "max_new_tokens": args.max_new_tokens,
        "predictions": _relative(predictions_path),
        "sample_ids": _relative(ids_path),
        "wall_s": time.perf_counter() - started,
        "env": {key: value for key, value in os.environ.items() if key.startswith("SATQUERY_VLM")},
    }
    groups = report_module.write_report(rows, out, provenance)
    if not args.no_worst_cases:
        report_module.write_worst_cases(rows, samples, out, args.root)

    overall = groups["all"]["all"]
    if not args.quiet:
        print((out / "results.md").read_text(encoding="utf-8"))
        print(f"→ {out}")

    if args.self_check:
        short = [
            key for key in SELF_CHECK_METRICS if key in overall.means and overall.means[key] < 0.999
        ]
        if overall.means.get("uncited_rate", 0.0) > 0.001:
            short.append("uncited_rate")
        if short:
            print(f"self-check below ceiling on: {', '.join(short)}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
