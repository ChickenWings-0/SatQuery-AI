#!/usr/bin/env python
"""Zero-shot VRSBench VQA baseline — the "before" column of the ablation table.

Master.md §8 Phase 4 asks this script for two things, and they are different
questions:

1. **How good is the un-adapted model?** Exact-match accuracy against the
   reference answers. This number only means something on the real VRSBench
   subset, and the script says so loudly when it is running on the built-in
   mock instead.
2. **Does it stay inside its evidence?** Every numeric span in every generated
   answer must resolve to a measurement through the Phase 3
   :mod:`~satquery.evidence.citation_validator`. This one is meaningful on any
   imagery at all, mock included, because it is a property of our prompt and our
   plumbing rather than of the benchmark — and it is the assertion the script
   fails on.

The default corpus is a deterministic five-item mock, so the script runs on a
laptop with no dataset and no network and still exercises the whole path:
ingestion, the renderer, the deterministic tools, the FactSheet, the prompt, the
model and the validator. Point ``--data-dir`` at a real VRSBench subset — a
``items.jsonl`` of ``{"id", "image", "question", "answer"}`` — to get the
accuracy column as well.

Usage::

    uv run python scripts/eval_vrsbench_zeroshot.py
    uv run python scripts/eval_vrsbench_zeroshot.py --data-dir data/vrsbench/subset
    uv run python scripts/eval_vrsbench_zeroshot.py --backend llamacpp --json out.json

Exit codes: ``0`` every answer was fully grounded · ``1`` some answer stated a
number no tool measured · ``2`` no VLM backend was servable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.agent.executor import ExecutionCache  # noqa: E402
from satquery.agent.pipeline import AnalysisRequest  # noqa: E402
from satquery.agent.pipeline import analyze as run_analysis  # noqa: E402
from satquery.core.config import get_settings  # noqa: E402
from satquery.evidence.citation_validator import CitationPolicy, extract_spans  # noqa: E402
from satquery.ingest.pipeline import SourceImage, ingest  # noqa: E402
from satquery.models.loader import BackendKind, available_backend, backend_config  # noqa: E402
from satquery.registry.registry import default_registry  # noqa: E402
from satquery.render.artifact_store import ArtifactStore  # noqa: E402
from satquery.schemas.trace import AuditTrace  # noqa: E402

VLM_TOOLS: Final[tuple[str, ...]] = ("vlm_vqa", "vlm_caption")

WGS84_UTM43N: Final[str] = "EPSG:32643"
VHR_GSD_M: Final[float] = 0.5
"""VRSBench imagery is VHR aerial, roughly 0.1-3 m. The mock sits in that band so
capability matching and the GSD checks behave as they would on the real corpus."""

TILE_PX: Final[int] = 512

_WORD: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Item:
    """One VRSBench-shaped VQA item."""

    id: str
    image: Path
    question: str
    answer: str = ""
    """The reference answer. Empty on the mock, where accuracy is meaningless."""


@dataclass
class Outcome:
    """What the system produced for one item, and how well grounded it was."""

    id: str
    question: str
    reference: str
    generated: str
    generator: str
    template_fallback: bool
    confidence: float
    numeric_spans: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    uncited_numeric_spans: list[str] = field(default_factory=list)
    fact_keys: list[str] = field(default_factory=list)
    exact_match: bool | None = None

    @property
    def grounded(self) -> bool:
        """True when every number the answer stated resolved to a measurement."""
        return not self.uncited_numeric_spans


# ------------------------------------------------------------------ mock corpus


def _scene(seed: int, size: int = TILE_PX) -> np.ndarray:
    """A deterministic VHR-looking RGB tile: fields, a road and some roofs.

    Structured rather than pure noise, because a model handed white noise
    produces a refusal, and a refusal tests nothing about grounding.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size

    vegetation = 0.35 + 0.25 * np.sin(6.0 * np.pi * (x + 0.15 * seed)) ** 2
    soil = 0.55 + 0.10 * np.cos(4.0 * np.pi * y)
    field_mask = (np.floor(x * 6) + np.floor(y * 6)) % 2 == 0

    red = np.where(field_mask, soil, 0.22 * vegetation).astype(np.float32)
    green = np.where(field_mask, soil * 0.92, 0.55 * vegetation).astype(np.float32)
    blue = np.where(field_mask, soil * 0.80, 0.20 * vegetation).astype(np.float32)

    road = np.abs(y - (0.30 + 0.05 * seed / 5.0)) < 0.02
    for channel in (red, green, blue):
        channel[road] = 0.42

    for index in range(4 + seed % 3):
        top = int(size * (0.55 + 0.08 * index))
        left = int(size * (0.10 + 0.17 * index))
        red[top : top + 26, left : left + 34] = 0.62
        green[top : top + 26, left : left + 34] = 0.38
        blue[top : top + 26, left : left + 34] = 0.33

    stack = np.stack([red, green, blue])
    stack += rng.normal(0.0, 0.012, stack.shape).astype(np.float32)
    clipped: np.ndarray = np.clip(stack, 0.0, 1.0)
    return clipped


MOCK_QUESTIONS: Final[tuple[tuple[str, str], ...]] = (
    ("How much of this scene is covered by vegetation?", ""),
    ("Is there a road crossing this scene, and where does it run?", ""),
    ("Describe the buildings visible in this image.", ""),
    ("What is the dominant land cover here?", ""),
    ("How bright is this scene overall, and what does that suggest?", ""),
)
"""Chosen so that some questions *are* answerable from the FactSheet and some are
not. An evaluation where every question has a measured answer would never catch
the failure this script exists to catch — a model that invents a number when it
has none."""


def build_mock_corpus(output_dir: Path, count: int = 5) -> list[Item]:
    """Write *count* deterministic VHR tiles and pair them with questions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = Affine(VHR_GSD_M, 0.0, 712_340.0, 0.0, -VHR_GSD_M, 3_161_780.0)
    items: list[Item] = []
    for index in range(count):
        question, reference = MOCK_QUESTIONS[index % len(MOCK_QUESTIONS)]
        path = output_dir / f"vrsbench_mock_{index:02d}.tif"
        data = (_scene(index) * 10_000).astype(np.uint16)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=TILE_PX,
            height=TILE_PX,
            count=3,
            dtype="uint16",
            crs=CRS.from_string(WGS84_UTM43N),
            transform=transform,
        ) as handle:
            handle.write(data)
            handle.descriptions = ("red", "green", "blue")
        items.append(
            Item(id=f"mock:{index:02d}", image=path, question=question, answer=reference)
        )
    return items


def load_corpus(data_dir: Path, limit: int) -> list[Item]:
    """Read a real VRSBench subset from ``items.jsonl``.

    Raises:
        FileNotFoundError: The manifest is not there.
        ValueError: A record is missing a field, or names an image that is not.
    """
    manifest = data_dir / "items.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} does not exist")

    items: list[Item] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record: dict[str, Any] = json.loads(line)
        try:
            image = (data_dir / str(record["image"])).resolve()
            item = Item(
                id=str(record.get("id", f"item:{line_number}")),
                image=image,
                question=str(record["question"]),
                answer=str(record.get("answer", "")),
            )
        except KeyError as error:
            raise ValueError(f"{manifest}:{line_number} has no {error} field") from error
        if not item.image.is_file():
            raise ValueError(f"{manifest}:{line_number} names a missing image {item.image}")
        items.append(item)
        if len(items) >= limit:
            break
    return items


# -------------------------------------------------------------------- execution


def normalise(text: str) -> str:
    """Reduce an answer to the token bag VRSBench's exact match compares.

    Lower-cased alphanumerics, punctuation and articles dropped. This is a
    reasonable reading of the benchmark's protocol, not a verified reproduction
    of it, and Phase 8 replaces it with the harness that is.
    """
    words = [word for word in _WORD.findall(text.lower()) if word not in {"a", "an", "the"}]
    return " ".join(words)


async def evaluate_item(
    item: Item, store: ArtifactStore, registry: Any, cache: ExecutionCache
) -> tuple[Outcome, AuditTrace]:
    """Run one item all the way through the pipeline and score its answer."""
    sources = [SourceImage(path=item.image, filename=item.image.name)]
    ingested = ingest(sources)
    analysis = await run_analysis(
        AnalysisRequest(
            query=item.question,
            sources=sources,
            ingest=ingested,
            # Flagging is the point: a stripped answer would hide the very spans
            # this script reports on.
            citation_policy=CitationPolicy.FLAG,
        ),
        store=store,
        registry=registry,
        cache=cache,
    )
    trace = analysis.trace
    answer = trace.answer
    outcome = Outcome(
        id=item.id,
        question=item.question,
        reference=item.answer,
        generated=answer.text,
        generator=answer.generator,
        template_fallback=answer.template_fallback,
        confidence=trace.confidence.overall,
        numeric_spans=[span.text for span in extract_spans(answer.text)],
        citations=[citation.source for citation in answer.citations],
        uncited_numeric_spans=list(answer.uncited_numeric_spans),
        fact_keys=sorted(trace.fact_sheet),
        exact_match=(
            normalise(answer.text) == normalise(item.answer) if item.answer else None
        ),
    )
    return outcome, trace


async def run(items: Sequence[Item], store: ArtifactStore, registry: Any) -> list[Outcome]:
    """Evaluate every item in order, sharing one execution cache."""
    cache = ExecutionCache()
    outcomes: list[Outcome] = []
    for index, item in enumerate(items, 1):
        outcome, _ = await evaluate_item(item, store, registry, cache)
        outcomes.append(outcome)
        print(f"[{index}/{len(items)}] {item.id}", file=sys.stderr)
    return outcomes


# ---------------------------------------------------------------------- reporting


def report(outcomes: Sequence[Outcome], mocked: bool) -> dict[str, Any]:
    """Summarise the run, keeping the two questions it answers separate."""
    total = len(outcomes)
    scored = [o for o in outcomes if o.exact_match is not None]
    numeric = [o for o in outcomes if o.numeric_spans]
    grounded = [o for o in outcomes if o.grounded]
    synthesised = [o for o in outcomes if not o.template_fallback]

    summary: dict[str, Any] = {
        "items": total,
        "corpus": "mock" if mocked else "vrsbench_subset",
        "answers_from_the_model": len(synthesised),
        "answers_from_the_template": total - len(synthesised),
        "items_stating_a_number": len(numeric),
        "fully_grounded": len(grounded),
        "grounding_rate": round(len(grounded) / total, 4) if total else 0.0,
        "uncited_spans": sorted(
            {span for outcome in outcomes for span in outcome.uncited_numeric_spans}
        ),
        "mean_confidence": (
            round(sum(o.confidence for o in outcomes) / total, 4) if total else 0.0
        ),
    }
    if scored:
        summary["exact_match"] = round(
            sum(1 for o in scored if o.exact_match) / len(scored), 4
        )
        summary["exact_match_items"] = len(scored)
    return summary


def print_report(outcomes: Sequence[Outcome], summary: dict[str, Any], mocked: bool) -> None:
    """Write the human-readable report to stdout."""
    print("\n=== Zero-shot VQA, answers ===\n")
    for outcome in outcomes:
        flag = "OK " if outcome.grounded else "!! "
        print(f"{flag}{outcome.id}  ({outcome.generator})")
        print(f"    Q: {outcome.question}")
        print(f"    A: {outcome.generated}")
        if outcome.citations:
            print(f"    cited: {', '.join(outcome.citations)}")
        if outcome.uncited_numeric_spans:
            print(f"    UNCITED: {', '.join(outcome.uncited_numeric_spans)}")
        print()

    print("=== Summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if mocked:
        print(
            "\n  NOTE: this ran on the built-in mock corpus. The grounding result is\n"
            "  meaningful — it is a property of our prompt and validator — but the\n"
            "  accuracy column is not, and no ablation row may be filled in from it.\n"
            "  Pass --data-dir with a real VRSBench subset for that."
        )


# --------------------------------------------------------------------------- cli


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="A VRSBench subset directory containing items.jsonl. Omit for the mock.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Items to evaluate.")
    parser.add_argument(
        "--backend",
        choices=["auto", "hf", "llamacpp"],
        default="hf",
        help="Serving path. Phase 4's baseline is measured on 'hf' (zero-shot bf16).",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Where to write rendered evidence. Defaults to a temporary directory.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the full report here.")
    parser.add_argument(
        "--allow-template",
        action="store_true",
        help="Do not fail when no model answered. For smoke-testing the harness.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation and return the process exit code."""
    args = parse_args(argv)

    settings = get_settings().model_copy(update={"vlm_backend": args.backend})
    kind = available_backend(backend_config(settings))
    if kind is BackendKind.NONE and not args.allow_template:
        print(
            f"No VLM backend is servable for --backend {args.backend}. Download the "
            f"weights, start scripts/serve_vlm.sh, or pass --allow-template to "
            f"exercise the harness against the deterministic answers.",
            file=sys.stderr,
        )
        return 2
    print(f"backend: {kind.value}", file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="vrsbench-zeroshot-") as scratch:
        workspace = Path(scratch)
        mocked = args.data_dir is None
        items = (
            build_mock_corpus(workspace / "corpus", args.limit)
            if mocked
            else load_corpus(args.data_dir, args.limit)
        )
        if not items:
            print("no items to evaluate", file=sys.stderr)
            return 2

        store = ArtifactStore(args.artifacts or workspace / "artifacts")
        registry = default_registry().with_availability(
            dict.fromkeys(VLM_TOOLS, kind is not BackendKind.NONE)
        )
        outcomes = asyncio.run(run(items, store, registry))

    summary = report(outcomes, mocked)
    print_report(outcomes, summary, mocked)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"summary": summary, "items": [asdict(o) for o in outcomes]},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    # The assertion Phase 4 is verified against: every numeric span in every
    # generated answer resolved to a citation.
    ungrounded = [o for o in outcomes if not o.grounded]
    if ungrounded:
        print(
            f"\nFAIL: {len(ungrounded)} of {len(outcomes)} answers state a number no tool "
            f"measured: "
            + "; ".join(f"{o.id} -> {', '.join(o.uncited_numeric_spans)}" for o in ungrounded),
            file=sys.stderr,
        )
        return 1
    print("\nPASS: every number in every answer resolves to a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
