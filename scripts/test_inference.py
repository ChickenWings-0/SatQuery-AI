#!/usr/bin/env python
"""Serve the Phase 7 QLoRA adapter on one held-out patch and audit what it says.

    uv run python scripts/test_inference.py --adapter runs/full-epoch-v1/adapter

This is the Phase 8 smoke test for *grounding*, not for accuracy. It asks four
questions of one BigEarthNet patch the adapter never trained on, and checks the
two mechanical guarantees the whole design rests on:

* **Boxes round-trip.** The GROUNDING turn must emit boxes that
  :func:`satquery.models.prompts.box_format.parse` reads back — the same parser
  ``tools/text_grounding.py`` runs at serving time. A checkpoint that grounds
  beautifully in a format the parser does not accept is worth nothing, and that
  failure is silent (DATA_ADAPTATION_PLAN §4.5).
* **Numbers resolve to measurements.** Every numeric span in a prose answer is
  put through :func:`satquery.evidence.citation_validator.validate` against the
  FactSheet the render pass actually measured for this patch. A number that
  resolves to nothing is reported as an uncited span — that list is the result,
  not a warning about it.

The last two probes are the ones that can fail interestingly. Probe 3 asks for a
percentage *nothing on the sheet measures*; probe 4 asks the same question with
the sheet emptied, where the prompt forbids stating any number at all. They
differ from probe 2 in one input only, so the comparison isolates the guardrail
from the model's general fluency: anything numeric the model produces there is
an invention, and the validator is what has to catch it.

Evidence comes from ``factsheets.jsonl`` — the sheets the Phase 2 render pass
measured off the real bands — rather than from a tool run here, so the numbers
the answer is checked against are the same ones the corpus was built from.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.evidence.citation_validator import (  # noqa: E402
    CitationPolicy,
    validate,
)
from satquery.evidence.fact_sheet import Fact, FactSheet  # noqa: E402
from satquery.models.prompts.box_format import (  # noqa: E402
    BOX_SCALE,
    NormalisedBox,
    strip_boxes,
)
from satquery.models.prompts.box_format import (
    parse as parse_boxes,
)
from satquery.models.prompts.builder import (  # noqa: E402
    ViewInput,
    build_prompt,
    strip_citation_markers,
)
from satquery.schemas.enums import PairType, TaskType  # noqa: E402

DEFAULT_BASE_MODEL: Final[str] = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_ADAPTER: Final[Path] = Path("runs/full-epoch-v1/adapter")
DEFAULT_FACTSHEETS: Final[Path] = Path(
    "data/processed/views/bigearthnet_v2/factsheets.jsonl"
)
DEFAULT_TRAIN_CORPUS: Final[Path] = Path("data/processed/corpus/full.train.jsonl")

HELD_OUT_SPLIT: Final[str] = "validation"
"""Only patches reBEN itself calls validation are eligible. The adapter trained
on the train split, so this is held out by the dataset's own definition rather
than by anything this script decides."""

DISCRETE_LABEL: Final[str] = "Inland waters"
"""The one CORINE class in reBEN that names a *thing* rather than a cover type.

Every other label — arable land, pastures, forest — describes what the ground is
made of, and its honest bounding box is the whole frame. A lake or a river has
edges, so it is the only class in this dataset a localisation claim can be
checked against."""

DEFAULT_RAW_ROOT: Final[Path] = Path("data/raw/ben/BigEarthNet-S2")

MAX_REFERENCE_AREA: Final[float] = 0.35
"""The measured water box may cover at most this much of the frame. Above it, a
full-frame guess starts scoring well by accident."""

WATER_FRACTION: Final[tuple[float, float]] = (0.03, 0.35)
"""Wet-pixel fraction the patch must fall inside: enough water to be visible in
a 448 px view, not so much that the scene simply is a lake."""

FULL_FRAME_IOU_FLOOR: Final[float] = 0.90
"""A box this close to (0,0),(1000,1000) is the degenerate answer, and is
reported as such however well it happens to overlap the reference."""

MAX_VIEWS: Final[int] = 6
"""The profile's ``max_views_per_sample``. Serving with more views than training
used changes the token budget the adapter was fitted against."""


class ProbeError(RuntimeError):
    """The test could not be run — bad inputs, missing weights, no patch."""


# ----------------------------------------------------------------- evidence


def load_fact_sheet(raw: dict[str, Any]) -> FactSheet:
    """Rebuild the measured sheet from a ``factsheets.jsonl`` row.

    The keys are already namespaced ``tool.scalar`` by the render pass, so the
    split here reproduces exactly what :func:`evidence.fact_sheet.build` would
    have produced from the live Execution — including the ``step`` the citation
    grammar prints, which is 1 because one tool measured this patch.
    """
    facts: dict[str, Fact] = {}
    for key, value in raw.items():
        tool, _, scalar = key.partition(".")
        if not scalar:
            continue
        facts[key] = Fact(key=key, tool=tool, scalar=scalar, step=1, value=value)
    return FactSheet(facts=facts)


def training_patch_ids(corpus: Path) -> set[str]:
    """Every patch id the adapter saw, read out of the training corpus itself.

    Read rather than assumed. The corpus id is ``ben2_txt:<patch>:<row>``, so the
    middle field is the patch — and checking against the file that was actually
    trained on is the only way to be sure a "validation" patch did not also enter
    training through some other source.
    """
    seen: set[str] = set()
    if not corpus.is_file():
        return seen
    with corpus.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            parts = str(row.get("id", "")).split(":")
            if len(parts) >= 2:
                seen.add(parts[1])
    return seen


def select_patch(
    factsheets: Path, corpus: Path, patch_id: str | None
) -> dict[str, Any]:
    """Pick one held-out patch carrying real measurements and readable views.

    Raises:
        ProbeError: No eligible patch, or the named one is not held out.
    """
    if not factsheets.is_file():
        raise ProbeError(f"{factsheets} does not exist. Run the Phase 2 render pass.")

    trained_on = training_patch_ids(corpus)
    with factsheets.open() as handle:
        for line in handle:
            try:
                sheet = json.loads(line)
            except json.JSONDecodeError:
                continue
            if patch_id is not None:
                if sheet.get("patch_id") != patch_id:
                    continue
                if sheet["patch_id"] in trained_on:
                    raise ProbeError(
                        f"{patch_id} is in {corpus} — the adapter trained on it, so "
                        "it cannot measure generalisation. Omit --patch to let the "
                        "script choose a held-out one."
                    )
                return sheet
            if sheet.get("split") != HELD_OUT_SPLIT:
                continue
            if sheet.get("patch_id") in trained_on:
                continue
            if not sheet.get("fact_sheet"):
                continue
            if all(Path(p).is_file() for p in sheet.get("views", {}).values()):
                return sheet

    raise ProbeError(
        f"no {HELD_OUT_SPLIT} patch in {factsheets} has both measurements and "
        "readable views on disk."
    )


def load_views(sheet: dict[str, Any]) -> list[ViewInput]:
    """Read the rendered views as the labelled inputs the prompt builder wants."""
    from PIL import Image

    views: list[ViewInput] = []
    labels: dict[str, str] = sheet.get("view_labels", {})
    for view_id, path in sheet.get("views", {}).items():
        with Image.open(path) as handle:
            rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        views.append(ViewInput(label=labels.get(view_id, view_id), rgb=rgb))
    return views[:MAX_VIEWS]


# ------------------------------------------------- discrete-object reference


def reference_water_box(
    patch_id: str, raw_root: Path
) -> tuple[float, tuple[int, int, int, int]] | None:
    """Measure where the water actually is, from the raw bands.

    This is the ground truth the grounding probe is scored against, and it is
    *measured* rather than annotated: NDWI over B03 and B08 — both native 10 m,
    both 120 x 120, so no resampling is involved and no interpolation can move an
    edge — thresholded at 0, then the bounding box of what is left.

    BigEarthNet carries no object annotations at all, only CORINE land-cover
    labels, so "there is a river in this patch" is the most the metadata can say
    and it never says *where*. Deriving the box from the pixels is what turns
    that into a checkable claim.

    Returns:
        The water fraction and its box in the same 0-1000 frame the model emits,
        or None when the patch is unreadable or holds no water at all.
    """
    import rasterio

    directory = raw_root / patch_id.rsplit("_", 2)[0] / patch_id
    if not directory.is_dir():
        return None

    channels: dict[str, np.ndarray] = {}
    for band in ("B03", "B08"):
        matches = sorted(directory.glob(f"*_{band}.tif*"))
        if not matches:
            return None
        with rasterio.open(matches[0]) as handle:
            channels[band] = handle.read(1).astype(np.float32)

    green, nir = channels["B03"], channels["B08"]
    total = green + nir
    ndwi = np.where(total != 0, (green - nir) / np.where(total == 0, 1, total), 0.0)
    mask = ndwi > 0.0
    if not mask.any():
        return None

    rows, columns = np.where(mask)
    height, width = mask.shape
    box = (
        int(columns.min() * BOX_SCALE / width),
        int(rows.min() * BOX_SCALE / height),
        int((columns.max() + 1) * BOX_SCALE / width),
        int((rows.max() + 1) * BOX_SCALE / height),
    )
    return float(mask.mean()), box


def box_area(box: Sequence[int]) -> float:
    """Fraction of the frame a 0-1000 box covers."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1]) / float(BOX_SCALE**2)


def iou(first: Sequence[int], second: Sequence[int]) -> float:
    """Intersection over union of two 0-1000 boxes."""
    x_min = max(first[0], second[0])
    y_min = max(first[1], second[1])
    x_max = min(first[2], second[2])
    y_max = min(first[3], second[3])
    overlap = max(0, x_max - x_min) * max(0, y_max - y_min) / float(BOX_SCALE**2)
    union = box_area(first) + box_area(second) - overlap
    return round(overlap / union, 4) if union > 0 else 0.0


def select_discrete_patch(
    factsheets: Path,
    corpus: Path,
    raw_root: Path,
    max_reference_area: float,
    water_fraction: tuple[float, float],
) -> tuple[dict[str, Any], float, tuple[int, int, int, int]]:
    """Find a held-out patch whose water body is small and localised.

    Both bounds matter. The *fraction* bound keeps out patches that are mostly
    lake — grounding those correctly means drawing the whole frame, which is the
    degenerate answer this probe exists to rule out. The *box area* bound keeps
    out patches where the water is scattered in specks across the scene: its
    bounding box is then the whole frame even though very few pixels are wet, so
    a full-frame answer would score as correct and prove nothing.

    Raises:
        ProbeError: Nothing eligible, usually because the raw bands are absent.
    """
    trained_on = training_patch_ids(corpus)
    best: tuple[float, dict[str, Any], float, tuple[int, int, int, int]] | None = None

    with factsheets.open() as handle:
        for line in handle:
            try:
                sheet = json.loads(line)
            except json.JSONDecodeError:
                continue
            if sheet.get("split") != HELD_OUT_SPLIT:
                continue
            if DISCRETE_LABEL not in sheet.get("labels", []):
                continue
            if sheet.get("patch_id") in trained_on or not sheet.get("fact_sheet"):
                continue
            if not all(Path(p).is_file() for p in sheet.get("views", {}).values()):
                continue
            measured = reference_water_box(sheet["patch_id"], raw_root)
            if measured is None:
                continue
            fraction, box = measured
            area = box_area(box)
            if not water_fraction[0] <= fraction <= water_fraction[1]:
                continue
            if area > max_reference_area:
                continue
            if best is None or area < best[0]:
                best = (area, sheet, fraction, box)

    if best is None:
        raise ProbeError(
            f"no held-out '{DISCRETE_LABEL}' patch has a localised water body "
            f"under {max_reference_area:.0%} of frame. Are the raw bands under "
            f"{raw_root}?"
        )
    _, sheet, fraction, box = best
    return sheet, fraction, box


# -------------------------------------------------------------------- model


class AdaptedModel:
    """Base Qwen3-VL with the LoRA adapter applied, loaded once for every probe.

    **Served in bf16, not NF4, and that is not the obvious choice.** The adapter
    was *trained* under NF4, so matching the training path would argue for
    serving under it too. Measured on this box (gfx1100, ROCm 6.4, torch 2.9.1,
    bitsandbytes 4-bit) that path does not generate: the base weights alone,
    with no adapter attached, answer "Name three primary colours" with a run of
    close-parens, while the same weights in bf16 answer it correctly. So NF4
    inference is broken here independently of anything this project trained —
    4-bit *training* was unaffected, which is why it went unnoticed for a
    19-hour run.

    Serving bf16 costs ~18.7 GiB against the 22 GiB budget, which fits, and is
    the only path on this machine that produces output at all. ``--quantise``
    re-enables NF4 for whoever wants to re-check the bitsandbytes bug against a
    newer build.
    """

    def __init__(self, base_model: str, adapter: Path, quantise: bool) -> None:
        """Load the base weights, then inject the adapter."""
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, BitsAndBytesConfig

        if not (adapter / "adapter_config.json").is_file():
            raise ProbeError(f"{adapter} holds no adapter_config.json.")

        kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "device_map": "auto"}
        if quantise:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = self._base(base_model, kwargs)
        self.model = PeftModel.from_pretrained(model, str(adapter))
        self.model.eval()
        # The processor travels with the adapter: the run wrote the chat template
        # and the image settings it trained under beside the weights, and the
        # base repo's copy can differ.
        source = str(adapter) if (adapter / "processor_config.json").is_file() else base_model
        self.processor = AutoProcessor.from_pretrained(source)
        self.torch = torch

    @staticmethod
    def _base(base_model: str, kwargs: dict[str, Any]) -> Any:
        """Load the base weights through whichever class this transformers has."""
        import transformers

        for name in ("Qwen3VLForConditionalGeneration", "AutoModelForImageTextToText"):
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            return cls.from_pretrained(base_model, **kwargs)
        raise ProbeError("this transformers build exposes no Qwen3-VL model class.")

    def generate(self, prompt: Any, max_new_tokens: int) -> str:
        """Answer one built prompt greedily, keeping the box sentinels intact.

        Decoded with ``skip_special_tokens=False``: the box markers Qwen grounds
        with are special tokens, and skipping them deletes precisely the output
        this test exists to parse. The chat scaffolding is trimmed by hand
        afterwards instead.
        """
        from PIL import Image

        from satquery.models.prompts.layout import chat_prompt

        # The same layout function the training records and the serving
        # backend use, so this probe measures the adapter under the token
        # stream it was trained on rather than a third hand-built variant.
        messages = chat_prompt(
            prompt.system,
            [image.label for image in prompt.images],
            prompt.user,
            images=[
                Image.fromarray(np.ascontiguousarray(image.rgb, dtype=np.uint8), mode="RGB")
                for image in prompt.images
            ],
        )

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        with self.torch.inference_mode():
            produced = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        text = self.processor.decode(
            produced[0][prompt_tokens:], skip_special_tokens=False
        )
        for marker in ("<|im_end|>", "<|endoftext|>"):
            text = text.split(marker)[0]
        return text.strip()

    def peak_vram_gib(self) -> float | None:
        """Peak allocation since load, for the record."""
        if not self.torch.cuda.is_available():  # pragma: no cover - CPU box
            return None
        return round(self.torch.cuda.max_memory_allocated() / 1024**3, 2)


# ------------------------------------------------------------------- probes


def box_report(text: str, size_px: int) -> dict[str, Any]:
    """Parse the boxes back out and say whether each one is usable.

    A box is in range when every coordinate sits inside the 0-1000 frame and the
    corners are ordered. Out-of-range coordinates are the characteristic failure
    of a checkpoint that learned absolute pixels instead of thousandths, so they
    are reported per box rather than summarised away.
    """
    boxes: list[NormalisedBox] = parse_boxes(text, width=size_px, height=size_px)
    parsed: list[dict[str, Any]] = []
    for box in boxes:
        in_range = (
            0 <= box.x_min < box.x_max <= BOX_SCALE
            and 0 <= box.y_min < box.y_max <= BOX_SCALE
        )
        parsed.append(
            {
                "label": box.label,
                "normalised": [box.x_min, box.y_min, box.x_max, box.y_max],
                "pixels": list(box.to_pixels(size_px, size_px)),
                "in_range": in_range,
            }
        )
    return {
        "n_boxes": len(parsed),
        "all_in_range": all(b["in_range"] for b in parsed) if parsed else False,
        "declined": "NONE" in text.upper() and not parsed,
        "boxes": parsed,
    }


def _verdict(uncited: Sequence[str], invented: Sequence[str]) -> str:
    """Say plainly what the fact-checker caught, if anything."""
    problems: list[str] = []
    if uncited:
        problems.append(f"{len(uncited)} number(s) resolved to no measurement")
    if invented:
        problems.append(f"{len(invented)} invented citation key(s)")
    if not problems:
        return "GROUNDED — every number resolved to a measurement"
    return "INTERCEPTED — " + "; ".join(problems)


def citation_report(text: str, sheet: FactSheet) -> dict[str, Any]:
    """Run the fact-checker over one answer and report what it intercepted.

    Two things come off the text before it is validated, and both matter:

    * **Boxes.** A coordinate is a position in the frame, not a claim about the
      scene, and there is no measurement it could ever resolve against.
    * **Citation markers.** ``[sar_backscatter_analyzer.sigma0_vv_db_mean]`` is
      scaffolding the prompt asked for, not prose — and the key contains a digit,
      so leaving it in makes the validator read the ``0`` of ``sigma0`` as an
      uncited numeric claim. That is a false positive against an answer that
      cited everything correctly.

    The keys themselves are not discarded: a key the model quoted that is not on
    the sheet is an invented measurement, and is reported as ``unknown_keys``.
    """
    marked = strip_citation_markers(strip_boxes(text), sheet)
    result = validate(marked.text, sheet, policy=CitationPolicy.FLAG)
    return {
        "facts_available": len(sheet.facts),
        "citations_resolved": len(result.citations),
        "cited_keys": marked.cited_keys,
        "invented_keys": marked.unknown_keys,
        "cited_values": [
            {"claim": c.claim, "source": c.source, "value": c.value}
            for c in result.citations
        ],
        "uncited_numeric_spans": result.uncited_numeric_spans,
        "verdict": _verdict(result.uncited_numeric_spans, marked.unknown_keys),
    }


def localisation_report(
    boxes: Sequence[dict[str, Any]], reference: Sequence[int]
) -> dict[str, Any]:
    """Score the emitted boxes against the box measured from the raw bands.

    The headline is IoU against the measured extent, but a full-frame box is
    called out separately and on its own terms. It has to be: on a patch where
    the target happens to be large, "everything" can post a respectable IoU
    while demonstrating no localisation whatsoever, and that is precisely the
    failure this probe was added to detect. ``tightness`` states the same thing
    as a ratio — how much smaller the model's box is than the whole frame.
    """
    if not boxes:
        return {"scored": False, "reason": "no box emitted"}

    best = max(boxes, key=lambda b: iou(b["normalised"], reference))
    predicted = best["normalised"]
    full_frame = [0, 0, BOX_SCALE, BOX_SCALE]
    overlap = iou(predicted, reference)
    frame_overlap = iou(predicted, full_frame)
    degenerate = frame_overlap >= FULL_FRAME_IOU_FLOOR
    return {
        "scored": True,
        "predicted": predicted,
        "reference": list(reference),
        "iou": overlap,
        "predicted_area_fraction": round(box_area(predicted), 4),
        "reference_area_fraction": round(box_area(reference), 4),
        "tightness": round(1.0 - box_area(predicted), 4),
        "full_frame": degenerate,
        "verdict": (
            "DEGENERATE — the box is the whole frame, so it localises nothing"
            if degenerate
            else f"LOCALISED — IoU {overlap:.2f} against the measured extent, "
            f"covering {box_area(predicted):.1%} of frame"
        ),
    }


def run_probe(
    model: AdaptedModel,
    name: str,
    task: TaskType,
    question: str,
    sheet: FactSheet,
    views: Sequence[ViewInput],
    size_px: int,
    max_new_tokens: int,
    reference: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build one prompt, answer it, and audit the answer.

    When *reference* is given the grounding report is scored against it, which
    turns "the box parsed" into "the box is in the right place".
    """
    prompt = build_prompt(
        task=task,
        pair_type=PairType.CROSS_MODAL,
        sheet=sheet,
        views=views,
        question=question,
        max_views=MAX_VIEWS,
    )
    text = model.generate(prompt, max_new_tokens=max_new_tokens)
    report: dict[str, Any] = {
        "probe": name,
        "task": str(task),
        "question": question,
        "facts_in_prompt": len(sheet.facts),
        "raw_answer": text,
    }
    if task is TaskType.GROUNDING:
        report["grounding"] = box_report(text, size_px)
        if reference is not None:
            report["localisation"] = localisation_report(
                report["grounding"]["boxes"], reference
            )
    report["fact_check"] = citation_report(text, sheet)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--factsheets", type=Path, default=DEFAULT_FACTSHEETS)
    parser.add_argument("--train-corpus", type=Path, default=DEFAULT_TRAIN_CORPUS)
    parser.add_argument(
        "--patch",
        default=None,
        help="Test this patch id instead of the first eligible held-out one.",
    )
    parser.add_argument("--size-px", type=int, default=448)
    parser.add_argument(
        "--discrete",
        action="store_true",
        help="Ground a discrete object instead of continuous cover: select a "
        "held-out patch with a small, localised water body and score the box "
        "against one measured from the raw bands.",
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--quantise",
        action="store_true",
        help="Load the base weights in NF4 rather than bf16. Matches the "
        "training path, but 4-bit generation is broken on this ROCm build — see "
        "AdaptedModel. For re-checking that bug, not for reading results off.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Select a patch, run the three probes, print the audit."""
    args = parse_args(argv)
    reference: tuple[int, int, int, int] | None = None
    water_fraction: float | None = None
    try:
        if args.discrete:
            selected, water_fraction, reference = select_discrete_patch(
                args.factsheets,
                args.train_corpus,
                args.raw_root,
                MAX_REFERENCE_AREA,
                WATER_FRACTION,
            )
        else:
            selected = select_patch(args.factsheets, args.train_corpus, args.patch)
    except ProbeError as error:
        print(str(error), file=sys.stderr)
        return 2

    measured = load_fact_sheet(selected["fact_sheet"])
    views = load_views(selected)
    patch = selected["patch_id"]

    print(f"patch      : {patch}")
    print(f"split      : {selected.get('split')} (held out from training)")
    print(f"labels     : {', '.join(selected.get('labels', [])) or 'none'}")
    print(f"views      : {len(views)} attached")
    print(f"facts      : {len(measured.facts)} measured scalars")
    print(f"adapter    : {args.adapter}")
    if reference is not None:
        print(
            f"target     : {DISCRETE_LABEL} — measured extent {list(reference)}, "
            f"{box_area(reference):.1%} of frame, {water_fraction:.1%} wet pixels"
        )
    print()

    try:
        model = AdaptedModel(args.base_model, args.adapter, quantise=args.quantise)
    except ProbeError as error:
        print(str(error), file=sys.stderr)
        return 2
    except ImportError as error:
        print(f"{error}\n\nInference needs: uv sync --extra train", file=sys.stderr)
        return 2

    # The render pass measured index and backscatter *statistics*; it did not
    # measure class fractions. So a question about the built-up percentage has
    # no measurement behind it on this sheet, and that is what makes probe 3 a
    # trap rather than a duplicate of probe 2.
    area_question = (
        "What percentage of this scene's area is built-up, and does the SAR "
        "backscatter agree with the optical indices?"
    )
    probes = [
        # 1. Boxes. On --discrete the target is a thing with edges and the box is
        #    scored against the measured extent; otherwise it is continuous cover
        #    and only the format is being checked.
        (
            "grounding",
            TaskType.GROUNDING,
            "the lake or river" if args.discrete else "the vegetated areas",
        ),
        # 2. Numbers that ARE on the sheet. The right answer copies them and the
        #    validator resolves the digits back to the measurement — this is the
        #    citation path working, and the control for probe 3.
        (
            "cited_measurements",
            TaskType.CROSS_MODAL_VQA,
            "What is the mean NDVI of this scene, and what is the mean VV "
            "backscatter in dB?",
        ),
        # 3. A percentage nothing measured. Any figure the model puts here is
        #    invented, and the validator is what has to intercept it.
        ("uncited_area_percentage", TaskType.CROSS_MODAL_VQA, area_question),
        # 4. The same question with the sheet emptied, where the prompt forbids
        #    stating any number at all. Isolates the guardrail from the model's
        #    general fluency: probes 3 and 4 differ in one input only.
        ("guardrail_empty_sheet", TaskType.CROSS_MODAL_VQA, area_question),
    ]

    reports: list[dict[str, Any]] = []
    for name, task, question in probes:
        sheet = FactSheet() if name == "guardrail_empty_sheet" else measured
        reports.append(
            run_probe(
                model,
                name,
                task,
                question,
                sheet,
                views,
                args.size_px,
                args.max_new_tokens,
                reference=reference if task is TaskType.GROUNDING else None,
            )
        )

    result = {
        "patch_id": patch,
        "split": selected.get("split"),
        "labels": selected.get("labels", []),
        "adapter": str(args.adapter),
        "base_model": args.base_model,
        "quantised_nf4": args.quantise,
        "peak_vram_gib": model.peak_vram_gib(),
        "discrete_object": (
            {
                "label": DISCRETE_LABEL,
                "reference_box": list(reference),
                "reference_area_fraction": round(box_area(reference), 4),
                "water_fraction": round(water_fraction, 4),
            }
            if reference is not None
            else None
        ),
        "measured": {
            "ndvi_mean": measured.number("spectral_index_analyzer.ndvi_mean"),
            "sigma0_vv_db_mean": measured.number(
                "sar_backscatter_analyzer.sigma0_vv_db_mean"
            ),
            "built_up_fraction_pct": measured.number(
                "spectral_index_analyzer.built_up_fraction_pct"
            ),
        },
        "probes": reports,
    }

    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"\nwritten to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
