"""The unified instruction-corpus builder (DATA_ADAPTATION_PLAN §3-§5, §7.3).

Six sources become one JSONL schema, and the whole module exists to enforce three
invariants that are invisible at build time and fatal at training time:

* **One label writer.** View labels come from
  :mod:`satquery.render.view_labels` and the system prompt from
  :mod:`satquery.models.prompts.builder` — the *same* functions the Phase 4
  inference path calls. A corpus that phrased a label itself would train the
  model on a dialect the server never speaks, and every test would still pass
  (§2.5, delta #6).
* **One box serialiser.** Grounding answers are emitted by
  :func:`satquery.models.prompts.box_format.serialise`, which
  ``tools/text_grounding.py`` parses back at inference (§4.5, delta #7).
* **Every number is a citation.** Answers quote measurements as
  ``7.42% [change_statistics.changed_area_pct]``, printed with
  :func:`~satquery.evidence.citation_validator.format_number` so the number the
  model copies resolves back to its own fact under the validator's tolerance.
  :func:`assemble` runs the validator over every sample it builds, so a template
  that drifts out of citable form fails the *build*, not the eval (§4.6).

Nothing here downloads anything. Adapters take dataset rows as mappings — the
row a Hugging Face dataset yields, or a mock — and pre-rendered view paths from
the Phase 2 render pass (§7.2). Importing this module pulls in no dataset, no
model and no torch.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from satquery.evidence.citation_validator import format_number, validate
from satquery.evidence.fact_sheet import Fact, FactSheet
from satquery.models.prompts.box_format import (
    BOX_SCALE,
    BoxFormatError,
    NormalisedBox,
    from_pixels,
    is_canonical_answer,
    serialise_answer,
)
from satquery.models.prompts.builder import (
    build_system_prompt,
    build_user_prompt,
    strip_citation_markers,
)
from satquery.models.prompts.templates import DEFAULT_VERSION
from satquery.render.view_labels import label_for_view
from satquery.render.views import ViewId
from satquery.schemas.enums import ImageRole, Modality, PairType, TaskType


class CorpusSource(StrEnum):
    """The six sources of §5, and the value written to ``sample.source``."""

    BIGEARTHNET_V2 = "bigearthnet_v2"
    VRSBENCH = "vrsbench"
    RSVQA_HR = "rsvqa_hr"
    CDVQA = "cdvqa"
    DIOR_RSVG = "dior_rsvg"
    EVIDENCE_QA = "evidence_qa"


COMPOSITION: Final[dict[CorpusSource, int]] = {
    CorpusSource.BIGEARTHNET_V2: 18_000,
    CorpusSource.VRSBENCH: 20_000,
    CorpusSource.RSVQA_HR: 10_000,
    CorpusSource.CDVQA: 8_000,
    CorpusSource.DIOR_RSVG: 6_000,
    CorpusSource.EVIDENCE_QA: 3_000,
}
"""Target sample count per source (§5). Totals 65,000.

**This total is the plan's, not the budget's.** The 200-step throughput probe
measured 0.265 samples/s on the 7900 XTX (``runs/throughput-probe``), so one
pass over 65 000 samples is ~68 h — roughly ten times the "one overnight run"
the plan assumed. A 12 h epoch is ~11 500 samples. The mix for the next run is
an owner's decision (proposed: BEN-txt 4 000 · BEN-labels 1 500 · evidence_qa
2 500 · VRSBench 2 500 · CDVQA 1 000; DIOR-RSVG and RSVQA-HR 0 until resolved)
and should be applied through ``--composition`` on the build rather than by
editing this table, so the §5 targets stay on record as the goal."""

TRACK: Final[dict[CorpusSource, str]] = {
    CorpusSource.BIGEARTHNET_V2: "A",
    CorpusSource.EVIDENCE_QA: "A",
    CorpusSource.VRSBENCH: "B",
    CorpusSource.RSVQA_HR: "B",
    CorpusSource.CDVQA: "B",
    CorpusSource.DIOR_RSVG: "B",
}
"""Track A grounds sensor physics at 10 m; track B aligns task format and VHR
resolution (§5). Recorded so a composition report can show both are present."""

QUARANTINED_SPLITS: Final[frozenset[str]] = frozenset({"test", "testA", "testB", "eval"})
"""Split names that belong to Phase 8 and must never enter the corpus (§4.7)."""


class CorpusError(RuntimeError):
    """A sample could not be built, or the corpus violates an invariant."""


class CitationError(CorpusError):
    """An assistant turn stated a number that does not resolve to its FactSheet.

    Fatal at build time by design: an uncitable training answer teaches the model
    exactly the behaviour ``evidence_qa`` exists to suppress.
    """


class LeakageError(CorpusError):
    """An image from a quarantined test split reached the training corpus (§4.7)."""


# --------------------------------------------------------------------- schema


class ViewRecord(BaseModel):
    """One rendered view as it appears in a corpus line (§3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slot: int = Field(ge=1)
    view_id: str
    path: str
    label: str


class Message(BaseModel):
    """One Hugging Face conversation turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class SampleMeta(BaseModel):
    """Provenance for one sample (§3, §7.3)."""

    model_config = ConfigDict(extra="forbid")

    sensor: str | None = None
    gsd_m: float | None = None
    labels: list[str] = Field(default_factory=list)
    split: Literal["train", "val"] = "train"
    source_split: str = "train"
    augmentations: list[str] = Field(default_factory=list)
    image_sha256: str | None = None
    phash: str | None = None


class CorpusSample(BaseModel):
    """One JSONL line of ``data/processed/corpus/{train,val}.jsonl`` (§3)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source: CorpusSource
    task: TaskType
    pair_type: PairType
    views: list[ViewRecord]
    messages: list[Message]
    fact_sheet: dict[str, float | str] = Field(default_factory=dict)
    meta: SampleMeta = Field(default_factory=SampleMeta)

    @field_validator("messages")
    @classmethod
    def _three_turns(cls, messages: list[Message]) -> list[Message]:
        """Enforce the system -> user -> assistant shape the trainer expects."""
        roles = [message.role for message in messages]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"expected system/user/assistant turns, got {roles}")
        return messages

    @property
    def system(self) -> str:
        """The system turn, carrying the injected FactSheet."""
        return self.messages[0].content

    @property
    def user(self) -> str:
        """The analyst's turn."""
        return self.messages[1].content

    @property
    def assistant(self) -> str:
        """The supervised answer, with its ``[tool.scalar]`` citations."""
        return self.messages[2].content

    def to_json_line(self) -> str:
        """Serialise to one JSONL line, enums rendered as their contract strings."""
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False)


@dataclass(frozen=True)
class SourceView:
    """A pre-rendered view on its way into a sample, before it is labelled.

    Slot numbers are assigned by :func:`label_views`, never carried here: a view
    dropped by augmentation renumbers everything after it, and the label — not
    the index — is what carries meaning (§2.4).
    """

    view_id: ViewId | str
    path: str | Path
    modality: Modality
    sensor: str | None = None
    role: ImageRole | None = None
    acquisition_time: datetime | None = None


def label_views(views: Sequence[SourceView]) -> list[ViewRecord]:
    """Assign slots and build each view's label through the shared builder.

    Raises:
        CorpusError: A view id is not in the frozen catalogue.
    """
    records: list[ViewRecord] = []
    for slot, view in enumerate(views, start=1):
        view_id = str(view.view_id)
        try:
            label = label_for_view(
                view_id=view_id,
                slot=slot,
                modality=view.modality,
                sensor=view.sensor,
                role=view.role,
                acquisition_time=view.acquisition_time,
            )
        except KeyError as error:
            raise CorpusError(f"sample references unknown view {view_id!r}") from error
        records.append(
            ViewRecord(slot=slot, view_id=view_id, path=str(view.path), label=label)
        )
    return records


# ---------------------------------------------------------------- fact sheets


def fact_sheet_from(mapping: Mapping[str, float | str], step: int = 1) -> FactSheet:
    """Build a :class:`FactSheet` from a ``{tool.scalar: value}`` mapping.

    The runtime sheet is assembled from tool executions; a corpus sample has no
    executions, only the scalars the Phase 2 render pass measured. Keys are
    required to be namespaced because the prompt is what the model learns to
    copy, and an unnamespaced key in training would teach it to emit citations
    the validator can never resolve.

    Raises:
        CorpusError: A key is not of the form ``tool.scalar``.
    """
    sheet = FactSheet()
    for key, value in mapping.items():
        tool, _, scalar = key.partition(".")
        if not tool or not scalar or "." in scalar:
            raise CorpusError(f"fact key {key!r} is not of the form 'tool.scalar'")
        sheet.facts[key] = Fact(key=key, tool=tool, scalar=scalar, step=step, value=value)
    return sheet


_UNIT_SUFFIX: Final[dict[str, str]] = {
    "pct": "%",
    "db": " dB",
    "km2": " km²",
    "m2": " m²",
    "px": " px",
}
"""How a scalar's unit is written in prose.

Not cosmetic: :func:`~satquery.evidence.citation_validator.unit_compatible`
refuses to let a bare number cite a scalar whose name carries a unit, so an
answer that wrote ``7.42 [change_statistics.changed_area_pct]`` would fail its
own citation check. The suffix is read off the scalar name, which is the same
place the validator reads it from."""


def unit_suffix(key: str) -> str:
    """The unit string a claim citing *key* must carry, possibly empty."""
    scalar = key.partition(".")[2] or key
    for segment in scalar.lower().split("_"):
        suffix = _UNIT_SUFFIX.get(segment)
        if suffix is not None:
            return suffix
    return ""


def cite(key: str, value: float | str) -> str:
    """Render one measurement as a citable span: ``"7.42% [tool.scalar]"``.

    Numbers are printed with the inference-side formatter, so a value copied out
    of the injected sheet lands inside the validator's tolerance of its own fact.
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        text = f"{format_number(float(value))}{unit_suffix(key)}"
    else:
        text = str(value)
    return f"{text} [{key}]"


# ------------------------------------------------------------------- assembly


def assemble(
    *,
    sample_id: str,
    source: CorpusSource,
    task: TaskType,
    pair_type: PairType,
    views: Sequence[SourceView],
    question: str,
    answer: str,
    fact_sheet: Mapping[str, float | str] | None = None,
    slots: Mapping[str, object] | None = None,
    meta: SampleMeta | None = None,
    prompt_version: str | None = None,
    audit: bool = True,
) -> CorpusSample:
    """Turn one source item into a validated conversation sample.

    The system turn is produced by :func:`build_system_prompt` — the inference
    builder itself — so the FactSheet block, the image index and the rules the
    model trains against are byte-identical to the ones it is served.

    Args:
        sample_id: Stable id, conventionally ``"{source}:{native_id}"``.
        source: Which corpus this came from.
        task: The resolved task; selects the task instruction in the prompt.
        pair_type: Recorded in the prompt's context block.
        views: Pre-rendered views in slot order.
        question: The analyst's turn; empty is filled per task.
        answer: The supervised assistant turn, citations included.
        fact_sheet: ``{tool.scalar: value}`` measured for this scene.
        slots: Resolved query slots, printed when filled.
        meta: Provenance; a default is built when omitted.
        prompt_version: Template version to train against.
        audit: Run the citation audit. Only turned off for grounding, whose
            answer is boxes rather than measurements.

    Raises:
        CitationError: The answer cites a key absent from the sheet, or states a
            number no measurement supports.
    """
    sheet = fact_sheet_from(fact_sheet or {})
    records = label_views(views)
    labels = [record.label for record in records]
    system = build_system_prompt(
        task=task,
        pair_type=pair_type,
        sheet=sheet,
        view_labels=labels,
        slots=slots or {},
        template=None if prompt_version is None else _template(prompt_version),
    )
    if audit:
        audit_answer(sample_id, answer, sheet)
    if task is TaskType.GROUNDING and not is_canonical_answer(answer):
        # Every grounding target must be the format the TASK block dictates:
        # tagged boxes or NONE. A bare box contradicts the instruction it is
        # trained under (ML_PIPELINE_RECOVERY_PLAN fact 13, §2.5).
        raise CorpusError(
            f"{sample_id}: grounding answer is not in the canonical tagged form: {answer!r}"
        )
    return CorpusSample(
        id=sample_id,
        source=source,
        task=task,
        pair_type=pair_type,
        views=records,
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=build_user_prompt(task, question)),
            Message(role="assistant", content=answer),
        ],
        fact_sheet=dict(sheet.as_dict()),
        meta=meta or SampleMeta(),
    )


def _template(version: str) -> Any:
    """Resolve a prompt template version, defaulting to the frozen one."""
    from satquery.models.prompts.templates import get_template

    return get_template(version or DEFAULT_VERSION)


def audit_answer(sample_id: str, answer: str, sheet: FactSheet) -> None:
    """Refuse an assistant turn whose numbers do not resolve to the sheet.

    Raises:
        CitationError: A cited key is unknown, or a numeric span is unsupported.
    """
    marked = strip_citation_markers(answer, sheet)
    if marked.unknown_keys:
        raise CitationError(
            f"{sample_id}: answer cites keys absent from its FactSheet: "
            f"{sorted(set(marked.unknown_keys))}"
        )
    result = validate(marked.text, sheet)
    if result.uncited_numeric_spans:
        raise CitationError(
            f"{sample_id}: answer states numbers no measurement supports: "
            f"{result.uncited_numeric_spans}"
        )


# --------------------------------------------------------------- augmentation


class Augmentation(StrEnum):
    """Corpus-build-time augmentations (§7.3), recorded in ``meta.augmentations``."""

    RESCALE = "rescale"
    SWIR_DROPOUT = "swir_dropout"
    SINGLE_POL = "single_pol"


AUGMENTATION_RATES: Final[dict[CorpusSource, dict[Augmentation, float]]] = {
    CorpusSource.BIGEARTHNET_V2: {
        Augmentation.RESCALE: 0.30,
        Augmentation.SWIR_DROPOUT: 0.20,
        Augmentation.SINGLE_POL: 0.20,
    },
    CorpusSource.VRSBENCH: {Augmentation.RESCALE: 0.30},
    CorpusSource.DIOR_RSVG: {Augmentation.RESCALE: 0.30},
}
"""The §7.3 table. Sources absent from it are never augmented."""

_SWIR_DEPENDENT: Final[frozenset[str]] = frozenset({ViewId.SWIR.value, ViewId.NDBI.value})
_SWIR_FACT_PREFIXES: Final[tuple[str, ...]] = ("ndbi_", "built_up_")
_VH_FACT_MARKERS: Final[tuple[str, ...]] = ("sigma0_vh", "vv_vh_ratio")


def plan_augmentations(
    source: CorpusSource, rng: random.Random
) -> list[tuple[Augmentation, str]]:
    """Draw this sample's augmentations, returning each with its meta tag.

    The rescale factor is drawn here rather than at render time so the tag in
    ``meta.augmentations`` fully determines the pixels: the pre-render pass reads
    the tag back and any run is reproducible from the corpus alone (§7.3).
    """
    drawn: list[tuple[Augmentation, str]] = []
    for augmentation, rate in AUGMENTATION_RATES.get(source, {}).items():
        if rng.random() >= rate:
            continue
        if augmentation is Augmentation.RESCALE:
            factor = 2.0 if source is CorpusSource.BIGEARTHNET_V2 else rng.uniform(1.0, 4.0)
            drawn.append((augmentation, f"rescale_{factor:.1f}x"))
        else:
            drawn.append((augmentation, augmentation.value))
    return drawn


def apply_augmentations(
    drawn: Sequence[tuple[Augmentation, str]],
    views: Sequence[SourceView],
    fact_sheet: Mapping[str, float | str],
) -> tuple[list[SourceView], dict[str, float | str], list[str]]:
    """Drop the views and facts an augmentation removes, and return its tags.

    A dropped view must take its measurements with it. Leaving ``ndbi_mean`` in
    the sheet after a SWIR dropout would train the model to cite a built-up
    number from a scene whose built-up evidence it cannot see — the precise
    hallucination the Cartosat-2S mitigation exists to prevent (§2.2, §7.3).
    """
    kept_views = list(views)
    kept_facts = dict(fact_sheet)
    tags: list[str] = []
    for augmentation, tag in drawn:
        tags.append(tag)
        if augmentation is Augmentation.SWIR_DROPOUT:
            kept_views = [v for v in kept_views if str(v.view_id) not in _SWIR_DEPENDENT]
            kept_facts = {
                key: value
                for key, value in kept_facts.items()
                if not key.partition(".")[2].startswith(_SWIR_FACT_PREFIXES)
            }
        elif augmentation is Augmentation.SINGLE_POL:
            kept_views = [v for v in kept_views if str(v.view_id) != ViewId.SARFC.value]
            kept_facts = {
                key: value
                for key, value in kept_facts.items()
                if not any(marker in key for marker in _VH_FACT_MARKERS)
            }
    return kept_views, kept_facts, tags


# ------------------------------------------------------------------- builders

BEN19_CLASSES: Final[tuple[str, ...]] = (
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
)
"""The official BigEarthNet-v2 19-class nomenclature — the primary mandate
metric's label space (§4.1, §8.1)."""

BEN6_MERGE: Final[dict[str, str]] = {
    "Urban fabric": "Urban",
    "Industrial or commercial units": "Urban",
    "Arable land": "Agriculture",
    "Permanent crops": "Agriculture",
    "Pastures": "Agriculture",
    "Complex cultivation patterns": "Agriculture",
    "Land principally occupied by agriculture, with significant areas of natural vegetation": (
        "Agriculture"
    ),
    "Agro-forestry areas": "Forest",
    "Broad-leaved forest": "Forest",
    "Coniferous forest": "Forest",
    "Mixed forest": "Forest",
    "Natural grassland and sparsely vegetated areas": "Grassland and shrub",
    "Moors, heathland and sclerophyllous vegetation": "Grassland and shrub",
    "Transitional woodland, shrub": "Grassland and shrub",
    "Beaches, dunes, sands": "Bare and sparse",
    "Inland wetlands": "Wetland",
    "Coastal wetlands": "Wetland",
    "Inland waters": "Water",
    "Marine waters": "Water",
}
"""The 6-class merge used for the reference-replication metric (§8.2).

Reconstructed from the CORINE level-1 groupings the reference study describes
rather than recovered from it, so §4.1's honesty note applies: results under this
merge are reported as *comparable, not identical*, and our claimed delta is
anchored to our own measured zero-shot baseline."""


def ben6_labels(labels: Sequence[str]) -> list[str]:
    """Merge 19-class labels into the 6-class space, order preserved, deduped."""
    merged: list[str] = []
    for label in labels:
        name = BEN6_MERGE.get(label)
        if name is not None and name not in merged:
            merged.append(name)
    return merged


def _require(row: Mapping[str, Any], *names: str) -> tuple[Any, ...]:
    """Pull required fields off a dataset row.

    Raises:
        CorpusError: A field the builder needs is absent.
    """
    missing = [name for name in names if row.get(name) is None]
    if missing:
        raise CorpusError(f"dataset row is missing {missing}; got keys {sorted(row)}")
    return tuple(row[name] for name in names)


def _split_of(row: Mapping[str, Any], default: str = "train") -> str:
    """The source split this row declares."""
    return str(row.get("split") or row.get("source_split") or default)


def _guard_quarantine(sample_id: str, source_split: str) -> None:
    """Refuse a row from a split reserved for Phase 8.

    Raises:
        LeakageError: The row belongs to a quarantined split (§4.2, §4.7).
    """
    if source_split in QUARANTINED_SPLITS:
        raise LeakageError(
            f"{sample_id}: source split {source_split!r} is quarantined for Phase 8 "
            "and must never enter the corpus"
        )


def _corpus_split(source_split: str) -> Literal["train", "val"]:
    """Map an official source split onto the corpus split it feeds (§5)."""
    return "val" if source_split.startswith(("val", "dev")) else "train"


def _meta_for(
    row: Mapping[str, Any],
    source_split: str,
    *,
    sensor: str | None = None,
    gsd_m: float | None = None,
    labels: Sequence[str] = (),
    augmentations: Sequence[str] = (),
) -> SampleMeta:
    """Assemble one sample's provenance block."""
    return SampleMeta(
        sensor=sensor if sensor is not None else row.get("sensor"),
        gsd_m=gsd_m if gsd_m is not None else row.get("gsd_m"),
        labels=list(labels),
        split=_corpus_split(source_split),
        source_split=source_split,
        augmentations=list(augmentations),
        image_sha256=row.get("image_sha256"),
        phash=row.get("phash"),
    )


def _join(labels: Sequence[str]) -> str:
    """Render a class list as prose: ``"a, b and c"``."""
    if not labels:
        return "no mapped class"
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])} and {labels[-1]}"


# --- 4.1 BigEarthNet-v2 -------------------------------------------------------

BEN_SENSOR: Final[str] = "Sentinel-2 (BigEarthNet-v2, 12-band)"
BEN_SAR_SENSOR: Final[str] = "Sentinel-1 GRD"
BEN_GSD_M: Final[float] = 10.0

_BEN_OPTICAL_VIEWS: Final[tuple[ViewId, ...]] = (
    ViewId.TC,
    ViewId.FCIR,
    ViewId.SWIR,
    ViewId.NDVI,
    ViewId.NDBI,
)
_BEN_SAR_VIEWS: Final[tuple[ViewId, ...]] = (ViewId.SARFC, ViewId.SARDB)


def _ben_views(row: Mapping[str, Any]) -> list[SourceView]:
    """The cross-modal view stack for one BigEarthNet patch (§2.4).

    ``view_paths`` maps a view id to the file the Phase 2 pass pre-rendered; a
    view with no path was unavailable for this patch and is simply absent —
    never substituted (§2.2).
    """
    paths: Mapping[str, Any] = row.get("view_paths") or {}
    views = [
        SourceView(
            view_id=view_id,
            path=paths[view_id.value],
            modality=Modality.OPTICAL,
            sensor=BEN_SENSOR,
            role=ImageRole.OPTICAL,
        )
        for view_id in _BEN_OPTICAL_VIEWS
        if view_id.value in paths
    ]
    views.extend(
        SourceView(
            view_id=view_id,
            path=paths[view_id.value],
            modality=Modality.SAR,
            sensor=BEN_SAR_SENSOR,
            role=ImageRole.SAR,
        )
        for view_id in _BEN_SAR_VIEWS
        if view_id.value in paths
    )
    if not views:
        raise CorpusError(f"BigEarthNet row {row.get('patch_id')!r} has no rendered views")
    return views


def _radar_built_up_answer(facts: Mapping[str, float | str]) -> str | None:
    """The radar built-up answer, when the sheet carries the radar evidence."""
    vv_key = "sar_backscatter_analyzer.sigma0_vv_db_mean"
    ratio_key = "sar_backscatter_analyzer.vv_vh_ratio_db_mean"
    vv = facts.get(vv_key)
    if not isinstance(vv, int | float):
        return None
    verdict = (
        "bright, structured returns consistent with built-up surfaces"
        if float(vv) > -8.0
        else "moderate returns more consistent with vegetated or open terrain than "
        "with dense structures"
    )
    sentence = f"Mean σ⁰_VV is {cite(vv_key, vv)}, {verdict}"
    ratio = facts.get(ratio_key)
    if isinstance(ratio, int | float):
        sentence += f", and the VV/VH ratio averages {cite(ratio_key, ratio)}"
    return sentence + "."


def _cross_modal_summary(labels: Sequence[str], facts: Mapping[str, float | str]) -> str:
    """Two sentences naming both sensors, citing whatever each measured."""
    optical = [
        f"the optical views show {_join(labels).lower()}",
    ]
    ndvi = facts.get("spectral_index_analyzer.ndvi_mean")
    if isinstance(ndvi, int | float):
        optical.append(
            f"with a mean NDVI of {cite('spectral_index_analyzer.ndvi_mean', ndvi)}"
        )
    radar = _radar_built_up_answer(facts)
    first = f"Across this scene {', '.join(optical)}."
    return first if radar is None else f"{first} On the radar side, {radar[0].lower()}{radar[1:]}"


def from_bigearthnet(
    row: Mapping[str, Any],
    *,
    rng: random.Random | None = None,
    augment: bool = True,
) -> list[CorpusSample]:
    """Build every task this BigEarthNet-v2 patch supports (§4.1).

    Args:
        row: One reBEN metadata row: ``patch_id``, ``labels``, ``view_paths``,
            optionally ``fact_sheet``, ``split`` and ``season``.
        rng: Seeded source for negative sampling and augmentation draws.
        augment: Apply the §7.3 augmentations.

    Returns:
        A ``SCENE_CLASSIFY`` sample, a balanced yes/no presence sample, and —
        where the patch has radar views — a cross-modal summary.
    """
    random_ = rng or random.Random(0)
    patch_id, labels = _require(row, "patch_id", "labels")
    sample_id_base = f"ben2:{patch_id}"
    source_split = _split_of(row)
    _guard_quarantine(sample_id_base, source_split)

    views = _ben_views(row)
    facts: dict[str, float | str] = dict(row.get("fact_sheet") or {})
    tags: list[str] = []
    if augment:
        drawn = plan_augmentations(CorpusSource.BIGEARTHNET_V2, random_)
        views, facts, tags = apply_augmentations(drawn, views, facts)

    class_list = [str(label) for label in labels]
    has_sar = any(view.modality is Modality.SAR for view in views)
    pair_type = PairType.CROSS_MODAL if has_sar else PairType.SINGLE
    meta = _meta_for(
        row,
        source_split,
        sensor=f"{BEN_SENSOR} + {BEN_SAR_SENSOR}" if has_sar else BEN_SENSOR,
        gsd_m=BEN_GSD_M,
        labels=class_list,
        augmentations=tags,
    )

    def make(task: TaskType, question: str, answer: str, suffix: str) -> CorpusSample:
        return assemble(
            sample_id=f"{sample_id_base}:{suffix}",
            source=CorpusSource.BIGEARTHNET_V2,
            task=task,
            pair_type=pair_type,
            views=views,
            question=question,
            answer=answer,
            fact_sheet=facts,
            meta=meta.model_copy(deep=True),
        )

    samples = [
        make(
            TaskType.SCENE_CLASSIFY,
            "Which land-cover classes are present in this scene?",
            f"The scene contains {_join(class_list)}.",
            "classify",
        )
    ]

    present = random_.random() < 0.5 and bool(class_list)
    if present:
        target = random_.choice(class_list)
        verdict = f"Yes, {target.lower()} is present in this scene."
    else:
        absent = [name for name in BEN19_CLASSES if name not in class_list]
        target = random_.choice(absent or list(BEN19_CLASSES))
        verdict = (
            f"No, there is no {target.lower()} in this scene; the mapped cover is "
            f"{_join(class_list)}."
        )
    samples.append(
        make(TaskType.VQA, f"Is there {target.lower()} in this image?", verdict, "presence")
    )

    if has_sar:
        radar = _radar_built_up_answer(facts)
        if radar is not None:
            samples.append(
                make(
                    TaskType.CROSS_MODAL_VQA,
                    "Does the radar imagery indicate built-up structures here?",
                    radar,
                    "radar_builtup",
                )
            )
        samples.append(
            make(
                TaskType.CROSS_MODAL_COMPARE,
                "Summarise the land cover, citing both the optical and radar evidence.",
                _cross_modal_summary(class_list, facts),
                "crossmodal_summary",
            )
        )
    return samples


# --- 4.1b BigEarthNet.txt — the QA annotation table ---------------------------

BEN_TXT_TASKS: Final[dict[str, TaskType]] = {
    "binary": TaskType.VQA,
    "mcq": TaskType.VQA,
    "captioning": TaskType.CAPTION,
    "bounding box": TaskType.GROUNDING,
}
"""``type`` column -> our task. The reBEN text export supervises four annotation
types over the same S1/S2 patches the label table covers (§4.1)."""

_BEN_TXT_BOX: Final[re.Pattern[str]] = re.compile(
    r"\[\s*(?P<x1>[\d.]+)\s+(?P<y1>[\d.]+)\s*,\s*(?P<x2>[\d.]+)\s+(?P<y2>[\d.]+)\s*\]"
)
"""``[0.64 0.0, 1.0 0.71]`` — corners separated by a comma, coordinates within a
corner by a space, all normalised to 0-1. Nothing else in the corpus uses this
form, and it is never emitted: it is converted to the one serialiser's output."""

_BEN_TXT_REF: Final[re.Pattern[str]] = re.compile(r"<ref>(?P<text>.*?)</ref>", re.DOTALL)


def noncanonical_grounding(samples: Iterable[CorpusSample]) -> list[tuple[str, str]]:
    """``(id, answer)`` of every GROUNDING sample whose answer is not canonical.

    The corpus-level check of ML_PIPELINE_RECOVERY_PLAN §2.5: run over a
    written JSONL (``iter_corpus``) after every build, and in the test suite
    over whatever the adapters produce. An empty list is the only acceptable
    result.
    """
    return [
        (sample.id, sample.assistant)
        for sample in samples
        if sample.task is TaskType.GROUNDING and not is_canonical_answer(sample.assistant)
    ]


class UnlabelledGroundingError(CorpusError):
    """A grounding row names no class, so its box cannot be written canonically.

    The ``<point>(x, y)</point>`` questions of the reBEN text export ask for
    "the land cover class instance at" a point and never say which class. The
    system prompt's format requires a NAME on every box; without one the row
    is excluded rather than emitted bare (ML_PIPELINE_RECOVERY_PLAN §2.5).
    """


def grounding_label(row: Mapping[str, Any]) -> str | None:
    """The NAME a BEN-txt grounding box is tagged with, or None when unresolvable.

    In order: the ``<ref>…</ref>`` the question carries; failing that, the
    patch's own label set when — and only when — it holds exactly one class,
    which is the one case the box's class is not ambiguous.
    """
    reference = _BEN_TXT_REF.search(str(row.get("input") or ""))
    if reference:
        text = reference.group("text").strip()
        if text:
            return text
    labels = [str(label).strip() for label in (row.get("labels") or ()) if str(label).strip()]
    if len(labels) == 1:
        return labels[0]
    return None


def parse_ben_txt_box(answer: str, label: str | None = None) -> list[NormalisedBox]:
    """Convert reBEN's ``[x1 y1, x2 y2]`` answer onto the 0-1000 frame.

    Its coordinates are already normalised, so this is a scale by
    :data:`~satquery.models.prompts.box_format.BOX_SCALE` rather than a
    projection — but it still goes through :class:`NormalisedBox`, because that
    is what validates the box and what :func:`serialise` will re-emit.
    """
    boxes: list[NormalisedBox] = []
    for match in _BEN_TXT_BOX.finditer(answer):
        corners = [float(match.group(name)) for name in ("x1", "y1", "x2", "y2")]
        scaled = [max(0, min(BOX_SCALE, round(value * BOX_SCALE))) for value in corners]
        try:
            boxes.append(
                NormalisedBox(
                    x_min=min(scaled[0], scaled[2]),
                    y_min=min(scaled[1], scaled[3]),
                    x_max=max(scaled[0], scaled[2]),
                    y_max=max(scaled[1], scaled[3]),
                    label=label,
                )
            )
        except BoxFormatError:
            # A zero-extent annotation is one bad box, not a bad dataset.
            continue
    return boxes


def _ben_txt_answer(row: Mapping[str, Any], task: TaskType) -> str:
    """The supervised answer for one annotation row.

    Answers are taken **verbatim** apart from sentence casing. Rewriting an
    mcq's letter into its option text, or a binary's "yes" into a sentence
    quoting the area band the question mentions, would inject numbers that no
    tool in our plan measured — and the citation audit exists precisely to stop
    that entering the corpus.
    """
    answer = str(row["output"]).strip()
    if task is TaskType.GROUNDING:
        label = grounding_label(row)
        if label is None:
            raise UnlabelledGroundingError(
                f"grounding row {row.get('ID')!r} names no class for its box "
                f"({str(row.get('input'))[:80]!r}); excluded rather than emitted bare"
            )
        boxes = parse_ben_txt_box(answer, label)
        if not boxes:
            raise CorpusError(f"unparseable bounding-box answer {answer!r}")
        return serialise_answer(boxes)
    if task is TaskType.CAPTION:
        return answer
    lowered = answer.lower()
    if lowered in {"yes", "no"}:
        return f"{lowered.capitalize()}."
    return answer


def from_bigearthnet_txt(
    row: Mapping[str, Any],
    *,
    rng: random.Random | None = None,
    augment: bool = True,
    skip_uncitable: bool = True,
) -> list[CorpusSample]:
    """Build one sample from a BigEarthNet.txt annotation row (§4.1).

    This is the *text* export of reBEN: one row per question/answer pair over an
    S1/S2 patch, with the columns ``ID``, ``patch_id``, ``s1_name``, ``input``,
    ``output``, ``type``, ``category``, ``split`` and the patch's geography. It
    carries no class-label vector, so :func:`from_bigearthnet` — which builds
    ``SCENE_CLASSIFY`` and presence questions out of the 19-class labels — cannot
    read it, and the two are dispatched on the row's shape rather than merged.

    The FactSheet is empty: this export publishes annotations, not measurements.
    That is not a gap to paper over — the system prompt then states plainly that
    no number may be quoted, which is the correct instruction for these answers.
    Numeric supervision comes from ``evidence_qa``, where the numbers are real.

    Args:
        row: One annotation row.
        rng: Seeded source for the §7.3 augmentation draws.
        augment: Apply the augmentations.
        skip_uncitable: Return no sample for an answer whose numbers cannot be
            attributed, instead of failing the build. Captions occasionally
            quote a percentage the export does not otherwise expose; with 9.6 M
            rows available for an 18 k budget, dropping those is cheaper than
            weakening the audit for every source.

    Returns:
        One sample, or none when the row is skipped — including every
        ``bounding box`` row whose class cannot be named (see
        :func:`grounding_label`): the prompt's format has no unnamed box.

    Raises:
        CorpusError: The row is missing a required column, or its annotation
            type is not one of the four the export documents.
    """
    random_ = rng or random.Random(0)
    row_id, patch_id, question, answer, kind = _require(
        row, "ID", "patch_id", "input", "output", "type"
    )
    sample_id = f"ben2_txt:{patch_id}:{row_id}"
    source_split = _split_of(row)
    _guard_quarantine(sample_id, source_split)

    task = BEN_TXT_TASKS.get(str(kind).strip().lower())
    if task is None:
        raise CorpusError(
            f"{sample_id}: unknown annotation type {kind!r}; expected one of "
            f"{sorted(BEN_TXT_TASKS)}"
        )

    views = _ben_views(row)
    tags: list[str] = []
    if augment:
        views, _, tags = apply_augmentations(
            plan_augmentations(CorpusSource.BIGEARTHNET_V2, random_), views, {}
        )
    has_sar = any(view.modality is Modality.SAR for view in views)

    try:
        text = _ben_txt_answer(row, task)
    except UnlabelledGroundingError:
        # Not a citation problem and not optional: a box with no name cannot be
        # written in the prompt's format, whatever the caller's policy.
        return []
    except CorpusError:
        if skip_uncitable:
            return []
        raise

    try:
        return [
            assemble(
                sample_id=sample_id,
                source=CorpusSource.BIGEARTHNET_V2,
                task=task,
                pair_type=PairType.CROSS_MODAL if has_sar else PairType.SINGLE,
                views=views,
                question=str(question).strip(),
                answer=text,
                meta=_meta_for(
                    row,
                    source_split,
                    sensor=f"{BEN_SENSOR} + {BEN_SAR_SENSOR}" if has_sar else BEN_SENSOR,
                    gsd_m=BEN_GSD_M,
                    augmentations=tags,
                ),
                audit=task is not TaskType.GROUNDING,
            )
        ]
    except CitationError:
        if skip_uncitable:
            return []
        raise


# --- 4.2 VRSBench -------------------------------------------------------------


def _vhr_view(row: Mapping[str, Any], key: str = "image_path") -> SourceView:
    """The single true-colour view a 3-channel VHR benchmark image renders to."""
    (path,) = _require(row, key)
    return SourceView(
        view_id=ViewId.TC,
        path=path,
        modality=Modality.OPTICAL,
        sensor=row.get("sensor"),
        role=ImageRole.SINGLE,
    )


def _boxes_from(row: Mapping[str, Any], boxes: Sequence[Sequence[float]], label: str | None
                ) -> list[NormalisedBox]:
    """Normalise pixel boxes onto the 0-1000 frame the serialiser emits.

    Raises:
        CorpusError: The row does not carry the frame the boxes were measured on.
    """
    width, height = row.get("width"), row.get("height")
    if not width or not height:
        raise CorpusError(
            f"grounding row {row.get('image_id')!r} needs width and height to "
            "normalise its boxes"
        )
    return [from_pixels(box, int(width), int(height), label=label) for box in boxes]


def _locate_question(expression: str) -> str:
    """Phrase a grounding request around whatever form the expression takes.

    The two corpora phrase their references differently: DIOR-RSVG gives a noun
    phrase ("the white aircraft on the left apron"), VRSBench gives a full
    declarative sentence ("The toll station is positioned at the center."). One
    template cannot carry both — "Locate The toll station is positioned at the
    center." is the kind of malformed instruction a model learns to imitate.
    """
    text = expression.strip()
    if text.endswith(".") and len(text.split()) > 4:
        return f"Locate the object described here: {text}"
    text = text.rstrip(".")
    if text.lower().startswith(("the ", "a ", "an ")):
        return f"Locate {text}."
    return f"Locate the {text}."


def _cited_count(answer: str) -> tuple[str, dict[str, float]] | None:
    """Supervise a bare integer answer as a measured, cited count.

    A count is a measurement, and the deterministic counter is the tool that
    would produce it at serving time — so the answer is written in the same
    citation form as every other number, rather than as a naked digit the
    citation audit would (correctly) reject.
    """
    text = answer.strip().rstrip(".")
    try:
        count = int(text)
    except ValueError:
        return None
    key = "object_counter.count"
    return f"The measured count is {cite(key, count)}.", {key: float(count)}


def from_vrsbench(
    row: Mapping[str, Any],
    *,
    rng: random.Random | None = None,
    augment: bool = True,
    skip_uncitable: bool = True,
) -> list[CorpusSample]:
    """Build every sample one VRSBench image supports (§4.2).

    The published row is one image with three nested annotation sets, not three
    flat rows: ``caption`` (a detailed description), ``qa_pairs``
    (``{ques_id, question, type, answer}``) and ``objects``
    (``{obj_id, referring_sentence, obj_cls, obj_coord, ...}``). One image
    therefore yields one caption sample, one VQA sample per question, and one
    grounding sample per referred object.

    ``obj_coord`` is already normalised to 0-1, so grounding answers scale onto
    the 0-1000 frame and are emitted by the shared serialiser — the same bytes
    ``tools/text_grounding.py`` parses back (§4.5).

    Args:
        row: One VRSBench row.
        rng: Seeded source for the §7.3 augmentation draws.
        augment: Apply the augmentations.
        skip_uncitable: Drop an annotation whose answer states a number no tool
            measured, rather than failing the build.

    Raises:
        LeakageError: The row comes from the quarantined VRSBench test split.
    """
    random_ = rng or random.Random(0)
    image = row.get("image") or row.get("image_id")
    if not image:
        raise CorpusError(f"VRSBench row has no image name; got keys {sorted(row)}")
    image_id = str(image)
    sample_id = f"vrsbench:{Path(image_id).stem}"
    source_split = _split_of(row)
    _guard_quarantine(sample_id, source_split)

    views = [_vhr_view(dict(row), "image_path")] if row.get("image_path") else [
        SourceView(
            view_id=ViewId.TC,
            path=image_id,
            modality=Modality.OPTICAL,
            sensor=row.get("sensor"),
            role=ImageRole.SINGLE,
        )
    ]
    tags: list[str] = []
    if augment:
        _, _, tags = apply_augmentations(
            plan_augmentations(CorpusSource.VRSBENCH, random_), views, {}
        )
    meta = _meta_for(row, source_split, augmentations=tags)

    def make(
        suffix: str,
        task: TaskType,
        question: str,
        answer: str,
        facts: Mapping[str, float | str] | None = None,
        audit: bool = True,
    ) -> CorpusSample | None:
        try:
            return assemble(
                sample_id=f"{sample_id}:{suffix}",
                source=CorpusSource.VRSBENCH,
                task=task,
                pair_type=PairType.SINGLE,
                views=views,
                question=question,
                answer=answer,
                fact_sheet=facts,
                meta=meta.model_copy(deep=True),
                audit=audit,
            )
        except CitationError:
            if skip_uncitable:
                return None
            raise

    samples: list[CorpusSample | None] = []
    caption = str(row.get("caption") or "").strip()
    if caption:
        samples.append(make("caption", TaskType.CAPTION, "Describe this scene.", caption))

    for pair in row.get("qa_pairs") or ():
        question = str(pair.get("question") or "").strip()
        answer = str(pair.get("answer") or "").strip()
        if not question or not answer:
            continue
        counted = _cited_count(answer)
        if counted is not None:
            text, facts = counted
            samples.append(
                make(f"qa{pair.get('ques_id', len(samples))}", TaskType.COUNT, question,
                     text, facts)
            )
        else:
            samples.append(
                make(
                    f"qa{pair.get('ques_id', len(samples))}",
                    TaskType.VQA,
                    question,
                    answer[0].upper() + answer[1:] if answer else answer,
                )
            )

    for obj in row.get("objects") or ():
        coordinates = obj.get("obj_coord")
        expression = str(obj.get("referring_sentence") or "").strip()
        if not coordinates or len(coordinates) != 4 or not expression:
            continue
        # The class where VRSBench gives one, the referring expression where it
        # does not: a grounding answer names every box (§2.5), never bare.
        label = str(obj.get("obj_cls") or "").strip() or expression
        scaled = [
            max(0, min(BOX_SCALE, round(float(value) * BOX_SCALE))) for value in coordinates
        ]
        try:
            box = NormalisedBox(
                x_min=min(scaled[0], scaled[2]),
                y_min=min(scaled[1], scaled[3]),
                x_max=max(scaled[0], scaled[2]),
                y_max=max(scaled[1], scaled[3]),
                label=label,
            )
        except BoxFormatError:
            continue
        samples.append(
            make(
                f"ground{obj.get('obj_id', len(samples))}",
                TaskType.GROUNDING,
                _locate_question(expression),
                serialise_answer([box]),
                audit=False,
            )
        )

    return [sample for sample in samples if sample is not None]


# --- 4.3 RSVQA-HR -------------------------------------------------------------

RSVQA_TYPES: Final[tuple[str, ...]] = ("presence", "comparison", "count", "area")
"""The four question types §4.3 balances across. Sampling the raw distribution
instead would teach the model to guess the majority type."""


def _rsvqa_area_m2(sample_id: str, answer: Any) -> float:
    """The square metres an RSVQA-HR ``area`` answer states.

    The release writes them as ``"521m2"`` — value and unit, no space. The unit
    is stripped rather than kept, because :func:`cite` writes it back from the
    scalar's own name: leaving it in produces ``"521m2 m²"``, which is both
    wrong and, since the validator reads the unit off the key, uncitable.

    Raises:
        CorpusError: The answer is not a number followed by ``m2``.
    """
    text = str(answer).strip().lower().removesuffix("m2").strip()
    try:
        return float(text)
    except ValueError as error:
        raise CorpusError(
            f"{sample_id}: area answer {answer!r} is not a number of square metres"
        ) from error


def from_rsvqa(row: Mapping[str, Any]) -> CorpusSample:
    """Build one RSVQA-HR sample (§4.3).

    A ``count`` question becomes a ``COUNT`` task carrying an
    ``object_counter.count`` fact, and its answer cites that key. RSVQA's own
    answer is a bare integer; supervising it in citation form is what keeps the
    counting task inside the same evidence discipline as every other task, and
    the count really is a measurement rather than a guess.

    An ``area`` question gets the same treatment against
    ``semantic_segmenter.area_m2``. It has to: RSVQA answers these with a bare
    ``"521m2"``, and :func:`audit_answer` rejects an assistant turn stating a
    number the sheet does not support — which made ``area`` a build error the
    moment RSVQA-HR was wired up, on the 3 040 train rows answering ``0m2``
    alone. Passing it through uncited was never the alternative: an area *is* a
    measurement, and ``semantic_segmenter`` is the tool that measures one at
    inference, under that exact scalar name.

    ``presence`` and ``comp`` answer yes or no and carry no number, so they stay
    plain ``VQA``.
    """
    image_id, question, answer = _require(row, "image_id", "question", "answer")
    question_type = str(row.get("type") or "presence").lower()
    source_split = _split_of(row)
    sample_id = f"rsvqa_hr:{image_id}:{row.get('question_id', question_type)}"
    _guard_quarantine(sample_id, source_split)

    views = [_vhr_view(row)]
    meta = _meta_for(row, source_split, gsd_m=row.get("gsd_m", 0.15))

    if question_type == "count":
        try:
            count = int(str(answer).strip())
        except ValueError as error:
            raise CorpusError(f"{sample_id}: count answer {answer!r} is not an integer") from error
        key = "object_counter.count"
        return assemble(
            sample_id=sample_id,
            source=CorpusSource.RSVQA_HR,
            task=TaskType.COUNT,
            pair_type=PairType.SINGLE,
            views=views,
            question=str(question).strip(),
            answer=f"The measured count is {cite(key, count)}.",
            fact_sheet={key: float(count)},
            meta=meta,
        )

    if question_type == "area":
        key = "semantic_segmenter.area_m2"
        area = _rsvqa_area_m2(sample_id, answer)
        return assemble(
            sample_id=sample_id,
            source=CorpusSource.RSVQA_HR,
            task=TaskType.VQA,
            pair_type=PairType.SINGLE,
            views=views,
            question=str(question).strip(),
            answer=f"The measured area is {cite(key, area)}.",
            fact_sheet={key: area},
            meta=meta,
        )

    return assemble(
        sample_id=sample_id,
        source=CorpusSource.RSVQA_HR,
        task=TaskType.VQA,
        pair_type=PairType.SINGLE,
        views=views,
        question=str(question).strip(),
        answer=str(answer).strip().capitalize().rstrip(".") + ".",
        meta=meta,
    )


# --- 4.4 CDVQA ----------------------------------------------------------------


def _bi_temporal_views(row: Mapping[str, Any]) -> list[SourceView]:
    """The pre/post true-colour pair a change sample renders to (§2.4)."""
    pre, post = _require(row, "pre_path", "post_path")
    return [
        SourceView(
            view_id=ViewId.TC,
            path=pre,
            modality=Modality.OPTICAL,
            sensor=row.get("sensor"),
            role=ImageRole.PRE,
            acquisition_time=row.get("pre_time"),
        ),
        SourceView(
            view_id=ViewId.TC,
            path=post,
            modality=Modality.OPTICAL,
            sensor=row.get("sensor"),
            role=ImageRole.POST,
            acquisition_time=row.get("post_time"),
        ),
    ]


def from_cdvqa(row: Mapping[str, Any]) -> CorpusSample:
    """Build one CDVQA bi-temporal change-VQA sample (§4.4)."""
    pair_id, question, answer = _require(row, "pair_id", "question", "answer")
    source_split = _split_of(row)
    sample_id = f"cdvqa:{pair_id}:{row.get('question_id', 'q')}"
    _guard_quarantine(sample_id, source_split)

    facts: dict[str, float | str] = dict(row.get("fact_sheet") or {})
    text = str(answer).strip().capitalize().rstrip(".")
    changed = facts.get("change_statistics.changed_area_pct")
    if isinstance(changed, int | float):
        text += (
            f", over {cite('change_statistics.changed_area_pct', changed)} of the scene"
        )
    return assemble(
        sample_id=sample_id,
        source=CorpusSource.CDVQA,
        task=TaskType.CHANGE_VQA,
        pair_type=PairType.BI_TEMPORAL,
        views=_bi_temporal_views(row),
        question=str(question).strip(),
        answer=text + ".",
        fact_sheet=facts,
        meta=_meta_for(row, source_split),
    )


# --- 4.5 DIOR-RSVG ------------------------------------------------------------


def from_dior_rsvg(
    row: Mapping[str, Any], *, rng: random.Random | None = None, augment: bool = True
) -> CorpusSample:
    """Build one DIOR-RSVG referring-expression grounding sample (§4.5).

    The answer is emitted by the shared serialiser and by nothing else, so what
    the model is trained to write is exactly what ``tools/text_grounding.py``
    parses back.
    """
    random_ = rng or random.Random(0)
    image_id, expression = _require(row, "image_id", "expression")
    boxes = row.get("boxes") or ([row["bbox"]] if row.get("bbox") else None)
    if not boxes:
        raise CorpusError(f"dior_rsvg:{image_id} carries no box for its expression")
    source_split = _split_of(row)
    sample_id = f"dior_rsvg:{image_id}:{row.get('expression_id', 0)}"
    _guard_quarantine(sample_id, source_split)

    views = [_vhr_view(row)]
    tags: list[str] = []
    if augment:
        _, _, tags = apply_augmentations(
            plan_augmentations(CorpusSource.DIOR_RSVG, random_), views, {}
        )
    return assemble(
        sample_id=sample_id,
        source=CorpusSource.DIOR_RSVG,
        task=TaskType.GROUNDING,
        pair_type=PairType.SINGLE,
        views=views,
        question=_locate_question(str(expression)),
        answer=serialise_answer(_boxes_from(row, boxes, str(expression).strip())),
        slots={"target_class": str(expression).strip()},
        meta=_meta_for(row, source_split, gsd_m=row.get("gsd_m", 0.5)),
        audit=False,
    )


# --- 4.6 evidence_qa ----------------------------------------------------------


def describe_ndvi(value: float) -> str:
    """Say what a mean NDVI indicates, in the fixed domain the views render on."""
    if value >= 0.6:
        return "dense, healthy vegetation across most of the scene"
    if value >= 0.3:
        return "moderate vegetation cover, consistent with cropland or mixed cover"
    if value >= 0.1:
        return "sparse vegetation over largely bare or built surfaces"
    if value >= -0.1:
        return "essentially unvegetated ground"
    return "open water or another non-vegetated surface"


def describe_ndbi(value: float) -> str:
    """Say what a mean NDBI indicates."""
    if value >= 0.1:
        return "a strong built-up signal"
    if value >= -0.05:
        return "an ambiguous built-up signal, which bare soil also produces"
    return "a low built-up signal"


def describe_backscatter(value: float) -> str:
    """Say what a mean σ⁰_VV indicates."""
    if value >= -6.0:
        return "bright returns typical of structures or rough surfaces"
    if value >= -14.0:
        return "moderate returns typical of vegetated terrain"
    return "low returns typical of smooth surfaces such as open water"


def describe_change(value: float) -> str:
    """Say how substantial a changed-area fraction is."""
    if value >= 20.0:
        return "a large fraction of the scene"
    if value >= 5.0:
        return "a substantial but localised part of the scene"
    return "a small part of the scene"


_NDVI_KEY: Final[str] = "spectral_index_analyzer.ndvi_mean"
_NDBI_KEY: Final[str] = "spectral_index_analyzer.ndbi_mean"
_VV_KEY: Final[str] = "sar_backscatter_analyzer.sigma0_vv_db_mean"
_RATIO_KEY: Final[str] = "sar_backscatter_analyzer.vv_vh_ratio_db_mean"
_CHANGED_KEY: Final[str] = "change_statistics.changed_area_pct"

_UNMEASURED_QUESTIONS: Final[tuple[tuple[str, str], ...]] = (
    (
        "What is the mean surface temperature of this scene?",
        "This system did not measure surface temperature; no thermal band was "
        "available and no tool in this plan produced that number.",
    ),
    (
        "How many buildings are there in this scene?",
        "This system did not count objects in this scene, so there is no measured "
        "count to report.",
    ),
)
"""Refusal supervision. Rule 3 of the system prompt says an unmeasured quantity
must be declined rather than guessed, and a corpus in which every question has a
number behind it teaches the opposite. These pair a question with an answer that
states the absence — the only samples whose correct answer contains no citation."""


def build_evidence_qa(
    *,
    sample_id: str,
    views: Sequence[SourceView],
    fact_sheet: Mapping[str, float | str],
    pair_type: PairType = PairType.SINGLE,
    meta: SampleMeta | None = None,
    rng: random.Random | None = None,
    refusal_probability: float = 1.0,
) -> list[CorpusSample]:
    """Generate citation-behaviour samples from a FactSheet (§4.6).

    Every number in every answer is copied from *fact_sheet* and tagged with its
    key, so the supervision is correct and attributable by construction. This is
    the only source in the corpus that teaches citation at all — the five public
    ones teach vocabulary and task format, and all of their answers are unsourced
    prose.

    Args:
        sample_id: Stable id prefix for this scene's samples.
        views: The scene's rendered views, in slot order.
        fact_sheet: The measurements every answer is generated from.
        pair_type: Recorded in the prompt's context block.
        meta: Provenance, copied onto each sample.
        rng: Seeded source for template choice.
        refusal_probability: How often a scene also yields an "I did not measure
            that" sample. Refusal is worth teaching — rule 3 says an unmeasured
            quantity is declined rather than guessed — but it is worth teaching
            in proportion. At 1.0 every scene emits one, which is a third of this
            source and enough to bias a model toward declining answers it could
            have supported.

    Returns:
        One sample per template the sheet supports, and sometimes a refusal.
    """
    random_ = rng or random.Random(0)
    provenance = meta or SampleMeta()
    samples: list[CorpusSample] = []

    def make(task: TaskType, question: str, answer: str, suffix: str) -> None:
        samples.append(
            assemble(
                sample_id=f"{sample_id}:{suffix}",
                source=CorpusSource.EVIDENCE_QA,
                task=task,
                pair_type=pair_type,
                views=views,
                question=question,
                answer=answer,
                fact_sheet=fact_sheet,
                meta=provenance.model_copy(deep=True),
            )
        )

    ndvi = fact_sheet.get(_NDVI_KEY)
    if isinstance(ndvi, int | float):
        make(
            TaskType.VQA,
            "What is the mean NDVI, and what does it indicate?",
            f"The mean NDVI is {cite(_NDVI_KEY, ndvi)}, indicating "
            f"{describe_ndvi(float(ndvi))}.",
            "ndvi",
        )

    ndbi = fact_sheet.get(_NDBI_KEY)
    vv = fact_sheet.get(_VV_KEY)
    if isinstance(ndbi, int | float) and isinstance(vv, int | float):
        agree = (float(ndbi) >= 0.1) == (float(vv) >= -6.0)
        sentence = (
            f"{'Yes' if agree else 'Partially'}. NDBI averages {cite(_NDBI_KEY, ndbi)}, "
            f"{describe_ndbi(float(ndbi))}, while σ⁰_VV averages {cite(_VV_KEY, vv)}, "
            f"{describe_backscatter(float(vv))}"
        )
        ratio = fact_sheet.get(_RATIO_KEY)
        if isinstance(ratio, int | float):
            sentence += f" with a VV/VH ratio of {cite(_RATIO_KEY, ratio)}"
        sentence += (
            ", so the two sensors agree." if agree else ", so the two sensors disagree."
        )
        make(
            TaskType.CROSS_MODAL_VQA,
            "Do the optical and radar signals agree about built-up area?",
            sentence,
            "agreement",
        )

    changed = fact_sheet.get(_CHANGED_KEY)
    if isinstance(changed, int | float):
        sentence = (
            f"{cite(_CHANGED_KEY, changed)} of the scene changed, "
            f"{describe_change(float(changed))}"
        )
        pre = fact_sheet.get(f"{_NDBI_KEY}_pre")
        post = fact_sheet.get(f"{_NDBI_KEY}_post")
        if isinstance(pre, int | float) and isinstance(post, int | float):
            direction = "rose" if float(post) > float(pre) else "fell"
            reading = (
                "consistent with new built-up surface"
                if float(post) > float(pre)
                else "consistent with built-up surface giving way to vegetation or water"
            )
            sentence += (
                f". Mean NDBI {direction} from {cite(f'{_NDBI_KEY}_pre', pre)} to "
                f"{cite(f'{_NDBI_KEY}_post', post)}, {reading}"
            )
        make(
            TaskType.CHANGE_VQA,
            "How much of this scene changed, and in which direction?",
            sentence + ".",
            "change",
        )

    if random_.random() < refusal_probability:
        question, answer = random_.choice(_UNMEASURED_QUESTIONS)
        make(TaskType.VQA, question, answer, "unmeasured")
    return samples


# --------------------------------------------------------- dedup and leakage

_PHASH_SIDE: Final[int] = 32
_PHASH_LOW: Final[int] = 8
PHASH_BITS: Final[int] = _PHASH_LOW * _PHASH_LOW
NEAR_DUP_DISTANCE: Final[int] = 5
"""Hamming distance at or below which two pHashes are the same image (§4.7)."""

_BAND_BITS: Final[int] = 8
_BAND_COUNT: Final[int] = PHASH_BITS // _BAND_BITS
"""Eight 8-bit bands. Two hashes within Hamming 5 differ in at most five bands,
so at least three bands match exactly — bucketing on bands finds every near
duplicate without scanning 65,000 hashes per insert."""


def sha256_of(data: bytes | str | Path) -> str:
    """Hash raw bytes, or the contents of a file path, as lowercase hex."""
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha256(Path(data).read_bytes()).hexdigest()


def _dct_matrix(size: int) -> np.ndarray:
    """The orthonormal DCT-II basis, built here to avoid a scipy dependency."""
    rows = np.arange(size).reshape(-1, 1)
    cols = np.arange(size).reshape(1, -1)
    matrix = np.cos(np.pi * (2 * cols + 1) * rows / (2 * size))
    matrix *= np.sqrt(2.0 / size)
    matrix[0] *= np.sqrt(0.5)
    return matrix.astype(np.float64)


def phash(image: Any) -> int:
    """Perceptual hash of an image, as a 64-bit integer.

    Args:
        image: A path, a PIL image, or a 2-D/3-D numpy array.

    Standard DCT pHash: greyscale, 32x32, keep the low-frequency 8x8 block minus
    its DC term, and threshold on the median. Two crops of one scene, or the same
    scene at two JPEG qualities, land within a few bits of each other — which is
    the overlap between VRSBench and DIOR-RSVG that §4.7 exists to catch.
    """
    grey = _greyscale(image)
    basis = _dct_matrix(_PHASH_SIDE)
    coefficients = basis @ grey @ basis.T
    block = coefficients[:_PHASH_LOW, :_PHASH_LOW].flatten()
    median = float(np.median(block[1:]))
    bits = 0
    for index, value in enumerate(block):
        if value > median:
            bits |= 1 << index
    return bits


def _greyscale(image: Any) -> np.ndarray:
    """Coerce any accepted image form to a 32x32 float greyscale array."""
    from PIL import Image

    if isinstance(image, str | Path):
        with Image.open(image) as handle:
            pil = handle.convert("L").resize((_PHASH_SIDE, _PHASH_SIDE), Image.Resampling.BILINEAR)
        return np.asarray(pil, dtype=np.float64)
    if isinstance(image, np.ndarray):
        array = image.astype(np.float64)
        if array.ndim == 3:
            array = array.mean(axis=2)
        pil = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    else:  # a PIL image
        pil = image
    pil = pil.convert("L").resize((_PHASH_SIDE, _PHASH_SIDE), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float64)


def hamming(left: int, right: int) -> int:
    """Bit distance between two hashes."""
    return int(left ^ right).bit_count()


class DedupVerdict(StrEnum):
    """Why the deduplicator accepted or rejected an image."""

    NEW = "new"
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    QUARANTINED = "quarantined"


@dataclass
class DedupResult:
    """One deduplication decision, and what it collided with."""

    verdict: DedupVerdict
    key: str
    collided_with: str | None = None
    distance: int | None = None

    @property
    def accepted(self) -> bool:
        """True when the image may enter the corpus."""
        return self.verdict is DedupVerdict.NEW


class Deduplicator:
    """SHA-256 and pHash dedup with a hard quarantine check (§4.7).

    Runs in the order §4.7 specifies: exact hash first (cheap and total), then
    perceptual hash across *all* sources, then the quarantine assertion. A
    quarantined collision is not a warning — :meth:`assert_clean` raises, because
    a leaked benchmark is worse than no benchmark.
    """

    def __init__(self, max_distance: int = NEAR_DUP_DISTANCE) -> None:
        """Start an empty index."""
        self.max_distance = max_distance
        self._sha: dict[str, str] = {}
        self._bands: list[dict[int, list[tuple[int, str]]]] = [
            {} for _ in range(_BAND_COUNT)
        ]
        self._quarantine_sha: dict[str, str] = {}
        self._quarantine_bands: list[dict[int, list[tuple[int, str]]]] = [
            {} for _ in range(_BAND_COUNT)
        ]
        self.leaks: list[DedupResult] = []
        self.dropped: list[DedupResult] = []

    @staticmethod
    def _band_keys(value: int) -> list[int]:
        """Split a hash into its 8-bit bands."""
        return [(value >> (index * _BAND_BITS)) & 0xFF for index in range(_BAND_COUNT)]

    def quarantine(self, key: str, sha256: str | None = None, image_hash: int | None = None
                   ) -> None:
        """Register a test-split image that nothing in the corpus may match."""
        if sha256:
            self._quarantine_sha[sha256] = key
        if image_hash is not None:
            for band, bucket in zip(self._band_keys(image_hash), self._quarantine_bands,
                                    strict=True):
                bucket.setdefault(band, []).append((image_hash, key))

    def _nearest(
        self,
        bands: list[dict[int, list[tuple[int, str]]]],
        image_hash: int,
    ) -> tuple[str, int] | None:
        """The closest indexed hash within ``max_distance``, if any."""
        best: tuple[str, int] | None = None
        seen: set[str] = set()
        for band, bucket in zip(self._band_keys(image_hash), bands, strict=True):
            for candidate, key in bucket.get(band, ()):
                if key in seen:
                    continue
                seen.add(key)
                distance = hamming(candidate, image_hash)
                if distance <= self.max_distance and (best is None or distance < best[1]):
                    best = (key, distance)
        return best

    def check(
        self,
        key: str,
        sha256: str | None = None,
        image_hash: int | None = None,
        *,
        image_key: str | None = None,
    ) -> DedupResult:
        """Decide whether one image may enter the corpus, and index it if so.

        Args:
            key: Identifies the *sample* being checked; what a report names.
            sha256: The source image's exact hash.
            image_hash: The source image's pHash.
            image_key: Identifies the *image*, when several samples share one.
                Defaults to *key*, which is right when they are the same thing.

        **``image_key`` is what stops the corpus collapsing.** §4.7 dedups on
        source image, but four of the six sources publish many samples per
        image — VRSBench averages ten annotations per tile, CDVQA around forty
        questions per bi-temporal pair. Checked as if each sample were its own
        image, the second annotation of a tile collides with the first on an
        exact hash and is dropped, which takes CDVQA from 9 000 candidates to
        the ~215 distinct pairs behind them and puts every §5 target out of
        reach. Passing the same *image_key* for every sample of one image makes
        the repeat an accepted re-use rather than a duplicate; a collision
        between two *different* images is still exactly as fatal as before.
        """
        image_key = image_key if image_key is not None else key
        if sha256 and sha256 in self._quarantine_sha:
            result = DedupResult(DedupVerdict.QUARANTINED, key, self._quarantine_sha[sha256], 0)
            self.leaks.append(result)
            return result
        if image_hash is not None:
            hit = self._nearest(self._quarantine_bands, image_hash)
            if hit is not None:
                result = DedupResult(DedupVerdict.QUARANTINED, key, hit[0], hit[1])
                self.leaks.append(result)
                return result
        seen = self._sha.get(sha256) if sha256 else None
        if seen is not None and seen != image_key:
            result = DedupResult(DedupVerdict.EXACT_DUPLICATE, key, seen, 0)
            self.dropped.append(result)
            return result
        if seen is None and image_hash is not None:
            hit = self._nearest(self._bands, image_hash)
            if hit is not None:
                if hit[0] != image_key:
                    result = DedupResult(DedupVerdict.NEAR_DUPLICATE, key, hit[0], hit[1])
                    self.dropped.append(result)
                    return result
                # Same image, reached through the pHash index instead of the
                # exact one — an unhashed source, where pHash is all there is.
                seen = hit[0]

        if seen is not None:
            # A further sample of an image already admitted. Accepted, and
            # deliberately *not* re-indexed: appending its pHash again would
            # grow the near-dup index by a factor of the annotations per image
            # — 205 000 entries for VRSBench's 20 264 tiles — and slow every
            # subsequent lookup for no added coverage.
            return DedupResult(DedupVerdict.NEW, key, seen, 0)
        if sha256:
            self._sha[sha256] = image_key
        if image_hash is not None:
            for band, bucket in zip(self._band_keys(image_hash), self._bands, strict=True):
                bucket.setdefault(band, []).append((image_hash, image_key))
        return DedupResult(DedupVerdict.NEW, key)

    def assert_clean(self) -> None:
        """Fail the build if anything matched a quarantined test image.

        Raises:
            LeakageError: At least one corpus image collided with a test split.
        """
        if self.leaks:
            listed = ", ".join(
                f"{leak.key} ~ {leak.collided_with} (d={leak.distance})"
                for leak in self.leaks[:10]
            )
            raise LeakageError(
                f"{len(self.leaks)} corpus image(s) matched a quarantined test split: {listed}"
            )


def deduplicate(
    samples: Iterable[CorpusSample], dedup: Deduplicator | None = None
) -> tuple[list[CorpusSample], Deduplicator]:
    """Filter a stream of samples through :class:`Deduplicator`.

    Samples carrying neither hash pass through: the corpus builder does not have
    the pixels for every source, and dropping unhashed samples silently would
    shrink the corpus without saying so. What is hashed is checked.
    """
    index = dedup or Deduplicator()
    kept: list[CorpusSample] = []
    for sample in samples:
        image_hash = int(sample.meta.phash, 16) if sample.meta.phash else None
        result = index.check(
            sample.id,
            sample.meta.image_sha256,
            image_hash,
            image_key=image_key_of(sample),
        )
        if result.accepted:
            kept.append(sample)
    return kept, index


def image_key_of(sample: CorpusSample) -> str:
    """What identifies the *image* several samples of one scene share (§4.7).

    The first view's path, not the sample id. Both work for the ids the adapters
    happen to mint today, but the path is what the corpus line actually points
    at: two samples of one tile name the same file, two tiles never do, and the
    same bytes appearing under two paths is precisely the case the exact-hash
    check exists to catch rather than something this key should paper over.
    """
    return sample.views[0].path if sample.views else sample.id


# --------------------------------------------------------------- corpus build


class SourceReport(BaseModel):
    """What one source contributed to the corpus."""

    model_config = ConfigDict(extra="forbid")

    source: CorpusSource
    track: str
    target: int
    built: int
    kept: int
    train: int
    val: int


class CorpusReport(BaseModel):
    """The build's own record of what it produced (§5)."""

    model_config = ConfigDict(extra="forbid")

    sources: list[SourceReport]
    train: int
    val: int
    dropped_exact: int
    dropped_near: int
    seed: int
    prompt_version: str = DEFAULT_VERSION

    @property
    def total(self) -> int:
        """Every sample written."""
        return self.train + self.val


def write_jsonl(path: Path, samples: Iterable[CorpusSample]) -> int:
    """Write samples as JSONL, creating parent directories. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.to_json_line() + "\n")
            written += 1
    return written


def read_jsonl(path: Path) -> list[CorpusSample]:
    """Read a corpus file back into validated samples."""
    with path.open(encoding="utf-8") as handle:
        return [CorpusSample.model_validate_json(line) for line in handle if line.strip()]


def take(samples: Sequence[CorpusSample], target: int, rng: random.Random) -> list[CorpusSample]:
    """Subsample to *target* without replacement, preserving corpus order.

    Order is preserved rather than shuffled because the trainer shuffles anyway,
    and a stable order makes two builds of the same seed diffable.
    """
    if target >= len(samples):
        return list(samples)
    chosen = set(rng.sample(range(len(samples)), target))
    return [sample for index, sample in enumerate(samples) if index in chosen]


VAL_SHARE: Final[float] = 0.10
"""The share of each source's §5 target reserved for its validation split.

Subsampling used to draw one uniform reservoir per *source* and split it
afterwards, so a source's validation count was whatever the draw happened to
contain: with BigEarthNet's few-percent validation share and a 200-sample probe
target, the expected count was a handful and zero was a live outcome — and
nothing said so. The split is now decided when the target is allocated, so
every source that offers validation samples keeps at least one."""


def allocate_split(
    target: int | None, offered_train: int, offered_val: int, val_share: float = VAL_SHARE
) -> tuple[int, int]:
    """How many ``(train, val)`` samples one source keeps from what it was offered.

    Validation gets its reserved share first — at least one sample whenever any
    was offered and the target allows — and whatever a split cannot fill goes to
    the other, so a source that publishes only one split still fills its whole
    target. ``None`` keeps everything.
    """
    if target is None:
        return offered_train, offered_val
    if target <= 0:
        return 0, 0
    reserved_val = min(max(1, round(target * val_share)), target - 1) if target > 1 else 0
    val = min(offered_val, max(reserved_val, target - offered_train))
    train = min(offered_train, target - val)
    return train, val


class Reservoir:
    """A bounded uniform random sample of a stream (Vitter's Algorithm R).

    The corpus subsamples ~9.6 M candidate rows down to an 18 k budget, so
    :func:`take` — which needs the whole population as a ``Sequence`` before it
    can choose — is the wrong shape: materialising that population is what made
    ``scripts/build_corpus.py`` exhaust RAM and get OOM-killed. A reservoir makes
    the same uniform draw in one pass while holding at most *capacity* samples,
    so peak memory is set by the §5 composition target rather than by the size of
    the source dataset.

    Arrival order is restored on :meth:`collect` for the reason :func:`take`
    preserves corpus order: the trainer shuffles anyway, and a stable order makes
    two builds of the same seed diffable.
    """

    def __init__(self, capacity: int | None, rng: random.Random) -> None:
        """Start an empty reservoir. ``capacity=None`` keeps everything offered."""
        self.capacity = capacity
        self._rng = rng
        self._items: list[tuple[int, CorpusSample]] = []
        self.seen = 0

    def offer(self, sample: CorpusSample) -> None:
        """Show one sample to the reservoir, keeping or discarding it."""
        self.seen += 1
        if self.capacity is None or len(self._items) < self.capacity:
            self._items.append((self.seen, sample))
            return
        if self.capacity == 0:
            return
        index = self._rng.randrange(self.seen)
        if index < self.capacity:
            self._items[index] = (self.seen, sample)

    def collect(self) -> list[CorpusSample]:
        """The retained samples, back in the order they arrived."""
        return [sample for _, sample in sorted(self._items, key=lambda pair: pair[0])]


def _refuse_empty_split(
    source: CorpusSource, offered_val: int, kept_val: int, target: int | None
) -> None:
    """Fail a build that was offered validation samples and wrote none.

    Raises:
        CorpusError: *source* had validation candidates, a non-zero target, and
            still contributes nothing to ``val.jsonl``.
    """
    if offered_val and not kept_val and target != 0:
        raise CorpusError(
            f"{source.value}: {offered_val} validation sample(s) were offered and none "
            f"were kept (target {target}); the corpus would have no held-out data for "
            "this source"
        )


def build_corpus_streaming(
    samples: Iterable[tuple[CorpusSource, CorpusSample]],
    out_dir: Path,
    *,
    composition: Mapping[CorpusSource, int] | None = None,
    dedup: Deduplicator | None = None,
    seed: int = 42,
) -> CorpusReport:
    """Deduplicate, subsample, split and write a corpus from a *stream*.

    Same contract as :func:`build_corpus` — same report, same two files — but it
    never holds the built population. Samples arrive tagged with their source,
    are deduplicated as they pass (:class:`Deduplicator` is already incremental),
    and land in a per-source :class:`Reservoir` capped at that source's §5 target.
    Peak memory is therefore ``sum(composition.values())`` samples, whatever the
    input size.

    Prefer this over :func:`build_corpus` for anything reading a Hub dataset;
    :func:`build_corpus` remains the right call when the samples are already a
    list in memory, as in the tests.

    Args:
        samples: ``(source, sample)`` pairs, in any order and any interleaving.
        out_dir: Directory receiving ``train.jsonl`` and ``val.jsonl``.
        composition: Per-source targets; defaults to :data:`COMPOSITION`. A
            source with no entry is kept in full, which is unbounded memory.
        dedup: A pre-loaded deduplicator, typically already carrying the
            quarantined test hashes.
        seed: Subsampling seed, recorded in the report.

    Raises:
        LeakageError: A corpus image matched a quarantined test image.
    """
    targets = dict(composition or COMPOSITION)
    index = dedup or Deduplicator()
    rng = random.Random(seed)

    built: dict[CorpusSource, int] = dict.fromkeys(CorpusSource, 0)
    kept: dict[CorpusSource, int] = dict.fromkeys(CorpusSource, 0)
    # One reservoir per (source, split), each able to hold the whole target, so
    # a source's validation share is decided by :func:`allocate_split` rather
    # than by the dice. Worst-case memory is two targets per source instead of
    # one; in practice the validation stream is a small fraction of the train
    # stream and its reservoir never fills.
    reservoirs: dict[tuple[CorpusSource, str], Reservoir] = {
        (source, split): Reservoir(targets.get(source), rng)
        for source in CorpusSource
        for split in ("train", "val")
    }

    for source, sample in samples:
        built[source] += 1
        image_hash = int(sample.meta.phash, 16) if sample.meta.phash else None
        verdict = index.check(
            sample.id,
            sample.meta.image_sha256,
            image_hash,
            image_key=image_key_of(sample),
        )
        if not verdict.accepted:
            continue
        kept[source] += 1
        reservoirs[source, sample.meta.split].offer(sample)

    # Only now is anything held: at most one §5 budget's worth of samples.
    reports: list[SourceReport] = []
    train: list[CorpusSample] = []
    val: list[CorpusSample] = []
    for source in CorpusSource:
        train_pool = reservoirs[source, "train"].collect()
        val_pool = reservoirs[source, "val"].collect()
        train_n, val_n = allocate_split(targets.get(source), len(train_pool), len(val_pool))
        # A uniform subset of a uniform reservoir is still uniform.
        source_train = take(train_pool, train_n, rng)
        source_val = take(val_pool, val_n, rng)
        _refuse_empty_split(
            source, reservoirs[source, "val"].seen, len(source_val), targets.get(source)
        )
        train.extend(source_train)
        val.extend(source_val)
        reports.append(
            SourceReport(
                source=source,
                track=TRACK[source],
                target=targets.get(source, 0),
                built=built[source],
                kept=kept[source],
                train=len(source_train),
                val=len(source_val),
            )
        )

    index.assert_clean()
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    return CorpusReport(
        sources=reports,
        train=len(train),
        val=len(val),
        dropped_exact=sum(
            1 for d in index.dropped if d.verdict is DedupVerdict.EXACT_DUPLICATE
        ),
        dropped_near=sum(
            1 for d in index.dropped if d.verdict is DedupVerdict.NEAR_DUPLICATE
        ),
        seed=seed,
    )


def build_corpus(
    sources: Mapping[CorpusSource, Sequence[CorpusSample]],
    out_dir: Path,
    *,
    composition: Mapping[CorpusSource, int] | None = None,
    dedup: Deduplicator | None = None,
    seed: int = 42,
) -> CorpusReport:
    """Deduplicate, subsample to the §5 mix, split and write the corpus.

    Args:
        sources: Built samples per source, in any order.
        out_dir: Directory receiving ``train.jsonl`` and ``val.jsonl``.
        composition: Per-source targets; defaults to :data:`COMPOSITION`.
        dedup: A pre-loaded deduplicator, typically already carrying the
            quarantined test hashes.
        seed: Subsampling seed, recorded in the report.

    Raises:
        LeakageError: A corpus image matched a quarantined test image.
    """
    targets = dict(composition or COMPOSITION)
    index = dedup or Deduplicator()
    rng = random.Random(seed)

    reports: list[SourceReport] = []
    train: list[CorpusSample] = []
    val: list[CorpusSample] = []
    for source in CorpusSource:
        built = list(sources.get(source, ()))
        kept, index = deduplicate(built, index)
        train_pool = [s for s in kept if s.meta.split == "train"]
        val_pool = [s for s in kept if s.meta.split == "val"]
        train_n, val_n = allocate_split(targets.get(source), len(train_pool), len(val_pool))
        source_train = take(train_pool, train_n, rng)
        source_val = take(val_pool, val_n, rng)
        _refuse_empty_split(source, len(val_pool), len(source_val), targets.get(source))
        train.extend(source_train)
        val.extend(source_val)
        reports.append(
            SourceReport(
                source=source,
                track=TRACK[source],
                target=targets.get(source, 0),
                built=len(built),
                kept=len(kept),
                train=len(source_train),
                val=len(source_val),
            )
        )

    index.assert_clean()
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    return CorpusReport(
        sources=reports,
        train=len(train),
        val=len(val),
        dropped_exact=sum(
            1 for d in index.dropped if d.verdict is DedupVerdict.EXACT_DUPLICATE
        ),
        dropped_near=sum(
            1 for d in index.dropped if d.verdict is DedupVerdict.NEAR_DUPLICATE
        ),
        seed=seed,
    )


__all__ = [
    "AUGMENTATION_RATES",
    "BEN19_CLASSES",
    "BEN_TXT_TASKS",
    "BEN6_MERGE",
    "COMPOSITION",
    "NEAR_DUP_DISTANCE",
    "PHASH_BITS",
    "QUARANTINED_SPLITS",
    "RSVQA_TYPES",
    "TRACK",
    "Augmentation",
    "CitationError",
    "CorpusError",
    "CorpusReport",
    "CorpusSample",
    "CorpusSource",
    "DedupResult",
    "DedupVerdict",
    "Deduplicator",
    "LeakageError",
    "Message",
    "Reservoir",
    "VAL_SHARE",
    "SampleMeta",
    "SourceReport",
    "SourceView",
    "ViewRecord",
    "apply_augmentations",
    "assemble",
    "audit_answer",
    "UnlabelledGroundingError",
    "ben6_labels",
    "grounding_label",
    "noncanonical_grounding",
    "build_corpus",
    "build_corpus_streaming",
    "build_evidence_qa",
    "cite",
    "deduplicate",
    "image_key_of",
    "describe_backscatter",
    "describe_change",
    "describe_ndbi",
    "describe_ndvi",
    "fact_sheet_from",
    "from_bigearthnet",
    "from_bigearthnet_txt",
    "from_cdvqa",
    "from_dior_rsvg",
    "from_rsvqa",
    "from_vrsbench",
    "hamming",
    "label_views",
    "parse_ben_txt_box",
    "phash",
    "plan_augmentations",
    "read_jsonl",
    "sha256_of",
    "allocate_split",
    "take",
    "unit_suffix",
    "write_jsonl",
]
