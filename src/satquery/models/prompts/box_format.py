"""The one box serialiser, shared by corpus construction and inference.

DATA_ADAPTATION_PLAN.md §4.5 states the constraint this module exists to
enforce: *"the assistant's answer must use Qwen3-VL's native normalised-box
format, byte-identical to what ``tools/text_grounding.py`` parses at inference.
Both sides import one serialiser."* A training/inference format drift produces a
model that grounds perfectly and a parser that returns nothing, and because the
model is fine-tuned on whatever :func:`serialise` emits, that failure is silent
and total.

**The canonical form** is Qwen's normalised box syntax: coordinates scaled to
:data:`BOX_SCALE` (0-1000) of the image's width and height, wrapped in the box
sentinels, optionally preceded by an object reference::

    <|object_ref_start|>aircraft<|object_ref_end|><|box_start|>(112,340),(288,512)<|box_end|>

Normalised rather than absolute because the model sees a 448 px canvas while the
analyst's raster may be 10,000 px across: a box in canvas pixels means nothing
once it leaves the canvas, whereas a fraction of the frame survives every resize
in the chain.

**Parsing is deliberately more permissive than emission.** :func:`parse` also
accepts the JSON ``{"bbox_2d": [...]}`` form that Qwen2.5-VL-derived checkpoints
emit, and absolute-pixel coordinates when told the frame size, because at serving
time the alternative to a tolerant parser is an empty answer. Emission has no
such latitude — there is exactly one training format, and it is the one above.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final

BOX_SCALE: Final[int] = 1000
"""Qwen normalises box coordinates to 0-1000 of each axis, not to 0-1."""

REF_START: Final[str] = "<|object_ref_start|>"
REF_END: Final[str] = "<|object_ref_end|>"
BOX_START: Final[str] = "<|box_start|>"
BOX_END: Final[str] = "<|box_end|>"

_BOX_BODY: Final[str] = r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*,\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"

_TAGGED = re.compile(
    rf"(?:{re.escape(REF_START)}(?P<label>.*?){re.escape(REF_END)}\s*)?"
    rf"{re.escape(BOX_START)}\s*(?P<body>{_BOX_BODY})\s*{re.escape(BOX_END)}",
    re.DOTALL,
)
_BODY = re.compile(_BOX_BODY)
_JSON_BLOCK = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)

NONE_ANSWER: Final[str] = "NONE"
"""The one non-box grounding answer the system prompt allows."""

_CANONICAL_BOX: Final[str] = (
    rf"{re.escape(REF_START)}[^<>]+{re.escape(REF_END)}"
    rf"{re.escape(BOX_START)}\(\d{{1,4}},\d{{1,4}}\),\(\d{{1,4}},\d{{1,4}}\){re.escape(BOX_END)}"
)
CANONICAL_ANSWER: Final[re.Pattern[str]] = re.compile(
    rf"(?:{_CANONICAL_BOX})+|{NONE_ANSWER}"
)
"""What a GROUNDING assistant turn must match in full: one or more *tagged*
boxes — name and coordinates, no separators — or exactly ``NONE``. This is the
format the system prompt's TASK block dictates (``templates.GROUNDED_V1``), and
the corpus builder refuses any grounding answer that does not match it. Half
of the previous corpus's grounding targets were bare ``<|box_start|>…`` with no
object reference, contradicting the instruction they were trained under
(ML_PIPELINE_RECOVERY_PLAN fact 13)."""

_SENTENCE_BREAK: Final[re.Pattern[str]] = re.compile(r"[\n\r.;!?]")
"""A label does not span a sentence. Only the text after the last break is
considered, so prose before a box cannot be swallowed whole."""

_LABEL_STRIP: Final[str] = " \t\r\n,:;-\u2013\u2014\"'`([{"
"""Separators that sit between a label and its box, or bracket the label."""

MAX_LABEL_CHARS: Final[int] = 64
"""A referring expression longer than this is prose that happens to precede a
box, not a label. The tail is kept rather than the head: the words nearest the
box are the ones describing it."""


class BoxFormatError(ValueError):
    """A box could not be represented in the canonical format."""


@dataclass(frozen=True)
class NormalisedBox:
    """One box in Qwen's normalised 0-1000 frame, with an optional label.

    Coordinates are ``x_min, y_min, x_max, y_max`` with a top-left origin — the
    same ordering the contract uses for pixel boxes (API_CONTRACT §1), so the
    only thing :func:`to_pixels` changes is the scale.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: str | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        """Reject a box that is inverted or outside the normalised frame.

        Raises:
            BoxFormatError: The box has non-positive extent, or a coordinate
                falls outside ``[0, BOX_SCALE]``.
        """
        for name, value in (
            ("x_min", self.x_min),
            ("y_min", self.y_min),
            ("x_max", self.x_max),
            ("y_max", self.y_max),
        ):
            if not 0 <= value <= BOX_SCALE:
                raise BoxFormatError(
                    f"{name}={value} is outside the normalised frame [0, {BOX_SCALE}]"
                )
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise BoxFormatError(
                f"box ({self.x_min},{self.y_min}),({self.x_max},{self.y_max}) has no extent"
            )

    @property
    def area(self) -> int:
        """Area in normalised units, used to rank and to drop degenerate boxes."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Scale onto a *width* x *height* frame as an integer pixel box.

        The contract's pixel boxes are half-open (``x_max`` exclusive), so the
        maxima round up and the minima round down: a box that rounded inward on
        both edges would shrink by up to a pixel per side on every conversion.
        """
        x_min = int(self.x_min * width // BOX_SCALE)
        y_min = int(self.y_min * height // BOX_SCALE)
        x_max = -int(-self.x_max * width // BOX_SCALE)
        y_max = -int(-self.y_max * height // BOX_SCALE)
        return (
            max(0, min(x_min, width - 1)),
            max(0, min(y_min, height - 1)),
            max(x_min + 1, min(x_max, width)),
            max(y_min + 1, min(y_max, height)),
        )


def from_pixels(
    box: Sequence[float],
    width: int,
    height: int,
    label: str | None = None,
    score: float | None = None,
) -> NormalisedBox:
    """Normalise a pixel box onto the 0-1000 frame.

    Args:
        box: ``(x_min, y_min, x_max, y_max)`` in pixels, top-left origin.
        width: Frame width the box was measured on.
        height: Frame height the box was measured on.
        label: Optional class or referring expression.
        score: Optional confidence.

    Raises:
        BoxFormatError: The frame has no extent, or the box is degenerate.
    """
    if width <= 0 or height <= 0:
        raise BoxFormatError(f"cannot normalise against a {width}x{height} frame")
    x_min, y_min, x_max, y_max = (float(v) for v in box)
    return NormalisedBox(
        x_min=_clamp(round(x_min * BOX_SCALE / width)),
        y_min=_clamp(round(y_min * BOX_SCALE / height)),
        x_max=_clamp(round(x_max * BOX_SCALE / width)),
        y_max=_clamp(round(y_max * BOX_SCALE / height)),
        label=label,
        score=score,
    )


def _clamp(value: int) -> int:
    """Hold a normalised coordinate inside the frame."""
    return max(0, min(BOX_SCALE, int(value)))


def serialise(boxes: Iterable[NormalisedBox]) -> str:
    """Render boxes in the canonical training format.

    This is the *only* emission path. The corpus builder writes assistant turns
    with it and the grounding tool parses them back with :func:`parse`, which is
    what keeps the two byte-identical.
    """
    return "".join(
        (f"{REF_START}{box.label}{REF_END}" if box.label else "")
        + f"{BOX_START}({box.x_min},{box.y_min}),({box.x_max},{box.y_max}){BOX_END}"
        for box in boxes
    )


def serialise_answer(boxes: Iterable[NormalisedBox]) -> str:
    """Render a *grounding answer*: every box tagged with its name, or ``NONE``.

    The strict form of :func:`serialise` for the corpus. A box without a label
    cannot be written in the format the system prompt dictates, so it is an
    error here rather than a bare box in the training data.

    Raises:
        BoxFormatError: A box carries no label.
    """
    listed = list(boxes)
    if not listed:
        return NONE_ANSWER
    for box in listed:
        if not box.label or not box.label.strip():
            raise BoxFormatError(
                f"box ({box.x_min},{box.y_min}),({box.x_max},{box.y_max}) has no label; a "
                "grounding answer names every box or the prompt's format is violated"
            )
    text = serialise(listed)
    if not is_canonical_answer(text):  # pragma: no cover - serialise() guarantees it
        raise BoxFormatError(f"serialised answer is not canonical: {text!r}")
    return text


def is_canonical_answer(text: str) -> bool:
    """True when *text* is exactly what the GROUNDING instruction asks for."""
    return CANONICAL_ANSWER.fullmatch(text) is not None


def strip_boxes(text: str) -> str:
    """Remove every box form this module understands, leaving the prose around it.

    For the citation validator. A box coordinate is not a claim about the scene:
    it is a position in the frame, and there is no FactSheet key it could ever
    resolve against. Validating one as a measurement marks it uncited, which
    degrades the step and caps the confidence — so a grounding answer was
    penalised for containing exactly what :data:`templates.GROUNDED_V1` orders it
    to contain, and every grounding run degraded no matter how good the boxes
    were. Prose numbers in the same answer still have to be cited, which is why
    this removes the boxes rather than skipping validation for the task.
    """
    without = _TAGGED.sub(" ", text)
    without = _JSON_BLOCK.sub(" ", without)
    return _BODY.sub(" ", without)


def parse(text: str, width: int | None = None, height: int | None = None) -> list[NormalisedBox]:
    """Extract every box from a model answer, in the order it emitted them.

    Args:
        text: The raw generation.
        width: Frame width, needed only to interpret absolute-pixel JSON boxes.
        height: Frame height, same.

    Returns:
        The boxes found, deduplicated and with degenerate ones dropped. A
        generation carrying no boxes yields an empty list rather than raising:
        "I could not find one" is a legitimate grounding answer, and the tool
        reports it as ``n_boxes: 0`` rather than as a failure.
    """
    found = list(_parse_tagged(text))
    if not found:
        found = list(_parse_json(text, width, height))
    if not found:
        found = list(_parse_bare(text))
    seen: set[tuple[int, int, int, int, str | None]] = set()
    unique: list[NormalisedBox] = []
    for box in found:
        key = (box.x_min, box.y_min, box.x_max, box.y_max, box.label)
        if key not in seen:
            seen.add(key)
            unique.append(box)
    return unique


def _parse_tagged(text: str) -> Iterator[NormalisedBox]:
    """Boxes in the canonical sentinel form."""
    for match in _TAGGED.finditer(text):
        body = _BODY.search(match.group("body"))
        if body is None:  # pragma: no cover - the outer pattern guarantees it
            continue
        label = (match.group("label") or "").strip() or None
        box = _build(tuple(int(v) for v in body.groups()), label, None)
        if box is not None:
            yield box


def _parse_bare(text: str) -> Iterator[NormalisedBox]:
    """Boxes written as bare ``(x1,y1),(x2,y2)``, label in front, no sentinels.

    Tried last, and only when nothing better matched, because it is the most
    permissive reading available: any pair of coordinate pairs in the text
    becomes a box. That is the right trade for a grounding answer, where the
    alternative is an empty box list, and the wrong one for anything else — which
    is why the canonical and JSON forms get first refusal.

    It exists because a fine-tuned checkpoint does not necessarily keep the
    sentinels. Ours emits ``buildings(170,527),(238,577)``: the coordinates are
    correct and in the right frame, and only the ``<|box_start|>`` wrapper is
    missing. Parsing returned nothing, so no box reached the frontend and the run
    was marked ungrounded for stating numbers it could not cite — a total,
    silent failure caused entirely by a missing pair of tags.

    Labels carry forward, because the observed form puts one label in front of a
    run of boxes: ``buildings(100,738),(330,998)(370,768),(610,1000)`` is two
    buildings, not one building and one anonymous box.
    """
    cursor = 0
    label: str | None = None
    for match in _BODY.finditer(text):
        found = _label_before(text[cursor:match.start()])
        if found is not None:
            label = found
        cursor = match.end()
        box = _build(tuple(int(value) for value in match.groups()), label, None)
        if box is not None:
            yield box


def _label_before(gap: str) -> str | None:
    """Read the label out of the text between the previous box and this one.

    Returns None when the gap holds no label — an empty gap, or nothing but
    separators — which is what makes a label carry forward across a run of
    boxes rather than resetting to unlabelled.
    """
    tail = _SENTENCE_BREAK.split(gap)[-1].strip(_LABEL_STRIP)
    if len(tail) > MAX_LABEL_CHARS:
        # Keep the words nearest the box, and drop the leading fragment the cut
        # is likely to have sliced through.
        _, _, tail = tail[-MAX_LABEL_CHARS:].partition(" ")
        tail = tail.strip(_LABEL_STRIP)
    return tail or None


def _parse_json(
    text: str, width: int | None, height: int | None
) -> Iterator[NormalisedBox]:
    """Boxes in the ``{"bbox_2d": [...]}`` form some Qwen checkpoints emit.

    Tolerated, never emitted. Coordinates here may be absolute pixels, which is
    only detectable by them exceeding the normalised frame — hence the frame
    size being needed to interpret them at all.
    """
    for block in _JSON_BLOCK.finditer(text):
        try:
            entries = json.loads(block.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(entries, list):  # pragma: no cover - regex shape
            continue
        for entry in entries:
            box = _from_json_entry(entry, width, height)
            if box is not None:
                yield box


def _from_json_entry(
    entry: Any, width: int | None, height: int | None
) -> NormalisedBox | None:
    """One JSON detection object, or None when it is not one."""
    if not isinstance(entry, dict):
        return None
    raw = entry.get("bbox_2d") or entry.get("bbox") or entry.get("box")
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        return None
    try:
        values = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None

    # Absolute pixels are the only reading under which a coordinate can exceed
    # the normalised frame, and rescaling needs the frame it was measured on.
    if max(values) > BOX_SCALE:
        if not width or not height:
            return None
        try:
            return from_pixels(values, width, height, _label_of(entry), _score_of(entry))
        except BoxFormatError:
            return None
    return _build(
        tuple(int(round(v)) for v in values), _label_of(entry), _score_of(entry)
    )


def _label_of(entry: dict[str, Any]) -> str | None:
    """The label field under any of the names Qwen has used for it."""
    for key in ("label", "text", "ref", "object"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _score_of(entry: dict[str, Any]) -> float | None:
    """The confidence field, when the generation carried one."""
    for key in ("score", "confidence"):
        value = entry.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _build(
    values: tuple[int, ...], label: str | None, score: float | None
) -> NormalisedBox | None:
    """Construct a box, returning None instead of raising on a degenerate one.

    A model that emits an inverted or zero-area box has made a mistake in one
    box, not in its whole answer. Dropping it keeps the rest of the detections.
    """
    x_min, y_min, x_max, y_max = values
    try:
        return NormalisedBox(
            x_min=_clamp(min(x_min, x_max)),
            y_min=_clamp(min(y_min, y_max)),
            x_max=_clamp(max(x_min, x_max)),
            y_max=_clamp(max(y_min, y_max)),
            label=label,
            score=score,
        )
    except BoxFormatError:
        return None


parse_boxes = parse
strip_box_syntax = strip_boxes
serialise_boxes = serialise
"""Aliases for re-export from the package namespace, where a bare ``parse`` next
to ``build_prompt`` would not say what it parses."""


__all__ = [
    "BOX_END",
    "BOX_SCALE",
    "BOX_START",
    "CANONICAL_ANSWER",
    "NONE_ANSWER",
    "REF_END",
    "REF_START",
    "BoxFormatError",
    "NormalisedBox",
    "from_pixels",
    "is_canonical_answer",
    "parse",
    "parse_boxes",
    "serialise",
    "serialise_answer",
    "serialise_boxes",
]
