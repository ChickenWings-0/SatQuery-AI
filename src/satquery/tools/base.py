"""The tool protocol: one interface, every capability behind it (Master.md §5).

A tool is a pure function of its declared inputs and its parameters. It never
decides *whether* it should run — capability matching does that — and it never
decides what runs next — the policy table does that. What it owns is one
measurement and the evidence for it.

Two conventions make the executor's job possible:

* **Artifacts are drafted, not stored.** A tool returns :class:`ArtifactDraft`
  objects with no id. The executor assigns ``art_{n}`` in ascending step order
  after each wave, so ids are identical across reruns even though the wave ran
  concurrently.
* **Arrays travel out of band.** ``ToolResult.data`` carries numpy payloads keyed
  by draft key; the executor re-keys them by artifact id so a downstream step
  reading ``@2:CHANGE_MASK`` gets the actual mask, not a URL.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from satquery.evidence.fact_sheet import FactSheet
from satquery.render.artifact_store import ArtifactStore
from satquery.render.tiling import VIEW_SIZE_PX, ViewGeometry, render_source
from satquery.render.views import ImageFormat
from satquery.schemas.enums import ArtifactType, Device, ImageRole, Modality, PairType, ToolStatus
from satquery.schemas.manifest import InputManifest
from satquery.schemas.trace import ArtifactRef

Array = npt.NDArray[np.float32]
Mask = npt.NDArray[np.bool_]
Rgb = npt.NDArray[np.uint8]


class ToolError(RuntimeError):
    """A tool could not produce its measurement.

    Raising this is how a tool asks for its fallback. It is never fatal on its
    own: the executor substitutes ``spec.fallback`` and records ``DEGRADED``.
    """


class MissingInputError(ToolError):
    """A required input or band was not present on the resolved inputs."""


@dataclass(frozen=True)
class ImageBundle:
    """One input image with everything a tool needs to read it.

    This is the ingestion layer's output joined to the band alias table, so no
    tool has to know how a sensor names its bands.
    """

    manifest: InputManifest
    path: Path
    resolved_bands: dict[str, int]

    @property
    def id(self) -> str:
        """The ``img_*`` reference this bundle answers to."""
        return self.manifest.id

    @property
    def modality(self) -> Modality:
        """Sensing modality, as classified during ingestion."""
        return self.manifest.modality

    @property
    def role(self) -> ImageRole | None:
        """Role within the pair, or None for an unroled single image."""
        return self.manifest.role

    @property
    def sensor(self) -> str | None:
        """The guessed sensor, used to pick the band alias map."""
        return self.manifest.sensor_guess

    @property
    def acquisition_time(self) -> datetime | None:
        """Acquisition timestamp when the metadata carried one."""
        return self.manifest.acquisition_time

    def has(self, *bands: str) -> bool:
        """True when every named logical band resolves on this image."""
        return all(band in self.resolved_bands for band in bands)

    def missing(self, bands: Sequence[str]) -> list[str]:
        """The named logical bands this image cannot provide."""
        return [band for band in bands if band not in self.resolved_bands]


@dataclass(frozen=True)
class Bands:
    """A band stack read onto the shared square view canvas.

    Every tool reads through this, so two tools measuring the same image are
    measuring the same pixels — which is what makes cross-tool agreement (§7.2)
    mean anything.
    """

    order: tuple[str, ...]
    stack: Array
    valid: Mask
    geometry: ViewGeometry

    def band(self, name: str) -> Array:
        """Return one logical band from the stack."""
        if name not in self.order:
            raise MissingInputError(f"band {name!r} was not read into this stack")
        selected: Array = self.stack[self.order.index(name)]
        return selected

    @property
    def pixel_area_m2(self) -> float | None:
        """Ground area of one canvas pixel, or None when not georeferenced.

        The canvas transform is the *rendered* grid's, not the source raster's,
        so this is the area the masks a tool produces are actually measured over.
        """
        transform = self.geometry.transform
        if transform is None or not self.geometry.crs:
            return None
        return abs(float(transform.a) * float(transform.e))


class PixelReader:
    """Memoised band reads, shared across every step of one analysis.

    A five-step plan over a bi-temporal pair would otherwise open and downsample
    the same rasters a dozen times. Keying the cache on (path, bands, size) also
    guarantees two steps see byte-identical arrays.
    """

    def __init__(self, size: int = VIEW_SIZE_PX) -> None:
        """Create a reader rendering onto a *size* x *size* canvas."""
        self.size = size
        self._cache: dict[tuple[str, tuple[str, ...], int], Bands] = {}

    def read(self, image: ImageBundle, bands: Sequence[str], size: int | None = None) -> Bands:
        """Read the named logical bands of *image* onto the view canvas.

        Raises:
            MissingInputError: The image does not carry one of the bands.
        """
        missing = image.missing(bands)
        if missing:
            raise MissingInputError(
                f"{image.id} does not provide band(s) {', '.join(missing)}"
            )
        order = tuple(dict.fromkeys(bands))
        edge = size or self.size
        key = (str(image.path), order, edge)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        indexes = [image.resolved_bands[name] for name in order]
        stack, valid, geometry = render_source(image.path, indexes, size=edge)
        result = Bands(order=order, stack=stack, valid=valid, geometry=geometry)
        self._cache[key] = result
        return result


@dataclass(frozen=True)
class MaskPayload:
    """A binary mask plus the geometry needed to turn pixels into ground area.

    Masks travel between steps in memory: ``change_statistics`` needs the actual
    pixels, and a PNG URL is not pixels.
    """

    mask: Mask
    pixel_area_m2: float | None
    geometry: ViewGeometry

    @property
    def total_px(self) -> int:
        """Number of pixels on the canvas the mask covers."""
        return int(self.mask.size)


@dataclass(frozen=True)
class ClassMap:
    """A labelled land-cover raster, on the same canvas as the masks beside it.

    The payload a ``SEGMENTATION`` artifact carries in memory. It is declared here
    in Phase 5 rather than in Phase 6 with the segmenter that produces it, because
    ``change_statistics`` already reads it: per-class change is a *join* between a
    change mask and a label map, and the join has to agree about the contract from
    both sides or it silently produces nothing.

    ``labels`` is 0-based and indexes :attr:`classes`; ``-1`` marks unlabelled
    pixels, which are excluded from every per-class figure rather than being
    attributed to class 0.
    """

    labels: npt.NDArray[np.int16]
    classes: tuple[str, ...]
    geometry: ViewGeometry
    pixel_area_m2: float | None = None

    def mask_for(self, index: int) -> Mask:
        """Boolean selection of one class."""
        selected: Mask = self.labels == index
        return selected

    def named(self) -> list[tuple[str, int]]:
        """``(class name, index)`` pairs, in declaration order."""
        return [(name, index) for index, name in enumerate(self.classes)]


@dataclass(frozen=True)
class ArtifactDraft:
    """One artifact a tool produced, before the executor gives it an id.

    Exactly one payload field is normally set. ``inline`` alone means a SCALARS
    or TEXT artifact with no blob, which the contract serves inside the trace
    rather than over ``/v1/artifacts``.
    """

    key: str
    type: ArtifactType
    label: str
    image: Rgb | None = None
    image_format: ImageFormat = ImageFormat.PNG
    raster: npt.NDArray[Any] | None = None
    geojson: Any | None = None
    inline: dict[str, Any] | None = None
    stats: dict[str, Any] | None = None
    geometry: ViewGeometry | None = None


@dataclass(frozen=True)
class RenderDraft:
    """The renderer's artifacts, expanded at materialisation time.

    ``render_views`` numbers its own artifacts, so it has to run once the
    executor knows the starting index. The tool therefore returns the request
    and the executor performs the render.
    """

    key: str
    sources: tuple[Any, ...]
    pair_type: PairType
    size: int


Draft = ArtifactDraft | RenderDraft


@dataclass
class ToolResult:
    """What one tool invocation produced."""

    status: ToolStatus = ToolStatus.OK
    scalars: dict[str, float | str] = field(default_factory=dict)
    artifacts: list[Draft] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    device: Device = Device.CPU
    notes: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class ToolContext:
    """Everything one step is allowed to see.

    Deliberately narrow: a tool receives its *resolved* inputs and the evidence
    already measured, never the plan and never the registry. Control flow is not
    a tool's business (Master.md §3.1).

    Two fields exist for the ``vlm_*`` synthesisers and nothing else:

    * :attr:`question` is the analyst's raw query. A VQA tool structurally cannot
      answer without it. It is passed as *data to be answered*, not as something
      a tool may act on: no tool reads it to decide what to run, because by the
      time a tool exists the plan is already fixed.
    * :attr:`facts` is the FactSheet as it stands *at this step* — every scalar
      the step's upstream dependencies measured, namespaced and schema-checked.
      It is what constrains the VLM to the numbers the system actually measured.
      A deterministic tool has no business reading it: measuring is its job, and a
      measurement that depended on another tool's answer would not be one.
    """

    trace_id: str
    step: int
    pair_type: PairType
    images: list[ImageBundle]
    artifacts: list[ArtifactRef] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    pixels: PixelReader = field(default_factory=PixelReader)
    slots: dict[str, Any] = field(default_factory=dict)
    question: str = ""
    facts: FactSheet = field(default_factory=FactSheet)
    seed: int = 0
    store: ArtifactStore | None = None

    def by_role(self, *roles: ImageRole) -> list[ImageBundle]:
        """Return the input images holding any of *roles*, in input order."""
        wanted = set(roles)
        return [image for image in self.images if image.role in wanted]

    def one(self) -> ImageBundle:
        """Return the sole input image.

        Raises:
            MissingInputError: The step did not resolve to exactly one image.
        """
        if len(self.images) != 1:
            raise MissingInputError(f"expected exactly one image, got {len(self.images)}")
        return self.images[0]

    def pair(self) -> tuple[ImageBundle, ImageBundle]:
        """Return the two input images in input order.

        Raises:
            MissingInputError: The step did not resolve to exactly two images.
        """
        if len(self.images) != 2:
            raise MissingInputError(f"expected exactly two images, got {len(self.images)}")
        return self.images[0], self.images[1]

    def payload(self, artifact_type: ArtifactType) -> Any:
        """Return the in-memory payload of the first input artifact of a type.

        Raises:
            MissingInputError: No input artifact of that type carried a payload.
        """
        for artifact in self.artifacts:
            if artifact.type is artifact_type and artifact.id in self.data:
                return self.data[artifact.id]
        raise MissingInputError(f"no {artifact_type.value} payload among the step's inputs")


@runtime_checkable
class Tool(Protocol):
    """The one interface every capability implements.

    Implementations must be deterministic: identical inputs and identical params
    must produce identical scalars, because the trace is replayed field by field
    in the reproducibility test (AGENT_POLICY_DAG §8).
    """

    name: str

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Perform this tool's measurement over the resolved inputs."""
        ...


def suffix_for(images: Sequence[ImageBundle], index: int, suffixes: Sequence[str]) -> str:
    """Pick the scalar suffix for the *index*-th image of a multi-image step.

    Suffixes come from the policy table, which is responsible for supplying
    distinct ones when a tool runs over several images — the FactSheet namespaces
    by tool, not by step, so ``ndvi_mean_pre`` and ``ndvi_mean_post`` are what
    keeps the two invocations apart (AGENT_POLICY_DAG §6.4).
    """
    if len(images) <= 1:
        return ""
    if index < len(suffixes):
        return str(suffixes[index])
    role = images[index].role
    return f"_{role.value}" if role else f"_{index}"
