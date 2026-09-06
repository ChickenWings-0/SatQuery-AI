"""Content-addressed artifact store (API_CONTRACT §4.6).

Artifacts are immutable: an id names a specific set of bytes, so the store keys
blobs by the SHA-256 of their encoded content and keeps a small per-trace index
mapping ``art_0.png`` to a digest. Two traces that render identical pixels share
one blob, and the endpoint can promise ``Cache-Control: immutable`` honestly.

Encoding rules are frozen by DATA_ADAPTATION_PLAN §2.1: measurement views
(indices, SAR backscatter, change overlays) are PNG, because JPEG chroma
subsampling on a colormapped field visibly shifts the apparent value; natural
composites are JPEG q92.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import rasterio
from affine import Affine
from PIL import Image
from rasterio.crs import CRS

from satquery.render.views import ImageFormat

JPEG_QUALITY: Final[int] = 92
"""Frozen by DATA_ADAPTATION_PLAN §2.1."""

JPEG_SUBSAMPLING: Final[int] = 0
"""4:4:4. These composites are evidence, so chroma is kept at full resolution
even though the format tolerates less."""

MIME_TYPES: Final[dict[str, str]] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "geojson": "application/geo+json",
    "json": "application/json",
}
"""Extensions the artifact endpoint serves. ``jpg`` extends the API_CONTRACT §4.6
list additively, because §2.1 of the rendering plan mandates JPEG composites."""

_INDEX_NAME: Final[str] = "index.json"
_EVICTED_KEY: Final[str] = "evicted"


class ArtifactNotFoundError(LookupError):
    """No such artifact in this trace."""


class TraceEvictedError(LookupError):
    """The trace existed but its artifacts have been evicted (HTTP 410)."""


@dataclass(frozen=True)
class StoredBlob:
    """A stored artifact: where its bytes are and how to serve them."""

    trace_id: str
    artifact_id: str
    extension: str
    digest: str
    path: Path
    size_bytes: int

    @property
    def mime(self) -> str:
        """The Content-Type this blob is served as."""
        return MIME_TYPES[self.extension]

    @property
    def etag(self) -> str:
        """A strong ETag — the content digest, which *is* the identity."""
        return f'"{self.digest}"'

    @property
    def url(self) -> str:
        """The contract-shaped URL this blob is reachable at."""
        return f"/v1/artifacts/{self.trace_id}/{self.artifact_id}.{self.extension}"


def encode_image(rgb: npt.NDArray[np.uint8], image_format: ImageFormat) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 image to PNG or JPEG q92."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got {rgb.shape}")
    buffer = io.BytesIO()
    image = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), mode="RGB")
    if image_format is ImageFormat.PNG:
        image.save(buffer, format="PNG", optimize=True)
    else:
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
    return buffer.getvalue()


def encode_geotiff(
    array: npt.NDArray[Any],
    transform: Affine | None,
    crs: str | None,
    nodata: float | None = None,
) -> bytes:
    """Encode a georeferenced companion raster.

    Accepts ``(H, W)`` or ``(bands, H, W)``. The transform and CRS must describe
    the *rendered* grid, so the file reopens exactly where the view sits on the
    ground.
    """
    data = array if array.ndim == 3 else array[np.newaxis, ...]
    count, height, width = data.shape
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": data.dtype.name,
        "compress": "deflate",
    }
    if crs and transform is not None:
        profile["crs"] = CRS.from_string(crs)
        profile["transform"] = transform
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.MemoryFile() as memory:
        with memory.open(**profile) as dataset:
            dataset.write(data)
        return bytes(memory.read())


def encode_json(payload: Any) -> bytes:
    """Encode a JSON or GeoJSON payload deterministically.

    Sorted keys and a fixed separator keep the digest stable, so re-rendering the
    same result reuses the same blob instead of growing the store.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


class ArtifactStore:
    """A filesystem-backed, content-addressed store for trace artifacts."""

    def __init__(self, root: str | Path) -> None:
        """Create (or attach to) a store rooted at *root*."""
        self.root = Path(root)
        self.blobs = self.root / "blobs"
        self.traces = self.root / "traces"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.traces.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ writing

    def put(self, trace_id: str, artifact_id: str, extension: str, payload: bytes) -> StoredBlob:
        """Store *payload* and register it under ``{artifact_id}.{extension}``.

        Raises:
            ValueError: The extension is not one the endpoint can serve.
        """
        if extension not in MIME_TYPES:
            raise ValueError(f"unsupported artifact extension {extension!r}")

        digest = hashlib.sha256(payload).hexdigest()
        path = self._blob_path(digest, extension)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write via a temporary name so a reader never sees a partial blob.
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(path)

        blob = StoredBlob(
            trace_id=trace_id,
            artifact_id=artifact_id,
            extension=extension,
            digest=digest,
            path=path,
            size_bytes=len(payload),
        )
        self._register(blob)
        return blob

    def put_image(
        self,
        trace_id: str,
        artifact_id: str,
        rgb: npt.NDArray[np.uint8],
        image_format: ImageFormat,
    ) -> StoredBlob:
        """Encode and store a rendered view."""
        return self.put(
            trace_id, artifact_id, image_format.extension, encode_image(rgb, image_format)
        )

    def put_geotiff(
        self,
        trace_id: str,
        artifact_id: str,
        array: npt.NDArray[Any],
        transform: Affine | None,
        crs: str | None,
        nodata: float | None = None,
    ) -> StoredBlob:
        """Encode and store a georeferenced companion raster."""
        return self.put(trace_id, artifact_id, "tif", encode_geotiff(array, transform, crs, nodata))

    def put_geojson(self, trace_id: str, artifact_id: str, payload: Any) -> StoredBlob:
        """Store vector features in EPSG:4326."""
        return self.put(trace_id, artifact_id, "geojson", encode_json(payload))

    def put_json(self, trace_id: str, artifact_id: str, payload: Any) -> StoredBlob:
        """Store an inline scalar or text payload."""
        return self.put(trace_id, artifact_id, "json", encode_json(payload))

    # ------------------------------------------------------------------ reading

    def resolve(self, trace_id: str, artifact_id: str, extension: str) -> StoredBlob:
        """Look up one artifact.

        Raises:
            TraceEvictedError: The trace is known but its artifacts are gone.
            ArtifactNotFoundError: No such trace or artifact.
        """
        index = self._read_index(trace_id)
        if index is None:
            raise ArtifactNotFoundError(f"unknown trace {trace_id}")
        if index.get(_EVICTED_KEY):
            raise TraceEvictedError(f"trace {trace_id} has been evicted")

        key = f"{artifact_id}.{extension}"
        entry = index.get("artifacts", {}).get(key)
        if entry is None:
            raise ArtifactNotFoundError(f"unknown artifact {key}")

        path = self._blob_path(entry["digest"], extension)
        if not path.exists():
            raise ArtifactNotFoundError(f"blob for {key} is missing from the store")
        return StoredBlob(
            trace_id=trace_id,
            artifact_id=artifact_id,
            extension=extension,
            digest=entry["digest"],
            path=path,
            size_bytes=int(entry["size_bytes"]),
        )

    def list_artifacts(self, trace_id: str) -> list[str]:
        """Every ``{artifact_id}.{ext}`` registered for a trace."""
        index = self._read_index(trace_id)
        if index is None or index.get(_EVICTED_KEY):
            return []
        return sorted(index.get("artifacts", {}))

    def evict(self, trace_id: str) -> None:
        """Mark a trace's artifacts as gone, so requests answer 410 rather than 404.

        Blobs are left alone: they are content-addressed and may be shared with
        another trace.
        """
        index = self._read_index(trace_id) or {"artifacts": {}}
        index[_EVICTED_KEY] = True
        index["evicted_at"] = datetime.now(UTC).isoformat()
        index["artifacts"] = {}
        self._write_index(trace_id, index)

    # ------------------------------------------------------------------ internals

    def _blob_path(self, digest: str, extension: str) -> Path:
        """Shard blobs two hex characters deep to keep directories small."""
        return self.blobs / digest[:2] / f"{digest}.{extension}"

    def _index_path(self, trace_id: str) -> Path:
        return self.traces / trace_id / _INDEX_NAME

    def _read_index(self, trace_id: str) -> dict[str, Any] | None:
        path = self._index_path(trace_id)
        if not path.exists():
            return None
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def _write_index(self, trace_id: str, index: dict[str, Any]) -> None:
        path = self._index_path(trace_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def _register(self, blob: StoredBlob) -> None:
        index = self._read_index(blob.trace_id) or {
            "created_at": datetime.now(UTC).isoformat(),
            "artifacts": {},
        }
        index.pop(_EVICTED_KEY, None)
        index.setdefault("artifacts", {})[f"{blob.artifact_id}.{blob.extension}"] = {
            "digest": blob.digest,
            "size_bytes": blob.size_bytes,
            "mime": blob.mime,
        }
        self._write_index(blob.trace_id, index)
