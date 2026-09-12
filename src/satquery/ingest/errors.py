"""Ingestion failures mapped onto the frozen error taxonomy (API_CONTRACT §6)."""

from __future__ import annotations

from satquery.schemas.api import ApiError


class IngestError(Exception):
    """A user-fixable ingestion failure that carries its own wire representation.

    ``message`` and ``hint`` are user-facing and must never contain a filesystem
    path or a stack trace.
    """

    def __init__(  # noqa: D107 - documented by the class docstring above
        self,
        code: str,
        http_status: int,
        message: str,
        hint: str | None = None,
        ref: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.message = message
        self.hint = hint
        self.ref = ref

    def to_api_error(self, trace_id: str | None = None) -> ApiError:
        """Render this failure as the frozen :class:`ApiError` payload."""
        return ApiError(
            code=self.code,
            http_status=self.http_status,
            message=self.message,
            hint=self.hint,
            ref=self.ref,
            trace_id=trace_id,
        )


class InvalidRasterError(IngestError):
    """rasterio could not open the file, or it opened as something unusable."""

    def __init__(self, ref: str | None = None, detail: str | None = None) -> None:
        """Build the error for image *ref*, with an optional user-safe *detail*."""
        super().__init__(
            code="INVALID_RASTER",
            http_status=400,
            message=detail or "The file could not be opened as a raster image.",
            hint="Upload a GeoTIFF, TIFF, PNG or JPEG produced by a standard tool.",
            ref=ref,
        )


class UnsupportedFormatError(IngestError):
    """The driver rasterio chose is outside the allow-list."""

    def __init__(self, driver: str, ref: str | None = None) -> None:
        """Build the error naming the rejected *driver*."""
        super().__init__(
            code="UNSUPPORTED_FORMAT",
            http_status=415,
            message=f"Files of type '{driver}' are not supported.",
            hint="Upload a GeoTIFF (preferred), TIFF, PNG or JPEG.",
            ref=ref,
        )


class TooManyImagesError(IngestError):
    """More images than the pipeline can relate to one another."""

    def __init__(self, count: int, maximum: int) -> None:
        """Build the error reporting *count* images against a *maximum*."""
        super().__init__(
            code="TOO_MANY_IMAGES",
            http_status=400,
            message=f"{count} images were uploaded; at most {maximum} can be analysed together.",
            hint=f"Upload {maximum} images or fewer.",
        )


class InvalidOptionsError(IngestError):
    """The ``options`` part was not JSON, or not a valid ``AnalyzeOptions``."""

    def __init__(self, detail: str) -> None:
        """Build the error with the validator's own explanation as the hint."""
        super().__init__(
            code="INVALID_OPTIONS",
            http_status=400,
            message="The options part could not be parsed.",
            hint=detail,
        )


class NoImagesError(IngestError):
    """The request carried no image parts at all."""

    def __init__(self) -> None:
        """Build the empty-request error."""
        super().__init__(
            code="INVALID_RASTER",
            http_status=400,
            message="No images were uploaded.",
            hint="Attach at least one image file to the 'images' field.",
        )


class ImageTooLargeError(IngestError):
    """The upload exceeds the configured byte or pixel budget."""

    def __init__(self, detail: str, ref: str | None = None) -> None:
        """Build the error with a user-safe *detail* naming the limit."""
        super().__init__(
            code="IMAGE_TOO_LARGE",
            http_status=413,
            message=detail,
            hint="Crop or downsample the image before uploading it.",
            ref=ref,
        )


class MissingGeoreferenceError(IngestError):
    """Exactly one image of a pair is georeferenced — an unresolvable mix."""

    def __init__(self, ref: str | None = None) -> None:
        """Build the error naming the image that lacks georeferencing."""
        super().__init__(
            code="MISSING_GEOREFERENCE",
            http_status=422,
            message=(
                "One image is georeferenced and the other is not, so they cannot be "
                "aligned to a common grid."
            ),
            hint="Upload two GeoTIFFs, or two plain images, but not one of each.",
            ref=ref,
        )


class CrsUnresolvableError(IngestError):
    """The two coordinate reference systems could not be reconciled."""

    def __init__(self, detail: str, ref: str | None = None) -> None:
        """Build the error with a user-safe *detail* naming the two CRSs."""
        super().__init__(
            code="CRS_MISMATCH_UNRESOLVABLE",
            http_status=422,
            message=detail,
            hint="Reproject both images to a common projected CRS before uploading.",
            ref=ref,
        )


class IncompatibleInputsError(IngestError):
    """The compatibility battery returned ``FAIL``, so no analysis is honest.

    ``/v1/validate`` reports a ``FAIL`` verdict and an empty ``supported_tasks``
    rather than raising, because pre-flight is meant to explain the problem.
    ``/v1/analyze`` has to refuse: running tools over images that cannot be
    compared would produce a confidently-worded answer about a relationship the
    data does not support.
    """

    CODES: dict[str, tuple[str, str]] = {
        "bounds_overlap_iou": (
            "INSUFFICIENT_OVERLAP",
            "Upload images covering the same footprint, or crop them to their shared area.",
        ),
        "gsd_ratio": (
            "GSD_MISMATCH",
            "Resample the finer image to the coarser one, or upload a matched pair.",
        ),
        "crs_match": (
            "CRS_MISMATCH_UNRESOLVABLE",
            "Reproject both images to a common projected CRS before uploading.",
        ),
        "modality_distinct": (
            "MODALITY_AMBIGUOUS",
            "Cross-modal analysis needs one optical and one SAR acquisition.",
        ),
    }
    """Which failing check maps onto which code in the frozen taxonomy (§6)."""

    DEFAULT_CODE = "NO_CAPABLE_TOOL"
    DEFAULT_HINT = "Upload a pair the compatibility checks accept, or a single image."

    def __init__(self, check: str, detail: str, ref: str | None = None) -> None:
        """Build the error from the name and detail of the check that failed."""
        code, hint = self.CODES.get(check, (self.DEFAULT_CODE, self.DEFAULT_HINT))
        super().__init__(code=code, http_status=422, message=detail, hint=hint, ref=ref)
