"""Geospatial ingestion: raster reading, classification and compatibility checking."""

from satquery.ingest.compatibility import assess, derive_overall, detect_pair_type
from satquery.ingest.errors import IngestError
from satquery.ingest.manifest_builder import build_manifest
from satquery.ingest.modality import ModalityResult, classify
from satquery.ingest.pipeline import IngestResult, SourceImage, ingest
from satquery.ingest.reader import RasterInfo, open_raster

__all__ = [
    "IngestError",
    "IngestResult",
    "ModalityResult",
    "RasterInfo",
    "SourceImage",
    "assess",
    "build_manifest",
    "classify",
    "derive_overall",
    "detect_pair_type",
    "ingest",
    "open_raster",
]
