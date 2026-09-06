"""Spectral rendering and the evidence artifact layer: one renderer, two consumers."""

from satquery.render.artifact_store import ArtifactStore, StoredBlob
from satquery.render.renderer import RenderedView, RenderSource, render_views
from satquery.render.view_labels import build_view_label, label_for_view
from satquery.render.views import CATALOGUE, ImageFormat, ViewId, ViewSpec, select_views

__all__ = [
    "CATALOGUE",
    "ArtifactStore",
    "ImageFormat",
    "RenderSource",
    "RenderedView",
    "StoredBlob",
    "ViewId",
    "ViewSpec",
    "build_view_label",
    "label_for_view",
    "render_views",
    "select_views",
]
