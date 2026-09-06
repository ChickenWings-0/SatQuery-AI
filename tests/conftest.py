"""Shared fixtures: the synthetic raster corpus every ingestion test reads."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_synthetic_fixtures import Scene, build_fixtures  # noqa: E402


@pytest.fixture(scope="session")
def scenes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Scene]:
    """Generate the synthetic corpus once per session, keyed by scene name.

    Generating rather than committing binaries keeps the ground truth and the
    pixels in the same file: a fixture cannot drift from what a test claims it is.
    """
    output_dir = tmp_path_factory.mktemp("synthetic")
    return {scene.name: scene for scene in build_fixtures(output_dir)}


@pytest.fixture(scope="session")
def scene_paths(scenes: dict[str, Scene]) -> dict[str, Path]:
    """Just the paths, for tests that do not need the ground-truth dict."""
    return {name: scene.path for name, scene in scenes.items()}


@pytest.fixture
def upload_files(scene_paths: dict[str, Path]) -> Iterator[object]:
    """Return a helper that turns scene names into multipart file tuples."""

    def build(*names: str) -> list[tuple[str, tuple[str, bytes, str]]]:
        return [
            (
                "images",
                (
                    scene_paths[name].name,
                    scene_paths[name].read_bytes(),
                    "image/tiff" if scene_paths[name].suffix == ".tif" else "image/png",
                ),
            )
            for name in names
        ]

    yield build
