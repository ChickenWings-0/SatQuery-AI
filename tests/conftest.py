"""Shared fixtures: the synthetic raster corpus every ingestion test reads.

The suite also pins itself to the deterministic-only baseline here, before any
``satquery`` module is imported. Tool availability is probed from the machine —
``catalog.vlm_servable()`` asks whether a backend could answer a prompt — so once
the Qwen weights land in the HF cache, tests written against the templated,
tool-driven path silently start exercising a live 8B generation instead. That
turns assertions about *our* code into assertions about the model's prose: a
scene description acquires an uncited number, a skipped step becomes an OK one,
and two identical requests stop producing identical params. It also takes the
suite from seconds to minutes.

``setdefault`` rather than an unconditional set: a run that deliberately exports
``SATQUERY_VLM_DISABLED=false`` to exercise the served path keeps its choice.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("SATQUERY_VLM_DISABLED", "true")

import pytest  # noqa: E402

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
