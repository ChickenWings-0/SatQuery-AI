#!/usr/bin/env python
"""Create placeholder view images so a sanity check can reach the first step.

    uv run python scripts/patch_dummy_images.py
    uv run python scripts/patch_dummy_images.py --verify

The corpus records the paths the Phase 2 render pass *will* write
(DATA_ADAPTATION_PLAN §7.2); it does not write images itself. Until that pass
runs, every path in ``train.jsonl`` points at nothing, and the trainer dies in
the processor before a single gradient is computed — which hides exactly the
things ``--sanity-check`` exists to measure: peak VRAM, gradient flow through
both towers, and whether bitsandbytes and ROCm agree today.

This writes a solid black 448x448 image at each missing path so that run can
proceed.

**These are not training data.** A model trained on black squares learns that
every scene is a black square, and the resulting adapter is worse than no
adapter. Two properties keep that from happening by accident:

* every file written is recorded in :data:`MANIFEST_NAME` beside the views root,
  with its size and hash, so a placeholder can always be told from a render;
* ``--verify`` re-checks the manifest against the corpus and reports how many
  placeholders a corpus still depends on, which is the check to run before a
  real run rather than after it.

Delete the manifest and its files once the render pass has produced the real
views: ``--clean`` does exactly that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_CORPUS: tuple[Path, ...] = (
    Path("data/processed/corpus/train.jsonl"),
    Path("data/processed/corpus/val.jsonl"),
)
MANIFEST_NAME: str = "PLACEHOLDER_VIEWS.json"
DEFAULT_SIZE: int = 448
"""``view_size_px`` from the training profile — a placeholder that is not the
size the processor expects would exercise a different resize path than the real
views do, which is a poor rehearsal."""

FORMATS: dict[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
}
"""Extension to Pillow format. The catalogue writes PNG for index and SAR views
and JPEG for composites (§2.1), and the placeholder keeps whichever the corpus
recorded so the file the trainer opens matches the name it was given."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, nargs="*", default=list(DEFAULT_CORPUS))
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument(
        "--views-root",
        type=Path,
        default=Path("data/processed/views"),
        help="Where the manifest is written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Also overwrite files that already have content. Off by default, so "
        "a real rendered view is never replaced by a black square.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Report how many of the corpus's views are missing or placeholders, "
        "and write nothing.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete every file recorded in the manifest, then the manifest.",
    )
    return parser.parse_args(argv)


def corpus_view_paths(corpus: Iterable[Path]) -> list[Path]:
    """Every distinct view path the corpus files reference, in first-seen order."""
    seen: dict[str, None] = {}
    for path in corpus:
        if not path.is_file():
            raise SystemExit(f"corpus file {path} does not exist")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                for view in json.loads(line).get("views", ()):
                    seen.setdefault(str(view["path"]), None)
    return [Path(value) for value in seen]


def needs_placeholder(path: Path, force: bool = False) -> bool:
    """True when this path has no usable image behind it."""
    if not path.exists():
        return True
    if path.stat().st_size == 0:
        return True
    return force


def write_placeholder(path: Path, size: int) -> dict[str, Any]:
    """Write one solid black image, and describe what was written.

    Raises:
        SystemExit: The extension is not one Pillow can write here.
    """
    from PIL import Image

    image_format = FORMATS.get(path.suffix.lower())
    if image_format is None:
        raise SystemExit(
            f"cannot write a placeholder for {path}: unknown extension "
            f"{path.suffix!r}; expected one of {sorted(FORMATS)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size)).save(path, format=image_format)
    payload = path.read_bytes()
    return {
        "path": str(path),
        "format": image_format,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def load_manifest(manifest: Path) -> list[dict[str, Any]]:
    """Read the manifest's file entries, or an empty list."""
    if not manifest.is_file():
        return []
    loaded = json.loads(manifest.read_text(encoding="utf-8"))
    entries = loaded.get("files", []) if isinstance(loaded, dict) else []
    return [entry for entry in entries if isinstance(entry, dict) and "path" in entry]


def verify(paths: Sequence[Path], manifest: Path) -> int:
    """Report what the corpus's views actually are. Returns an exit code."""
    recorded = {str(entry["path"]): entry for entry in load_manifest(manifest)}
    missing = [p for p in paths if not p.exists() or p.stat().st_size == 0]
    placeholders = [
        p
        for p in paths
        if str(p) in recorded
        and p.is_file()
        and hashlib.sha256(p.read_bytes()).hexdigest() == recorded[str(p)]["sha256"]
    ]
    real = len(paths) - len(missing) - len(placeholders)
    print(f"views referenced : {len(paths)}")
    print(f"  missing/empty  : {len(missing)}")
    print(f"  placeholders   : {len(placeholders)}")
    print(f"  rendered       : {real}")
    if placeholders:
        print(
            "\nThis corpus still depends on placeholder images. They are valid "
            "files and will train without error, and the resulting adapter will "
            "have learned nothing about imagery. Use them for --sanity-check only."
        )
    return 1 if (missing or placeholders) else 0


def clean(manifest: Path) -> int:
    """Delete every recorded placeholder. Returns an exit code."""
    entries = load_manifest(manifest)
    removed = 0
    for entry in entries:
        path = Path(str(entry["path"]))
        if path.is_file():
            path.unlink()
            removed += 1
    manifest.unlink(missing_ok=True)
    print(f"removed {removed} placeholder image(s) and the manifest")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Patch, verify or clean. Returns a process exit code."""
    args = parse_args(argv)
    manifest_path = args.views_root / MANIFEST_NAME

    if args.clean:
        return clean(manifest_path)

    paths = corpus_view_paths(args.corpus)
    if args.verify:
        return verify(paths, manifest_path)

    targets = [path for path in paths if needs_placeholder(path, args.force)]
    print(f"views referenced : {len(paths)}")
    print(f"needing a file   : {len(targets)}")

    written = [write_placeholder(path, args.size) for path in targets]
    if written:
        existing = {str(entry["path"]): entry for entry in load_manifest(manifest_path)}
        existing.update({str(entry["path"]): entry for entry in written})
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "written_at": datetime.now(UTC).isoformat(),
                    "size_px": args.size,
                    "note": (
                        "Solid black placeholders for --sanity-check only. Not "
                        "training data; delete with --clean once the Phase 2 render "
                        "pass has written the real views."
                    ),
                    "files": sorted(existing.values(), key=lambda entry: str(entry["path"])),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"placeholders written: {len(written)} -> manifest {manifest_path}")
    print(
        "\nThese are solid black 448x448 images, not imagery. They exist so the "
        "sanity check can measure VRAM and gradient flow; a real run on them "
        "would produce an adapter that has learned nothing. Run "
        "`--verify` before any run you intend to keep."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
