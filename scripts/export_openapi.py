#!/usr/bin/env python
"""Export the FastAPI schema to ``openapi.json`` at the project root.

``openapi.json`` is the machine contract and is committed, so the frontend can
generate a typed client (``openapi-typescript``, ``orval``) without running the
backend. ``DOCS/API_CONTRACT.md`` is the prose contract: if the two disagree,
the prose wins and this file is regenerated.

Usage:  uv run python scripts/export_openapi.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from satquery.api.app import app  # noqa: E402  (import needs the sys.path tweak above)

DEFAULT_OUTPUT = PROJECT_ROOT / "openapi.json"


def export(output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Write ``app.openapi()`` to *output_path* and return it."""
    schema = app.openapi()
    output_path.write_text(json.dumps(schema, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    ``argparse`` rather than ``sys.argv[1]``: the old form treated *any* first
    argument as the output path, so ``--help`` wrote a file named ``--help``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"where to write the schema (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Export the schema to ``--out``, or the default."""
    args = parse_args(argv)
    written = export(args.out.resolve())
    paths = len(app.openapi().get("paths", {}))
    print(f"Wrote {written} ({paths} paths, {written.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
