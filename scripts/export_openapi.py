#!/usr/bin/env python
"""Export the FastAPI schema to ``openapi.json`` at the project root.

``openapi.json`` is the machine contract and is committed, so the frontend can
generate a typed client (``openapi-typescript``, ``orval``) without running the
backend. ``DOCS/API_CONTRACT.md`` is the prose contract: if the two disagree,
the prose wins and this file is regenerated.

Usage:  uv run python scripts/export_openapi.py [output_path]
"""

from __future__ import annotations

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


def main() -> int:
    """Export the schema to the path given on argv, or the default."""
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT
    written = export(output)
    paths = len(app.openapi().get("paths", {}))
    print(f"Wrote {written} ({paths} paths, {written.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
