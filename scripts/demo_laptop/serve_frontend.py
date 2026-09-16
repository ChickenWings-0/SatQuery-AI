#!/usr/bin/env python
"""Serve the built console (``frontend/dist``) with no Node and no network.

The air-gapped laptop has Python (for the API) but should not need Node or an
``npx`` download to put a static bundle in front of a browser. This is the
standard-library file server with one addition: any path that is not a file
falls back to ``index.html``, so the console's own router owns ``/report/…``,
``/maps`` and a refresh on any of them.

    uv run --no-sync python scripts/demo_laptop/serve_frontend.py            # :5173
    uv run --no-sync python scripts/demo_laptop/serve_frontend.py --port 8081
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DIST = REPO / "frontend" / "dist"


class SpaHandler(SimpleHTTPRequestHandler):
    """Static files, with ``index.html`` for every route the bundle owns."""

    def send_head(self):  # type: ignore[no-untyped-def]  # noqa: ANN201 - stdlib signature
        """Serve the file when it exists, else the app shell."""
        target = Path(self.translate_path(self.path))
        if not target.exists() and "." not in Path(self.path).name:
            self.path = "/index.html"
        return super().send_head()

    def end_headers(self) -> None:
        """Never cache: a rebuilt bundle must show up on the next refresh."""
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name
        """One line per request, to stdout, without the date noise."""
        sys.stdout.write(f"{self.address_string()} {format % args}\n")


def main() -> int:
    """Run the server until Ctrl-C."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dist", type=Path, default=DEFAULT_DIST)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173)
    args = parser.parse_args()
    if not (args.dist / "index.html").is_file():
        print(
            f"{args.dist} holds no index.html — copy frontend/dist from the stick", file=sys.stderr
        )
        return 66
    handler = partial(SpaHandler, directory=str(args.dist))
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"console at http://{args.host}:{args.port}  (serving {args.dist})")
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
