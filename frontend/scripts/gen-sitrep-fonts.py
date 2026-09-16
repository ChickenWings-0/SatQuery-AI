"""Static TTF instances of Geist for the SITREP PDF.

pdf-lib's browser build of fontkit cannot decode WOFF2 (its Buffer shim throws
"Index out of range" on the brotli stream), so the PDF embeds plain TTFs. These
are cut from the self-hosted variable WOFF2s in ``public/fonts/`` — the same
faces the console uses — as three static instances, subset to Latin.

    uv run --no-sync python frontend/scripts/gen-sitrep-fonts.py
"""

from __future__ import annotations

from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

FONTS = Path(__file__).resolve().parents[1] / "public" / "fonts"

INSTANCES = (
    ("geist-latin-wght-normal.woff2", 400, "geist-sitrep-400.ttf"),
    ("geist-latin-wght-normal.woff2", 600, "geist-sitrep-600.ttf"),
    ("geist-mono-latin-wght-normal.woff2", 400, "geist-mono-sitrep-400.ttf"),
)

# Basic Latin, Latin-1 Supplement, the punctuation and symbols the brief uses.
UNICODES = "U+0020-007E,U+00A0-00FF,U+2013,U+2014,U+2018,U+2019,U+201C,U+201D,U+2022,U+2026,U+2032,U+2033,U+2192,U+00B2,U+00B0,U+00B7"


def main() -> None:
    for source, weight, target in INSTANCES:
        font = TTFont(FONTS / source)
        static = instancer.instantiateVariableFont(font, {"wght": weight}, inplace=False)
        static.flavor = None
        options = subset.Options()
        options.flavor = None
        options.layout_features = ["kern", "liga", "tnum"]
        options.name_IDs = ["*"]
        options.notdef_outline = True
        subsetter = subset.Subsetter(options)
        subsetter.populate(unicodes=subset.parse_unicodes(UNICODES))
        subsetter.subset(static)
        out = FONTS / target
        static.save(out)
        print(f"{target}: {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
