"""Build the Use Cases gallery's sample rasters from real, openly licensed imagery.

Every card in ``frontend/src/pages/usecases/catalogue.ts`` names files under
``frontend/public/samples/<slug>/``. This script produces them:

* Sentinel-2 L2A and Sentinel-1 RTC windows through the API's own
  ``POST /v1/imagery/fetch`` (the band stacks ingestion recognises), searched on
  Microsoft Planetary Computer for the card's place and dates. A bi-temporal
  pair is taken from one MGRS tile so both epochs share a grid.
* NAIP 0.3 m aerial RGB for the aircraft-count card, read straight from the
  Planetary Computer COG (the fetch endpoint serves Sentinel only).

It also writes a small ``thumb.webp`` per card and prints each card's size so
the catalogue's ``sizeMb`` can say what a judge will actually download.

Usage (API running, code from .env)::

    python scripts/build_use_case_samples.py --api http://127.0.0.1:8000 --code <code>
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
OUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "samples"
MAX_PX = 512


@dataclass(frozen=True)
class Epoch:
    """One acquisition to find: collection, date range, cloud ceiling."""

    collection: str
    dates: str
    max_cloud: float | None = 10.0
    min_cloud: float | None = None


@dataclass(frozen=True)
class Card:
    """A gallery card's place and the epochs that fill its files."""

    slug: str
    bbox: tuple[float, float, float, float]
    files: tuple[str, ...]
    epochs: tuple[Epoch, ...]
    same_tile: bool = True
    notes: dict[str, Any] = field(default_factory=dict)


S2, S1 = "sentinel-2-l2a", "sentinel-1-rtc"

CARDS: tuple[Card, ...] = (
    Card("bengaluru-sprawl", (77.695, 12.935, 77.760, 12.995), ("pre.tif", "post.tif"),
         (Epoch(S2, "2019-01-01/2019-03-31"), Epoch(S2, "2024-01-01/2024-03-31"))),
    Card("brahmaputra-flood", (91.560, 26.120, 91.680, 26.230), ("scene.tif",),
         (Epoch(S1, "2024-07-01/2024-07-12", max_cloud=None),)),
    Card("sundarbans-ndvi", (88.865, 21.920, 88.925, 21.980), ("pre.tif", "post.tif"),
         (Epoch(S2, "2018-01-01/2018-03-31"), Epoch(S2, "2024-01-01/2024-03-31"))),
    Card("chennai-reservoir", (80.125, 13.125, 80.190, 13.190), ("pre.tif", "post.tif"),
         (Epoch(S2, "2019-05-15/2019-06-30"), Epoch(S2, "2023-05-15/2023-06-30"))),
    Card("kerala-cross-modal", (76.235, 9.900, 76.300, 9.965), ("optical.tif", "sar.tif"),
         (Epoch(S2, "2024-01-01/2024-03-31", max_cloud=60.0, min_cloud=15.0),
          Epoch(S1, "2024-01-01/2024-03-31", max_cloud=None)), same_tile=False),
    Card("kutch-landcover", (69.715, 23.815, 69.785, 23.885), ("scene.tif",),
         (Epoch(S2, "2024-01-01/2024-03-31"),)),
    Card("mundra-port", (69.670, 22.715, 69.735, 22.775), ("pre.tif", "post.tif"),
         (Epoch(S2, "2017-01-01/2017-04-30"), Epoch(S2, "2024-01-01/2024-04-30"))),
)

NAIP_CARD = ("airbase-apron", (-112.0162, 33.4318, -112.0098, 33.4362), "scene.tif")


def _post(url: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"content-type": "application/json", **(headers or {})}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def _contains(item: dict[str, Any], bbox: tuple[float, ...]) -> bool:
    b = item["bbox"]
    return b[0] <= bbox[0] and b[1] <= bbox[1] and b[2] >= bbox[2] and b[3] >= bbox[3]


def _search(epoch: Epoch, bbox: tuple[float, ...], tile: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "collections": [epoch.collection],
        "bbox": list(bbox),
        "datetime": epoch.dates,
        "limit": 100,
    }
    query: dict[str, Any] = {}
    if epoch.max_cloud is not None:
        query["eo:cloud_cover"] = {"lt": epoch.max_cloud}
    if epoch.min_cloud is not None:
        query.setdefault("eo:cloud_cover", {})["gt"] = epoch.min_cloud
    if epoch.collection == S2:
        # Tile-wide nodata, a proxy for "this window sits in the swath": an item at
        # the swath edge can cover the bbox on paper and be three-quarters empty.
        query["s2:nodata_pixel_percentage"] = {"lt": 5}
    if query:
        body["query"] = query
    items = [i for i in _post(STAC, body)["features"] if _contains(i, bbox)]
    if tile:
        items = [i for i in items if i["properties"].get("s2:mgrs_tile") == tile]
    if not items:
        raise SystemExit(f"no {epoch.collection} item over {bbox} in {epoch.dates}")
    return min(items, key=lambda i: i["properties"].get("eo:cloud_cover", 0.0))


def _thumb(path: Path, out: Path) -> None:
    import rasterio
    from PIL import Image

    with rasterio.open(path) as src:
        if src.count >= 12:
            bands = src.read([4, 3, 2]).astype(np.float32)
        elif src.count == 2:
            vv = src.read(1).astype(np.float32)
            db = 10 * np.log10(np.clip(vv, 1e-6, None))
            bands = np.stack([db, db, db])
        else:
            bands = src.read([1, 2, 3]).astype(np.float32)
    rgb = np.zeros((*bands.shape[1:], 3), dtype=np.uint8)
    for c in range(3):
        finite = bands[c][np.isfinite(bands[c])]
        lo, hi = np.percentile(finite, (2, 98)) if finite.size else (0, 1)
        rgb[..., c] = (np.clip((bands[c] - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(rgb)
    w, h = image.size
    target = 4 / 3
    if w / h > target:
        nw = int(h * target)
        image = image.crop(((w - nw) // 2, 0, (w + nw) // 2, h))
    else:
        nh = int(w / target)
        image = image.crop((0, (h - nh) // 2, w, (h + nh) // 2))
    image.resize((320, 240)).save(out, "WEBP", quality=70)


def build_sentinel(card: Card, api: str, code: str) -> None:
    """Fetch one card's epochs through the API and write its files."""
    headers = {"X-SatQuery-Code": code} if code else {}
    tile: str | None = None
    items = []
    for epoch in card.epochs:
        item = _search(epoch, card.bbox, tile if card.same_tile else None)
        if card.same_tile and epoch.collection == S2:
            tile = item["properties"].get("s2:mgrs_tile")
        items.append(item)
        cloud = item["properties"].get("eo:cloud_cover")
        print(f"  {card.slug}: {item['id']} {item['properties']['datetime'][:10]} cloud={cloud}")
    response = _post(
        f"{api}/v1/imagery/fetch",
        {
            "items": [{"collection": i["collection"], "id": i["id"]} for i in items],
            "bbox": list(card.bbox),
            "max_px": MAX_PX,
        },
        headers,
    )
    folder = OUT / card.slug
    folder.mkdir(parents=True, exist_ok=True)
    for file, name in zip(response["files"], card.files, strict=True):
        request = urllib.request.Request(api + file["url"], headers=headers)
        with urllib.request.urlopen(request, timeout=300) as source:
            (folder / name).write_bytes(source.read())
    _thumb(folder / card.files[0], folder / "thumb.webp")


def build_naip(slug: str, bbox: tuple[float, ...], name: str) -> None:
    """Read a NAIP window straight from its COG and write an RGB GeoTIFF."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    feats = _post(STAC, {"collections": ["naip"], "bbox": list(bbox), "limit": 10})["features"]
    item = next(f for f in feats if _contains(f, bbox))
    print(f"  {slug}: {item['id']} {item['properties']['datetime'][:10]}")
    with urllib.request.urlopen(SAS.format(collection="naip"), timeout=60) as response:
        token = json.load(response)["token"]
    with rasterio.open(item["assets"]["image"]["href"] + "?" + token) as src:
        native = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(*native, transform=src.transform)
        rgb = src.read([1, 2, 3], window=window)
        profile = {
            "driver": "GTiff", "count": 3, "dtype": "uint8", "crs": src.crs,
            "width": rgb.shape[2], "height": rgb.shape[1], "compress": "deflate",
            "transform": src.window_transform(window),
        }
    folder = OUT / slug
    folder.mkdir(parents=True, exist_ok=True)
    with rasterio.open(folder / name, "w", **profile) as dst:
        dst.write(rgb)
    _thumb(folder / name, folder / "thumb.webp")


def main(argv: list[str] | None = None) -> int:
    """Build every card, or the ones named with --only."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--code", default="")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(argv)
    for card in CARDS:
        if args.only and card.slug not in args.only:
            continue
        build_sentinel(card, args.api, args.code)
    if not args.only or NAIP_CARD[0] in args.only:
        build_naip(*NAIP_CARD)
    skip = {"globe", "stac"}
    for folder in sorted(p for p in OUT.iterdir() if p.is_dir() and p.name not in skip):
        size = sum(f.stat().st_size for f in folder.glob("*.tif"))
        print(f"{folder.name}: {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
