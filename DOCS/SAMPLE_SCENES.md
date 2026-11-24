# Sample scenes for the Use Cases gallery

One folder per `slug` in `frontend/src/pages/usecases/catalogue.ts`, holding the
rasters the row names in `files` plus an optional `thumb.webp` (4:3, ≤ 80 KB):

```
public/samples/bengaluru-sprawl/pre.tif
public/samples/bengaluru-sprawl/post.tif
public/samples/bengaluru-sprawl/thumb.webp
```

A row whose first raster is missing is shown with its plate (the drawn
footprint) but cannot be loaded, and the card says so.

**Building them.** `scripts/build_use_case_samples.py` (API running) fetches every
card's rasters: Sentinel-2 L2A / Sentinel-1 RTC windows through
`POST /v1/imagery/fetch` (pairs from one MGRS tile, full-swath items only, 512 px)
and a 0.3 m NAIP airport apron read from its Planetary Computer COG, plus a
`thumb.webp` each, and prints the per-card size for the catalogue's `sizeMb`. The
rasters are committed (`.gitignore` negates `frontend/public/samples/*/*.tif`);
~50 MB in all. Nothing here is fetched
until the user asks; the gallery itself costs no bandwidth.
