# Sample scenes for the Use Cases gallery

One folder per `slug` in `frontend/src/pages/usecases/catalogue.ts`, holding the
rasters the row names in `files` plus an optional `thumb.webp` (4:3, ≤ 80 KB):

```
public/samples/bengaluru-sprawl/pre.tif
public/samples/bengaluru-sprawl/post.tif
public/samples/bengaluru-sprawl/thumb.webp
```

A row whose first raster is missing is shown with its plate (the drawn
footprint) but cannot be loaded, and the card says so. Nothing here is fetched
until the user asks; the gallery itself costs no bandwidth.
