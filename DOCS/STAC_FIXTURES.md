# Recorded discovery fixtures (`?mock=1`)

What the Maps → *Find imagery* HUD runs on when the venue has no route out
(DOCS/UI_TRACK_4_ARCHITECTURE.md §2.4). Every third-party host the HUD talks
to has an explicit MSW handler in `frontend/src/mocks/handlers.ts`; the
handlers `fetch()` the files here and rewrite thumbnail hrefs to `thumbs/`, so
nothing relies on the worker's `bypass` and the whole flow works with the
network switched off (`frontend/e2e/track4.spec.ts` proves it that way).

```
places/<city>.json         Nominatim `/search?format=jsonv2&limit=6` — the reply verbatim
search/<s1|s2>-<city>.json Planetary Computer `POST /api/stac/v1/search` — the reply verbatim
thumbs/<item-id>.jpg       each item's `rendered_preview`, 256 px, ≤ 12 KB (52 files)
fetch/pair-ahmedabad.json  `POST /v1/imagery/fetch` for the rehearsed S1 pair
fetch/*.tif                the two clipped GeoTIFFs that response points at
```

| fixture | scenes | window | picked by |
|---|---|---|---|
| `search/s1-ahmedabad.json` | 16 · Sentinel-1 RTC | 2025-09-04 → 2026-09-06 | SAR + bbox centre near 72.6 E 23.0 N |
| `search/s2-ahmedabad.json` | 12 · Sentinel-2 L2A | 2026-05-11 → 2026-06-27 | optical + same centre |
| `search/s1-bengaluru.json` | 12 · Sentinel-1 RTC | 2026-04-17 → 2026-09-03 | SAR + centre near 77.6 E 13.0 N |
| `search/s2-chennai.json` | 12 · Sentinel-2 L2A | 2026-05-24 → 2026-09-01 | optical + centre near 80.3 E 13.1 N |

The place handler matches the typed query's first letters against the file
names (`Ahm` → `ahmedabad.json`); an unknown place answers `[]` after a short
delay. The search handler picks sensor from `collections` and place from the
bbox centre, ignoring the dates — the shelf therefore shows the same scenes
whatever window the slider chooses, which is fine for a rehearsal and wrong
for anything else.

`fetch/pair-ahmedabad.json` is the recorded response for the two descending
S1A/S1D scenes of 2026-03-03 and 2026-09-06 (the "6 mo apart" preset, same
orbit, so the pair passes the orbit check and co-registration). Its `url`s
point back into this folder; the GeoTIFFs are real 2-band VV/VH windows
(356 × 384 px, EPSG:32643) and go through the same `/v1/validate` mock as a
dropped file.

## Provenance

Recorded 2026-09-15 against the live services with online features on:

- Nominatim — © OpenStreetMap contributors, ODbL 1.0.
- Planetary Computer STAC — items are Sentinel-1 RTC (Microsoft) and
  Sentinel-2 L2A (ESA/Copernicus); previews are the catalogue's own
  `rendered_preview` renders.

Nothing here is synthetic. Do not edit a reply by hand; re-record it.

## Re-recording

With the API running and `SATQUERY_IMAGERY_STAC_URL` at its default, from
`frontend/`:

```sh
# places: one call per city, jsonv2, six results
for city in ahmedabad bengaluru chennai; do
  curl -s -H 'Referer: https://satquery.local/' \
    "https://nominatim.openstreetmap.org/search?q=${city}&format=jsonv2&limit=6&addressdetails=0" \
    > public/samples/stac/places/${city}.json; sleep 1; done

# searches: the body `geo/stac.ts` sends (limit 24, newest first); one per row above
curl -s -X POST https://planetarycomputer.microsoft.com/api/stac/v1/search \
  -H 'content-type: application/json' \
  -d '{"collections":["sentinel-1-rtc"],"bbox":[72.44,22.90,72.72,23.14],"datetime":"2025-09-01T00:00:00Z/2026-09-15T00:00:00Z","limit":24,"sortby":[{"field":"properties.datetime","direction":"desc"}]}' \
  > public/samples/stac/search/s1-ahmedabad.json

# thumbs: every rendered_preview in every search file, 256 px, JPEG q70
python3 - <<'PY'
import json, glob, subprocess, pathlib
for f in glob.glob('public/samples/stac/search/*.json'):
    for item in json.load(open(f))['features']:
        out = pathlib.Path('public/samples/stac/thumbs') / f"{item['id']}.jpg"
        if out.exists(): continue
        href = item['assets']['rendered_preview']['href']
        subprocess.run(['magick', href, '-resize', '256x256', '-quality', '70', str(out)], check=True)
PY

# fetch: the rehearsed pair, through the local API
curl -s -X POST http://127.0.0.1:8000/v1/imagery/fetch -H 'content-type: application/json' \
  -d '{"items":[{"collection":"sentinel-1-rtc","id":"<T1 id>"},{"collection":"sentinel-1-rtc","id":"<T2 id>"}],"bbox":[72.5201,22.9615,72.6401,23.0815],"max_px":512}' \
  > public/samples/stac/fetch/pair-ahmedabad.json
# then copy the two files it names from <artifact_root>/imagery/<fetch_id>/ into fetch/
# and rewrite each `url` to /samples/stac/fetch/<name>.
```

After re-recording, `e2e/track4.spec.ts` pins the shelf order (`useT1.nth(8)`
is the 2026-03-03 scene); update the index if the recording moves.
