"""
Rasterise Natural Earth 110m land polygons to the 1024x512 equirectangular
1-bit mask the landing globe's shader samples (`public/samples/globe/land.png`).

    curl -sL -o /tmp/land.geojson https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson
    python frontend/scripts/build-land-mask.py /tmp/land.geojson

Natural Earth is public domain. The result is ~10 KB.
"""
import json
import sys

from PIL import Image, ImageDraw

W, H = 1024, 512
src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else 'frontend/public/samples/globe/land.png'
data = json.load(open(src))
img = Image.new('1', (W, H), 0)
draw = ImageDraw.Draw(img)


def px(lon, lat):
    return ((lon + 180) / 360 * W, (90 - lat) / 180 * H)


for feature in data['features']:
    geom = feature['geometry']
    polys = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
    for poly in polys:
        outer, *holes = poly
        draw.polygon([px(x, y) for x, y in outer], fill=1)
        for hole in holes:
            draw.polygon([px(x, y) for x, y in hole], fill=0)

img.save(out, optimize=True)
print(out, img.size)
