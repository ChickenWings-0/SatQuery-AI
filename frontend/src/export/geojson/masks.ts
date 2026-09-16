/**
 * Mask rasters → polygons, on the client, lazily.
 *
 * A change mask or a class raster is a PNG the pipeline already rendered; a
 * GIS user wants it as vectors with the class in the attribute table. This
 * module traces pixel boundaries (every inside pixel contributes the sides
 * it does not share with another inside pixel, and the sides are chained
 * into closed rings), then simplifies each ring with Douglas–Peucker at
 * half a pixel. Exact, dependency-free, and — because it is reached only
 * through `import()` behind the "Include masks" checkbox — never in the
 * console entry. The one published marching-squares package is AGPL, which
 * is why this file exists instead.
 */
import { BOX_SCALE } from '@/thread/bbox'
import { type Bounds, type Feature } from '@/export/geojson/build'
import { rewind, roundRing, signedArea, simplifyRing, type Position, type Ring } from '@/export/geojson/winding'

export interface MaskSource {
  /** Same-origin URL of the PNG. */
  url: string
  /** Artifact id, for `source_artifact`. */
  id: string
  kind: 'CHANGE_MASK' | 'SEGMENTATION'
  label: string
  produced_by_step: number
  /** Optional class names for a segmentation raster, keyed by its index or colour. */
  classes?: Record<string, string>
}

export interface MaskOptions {
  bounds: Bounds | null
  /** Pixels; 0.5 by default. */
  tolerance?: number
  /** Longest side the raster is read at; larger rasters are resampled nearest. */
  maxSide?: number
  extraProperties?: Record<string, unknown>
}

interface Grid {
  width: number
  height: number
  cells: Uint16Array
  /** Class id → display label. */
  labels: Map<number, string>
}

const MAX_CLASSES = 16

async function readGrid(source: MaskSource, maxSide: number): Promise<Grid | null> {
  if (typeof document === 'undefined') return null
  const img = new Image()
  img.decoding = 'async'
  img.src = source.url
  await img.decode()
  const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight))
  const width = Math.max(1, Math.round(img.naturalWidth * scale))
  const height = Math.max(1, Math.round(img.naturalHeight * scale))
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) return null
  ctx.imageSmoothingEnabled = false
  ctx.drawImage(img, 0, 0, width, height)
  const { data } = ctx.getImageData(0, 0, width, height)
  const cells = new Uint16Array(width * height)
  const labels = new Map<number, string>()

  if (source.kind === 'CHANGE_MASK') {
    for (let i = 0; i < width * height; i += 1) {
      const r = data[i * 4]!
      const g = data[i * 4 + 1]!
      const b = data[i * 4 + 2]!
      const a = data[i * 4 + 3]!
      cells[i] = a > 127 && (r + g + b) / 3 > 127 ? 1 : 0
    }
    labels.set(1, 'changed')
    return { width, height, cells, labels }
  }

  // Segmentation: quantise by colour, keep the most frequent classes.
  const counts = new Map<number, number>()
  const keys = new Uint32Array(width * height)
  for (let i = 0; i < width * height; i += 1) {
    const a = data[i * 4 + 3]!
    const key = a < 128 ? 0 : (data[i * 4]! << 16) | (data[i * 4 + 1]! << 8) | data[i * 4 + 2]! | 0x1000000
    keys[i] = key
    if (key) counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, MAX_CLASSES)
  const idOf = new Map<number, number>()
  ranked.forEach(([key], index) => {
    const id = index + 1
    idOf.set(key, id)
    const hex = `#${(key & 0xffffff).toString(16).padStart(6, '0')}`
    labels.set(id, source.classes?.[hex] ?? source.classes?.[String(index)] ?? hex)
  })
  for (let i = 0; i < width * height; i += 1) cells[i] = idOf.get(keys[i]!) ?? 0
  return { width, height, cells, labels }
}

/** Every closed ring around the cells of one class, in pixel corners, y-down. */
export function traceRings(cells: ArrayLike<number>, width: number, height: number, cls: number): Ring[] {
  const inside = (x: number, y: number) => x >= 0 && y >= 0 && x < width && y < height && cells[y * width + x] === cls
  // Directed edges keyed by their start vertex; a vertex is `x + y * (width + 1)`.
  const stride = width + 1
  const key = (x: number, y: number) => x + y * stride
  const out = new Map<number, number[]>()
  const push = (from: number, to: number) => {
    const list = out.get(from)
    if (list) list.push(to)
    else out.set(from, [to])
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!inside(x, y)) continue
      if (!inside(x, y - 1)) push(key(x, y), key(x + 1, y))
      if (!inside(x + 1, y)) push(key(x + 1, y), key(x + 1, y + 1))
      if (!inside(x, y + 1)) push(key(x + 1, y + 1), key(x, y + 1))
      if (!inside(x - 1, y)) push(key(x, y + 1), key(x, y))
    }
  }
  const rings: Ring[] = []
  const toPos = (k: number): Position => [k % stride, Math.floor(k / stride)]
  for (const [start, targets] of out) {
    while (targets.length > 0) {
      const ring: Position[] = [toPos(start)]
      let current = start
      let next = targets.pop()!
      let guard = width * height * 4
      while (next !== start && guard-- > 0) {
        ring.push(toPos(next))
        const list = out.get(next)
        if (!list || list.length === 0) break
        // Prefer continuing straight over turning, which keeps diagonally
        // touching regions from swapping rings mid-trace.
        current = next
        next = list.pop()!
      }
      void current
      if (ring.length >= 4) rings.push(dropCollinear(ring))
    }
  }
  return rings
}

/** Axis-aligned tracing emits a vertex at every pixel; keep only the corners. */
function dropCollinear(ring: Position[]): Ring {
  const out: Position[] = []
  const n = ring.length
  for (let i = 0; i < n; i += 1) {
    const prev = ring[(i - 1 + n) % n]!
    const cur = ring[i]!
    const nxt = ring[(i + 1) % n]!
    const straight = (prev[0] === cur[0] && cur[0] === nxt[0]) || (prev[1] === cur[1] && cur[1] === nxt[1])
    if (!straight) out.push(cur)
  }
  return out
}

function contains(ring: Ring, [px, py]: Position): boolean {
  let inside = false
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
    const [xi, yi] = ring[i]!
    const [xj, yj] = ring[j]!
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

/** Group rings into polygons: each hole goes to the smallest exterior that contains it. */
export function assemble(rings: Ring[]): Ring[][] {
  const exteriors: { ring: Ring; area: number; holes: Ring[] }[] = []
  const holes: Ring[] = []
  for (const ring of rings) {
    // The tracer walks exteriors clockwise on a y-down screen, which the
    // plain shoelace reads as positive; holes come out the other way.
    const area = signedArea(ring)
    if (area > 0) exteriors.push({ ring, area, holes: [] })
    else holes.push(ring)
  }
  exteriors.sort((a, b) => a.area - b.area)
  for (const hole of holes) {
    const home = exteriors.find((ext) => contains(ext.ring, hole[0]!))
    home?.holes.push(hole)
  }
  return exteriors.map((ext) => [ext.ring, ...ext.holes])
}

function project(grid: Grid, bounds: Bounds | null): (p: Position) => Position {
  if (bounds) {
    const [w, s, e, n] = bounds
    return ([x, y]) => [w + (x / grid.width) * (e - w), n - (y / grid.height) * (n - s)]
  }
  return ([x, y]) => [(x / grid.width) * BOX_SCALE, (y / grid.height) * BOX_SCALE]
}

export async function vectoriseMask(source: MaskSource, options: MaskOptions): Promise<Feature[]> {
  const grid = await readGrid(source, options.maxSide ?? 1024)
  if (!grid) return []
  const tolerance = options.tolerance ?? 0.5
  const toWorld = project(grid, options.bounds)
  const yDown = options.bounds === null
  const features: Feature[] = []
  for (const [cls, label] of grid.labels) {
    const rings = traceRings(grid.cells, grid.width, grid.height, cls).map((ring) => simplifyRing(ring, tolerance))
    const polygons = assemble(rings)
    if (polygons.length === 0) continue
    const coordinates = polygons.map(([exterior, ...holes]) => [
      roundRing(rewind(exterior!.map(toWorld), true, yDown)),
      ...holes.map((hole) => roundRing(rewind(hole.map(toWorld), false, yDown))),
    ])
    const pixels = grid.cells.reduce((n, c) => n + (c === cls ? 1 : 0), 0)
    features.push({
      type: 'Feature',
      geometry: { type: 'MultiPolygon', coordinates },
      properties: {
        kind: 'mask',
        label,
        class_id: cls,
        mask_kind: source.kind,
        mask_label: source.label,
        source_artifact: source.id,
        source_step: source.produced_by_step,
        coverage_pct: Math.round((pixels / grid.cells.length) * 1000) / 10,
        ...(options.extraProperties ?? {}),
        ...(options.bounds ? {} : { crs: 'pixel', frame: BOX_SCALE }),
      },
    })
  }
  return features
}
