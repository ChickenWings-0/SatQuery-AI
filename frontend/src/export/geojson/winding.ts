/**
 * The geometry rules RFC 7946 asks for, in sixty lines rather than a
 * dependency: right-hand-rule winding (exterior rings counter-clockwise,
 * holes clockwise), closed rings, a `bbox`, and coordinates rounded to
 * seven decimals — about a centimetre at the equator, and the precision
 * §11.2 recommends.
 */
export type Position = [number, number]
export type Ring = Position[]

/** Shoelace; positive when the ring runs counter-clockwise in a y-up frame. */
export function signedArea(ring: Ring): number {
  let sum = 0
  for (let i = 0, n = ring.length; i < n; i += 1) {
    const [x1, y1] = ring[i]!
    const [x2, y2] = ring[(i + 1) % n]!
    sum += x1 * y2 - x2 * y1
  }
  return sum / 2
}

/** Append the first position if the ring is not already closed. */
export function closeRing(ring: Ring): Ring {
  if (ring.length === 0) return ring
  const [fx, fy] = ring[0]!
  const [lx, ly] = ring[ring.length - 1]!
  return fx === lx && fy === ly ? ring : [...ring, [fx, fy]]
}

/**
 * Wind the ring so that it is counter-clockwise when `exterior`, clockwise
 * otherwise. Pixel frames are y-down, which flips the sign of the shoelace:
 * pass `yDown` so "counter-clockwise" still means what a viewer sees.
 */
export function rewind(ring: Ring, exterior = true, yDown = false): Ring {
  const closed = closeRing(ring)
  const area = signedArea(closed) * (yDown ? -1 : 1)
  const isCcw = area > 0
  return isCcw === exterior ? closed : [...closed].reverse()
}

export function round7(value: number): number {
  return Math.round(value * 1e7) / 1e7
}

export function roundRing(ring: Ring): Ring {
  return ring.map(([x, y]) => [round7(x), round7(y)])
}

/** `[west, south, east, north]` over every position in every ring. */
export function bboxOfRings(rings: Ring[]): [number, number, number, number] | null {
  let w = Infinity
  let s = Infinity
  let e = -Infinity
  let n = -Infinity
  for (const ring of rings) {
    for (const [x, y] of ring) {
      if (x < w) w = x
      if (x > e) e = x
      if (y < s) s = y
      if (y > n) n = y
    }
  }
  return Number.isFinite(w) ? [round7(w), round7(s), round7(e), round7(n)] : null
}

function perpendicularDistance(p: Position, a: Position, b: Position): number {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  if (dx === 0 && dy === 0) return Math.hypot(p[0] - a[0], p[1] - a[1])
  const t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)
  const cx = a[0] + t * dx
  const cy = a[1] + t * dy
  return Math.hypot(p[0] - cx, p[1] - cy)
}

/** Douglas–Peucker on an open polyline. */
export function simplify(points: Position[], tolerance: number): Position[] {
  if (points.length <= 2 || tolerance <= 0) return points
  let maxDist = 0
  let index = 0
  const first = points[0]!
  const last = points[points.length - 1]!
  for (let i = 1; i < points.length - 1; i += 1) {
    const d = perpendicularDistance(points[i]!, first, last)
    if (d > maxDist) {
      maxDist = d
      index = i
    }
  }
  if (maxDist <= tolerance) return [first, last]
  const left = simplify(points.slice(0, index + 1), tolerance)
  const right = simplify(points.slice(index), tolerance)
  return [...left.slice(0, -1), ...right]
}

/** Simplify a closed ring; it stays closed and keeps at least four positions. */
export function simplifyRing(ring: Ring, tolerance: number): Ring {
  const open = ring.length > 1 && ring[0]![0] === ring[ring.length - 1]![0] && ring[0]![1] === ring[ring.length - 1]![1]
    ? ring.slice(0, -1)
    : ring
  if (open.length < 4) return closeRing(open)
  // Split at the farthest point from the start so DP has two real ends.
  let far = 0
  let farDist = 0
  for (let i = 1; i < open.length; i += 1) {
    const d = Math.hypot(open[i]![0] - open[0]![0], open[i]![1] - open[0]![1])
    if (d > farDist) {
      farDist = d
      far = i
    }
  }
  const a = simplify(open.slice(0, far + 1), tolerance)
  const b = simplify([...open.slice(far), open[0]!], tolerance)
  const merged = [...a.slice(0, -1), ...b.slice(0, -1)]
  return merged.length >= 3 ? closeRing(merged) : closeRing(open)
}
