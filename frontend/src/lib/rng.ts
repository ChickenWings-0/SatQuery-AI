/**
 * A tiny deterministic generator: FNV-1a folds the seed string into 32 bits,
 * mulberry32 walks from there. Shared by the Use Cases plates and the landing
 * HUDs so the same slug draws the same parcels on both surfaces, and a
 * re-render never reshuffles a drawing.
 */
export function rng(seed: string): () => number {
  let h = 2166136261
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return () => {
    h = (h + 0x6d2b79f5) | 0
    let t = h
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export const between = (r: () => number, lo: number, hi: number): number => lo + r() * (hi - lo)
