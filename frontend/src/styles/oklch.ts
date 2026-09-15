/**
 * OKLCH → sRGB, implemented rather than imported.
 *
 * The palette is written in OKLCH (`theme.css`) and two things need it back as
 * sRGB: the contrast test, which must compute WCAG luminance from what the
 * browser will actually paint, and the landing globe, whose shader wants the
 * accent as linear RGB. Both read this one file so a token change cannot leave
 * the globe or the test on a stale copy of the colour.
 *
 * Matrices are Björn Ottosson's published OKLab ↔ linear-sRGB set. Out-of-gamut
 * results are clipped per channel, which is what browsers do for `oklch()`.
 */
export type Rgb = readonly [r: number, g: number, b: number]

export interface Oklch {
  l: number
  c: number
  h: number
  /** 0–1. `1` when the token carries no `/ alpha`. */
  alpha: number
}

const OKLCH_RE =
  /^oklch\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s*\/\s*([\d.]+%?))?\s*\)$/i

export function parseOklch(value: string): Oklch | null {
  const m = OKLCH_RE.exec(value.trim())
  if (!m) return null
  const alphaRaw = m[4]
  const alpha =
    alphaRaw === undefined
      ? 1
      : alphaRaw.endsWith('%')
        ? Number.parseFloat(alphaRaw) / 100
        : Number.parseFloat(alphaRaw)
  return { l: Number(m[1]), c: Number(m[2]), h: Number(m[3]), alpha }
}

/** Linear-light sRGB, 0–1 per channel, clipped to gamut. */
export function oklchToLinear({ l, c, h }: Pick<Oklch, 'l' | 'c' | 'h'>): Rgb {
  const rad = (h * Math.PI) / 180
  const a = c * Math.cos(rad)
  const b = c * Math.sin(rad)
  const l_ = l + 0.3963377774 * a + 0.2158037573 * b
  const m_ = l - 0.1055613458 * a - 0.0638541728 * b
  const s_ = l - 0.0894841775 * a - 1.291485548 * b
  const L = l_ ** 3
  const M = m_ ** 3
  const S = s_ ** 3
  const clip = (x: number) => Math.min(1, Math.max(0, x))
  return [
    clip(4.0767416621 * L - 3.3077115913 * M + 0.2309699292 * S),
    clip(-1.2684380046 * L + 2.6097574011 * M - 0.3413193965 * S),
    clip(-0.0041960863 * L - 0.7034186147 * M + 1.707614701 * S),
  ]
}

function gamma(c: number): number {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055
}

/** Display sRGB, 0–1 per channel. */
export function oklchToSrgb(color: Pick<Oklch, 'l' | 'c' | 'h'>): Rgb {
  return oklchToLinear(color).map(gamma) as unknown as Rgb
}

export function oklchToHex(color: Pick<Oklch, 'l' | 'c' | 'h'>): string {
  return `#${oklchToSrgb(color)
    .map((c) => Math.round(c * 255).toString(16).padStart(2, '0'))
    .join('')}`
}

/** `#rrggbb` for an `oklch(...)` string, or the input when it is already hex. */
export function toHex(value: string): string | null {
  const v = value.trim()
  if (/^#[0-9a-fA-F]{6}$/.test(v)) return v.toLowerCase()
  if (/^#[0-9a-fA-F]{3}$/.test(v)) return `#${[...v.slice(1)].map((ch) => ch + ch).join('')}`
  const parsed = parseOklch(v)
  return parsed ? oklchToHex(parsed) : null
}
