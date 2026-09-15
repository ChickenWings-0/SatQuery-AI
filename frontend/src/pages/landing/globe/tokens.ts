/**
 * The palette, as linear RGB for the shader.
 *
 * three.js wants linear light. These are derived at module load from the
 * same OKLCH values `theme.css` declares, through the same matrices the
 * contrast test uses, so a token change reaches the globe without anyone
 * remembering to. The strings below are the canonical token values; a test
 * asserts they match `theme.css`.
 */
import { oklchToLinear, parseOklch, type Rgb } from '@/styles/oklch'

export const TOKEN_SOURCE = {
  bgMain: 'oklch(0.1587 0.0081 48.16)',
  sand: 'oklch(0.772 0.0907 61.97)',
  terracotta: 'oklch(0.6188 0.1051 44.4)',
  textHi: 'oklch(0.9309 0.0144 57.58)',
  paper: 'oklch(0.975 0.01 80)',
  paperAccent: 'oklch(0.56 0.11 40)',
  paperText: 'oklch(0.22 0.02 50)',
} as const

function linear(value: string): Rgb {
  const parsed = parseOklch(value)
  if (!parsed) throw new Error(`bad token ${value}`)
  return oklchToLinear(parsed)
}

export const tokens: Record<keyof typeof TOKEN_SOURCE, Rgb> = {
  bgMain: linear(TOKEN_SOURCE.bgMain),
  sand: linear(TOKEN_SOURCE.sand),
  terracotta: linear(TOKEN_SOURCE.terracotta),
  textHi: linear(TOKEN_SOURCE.textHi),
  paper: linear(TOKEN_SOURCE.paper),
  paperAccent: linear(TOKEN_SOURCE.paperAccent),
  paperText: linear(TOKEN_SOURCE.paperText),
}
