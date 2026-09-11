/**
 * The palette's contrast contract, enforced.
 *
 * `theme.css` has always *stated* its rules — "rust is for borders, icons and
 * large text; orange is a FILL behind dark text, never orange text on cream".
 * An audit then found eight components doing exactly that, at 1.99:1. Writing a
 * rule in a comment does not enforce it, and no reviewer can eyeball the
 * difference between 4.6:1 and 2.6:1 on a dark ground.
 *
 * So the rule lives here instead. Every pair below is a pair the app actually
 * renders; the tokens are parsed out of `theme.css` at run time, so changing a
 * hex there and breaking a pair fails this test rather than shipping.
 *
 * The maths is WCAG 2.x relative luminance, implemented rather than imported:
 * it is nine lines, and a dependency that computes it would be a dependency
 * that has to be trusted about the one thing this file exists to be sure of.
 */
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const css = readFileSync(new URL('../theme.css', import.meta.url), 'utf8')

/** Every `--color-*` custom property declared in the theme. */
const TOKENS: Record<string, string> = Object.fromEntries(
  [...css.matchAll(/--color-([a-z0-9-]+):\s*(#[0-9a-fA-F]{3,6})\b/g)].map((m) => [m[1]!, m[2]!]),
)

function channels(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  const full = h.length === 3 ? [...h].map((c) => c + c).join('') : h
  return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16) / 255) as [number, number, number]
}

function luminance(hex: string): number {
  const [r, g, b] = channels(hex).map((c) =>
    c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4,
  ) as [number, number, number]
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

function contrast(fg: string, bg: string): number {
  const [a, b] = [luminance(fg), luminance(bg)]
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05)
}

/**
 * Composite a translucent fill over an opaque ground.
 *
 * Tailwind's `bg-ok/12` is not a colour, it is a colour *and* a ground, and the
 * result is what the text actually sits on. Checking the token against white
 * instead — which is the intuitive thing to do — is precisely how the status
 * chips shipped at 2.72:1.
 */
function over(fg: string, bg: string, alpha: number): string {
  const [f, b] = [channels(fg), channels(bg)]
  const mix = f.map((c, i) => c * alpha + b[i]! * (1 - alpha))
  return `#${mix.map((c) => Math.round(c * 255).toString(16).padStart(2, '0')).join('')}`
}

const t = (name: string): string => {
  const value = TOKENS[name]
  if (!value) throw new Error(`--color-${name} is not declared in theme.css`)
  return value
}

/** WCAG 1.4.3 body text. */
const BODY = 4.5
/** WCAG 1.4.11 non-text: status dots, control borders, focus rings, meters. */
const MARK = 3

describe('the palette declares every role it uses', () => {
  it.each([
    'bg-main', 'surface-card', 'sidebar', 'sidebar-hi', 'sidebar-text', 'sidebar-text-lo',
    'sidebar-ok', 'sidebar-warn', 'text-hi', 'text-lo', 'accent-warm', 'accent-warm-text',
    'accent-warm-strong', 'accent-cool', 'evidence', 'on-evidence', 'on-accent-cool',
    'ok', 'ok-text', 'warn', 'warn-text', 'fail', 'skip', 'skip-text', 'line',
  ])('--color-%s', (name) => {
    expect(t(name)).toMatch(/^#[0-9a-fA-F]{6}$/)
  })

  it('has no --color-fail-text, because --color-fail never needed one', () => {
    // Symmetry is not a reason to add a token. Guarded so that if `--color-fail`
    // is ever darkened or lightened, this line is where someone re-checks.
    expect(TOKENS['fail-text']).toBeUndefined()
    expect(contrast(t('fail'), t('bg-main'))).toBeGreaterThanOrEqual(BODY)
  })
})

describe('text on the application ground', () => {
  const grounds = () => [
    ['ground', t('bg-main')],
    ['card', t('surface-card')],
  ]

  it.each(['text-hi', 'text-lo', 'ok-text', 'warn-text', 'skip-text', 'fail', 'accent-warm-text'])(
    '%s reads at 4.5:1 on both surfaces',
    (name) => {
      for (const [label, ground] of grounds()) {
        expect(contrast(t(name), ground!), `${name} on ${label}`).toBeGreaterThanOrEqual(BODY)
      }
    },
  )

  it('keeps every -text token at body contrast even where it equals its fill', () => {
    // On the dark ground the saturated status fills happen to clear 4.5:1 as
    // text too, so `ok-text` and `warn-text` currently equal `ok` and `warn`.
    // The -text tokens stay because components reach for them by role: if a
    // fill is ever re-tuned to something that only clears the 3:1 mark bar,
    // this is the assertion that catches the text going with it.
    for (const [text, fill] of [
      ['ok-text', 'ok'],
      ['warn-text', 'warn'],
      ['skip-text', 'skip'],
    ] as const) {
      expect(contrast(t(text), t('bg-main')), `${text}`).toBeGreaterThanOrEqual(BODY)
      expect(contrast(t(fill), t('bg-main')), `${fill} as a mark`).toBeGreaterThanOrEqual(MARK)
    }
  })

  it('carries the accent as sand at text weight, never the button fill', () => {
    // `accent-warm-text` is the small-text voice of the accent. It must be
    // brighter than the fill it accompanies, or a label and a button read as
    // the same hue at two contrasts.
    expect(contrast(t('accent-warm-text'), t('bg-main'))).toBeGreaterThan(
      contrast(t('accent-warm'), t('bg-main')),
    )
  })
})

describe('a chip puts its hue on a wash of itself', () => {
  it.each([
    ['ok-text', 'ok', 0.12],
    ['skip-text', 'skip', 0.15],
    ['warn-text', 'warn', 0.2],
    ['warn-text', 'warn', 0.08],
    ['fail', 'fail', 0.08],
  ])('%s on bg-%s/%d', (text, fill, alpha) => {
    const ground = over(t(fill), t('surface-card'), alpha)
    expect(contrast(t(text), ground)).toBeGreaterThanOrEqual(BODY)
  })
})

describe('the navigation column', () => {
  it('carries secondary text at 4.5:1 on the column AND its hover state', () => {
    // Both grounds, because a row that passes until the pointer touches it is
    // not a row that passes. This is what nine `opacity-*` modifiers hid.
    for (const ground of [t('sidebar'), t('sidebar-hi')]) {
      expect(contrast(t('sidebar-text-lo'), ground)).toBeGreaterThanOrEqual(BODY)
      expect(contrast(t('sidebar-text'), ground)).toBeGreaterThanOrEqual(BODY)
    }
  })

  it('keeps the status hues legible on the column and its hover state', () => {
    // The column shares the card surface now, so the sidebar status tokens are
    // the base hues — pinned here so a future re-tune of the column ground
    // cannot quietly take the health badge with it.
    for (const ground of [t('sidebar'), t('sidebar-hi')]) {
      expect(contrast(t('sidebar-ok'), ground)).toBeGreaterThanOrEqual(BODY)
      expect(contrast(t('sidebar-warn'), ground)).toBeGreaterThanOrEqual(BODY)
    }
  })

  it('shows a focus ring the keyboard user can see', () => {
    // WCAG 1.4.11: both the global ring and the column's own ring clear 3:1
    // on the column. The column keeps its brighter ring because it sits beside
    // the 3px terracotta active indicator and must not read as part of it.
    expect(contrast(t('accent-warm'), t('sidebar'))).toBeGreaterThanOrEqual(MARK)
    expect(contrast(t('sidebar-text'), t('sidebar'))).toBeGreaterThanOrEqual(MARK)
  })
})

describe('fills that carry light text', () => {
  it('keeps white legible on the primary button', () => {
    expect(contrast('#ffffff', t('accent-warm-strong'))).toBeGreaterThanOrEqual(BODY)
  })

  it('keeps the citation pill legible, including on hover', () => {
    expect(contrast(t('on-evidence'), t('evidence'))).toBeGreaterThanOrEqual(BODY)
    const hovered = over(t('evidence'), t('surface-card'), 0.8)
    expect(contrast(t('on-evidence'), hovered)).toBeGreaterThanOrEqual(BODY)
  })

  it('keeps the selected navigation row legible', () => {
    expect(contrast(t('on-accent-cool'), t('accent-cool'))).toBeGreaterThanOrEqual(BODY)
  })
})

describe('non-text marks', () => {
  it.each(['ok', 'warn', 'fail', 'skip', 'accent-warm'])(
    'the %s status dot is visible on both surfaces',
    (name) => {
      for (const ground of [t('bg-main'), t('surface-card')]) {
        expect(contrast(t(name), ground)).toBeGreaterThanOrEqual(MARK)
      }
    },
  )

  it('draws the ungrounded-number underline at mark contrast', () => {
    // §8.5's honesty signal is a wavy underline and nothing else visual, so it
    // has to clear 3:1 or the guard is invisible rather than live.
    expect(contrast(t('warn'), t('surface-card'))).toBeGreaterThanOrEqual(MARK)
  })

  it('draws control edges and meters at mark contrast', () => {
    expect(contrast(t('accent-warm-text'), t('bg-main'))).toBeGreaterThanOrEqual(MARK)
    expect(contrast(t('accent-warm-strong'), t('line'))).toBeGreaterThanOrEqual(MARK)
    expect(contrast(t('accent-warm'), t('bg-main'))).toBeGreaterThanOrEqual(MARK)
  })
})
