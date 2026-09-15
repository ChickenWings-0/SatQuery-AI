/**
 * The reduced-motion contract.
 *
 * `prefers-reduced-motion` is easy to satisfy destructively — one
 * `* { animation: 0.01ms !important }` and every audit tool goes quiet. In this
 * product that would delete the running-state signal, which is the one thing
 * the interface exists to show. These assertions pin the *intentional*
 * alternative: less movement, same meaning.
 *
 * Read from `theme.css` for the same reason the contrast suite is: a rule that
 * only lives in a comment is a rule that gets edited away.
 */
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const css = readFileSync(new URL('../theme.css', import.meta.url), 'utf8')

/** The body of the `prefers-reduced-motion: reduce` block, brace-matched. */
/** Every `prefers-reduced-motion` block in the file, concatenated. */
function reducedMotionBlock(): string {
  const needle = '@media (prefers-reduced-motion: reduce)'
  const blocks: string[] = []
  let start = css.indexOf(needle)
  expect(start, 'theme.css declares a reduced-motion block').toBeGreaterThan(-1)
  while (start !== -1) {
    let depth = 0
    let end = -1
    for (let i = css.indexOf('{', start); i < css.length; i++) {
      if (css[i] === '{') depth++
      else if (css[i] === '}' && --depth === 0) {
        end = i + 1
        break
      }
    }
    if (end === -1) throw new Error('unbalanced braces in the reduced-motion block')
    blocks.push(css.slice(start, end))
    start = css.indexOf(needle, end)
  }
  return blocks.join('\n')
}

describe('prefers-reduced-motion', () => {
  const block = reducedMotionBlock()

  it('does not blanket-kill every animation', () => {
    // The `*`-selector recipe. It passes checkers and loses the product.
    expect(block).not.toMatch(/^\s*\*[\s,{]/m)
    expect(block).not.toMatch(/animation:\s*[^;]*0\.01m?s/)
  })

  it('stills the running pulse instead of hiding the element', () => {
    expect(block).toMatch(/\[data-status='RUNNING'\]/)
    expect(block).toMatch(/animation:\s*none/)
    // Forced opaque: a pulse frozen mid-cycle at 0.45 opacity would read as
    // disabled rather than running.
    expect(block).toMatch(/opacity:\s*1/)
  })

  it('keeps colour and state transitions', () => {
    // Feedback that confirms an action must survive. Killing `transition`
    // wholesale makes every control feel broken rather than calm.
    expect(block).not.toMatch(/transition[^;]*:\s*none/)
    expect(block).not.toMatch(/transition-duration/)
  })

  it('swaps the dialog entrance for a fade rather than removing it', () => {
    expect(block).toMatch(/\[data-sq-dialog\]\[data-state='open'\]/)
    expect(block).toMatch(/animation-name:\s*sq-overlay-in/)
  })

  it('stops React Flow’s edge dash, which ships as its own animation', () => {
    expect(block).toMatch(/\.react-flow__edge-path/)
  })
})

describe('the motion budget', () => {
  it('loops only the animations on the budget, each reporting a real ongoing state', () => {
    // Anything `infinite` is running forever on someone's battery. Each loop
    // here is gated: the RUNNING dot stops when the step does, the health
    // breath reports a poll that persists, the shimmer ends with the fetch,
    // and the landing/empty-state loops (`data-breathe`, `data-sweep`, the
    // pipeline beam's fallback loop on browsers without scroll timelines)
    // are paused by their owners when off-screen or closed. A seventh is a
    // design review, not a merge.
    const infinite = [...css.matchAll(/animation:[^;]*infinite[^;]*;/g)].map((m) => m[0])
    const names = ['sq-status-pulse', 'sq-glow', 'sq-shimmer', 'sq-breathe', 'sq-sweep', 'sq-beam-loop']
    expect(infinite).toHaveLength(names.length)
    for (const name of names) {
      expect(infinite.some((rule) => rule.includes(name)), name).toBe(true)
    }
  })

  it('stills every loop under reduced motion', () => {
    const block = reducedMotionBlock()
    expect(block).toMatch(/\[data-breathe\],\s*\[data-sweep\],\s*\[data-badge-new\]\s*\{[^}]*animation:\s*none/)
    expect(block).toMatch(/\[data-hero='glow'\]\s*\{[^}]*animation:\s*none/)
    expect(block).toMatch(/\.skeleton\s*\{[^}]*animation:\s*none/)
  })

  it('stills the health glow under reduced motion', () => {
    const block = reducedMotionBlock()
    expect(block).toMatch(/\.sq-glow\s*\{[^}]*animation:\s*none/)
  })

  it('animates only compositable properties', () => {
    // No `width`, `height`, `top`, `left` or `margin` in a keyframe: this UI
    // runs beside a saturated GPU and may not touch layout to draw itself.
    const keyframes = [...css.matchAll(/@keyframes\s+sq-[\w-]+\s*\{[\s\S]*?\n {2}\}/g)].map(
      (m) => m[0],
    )
    expect(keyframes.length).toBeGreaterThanOrEqual(4)
    for (const frame of keyframes) {
      expect(frame, frame.slice(0, 40)).not.toMatch(
        /^\s*(width|height|top|left|right|bottom|margin|padding)\s*:/m,
      )
    }
  })

  it('uses deceleration rather than bounce for arrivals', () => {
    expect(css).toContain('cubic-bezier(0.16, 1, 0.3, 1)')
  })
})
