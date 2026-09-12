/**
 * Runs before every test file, after the environment is created.
 *
 * `react-compare-slider` registers a CSS custom property in a module-level
 * IIFE and `console.debug`s when `CSS.registerProperty` is missing — which it
 * is under happy-dom and jsdom. That line landed in the output of every test
 * file that mounts `ImageViewer`, and noise like that hides real signal. The
 * stub gives the library what it probes for; nothing in the suite depends on
 * the property actually being registered.
 *
 * Patched on the *prototype*: happy-dom's `window.CSS` getter returns a fresh
 * instance on every access, so a property set on one instance is gone by the
 * time the library reads it. Only stubbed when absent, so a future DOM
 * implementation that ships the real thing wins.
 */
type CssLike = { registerProperty?: (definition: unknown) => void; supports?: () => boolean }

const css = (globalThis as unknown as { CSS?: CssLike }).CSS
if (css) {
  const target = (Object.getPrototypeOf(css) as CssLike | null) ?? css
  target.registerProperty ??= () => undefined
  target.supports ??= () => false
}
