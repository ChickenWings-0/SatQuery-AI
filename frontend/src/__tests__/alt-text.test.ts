/**
 * Every `<img` in the source carries an `alt`. Decorative images say so with
 * `alt=""` *and* `aria-hidden`; everything else names what it shows.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

const SRC = new URL('..', import.meta.url).pathname

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name)
    if (name === '__tests__') continue
    if (statSync(full).isDirectory()) walk(full, out)
    else if (name.endsWith('.tsx')) out.push(full)
  }
  return out
}

describe('alt text', () => {
  it('is present on every <img>', () => {
    for (const file of walk(SRC)) {
      const source = readFileSync(file, 'utf8')
      for (const match of source.matchAll(/<img\b([\s\S]*?)\/?>/g)) {
        const attrs = match[1]!
        expect(attrs, `${file.slice(SRC.length)}: ${attrs.trim().slice(0, 60)}`).toMatch(/\balt=/)
        if (/\balt=""/.test(attrs)) expect(attrs, `${file.slice(SRC.length)}: decorative img needs aria-hidden`).toMatch(/aria-hidden/)
        expect(attrs, `${file.slice(SRC.length)}: alt must not be a bare empty fallback`).not.toMatch(/alt=\{[^}]*\?\? ''\}/)
      }
    }
  })
})
