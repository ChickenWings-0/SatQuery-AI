/**
 * Every heavy dependency has exactly one owner that imports it statically,
 * and that owner is reached only through `React.lazy`. A `grep` over the
 * source is the cheapest bundle guard there is, and it runs on every commit.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

const SRC = new URL('..', import.meta.url).pathname

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name)
    if (name === '__tests__' || name === 'node_modules') continue
    if (statSync(full).isDirectory()) walk(full, out)
    else if (/\.(ts|tsx)$/.test(name) && !name.endsWith('.d.ts')) out.push(full)
  }
  return out
}

const files = walk(SRC)
const rel = (file: string) => file.slice(SRC.length)

const OWNERS: Record<string, RegExp> = {
  three: /^pages\/landing\/globe\//,
  '@react-three/': /^pages\/landing\/globe\//,
  'maplibre-gl': /^pages\/maps\/MapStage\.tsx$/,
  '@xyflow/react': /^components\/pipeline\/DagCanvas\.tsx$/,
  'idb-keyval': /^state\/library\.ts$/,
  'pdf-lib': /^export\/sitrep\/render\.ts$/,
  '@pdf-lib/fontkit': /^export\/sitrep\/render\.ts$/,
}

describe('bundle boundaries', () => {
  it.each(Object.entries(OWNERS))('%s is imported statically only by its owner', (dep, owner) => {
    const importers = files.filter((file) => {
      const source = readFileSync(file, 'utf8')
      return new RegExp(`^import[^\\n]*from ['"]${dep.replace('/', '\\/')}`, 'm').test(source)
    })
    for (const file of importers) {
      expect(rel(file), `${rel(file)} imports ${dep}`).toMatch(owner)
    }
  })

  it('reaches every lazy owner only through import()', () => {
    for (const [name, pattern] of [
      ['Landing', /from '@\/pages\/Landing'/],
      ['MapStage', /from '@\/pages\/maps\/MapStage'/],
      ['Globe', /from '@\/pages\/landing\/globe\/Globe'/],
      // `import type` is erased by the compiler and pulls nothing in.
      ['library (outside its pages)', /^import(?!\s+type)[^\n]*from '@\/state\/library'/m],
      // pdf-lib rides with the SITREP builder, which loads on the click.
      ['SITREP builder', /^import[^\n]*from '@\/export\/sitrep\/build'/m],
      // The GeoJSON planner (stores + builder) loads on the click too.
      ['GeoJSON planner', /^import(?!\s+type)[^\n]*from '@\/export\/geojson\/plan'/m],
      // The mask vectoriser is its own chunk behind the "Include masks" switch.
      ['mask vectoriser', /^import(?!\s+type)[^\n]*from '@\/export\/geojson\/masks'/m],
    ] as const) {
      const importers = files.filter((file) => pattern.test(readFileSync(file, 'utf8')))
      // `SettingsDialog` (`App.tsx`) and `SidebarHistory` (`Sidebar.tsx`) are
      // themselves lazy chunks, so their static import of the library never
      // reaches the console entry.
      const allowed =
        name === 'library (outside its pages)'
          ? importers.filter(
              (f) =>
                !/^(pages\/(saved|projects|Saved|Projects|Report)|components\/shell\/(AccountPopover|SettingsDialog|SidebarHistory))/.test(rel(f)),
            )
          : importers
      expect(allowed.map(rel), `${name} is statically imported`).toEqual([])
    }
  })

  it('keeps the SITREP renderer behind its builder', () => {
    // `render.ts` is where pdf-lib lives; only the lazily loaded builder may
    // import it, or the click stops being the moment the chunk loads.
    const importers = files.filter((file) => /^import[^\n]*from '@\/export\/sitrep\/render'/m.test(readFileSync(file, 'utf8')))
    expect(importers.map(rel)).toEqual(['export/sitrep/build.ts'])
  })

  it('keeps the geo clients free of React, stores and the map', () => {
    // `geo/*` are plain fetch wrappers: MSW must see every call, vitest must
    // replay them without a DOM, and nothing in them may drag the Maps chunk
    // or a store into a place that only wanted a search.
    const geo = files.filter((file) => /^geo\//.test(rel(file)))
    expect(geo.length).toBeGreaterThan(0)
    for (const file of geo) {
      const source = readFileSync(file, 'utf8')
      expect(source, `${rel(file)} imports react`).not.toMatch(/from 'react'/)
      expect(source, `${rel(file)} imports a store`).not.toMatch(/from '@\/state\//)
      expect(source, `${rel(file)} imports maplibre`).not.toMatch(/from 'maplibre-gl'/)
    }
  })

  it('keeps the STAC store inside the Maps chunk', () => {
    const importers = files.filter((file) => /from '@\/state\/stac'/.test(readFileSync(file, 'utf8')))
    for (const file of importers) expect(rel(file)).toMatch(/^(pages\/maps\/|pages\/Maps\.tsx$|mocks\/)/)
  })

  it('keeps the display scale on the landing page', () => {
    // `.t-display` and `.t-section` are the landing page's voice; a display
    // size on a KPI card is the fastest way to make the console look assembled.
    const users = files.filter((file) => /\bt-display\b|\bt-section\b|--font-display/.test(readFileSync(file, 'utf8')))
    for (const file of users) expect(rel(file)).toMatch(/^pages\/landing\//)
  })
})
