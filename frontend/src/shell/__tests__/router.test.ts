import { describe, expect, it } from 'vitest'

import { normalisePath, pathFor, sectionFromPath } from '@/shell/router'
import { NAV_SECTIONS } from '@/state/ui'

describe('the router', () => {
  it('round-trips every section', () => {
    for (const section of [...NAV_SECTIONS, 'history'] as const) {
      expect(sectionFromPath(pathFor(section))).toBe(section)
    }
  })

  it('sends an unknown path to the 404 page, not to the workspace', () => {
    expect(sectionFromPath('/nope')).toBe('notFound')
    expect(sectionFromPath('/explore/extra')).toBe('notFound')
  })

  it('normalises a trailing slash and ignores the query', () => {
    expect(normalisePath('/use-cases/')).toBe('/use-cases')
    expect(normalisePath('/')).toBe('/')
    expect(sectionFromPath('/maps?mock=1')).toBe('maps')
  })

  it('keeps a project or report deep link inside its section', () => {
    expect(sectionFromPath('/projects/p-abc')).toBe('projects')
    expect(sectionFromPath('/report/t-abc')).toBe('saved')
  })
})
