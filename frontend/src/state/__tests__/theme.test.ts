// @vitest-environment happy-dom
import { beforeEach, describe, expect, it } from 'vitest'

import { applyTheme, useThemeStore } from '@/state/theme'

beforeEach(() => {
  document.documentElement.classList.remove('dark')
  useThemeStore.setState({ pref: 'dark', resolved: 'dark' })
})

describe('the theme', () => {
  it('puts .dark on <html> and flips color-scheme', () => {
    applyTheme('dark')
    expect(document.documentElement.classList.contains('dark')).toBe(true)
    expect(document.documentElement.style.colorScheme).toBe('dark')
    applyTheme('light')
    expect(document.documentElement.classList.contains('dark')).toBe(false)
    expect(document.documentElement.style.colorScheme).toBe('light')
  })

  it('toggles between dark and light and persists the choice', () => {
    useThemeStore.getState().toggle()
    expect(useThemeStore.getState().resolved).toBe('light')
    expect(document.documentElement.classList.contains('dark')).toBe(false)
    expect(localStorage.getItem('satquery.theme')).toBe('light')
    useThemeStore.getState().toggle()
    expect(useThemeStore.getState().resolved).toBe('dark')
  })
})
