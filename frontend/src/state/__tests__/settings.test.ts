/**
 * Custom instructions ride inside `query`, which the server caps at 1000
 * characters. These pin the arithmetic that keeps the composer honest about
 * what is left, and the shape of what is actually sent.
 */
import { beforeEach, describe, expect, it } from 'vitest'

import {
  composeQuery,
  INSTRUCTIONS_MAX,
  QUERY_MAX,
  queryRoom,
  useSettingsStore,
} from '@/state/settings'

describe('custom instructions', () => {
  beforeEach(() => useSettingsStore.getState().reset())

  it('leave the whole budget to the question when empty', () => {
    expect(queryRoom('')).toBe(QUERY_MAX)
    expect(queryRoom('   ')).toBe(QUERY_MAX)
    expect(composeQuery('How much water?', '  ')).toBe('How much water?')
  })

  it('cost their length plus the blank line between', () => {
    expect(queryRoom('Answer in hectares.')).toBe(QUERY_MAX - 'Answer in hectares.'.length - 2)
    expect(composeQuery('How much water?', ' Answer in hectares. ')).toBe(
      'Answer in hectares.\n\nHow much water?',
    )
  })

  it('never leave less than a usable question', () => {
    const longest = 'x'.repeat(INSTRUCTIONS_MAX)
    useSettingsStore.getState().setCustomInstructions(longest + 'overflow')
    expect(useSettingsStore.getState().customInstructions).toHaveLength(INSTRUCTIONS_MAX)
    expect(queryRoom(longest)).toBeGreaterThanOrEqual(QUERY_MAX - INSTRUCTIONS_MAX - 2)
  })

  it('keep the seed a non-negative integer, 0 meaning server default', () => {
    const { setSeed } = useSettingsStore.getState()
    setSeed(42)
    expect(useSettingsStore.getState().seed).toBe(42)
    setSeed(-3)
    expect(useSettingsStore.getState().seed).toBe(0)
    setSeed(Number.NaN)
    expect(useSettingsStore.getState().seed).toBe(0)
  })
})

describe('the settings dialog', () => {
  it('opens on a named pane and closes', () => {
    const store = useSettingsStore.getState()
    store.openSettings('data')
    expect(useSettingsStore.getState()).toMatchObject({ open: true, tab: 'data' })
    store.closeSettings()
    expect(useSettingsStore.getState().open).toBe(false)
    // Reopening without a pane keeps the last one.
    store.openSettings()
    expect(useSettingsStore.getState().tab).toBe('data')
  })
})
