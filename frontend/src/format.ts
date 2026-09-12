/**
 * Locale-aware formatting, in one place.
 *
 * Everything numeric in this UI passes through here rather than through
 * `String(n)`, `n.toFixed()` or a hand-rolled `n + 's'`. Three failures this
 * exists to prevent, all of which are reachable from a real server response:
 *
 *   - `toFixed` on a `null` throws, and a throw inside a render blanks the
 *     whole panel. API_CONTRACT §1 makes every unknown field `null`, so every
 *     one of these call sites is a live crash waiting for a CPU-only box or a
 *     PNG with no georeferencing.
 *   - `NaN` and `Infinity` reach the KPI cards as "NaN" — an assertion the
 *     server never made. They render as `—`, the same as any other unknown.
 *   - `${n} file${n === 1 ? '' : 's'}` is correct in English and wrong in most
 *     other languages. `Intl.PluralRules` knows the six categories.
 *
 * The locale is deliberately the browser's (`undefined` to every Intl
 * constructor). There is no translation layer yet, so the *copy* is English —
 * but the numbers, the separators and the plural selection are already the
 * reader's, which is the half that silently produces wrong output rather than
 * merely untranslated output.
 */

/** What every formatter here renders when the value is not a real number. */
export const ABSENT = '—'

const cache = new Map<string, Intl.NumberFormat>()

function formatter(precision: number, fixed = true): Intl.NumberFormat {
  const key = `${precision}:${fixed}`
  let found = cache.get(key)
  if (!found) {
    found = new Intl.NumberFormat(undefined, {
      minimumFractionDigits: fixed ? precision : 0,
      maximumFractionDigits: precision,
    })
    cache.set(key, found)
  }
  return found
}

/** True for a value that can actually be rendered as a number. */
export function isNumeric(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

/**
 * A number at a fixed precision, with the reader's thousands separator.
 * Anything that is not a finite number renders as {@link ABSENT}.
 */
export function decimal(value: unknown, precision = 2): string {
  return isNumeric(value) ? formatter(precision).format(value) : ABSENT
}

/** A whole number — counts, pixel dimensions, millisecond durations. */
export function integer(value: unknown): string {
  return decimal(value, 0)
}

/** A percentage that arrives already scaled 0–100. */
export function percent(value: unknown, precision = 1): string {
  return isNumeric(value) ? `${formatter(precision).format(value)}%` : ABSENT
}

/** A 0–1 ratio rendered as a percentage. */
export function ratioAsPercent(value: unknown, precision = 0): string {
  return isNumeric(value) ? `${formatter(precision).format(value * 100)}%` : ABSENT
}

/** Free precision, for a value whose natural number of decimals is unknown. */
const general = new Intl.NumberFormat(undefined, { maximumFractionDigits: 6 })

const plurals = new Intl.PluralRules(undefined)

/**
 * Pick the plural form for `count`.
 *
 * `other` is required and every other category optional, which is exactly the
 * shape English needs (`one` / `other`) while leaving room for a locale that
 * needs `few` or `many` without changing any call site.
 */
export function plural(
  count: number,
  forms: { one?: string; two?: string; few?: string; many?: string; zero?: string; other: string },
): string {
  if (!isNumeric(count)) return forms.other
  return forms[plurals.select(count)] ?? forms.other
}

/** `"2 files"` — the count and its noun, both localised. */
export function countOf(
  count: number,
  forms: { one?: string; two?: string; few?: string; many?: string; zero?: string; other: string },
): string {
  return `${integer(count)} ${plural(count, forms)}`
}

const BYTE_UNITS = ['B', 'kB', 'MB', 'GB', 'TB'] as const

/** A file size in the largest unit that keeps it above 1. */
export function bytes(value: unknown): string {
  if (!isNumeric(value) || value < 0) return ABSENT
  let size = value
  let unit = 0
  while (size >= 1024 && unit < BYTE_UNITS.length - 1) {
    size /= 1024
    unit += 1
  }
  // Not a fixed precision: a round limit should read "512 MB", not "512.0 MB".
  return `${formatter(unit === 0 ? 0 : 1, false).format(size)} ${BYTE_UNITS[unit]}`
}

/** A duration in milliseconds, promoted to seconds once it stops being legible. */
export function duration(ms: unknown): string {
  if (!isNumeric(ms) || ms < 0) return ABSENT
  if (ms < 1000) return `${integer(Math.round(ms))} ms`
  if (ms < 60_000) return `${decimal(ms / 1000, 1)} s`
  const minutes = Math.floor(ms / 60_000)
  return `${integer(minutes)} min ${integer(Math.round((ms % 60_000) / 1000))} s`
}

const timeFormat = new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' })
const stampFormat = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'medium',
})

/** Clock time for a session-local timestamp. */
export function timeOfDay(epochMs: unknown): string {
  return isNumeric(epochMs) ? timeFormat.format(new Date(epochMs)) : ABSENT
}

/** Date and time, for a run that may no longer be from today. */
export function timestamp(epochMs: unknown): string {
  return isNumeric(epochMs) ? stampFormat.format(new Date(epochMs)) : ABSENT
}

/**
 * Render an arbitrary `fact_sheet` value.
 *
 * The sheet is `Record<string, unknown>` on the wire and tools are free to put
 * a list or an object in it. `String(value)` turns those into `[object Object]`
 * — visible in the FactSheet tab of the one surface built to prove the run was
 * real, which is the worst possible place for it.
 */
export function scalar(value: unknown): string {
  if (value === null || value === undefined) return ABSENT
  if (typeof value === 'number') return Number.isFinite(value) ? general.format(value) : ABSENT
  if (typeof value === 'string' || typeof value === 'boolean') return String(value)
  try {
    return JSON.stringify(value)
  } catch {
    return ABSENT
  }
}

/** "Just now", "4 min ago", "2 hours ago", or the clock time past a day. */
export function relativeTime(epochMs: number, now = Date.now()): string {
  const seconds = Math.max(0, Math.round((now - epochMs) / 1000))
  if (seconds < 45) return 'Just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} min ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return hours === 1 ? '1 hour ago' : `${hours} hours ago`
  return timeOfDay(epochMs)
}
