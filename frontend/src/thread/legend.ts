/**
 * The change overlay's legend text.
 *
 * It read "New built-up area" on every change task regardless of what the
 * question asked about. The classifier resolves the question's class onto the
 * controlled vocabulary (`configs/class_vocabulary.yaml`) and records it in
 * `resolved_task.slots.target_class`; the legend follows that, and falls back
 * to a neutral label when no class was named — never to a specific one.
 */
import type { ResolvedTask } from '@/api/types'

/** Canonical vocabulary keys → how the legend names them. */
const CLASS_LABELS: Record<string, string> = {
  built_up: 'built-up area',
  water: 'water',
  vegetation: 'vegetation',
  bare_soil: 'bare soil',
  road: 'roads',
  aircraft: 'aircraft',
  ship: 'ships',
  vehicle: 'vehicles',
  airport: 'airport area',
  cloud: 'cloud',
}

export const DEFAULT_CHANGE_LEGEND = 'Detected change'

/** A vocabulary key, or a term kept verbatim, as legend prose. */
export function humaniseClass(raw: string): string {
  return CLASS_LABELS[raw] ?? raw.replace(/_/g, ' ').trim()
}

export function changeLegend(task: ResolvedTask | null | undefined): string {
  const slots = (task?.slots ?? {}) as Record<string, unknown>
  const target = slots['target_class']
  if (typeof target !== 'string' || target.trim() === '') return DEFAULT_CHANGE_LEGEND
  return `New ${humaniseClass(target)}`
}
