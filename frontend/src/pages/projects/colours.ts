import type { ProjectColour } from '@/state/library'

/** A project's tag hue, from the status vocabulary only. */
export const COLOUR: Record<ProjectColour, string> = {
  terracotta: 'bg-accent-warm',
  sand: 'bg-surface-sand',
  ok: 'bg-ok',
  warn: 'bg-warn',
}
