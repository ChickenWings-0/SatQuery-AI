/** Confidence as a colour: the same thresholds the answer card uses. */
export function confidenceTone(value: number | null): string {
  if (value === null) return 'text-text-lo'
  if (value >= 0.8) return 'text-ok-text'
  if (value >= 0.6) return 'text-warn-text'
  return 'text-fail'
}
