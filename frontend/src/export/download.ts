/**
 * Hand a file to the browser. Shared by every exporter — GeoJSON, SITREP, the
 * device export — so there is exactly one place that knows the object-URL
 * dance and its revoke timing.
 */
export function downloadBlob(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.append(a)
  a.click()
  a.remove()
  // Revoking synchronously races the click in Safari; a second is plenty.
  setTimeout(() => URL.revokeObjectURL(url), 1_000)
}

export function downloadJson(name: string, data: unknown, mime = 'application/geo+json'): void {
  downloadBlob(name, new Blob([JSON.stringify(data, null, 2)], { type: mime }))
}

/** `20260915-1432` — local time, filename-safe, sorts chronologically. */
export function fileStamp(at = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${at.getFullYear()}${pad(at.getMonth() + 1)}${pad(at.getDate())}-${pad(at.getHours())}${pad(at.getMinutes())}`
}
