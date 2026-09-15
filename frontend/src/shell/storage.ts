/**
 * `localStorage`, without assuming it works.
 *
 * It throws outright in a Safari private window and anywhere the user has
 * blocked site data, and several stores read it during module evaluation —
 * a throw there takes the app down before React mounts. Every persisted
 * preference goes through this so that failure means "not remembered", never
 * "not rendered".
 */
export const safeStorage = {
  getItem(key: string): string | null {
    try {
      return localStorage.getItem(key)
    } catch {
      return null
    }
  },
  setItem(key: string, value: string): void {
    try {
      localStorage.setItem(key, value)
    } catch {
      // A preference that cannot be persisted still applies to this session.
    }
  },
  removeItem(key: string): void {
    try {
      localStorage.removeItem(key)
    } catch {
      // Same.
    }
  },
}
