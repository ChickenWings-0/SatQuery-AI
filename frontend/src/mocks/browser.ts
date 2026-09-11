/** Starts MSW in the browser. Imported lazily so it never enters the prod bundle. */
import { setupWorker } from 'msw/browser'

import { handlers } from '@/mocks/handlers'

export const worker = setupWorker(...handlers)

/** True when the app was opened with `?mock=1`. */
export function mockRequested(): boolean {
  return new URLSearchParams(window.location.search).get('mock') === '1'
}
