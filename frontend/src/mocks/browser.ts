/** Starts MSW in the browser. Imported lazily so it never enters the prod bundle. */
import { setupWorker } from 'msw/browser'

import { handlers } from '@/mocks/handlers'

export const worker = setupWorker(...handlers)

export { mockRequested } from '@/mocks/mode'
