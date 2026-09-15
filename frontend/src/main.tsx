import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import App from '@/App'
import { bindRouter } from '@/shell/router'
import { bindTheme } from '@/state/theme'
import '@/styles/theme.css'

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, refetchOnWindowFocus: false } },
})

function render() {
  bindTheme()
  bindRouter()
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </StrictMode>,
  )
}

/**
 * `?mock=1` starts MSW before the first render, so the whole UI runs with the
 * backend down. The import is dynamic: MSW and the recorded fixtures must never
 * reach the production bundle.
 */
async function main() {
  const { mockRequested } = await import('@/mocks/browser')
  if (mockRequested()) {
    const { worker } = await import('@/mocks/browser')
    await worker.start({ onUnhandledRequest: 'bypass', quiet: true })
  }
  render()
}

void main()
