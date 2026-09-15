/**
 * Pairs with every lazy route. A chunk that 404s after a deploy — the stale
 * `index.html`, the new hashes — is the most common production-only crash in
 * an SPA, and the recovery is a reload, so the fallback offers exactly that.
 */
import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  title: string
  children: ReactNode
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  override state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(`[${this.props.title}]`, error, info.componentStack)
  }

  override render(): ReactNode {
    if (!this.state.error) return this.props.children
    const stale = /Failed to fetch dynamically imported module|Loading chunk|import\(\)/i.test(
      this.state.error.message,
    )
    return (
      <div role="alert" className="card mx-auto mt-10 max-w-md p-5">
        <p className="t-panel">{this.props.title} failed to load.</p>
        <p className="t-meta mt-1.5">
          {stale
            ? 'The app was updated while this tab was open. Reloading fetches the new version.'
            : this.state.error.message}
        </p>
        <button type="button" onClick={() => window.location.reload()} className="btn-primary mt-4">
          Reload
        </button>
      </div>
    )
  }
}
