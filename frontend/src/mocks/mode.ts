/**
 * Is the app running on recorded fixtures? Kept apart from `mocks/browser`
 * so a component can ask without pulling MSW into the production bundle.
 */
export function mockRequested(): boolean {
  if (typeof window === 'undefined') return false
  return new URLSearchParams(window.location.search).get('mock') === '1'
}
