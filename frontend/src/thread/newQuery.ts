/**
 * "New Query" from the rail, as an event.
 *
 * The rail cannot reach the run hook's `AbortController` — it lives inside
 * `ThreadPanel` — and resetting the job store under a stream that keeps
 * applying events would only be undone by the next chunk. So the rail
 * announces, and `useRun` abandons the run on both ends before the reset.
 */
export const NEW_QUERY_EVENT = 'satquery:new-query'
