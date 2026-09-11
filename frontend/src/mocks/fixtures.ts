/**
 * Offline fixtures: real responses, recorded from a real run.
 *
 * These JSON files were captured from a live FastAPI against the synthetic
 * `s2_pre` / `s2_post` corpus — the canonical bi-temporal scenario of
 * API_CONTRACT §7.4 — rather than written by hand. A hand-written fixture
 * drifts from the frozen schema the moment anything is added within 1.0; a
 * recording cannot, because it *is* what the server said.
 *
 * The trace id is normalised to {@link MOCK_TRACE_ID} so artifact URLs are
 * stable, and the 14 primary renders live in `public/mock-artifacts/` so the
 * evidence gallery shows genuine imagery with the GPU off.
 *
 * To re-record: run the API against the synthetic corpus, capture
 * `/v1/health`, `/v1/registry`, `/v1/validate` and the SSE stream, then
 * substitute the trace id.
 */
import events from '@/mocks/captured/events.bitemporal.json'
import health from '@/mocks/captured/health.json'
import registry from '@/mocks/captured/registry.json'
import validate from '@/mocks/captured/validate.bitemporal.json'
import validateSingle from '@/mocks/captured/validate.single.json'

import type {
  HealthResponse,
  RegistryResponse,
  ValidateResponse,
} from '@/api/types'

export const MOCK_TRACE_ID = '0f1e2d3c4b5a69788796a5b4c3d2e1f0'

export interface RecordedEvent {
  event: string
  data: unknown
}

export const healthFixture = health as unknown as HealthResponse
export const registryFixture = registry as unknown as RegistryResponse
export const validateFixture = validate as unknown as ValidateResponse

/**
 * The same pre-flight with one image. Recorded separately rather than derived,
 * because a single-image run is not a bi-temporal one with a manifest removed:
 * six of the ten checks become `SKIP`, the pair type is `SINGLE`, and the
 * supported task list is a different set entirely.
 */
export const validateSingleFixture = validateSingle as unknown as ValidateResponse
export const eventsFixture = events as unknown as RecordedEvent[]
