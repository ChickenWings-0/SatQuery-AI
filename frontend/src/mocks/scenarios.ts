/**
 * Mock scenarios.
 *
 * The canonical recording is a real run, and a real run with the VLM disabled
 * produces a *templated* answer — which is cited by construction, so its
 * `uncited_numeric_spans` is empty. That is honest, but it means the offline
 * demo never shows the anti-hallucination guard, which API_CONTRACT §8.5 calls
 * the proof the guard is live rather than claimed.
 *
 * So there is a second scenario. It takes the same recording and swaps in an
 * answer containing two numbers no tool measured, exactly as an ungrounded VLM
 * generation would. It is **synthetic and labelled as such** — reached only via
 * `?mock=1&scenario=ungrounded`, never the default — because a fabricated
 * recording passed off as a real one is the one thing this project cannot ship.
 */
import type { RecordedEvent } from '@/mocks/fixtures'

export const SCENARIOS = ['canonical', 'ungrounded', 'grounded'] as const
export type Scenario = (typeof SCENARIOS)[number]

export function activeScenario(): Scenario {
  if (typeof window === 'undefined') return 'canonical'
  const requested = new URLSearchParams(window.location.search).get('scenario')
  return (SCENARIOS as readonly string[]).includes(requested ?? '')
    ? (requested as Scenario)
    : 'canonical'
}

/** The synthetic ungrounded answer, replacing the recorded templated one. */
const UNGROUNDED_ANSWER = {
  text:
    'Approximately 5.34% of the analysed scene (0.35 km2) changed between the two ' +
    'acquisitions, concentrated in 1 distinct region covering 350,106.12 m2. The new ' +
    'construction appears to be residential housing built over roughly 12 hectares, and ' +
    'the surrounding vegetation declined by about 8.5% over the same period.',
  citations: [
    { claim: '5.34%', source: 'step:3/scalars.changed_area_pct', value: 5.3422 },
    { claim: '0.35 km2', source: 'step:3/scalars.changed_area_km2', value: 0.350106 },
    { claim: '1', source: 'step:3/scalars.component_count', value: 1.0 },
    { claim: '350,106.12 m2', source: 'step:3/scalars.changed_area_m2', value: 350106.12 },
  ],
  // Neither number resolves to any measurement: no tool emitted a hectare
  // figure, and nothing measured a vegetation decline.
  uncited_numeric_spans: ['12 hectares', '8.5%'],
  generator: 'vlm_change_vqa@1.0.0+adapter:sq-lora-v3 (SYNTHETIC FIXTURE)',
  template_fallback: false,
}

/**
 * The `grounded` scenario: a run that produced boxes, on a 2:1 image.
 *
 * Exists for the browser test of the overlay geometry — the one visual code
 * path nothing else exercises in a real layout engine. The pre-change true
 * colour view is swapped for a 896x448 render so `object-fit: contain` has to
 * letterbox, and a `BBOX_SET` artifact carries one box at a known position.
 */
export const GROUNDED_BOX = { xMin: 250, yMin: 250, xMax: 750, yMax: 750 } as const

const GROUNDED_BBOX_SET = {
  id: 'art_99',
  type: 'BBOX_SET',
  mime: 'application/json',
  label: 'Grounded boxes',
  url: null,
  geotiff_url: null,
  geojson_url: null,
  geo: null,
  width: null,
  height: null,
  stats: null,
  inline: {
    boxes: [
      {
        id: 'box_0',
        label: 'compound',
        score: 0.91,
        bbox_px: [224, 112, 672, 336],
        bbox_normalised: [GROUNDED_BOX.xMin, GROUNDED_BOX.yMin, GROUNDED_BOX.xMax, GROUNDED_BOX.yMax],
        bbox_wgs84: null,
      },
    ],
    frame: { width: 896, height: 448 },
    source_image: 'img_0',
  },
  produced_by_step: 5,
}

function groundedEvents(events: RecordedEvent[]): RecordedEvent[] {
  const out: RecordedEvent[] = []
  for (const event of events) {
    if (event.event === 'artifact') {
      const artifact = event.data as { id: string; url?: string | null; width?: number | null }
      if (artifact.id === 'art_0') {
        out.push({
          event: 'artifact',
          data: { ...artifact, url: '/v1/artifacts/grounded/wide.jpg', width: 896 },
        })
        continue
      }
    }
    if (event.event === 'done') {
      const done = event.data as { artifacts?: unknown[] }
      out.push({ event: 'artifact', data: GROUNDED_BBOX_SET })
      out.push({
        event: 'done',
        data: {
          ...done,
          artifacts: [
            ...(done.artifacts ?? []).map((artifact) => {
              const ref = artifact as { id: string }
              return ref.id === 'art_0'
                ? { ...ref, url: '/v1/artifacts/grounded/wide.jpg', width: 896 }
                : artifact
            }),
            GROUNDED_BBOX_SET,
          ],
        },
      })
      continue
    }
    out.push(event)
  }
  return out
}

/** Rewrite the recording's terminal payload for the requested scenario. */
export function eventsForScenario(
  events: RecordedEvent[],
  scenario: Scenario,
): RecordedEvent[] {
  if (scenario === 'canonical') return events
  if (scenario === 'grounded') return groundedEvents(events)

  return events.map((event) => {
    if (event.event !== 'done') return event
    const done = event.data as { answer: unknown; trace?: { answer?: unknown } | null }
    return {
      event: 'done',
      data: {
        ...done,
        answer: UNGROUNDED_ANSWER,
        // Keep the trace's copy consistent; the modal reads from it.
        ...(done.trace ? { trace: { ...done.trace, answer: UNGROUNDED_ANSWER } } : {}),
      },
    }
  })
}
