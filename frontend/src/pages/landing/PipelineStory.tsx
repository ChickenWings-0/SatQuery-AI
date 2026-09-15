/**
 * What happens between the question and the number — the four stages the
 * backend actually runs, each with its own small animation, played once as
 * it scrolls into view. Numbers come from the recorded bi-temporal fixture
 * (`mocks/captured/events.bitemporal.json`); nothing here is invented.
 */
import { useEffect, useRef } from 'react'

import { PlayIcon } from '@/components/ui/icons'
import { useReducedMotion } from '@/shell/useReducedMotion'
import { useLandingStore, type StageId } from '@/state/landing'

const CHECKS = ['CRS', 'GSD', 'overlap', 'co-reg', 'nodata', 'bands', 'dtype', 'modality', 'extent', 'cloud', 'dates']

const STAGES: { id: StageId; title: string; sub: string; body: string }[] = [
  {
    id: 1,
    title: 'Raster ingest & pre-flight',
    sub: 'POST /v1/validate · 11 checks',
    body: 'Both rasters are opened, their CRS, ground sampling distance, overlap and co-registration checked before any tool runs. A warning is reported, not hidden.',
  },
  {
    id: 2,
    title: 'Spatial policy DAG',
    sub: 'policy_key = TaskType | PairType | Modality',
    body: 'A rule cascade classifies the question; a frozen policy table turns the key into a plan. The LLM never chooses a tool.',
  },
  {
    id: 3,
    title: 'Co-register · render · mask',
    sub: 'spectral_renderer → siamese_change_detector',
    body: 'The pair is placed on a common grid, rendered as true-colour, NDVI and NDBI views, and the change mask is measured — pixels, m², km².',
  },
  {
    id: 4,
    title: 'Evidence-bound synthesis',
    sub: 'vlm_change_vqa → CitationValidator',
    body: 'The VLM writes the sentence; the validator binds each number to a scalar in the trace, and underlines any it cannot.',
  },
]

function Stage1({ played }: { played: boolean }) {
  return (
    <ul className="flex flex-wrap gap-x-1.5 gap-y-2" aria-hidden>
      {CHECKS.map((check, i) => (
        <li
          key={check}
          className={`chip border transition-colors duration-[220ms] ${
            played
              ? i === 9
                ? 'border-warn/60 bg-warn/12 text-warn-text'
                : 'border-ok/50 bg-ok/12 text-ok-text'
              : 'border-line text-text-lo'
          }`}
          style={{ transitionDelay: played ? `${i * 70}ms` : '0ms' }}
        >
          {check}
        </li>
      ))}
    </ul>
  )
}

function Stage2({ played }: { played: boolean }) {
  const nodes = [
    [20, 40],
    [80, 20],
    [80, 60],
    [140, 40],
    [200, 40],
  ] as const
  const edges = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [3, 4],
  ] as const
  return (
    <div aria-hidden>
      <p className="t-coord mb-3 text-accent-warm-text">
        <span
          className="inline-block align-bottom"
          style={{ animation: played ? 'sq-wipe 900ms linear both' : 'none', clipPath: played ? undefined : 'inset(0 100% 0 0)' }}
        >
          CHANGE_VQA|BI_TEMPORAL|optical → plan_change_optical_v1
        </span>
      </p>
      <svg viewBox="0 0 220 80" className="h-20 w-full max-w-[280px]" fill="none">
        {edges.map(([a, b], i) => (
          <line
            key={i}
            x1={nodes[a][0]}
            y1={nodes[a][1]}
            x2={nodes[b][0]}
            y2={nodes[b][1]}
            stroke="var(--color-accent-warm)"
            strokeWidth="1.2"
            pathLength={1}
            strokeDasharray={1}
            strokeDashoffset={played ? 0 : 1}
            style={{ transition: 'stroke-dashoffset 360ms var(--ease-out-quint)', transitionDelay: `${300 + i * 180}ms` }}
          />
        ))}
        {nodes.map(([x, y], i) => (
          <circle
            key={i}
            cx={x}
            cy={y}
            r="6"
            fill="var(--color-surface-card)"
            stroke="var(--color-surface-sand)"
            strokeWidth="1.2"
            style={{ opacity: played ? 1 : 0, transition: 'opacity 220ms', transitionDelay: `${200 + i * 180}ms` }}
          />
        ))}
      </svg>
    </div>
  )
}

function Stage3({ played }: { played: boolean }) {
  const tabs = ['TC', 'NDVI', 'NDBI', 'CHANGE']
  return (
    <div aria-hidden>
      <div className="relative h-24 w-full max-w-[280px] overflow-hidden rounded-lg border border-line bg-bg-main">
        <div className="graticule absolute inset-0" />
        <div
          className="absolute inset-3 rounded border border-accent-warm/60 transition-transform duration-[600ms] ease-[var(--ease-out-quint)]"
          style={{ transform: played ? 'translate(0,0)' : 'translate(4px,3px)' }}
        />
        <div className="absolute inset-3 rounded border border-surface-sand/60" />
        <div
          className="absolute right-6 bottom-5 h-8 w-14 rounded-sm bg-warn/12 outline outline-1 outline-warn/60 transition-opacity duration-[400ms]"
          style={{ opacity: played ? 1 : 0, transitionDelay: '900ms' }}
        />
      </div>
      <ul className="mt-2 flex gap-1.5">
        {tabs.map((tab, i) => (
          <li
            key={tab}
            className={`chip border transition-colors duration-[220ms] ${played ? 'border-accent-warm bg-accent-glow text-accent-warm-text' : 'border-line text-text-lo'}`}
            style={{ transitionDelay: played ? `${500 + i * 160}ms` : '0ms' }}
          >
            {tab}
          </li>
        ))}
      </ul>
    </div>
  )
}

function Stage4({ played }: { played: boolean }) {
  return (
    <p aria-hidden className="max-w-[28ch] text-[13px] leading-relaxed text-text-hi">
      <span
        className={`inline-block overflow-hidden align-bottom whitespace-nowrap transition-[max-width] duration-[1200ms] ease-linear ${played ? 'max-w-[60ch]' : 'max-w-0'}`}
      >
        Approximately{' '}
      </span>
      <span
        className={`inline-flex items-center rounded-md px-1.5 py-0.5 font-medium transition-colors duration-[300ms] ${played ? 'bg-evidence text-on-evidence' : 'text-text-hi'}`}
        style={{ transitionDelay: '1300ms' }}
      >
        5.34 %
      </span>{' '}
      of the scene (
      <span
        className={`inline-flex items-center rounded-md px-1.5 py-0.5 font-medium transition-colors duration-[300ms] ${played ? 'bg-evidence text-on-evidence' : 'text-text-hi'}`}
        style={{ transitionDelay: '1500ms' }}
      >
        0.35 km²
      </span>
      ) changed between the two acquisitions, across{' '}
      <span className={played ? 'uncited' : ''} style={{ transitionDelay: '1700ms' }}>
        1 region
      </span>
      .
      <span className="t-coord mt-2 block text-text-lo">
        step:3/scalars.changed_area_pct · confidence 0.65
      </span>
    </p>
  )
}

const PARTS = { 1: Stage1, 2: Stage2, 3: Stage3, 4: Stage4 } as const

export function PipelineStory() {
  const played = useLandingStore((state) => state.storyPlayed)
  const replaying = useLandingStore((state) => state.storyReplaying)
  const mark = useLandingStore((state) => state.markStagePlayed)
  const replay = useLandingStore((state) => state.replayStory)
  const reduced = useReducedMotion()
  const refs = useRef<(HTMLLIElement | null)[]>([])

  useEffect(() => {
    if (reduced) {
      for (const stage of STAGES) mark(stage.id)
      return
    }
    if (typeof IntersectionObserver === 'undefined') {
      for (const stage of STAGES) mark(stage.id)
      return
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue
          const id = Number((entry.target as HTMLElement).dataset['stage']) as StageId
          mark(id)
          io.unobserve(entry.target)
        }
      },
      { threshold: 0.2 },
    )
    for (const el of refs.current) if (el) io.observe(el)
    return () => io.disconnect()
  }, [mark, reduced])

  return (
    <section id="story" aria-labelledby="story-title" className="relative mx-auto max-w-[1280px] px-4 py-28 wide:px-6 desk:py-36">
      <div data-reveal>
        <h2 id="story-title" className="t-section max-w-[42ch] text-text-hi">
          What happens between the question and the number.
        </h2>
        <p className="t-lede mt-5">
          “The LLM performs slot filling and answer synthesis only. It never chooses a tool, never
          orders steps, and never invents a capability.” — <span className="t-coord">AGENT_POLICY_DAG.md §0</span>
        </p>
      </div>

      <ol className="sr-only">
        {STAGES.map((stage) => (
          <li key={stage.id}>
            {stage.title}: {stage.body}
          </li>
        ))}
      </ol>

      {/*
       * Four stages, one beam. The grid gutters are wide enough (48 px at
       * `desk`) that four columns of 14 px copy read as four columns rather
       * than one paragraph with rivers; inside a stage the three groups —
       * marker, text, diagram — get three different gaps because they are
       * three different things. The diagram sits in a fixed slot so every
       * column's drawing starts on the same line.
       */}
      <ol className="relative mt-16 grid grid-cols-1 gap-14 wide:grid-cols-2 wide:gap-x-10 wide:gap-y-16 desk:mt-20 desk:grid-cols-4 desk:gap-x-12" aria-hidden>
        <li aria-hidden className="pointer-events-none absolute top-[19.5px] right-0 left-0 hidden h-px desk:block">
          <svg className="size-full" preserveAspectRatio="none">
            <line x1="0" y1="0.5" x2="100%" y2="0.5" stroke="var(--color-line)" />
            {/* `pathLength` normalises the dash to the line, so the pulse
                travels once per cycle at any viewport width. Scroll-linked
                where the browser can (`data-beam` in theme.css), a 6 s loop
                where it cannot. */}
            <line
              data-beam
              x1="0"
              y1="0.5"
              x2="100%"
              y2="0.5"
              stroke="var(--color-surface-sand)"
              strokeWidth="2"
              pathLength={1000}
              strokeDasharray="14 986"
            />
          </svg>
        </li>
        {STAGES.map((stage, i) => {
          const Part = PARTS[stage.id]
          const on = played.includes(stage.id)
          return (
            <li
              key={stage.id}
              data-stage={stage.id}
              ref={(el) => {
                refs.current[i] = el
              }}
              className={`relative grid grid-cols-[2.5rem_1fr] gap-x-4 wide:flex wide:flex-col ${replaying === stage.id ? 'rounded-xl ring-1 ring-accent-warm/40 ring-offset-8 ring-offset-bg-main' : ''}`}
            >
              {/* On a phone the stages read as a timeline: badge in the
                  gutter, a rule from badge to badge, the text beside it. */}
              {i < STAGES.length - 1 ? (
                <span aria-hidden className="absolute top-10 -bottom-14 left-5 w-px bg-line wide:hidden" />
              ) : null}
              <span className="relative z-[1] grid size-10 place-items-center rounded-full border border-line bg-surface-card font-mono text-[13px] text-accent-warm-text outline outline-[6px] outline-bg-main">
                {stage.id}
              </span>
              <div className="pt-2.5 wide:mt-6 wide:pt-0">
                <h3 className="t-stage text-text-hi">{stage.title}</h3>
                <p className="t-coord mt-2 text-text-lo">{stage.sub}</p>
                <p className="mt-4 max-w-[30ch] text-[14px] leading-[1.6] text-text-lo">{stage.body}</p>
              </div>
              <div data-diagram className="col-start-2 mt-6 min-h-[128px] wide:mt-8">
                <Part played={on} />
              </div>
            </li>
          )
        })}
      </ol>

      <div className="mt-12 flex justify-end">
        <button type="button" onClick={() => void replay()} disabled={replaying !== null} className="btn-ghost-sm flex items-center gap-1.5">
          <PlayIcon size={11} />
          Replay
        </button>
      </div>
    </section>
  )
}
