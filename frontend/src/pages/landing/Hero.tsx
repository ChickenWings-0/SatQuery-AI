/**
 * The first viewport. Editorial, asymmetric: copy in the first seven columns,
 * the globe bleeding past the right edge. The headline is the page's `h1`;
 * the coordinate beside the reticle is the globe's initial target, so the
 * two agree. The entry sequence is one timeline (`[data-hero]` in theme.css).
 */
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'

import { Ambience } from '@/pages/landing/Ambience'
import { GlobePoster } from '@/pages/landing/globe/GlobePoster'
import { ProofStrip } from '@/pages/landing/ProofStrip'
import { launchWithFiles, useDropLaunch } from '@/pages/landing/useDropLaunch'
import { useGlobeVisibility } from '@/pages/landing/useGlobeVisibility'
import { useCursorLight, useScrollProgress } from '@/pages/landing/useScrollProgress'
import { dms } from '@/pages/usecases/catalogue'
import { ArrowRightIcon, PlayIcon } from '@/components/ui/icons'
import { Reticle } from '@/components/ui/Reticle'
import { useReducedMotion } from '@/shell/useReducedMotion'
import { useFocusStore } from '@/state/focus'
import { useLandingStore, type GlobeSupport } from '@/state/landing'
import { useUiStore } from '@/state/ui'

const Globe = lazy(() => import('@/pages/landing/globe/Globe'))
const ACCEPT = '.tif,.tiff,.png,.jpg,.jpeg,image/tiff,image/png,image/jpeg'
const TARGET: [number, number] = [12.9716, 77.5946]

function detectWebGL(): GlobeSupport {
  try {
    const canvas = document.createElement('canvas')
    const ok = canvas.getContext('webgl2') ?? canvas.getContext('webgl')
    return ok ? 'webgl' : 'none'
  } catch {
    return 'none'
  }
}

function lowEnd(): boolean {
  const nav = navigator as Navigator & { deviceMemory?: number; connection?: { saveData?: boolean } }
  return (nav.deviceMemory !== undefined && nav.deviceMemory < 2) || Boolean(nav.connection?.saveData)
}

export function Hero() {
  const setSection = useUiStore((state) => state.setSection)
  const focusComposer = useFocusStore((state) => state.focusComposer)
  const reduced = useReducedMotion()
  const globe = useLandingStore((state) => state.globe)
  const setGlobe = useLandingStore((state) => state.setGlobe)
  const dragOver = useLandingStore((state) => state.dragOver)
  const setPhase = useLandingStore((state) => state.setPhase)
  const [optIn, setOptIn] = useState(false)
  const globeBox = useRef<HTMLDivElement>(null)
  const hero = useRef<HTMLElement>(null)
  useDropLaunch()
  useGlobeVisibility(globeBox)
  // The hero's exit drives the parallax, the beam and the globe; the pointer
  // drives the glow and the globe's rim. Both are one listener each.
  useScrollProgress(hero)
  useCursorLight(hero, true)

  useEffect(() => {
    setPhase('acquiring')
    setGlobe(lowEnd() ? 'none' : detectWebGL())
    const timer = window.setTimeout(() => setPhase('live'), 1_200)
    return () => window.clearTimeout(timer)
  }, [setGlobe, setPhase])

  const wantsGlobe = globe === 'webgl' && (!reduced || optIn)

  // Preload the chunk after first paint, only when it will be used.
  useEffect(() => {
    if (!wantsGlobe) return
    const idle = (window as Window & { requestIdleCallback?: (cb: () => void) => number }).requestIdleCallback
    const run = () => void import('@/pages/landing/globe/Globe')
    if (idle) idle(run)
    else window.setTimeout(run, 1)
  }, [wantsGlobe])

  const launch = useCallback(() => {
    setSection('explore')
    window.setTimeout(focusComposer, 350)
  }, [setSection, focusComposer])

  return (
    <section ref={hero} aria-labelledby="hero-title" className="relative flex min-h-[100svh] flex-col justify-center overflow-hidden">
      <Ambience />

      {dragOver ? (
        <div aria-hidden className="pointer-events-none absolute inset-6 z-20 rounded-[var(--radius-hud)] border-2 border-dashed border-surface-sand">
          <span className="t-coord glass absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 px-4 py-2 text-text-hi">
            Drop the rasters — pre-flight starts here
          </span>
        </div>
      ) : null}

      <div className="relative mx-auto grid w-full max-w-[1280px] grid-cols-1 items-center gap-10 px-4 pt-28 pb-14 wide:px-6 desk:grid-cols-12 desk:gap-6 desk:pt-24 desk:pb-16">
        <div data-hero="copy" className="desk:col-span-7">
          <div className={`${dragOver ? 'opacity-40' : ''} transition-opacity duration-[220ms]`}>
          <div data-hero="coord" className="mb-6 flex flex-wrap items-center gap-x-3 gap-y-1">
            <Reticle data-hero="reticle" size={40} className="-ml-2" />
            <span className="t-coord text-accent-warm-text">{dms(TARGET, true)}</span>
            <span className="t-coord text-text-lo">initial target · Bengaluru</span>
          </div>

          <h1 id="hero-title" className="t-display text-text-hi">
            <span data-hero="line-1" className="block">
              From space to answers —
            </span>
            <span data-hero="line-2" className="block text-accent-warm-text">
              with the receipts.
            </span>
          </h1>

          <p data-hero="sub" className="t-display-sub mt-7">
            Bi-temporal, cross-modal SAR/optical analysis where every number is bound to a
            measurement and the tool graph is part of the answer.
          </p>

          <div data-hero="cta" className="mt-9 flex flex-wrap items-center gap-3">
            <button type="button" onClick={launch} className="btn-primary group flex h-11 items-center gap-2 !px-5 text-[15px]">
              <PlayIcon size={14} />
              Launch mission
              <ArrowRightIcon size={15} className="transition-transform duration-[120ms] group-hover:translate-x-0.5" />
            </button>
            <label className="btn-ghost flex h-11 cursor-pointer items-center gap-2 !px-4 text-[15px]">
              <span aria-hidden className="grid size-4 place-items-center rounded-[3px] border border-dashed border-current" />
              Drop imagery
              <input
                type="file"
                multiple
                accept={ACCEPT}
                className="sr-only"
                onChange={(event) => launchWithFiles(event.target.files)}
              />
            </label>
            <kbd className="kbd hidden !text-[11px] wide:inline-flex">⌘K</kbd>
          </div>

          <div className="mt-14 hidden desk:block">
            <ProofStrip />
          </div>
          </div>
        </div>

        <div data-hero="globe-wrap" className="relative desk:col-span-5">
          <div
            ref={globeBox}
            data-hero="globe"
            role="img"
            aria-label="Interactive globe. Drag to rotate; press plus or minus to zoom."
            tabIndex={0}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && reduced && globe === 'webgl') setOptIn(true)
            }}
            className="relative mx-auto aspect-square w-[min(72vw,520px)] cursor-grab active:cursor-grabbing focus-visible:outline-2 focus-visible:outline-offset-8 desk:w-[calc(100%+12vw)] desk:max-w-none desk:-mr-[12vw]"
          >
            {wantsGlobe ? (
              <Suspense fallback={<GlobePoster />}>
                <Globe />
              </Suspense>
            ) : (
              <GlobePoster />
            )}
          </div>
          {reduced && globe === 'webgl' && !optIn ? (
            <button type="button" onClick={() => setOptIn(true)} className="btn-ghost-sm mx-auto mt-3 flex items-center gap-1.5">
              <PlayIcon size={11} />
              Play globe
            </button>
          ) : null}
        </div>

        <div className="desk:hidden">
          <ProofStrip />
        </div>
      </div>
    </section>
  )
}
