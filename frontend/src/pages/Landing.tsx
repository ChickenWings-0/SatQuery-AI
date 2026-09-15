/**
 * The landing page (`/`). Persuade mode: one rehearsed focal sequence (the
 * hero's acquisition), product truth as proof, one door into the console.
 * Rendered outside `AppShell` — no sidebar, no header, no thread.
 */
import { useEffect, useRef } from 'react'

import { Capabilities } from '@/pages/landing/Capabilities'
import { ClosingCta, LandingFooter } from '@/pages/landing/ClosingCta'
import { Hero } from '@/pages/landing/Hero'
import { LandingNav } from '@/pages/landing/LandingNav'
import { PipelineStory } from '@/pages/landing/PipelineStory'
import { useReveal } from '@/pages/landing/useReveal'
import { DocumentMeta } from '@/shell/DocumentMeta'
import { useLandingStore } from '@/state/landing'

export default function Landing() {
  const root = useRef<HTMLDivElement>(null)
  const reset = useLandingStore((state) => state.reset)
  useReveal(root)
  useEffect(() => () => reset(), [reset])

  return (
    <div ref={root} className="min-h-full bg-bg-main text-text-hi">
      <DocumentMeta page="home" />
      <a href="#main" className="skip-link">
        Skip to main content
      </a>
      <LandingNav />
      <main id="main" className="-mt-14">
        <Hero />
        <PipelineStory />
        <Capabilities />
        <ClosingCta />
      </main>
      <LandingFooter />
    </div>
  )
}
