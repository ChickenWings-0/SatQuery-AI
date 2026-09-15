/**
 * Three things a chat box cannot do. The list is a real tablist: a click,
 * Enter, Space, the arrow keys, Home and End all select; hover does nothing
 * to the panel. A preview that changed under the cursor was a slideshow, not
 * a choice — and on touch the two states were indistinguishable.
 *
 * The preview stage is a drawn HUD (`landing/hud/*`) with a real
 * `public/samples/capabilities/<id>.webp` under it when one is supplied. On
 * a switch, the outgoing HUD slides out and the incoming one slides in, both
 * stacked in one grid cell so the outgoing side can animate at all (a `key`
 * remount could only animate the new one). The slide follows the list order.
 */
import { useEffect, useRef, useState, type KeyboardEvent } from 'react'

import { ArrowRightIcon } from '@/components/ui/icons'
import { CAPABILITIES, type Capability } from '@/pages/landing/capabilities'
import { ChangeHud } from '@/pages/landing/hud/ChangeHud'
import { CrossModalHud } from '@/pages/landing/hud/CrossModalHud'
import { GroundingHud } from '@/pages/landing/hud/GroundingHud'
import { useCursorLight } from '@/pages/landing/useScrollProgress'
import { useFocusStore } from '@/state/focus'
import { CAPABILITY_ORDER, useLandingStore, type CapabilityId } from '@/state/landing'
import { useUiStore } from '@/state/ui'

const HUDS = { change: ChangeHud, crossmodal: CrossModalHud, grounding: GroundingHud } as const

const byId = (id: CapabilityId) => CAPABILITIES.find((c) => c.id === id) ?? CAPABILITIES[0]!

function Preview({ cap, state }: { cap: Capability; state: 'in' | 'out' }) {
  const [hasImage, setHasImage] = useState(true)
  const card = useRef<HTMLDivElement>(null)
  // The card's specular follows the pointer, card-local.
  useCursorLight(card)
  const Hud = HUDS[cap.id]
  return (
    <div ref={card} data-cap-panel={state} className="card-flush group relative aspect-[16/10] w-full overflow-hidden bg-bg-main [grid-area:1/1]">
      {hasImage ? (
        <img
          src={cap.image}
          alt={cap.alt}
          loading="lazy"
          decoding="async"
          onError={() => setHasImage(false)}
          className="absolute inset-0 size-full object-cover saturate-[0.85] transition-[filter] duration-[220ms] group-hover:saturate-100"
        />
      ) : null}
      <Hud seed={cap.id} hasImage={hasImage} />
    </div>
  )
}

/** `/#cap-grounding` opens on that capability. Read once. */
function fromHash(): CapabilityId | null {
  if (typeof window === 'undefined') return null
  const id = window.location.hash.replace(/^#cap-/, '')
  return (CAPABILITY_ORDER as readonly string[]).includes(id) ? (id as CapabilityId) : null
}

export function Capabilities() {
  const active = useLandingStore((state) => state.activeCapability)
  const direction = useLandingStore((state) => state.capabilityDirection)
  const setCapability = useLandingStore((state) => state.setCapability)
  const setSection = useUiStore((state) => state.setSection)
  const proposeQuestion = useFocusStore((state) => state.proposeQuestion)
  const shown = byId(active)

  // The outgoing panel stays mounted for the length of its exit.
  const [leaving, setLeaving] = useState<CapabilityId | null>(null)
  const previous = useRef(active)
  useEffect(() => {
    if (previous.current === active) return
    setLeaving(previous.current)
    previous.current = active
    const timer = window.setTimeout(() => setLeaving(null), 220)
    return () => window.clearTimeout(timer)
  }, [active])

  useEffect(() => {
    const wanted = fromHash()
    if (wanted) setCapability(wanted)
  }, [setCapability])

  function tryIt() {
    setSection('explore')
    window.setTimeout(() => proposeQuestion(shown.question), 350)
  }

  function onKey(event: KeyboardEvent<HTMLButtonElement>, i: number) {
    const n = CAPABILITIES.length
    let next: number
    if (event.key === 'ArrowDown') next = (i + 1) % n
    else if (event.key === 'ArrowUp') next = (i + n - 1) % n
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = n - 1
    else return
    event.preventDefault()
    const target = CAPABILITIES[next]!
    setCapability(target.id)
    document.getElementById(`cap-tab-${target.id}`)?.focus()
  }

  return (
    <section id="capabilities" aria-labelledby="cap-title" className="relative mx-auto max-w-[1280px] px-4 py-28 wide:px-6 desk:py-36">
      <div data-reveal>
        <h2 id="cap-title" className="t-section text-text-hi">
          Three things a chat box cannot do.
        </h2>
      </div>
      <div className="mt-12 grid grid-cols-1 gap-10 desk:mt-16 desk:grid-cols-12 desk:gap-12">
        <div className="space-y-1 desk:col-span-5" role="tablist" aria-label="Capabilities" aria-orientation="vertical" data-reveal>
          {CAPABILITIES.map((cap, i) => {
            const on = cap.id === active
            return (
              <button
                key={cap.id}
                role="tab"
                id={`cap-tab-${cap.id}`}
                aria-selected={on}
                aria-controls="cap-panel"
                tabIndex={on ? 0 : -1}
                data-reveal-child
                data-state={on ? 'on' : 'off'}
                onClick={() => setCapability(cap.id)}
                onKeyDown={(event) => onKey(event, i)}
                className="cap-tab block w-full rounded-xl border px-5 py-4 text-left transition-colors duration-[220ms]"
              >
                <span className="t-stage block text-text-hi">{cap.label}</span>
                <span className="t-coord mt-2 block text-text-lo">{cap.tasks}</span>
                {/* `grid-template-rows: 0fr → 1fr` animates to the true height. */}
                <span className="cap-copy grid transition-[grid-template-rows,opacity] duration-[260ms] ease-[var(--ease-out-quint)]">
                  <span className="block min-h-0 overflow-hidden">
                    <span className="block pt-3 text-[14px] leading-[1.6] text-text-lo">{cap.copy}</span>
                  </span>
                </span>
              </button>
            )
          })}
        </div>
        <div id="cap-panel" role="tabpanel" aria-labelledby={`cap-tab-${shown.id}`} className="desk:col-span-7" data-reveal>
          <div className="grid" data-cap-direction={direction} style={{ contain: 'paint' }}>
            {leaving && leaving !== active ? <Preview key={leaving} cap={byId(leaving)} state="out" /> : null}
            <Preview key={active} cap={shown} state="in" />
          </div>
          <div key={shown.id} className="sq-arrive mt-5 flex flex-wrap items-center gap-2">
            {shown.tools.map((tool) => (
              <span key={tool} className="chip bg-line text-text-lo">
                {tool}
              </span>
            ))}
            <button type="button" onClick={tryIt} className="ml-auto flex items-center gap-1.5 text-[13px] font-medium text-accent-warm-text underline-offset-4 hover:underline">
              Try this in the console
              <ArrowRightIcon size={14} />
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}
