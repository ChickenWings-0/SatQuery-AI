/**
 * The question input. Disabled until imagery has passed pre-flight, because
 * `supported_tasks` is what makes the suggestions meaningful.
 *
 * The text lives in `useFocusStore.draft` rather than in local state so the
 * pre-flight's suggestion chips — two columns away — can fill it in. Losing the
 * user's typing is the one thing this component must never do, so the draft is
 * never cleared by the app: it survives a failed run, a cancelled run, a
 * section change and the narrow layout remounting the thread elsewhere. The
 * question also stays readable beside the answer it produced, which is what
 * you want in a console you are meant to audit.
 */
import { useEffect, useId, useRef } from 'react'

import { CameraIcon, ChevronDownIcon, SendIcon } from '@/components/ui/icons'
import { countOf } from '@/format'
import { useFocusStore } from '@/state/focus'
import { queryRoom, useSettingsStore } from '@/state/settings'
import { useUiStore } from '@/state/ui'

/** The counter appears once this many characters are left. */
const COUNTDOWN_AT = 100

export function QueryComposer({
  onSubmit,
  onCancel,
  busy,
}: {
  onSubmit: (query: string) => void
  onCancel: () => void
  busy: boolean
}) {
  const value = useFocusStore((state) => state.draft)
  // The server caps `query` at 1000 characters and custom instructions are
  // sent inside it, so the box's ceiling is what the instructions leave.
  const instructions = useSettingsStore((state) => state.customInstructions)
  const MAX_LENGTH = queryRoom(instructions)
  const setDraft = useFocusStore((state) => state.setDraft)
  const setComposer = useFocusStore((state) => state.setComposer)
  const validation = useUiStore((state) => state.validation)
  const validating = useUiStore((state) => state.validating)
  const textarea = useRef<HTMLTextAreaElement>(null)
  const hintId = useId()

  // Registered rather than queried by selector, so `/` focuses *this* textarea
  // even once the narrow layout renders the thread somewhere else.
  useEffect(() => {
    setComposer(textarea.current)
    return () => setComposer(null)
  }, [setComposer])

  const ready = Boolean(validation) && validation?.compatibility.overall !== 'FAIL'
  const blocked = validation?.compatibility.overall === 'FAIL'
  const trimmed = value.trim()
  const remaining = MAX_LENGTH - value.length

  const placeholder = ready
    ? 'Ask about this imagery…'
    : validating
      ? 'Checking the imagery…'
      : blocked
        ? 'These images cannot be analysed together'
        : 'Upload imagery to begin'

  const status = ready
    ? null
    : validating
      ? 'Running pre-flight on the selected imagery.'
      : blocked
        ? 'Pre-flight failed: these images cannot be analysed together. Fix the failing check or upload a different pair.'
        : 'Upload one or two scenes in the Data Stage to enable questions.'

  function send() {
    // Guarded here as well as on the button: `⌘↵` bypasses `disabled`.
    if (!ready || busy || trimmed === '') return
    onSubmit(trimmed)
  }

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        send()
      }}
    >
      <div className="rounded-xl border border-line bg-surface-card transition-colors focus-within:border-accent-warm focus-within:ring-2 focus-within:ring-accent-glow">
        <textarea
          ref={textarea}
          value={value}
          onChange={(event) => setDraft(event.target.value.slice(0, MAX_LENGTH))}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
              event.preventDefault()
              send()
            }
          }}
          rows={3}
          maxLength={MAX_LENGTH}
          disabled={!ready || busy}
          aria-label="Question about this imagery"
          aria-describedby={hintId}
          placeholder={placeholder}
          // `break-words` matters: a pasted 800-character URL or an unbroken
          // CJK run has no space to wrap at and would otherwise scroll the
          // textarea sideways under its own controls.
          className="w-full resize-none [overflow-wrap:anywhere] bg-transparent px-3.5 py-3 text-[13px] leading-relaxed outline-none placeholder:text-text-lo disabled:opacity-50"
        />
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5 border-t border-line-soft px-2.5 py-2">
          {/* Imagery goes in through the Data Stage's dropzone; this control
              is a scaffold for attaching it from here, and says so. */}
          <button
            type="button"
            aria-label="Attach imagery (upload in the Data Stage for now)"
            title="Attach imagery — upload in the Data Stage for now"
            disabled
            className="grid size-8 shrink-0 place-items-center rounded-lg text-text-lo transition-colors hover:bg-accent-glow hover:text-text-hi disabled:opacity-50"
          >
            <CameraIcon size={16} />
          </button>
          <button
            type="button"
            aria-haspopup="menu"
            aria-expanded={false}
            aria-label="Analysis mode: Auto"
            onClick={() => {
              /* Mode switching is not implemented yet. */
            }}
            className="flex h-8 shrink-0 items-center gap-1 rounded-lg border border-line px-2 text-[11px] font-medium text-text-lo transition-colors hover:border-accent-warm/40 hover:text-text-hi"
          >
            Auto
            <ChevronDownIcon size={12} />
          </button>

          <span
            id={hintId}
            className="flex min-w-0 items-center gap-1.5 text-[11px] text-text-lo"
            {...(status ? { role: 'status' } : {})}
          >
            {ready ? (
              <>
                <kbd className="kbd">/</kbd> focus
                <span aria-hidden className="text-line">
                  ·
                </span>
                <kbd className="kbd">⌘↵</kbd> run
              </>
            ) : (
              status
            )}
          </span>

          {/* Only once it is nearly a problem: a live counter on an empty box
              is a distraction, a silent truncation at 1000 is a bug. */}
          {remaining <= COUNTDOWN_AT && (
            <span
              role="status"
              className={`tabular text-[11px] ${remaining <= 0 ? 'font-medium' : 'text-text-lo'}`}
            >
              {remaining <= 0
                ? 'Character limit reached'
                : `${countOf(remaining, { one: 'character', other: 'characters' })} left`}
            </span>
          )}

          <span className="ml-auto flex items-center gap-2">
            {busy && (
              <button
                type="button"
                onClick={onCancel}
                className="rounded-lg border border-line px-2.5 py-1.5 text-[13px] text-text-lo transition-colors hover:border-surface-sand"
              >
                Stop
              </button>
            )}
            <button
              type="submit"
              disabled={!ready || busy || trimmed === ''}
              aria-label={busy ? 'Running' : 'Ask'}
              title={busy ? 'Running…' : 'Ask (⌘↵)'}
              className="btn-primary grid size-9 place-items-center !rounded-full !p-0"
            >
              <SendIcon size={18} className={busy ? 'sq-glow' : ''} />
            </button>
          </span>
        </div>
      </div>
    </form>
  )
}
