/**
 * The Data Stage's empty state: drop target → manifests → compatibility.
 *
 * This is the whole of F2. It runs before the user has typed anything, and its
 * job is to answer one question — *can these images be analysed together, and
 * for what?* — using only the GPU-free `/v1/validate` route.
 *
 * The task chips are the primary call to action on the first screen, so they
 * are real controls: clicking one writes its question into the composer and
 * hands over the caret. They spent a while as `<button>`s with hover styling
 * and no handler, which is the worst kind of broken — it looks fine.
 */
import { Dropzone } from '@/components/stage/Dropzone'
import { CompatibilityDetails } from '@/components/stage/CompatibilityDetails'
import { ManifestCard } from '@/components/stage/ManifestCard'
import { countOf } from '@/format'
import { useFocusStore } from '@/state/focus'
import { useUiStore } from '@/state/ui'
import type { TaskType } from '@/api/types'

/** Questions worth offering, per task. Seeded from API_CONTRACT §7. */
const SUGGESTIONS: Partial<Record<TaskType, string>> = {
  CAPTION: 'Describe this scene and identify water bodies.',
  VQA: 'What kind of terrain is this?',
  SCENE_CLASSIFY: 'What land-cover classes are present?',
  COUNT: 'How many buildings are visible?',
  GROUNDING: 'Where are the built-up areas?',
  SEGMENTATION: 'Segment the water bodies.',
  CHANGE_VQA: 'How much built-up area appeared?',
  CHANGE_CAPTION: 'Describe what changed between these two dates.',
  CHANGE_MAP: 'Produce a change map for this pair.',
  CROSS_MODAL_VQA: 'What does the SAR show that the optical misses?',
  CROSS_MODAL_COMPARE: 'Compare the optical and SAR views.',
}

/** A task the registry has no phrasing for still gets an askable question. */
function questionFor(task: TaskType): string {
  return SUGGESTIONS[task] ?? `Run ${task.toLowerCase().replace(/_/g, ' ')} on this imagery.`
}

export function PreflightPanel() {
  // Per-field selectors: `useUiStore()` re-rendered the whole pre-flight —
  // including every manifest card and the compatibility table — on any store
  // write at all, including ones from other columns.
  const files = useUiStore((state) => state.files)
  const validation = useUiStore((state) => state.validation)
  const validating = useUiStore((state) => state.validating)
  const validationError = useUiStore((state) => state.validationError)
  const clearFiles = useUiStore((state) => state.clearFiles)
  const proposeQuestion = useFocusStore((state) => state.proposeQuestion)

  const blocked = validation?.compatibility.overall === 'FAIL'
  const tasks = validation?.supported_tasks ?? []

  return (
    <div className="mx-auto max-w-4xl space-y-5 md:space-y-6">
      <header>
        <h2 className="t-page">Explore</h2>
        <p className="mt-1.5 text-[13px] text-text-lo">
          Upload imagery and SatQuery checks what it can answer before you ask.
        </p>
      </header>

      {files.length === 0 ? (
        <Dropzone />
      ) : (
        <div className="flex items-center justify-between gap-3">
          <p className="tabular min-w-0 truncate text-sm text-text-lo">
            {countOf(files.length, { one: 'file', other: 'files' })} selected
          </p>
          <button
            type="button"
            onClick={clearFiles}
            className="shrink-0 rounded px-1 py-1.5 text-sm text-accent-warm-text hover:underline"
          >
            Clear
          </button>
        </div>
      )}

      {/* `role="status"` on both: pre-flight is the first thing that happens
          after a drop, and a screen-reader user otherwise hears nothing at all
          between selecting a file and the manifests appearing. */}
      {validating && (
        <p
          role="status"
          className="rounded-lg border border-line bg-surface-card px-4 py-3 text-sm text-text-lo"
        >
          Running pre-flight — reading headers, checking compatibility. No GPU involved.
        </p>
      )}

      {validationError && (
        <p
          role="alert"
          className="rounded-lg border border-fail/40 bg-fail/8 px-4 py-3 text-sm [overflow-wrap:anywhere]"
        >
          {validationError}
        </p>
      )}

      {validation && (
        <>
          <section className="space-y-3">
            {validation.inputs.map((manifest, index) => (
              <ManifestCard
                key={manifest.id}
                manifest={manifest}
                {...(files[index] ? { previewUrl: files[index].previewUrl } : {})}
              />
            ))}
          </section>

          <CompatibilityDetails report={validation.compatibility} />

          <section>
            <h3 className="t-eyebrow">
              {blocked ? 'Unavailable' : 'Questions these images can answer'}
            </h3>

            {blocked ? (
              <p className="mt-2 rounded-lg border border-fail/40 bg-fail/8 px-4 py-3 text-sm">
                These images cannot be analysed together. Fix the failing check above, or upload
                a different pair.
              </p>
            ) : tasks.length === 0 ? (
              // A PASS with no supported tasks is possible — a single unusual
              // sensor with no matching policy row. Silence would read as a
              // rendering fault rather than as an answer.
              <p className="mt-2 rounded-lg border border-dashed border-line px-4 py-3 text-sm text-text-lo">
                Pre-flight found no task the registry knows how to run on these images. You can
                still ask a free-form question; the planner will classify it.
              </p>
            ) : (
              <ul className="mt-2.5 flex flex-wrap gap-2">
                {tasks.map((task) => {
                  const question = questionFor(task)
                  return (
                    <li key={task} className="min-w-0 max-w-full">
                      <button
                        type="button"
                        onClick={() => proposeQuestion(question)}
                        // Sized by padding, not by a hard height: the
                        // `pointer: coarse` floor in theme.css lifts this to
                        // 44px for a finger and leaves the desk layout dense.
                        className="flex w-full items-center gap-2 rounded-lg border border-line bg-surface-card px-3 py-2 text-left text-[13px] transition-colors hover:border-accent-cool hover:bg-accent-cool"
                      >
                        <span className="chip shrink-0 text-text-lo">{task}</span>
                        <span className="min-w-0 [overflow-wrap:anywhere]">{question}</span>
                      </button>
                    </li>
                  )
                })}
              </ul>
            )}
          </section>

          {(validation.warnings?.length ?? 0) > 0 && (
            <ul className="space-y-1">
              {/* The same warning arrives once per image, so code+message is not
                  unique — only the position in the array is. */}
              {validation.warnings?.map((warning, index) => (
                <li
                  key={`${warning.code}-${warning.ref ?? index}`}
                  className="flex gap-1.5 text-xs text-text-lo [overflow-wrap:anywhere]"
                >
                  <span aria-hidden className="shrink-0 text-warn-text">
                    ⚠
                  </span>
                  <span>
                    <span className="font-mono">{warning.code}</span> {warning.message}
                    {warning.ref && <span className="font-mono"> ({warning.ref})</span>}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  )
}
