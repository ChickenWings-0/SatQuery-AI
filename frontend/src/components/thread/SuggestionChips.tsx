/**
 * Quick-action chips under the composer.
 *
 * Before pre-flight there is nothing to ground a suggestion in, so the chips
 * are the four questions the product is built to answer — a demonstration of
 * range, not a menu. Once pre-flight has run, `supported_tasks` says what these
 * images can actually be asked, and the chips become those questions instead:
 * offering "find floods" on a single SAR scene would be a suggestion the run
 * cannot honour.
 *
 * Phrased differently from the pre-flight panel's own suggestions on purpose.
 * Both reach the composer through `proposeQuestion`, but two controls on one
 * screen with the same accessible name are one control too many for a screen
 * reader — and a chip is a nudge, where the pre-flight card is a full question.
 */
import type { TaskType } from '@/api/types'
import { useFocusStore } from '@/state/focus'
import { useUiStore } from '@/state/ui'

const DEFAULTS = [
  'Compare vegetation before and after 2020',
  'Find floods in my area',
  'Detect new constructions',
  'Show NDVI trend',
]

const BY_TASK: Partial<Record<TaskType, string>> = {
  CHANGE_VQA: 'Detect new constructions',
  CHANGE_CAPTION: 'Summarise what changed',
  CHANGE_MAP: 'Map the change footprint',
  CAPTION: 'Describe the scene',
  VQA: 'Identify the land use',
  SCENE_CLASSIFY: 'Classify the land cover',
  COUNT: 'Count the buildings',
  GROUNDING: 'Locate built-up areas',
  SEGMENTATION: 'Outline the water bodies',
  CROSS_MODAL_VQA: 'Compare optical with SAR',
  CROSS_MODAL_COMPARE: 'Where do optical and SAR disagree?',
}

export function SuggestionChips() {
  const validation = useUiStore((state) => state.validation)
  const proposeQuestion = useFocusStore((state) => state.proposeQuestion)

  const chips = validation
    ? validation.supported_tasks
        .map((task) => BY_TASK[task])
        .filter((chip): chip is string => Boolean(chip))
        .slice(0, 4)
    : DEFAULTS
  if (chips.length === 0) return null

  return (
    <ul className="flex flex-wrap gap-1.5" aria-label="Suggested questions">
      {chips.map((chip) => (
        <li key={chip}>
          <button
            type="button"
            onClick={() => proposeQuestion(chip)}
            className="rounded-full border border-line px-3 py-1.5 text-[12px] text-text-lo transition-colors hover:border-accent-warm hover:text-accent-warm-text"
          >
            {chip}
          </button>
        </li>
      ))}
    </ul>
  )
}
