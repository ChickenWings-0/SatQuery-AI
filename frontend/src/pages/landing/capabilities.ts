/**
 * The three capabilities the landing page argues for, as data: label, the
 * task types and tools behind it, the question "Try this in the console"
 * seeds, and the sample image path (`public/samples/capabilities/`).
 */
import type { CapabilityId } from '@/state/landing'

export interface Capability {
  id: CapabilityId
  label: string
  tasks: string
  tools: string[]
  copy: string
  question: string
  image: string
  alt: string
}

export const CAPABILITIES: Capability[] = [
  {
    id: 'change',
    label: 'Multimodal change detection',
    tasks: 'CHANGE_VQA · CHANGE_MAP',
    tools: ['siamese_change_detector', 'change_statistics'],
    copy: 'Two dates, one co-registered pair, and a change mask whose area is a measured scalar, not a guess.',
    question: 'How much of the scene transitioned to built-up between the two dates?',
    image: '/samples/capabilities/change.webp',
    alt: 'Sentinel-2 true-colour pair with the change mask on the right half',
  },
  {
    id: 'crossmodal',
    label: 'SAR / optical consistency',
    tasks: 'CROSS_MODAL_COMPARE',
    tools: ['physics_agreement', 'crossmodal_consistency'],
    copy: 'Backscatter and reflectance answer separately; physics_agreement says where they disagree, and the answer tells you.',
    question: 'Where do optical and SAR disagree about built-up area?',
    image: '/samples/capabilities/crossmodal.webp',
    alt: 'Optical true-colour beside SAR VV/VH with the disagreement mask',
  },
  {
    id: 'grounding',
    label: 'Sub-pixel grounding',
    tasks: 'GROUNDING · COUNT',
    tools: ['text_grounding', 'object_counter'],
    copy: 'Boxes are validated as positions against the raster grid before the VLM is allowed to cite them.',
    question: 'Count the aircraft on the apron and locate each one.',
    image: '/samples/capabilities/grounding.webp',
    alt: 'A very-high-resolution frame with three validated bounding boxes',
  },
]
