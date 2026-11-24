/**
 * The curated investigations. Each row names a real `TaskType`, the tools the
 * policy table will select for it, and where its rasters live under
 * `public/samples/<slug>/`. The rasters are supplied by the team (see
 * `DOCS/SAMPLE_SCENES.md`); a row whose manifest is missing is shown with
 * its plate but cannot be loaded, and says so.
 */
import type { TaskType } from '@/api/types'

export type Sensor = 'Sentinel-2' | 'Sentinel-1' | 'VHR optical' | 'Sentinel-2 + Sentinel-1'
export type SamplePair = 'SINGLE' | 'BI_TEMPORAL' | 'CROSS_MODAL'
export type TaskFamily = 'change' | 'crossmodal' | 'grounding' | 'classify'

export interface UseCase {
  slug: string
  title: string
  region: string
  centroid: [lat: number, lon: number]
  years: [string, string] | [string]
  sensor: Sensor
  pairType: SamplePair
  task: TaskType
  family: TaskFamily
  question: string
  tools: string[]
  files: string[]
  sizeMb: number
}

export const FAMILY_LABELS: Record<TaskFamily, string> = {
  change: 'Change',
  crossmodal: 'Cross-modal',
  grounding: 'Grounding',
  classify: 'Classify',
}

export const USE_CASES: readonly UseCase[] = [
  {
    slug: 'bengaluru-sprawl',
    title: 'Urban sprawl, Bengaluru',
    region: 'Karnataka, IN',
    centroid: [12.965, 77.7275],
    years: ['2019', '2024'],
    sensor: 'Sentinel-2',
    pairType: 'BI_TEMPORAL',
    task: 'CHANGE_VQA',
    family: 'change',
    question: 'How much of the scene transitioned to built-up between the two dates?',
    tools: ['image_diff_change', 'change_statistics', 'spectral_index_analyzer', 'vlm_change_vqa'],
    files: ['pre.tif', 'post.tif'],
    sizeMb: 9,
  },
  {
    slug: 'brahmaputra-flood',
    title: 'Brahmaputra flood delineation',
    region: 'Assam, IN',
    centroid: [26.175, 91.62],
    years: ['2024'],
    sensor: 'Sentinel-1',
    pairType: 'SINGLE',
    task: 'SEGMENTATION',
    family: 'grounding',
    question: 'Outline the inundated area and estimate its extent.',
    tools: ['sar_backscatter_analyzer', 'class_outliner', 'vlm_vqa'],
    files: ['scene.tif'],
    sizeMb: 2,
  },
  {
    slug: 'airbase-apron',
    title: 'Airport apron — aircraft count',
    region: 'Phoenix, US · NAIP 0.3 m',
    centroid: [33.434, -112.013],
    years: ['2023'],
    sensor: 'VHR optical',
    pairType: 'SINGLE',
    task: 'COUNT',
    family: 'grounding',
    question: 'Count the aircraft on the apron and locate each one.',
    tools: ['text_grounding', 'object_counter', 'vlm_vqa'],
    files: ['scene.tif'],
    sizeMb: 8,
  },
  {
    slug: 'sundarbans-ndvi',
    title: 'Sundarbans mangrove vigour',
    region: 'West Bengal, IN',
    centroid: [21.95, 88.895],
    years: ['2018', '2024'],
    sensor: 'Sentinel-2',
    pairType: 'BI_TEMPORAL',
    task: 'CHANGE_MAP',
    family: 'change',
    question: 'Map where vegetation vigour dropped between the dates.',
    tools: ['image_diff_change', 'change_statistics', 'spectral_index_analyzer'],
    files: ['pre.tif', 'post.tif'],
    sizeMb: 7,
  },
  {
    slug: 'chennai-reservoir',
    title: 'Chennai reservoir water extent',
    region: 'Tamil Nadu, IN',
    centroid: [13.1575, 80.1575],
    years: ['2019', '2023'],
    sensor: 'Sentinel-2',
    pairType: 'BI_TEMPORAL',
    task: 'CHANGE_VQA',
    family: 'change',
    question: 'Did the reservoir’s surface water shrink or grow?',
    tools: ['image_diff_change', 'change_statistics', 'spectral_index_analyzer', 'vlm_change_vqa'],
    files: ['pre.tif', 'post.tif'],
    sizeMb: 9,
  },
  {
    slug: 'kerala-cross-modal',
    title: 'Optical vs SAR under cloud, Kerala',
    region: 'Kerala, IN',
    centroid: [9.9325, 76.2675],
    years: ['2024'],
    sensor: 'Sentinel-2 + Sentinel-1',
    pairType: 'CROSS_MODAL',
    task: 'CROSS_MODAL_COMPARE',
    family: 'crossmodal',
    question: 'Where do optical and SAR disagree about built-up area?',
    tools: ['spectral_index_analyzer', 'sar_backscatter_analyzer', 'crossmodal_consistency', 'physics_agreement'],
    files: ['optical.tif', 'sar.tif'],
    sizeMb: 6,
  },
  {
    slug: 'kutch-landcover',
    title: 'Land cover, Rann of Kutch',
    region: 'Gujarat, IN',
    centroid: [23.85, 69.75],
    years: ['2024'],
    sensor: 'Sentinel-2',
    pairType: 'SINGLE',
    task: 'SCENE_CLASSIFY',
    family: 'classify',
    question: 'Classify the land cover in this scene.',
    tools: ['spectral_index_analyzer', 'vlm_vqa'],
    files: ['scene.tif'],
    sizeMb: 5,
  },
  {
    slug: 'mundra-port',
    title: 'Port expansion, Mundra',
    region: 'Gujarat, IN',
    centroid: [22.745, 69.7025],
    years: ['2017', '2024'],
    sensor: 'Sentinel-2',
    pairType: 'BI_TEMPORAL',
    task: 'CHANGE_CAPTION',
    family: 'change',
    question: 'Summarise what changed at the port.',
    tools: ['image_diff_change', 'change_statistics', 'spectral_index_analyzer', 'vlm_change_vqa'],
    files: ['pre.tif', 'post.tif'],
    sizeMb: 7,
  },
]

export function sampleUrl(useCase: UseCase, file: string): string {
  return `/samples/${useCase.slug}/${file}`
}

/** `12°58′N 77°35′E` from a decimal centroid. */
export function dms([lat, lon]: [number, number], seconds = false): string {
  const part = (value: number, pos: string, neg: string) => {
    const abs = Math.abs(value)
    const d = Math.floor(abs)
    const mFloat = (abs - d) * 60
    const m = Math.floor(mFloat)
    const s = Math.round((mFloat - m) * 60)
    return `${d}°${String(m).padStart(2, '0')}′${seconds ? `${String(s).padStart(2, '0')}″` : ''}${value >= 0 ? pos : neg}`
  }
  return `${part(lat, 'N', 'S')} ${part(lon, 'E', 'W')}`
}
