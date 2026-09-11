/**
 * Grouping streamed artifacts into the things the viewer can actually show.
 *
 * The renderer emits one artifact per (view, temporal role), and the evidence
 * tray needs them the other way round: one entry per *view*, holding its `pre`
 * and `post` halves so the A/B slider has two images to compare.
 *
 * Two label grammars arrive over the wire and both must be understood:
 *
 *   - the renderer's, from `src/satquery/render/view_labels.py` —
 *     "Image 3 (optical false colour infrared, pre-change, Sentinel-2, …)"
 *   - the index analyser's — "NDVI (img_0_pre), domain -1 to 1"
 *
 * Rather than parse either strictly, the view is recognised by its name and the
 * temporal role by any of the markers both grammars use. That is deliberately
 * tolerant: a label that gains a field must not empty the evidence tray.
 */
import type { ArtifactRef } from '@/api/types'

/** The fixed view vocabulary of `view_labels.py`. */
export type ViewCode =
  | 'TC'
  | 'FCIR'
  | 'SWIR'
  | 'NDVI'
  | 'NDWI'
  | 'NDBI'
  | 'SARFC'
  | 'SARDB'
  | 'PAN'
  | 'CHANGE'
  | 'OTHER'

export const VIEW_NAMES: Record<ViewCode, string> = {
  TC: 'True colour',
  FCIR: 'False colour IR',
  SWIR: 'SWIR composite',
  NDVI: 'NDVI',
  NDWI: 'NDWI',
  NDBI: 'NDBI',
  SARFC: 'SAR false colour',
  SARDB: 'SAR backscatter',
  PAN: 'Panchromatic',
  CHANGE: 'Change',
  OTHER: 'Evidence',
}

/**
 * Views rendered on a fixed domain, so their legend is a constant rather than
 * something to derive per image. Mirrors `SCALE_NOTES` in `view_labels.py`; do
 * not re-derive these from pixels.
 */
export const FIXED_DOMAIN: Partial<Record<ViewCode, { min: number; max: number; unit?: string }>> =
  {
    NDVI: { min: -1, max: 1 },
    NDWI: { min: -1, max: 1 },
    NDBI: { min: -1, max: 1 },
    SARDB: { min: -25, max: 0, unit: 'dB' },
  }

// Order matters: the more specific pattern must win. "false colour VV/VH" and
// "false colour infrared" both contain "false colour".
const VIEW_PATTERNS: ReadonlyArray<readonly [ViewCode, RegExp]> = [
  ['SARFC', /false colour vv|vv\/vh|sar false/i],
  ['FCIR', /false colour infrared|fcir/i],
  ['TC', /true colour/i],
  ['SWIR', /swir/i],
  ['NDVI', /ndvi/i],
  ['NDWI', /ndwi/i],
  ['NDBI', /ndbi/i],
  ['SARDB', /backscatter/i],
  ['PAN', /panchromatic/i],
  ['CHANGE', /change (mask|overlay)|overlaid/i],
]

export type TemporalRole = 'pre' | 'post' | 'single'

export function viewCodeOf(artifact: ArtifactRef): ViewCode {
  if (artifact.type === 'CHANGE_MASK' || artifact.type === 'OVERLAY_PNG') return 'CHANGE'
  for (const [code, pattern] of VIEW_PATTERNS) {
    if (pattern.test(artifact.label)) return code
  }
  return 'OTHER'
}

export function roleOf(artifact: ArtifactRef): TemporalRole {
  const label = artifact.label
  if (/post-change|_post\b|img_1/i.test(label)) return 'post'
  if (/pre-change|_pre\b|img_0/i.test(label)) return 'pre'
  return 'single'
}

/** One entry in the evidence tray: a view, with whichever halves exist. */
export interface ViewGroup {
  /** Stable id: view code plus a discriminator for same-code artifacts. */
  key: string
  code: ViewCode
  name: string
  pre: ArtifactRef | null
  post: ArtifactRef | null
  single: ArtifactRef | null
  /** True when both halves exist, so the A/B slider is meaningful. */
  comparable: boolean
  /** Every artifact in this group, for step lookups. */
  members: ArtifactRef[]
}

/** The image a group shows when it is not being swiped. */
export function primaryOf(group: ViewGroup): ArtifactRef | null {
  return group.post ?? group.single ?? group.pre
}

/**
 * Fold artifacts into tray entries, preserving arrival order.
 *
 * Only artifacts with a `url` are included: `SCALARS` and `TEXT` have no render
 * and belong in the KPI cards or the pipeline modal, not the gallery.
 */
export function groupViews(artifacts: ArtifactRef[]): ViewGroup[] {
  const order: string[] = []
  const groups = new Map<string, ViewGroup>()

  for (const artifact of artifacts) {
    if (!artifact.url) continue

    const code = viewCodeOf(artifact)
    const role = roleOf(artifact)

    // CHANGE_MASK and OVERLAY_PNG are both "CHANGE" but are different pictures,
    // so they get their own entries rather than fighting over one slot.
    const key = code === 'CHANGE' || code === 'OTHER' ? `${code}:${artifact.id}` : code

    let group = groups.get(key)
    if (!group) {
      group = {
        key,
        code,
        // Two CHANGE artifacts land in two entries, so they need two names —
        // a tray showing "Change" twice tells the viewer nothing.
        name:
          artifact.type === 'CHANGE_MASK'
            ? 'Change mask'
            : artifact.type === 'OVERLAY_PNG'
              ? 'Change overlay'
              : VIEW_NAMES[code],
        pre: null,
        post: null,
        single: null,
        comparable: false,
        members: [],
      }
      groups.set(key, group)
      order.push(key)
    }

    group.members.push(artifact)
    if (role === 'pre' && !group.pre) group.pre = artifact
    else if (role === 'post' && !group.post) group.post = artifact
    else if (!group.single) group.single = artifact

    group.comparable = Boolean(group.pre && group.post)
  }

  return order.map((key) => groups.get(key)!).filter((group) => primaryOf(group) !== null)
}

/** Find the tray entry that holds a given artifact id. */
export function groupForArtifact(groups: ViewGroup[], artifactId: string): ViewGroup | undefined {
  return groups.find((group) => group.members.some((member) => member.id === artifactId))
}

/**
 * Pick the tray entry a citation should reveal.
 *
 * A citation names a step, not a picture. Prefer an output of that step whose
 * label mentions the scalar — `ndbi_mean_pre` should reveal the NDBI view, not
 * whichever heatmap happens to be first — and fall back to any output. A step
 * that produced only `SCALARS` (change_statistics does) has nothing to show,
 * and returning `undefined` is the correct answer.
 */
export function groupForScalar(
  groups: ViewGroup[],
  stepOutputs: string[],
  scalarPath: string,
): ViewGroup | undefined {
  const candidates = groups.filter((group) =>
    group.members.some((member) => stepOutputs.includes(member.id)),
  )
  if (candidates.length === 0) return undefined

  const token = /(ndvi|ndwi|ndbi|sigma0|vv|vh|swir|change)/i.exec(scalarPath)?.[1]
  if (token) {
    const matched = candidates.find((group) =>
      group.members.some((member) => new RegExp(token, 'i').test(member.label)),
    )
    if (matched) return matched
  }
  return candidates[0]
}
