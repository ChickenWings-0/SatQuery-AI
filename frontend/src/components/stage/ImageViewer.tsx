/**
 * F4a — the large viewer at the top of the Data Stage.
 *
 * For a bi-temporal pair the A/B swipe is the money shot: one handle dragged
 * across the same footprint, before on the left, after on the right. When only
 * one image exists the same panel shows it plain, so the layout does not jump
 * between run types.
 *
 * Over the image, four pieces of chrome, each absolutely positioned and each
 * outside the zoom transform so it stays put while the raster moves:
 *
 *   top-left      T1 / T2 date pills, read out of the artifact labels
 *   top-right     zoom in / out / reset, driven through `useControls`
 *   bottom-left   a scale bar, derived from the manifest's GSD and the live
 *                 zoom — or absent, when the scale is unknown
 *   bottom-right  the feature legend, on change tasks only
 *
 * And one layer *inside* the transform: the bounding boxes the answer named.
 * The raster is `object-fit: contain`, so it is letterboxed inside its cell
 * whenever the cell's aspect differs from the image's; the box layer is sized
 * to the raster's rendered rectangle — computed from its natural size and the
 * cell's — rather than to the cell, or every box would land off its target on
 * a wide window.
 *
 * Two sizing rules earn their keep. The stage is a square-ish flex child with
 * `min-h-0`, so it can actually shrink inside the column — without that the
 * panel overflows its grid cell and the section beneath it gets painted over.
 * And the image is sized to *fill* that stage rather than sitting at its
 * intrinsic 256 px. `image-rendering: pixelated` is deliberate: these are
 * 256 px rasters blown up, and smoothing them invents detail the sensor never
 * saw.
 *
 * The zoom/pan transform is intentionally *shared across view switches* — the
 * key on `TransformWrapper` never changes when the active view does — so
 * flipping true colour → NDVI keeps the user's framing instead of resetting it.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider'
import {
  TransformComponent,
  TransformWrapper,
  useControls,
  useTransformEffect,
} from 'react-zoom-pan-pinch'

import { artifactUrl } from '@/api/client'
import { BboxOverlay } from '@/components/stage/BboxOverlay'
import { DotIcon, ResetIcon, ZoomInIcon, ZoomOutIcon } from '@/components/ui/icons'
import { scaleBar, sceneMeta } from '@/evidence/scene'
import { FIXED_DOMAIN, primaryOf, type ViewGroup } from '@/evidence/views'
import { useFocusStore } from '@/state/focus'
import { useJobStore } from '@/state/job'
import { useUiStore } from '@/state/ui'
import { boxesForResult } from '@/thread/boxes'
import { changeLegend } from '@/thread/legend'

/**
 * Index and SAR views are rendered on a fixed domain by the server, so the
 * legend is a constant. Reading it from `FIXED_DOMAIN` rather than the pixels
 * is what makes the colours *absolute* rather than per-image relative.
 */
function Legend({ group }: { group: ViewGroup }) {
  const domain = FIXED_DOMAIN[group.code]
  if (!domain) return null
  return (
    <div className="tabular flex items-center gap-2 font-mono text-[11px] text-text-lo">
      <span>
        {domain.min}
        {domain.unit ?? ''}
      </span>
      <span
        className="h-1.5 w-20 rounded-full ring-1 ring-line"
        style={{ background: 'linear-gradient(90deg,#100c0a,#ba704f,#dfa878,#f0e6df)' }}
      />
      <span>
        {domain.max}
        {domain.unit ?? ''}
      </span>
    </div>
  )
}

const FILL = 'h-full w-full [image-rendering:pixelated]'

/**
 * `ReactCompareSliderImage` writes `object-fit` as an inline style, which beats
 * any utility class — so the fit has to be passed as a style too or a 256 px
 * square gets stretched across an 880 px panel.
 */
const CONTAIN = { objectFit: 'contain' } as const

const PILL =
  'pointer-events-auto rounded-full border border-line bg-bg-main/85 backdrop-blur-sm'

/** A round control over the raster. Dark pill, terracotta on hover. */
function ViewerButton({
  label,
  onClick,
  children,
}: {
  label: string
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className={`${PILL} grid size-8 place-items-center text-text-hi transition-colors hover:border-accent-warm hover:text-accent-warm-text`}
    >
      {children}
    </button>
  )
}

/** Must render inside `TransformWrapper`: `useControls` reads its context. */
function ZoomControls() {
  const { zoomIn, zoomOut, resetTransform } = useControls()
  return (
    <div className="pointer-events-none absolute top-3 right-3 z-20 flex flex-col gap-1.5">
      <ViewerButton label="Zoom in" onClick={() => zoomIn()}>
        <ZoomInIcon size={16} />
      </ViewerButton>
      <ViewerButton label="Zoom out" onClick={() => zoomOut()}>
        <ZoomOutIcon size={16} />
      </ViewerButton>
      <ViewerButton label="Reset view" onClick={() => resetTransform()}>
        <ResetIcon size={16} />
      </ViewerButton>
      {/* North is up: the renderer writes north-up rasters, so the compass is
          a constant and a reminder, not a reading. */}
      <span
        aria-hidden
        className={`${PILL} grid size-8 place-items-center font-mono text-[10px] font-semibold text-text-lo`}
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1" opacity="0.5" />
          <path d="M8 2.5 10 8H6l2-5.5Z" fill="var(--color-accent-warm)" />
          <path d="M8 13.5 6 8h4l-2 5.5Z" fill="currentColor" opacity="0.5" />
        </svg>
      </span>
    </div>
  )
}

/**
 * The scale bar. Inside `TransformWrapper` for the live zoom; positioned in
 * the viewer, not the transform, so it never moves with the raster. Hidden,
 * rather than faked, when the scene's ground sample distance is unknown.
 */
function ScaleBar({ pxPerKmAtRest }: { pxPerKmAtRest: number | null }) {
  const [scale, setScale] = useState(1)
  useTransformEffect(({ state }) => {
    setScale(state.scale)
  })
  const bar = scaleBar(pxPerKmAtRest ? pxPerKmAtRest * scale : null, 160)
  if (!bar) return null
  return (
    <div
      aria-label={`Scale: ${bar.label}`}
      className={`${PILL} tabular absolute bottom-3 left-3 z-20 flex items-center gap-2 px-2.5 py-1.5 font-mono text-[10px] text-text-hi`}
    >
      <span
        aria-hidden
        className="block h-1.5 border-x-2 border-b-2 border-text-hi"
        style={{ width: `${Math.round(bar.px)}px` }}
      />
      <span>{bar.label}</span>
    </div>
  )
}

/** The rectangle `object-fit: contain` gives an image inside a cell. */
function containBox(
  cell: { width: number; height: number },
  natural: { width: number; height: number } | null,
): { left: number; top: number; width: number; height: number } | null {
  if (!natural || cell.width === 0 || cell.height === 0) return null
  const scale = Math.min(cell.width / natural.width, cell.height / natural.height)
  const width = natural.width * scale
  const height = natural.height * scale
  return { left: (cell.width - width) / 2, top: (cell.height - height) / 2, width, height }
}

export function ImageViewer({ group }: { group: ViewGroup }) {
  const swipe = useFocusStore((state) => state.swipe)
  const swipeEpoch = useFocusStore((state) => state.swipeEpoch)
  const setSwipe = useFocusStore((state) => state.setSwipe)
  const result = useJobStore((state) => state.result)
  const validation = useUiStore((state) => state.validation)

  const preUrl = group.pre ? artifactUrl(group.pre) : null
  const postUrl = group.post ? artifactUrl(group.post) : null
  const comparable = group.comparable && preUrl && postUrl
  const primary = primaryOf(group)
  const primaryUrl = primary ? artifactUrl(primary) : null

  const meta = useMemo(() => sceneMeta(validation, group), [validation, group])
  const task = result?.resolved_task.primary
  const isChange = task?.startsWith('CHANGE_') ?? false

  // The boxes to draw: the grounding tool's BBOX_SET artifact when one ran,
  // the answer text only as a fallback (see @/thread/boxes). Once per answer.
  const artifacts = useJobStore((state) => state.artifacts)
  const resolved = useMemo(() => boxesForResult(result, artifacts), [result, artifacts])
  const boxes = resolved.boxes

  // The raster's rendered rectangle inside its cell, for the box layer.
  const cellRef = useRef<HTMLDivElement>(null)
  const [cell, setCell] = useState({ width: 0, height: 0 })
  const [natural, setNatural] = useState<{ width: number; height: number } | null>(null)
  useEffect(() => {
    const element = cellRef.current
    if (!element || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setCell({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  const fit = containBox(cell, natural)
  const pxPerKmAtRest = fit && meta.widthKm ? fit.width / meta.widthKm : null

  function onImageLoad(event: React.SyntheticEvent<HTMLImageElement>) {
    const { naturalWidth, naturalHeight } = event.currentTarget
    if (naturalWidth > 0 && naturalHeight > 0) {
      setNatural((prev) =>
        prev && prev.width === naturalWidth && prev.height === naturalHeight
          ? prev
          : { width: naturalWidth, height: naturalHeight },
      )
    }
  }

  return (
    <section className="card-flush flex h-full min-h-0 flex-col">
      <header className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-line-soft px-5 py-3">
        <h2 className="t-panel">{group.name}</h2>
        {comparable && (
          <span className="chip bg-accent-cool text-on-accent-cool">pre → post</span>
        )}
        <div className="ml-auto flex items-center gap-4">
          <Legend group={group} />
          {comparable && (
            <span className="tabular font-mono text-[11px] text-text-lo">{swipe}%</span>
          )}
        </div>
      </header>

      {/* The checkered ground reads as "no data here", so letterboxing around a
          non-square raster looks intentional rather than broken. */}
      <div
        className="relative min-h-0 flex-1"
        style={{
          backgroundColor: '#150e0b',
          backgroundImage:
            'linear-gradient(45deg,#1a1210 25%,transparent 25%,transparent 75%,#1a1210 75%),linear-gradient(45deg,#1a1210 25%,transparent 25%,transparent 75%,#1a1210 75%)',
          backgroundSize: '16px 16px',
          backgroundPosition: '0 0, 8px 8px',
        }}
      >
        <TransformWrapper
          minScale={1}
          maxScale={8}
          doubleClick={{ mode: 'reset' }}
          // The slider handle must receive its own drag; without this the pan
          // layer swallows the pointer and the swipe never moves.
          panning={{ excluded: ['__rcs-handle-root', 'input'] }}
          wheel={{ step: 0.15 }}
        >
          <TransformComponent
            wrapperClass="!h-full !w-full"
            contentClass="!h-full !w-full"
          >
            <div ref={cellRef} className="relative h-full w-full">
              {comparable ? (
                /* v4 is uncontrolled: `defaultPosition` seeds it and
                   `onPositionChange` reports back. `onlyHandleDraggable` keeps
                   the swipe and the pan from fighting — drag the handle to
                   compare, drag anywhere else to pan. The handle is focusable,
                   so arrow keys move it too. */
                <ReactCompareSlider
                  // Remounting on `swipeEpoch` is how `[`/`]` move the handle:
                  // the component is uncontrolled, so a new key re-seeds it at
                  // the store's position. A drag leaves the epoch alone.
                  key={`${group.key}:${swipeEpoch}`}
                  defaultPosition={swipe}
                  onlyHandleDraggable
                  onPositionChange={(position) => setSwipe(Math.round(position))}
                  className="h-full w-full"
                  itemOne={
                    <ReactCompareSliderImage
                      src={preUrl}
                      alt={group.pre?.label ?? 'pre'}
                      className={FILL}
                      style={CONTAIN}
                      onLoad={onImageLoad}
                    />
                  }
                  itemTwo={
                    <ReactCompareSliderImage
                      src={postUrl}
                      alt={group.post?.label ?? 'post'}
                      className={FILL}
                      style={CONTAIN}
                    />
                  }
                />
              ) : primaryUrl ? (
                <img
                  src={primaryUrl}
                  alt={primary?.label ?? group.name}
                  className={FILL}
                  style={CONTAIN}
                  onLoad={onImageLoad}
                />
              ) : null}

              {/* Sized to the raster, not the cell — see the header comment. */}
              {fit && boxes.length > 0 && (
                <div
                  className="pointer-events-none absolute"
                  style={{
                    left: fit.left,
                    top: fit.top,
                    width: fit.width,
                    height: fit.height,
                  }}
                >
                  <BboxOverlay boxes={boxes} />
                </div>
              )}
              {/* Drawn from prose rather than from the grounding artifact:
                  worth saying, because it is the one case where the map and
                  the trace can disagree. */}
              {fit && resolved.source === 'text' && (
                <span
                  data-testid="bbox-source-hint"
                  className={`${PILL} absolute top-3 left-3 z-20 px-2 py-0.5 text-[10px] text-warn`}
                >
                  boxes read from the answer text
                </span>
              )}
            </div>
          </TransformComponent>

          <ZoomControls />
          <ScaleBar pxPerKmAtRest={pxPerKmAtRest} />
        </TransformWrapper>

        {/* The dates, from the labels the VLM was shown. On a single image the
            one pill sits alone; on a pair the two flank the handle's home. */}
        <div className="pointer-events-none absolute top-3 left-3 z-20 flex flex-wrap gap-1.5">
          {(meta.t1 || comparable) && (
            <span
              className={`${PILL} tabular flex items-center gap-1.5 px-2.5 py-1 font-mono text-[11px] text-text-hi`}
            >
              <span className="font-semibold text-accent-warm-text">T1</span>
              <span>{meta.t1 ?? '—'}</span>
            </span>
          )}
          {comparable && (
            <span
              className={`${PILL} tabular flex items-center gap-1.5 px-2.5 py-1 font-mono text-[11px] text-text-hi`}
            >
              <span className="font-semibold text-accent-warm-text">T2</span>
              <span>{meta.t2 ?? '—'}</span>
            </span>
          )}
        </div>

        {/* The feature legend. Only meaningful on a change task, where the
            overlay's colour has a fixed meaning; anywhere else it would be a
            label with nothing to point at. */}
        {isChange && (
          <div
            className={`${PILL} absolute right-3 bottom-3 z-20 flex items-center gap-1.5 px-2.5 py-1 text-[11px] text-text-hi`}
          >
            <DotIcon className="text-accent-warm" />
            <span>{changeLegend(result?.resolved_task)}</span>
          </div>
        )}
      </div>

      <footer className="flex shrink-0 items-center gap-4 border-t border-line-soft px-5 py-2.5">
        {/* The exact label grammar from view_labels.py — the string the VLM was
            shown, which is the point of displaying it. */}
        <p className="truncate text-[11px] text-text-lo" title={primary?.label}>
          {comparable ? group.pre?.label : primary?.label}
        </p>
        <p className="ml-auto flex shrink-0 items-center gap-2 text-[11px] text-text-lo">
          {comparable && (
            <>
              <kbd className="kbd">[</kbd>
              <kbd className="kbd">]</kbd>
              <span>swipe</span>
            </>
          )}
          <kbd className="kbd">←</kbd>
          <kbd className="kbd">→</kbd>
          <span>evidence</span>
        </p>
      </footer>
    </section>
  )
}
