/**
 * The scene panel of the SITREP, drawn on an offscreen canvas: one or two
 * views side by side, the answer's boxes burned in, a north arrow, a scale
 * bar from the manifest's GSD, and the bounds as a strip along the bottom.
 * Same-origin images only (`/v1/artifacts/...`), so the canvas is never
 * tainted — the invariant `pages/saved/capture.ts` already relies on.
 * No pdf-lib here: this file is small and static.
 */
import { formatDms } from '@/evidence/georef'
import type { SitrepModel } from '@/export/sitrep/compose'
import { BOX_SCALE } from '@/thread/bbox'

export interface ScenePng {
  bytes: Uint8Array
  width: number
  height: number
}

const PANE_W = 720
const PANE_H = 720
const GAP = 12
const STRIP_H = 36
const INK = '#1a1613'
const PAPER = '#f6f1ea'
const ACCENT = '#c96a3b'
const SAND = '#e6c79c'

async function decode(url: string): Promise<HTMLImageElement | null> {
  try {
    const img = new Image()
    img.decoding = 'async'
    img.src = url
    await img.decode()
    return img
  } catch {
    return null
  }
}

function roundedRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

/** A round scale-bar length in metres that fits inside `maxPx` pixels. */
export function scaleBarMetres(metresPerPx: number, maxPx: number): number {
  const maxM = metresPerPx * maxPx
  const steps = [1, 2, 5]
  let best = 1
  for (let mag = 1; mag <= 1e6; mag *= 10) {
    for (const s of steps) {
      if (s * mag <= maxM) best = s * mag
    }
  }
  return best
}

export async function drawScenePanel(
  model: SitrepModel,
  resolveUrl: (artifactId: string) => string | null,
): Promise<ScenePng | null> {
  if (typeof document === 'undefined') return null
  const panes = model.scene.panes.slice(0, 2)
  if (panes.length === 0) return null
  const images = await Promise.all(panes.map((pane) => (pane.artifactId ? decode(resolveUrl(pane.artifactId) ?? '') : null)))

  const width = panes.length * PANE_W + (panes.length - 1) * GAP
  const height = PANE_H + STRIP_H
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  ctx.fillStyle = PAPER
  ctx.fillRect(0, 0, width, height)

  panes.forEach((pane, i) => {
    const x0 = i * (PANE_W + GAP)
    const img = images[i]
    ctx.save()
    roundedRect(ctx, x0, 0, PANE_W, PANE_H, 6)
    ctx.clip()
    if (img) {
      // Cover-fit: the rendered views are square, but a non-square input
      // should not be stretched into one.
      const s = Math.max(PANE_W / img.naturalWidth, PANE_H / img.naturalHeight)
      const dw = img.naturalWidth * s
      const dh = img.naturalHeight * s
      ctx.drawImage(img, x0 + (PANE_W - dw) / 2, (PANE_H - dh) / 2, dw, dh)
    } else {
      ctx.fillStyle = '#2a2420'
      ctx.fillRect(x0, 0, PANE_W, PANE_H)
      ctx.fillStyle = SAND
      ctx.font = '500 18px Geist, Inter, system-ui, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('view unavailable', x0 + PANE_W / 2, PANE_H / 2)
      ctx.textAlign = 'left'
    }
    // Boxes: the normalised 0–1000 frame maps onto the pane.
    const sx = PANE_W / BOX_SCALE
    const sy = PANE_H / BOX_SCALE
    for (const box of model.scene.boxes) {
      const bx = x0 + box.xMin * sx
      const by = box.yMin * sy
      const bw = (box.xMax - box.xMin) * sx
      const bh = (box.yMax - box.yMin) * sy
      ctx.fillStyle = 'rgba(201,106,59,0.14)'
      ctx.fillRect(bx, by, bw, bh)
      ctx.lineWidth = 2.5
      ctx.strokeStyle = ACCENT
      ctx.strokeRect(bx, by, bw, bh)
      if (box.label) {
        ctx.font = '600 15px "Geist Mono", ui-monospace, monospace'
        const tw = ctx.measureText(box.label).width + 10
        const ly = by < 26 ? by + bh + 4 : by - 22
        ctx.fillStyle = 'rgba(26,22,19,0.85)'
        roundedRect(ctx, bx, ly, tw, 20, 3)
        ctx.fill()
        ctx.fillStyle = SAND
        ctx.fillText(box.label, bx + 5, ly + 15)
      }
    }
    // Pane label, top-left.
    ctx.font = '600 16px Geist, Inter, system-ui, sans-serif'
    const lw = ctx.measureText(pane.label).width + 14
    ctx.fillStyle = 'rgba(26,22,19,0.82)'
    roundedRect(ctx, x0 + 10, 10, lw, 26, 4)
    ctx.fill()
    ctx.fillStyle = PAPER
    ctx.fillText(pane.label, x0 + 17, 28)
    ctx.restore()
  })

  // North arrow, top-right of the last pane. WGS84-bounded rasters are
  // north-up by construction; a pixel-space scene gets no arrow, because
  // nothing in the manifest says which way is up.
  const last = (panes.length - 1) * (PANE_W + GAP)
  if (model.scene.bounds) {
    const ax = last + PANE_W - 34
    const ay = 46
    ctx.fillStyle = 'rgba(26,22,19,0.82)'
    ctx.beginPath()
    ctx.arc(ax, ay, 22, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = PAPER
    ctx.beginPath()
    ctx.moveTo(ax, ay - 14)
    ctx.lineTo(ax + 7, ay + 8)
    ctx.lineTo(ax, ay + 3)
    ctx.lineTo(ax - 7, ay + 8)
    ctx.closePath()
    ctx.fill()
    ctx.font = '700 11px Geist, Inter, system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('N', ax, ay + 19)
    ctx.textAlign = 'left'
  }

  // Scale bar, bottom-left of the first pane.
  if (model.scene.gsdM && model.scene.widthPx) {
    // The rendered view covers the whole raster, so metres per pane pixel
    // is the GSD times the raster's width, spread over the pane.
    const metresPerPx = (model.scene.gsdM * model.scene.widthPx) / PANE_W
    const metres = scaleBarMetres(metresPerPx, PANE_W * 0.3)
    const px = metres / metresPerPx
    const bx = 16
    const by = PANE_H - 26
    ctx.fillStyle = 'rgba(26,22,19,0.82)'
    roundedRect(ctx, bx - 6, by - 20, px + 12, 32, 4)
    ctx.fill()
    ctx.fillStyle = PAPER
    ctx.fillRect(bx, by, px, 4)
    ctx.fillRect(bx, by - 4, 2, 8)
    ctx.fillRect(bx + px - 2, by - 4, 2, 8)
    ctx.font = '600 12px "Geist Mono", ui-monospace, monospace'
    ctx.fillText(metres >= 1000 ? `${metres / 1000} km` : `${metres} m`, bx, by - 7)
  }

  // Bounds strip.
  ctx.fillStyle = INK
  ctx.fillRect(0, PANE_H, width, STRIP_H)
  ctx.fillStyle = SAND
  ctx.font = '500 14px "Geist Mono", ui-monospace, monospace'
  const strip = model.scene.bounds
    ? `${formatDms(model.scene.bounds[0], model.scene.bounds[3])}  →  ${formatDms(model.scene.bounds[2], model.scene.bounds[1])}   ${model.scene.crs ?? ''}`
    : 'Not georeferenced — boxes are in the image frame.'
  ctx.fillText(strip, 12, PANE_H + 23)
  if (model.scene.sensor) {
    ctx.textAlign = 'right'
    ctx.fillText(model.scene.sensor, width - 12, PANE_H + 23)
    ctx.textAlign = 'left'
  }

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'))
  if (!blob) return null
  return { bytes: new Uint8Array(await blob.arrayBuffer()), width, height }
}
