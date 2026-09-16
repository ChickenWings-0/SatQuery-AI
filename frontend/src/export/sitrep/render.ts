/**
 * The page. This is the only module that imports pdf-lib, and it is reached
 * only through `import()` from `useSitrep` — the console entry never pays
 * for it. A4 portrait with 18 mm margins, which is also inside US Letter's
 * printable area, so one file prints on both without clipping. Geist and
 * Geist Mono are fetched from `public/fonts/` (same origin, offline-safe)
 * as static TTF instances cut by `scripts/gen-sitrep-fonts.py` — fontkit's
 * browser build cannot decode WOFF2 — and embedded subsetted; if the fetch
 * fails the standard Helvetica is used so a brief is never lost to a font.
 */
import fontkit from '@pdf-lib/fontkit'
import { PDFDocument, PDFFont, PDFPage, StandardFonts, rgb, type RGB } from 'pdf-lib'

import type { ScenePng } from '@/export/sitrep/scene'
import type { SitrepModel } from '@/export/sitrep/compose'

const MM = 72 / 25.4
const PAGE_W = 210 * MM
const PAGE_H = 297 * MM
const MARGIN = 18 * MM
const GUTTER = 6 * MM

const INK = rgb(0.102, 0.086, 0.075)
const INK_LO = rgb(0.42, 0.38, 0.35)
const PAPER = rgb(0.965, 0.945, 0.918)
const ACCENT = rgb(0.788, 0.416, 0.231)
const RULE = rgb(0.82, 0.78, 0.74)
const OK = rgb(0.16, 0.5, 0.33)
const WARN = rgb(0.72, 0.5, 0.12)
const FAIL = rgb(0.72, 0.22, 0.18)

interface Fonts {
  sans: PDFFont
  sansBold: PDFFont
  mono: PDFFont
}

async function fetchFont(path: string): Promise<Uint8Array | null> {
  try {
    const response = await fetch(path)
    if (!response.ok) return null
    return new Uint8Array(await response.arrayBuffer())
  } catch {
    return null
  }
}

async function loadFonts(doc: PDFDocument): Promise<Fonts> {
  doc.registerFontkit(fontkit)
  const [sans, bold, mono] = await Promise.all([
    fetchFont('/fonts/geist-sitrep-400.ttf'),
    fetchFont('/fonts/geist-sitrep-600.ttf'),
    fetchFont('/fonts/geist-mono-sitrep-400.ttf'),
  ])
  try {
    if (!sans || !bold || !mono) throw new Error('fonts unavailable')
    return {
      sans: await doc.embedFont(sans, { subset: true }),
      sansBold: await doc.embedFont(bold, { subset: true }),
      mono: await doc.embedFont(mono, { subset: true }),
    }
  } catch {
    return {
      sans: await doc.embedFont(StandardFonts.Helvetica),
      sansBold: await doc.embedFont(StandardFonts.HelveticaBold),
      mono: await doc.embedFont(StandardFonts.Courier),
    }
  }
}

/** Greedy word wrap at `width` points. */
function wrap(text: string, font: PDFFont, size: number, width: number): string[] {
  const lines: string[] = []
  for (const paragraph of text.split(/\n+/)) {
    const words = paragraph.split(/\s+/).filter(Boolean)
    let line = ''
    for (const word of words) {
      const candidate = line ? `${line} ${word}` : word
      if (textWidth(font, candidate, size) <= width) line = candidate
      else {
        if (line) lines.push(line)
        // A single token wider than the column is broken by character.
        if (textWidth(font, word, size) > width) {
          let chunk = ''
          for (const ch of word) {
            if (textWidth(font, chunk + ch, size) > width) {
              lines.push(chunk)
              chunk = ch
            } else chunk += ch
          }
          line = chunk
        } else line = word
      }
    }
    if (line) lines.push(line)
  }
  return lines
}

interface Cursor {
  page: PDFPage
  fonts: Fonts
}

const FALLBACK_GLYPHS: Record<string, string> = {
  '→': '->',
  '←': '<-',
  '⇧': 'Shift+',
  '′': "'",
  '″': '"',
  '−': '-',
  '–': '-',
  '—': '-',
  '·': '.',
  '…': '...',
  '²': '2',
}

const glyphCache = new WeakMap<PDFFont, Map<string, boolean>>()

function canDraw(font: PDFFont, ch: string): boolean {
  let cache = glyphCache.get(font)
  if (!cache) {
    cache = new Map()
    glyphCache.set(font, cache)
  }
  const known = cache.get(ch)
  if (known !== undefined) return known
  let ok = true
  try {
    font.encodeText(ch)
  } catch {
    ok = false
  }
  cache.set(ch, ok)
  return ok
}

/** Replace what the font cannot encode: a brief must never fail on an arrow. */
function fit(str: string, font: PDFFont): string {
  let out = ''
  for (const ch of str) {
    if (canDraw(font, ch)) out += ch
    else {
      const alt = FALLBACK_GLYPHS[ch] ?? '?'
      out += canDraw(font, alt[0] ?? '?') ? alt : '?'
    }
  }
  return out
}

function text(c: Cursor, raw: string, x: number, y: number, size: number, opts: { font?: PDFFont; color?: RGB; bold?: boolean } = {}) {
  const font = opts.font ?? (opts.bold ? c.fonts.sansBold : c.fonts.sans)
  const color = opts.color ?? INK
  const str = fit(raw, font)
  c.page.drawText(str, { x, y, size, font, color })
  if (opts.bold && c.fonts.sansBold === c.fonts.sans) {
    // Fake bold for the variable face: a second pass a third of a point over.
    c.page.drawText(str, { x: x + size * 0.028, y, size, font, color })
  }
}

function textWidth(font: PDFFont, str: string, size: number): number {
  return font.widthOfTextAtSize(fit(str, font), size)
}

function eyebrowRule(c: Cursor, label: string, x: number, y: number, width: number) {
  text(c, label, x, y, 7.5, { color: INK_LO, bold: true })
  c.page.drawLine({ start: { x, y: y - 3.5 }, end: { x: x + width, y: y - 3.5 }, thickness: 0.5, color: RULE })
}

/** Answer text with superscript citation numbers, wrapped to the column; returns the y after it. */
function drawAnswer(c: Cursor, model: SitrepModel, x: number, top: number, width: number, maxHeight: number): { y: number; clipped: boolean } {
  const size = 9.5
  const lead = 13.5
  const superSize = 6
  type Run = { str: string; sup?: string }
  const runs: Run[] = []
  for (const seg of model.answer.segments) {
    if (seg.kind === 'citation' && seg.index !== undefined) runs.push({ str: seg.text, sup: String(seg.index + 1) })
    else if (seg.kind === 'uncited') runs.push({ str: seg.text, sup: '!' })
    else runs.push({ str: seg.text })
  }
  // Tokenise into words that keep their superscript attached to the last word.
  const tokens: { word: string; sup?: string; space: boolean }[] = []
  for (const run of runs) {
    const parts = run.str.split(/(\s+)/)
    parts.forEach((part) => {
      if (!part) return
      if (/^\s+$/.test(part)) {
        const last = tokens[tokens.length - 1]
        if (last) last.space = true
        return
      }
      tokens.push({ word: part, space: false })
    })
    const last = tokens[tokens.length - 1]
    if (run.sup && last) last.sup = run.sup
  }
  let y = top
  let cx = x
  let clipped = false
  const maxLines = Math.floor(maxHeight / lead)
  let lines = 1
  const spaceW = textWidth(c.fonts.sans, ' ', size)
  for (const token of tokens) {
    const w = textWidth(c.fonts.sans, token.word, size) + (token.sup ? textWidth(c.fonts.mono, token.sup, superSize) + 1 : 0)
    if (cx + w > x + width && cx > x) {
      lines += 1
      if (lines > maxLines) {
        clipped = true
        break
      }
      y -= lead
      cx = x
    }
    text(c, token.word, cx, y, size)
    cx += textWidth(c.fonts.sans, token.word, size)
    if (token.sup) {
      const isUncited = token.sup === '!'
      c.page.drawText(fit(token.sup, c.fonts.mono), { x: cx + 1, y: y + 4, size: superSize, font: c.fonts.mono, color: isUncited ? WARN : ACCENT })
      cx += textWidth(c.fonts.mono, token.sup, superSize) + 1
    }
    if (token.space) cx += spaceW
  }
  if (clipped) {
    text(c, '…', cx, y, size, { color: INK_LO })
  }
  return { y: y - lead, clipped }
}

function statusColor(status: SitrepModel['header']['status']): RGB {
  return status === 'OK' ? OK : status === 'DEGRADED' ? WARN : FAIL
}

export async function renderSitrep(model: SitrepModel, scene: ScenePng | null): Promise<Uint8Array> {
  const doc = await PDFDocument.create()
  doc.setTitle(`SITREP ${model.header.sceneId}`)
  doc.setProducer('SatQuery AI')
  doc.setCreator('SatQuery AI console')
  doc.setSubject(model.query)
  const fonts = await loadFonts(doc)
  const page = doc.addPage([PAGE_W, PAGE_H])
  const c: Cursor = { page, fonts }
  const contentW = PAGE_W - 2 * MARGIN
  const leftW = 92 * MM
  const rightX = MARGIN + leftW + GUTTER
  const rightW = contentW - leftW - GUTTER

  // ── Header band ──────────────────────────────────────────────────────
  const bandH = 22 * MM
  page.drawRectangle({ x: 0, y: PAGE_H - bandH, width: PAGE_W, height: bandH, color: INK })
  let y = PAGE_H - 9 * MM
  text(c, 'SATQUERY AI', MARGIN, y, 8, { color: PAPER, bold: true })
  text(c, '·  SITREP', MARGIN + textWidth(fonts.sansBold, 'SATQUERY AI', 8) + 3, y, 8, { color: rgb(0.9, 0.78, 0.61) })
  const stamp = `${model.header.sceneId}  ·  ${model.header.generatedAt}`
  text(c, stamp, PAGE_W - MARGIN - textWidth(fonts.mono, stamp, 8), y, 8, { font: fonts.mono, color: rgb(0.9, 0.78, 0.61) })
  y -= 7.5 * MM
  const statusLabel = `Status: ${model.header.status}`
  const statusW = textWidth(fonts.sansBold, statusLabel, 8.5)
  const queryLines = wrap(`Query: ${model.query}`, fonts.sans, 9, contentW - statusW - 8 * MM)
  text(c, queryLines[0] ?? 'Query: —', MARGIN, y, 9, { color: PAPER })
  if (queryLines.length > 1) text(c, `${queryLines[1]!.slice(0, 80)}…`, MARGIN, y - 11, 8, { color: rgb(0.8, 0.74, 0.68) })
  page.drawCircle({ x: PAGE_W - MARGIN - statusW - 4 * MM, y: y + 2.5, size: 2.2, color: statusColor(model.header.status) })
  text(c, statusLabel, PAGE_W - MARGIN - statusW, y, 8.5, { color: PAPER, bold: true })

  // ── Body columns ─────────────────────────────────────────────────────
  const bodyTop = PAGE_H - bandH - 8 * MM
  const footerFloor = MARGIN + 30 * MM
  const bodyH = bodyTop - footerFloor - 6 * MM

  // Scene panel, left.
  eyebrowRule(c, 'SCENE', MARGIN, bodyTop, leftW)
  const sceneTop = bodyTop - 6 * MM
  let sceneBottom = sceneTop
  if (scene) {
    const png = await doc.embedPng(scene.bytes)
    const ratio = scene.height / scene.width
    const h = Math.min(leftW * ratio, bodyH - 10 * MM)
    const drawW = h / ratio
    page.drawImage(png, { x: MARGIN, y: sceneTop - h, width: drawW, height: h })
    page.drawRectangle({ x: MARGIN, y: sceneTop - h, width: drawW, height: h, borderColor: RULE, borderWidth: 0.5 })
    sceneBottom = sceneTop - h
  } else {
    const h = leftW * 0.72
    page.drawRectangle({ x: MARGIN, y: sceneTop - h, width: leftW, height: h, color: rgb(0.93, 0.91, 0.88), borderColor: RULE, borderWidth: 0.5 })
    text(c, 'No rendered view for this run.', MARGIN + 4 * MM, sceneTop - h / 2, 8.5, { color: INK_LO })
    sceneBottom = sceneTop - h
  }
  const modeLabel = model.scene.mode === 'pair' ? 'Bi-temporal pair' : model.scene.mode === 'cross-modal' ? 'Optical + SAR' : 'Single scene'
  const boxesLabel = model.scene.boxes.length > 0 ? `  ·  ${model.scene.boxes.length} box${model.scene.boxes.length === 1 ? '' : 'es'} burned in` : ''
  const paneLabels = model.scene.panes.map((p) => p.label).join('  |  ')
  text(c, `${modeLabel}${boxesLabel}`, MARGIN, sceneBottom - 4 * MM, 7.5, { color: INK_LO })
  if (paneLabels) text(c, paneLabels, MARGIN, sceneBottom - 4 * MM - 9, 7, { font: fonts.mono, color: INK_LO })
  sceneBottom -= 8 * MM

  // Answer, right.
  eyebrowRule(c, 'ANSWER', rightX, bodyTop, rightW)
  let ry = bodyTop - 6 * MM
  const rowH = 7 * MM
  const measurementsH = (Math.min(model.measurements.length, 5) + (model.confidence !== null ? 1 : 0)) * rowH + 12 * MM
  const answerMax = bodyH - measurementsH - 8 * MM
  if (model.answer.empty) {
    text(c, 'The run finished without producing an answer. The measurements below are still real.', rightX, ry, 9, { color: INK_LO })
    ry -= 13.5
  } else {
    const drawn = drawAnswer(c, model, rightX, ry, rightW, answerMax)
    ry = drawn.y
    if (drawn.clipped) {
      text(c, `Full answer: /report/${model.header.traceId}`, rightX, ry, 7, { font: fonts.mono, color: INK_LO })
      ry -= 10
    }
  }

  // Key measurements: label and its fact_sheet key on the left, the value
  // in tabular mono on the right — the same rows the KPI cards show.
  ry -= 4 * MM
  eyebrowRule(c, 'KEY MEASUREMENTS', rightX, ry, rightW)
  ry -= 6.5 * MM
  if (model.measurements.length === 0) {
    text(c, 'No headline scalars in this run.', rightX, ry, 8.5, { color: INK_LO })
    ry -= rowH
  }
  const drawRow = (label: string, source: string | null, value: string) => {
    text(c, label, rightX, ry, 8.5)
    if (source) text(c, source, rightX, ry - 8, 5.5, { font: fonts.mono, color: INK_LO })
    const vw = textWidth(fonts.mono, value, 9.5)
    text(c, value, rightX + rightW - vw, ry, 9.5, { font: fonts.mono })
    page.drawLine({ start: { x: rightX, y: ry - 11.5 }, end: { x: rightX + rightW, y: ry - 11.5 }, thickness: 0.3, color: RULE })
    ry -= rowH
  }
  for (const m of model.measurements.slice(0, 5)) drawRow(m.label, m.source, m.unit ? `${m.value} ${m.unit}` : m.value)
  if (model.confidence !== null) drawRow('Confidence', 'confidence.overall', model.confidence.toFixed(2))

  // The footer sits under whichever column ran longer, never below the floor.
  const footerTop = Math.max(footerFloor, Math.min(sceneBottom, ry) - 4 * MM)

  // ── Footer band ──────────────────────────────────────────────────────
  page.drawLine({ start: { x: MARGIN, y: footerTop }, end: { x: PAGE_W - MARGIN, y: footerTop }, thickness: 0.8, color: INK })
  let fy = footerTop - 6 * MM
  text(c, 'TOOL CHAIN', MARGIN, fy, 7.5, { color: INK_LO, bold: true })
  const chainX = MARGIN + 24 * MM
  const chain =
    model.toolChain.length > 0
      ? [
          'parse',
          ...(model.checks ? [`checks ${model.checks.passed}/${model.checks.total}`] : []),
          ...model.toolChain.map((s) => `${s.tool}@${s.version.split('+')[0]}${s.state === 'OK' ? '' : ` (${s.state.toLowerCase()})`}`),
        ].join('  →  ')
      : 'not available — the trace is no longer on the server'
  const chainLines = wrap(chain, fonts.mono, 7.5, contentW - 24 * MM).slice(0, 2)
  for (const line of chainLines) {
    text(c, line, chainX, fy, 7.5, { font: fonts.mono })
    fy -= 10
  }
  fy -= 4 * MM
  text(c, 'CITATIONS', MARGIN, fy, 7.5, { color: INK_LO, bold: true })
  const cited = `${model.citations.bound} bound  ·  ${model.citations.uncited} uncited`
  text(c, cited, chainX, fy, 8, { font: fonts.mono })
  if (model.citations.uncited > 0) {
    const badge = `${model.citations.uncited} UNCITED`
    const bw = textWidth(fonts.sansBold, badge, 7) + 8
    const bx = chainX + textWidth(fonts.mono, cited, 8) + 4 * MM
    page.drawRectangle({ x: bx, y: fy - 3, width: bw, height: 12, color: WARN, borderWidth: 0 })
    text(c, badge, bx + 4, fy, 7, { color: PAPER, bold: true })
  }
  const traceLine = `trace ${model.header.traceId}  ·  schema v${model.header.version}`
  text(c, traceLine, PAGE_W - MARGIN - textWidth(fonts.mono, traceLine, 7), fy, 7, { font: fonts.mono, color: INK_LO })

  // Footnotes: what could not be included, said out loud.
  if (model.notes.length > 0) {
    let ny = MARGIN + 2 * MM
    for (const note of model.notes.slice(0, 2)) {
      text(c, note, MARGIN, ny, 6.5, { color: INK_LO })
      ny += 8
    }
  }

  return doc.save()
}
