/**
 * The map draws from the grounding artifact, not from the prose.
 *
 * `text_grounding.py` writes a `BBOX_SET` artifact with every box in pixels
 * and in the 0–1000 frame; the frontend used to ignore it and re-parse the
 * answer text. The resolver prefers the artifact and falls back to text only
 * when no usable artifact exists.
 */
import { describe, expect, it } from 'vitest'

import type { AnalyzeResponse, ArtifactRef } from '@/api/types'
import { boxesForAnswer, boxesForResult, isBboxSetInline, lastBboxSet } from '@/thread/boxes'
import { changeLegend, humaniseClass } from '@/thread/legend'

/** A `BBOX_SET` artifact as `text_grounding.py` writes it, with the given boxes. */
function bboxSet(
  id: string,
  boxes: Array<[number, number, number, number]>,
  sourceImage = 'img_0',
): ArtifactRef {
  return {
    id,
    type: 'BBOX_SET',
    mime: 'application/json',
    label: 'Grounded boxes',
    url: null,
    produced_by_step: 3,
    inline: {
      boxes: boxes.map((box, index) => ({
        id: `box_${index}`,
        label: 'aircraft',
        score: 0.9,
        bbox_px: box.map((n) => n * 2) as [number, number, number, number],
        bbox_normalised: box,
        bbox_wgs84: null,
      })),
      frame: { width: 2000, height: 2000 },
      source_image: sourceImage,
    },
  } as unknown as ArtifactRef
}

const TEXT_WITH_BOXES = 'Two aircraft: aircraft(100,100),(200,200) aircraft(300,300),(400,400).'

function resultWith(text: string): AnalyzeResponse {
  return { answer: { text } } as unknown as AnalyzeResponse
}

describe('boxesForAnswer', () => {
  it('draws from the artifact even when the text parses to a different set', () => {
    const resolved = boxesForAnswer(TEXT_WITH_BOXES, [bboxSet('art_1', [[10, 20, 30, 40]])])
    expect(resolved.source).toBe('artifact')
    expect(resolved.boxes).toEqual([
      { xMin: 10, yMin: 20, xMax: 30, yMax: 40, label: 'aircraft', score: 0.9 },
    ])
    expect(resolved.sourceImage).toBe('img_0')
  })

  it('falls back to the answer text when no artifact exists, and says so', () => {
    const resolved = boxesForAnswer(TEXT_WITH_BOXES, [])
    expect(resolved.source).toBe('text')
    expect(resolved.boxes).toHaveLength(2)
    expect(resolved.sourceImage).toBeNull()
  })

  it('falls back to text on a malformed artifact rather than throwing', () => {
    const broken = {
      ...bboxSet('art_1', [[1, 2, 3, 4]]),
      inline: { boxes: [{ id: 'x', bbox_normalised: 'nope' }], frame: {} },
    } as unknown as ArtifactRef
    expect(isBboxSetInline(broken.inline)).toBe(false)
    expect(boxesForAnswer(TEXT_WITH_BOXES, [broken]).source).toBe('text')
  })

  it('reports none when there is neither', () => {
    expect(boxesForAnswer('No objects here.', [])).toEqual({
      boxes: [],
      source: 'none',
      sourceImage: null,
    })
    expect(boxesForAnswer(null, [])).toMatchObject({ source: 'none' })
  })

  it('uses the last BBOX_SET in the run, like the aggregator uses the last synthesiser', () => {
    const first = bboxSet('art_1', [[1, 1, 5, 5]])
    const second = bboxSet('art_2', [[7, 7, 9, 9]], 'img_1')
    expect(lastBboxSet([first, second])?.id).toBe('art_2')
    const resolved = boxesForAnswer('', [first, second])
    expect(resolved.boxes[0]?.xMin).toBe(7)
    expect(resolved.sourceImage).toBe('img_1')
  })

  it('rights an inverted box, clamps to the frame and drops an empty one', () => {
    const resolved = boxesForAnswer('', [
      bboxSet('art_1', [
        [300, 400, 100, 200],
        [-50, 0, 1500, 10],
        [40, 40, 40, 90],
      ]),
    ])
    expect(resolved.boxes).toEqual([
      { xMin: 100, yMin: 200, xMax: 300, yMax: 400, label: 'aircraft', score: 0.9 },
      { xMin: 0, yMin: 0, xMax: 1000, yMax: 10, label: 'aircraft', score: 0.9 },
    ])
  })
})

describe('boxesForResult', () => {
  it("reads the streamed artifacts first and the result's own list second", () => {
    const set = bboxSet('art_1', [[1, 1, 5, 5]])
    const result = { ...resultWith(''), artifacts: [set] } as AnalyzeResponse
    expect(boxesForResult(result, []).source).toBe('artifact')
    expect(boxesForResult(resultWith(TEXT_WITH_BOXES), []).source).toBe('text')
    expect(boxesForResult(null, []).source).toBe('none')
  })
})

describe('changeLegend', () => {
  it('follows target_class instead of naming built-up area on every change task', () => {
    const task = (slots: Record<string, unknown>) =>
      ({ primary: 'CHANGE_VQA', secondary: [], slots, confidence: 0.9, classifier: 'rules_v1' }) as never
    expect(changeLegend(task({ target_class: 'water' }))).toBe('New water')
    expect(changeLegend(task({ target_class: 'built_up' }))).toBe('New built-up area')
    expect(changeLegend(task({ target_class: 'solar farm' }))).toBe('New solar farm')
    expect(changeLegend(task({ target_class: null }))).toBe('Detected change')
    expect(changeLegend(task({}))).toBe('Detected change')
    expect(changeLegend(null)).toBe('Detected change')
  })

  it('humanises vocabulary keys', () => {
    expect(humaniseClass('bare_soil')).toBe('bare soil')
    expect(humaniseClass('some_other_thing')).toBe('some other thing')
  })
})
