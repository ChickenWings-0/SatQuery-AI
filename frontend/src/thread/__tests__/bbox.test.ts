/**
 * The bbox parser, held to the backend's behaviour.
 *
 * Each case mirrors a rule in `box_format.py`: if the two ever disagree, the
 * map draws something the grounding tool did not parse — or nothing where it
 * did — and a grounding answer is wrong in a way no one reading the chat can
 * see.
 */
import { describe, expect, it } from 'vitest'

import { BOX_SCALE, parseBboxTokens, stripBboxTokens } from '@/thread/bbox'

const tagged = (label: string | null, body: string) =>
  `${label ? `<|object_ref_start|>${label}<|object_ref_end|>` : ''}<|box_start|>${body}<|box_end|>`

describe('the canonical tagged form', () => {
  it('parses a labelled box', () => {
    const boxes = parseBboxTokens(`Found one: ${tagged('aircraft', '(112,340),(288,512)')}.`)
    expect(boxes).toEqual([
      { xMin: 112, yMin: 340, xMax: 288, yMax: 512, label: 'aircraft', score: null },
    ])
  })

  it('parses an unlabelled box and tolerates whitespace inside the body', () => {
    const boxes = parseBboxTokens(tagged(null, '( 10 , 20 ) , ( 30 , 40 )'))
    expect(boxes).toEqual([{ xMin: 10, yMin: 20, xMax: 30, yMax: 40, label: null, score: null }])
  })

  it('keeps emission order and drops exact duplicates', () => {
    const a = tagged('a', '(0,0),(10,10)')
    const b = tagged('b', '(20,20),(30,30)')
    expect(parseBboxTokens(`${a}${b}${a}`).map((box) => box.label)).toEqual(['a', 'b'])
  })
})

describe('coordinate hygiene, matching _build', () => {
  it('clamps every coordinate into [0, BOX_SCALE]', () => {
    const [box] = parseBboxTokens(tagged(null, '(-50,-1),(1200,1000)'))
    expect(box).toMatchObject({ xMin: 0, yMin: 0, xMax: BOX_SCALE, yMax: BOX_SCALE })
  })

  it('puts an inverted box the right way round', () => {
    const [box] = parseBboxTokens(tagged(null, '(300,400),(100,200)'))
    expect(box).toMatchObject({ xMin: 100, yMin: 200, xMax: 300, yMax: 400 })
  })

  it('drops a box with no extent, and keeps the rest of the answer', () => {
    const text = `${tagged('empty', '(5,5),(5,5)')}${tagged('line', '(0,0),(100,0)')}${tagged('ok', '(1,1),(2,2)')}`
    expect(parseBboxTokens(text).map((box) => box.label)).toEqual(['ok'])
  })

  it('drops a box clamped into having no extent', () => {
    // Both corners beyond the frame on the same side collapse to one point.
    expect(parseBboxTokens(tagged(null, '(1200,1200),(1500,1500)'))).toEqual([])
  })
})

describe('the fallback forms', () => {
  it('reads the bare form the fine-tuned checkpoint emits, carrying the label forward', () => {
    const boxes = parseBboxTokens('buildings(100,738),(330,998)(370,768),(610,1000)')
    expect(boxes).toHaveLength(2)
    expect(boxes.map((box) => box.label)).toEqual(['buildings', 'buildings'])
    expect(boxes[1]).toMatchObject({ xMin: 370, yMin: 768, xMax: 610, yMax: 1000 })
  })

  it('does not let a label span a sentence', () => {
    const [box] = parseBboxTokens('The scene is urban. Two roads (10,10),(500,40)')
    expect(box?.label).toBe('Two roads')
  })

  it('reads JSON bbox_2d entries, and rescales absolute pixels only when told the frame', () => {
    const text = '[{"bbox_2d": [10, 20, 30, 40], "label": "tank"}, {"bbox_2d": [0, 0, 2000, 1000]}]'
    expect(parseBboxTokens(text)).toEqual([
      { xMin: 10, yMin: 20, xMax: 30, yMax: 40, label: 'tank', score: null },
    ])
    expect(parseBboxTokens(text, 2000, 2000)[1]).toMatchObject({
      xMin: 0,
      yMin: 0,
      xMax: 1000,
      yMax: 500,
    })
  })

  it('prefers the tagged form when both are present', () => {
    const text = `${tagged('a', '(0,0),(10,10)')} and (20,20),(30,30)`
    expect(parseBboxTokens(text)).toHaveLength(1)
  })

  it('yields nothing for prose with no boxes', () => {
    expect(parseBboxTokens('Built-up area grew by 5.34% across 1 region.')).toEqual([])
  })
})

describe('stripBboxTokens', () => {
  it('removes every form and leaves the prose readable', () => {
    const text = `Two aircraft: ${tagged('aircraft', '(1,1),(2,2)')} ${tagged('aircraft', '(3,3),(4,4)')}. Built-up area grew 5.34%.`
    expect(stripBboxTokens(text)).toBe('Two aircraft:. Built-up area grew 5.34%.')
  })

  it('leaves citation claims findable as substrings', () => {
    const text = `Largest ${tagged(null, '(0,0),(9,9)')} covering 350,106.12 m2.`
    expect(stripBboxTokens(text)).toContain('350,106.12 m2')
  })

  it('is the identity on box-free text', () => {
    expect(stripBboxTokens('nothing to see')).toBe('nothing to see')
  })
})
