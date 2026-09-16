/**
 * Track 4 in the browser, on the built bundle, on recorded fixtures, with
 * the network switched off after the page loads — which is the venue-Wi-Fi
 * rehearsal (ROADMAP Track 4, DOCS/UI_TRACK_4_ARCHITECTURE.md §4.4) run by
 * a machine instead of a person:
 *
 *   1. /maps?mock=1 → Find imagery → "Ahm" → Ahmedabad → the map flies
 *   2. Sentinel-1, T1/T2 pair, two scenes six months apart → Analyse change
 *   3. the console opens with two views and a passing pre-flight
 *   4. run the query → SITREP downloads as a one-page PDF
 *   5. Export GeoJSON → a WGS84 FeatureCollection with the provenance columns
 */
import { PDFDocument } from 'pdf-lib'
import { expect, test, type Page } from '@playwright/test'

const SHOTS = process.env['TRACK4_SHOTS']

async function shot(page: Page, name: string) {
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/${name}.png` })
}

async function readDownload(page: Page, trigger: () => Promise<void>) {
  const [download] = await Promise.all([page.waitForEvent('download'), trigger()])
  const stream = await download.createReadStream()
  const chunks: Buffer[] = []
  for await (const chunk of stream) chunks.push(Buffer.from(chunk))
  return { name: download.suggestedFilename(), bytes: Buffer.concat(chunks) }
}

test('finds imagery, sends a pair to the console, and exports a SITREP and GeoJSON — offline', async ({ page, context }) => {
  // The venue Wi-Fi drops: everything off this machine is refused. The app
  // itself is served from localhost, as on the demo laptop, so its chunks
  // still load; every third-party call must be answered by the worker.
  await context.route('**/*', (route) => {
    const host = new URL(route.request().url()).hostname
    if (host === '127.0.0.1' || host === 'localhost') return route.continue()
    return route.abort('internetdisconnected')
  })
  await page.goto('/maps?mock=1')

  await page.getByRole('button', { name: 'Find imagery' }).first().click()
  const search = page.getByRole('combobox', { name: 'Search for a place' })
  await search.fill('Ahm')
  await expect(page.getByRole('option', { name: /Ahmedabad/ }).first()).toBeVisible({ timeout: 10_000 })
  await shot(page, '1-place-search')
  await page.getByRole('option', { name: /Ahmedabad/ }).first().click()

  // Optical first: the default, with a cloud slider.
  await expect(page.getByText(/scenes, newest first/)).toBeVisible({ timeout: 10_000 })
  await shot(page, '2-optical-shelf')

  // Sentinel-1, as a pair.
  await page.getByRole('radio', { name: 'Sentinel-1 RTC, SAR' }).click()
  await expect(page.getByText(/scenes, newest first/)).toBeVisible({ timeout: 10_000 })
  await page.getByRole('switch', { name: 'Pick a T1/T2 pair' }).click()
  const cards = page.getByRole('list').filter({ has: page.getByRole('button', { name: /Use as T1/ }) })
  await expect(cards).toBeVisible()
  const useT2 = page.getByRole('button', { name: 'Use as T2' })
  const useT1 = page.getByRole('button', { name: 'Use as T1' })
  await useT2.first().click()
  // Six months earlier: the ninth card in the recording is 2026-03-03.
  await useT1.nth(8).click()
  await expect(page.getByRole('button', { name: 'Analyse change' })).toBeEnabled()
  await shot(page, '3-pair-picked')

  const analyse = page.getByRole('button', { name: 'Analyse change' })
  await expect(analyse).toBeEnabled()
  await analyse.click()

  // The console, with two files through pre-flight.
  const question = page.getByRole('textbox', { name: 'Question about this imagery' })
  await expect(question).toBeEnabled({ timeout: 20_000 })
  await expect(question).toHaveValue(/changed between these two/)
  await shot(page, '4-console-preflight')

  await page.getByRole('button', { name: 'Ask' }).click()
  await expect(page.getByRole('button', { name: /SITREP/ })).toBeVisible({ timeout: 40_000 })
  await expect(page.getByRole('button', { name: /Export GeoJSON/ })).toBeVisible()
  await shot(page, '5-answer')

  // SITREP: one page, A4, more than 20 KB, named after the scene.
  const pdf = await readDownload(page, () => page.getByRole('button', { name: /SITREP/ }).click())
  expect(pdf.name).toMatch(/^SITREP-.+-\d{8}-\d{4}\.pdf$/)
  expect(pdf.bytes.byteLength).toBeGreaterThan(20_000)
  const doc = await PDFDocument.load(pdf.bytes)
  expect(doc.getPageCount()).toBe(1)
  if (SHOTS) await import('node:fs/promises').then((fs) => fs.writeFile(`${SHOTS}/sitrep.pdf`, pdf.bytes))

  // GeoJSON: the recording has a change mask, so the dialog asks about masks.
  await page.getByRole('button', { name: /Export GeoJSON/ }).click()
  await expect(page.getByRole('dialog', { name: 'Export GeoJSON' })).toBeVisible()
  await shot(page, '6-geojson-dialog')
  const geo = await readDownload(page, () => page.getByRole('button', { name: 'Export', exact: true }).click())
  expect(geo.name).toMatch(/\.geojson$/)
  const collection = JSON.parse(geo.bytes.toString('utf8')) as {
    type: string
    bbox?: number[]
    crs?: unknown
    features: { properties: Record<string, unknown>; geometry: { type: string } }[]
  }
  expect(collection.type).toBe('FeatureCollection')
  expect(collection.crs).toBeUndefined()
  expect(collection.bbox).toHaveLength(4)
  const mask = collection.features.find((f) => f.properties['kind'] === 'mask')
  expect(mask?.geometry.type).toBe('MultiPolygon')
  expect(mask?.properties['trace_id']).toBeTruthy()

  // Keyboard: ⌘⇧S builds another brief.
  const again = await readDownload(page, () => page.keyboard.press('ControlOrMeta+Shift+S'))
  expect(again.name).toMatch(/^SITREP-/)
})

test('the online gate holds when online features are off and the mock is not on', async ({ page }) => {
  await page.goto('/maps')
  await page.getByRole('button', { name: 'Find imagery' }).first().click()
  await expect(page.getByText('Imagery search is off.')).toBeVisible()
  await expect(page.getByRole('combobox', { name: 'Search for a place' })).toHaveCount(0)
  await shot(page, '7-gate')
})

test('the globe toggle works with no run loaded', async ({ page }) => {
  await page.setViewportSize({ width: 1400, height: 860 })
  await page.goto('/maps?mock=1')
  await page.getByRole('button', { name: 'Find imagery' }).first().click()
  const globe = page.getByRole('switch', { name: 'Globe projection' })
  await globe.click()
  await expect(globe).toHaveAttribute('aria-checked', 'true')
  await page.waitForTimeout(800)
  await shot(page, '8-globe')
  await page.keyboard.press('Escape')
  await page.locator('[role="application"]').focus()
  await page.keyboard.press('b')
  await expect(globe).toHaveAttribute('aria-checked', 'false')
})
