/**
 * The bounding-box overlay lands on the pixels, in a real layout engine.
 *
 * `ImageViewer` sizes the overlay to the `object-fit: contain` rectangle of
 * the raster — computed from `naturalWidth/Height` and a `ResizeObserver` —
 * not to the cell. On a square image the two coincide and nothing can go
 * wrong; on a wide image in a tall cell (or the reverse) every box lands off
 * target unless that arithmetic is right. That is the highest-risk visual
 * path in the product, and the unit suite cannot see it: happy-dom has no
 * layout. So this drives the built bundle in Chromium with the `grounded`
 * mock scenario — a 896x448 render and one box at (250,250)-(750,750) — at
 * two viewports, one width-bound and one height-bound.
 */
import { expect, test, type Page } from '@playwright/test'

const TOLERANCE_PX = 1.5

async function startGroundedRun(page: Page) {
  await page.goto('/?mock=1&scenario=grounded')
  const input = page.locator('input[type="file"]')
  await input.setInputFiles([
    { name: 'pre.tif', mimeType: 'image/tiff', buffer: Buffer.from([1, 2, 3, 4]) },
    { name: 'post.tif', mimeType: 'image/tiff', buffer: Buffer.from([1, 2, 3, 4]) },
  ])
  const question = page.getByRole('textbox', { name: 'Question about this imagery' })
  await expect(question).toBeEnabled({ timeout: 15_000 })
  await question.fill('Where is the walled compound?')
  await page.getByRole('button', { name: 'Ask' }).click()
  await expect(page.getByTestId('bbox-overlay')).toBeVisible({ timeout: 30_000 })
}

/** The rectangle `object-fit: contain` renders an image into, from its own box. */
async function containRect(page: Page) {
  return page.evaluate(() => {
    const img = document.querySelector<HTMLImageElement>('img[alt^="Image 1"]')
    if (!img) throw new Error('the pre-change image is not on the stage')
    const box = img.getBoundingClientRect()
    const scale = Math.min(box.width / img.naturalWidth, box.height / img.naturalHeight)
    const width = img.naturalWidth * scale
    const height = img.naturalHeight * scale
    return {
      natural: { width: img.naturalWidth, height: img.naturalHeight },
      left: box.left + (box.width - width) / 2,
      top: box.top + (box.height - height) / 2,
      width,
      height,
    }
  })
}

async function overlayRect(page: Page) {
  // The overlay's parent is the absolutely positioned layer sized to `fit`.
  const layer = page.getByTestId('bbox-overlay').locator('..')
  const box = await layer.boundingBox()
  if (!box) throw new Error('the overlay layer has no box')
  return { left: box.x, top: box.y, width: box.width, height: box.height }
}

for (const viewport of [
  { name: 'width-bound (wide window)', width: 1400, height: 700 },
  { name: 'height-bound (narrow window)', width: 700, height: 900 },
]) {
  test(`the overlay matches the image's contain box: ${viewport.name}`, async ({ page }) => {
    await page.setViewportSize({ width: viewport.width, height: viewport.height })
    await startGroundedRun(page)

    const image = await containRect(page)
    expect(image.natural).toEqual({ width: 896, height: 448 })
    const overlay = await overlayRect(page)

    expect(Math.abs(overlay.left - image.left)).toBeLessThan(TOLERANCE_PX)
    expect(Math.abs(overlay.top - image.top)).toBeLessThan(TOLERANCE_PX)
    expect(Math.abs(overlay.width - image.width)).toBeLessThan(TOLERANCE_PX)
    expect(Math.abs(overlay.height - image.height)).toBeLessThan(TOLERANCE_PX)

    // The box is (250,250)-(750,750) in the 0-1000 frame: its centre must be
    // the centre of the rendered raster, and its size half of it on each axis.
    const rect = page.getByTestId('bbox-overlay').locator('svg rect').first()
    const drawn = await rect.boundingBox()
    if (!drawn) throw new Error('the box was not drawn')
    expect(Math.abs(drawn.x + drawn.width / 2 - (image.left + image.width / 2))).toBeLessThan(
      TOLERANCE_PX,
    )
    expect(Math.abs(drawn.y + drawn.height / 2 - (image.top + image.height / 2))).toBeLessThan(
      TOLERANCE_PX,
    )
    expect(Math.abs(drawn.width - image.width / 2)).toBeLessThan(TOLERANCE_PX)
    expect(Math.abs(drawn.height - image.height / 2)).toBeLessThan(TOLERANCE_PX)
  })
}

test('boxes come from the grounding artifact, not the prose', async ({ page }) => {
  await page.setViewportSize({ width: 1200, height: 800 })
  await startGroundedRun(page)
  await expect(page.getByTestId('bbox-source-hint')).toHaveCount(0)
})
