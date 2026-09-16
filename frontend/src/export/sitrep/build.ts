/**
 * Compose → draw → render, as one lazily loaded unit. `useSitrep` reaches
 * this file only through `import()`, so the composer, the canvas panel and
 * pdf-lib all ride in the `sitrep-*` chunk and the console entry pays for
 * none of them until the click.
 */
import { artifactUrl } from '@/api/client'
import type { ArtifactRef } from '@/api/types'
import { composeSitrep, sitrepFileName, type SitrepInput } from '@/export/sitrep/compose'
import { renderSitrep } from '@/export/sitrep/render'
import { drawScenePanel } from '@/export/sitrep/scene'

export async function buildSitrepFromInput(input: SitrepInput): Promise<{ name: string; bytes: Uint8Array }> {
  const model = composeSitrep(input)
  const artifacts: ArtifactRef[] = input.artifacts.length > 0 ? input.artifacts : (input.result?.artifacts ?? [])
  const byId = new Map(artifacts.map((a) => [a.id, a]))
  const scene = await drawScenePanel(model, (id) => {
    const ref = byId.get(id)
    return ref ? artifactUrl(ref) : null
  })
  const bytes = await renderSitrep(model, scene)
  return { name: sitrepFileName(model.header.sceneId, input.generatedAt), bytes }
}
