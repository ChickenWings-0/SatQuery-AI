/**
 * The entry-chunk budget, checked against `dist/.vite/manifest.json` after
 * `vite build`. The console entry must stay small: every heavy dependency
 * has one lazy owner (`vite.config.ts` manualChunks), and this is the line
 * that notices when one leaks back into the entry.
 */
import { readFileSync, statSync } from 'node:fs'
import { gzipSync } from 'node:zlib'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url))
const manifest = JSON.parse(readFileSync(`${root}/dist/.vite/manifest.json`, 'utf8'))
const entry = Object.values(manifest).find((chunk) => chunk.isEntry)
if (!entry) throw new Error('check-bundle: no entry chunk in the manifest')

const BUDGET_KB = 180
const files = [entry.file, ...(entry.imports ?? []).map((key) => manifest[key]?.file).filter(Boolean)]
let gz = 0
for (const file of files) {
  const bytes = readFileSync(`${root}/dist/${file}`)
  gz += gzipSync(bytes).length
  statSync(`${root}/dist/${file}`)
}
const kb = gz / 1024
const line = `check-bundle: entry + static imports = ${kb.toFixed(1)} KB gz (budget ${BUDGET_KB} KB)`
if (kb > BUDGET_KB) {
  console.error(line)
  process.exit(1)
}
console.log(line)

for (const name of ['globe', 'map', 'dag']) {
  const leaked = (entry.imports ?? []).some((key) => key.includes(name))
  if (leaked) {
    console.error(`check-bundle: the ${name} chunk is statically imported by the entry`)
    process.exit(1)
  }
}
