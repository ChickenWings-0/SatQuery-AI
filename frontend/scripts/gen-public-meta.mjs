/**
 * Generate `public/robots.txt`, `public/sitemap.xml` and `public/llms.txt`
 * from the route table in `src/shell/meta.ts`, so a new route cannot be
 * forgotten by one of them. Runs as `prebuild`; also runnable by hand.
 *
 * `VITE_PUBLIC_ORIGIN` (from the environment or `.env`) is the site root;
 * when empty, the sitemap and canonical hints are still written but the
 * origin is left blank, which is correct for a LAN deploy.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url))
const env = (() => {
  try {
    return Object.fromEntries(
      readFileSync(`${root}/.env`, 'utf8')
        .split('\n')
        .filter((l) => l.includes('=') && !l.startsWith('#'))
        .map((l) => l.split('=').map((s) => s.trim())),
    )
  } catch {
    return {}
  }
})()
const origin = (process.env.VITE_PUBLIC_ORIGIN ?? env.VITE_PUBLIC_ORIGIN ?? '').replace(/\/$/, '')

// A tiny extractor rather than a TS loader: the table is literal objects.
const source = readFileSync(`${root}/src/shell/meta.ts`, 'utf8')
const entries = [...source.matchAll(/(\w+): \{\s*title: '([^']*)',\s*description:\s*'([^']*)',\s*path: '([^']*)',(?:\s*ogImage: '[^']*',)?(\s*noindex: true,)?/g)].map(
  (m) => ({ key: m[1], title: m[2], description: m[3], path: m[4], noindex: Boolean(m[5]) }),
)
if (entries.length === 0) throw new Error('gen-public-meta: no routes found in src/shell/meta.ts')

const indexable = entries.filter((e) => !e.noindex)
const today = new Date().toISOString().slice(0, 10)

const robots = `# SatQuery AI — https://github.com/ChickenWings-0/SatQuery-AI
# AI crawlers welcome — see /llms.txt. Only device-local pages are disallowed.
User-agent: *
Allow: /
${entries
  .filter((e) => e.noindex && e.path !== '/404')
  .map((e) => `Disallow: ${e.path}`)
  .join('\n')}
Disallow: /report/
Disallow: /v1/
${origin ? `Sitemap: ${origin}/sitemap.xml` : '# Sitemap: set VITE_PUBLIC_ORIGIN to emit an absolute sitemap URL'}
`

const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${indexable.map((e) => `  <url>\n    <loc>${origin}${e.path}</loc>\n    <lastmod>${today}</lastmod>\n  </url>`).join('\n')}
</urlset>
`

const llms = `# SatQuery AI

> Evidence-bound geospatial intelligence: an agentic vision-language assistant for satellite imagery that shows its work. A query is parsed, inputs pass eleven compatibility checks, a deterministic policy table selects and sequences specialist tools, and the answer is grounded — every number bound to a measured scalar or flagged as uncited — with a versioned, schema-validated audit trace. Built for Smart India Hackathon 2026, problem statement 26167 (ISRO / Space Applications Centre).

## What it does

- VQA, captioning and grounding (validated bounding boxes) over pre-rendered views
- Bi-temporal change detection: Siamese change model + change statistics, co-registered at ingest
- Cross-modal optical + SAR: spectral-index and backscatter analysers feed one fact sheet; physics_agreement flags disagreement
- Input validation: 11 named checks (CRS, GSD, overlap, co-registration, …) before any tool runs
- Citation honesty: a CitationValidator that flags rather than strips
- Audit trace: built by the executor, persisted, returned verbatim

## How answers are built

1. Raster ingest & pre-flight — POST /v1/validate, 11 checks
2. Spatial policy DAG — policy_key = TaskType | PairType | Modality; a frozen table selects the tools
3. Co-register · render · mask — spectral views (TC, FCIR, NDVI, NDBI, NDWI, SAR), change masks, measured areas
4. Evidence-bound synthesis — the VLM (Qwen3-VL-8B + QLoRA) writes; the validator binds numbers to the trace

The LLM performs slot filling and answer synthesis only. It never chooses a tool, never orders steps, and never invents a capability.

## Pages

${indexable.map((e) => `- [${e.title}](${origin}${e.path}): ${e.description}`).join('\n')}

## Not for

- No accounts, no cloud storage: saved runs and projects live in the browser
- Runs fully local on one 24 GB consumer GPU; survives the network being pulled
`

writeFileSync(`${root}/public/robots.txt`, robots)
writeFileSync(`${root}/public/sitemap.xml`, sitemap)
writeFileSync(`${root}/public/llms.txt`, llms)
console.log(`gen-public-meta: ${indexable.length} indexable routes, origin ${origin || '(none)'}`)
