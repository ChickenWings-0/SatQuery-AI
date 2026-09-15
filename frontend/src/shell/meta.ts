/**
 * Per-route document metadata — one table, read by `DocumentMeta` in the
 * tree and by `scripts/gen-public-meta.mjs` at build time for `sitemap.xml`,
 * `robots.txt` and `llms.txt`, so a new route cannot be forgotten by one of
 * them.
 *
 * `origin` is empty on the lab LAN. A canonical URL pointing at
 * `http://192.168.x.x` would be wrong, so canonical and `og:url` are emitted
 * only when a public origin was set at build time.
 */
import type { Section } from '@/state/ui'

export const SITE = {
  name: 'SatQuery AI',
  origin: (import.meta.env.VITE_PUBLIC_ORIGIN as string | undefined)?.replace(/\/$/, '') ?? '',
  defaultDescription:
    'Mission control for grounded Earth-observation analysis. Ask a question of satellite imagery and get an answer where every number is traced to the tool that measured it.',
  ogImage: '/og/default.png',
  locale: 'en_IN',
} as const

export interface PageMeta {
  title: string
  description: string
  path: string
  ogImage?: string
  noindex?: boolean
}

export type MetaPage = Section | 'report'

export const META: Record<MetaPage, PageMeta> = {
  home: {
    title: 'SatQuery AI — From space to answers, with the receipts',
    description:
      'Bi-temporal and cross-modal SAR/optical satellite analysis where every number is bound to a measurement and the tool graph is part of the answer. Built for SIH 2026, ISRO / SAC.',
    path: '/',
    ogImage: '/og/landing.png',
  },
  explore: {
    title: 'New Query · SatQuery AI',
    description:
      'The workspace: drop satellite imagery, let the eleven-check pre-flight say what it can answer, and ask.',
    path: '/explore',
  },
  usecases: {
    title: 'Use cases · SatQuery AI',
    description: 'Runnable geospatial investigations over Indian scenes: sprawl, floods, grounding, cross-modal checks.',
    path: '/use-cases',
  },
  maps: {
    title: 'Maps · SatQuery AI',
    description: 'The loaded scene in the world, with every rendered view switchable and a swipe compare.',
    path: '/maps',
  },
  saved: {
    title: 'Saved · SatQuery AI',
    description: 'Runs kept on this device, with their boxes, numbers and trace.',
    path: '/saved',
    noindex: true,
  },
  projects: {
    title: 'Projects · SatQuery AI',
    description: 'Investigations grouped by question: one footprint, one report.',
    path: '/projects',
    noindex: true,
  },
  datasets: {
    title: 'Datasets · SatQuery AI',
    description: 'The corpus sources behind the model, as the API reports them.',
    path: '/datasets',
  },
  tools: {
    title: 'Tools · SatQuery AI',
    description: 'The tool registry, live from /v1/registry: what can run on this machine right now.',
    path: '/tools',
  },
  history: {
    title: 'History · SatQuery AI',
    description: 'Every question asked this session and what became of it.',
    path: '/history',
    noindex: true,
  },
  report: {
    title: 'Report · SatQuery AI',
    description: 'A printable record of a run: answer, measurements, pipeline and trace.',
    path: '/report',
    noindex: true,
  },
  notFound: {
    title: 'Not found · SatQuery AI',
    description: 'That address does not point at anything in SatQuery AI.',
    path: '/404',
    noindex: true,
  },
}

/** Public, indexable routes — what the sitemap lists. */
export function indexablePages(): PageMeta[] {
  return Object.values(META).filter((page) => !page.noindex)
}
