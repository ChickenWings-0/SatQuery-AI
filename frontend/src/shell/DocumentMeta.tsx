/**
 * The one component that renders `<head>` tags.
 *
 * React 19 hoists `<title>`, `<meta>` and `<link>` rendered anywhere in the
 * tree into the document head and de-duplicates them by `name`/`property`/
 * `rel`, so this needs no helmet library — a dependency that re-implements a
 * platform feature is the thing this file exists not to be.
 *
 * Title grammar: `Subject · Section · SatQuery AI` — most specific first,
 * the brand last. The landing page is the one exception (brand first, then
 * the claim), because there the brand *is* the subject.
 */
import { META, SITE, type MetaPage } from '@/shell/meta'

export function DocumentMeta({ page, subject }: { page: MetaPage; subject?: string | undefined }) {
  const meta = META[page]
  const title = subject ? `${subject} · ${meta.title}` : meta.title
  const url = SITE.origin ? `${SITE.origin}${meta.path}` : null
  const image = `${SITE.origin}${meta.ogImage ?? SITE.ogImage}`
  return (
    <>
      <title>{title}</title>
      <meta name="description" content={meta.description} />
      {meta.noindex ? <meta name="robots" content="noindex" /> : null}
      {url ? <link rel="canonical" href={url} /> : null}
      <meta property="og:type" content="website" />
      <meta property="og:site_name" content={SITE.name} />
      <meta property="og:title" content={title} />
      <meta property="og:description" content={meta.description} />
      <meta property="og:image" content={image} />
      <meta property="og:image:width" content="1200" />
      <meta property="og:image:height" content="630" />
      <meta property="og:image:alt" content={`${SITE.name} — ${meta.description}`} />
      {url ? <meta property="og:url" content={url} /> : null}
      <meta property="og:locale" content={SITE.locale} />
      <meta name="twitter:card" content="summary_large_image" />
    </>
  )
}
