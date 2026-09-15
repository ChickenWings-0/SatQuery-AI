/**
 * A curated gallery of runnable investigations. Pick one and be in the
 * console with imagery loaded and a question in the box, in one click.
 */
import { useMemo, useState } from 'react'

import { Graticule } from '@/components/ui/Graticule'
import { SectionHeader } from '@/components/ui/SectionHeader'
import { FAMILY_LABELS, USE_CASES, type TaskFamily } from '@/pages/usecases/catalogue'
import { UseCaseCard } from '@/pages/usecases/UseCaseCard'
import { useLoadSample } from '@/pages/usecases/useLoadSample'
import { DocumentMeta } from '@/shell/DocumentMeta'

const FAMILIES = Object.keys(FAMILY_LABELS) as TaskFamily[]

export default function UseCases() {
  const [families, setFamilies] = useState<TaskFamily[]>([])
  const { load, loading, progress, error } = useLoadSample()

  const rows = useMemo(
    () => (families.length === 0 ? USE_CASES : USE_CASES.filter((u) => families.includes(u.family))),
    [families],
  )

  function toggle(family: TaskFamily) {
    setFamilies((current) =>
      current.includes(family) ? current.filter((f) => f !== family) : [...current, family],
    )
  }

  return (
    <>
      <DocumentMeta page="usecases" />
      <SectionHeader title="Use cases" meta={`${USE_CASES.length} investigations`}>
        <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Filter by task family">
          {FAMILIES.map((family) => {
            const on = families.includes(family)
            return (
              <button
                key={family}
                type="button"
                aria-pressed={on}
                onClick={() => toggle(family)}
                className={`rounded-full border px-3 py-1 text-[12px] transition-colors duration-[120ms] ${
                  on
                    ? 'border-accent-warm bg-accent-glow text-accent-warm-text'
                    : 'border-line text-text-lo hover:border-accent-warm hover:text-accent-warm-text'
                }`}
              >
                {FAMILY_LABELS[family]}
              </button>
            )
          })}
          {families.length > 0 ? (
            <button type="button" onClick={() => setFamilies([])} className="t-meta ml-1 hover:text-text-hi">
              Clear
            </button>
          ) : null}
        </div>
      </SectionHeader>

      {rows.length === 0 ? (
        <div className="relative overflow-hidden rounded-xl border border-line py-16 text-center">
          <Graticule />
          <p className="t-panel relative">No use cases match those filters.</p>
          <button type="button" onClick={() => setFamilies([])} className="btn-ghost relative mt-4">
            Clear filters
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 wide:grid-cols-2 desk:grid-cols-3">
          {rows.map((useCase) => (
            <UseCaseCard
              key={useCase.slug}
              useCase={useCase}
              onLoad={() => void load(useCase)}
              busy={loading === useCase.slug}
              progress={progress?.slug === useCase.slug ? progress : null}
              error={error?.slug === useCase.slug ? error.message : null}
              disabled={loading !== null && loading !== useCase.slug}
            />
          ))}
        </div>
      )}
    </>
  )
}
