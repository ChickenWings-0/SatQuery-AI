/**
 * Loading placeholders that reserve the final layout's dimensions, so nothing
 * shifts when data lands. The shimmer is a background-position loop; under
 * reduced motion theme.css stills it.
 */
export function Skeleton({ className = '' }: { className?: string }) {
  return <div aria-hidden className={`skeleton ${className}`} />
}

export function CardSkeleton() {
  return (
    <div className="card-flush">
      <Skeleton className="aspect-[16/10] w-full rounded-none" />
      <div className="space-y-2.5 p-4">
        <Skeleton className="h-3.5 w-3/4" />
        <Skeleton className="h-3 w-1/2" />
        <Skeleton className="h-3 w-1/3" />
      </div>
    </div>
  )
}

export function SectionSkeleton({ title }: { title?: string }) {
  return (
    <div aria-busy="true" aria-label={title ? `Loading ${title}` : 'Loading'}>
      <div className="mb-5 border-b border-line pb-4">
        <Skeleton className="h-7 w-40" />
      </div>
      <div className="grid grid-cols-1 gap-4 wide:grid-cols-2 desk:grid-cols-3">
        {Array.from({ length: 6 }, (_, i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
    </div>
  )
}
