/**
 * A pill switch: `role="radiogroup"`, arrow keys move, one tab stop.
 */
import type { ReactNode } from 'react'

export interface SegmentedOption<T extends string> {
  value: T
  label: ReactNode
  /** Accessible name when `label` is an icon. */
  title?: string
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  size = 'md',
}: {
  value: T
  options: readonly SegmentedOption<T>[]
  onChange: (value: T) => void
  label: string
  size?: 'sm' | 'md'
}) {
  const pad = size === 'sm' ? 'h-7 px-2 text-[11.5px]' : 'h-8 px-2.5 text-[12.5px]'
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="inline-flex items-center gap-0.5 rounded-lg border border-line bg-bg-main p-0.5"
      onKeyDown={(event) => {
        if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
        event.preventDefault()
        const index = options.findIndex((option) => option.value === value)
        const next = options[(index + (event.key === 'ArrowRight' ? 1 : options.length - 1)) % options.length]
        if (next) onChange(next.value)
      }}
    >
      {options.map((option) => {
        const active = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={option.title}
            title={option.title}
            tabIndex={active ? 0 : -1}
            onClick={() => onChange(option.value)}
            className={`flex items-center gap-1.5 rounded-md font-medium transition-colors duration-[120ms] ${pad} ${
              active
                ? 'bg-surface-elevated text-text-hi shadow-[inset_0_1px_0_var(--color-glass-hi)]'
                : 'text-text-lo hover:text-text-hi'
            }`}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
