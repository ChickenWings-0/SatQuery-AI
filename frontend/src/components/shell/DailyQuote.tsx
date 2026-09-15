/**
 * The sidebar's mission slot: one sourced line a day.
 *
 * The caption is the speaker's name and nothing else. Every quote still
 * carries its provenance in `quotes.ts` — that is the product's own rule, a
 * line without a source does not ship — but a second caption line naming
 * the speech or the book was hierarchy the rail does not have room for: the
 * quote, the name, the terracotta rule. The source rides on the figure's
 * `title`, so it is one hover away rather than gone.
 */
import { useDailyQuote } from '@/shell/useDailyQuote'

export function DailyQuote() {
  const { quote, index, total } = useDailyQuote()
  return (
    <figure
      className="mt-8 hidden px-2.5 wide:block"
      title={`${quote.by}, ${quote.source} · quote ${index + 1} of ${total}, changes daily`}
    >
      <blockquote className="t-quote text-sidebar-text-lo">{quote.text}</blockquote>
      <figcaption className="mt-2 text-[11px] leading-snug font-medium text-sidebar-text">
        — {quote.by}
      </figcaption>
      <span aria-hidden className="mt-2.5 block h-0.5 w-8 rounded-full bg-accent-warm" />
    </figure>
  )
}
