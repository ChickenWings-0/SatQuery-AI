/**
 * Today's quote, deterministically: day-of-month indexes the list, so every
 * machine in the room shows the same line and it changes at local midnight.
 *
 * The hook re-arms itself for the next midnight while the tab is open and on
 * `visibilitychange`, so a console left running overnight rolls over rather
 * than showing yesterday until someone reloads.
 */
import { useEffect, useState } from 'react'

import { QUOTES, type Quote } from '@/shell/quotes'

export function quoteFor(date: Date): Quote {
  return QUOTES[(date.getDate() - 1) % QUOTES.length]!
}

export function quoteIndex(date: Date): number {
  return (date.getDate() - 1) % QUOTES.length
}

function msUntilMidnight(now: Date): number {
  const next = new Date(now)
  next.setHours(24, 0, 0, 0)
  return Math.max(1_000, next.getTime() - now.getTime())
}

export function useDailyQuote(): { quote: Quote; index: number; total: number } {
  const [date, setDate] = useState(() => new Date())

  useEffect(() => {
    let timer = window.setTimeout(function roll() {
      setDate(new Date())
      timer = window.setTimeout(roll, msUntilMidnight(new Date()))
    }, msUntilMidnight(date))
    const onVisible = () => {
      if (document.visibilityState === 'visible') setDate(new Date())
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      window.clearTimeout(timer)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [date])

  return { quote: quoteFor(date), index: quoteIndex(date), total: QUOTES.length }
}
