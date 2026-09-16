/**
 * The date window. Two native date inputs — the keyboard story is free and
 * a judge knows how they work — with three presets that are the demo's
 * whole vocabulary: the last month, the last six months, a year apart.
 */
import { CalendarIcon } from '@/components/ui/icons'
import { useStacStore } from '@/state/stac'

const isoDate = (d: Date) => d.toISOString().slice(0, 10)

function preset(months: number): [string, string] {
  const to = new Date()
  const from = new Date(to)
  from.setMonth(from.getMonth() - months)
  return [isoDate(from), isoDate(to)]
}

const PRESETS: { label: string; months: number }[] = [
  { label: '30 d', months: 1 },
  { label: '6 mo', months: 6 },
  { label: '1 yr', months: 12 },
]

export function TimeSlider() {
  const [from, to] = useStacStore((state) => state.query.datetime)
  const pair = useStacStore((state) => state.query.pair)
  const setDatetime = useStacStore((state) => state.setDatetime)
  const setPair = useStacStore((state) => state.setPair)
  const input = 'field h-7 !w-auto !px-1.5 !py-0 text-[12px] tabular'

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <CalendarIcon size={14} className="shrink-0 text-text-lo" />
        <input
          type="date"
          value={from}
          max={to}
          aria-label="From date"
          onChange={(event) => event.target.value && setDatetime([event.target.value, to])}
          className={input}
        />
        <span className="text-text-lo" aria-hidden>
          →
        </span>
        <input
          type="date"
          value={to}
          min={from}
          aria-label="To date"
          onChange={(event) => event.target.value && setDatetime([from, event.target.value])}
          className={input}
        />
      </div>
      <div className="flex items-center gap-1.5">
        <div role="group" aria-label="Date presets" className="flex gap-1">
          {PRESETS.map((p) => (
            <button key={p.label} type="button" onClick={() => setDatetime(preset(p.months))} className="chip border border-line hover:border-accent-warm hover:text-text-hi">
              {p.label}
            </button>
          ))}
        </div>
        <label className="ml-auto flex items-center gap-2 text-[12px] text-text-lo">
          <span>T1 / T2 pair</span>
          <button type="button" role="switch" aria-checked={pair} aria-label="Pick a T1/T2 pair" onClick={() => setPair(!pair)} className="switch" />
        </label>
      </div>
    </div>
  )
}
