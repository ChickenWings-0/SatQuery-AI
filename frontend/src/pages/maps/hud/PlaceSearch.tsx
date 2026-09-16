/**
 * Type a place, get a list, pick one: the map flies there and the scene
 * search runs. 400 ms debounce and a 3-character floor keep Nominatim's
 * one-request-a-second policy honest; ↑↓ move, ⏎ picks, Esc closes.
 */
import { useEffect, useId, useRef, useState } from 'react'

import { PinDropIcon, SearchIcon } from '@/components/ui/icons'
import { isOffline, UpstreamError } from '@/geo/deadline'
import { MIN_QUERY, search, type Place } from '@/geo/nominatim'
import { useStacStore } from '@/state/stac'

const DEBOUNCE_MS = 400

type ListState =
  | { kind: 'idle' }
  | { kind: 'searching' }
  | { kind: 'results'; places: Place[] }
  | { kind: 'empty' }
  | { kind: 'offline' }
  | { kind: 'error'; message: string }

export function PlaceSearch({ autoFocus = false }: { autoFocus?: boolean }) {
  const text = useStacStore((state) => state.query.placeText)
  const place = useStacStore((state) => state.query.place)
  const setPlaceText = useStacStore((state) => state.setPlaceText)
  const setPlace = useStacStore((state) => state.setPlace)
  const [list, setList] = useState<ListState>({ kind: 'idle' })
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const controller = useRef<AbortController | null>(null)
  const listId = useId()

  // Debounced search on the text, unless the text is the chosen place's name.
  const q = text.trim()
  const searchable = open && q.length >= MIN_QUERY && !(place && q === place.name)
  useEffect(() => {
    if (!searchable) return
    const timer = setTimeout(() => {
      controller.current?.abort()
      const ctl = new AbortController()
      controller.current = ctl
      setList({ kind: 'searching' })
      search(q, ctl.signal)
        .then((places) => {
          if (ctl.signal.aborted) return
          setActive(0)
          setList(places.length > 0 ? { kind: 'results', places } : { kind: 'empty' })
        })
        .catch((error: unknown) => {
          if (ctl.signal.aborted) return
          if (isOffline(error)) setList({ kind: 'offline' })
          else setList({ kind: 'error', message: error instanceof UpstreamError ? error.message : 'Place search failed.' })
        })
    }, DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [q, searchable])

  function choose(next: Place) {
    setPlace(next)
    setOpen(false)
    setList({ kind: 'idle' })
  }

  // Below the floor, or the chosen place's own name: nothing to list.
  const shown: ListState = searchable ? list : { kind: 'idle' }
  const places = shown.kind === 'results' ? shown.places : []

  return (
    <div className="relative">
      <div className="flex items-center gap-2">
        <SearchIcon size={14} className="shrink-0 text-text-lo" />
        <input
          autoFocus={autoFocus}
          value={text}
          role="combobox"
          aria-expanded={open && places.length > 0}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={open && places[active] ? `${listId}-${places[active]!.id}` : undefined}
          aria-label="Search for a place"
          placeholder="Search a place — Ahmedabad, Sundarbans, 12.97 77.59"
          onChange={(event) => {
            setPlaceText(event.target.value)
            setOpen(true)
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 120)}
          onKeyDown={(event) => {
            if (event.key === 'Escape') {
              setOpen(false)
              return
            }
            if (places.length === 0) return
            if (event.key === 'ArrowDown') {
              event.preventDefault()
              setActive((i) => (i + 1) % places.length)
            } else if (event.key === 'ArrowUp') {
              event.preventDefault()
              setActive((i) => (i + places.length - 1) % places.length)
            } else if (event.key === 'Enter') {
              event.preventDefault()
              const chosen = places[active]
              if (chosen) choose(chosen)
            }
          }}
          className="field h-8 !py-1 text-[13px]"
        />
      </div>

      {open && shown.kind !== 'idle' ? (
        <div className="glass absolute top-full left-0 z-20 mt-1.5 w-full p-1" role="presentation">
          {shown.kind === 'searching' ? (
            <p className="t-meta px-2.5 py-2" role="status">
              Searching…
            </p>
          ) : null}
          {shown.kind === 'empty' ? <p className="t-meta px-2.5 py-2">No place by that name.</p> : null}
          {shown.kind === 'offline' ? (
            <p className="px-2.5 py-2 text-[12px] text-warn-text" role="status">
              Search needs a network — the loaded scene is still here.
            </p>
          ) : null}
          {shown.kind === 'error' ? (
            <p className="px-2.5 py-2 text-[12px] text-warn-text" role="status">
              {shown.message}
            </p>
          ) : null}
          {shown.kind === 'results' ? (
            <ul id={listId} role="listbox" aria-label="Places" className="max-h-64 overflow-y-auto">
              {shown.places.map((candidate, index) => (
                <li
                  key={candidate.id}
                  id={`${listId}-${candidate.id}`}
                  role="option"
                  aria-selected={index === active}
                  onMouseEnter={() => setActive(index)}
                  onMouseDown={(event) => {
                    event.preventDefault()
                    choose(candidate)
                  }}
                  className={`menu-row !py-1.5 ${index === active ? 'bg-accent-glow' : ''}`}
                >
                  <PinDropIcon size={14} className="shrink-0 text-text-lo" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[13px] text-text-hi">{candidate.name}</span>
                    <span className="block truncate text-[11px] text-text-lo">{candidate.detail}</span>
                  </span>
                  <span className="t-coord text-text-lo">{candidate.kind}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
