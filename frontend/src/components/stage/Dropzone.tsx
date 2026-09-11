/**
 * File selection. Validation fires on *select*, not on submit.
 *
 * API_CONTRACT §8.2: call `/v1/validate` on file select, before the user types.
 * It is fast and GPU-free, and its `supported_tasks` is what tells the UI which
 * questions are worth offering — so waiting until submit would throw away the
 * whole point of the endpoint.
 *
 * Everything else here is about what a file picker actually receives. Users
 * drop folders, `.zip`s, `.DS_Store`, 4 GB scenes and empty files, and they drop
 * a second set while the first is still validating. Each of those is caught
 * before it costs a round trip — a rejected 512 MB upload that had to reach the
 * server first is a two-minute mistake, not a two-second one. The server
 * remains the authority; this only avoids the obviously doomed request.
 */
import { useRef, useState } from 'react'

import { bytes, countOf } from '@/format'
import { useUiStore } from '@/state/ui'

const ACCEPT = '.tif,.tiff,.png,.jpg,.jpeg,image/tiff,image/png,image/jpeg'
const MAX_IMAGES = 2
/** Mirrors `max_upload_mb` in `satquery.core.config` — the server's own cap. */
const MAX_BYTES = 512 * 1024 * 1024
const EXTENSIONS = /\.(tiff?|png|jpe?g)$/i

interface Rejection {
  name: string
  reason: string
}

/** Split a raw selection into what can be sent and what cannot, with reasons. */
function triage(list: File[]): { accepted: File[]; rejected: Rejection[] } {
  const accepted: File[] = []
  const rejected: Rejection[] = []

  for (const file of list) {
    // A dropped *directory* arrives as a File with no type and no extension,
    // and uploading it produces a confusing server-side error rather than an
    // obvious client-side one.
    if (!EXTENSIONS.test(file.name)) {
      rejected.push({ name: file.name, reason: 'not a GeoTIFF, PNG or JPEG' })
    } else if (file.size === 0) {
      rejected.push({ name: file.name, reason: 'empty file' })
    } else if (file.size > MAX_BYTES) {
      rejected.push({ name: file.name, reason: `${bytes(file.size)} exceeds the ${bytes(MAX_BYTES)} limit` })
    } else {
      accepted.push(file)
    }
  }

  return { accepted, rejected }
}

export function Dropzone() {
  const input = useRef<HTMLInputElement>(null)
  // A depth counter, not a boolean: `dragleave` fires every time the pointer
  // crosses into a child element, so a boolean makes the highlight strobe.
  const depth = useRef(0)
  const [dragging, setDragging] = useState(false)
  const [rejected, setRejected] = useState<Rejection[]>([])
  /** True when more usable images were offered than the contract allows. */
  const [truncated, setTruncated] = useState(false)
  // The pre-flight request itself belongs to the store, not to this component:
  // `selectFiles` replaces the drop target with the manifest list, so this
  // component is gone long before `/v1/validate` answers.
  const selectFiles = useUiStore((state) => state.selectFiles)

  function accept(list: FileList | null) {
    if (!list || list.length === 0) return
    const { accepted, rejected: bad } = triage([...list])
    setRejected(bad)
    setTruncated(accepted.length > MAX_IMAGES)

    if (accepted.length === 0) {
      // Nothing usable: keep whatever was already selected rather than
      // clearing a good selection because of a stray drop.
      return
    }

    // The contract caps a request at two images; catching it here means an
    // obvious mistake never costs a round trip.
    void selectFiles(accepted.slice(0, MAX_IMAGES))
  }

  return (
    <div>
      <div
        onDragEnter={(event) => {
          event.preventDefault()
          depth.current += 1
          setDragging(true)
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => {
          depth.current = Math.max(0, depth.current - 1)
          if (depth.current === 0) setDragging(false)
        }}
        onDrop={(event) => {
          event.preventDefault()
          depth.current = 0
          setDragging(false)
          accept(event.dataTransfer.files)
        }}
        className={`rounded-xl border-2 border-dashed p-6 text-center transition-colors md:p-10 ${
          dragging ? 'border-accent-warm bg-accent-cool/25' : 'border-line'
        }`}
      >
        <p className="text-lg font-medium">Drop one or two scenes</p>
        <p className="mt-1.5 text-sm text-text-lo">
          GeoTIFF preferred; PNG and JPEG accepted, up to {bytes(MAX_BYTES)} each. Two images
          unlock bi-temporal and cross-modal questions.
        </p>
        <button
          type="button"
          onClick={() => input.current?.click()}
          className="mt-5 min-h-11 rounded-lg border border-accent-warm-text px-4 py-2 text-sm font-medium text-accent-warm-text transition-colors hover:bg-accent-warm-strong hover:text-white"
        >
          Choose files
        </button>
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          multiple
          hidden
          onChange={(event) => {
            accept(event.target.files)
            // Reset, so re-picking the same file after a rejection still fires
            // `change` — otherwise the second attempt is silently ignored.
            event.target.value = ''
          }}
        />
      </div>

      {(rejected.length > 0 || truncated) && (
        <div role="status" className="mt-2 space-y-1 text-sm">
          {truncated && (
            <p className="text-text-lo">
              <span aria-hidden className="text-warn-text">
                ⚠
              </span>{' '}
              At most {countOf(MAX_IMAGES, { one: 'image', other: 'images' })} can be analysed
              together; the first {MAX_IMAGES} were kept.
            </p>
          )}
          {rejected.map((entry) => (
            <p key={entry.name} className="flex gap-1.5 text-text-lo [overflow-wrap:anywhere]">
              <span aria-hidden className="shrink-0 text-warn-text">
                ⚠
              </span>
              <span>
                <span className="font-medium">{entry.name}</span> was not used — {entry.reason}.
              </span>
            </p>
          ))}
        </div>
      )}
    </div>
  )
}
