/**
 * One `InputManifest`, as the pre-flight renders it.
 *
 * API_CONTRACT §1's nulls rule is load-bearing here: a field that is
 * inapplicable or unknown is `null`, never `0` and never `""`. So every field
 * renders `—` when absent, and the manifest's own `warnings[]` are shown
 * alongside — that is where the *reason* for an unknown value lives.
 */
import { ABSENT, decimal, integer, percent, ratioAsPercent } from '@/format'
import type { InputManifest } from '@/api/types'

function Field({ label, value }: { label: string; value: string | number | null | undefined }) {
  const shown = value === null || value === undefined || value === '' ? ABSENT : String(value)
  const absent = shown === ABSENT
  return (
    <div className="min-w-0">
      <dt className="text-[10px] tracking-[0.08em] text-text-lo uppercase">{label}</dt>
      {/* CRS strings, driver names and band lists are arbitrary-length tokens
          from GDAL; without this one long WKT pushes the grid past the card. */}
      <dd
        className={`tabular font-mono text-[13px] [overflow-wrap:anywhere] ${absent ? 'text-text-lo' : ''}`}
      >
        {shown}
      </dd>
    </div>
  )
}

export function ManifestCard({
  manifest,
  previewUrl,
}: {
  manifest: InputManifest
  previewUrl?: string
}) {
  const warnings = manifest.warnings ?? []

  return (
    <article className="card-flush">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-line-soft px-4 py-2.5">
        <span className="chip bg-accent-cool text-on-accent-cool">
          {manifest.id}
        </span>
        {manifest.role && (
          <span className="font-mono text-[11px] text-text-lo">role: {manifest.role}</span>
        )}
        {/* `min-w-0` is what makes `truncate` work at all inside a flex row:
            a flex item defaults to `min-width: auto`, so a 200-character
            filename widens the card instead of ellipsising. */}
        <span className="min-w-0 flex-1 truncate text-sm" title={manifest.filename}>
          {manifest.filename}
        </span>
        <span className="tabular ml-auto shrink-0 font-mono text-[11px] text-text-lo">
          {manifest.modality} · {ratioAsPercent(manifest.modality_confidence)}
        </span>
      </div>

      <div className="flex flex-col gap-4 p-4 min-[400px]:flex-row">
        {previewUrl && (
          <img
            src={previewUrl}
            alt=""
            className="size-24 shrink-0 rounded-lg border border-line object-cover [image-rendering:pixelated]"
          />
        )}

        {/* One column on the narrowest phones: two ~150px columns cannot hold
            an EPSG string or a `256 × 256 px` size without wrapping mid-value. */}
        <dl className="grid min-w-0 flex-1 grid-cols-1 gap-x-4 gap-y-2.5 min-[400px]:grid-cols-2 sm:grid-cols-3">
          <Field label="Sensor" value={manifest.sensor_guess} />
          <Field label="CRS" value={manifest.crs} />
          <Field label="GSD" value={manifest.gsd_m ? `${decimal(manifest.gsd_m, 2)} m` : null} />
          <Field
            label="Size"
            value={`${integer(manifest.width)} × ${integer(manifest.height)} px`}
          />
          <Field label="Bands" value={integer(manifest.band_count)} />
          <Field label="dtype" value={manifest.dtype} />
          {/* §1's nulls rule cuts both ways: these were `.toFixed()` calls on
              fields the contract allows to be null, which throws inside render
              and blanks the entire pre-flight rather than showing one dash. */}
          <Field label="No-data" value={percent(manifest.nodata_pct)} />
          <Field label="Acquired" value={manifest.acquisition_time?.replace('T', ' ')} />
          <Field label="Driver" value={manifest.driver} />
        </dl>
      </div>

      {(manifest.band_names?.length ?? 0) > 0 && (
        <p className="border-t border-line-soft px-4 py-2 font-mono text-[11px] text-text-lo">
          {manifest.band_names?.join(' · ')}
        </p>
      )}

      {warnings.length > 0 && (
        <ul className="border-t border-line-soft bg-warn/8 px-4 py-2">
          {warnings.map((warning) => (
            <li key={warning} className="text-xs text-text-lo">
              <span className="text-warn-text">⚠</span> {warning}
            </li>
          ))}
        </ul>
      )}
    </article>
  )
}
