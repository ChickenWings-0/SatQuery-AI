/**
 * Placeholder for the local scene browser.
 *
 * The backend exposes no route over `data/processed/views/`, so there is
 * nothing honest to list yet. Saying so beats inventing a fake catalogue.
 */
export function DatasetsPanel() {
  return (
    <div className="mx-auto max-w-4xl">
      <h1 className="t-page">Datasets</h1>
      <p className="mt-1.5 text-[13px] text-text-lo">
        The local corpus under <span className="font-mono">data/processed/views/</span> —
        BigEarthNet-v2 and VRSBench scenes with their pre-rendered views.
      </p>
      <p className="mt-8 rounded-lg border border-dashed border-line p-8 text-center text-text-lo">
        Not wired up: the API has no route that lists local scenes. Until one exists, start a
        new query and drop imagery there.
      </p>
    </div>
  )
}
