/**
 * The discovery column: place → dates → sensor → shelf → analyse.
 *
 * Gated. With online features off (the default, for judging halls with no
 * route out) and not in mock mode, this renders one sentence and a button
 * to Settings; no request is ever constructed, not even a DNS prefetch.
 * When the network drops mid-session the `offline` event is what the
 * store listens to, and one automatic retry runs when it comes back.
 */
import { useEffect } from 'react'

import { CloseIcon, SettingsIcon } from '@/components/ui/icons'
import { mockRequested } from '@/mocks/mode'
import { CollectionPicker } from '@/pages/maps/hud/CollectionPicker'
import { FetchBar } from '@/pages/maps/hud/FetchBar'
import { PlaceSearch } from '@/pages/maps/hud/PlaceSearch'
import { SceneShelf } from '@/pages/maps/hud/SceneShelf'
import { TimeSlider } from '@/pages/maps/hud/TimeSlider'
import { useSettingsStore } from '@/state/settings'
import { useStacStore } from '@/state/stac'

function discoveryAllowed(onlineFeatures: boolean): boolean {
  return onlineFeatures || mockRequested()
}

export function Discover({ onClose }: { onClose: () => void }) {
  const onlineFeatures = useSettingsStore((state) => state.onlineFeatures)
  const openSettings = useSettingsStore((state) => state.openSettings)
  const markOnline = useStacStore((state) => state.markOnline)
  const allowed = discoveryAllowed(onlineFeatures)

  useEffect(() => {
    if (!allowed) return
    window.addEventListener('online', markOnline)
    return () => window.removeEventListener('online', markOnline)
  }, [allowed, markOnline])

  return (
    <aside
      aria-label="Find imagery"
      className="glass glass-glow flex max-h-full w-full flex-col gap-3 p-3 wide:w-[380px]"
    >
      <div className="flex items-center gap-2">
        <h2 className="t-panel">Find imagery</h2>
        <span className="t-coord text-text-lo">Sentinel · Planetary Computer</span>
        <button type="button" onClick={onClose} aria-label="Close imagery search" className="ml-auto grid size-7 place-items-center rounded-md text-text-lo hover:bg-sidebar-hi hover:text-text-hi">
          <CloseIcon size={14} />
        </button>
      </div>

      {allowed ? (
        <>
          <PlaceSearch autoFocus />
          <TimeSlider />
          <CollectionPicker />
          <SceneShelf />
          <FetchBar />
        </>
      ) : (
        <div className="rounded-lg border border-line-soft p-3">
          <p className="text-[13px] text-text-hi">Imagery search is off.</p>
          <p className="t-meta mt-1">
            Turn on online features in Settings to search Sentinel scenes by place and date. Nothing leaves this machine until you do.
          </p>
          <button type="button" onClick={() => openSettings('data')} className="btn-ghost-sm mt-3 inline-flex items-center gap-1.5">
            <SettingsIcon size={13} />
            Open settings
          </button>
        </div>
      )}
    </aside>
  )
}
