/**
 * Every run this browser has kept — dense, scannable, re-openable, exportable.
 */
import { SectionHeader } from '@/components/ui/SectionHeader'
import { SavedList } from '@/pages/saved/SavedList'
import { DocumentMeta } from '@/shell/DocumentMeta'
import { useLibraryStore } from '@/state/library'

export default function Saved() {
  const count = useLibraryStore((state) => Object.keys(state.runs).length)
  const hydrated = useLibraryStore((state) => state.hydrated)
  return (
    <>
      <DocumentMeta page="saved" />
      <SectionHeader
        title="Saved"
        meta={hydrated ? `${count} ${count === 1 ? 'run' : 'runs'} · on this device` : 'on this device'}
      />
      <SavedList />
    </>
  )
}
