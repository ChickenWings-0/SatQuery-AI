import { useEffect } from 'react'

import { useLibraryStore } from '@/state/library'

/** Hydrate the library from IndexedDB once; returns whether it has. */
export function useLibrary(): boolean {
  const hydrate = useLibraryStore((state) => state.hydrate)
  const hydrated = useLibraryStore((state) => state.hydrated)
  useEffect(() => {
    void hydrate()
  }, [hydrate])
  return hydrated
}
