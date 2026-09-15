/**
 * Create a project. The name field is focused and its placeholder is a real
 * example, not "Project name".
 */
import { useState, type FormEvent } from 'react'

import { COLOUR } from '@/pages/projects/colours'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { useLibraryStore, type ProjectColour } from '@/state/library'

const COLOURS: ProjectColour[] = ['terracotta', 'sand', 'ok', 'warn']

export function NewProjectDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onCreated?: (id: string) => void
}) {
  const createProject = useLibraryStore((state) => state.createProject)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [colour, setColour] = useState<ProjectColour>('terracotta')
  const [busy, setBusy] = useState(false)

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!name.trim() || busy) return
    setBusy(true)
    const project = await createProject({ name, description, colour })
    setBusy(false)
    setName('')
    setDescription('')
    onOpenChange(false)
    onCreated?.(project.id)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="!inset-auto !top-[16vh] !left-1/2 w-[min(440px,calc(100vw-1.5rem))] !-translate-x-1/2 !bg-transparent !border-0 !shadow-none">
        <form onSubmit={(event) => void submit(event)} className="glass glass-glow p-5">
          <DialogTitle className="t-panel">New project</DialogTitle>
          <DialogDescription className="t-meta mt-1">
            A folder for runs that belong to the same question.
          </DialogDescription>
          <label className="mt-4 block">
            <span className="t-eyebrow">Name</span>
            <input
              autoFocus
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Brahmaputra monsoon 2025"
              required
              className="mt-1.5 h-9 w-full rounded-lg border border-line bg-bg-main px-3 text-[13.5px] text-text-hi placeholder:text-text-lo focus:border-accent-warm focus:outline-none"
            />
          </label>
          <label className="mt-3 block">
            <span className="t-eyebrow">Description</span>
            <textarea
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              rows={2}
              placeholder="What you are trying to find out, in a sentence."
              className="mt-1.5 w-full resize-none rounded-lg border border-line bg-bg-main px-3 py-2 text-[13px] text-text-hi placeholder:text-text-lo focus:border-accent-warm focus:outline-none"
            />
          </label>
          <div className="mt-3">
            <span className="t-eyebrow">Tag</span>
            <div role="radiogroup" aria-label="Tag colour" className="mt-1.5 flex gap-2">
              {COLOURS.map((c) => (
                <button
                  key={c}
                  type="button"
                  role="radio"
                  aria-checked={colour === c}
                  aria-label={c}
                  onClick={() => setColour(c)}
                  className={`size-6 rounded-full ${COLOUR[c]} ${colour === c ? 'ring-2 ring-text-hi ring-offset-2 ring-offset-bg-main' : ''}`}
                />
              ))}
            </div>
          </div>
          <div className="mt-5 flex justify-end gap-2">
            <button type="button" onClick={() => onOpenChange(false)} className="btn-ghost">
              Cancel
            </button>
            <button type="submit" disabled={!name.trim() || busy} className="btn-primary">
              Create project
            </button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}
