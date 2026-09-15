/**
 * Investigations grouped by question. `/projects/<id>` opens one; the id is
 * read from the URL here (the router only knows the section).
 */
import { useEffect, useState } from 'react'
import { useShallow } from 'zustand/shallow'

import { Graticule } from '@/components/ui/Graticule'
import { PlusIcon } from '@/components/ui/icons'
import { Reticle } from '@/components/ui/Reticle'
import { SectionHeader } from '@/components/ui/SectionHeader'
import { CardSkeleton } from '@/components/ui/Skeleton'
import { NewProjectDialog } from '@/pages/projects/NewProjectDialog'
import { ProjectCard } from '@/pages/projects/ProjectCard'
import { ProjectDetail } from '@/pages/projects/ProjectDetail'
import { useLibrary } from '@/pages/saved/useLibrary'
import { DocumentMeta } from '@/shell/DocumentMeta'
import { projectsSorted, useLibraryStore } from '@/state/library'
import { toast } from '@/state/notifications'

function idFromLocation(): string | null {
  const m = /^\/projects\/([^/]+)/.exec(window.location.pathname)
  return m?.[1] ? decodeURIComponent(m[1]) : null
}

export default function Projects() {
  const hydrated = useLibrary()
  const projects = useLibraryStore(useShallow(projectsSorted))
  const runCount = useLibraryStore((state) => Object.keys(state.runs).length)
  const deleteProject = useLibraryStore((state) => state.deleteProject)
  const restoreProject = useLibraryStore((state) => state.restoreProject)
  const [openId, setOpenId] = useState<string | null>(() => idFromLocation())
  const [creating, setCreating] = useState(false)

  useEffect(() => {
    const onPop = () => setOpenId(idFromLocation())
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  function open(id: string) {
    window.history.pushState({ section: 'projects', id }, '', `/projects/${encodeURIComponent(id)}`)
    setOpenId(id)
  }
  function back() {
    window.history.pushState({ section: 'projects' }, '', '/projects')
    setOpenId(null)
  }

  async function del(id: string) {
    const removed = await deleteProject(id, { keepRuns: true })
    if (!removed) return
    if (openId === id) back()
    toast(`“${removed.project.name}” deleted — its runs stay in Saved.`, 'info', {
      label: 'Undo',
      run: () => void restoreProject(removed.project, removed.runs),
    })
  }

  const current = openId ? useLibraryStore.getState().projects[openId] : null
  if (hydrated && openId && current) {
    return <ProjectDetail project={current} onBack={back} />
  }

  return (
    <>
      <DocumentMeta page="projects" />
      <SectionHeader
        title="Projects"
        meta={hydrated ? `${projects.length} ${projects.length === 1 ? 'project' : 'projects'} · ${runCount} runs` : undefined}
        action={
          <button type="button" onClick={() => setCreating(true)} className="btn-primary flex items-center gap-1.5 !py-1.5">
            <PlusIcon size={14} />
            New project
          </button>
        }
      />

      {!hydrated ? (
        <div className="grid grid-cols-1 gap-4 wide:grid-cols-2 desk:grid-cols-3" aria-busy="true">
          {Array.from({ length: 3 }, (_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      ) : projects.length === 0 ? (
        <div className="relative overflow-hidden rounded-xl border border-line px-6 py-20 text-center">
          <Graticule major />
          <Reticle size={64} breathe className="absolute top-10 left-[18%]" />
          <Reticle size={64} breathe className="absolute top-[45%] right-[14%]" style={{ animationDelay: '1.3s' }} />
          <Reticle size={64} breathe className="absolute bottom-8 left-[40%]" style={{ animationDelay: '2.6s' }} />
          <div className="relative mx-auto max-w-sm">
            <p className="t-page">No projects yet.</p>
            <p className="t-meta mt-2">
              A project is a folder for runs that belong to the same question — one place, one
              footprint, one report.
            </p>
            <button type="button" onClick={() => setCreating(true)} className="btn-primary mt-6 inline-flex items-center gap-1.5">
              <PlusIcon size={14} />
              Create your first project
            </button>
            <p className="t-meta mt-3">Saved runs can be filed later.</p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 wide:grid-cols-2 desk:grid-cols-3">
          {projects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onOpen={() => open(project.id)}
              onDelete={() => void del(project.id)}
            />
          ))}
        </div>
      )}

      <NewProjectDialog open={creating} onOpenChange={setCreating} onCreated={open} />
    </>
  )
}
