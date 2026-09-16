/**
 * One project: its runs (the shared list, filtered), the union footprint, and
 * the print route. The description edits in place and saves on blur.
 */
import { useState } from 'react'
import { useShallow } from 'zustand/shallow'

import { downloadJson } from '@/export/download'
import { footprintToGeoJson } from '@/export/geojson/build'
import { ArrowRightIcon, DownloadIcon, FileIcon } from '@/components/ui/icons'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { SectionHeader } from '@/components/ui/SectionHeader'
import { COLOUR } from '@/pages/projects/colours'
import { SavedList } from '@/pages/saved/SavedList'
import { DocumentMeta } from '@/shell/DocumentMeta'
import { runsInProject, useLibraryStore, type Project } from '@/state/library'
import { toast } from '@/state/notifications'

export function ProjectDetail({ project, onBack }: { project: Project; onBack: () => void }) {
  const runs = useLibraryStore(useShallow((state) => runsInProject(state, project.id)))
  const updateProject = useLibraryStore((state) => state.updateProject)
  const [description, setDescription] = useState(project.description)

  function exportFootprint() {
    if (!project.aoi) return
    downloadJson(
      `satquery-project-${project.name.toLowerCase().replace(/\s+/g, '-')}.geojson`,
      footprintToGeoJson(project.aoi, { project: project.name, runs: runs.map((r) => r.traceId) }),
    )
    toast('Project footprint exported.', 'ok')
  }

  return (
    <>
      <DocumentMeta page="projects" subject={project.name} />
      <SectionHeader
        title={project.name}
        crumb={
          <>
            <button type="button" onClick={onBack} className="hover:text-text-hi">
              Projects
            </button>
            <span aria-hidden>/</span>
          </>
        }
        meta={`${runs.length} ${runs.length === 1 ? 'run' : 'runs'}`}
        action={
          <Popover>
            <PopoverTrigger asChild>
              <button type="button" className="btn-ghost flex items-center gap-1.5 !py-1.5">
                <DownloadIcon size={14} />
                Export
              </button>
            </PopoverTrigger>
            <PopoverContent align="end" className="w-56 p-1.5" aria-label="Export project">
              <button
                type="button"
                className="menu-row disabled:opacity-40"
                disabled={!project.aoi}
                title={project.aoi ? undefined : 'No member run carries a georeference'}
                onClick={exportFootprint}
              >
                <FileIcon size={15} className="text-text-lo" />
                Footprint GeoJSON
              </button>
              <a
                className="menu-row"
                href={`/report/project/${project.id}`}
                target="_blank"
                rel="noopener"
                aria-label="Project report, opens in a new tab"
              >
                <FileIcon size={15} className="text-text-lo" />
                PDF report
                <ArrowRightIcon size={13} className="ml-auto -rotate-45 text-text-lo" />
              </a>
            </PopoverContent>
          </Popover>
        }
      >
        <div className="flex items-start gap-3">
          <span aria-hidden className={`mt-1.5 h-4 w-1 shrink-0 rounded-full ${COLOUR[project.colour]}`} />
          <textarea
            aria-label="Project description"
            value={description}
            rows={2}
            placeholder="What you are trying to find out, in a sentence."
            onChange={(event) => setDescription(event.target.value)}
            onBlur={() => {
              if (description !== project.description) void updateProject(project.id, { description })
            }}
            className="t-lede w-full resize-none rounded-md bg-transparent px-1 py-0.5 text-text-hi placeholder:text-text-lo hover:bg-surface-card focus:bg-surface-card focus:outline-none"
          />
        </div>
      </SectionHeader>
      <SavedList
        projectId={project.id}
        emptyTitle="No runs filed here yet."
        emptyBody="Save a run from the answer card, then add it to this project from Saved."
      />
    </>
  )
}
