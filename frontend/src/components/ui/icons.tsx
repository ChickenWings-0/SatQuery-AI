/**
 * The icon set, inline.
 *
 * No icon library: the whole product needs twenty-odd glyphs, and a dependency
 * that ships two thousand to supply them is a bundle-size decision made for
 * nothing. Each icon is a 20×20 stroke drawing on `currentColor`, so it takes
 * the text colour of wherever it lands and needs no colour prop of its own.
 *
 * `aria-hidden` by default. Every icon here sits beside a text label or inside
 * a control that carries its own `aria-label`; an icon that must speak for
 * itself passes `aria-hidden={false}` and a `<title>` through `children`.
 */
import type { ReactNode, SVGProps } from 'react'

export type IconProps = SVGProps<SVGSVGElement> & { size?: number }

function Icon({
  size = 20,
  children,
  strokeWidth = 1.75,
  ...rest
}: IconProps & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...rest}
    >
      {children}
    </svg>
  )
}

/* ── navigation ─────────────────────────────────────────────────────────── */

export function HomeIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3 9.5 10 3.5l7 6V16a1 1 0 0 1-1 1h-3.5v-4.5h-5V17H4a1 1 0 0 1-1-1V9.5Z" />
    </Icon>
  )
}

export function ExploreIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="10" cy="10" r="7.25" />
      <path d="m13 7-1.75 4.25L7 13l1.75-4.25L13 7Z" fill="currentColor" stroke="none" />
    </Icon>
  )
}

export function DatasetsIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <rect x="3" y="3" width="6" height="6" rx="1.25" />
      <rect x="11" y="3" width="6" height="6" rx="1.25" />
      <rect x="3" y="11" width="6" height="6" rx="1.25" />
      <rect x="11" y="11" width="6" height="6" rx="1.25" />
    </Icon>
  )
}

export function ToolsIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12.5 3.5a3.5 3.5 0 0 0 4 4.75L9.75 15a1.75 1.75 0 0 1-2.5-2.5l6.75-6.75a3.5 3.5 0 0 0-1.5-2.25Z" />
      <path d="m4.5 15.5.75.75" />
    </Icon>
  )
}

export function UseCasesIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M7.5 14.5v-1.1a5 5 0 1 1 5 0v1.1" />
      <path d="M8 17h4M8.5 14.5h3" />
    </Icon>
  )
}

export function MapsIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="10" cy="10" r="7.25" />
      <path d="M2.75 10h14.5M10 2.75c2.25 2.25 2.25 12.25 0 14.5M10 2.75c-2.25 2.25-2.25 12.25 0 14.5" />
    </Icon>
  )
}

export function SavedIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M5 3.5h10v13l-5-3.25L5 16.5v-13Z" />
    </Icon>
  )
}

export function ProjectsIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3 5.5A1.5 1.5 0 0 1 4.5 4h3.25l1.5 1.75h6.25A1.5 1.5 0 0 1 17 7.25v7.25A1.5 1.5 0 0 1 15.5 16h-11A1.5 1.5 0 0 1 3 14.5v-9Z" />
    </Icon>
  )
}

/* ── actions ────────────────────────────────────────────────────────────── */

export function PlusIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M10 4v12M4 10h12" />
    </Icon>
  )
}

export function ShareIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="14.5" cy="5" r="2" />
      <circle cx="5.5" cy="10" r="2" />
      <circle cx="14.5" cy="15" r="2" />
      <path d="m7.3 9 5.4-3M7.3 11l5.4 3" />
    </Icon>
  )
}

export function DownloadIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M10 3v9.5M6 9l4 4 4-4M4 16.5h12" />
    </Icon>
  )
}

export function FullscreenIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3.5 7.5v-4h4M16.5 7.5v-4h-4M3.5 12.5v4h4M16.5 12.5v4h-4" />
    </Icon>
  )
}

export function ZoomInIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="9" cy="9" r="5.5" />
      <path d="m13.25 13.25 3.25 3.25M9 6.5v5M6.5 9h5" />
    </Icon>
  )
}

export function ZoomOutIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="9" cy="9" r="5.5" />
      <path d="m13.25 13.25 3.25 3.25M6.5 9h5" />
    </Icon>
  )
}

export function ResetIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 10a6 6 0 1 0 1.75-4.25" />
      <path d="M4 3.5v3h3" />
    </Icon>
  )
}

export function SettingsIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="10" cy="10" r="2.25" />
      <path d="M10 2.75v2M10 15.25v2M2.75 10h2M15.25 10h2M4.87 4.87l1.42 1.42M13.71 13.71l1.42 1.42M4.87 15.13l1.42-1.42M13.71 6.29l1.42-1.42" />
    </Icon>
  )
}

export function BellIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M5.5 13.5V9a4.5 4.5 0 0 1 9 0v4.5l1 1.5h-11l1-1.5ZM8.25 17h3.5" />
    </Icon>
  )
}

export function SendIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 10h12M11 5l5 5-5 5" />
    </Icon>
  )
}

export function CameraIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3 7.5A1.5 1.5 0 0 1 4.5 6h2l1.25-2h4.5L13.5 6h2A1.5 1.5 0 0 1 17 7.5v7A1.5 1.5 0 0 1 15.5 16h-11A1.5 1.5 0 0 1 3 14.5v-7Z" />
      <circle cx="10" cy="11" r="2.75" />
    </Icon>
  )
}

export function CopyIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <rect x="7" y="7" width="10" height="10" rx="1.5" />
      <path d="M13 7V4.5A1.5 1.5 0 0 0 11.5 3h-7A1.5 1.5 0 0 0 3 4.5v7A1.5 1.5 0 0 0 4.5 13H7" />
    </Icon>
  )
}

export function BookmarkIcon(props: IconProps) {
  return <SavedIcon {...props} />
}

export function ChevronDownIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="m5.5 8 4.5 4.5L14.5 8" />
    </Icon>
  )
}

export function CheckIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="m4.5 10.5 3.5 3.5 7.5-8" />
    </Icon>
  )
}

export function SatelliteIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="m9 11 4.5-4.5M6 7.5 9.5 4l3 3L9 10.5l-3-3ZM9.5 12.5 13 16l3.5-3.5-3-3-3.5 3.5-1-1" />
      <path d="M4 16a3 3 0 0 1 0-4.25M2.25 17.75a5.5 5.5 0 0 1 0-7.75" />
    </Icon>
  )
}

/* ── status ─────────────────────────────────────────────────────────────── */

/** A filled 8px circle that takes the text colour — the status indicator. */
export function DotIcon({ size = 8, className = '', ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 8 8"
      aria-hidden="true"
      className={`inline-block shrink-0 ${className}`}
      {...rest}
    >
      <circle cx="4" cy="4" r="4" fill="currentColor" />
    </svg>
  )
}
