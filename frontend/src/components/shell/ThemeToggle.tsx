/**
 * Dark / light / system.
 *
 * Two shapes: the three-way segmented control for the account popover, and a
 * single icon button for the landing nav. Both drive `useThemeStore`; the
 * icon is one drawing whose moon-to-sun change is a 220 ms crossfade rather
 * than two icons swapping, so the control reads as a state, not a menu.
 */
import { ContrastIcon, MoonIcon, SunIcon } from '@/components/ui/icons'
import { Segmented } from '@/components/ui/Segmented'
import { useThemeStore, type ThemePref } from '@/state/theme'

export function ThemeSegmented() {
  const pref = useThemeStore((state) => state.pref)
  const setPref = useThemeStore((state) => state.setPref)
  return (
    <Segmented<ThemePref>
      size="sm"
      label="Theme"
      value={pref}
      onChange={setPref}
      options={[
        { value: 'dark', label: <MoonIcon size={13} />, title: 'Dark' },
        { value: 'light', label: <SunIcon size={13} />, title: 'Light' },
        { value: 'system', label: <ContrastIcon size={13} />, title: 'Follow system' },
      ]}
    />
  )
}

export function ThemeButton({ className = '' }: { className?: string }) {
  const resolved = useThemeStore((state) => state.resolved)
  const toggle = useThemeStore((state) => state.toggle)
  const dark = resolved === 'dark'
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
      aria-pressed={!dark}
      title={`${dark ? 'Light' : 'Dark'} theme  ⌘J`}
      className={`relative grid size-9 place-items-center rounded-full text-text-lo transition-colors duration-[120ms] hover:text-text-hi ${className}`}
    >
      <span
        aria-hidden
        className="absolute inset-0 grid place-items-center transition-opacity duration-[220ms]"
        style={{ opacity: dark ? 1 : 0 }}
      >
        <MoonIcon size={16} />
      </span>
      <span
        aria-hidden
        className="absolute inset-0 grid place-items-center transition-opacity duration-[220ms]"
        style={{ opacity: dark ? 0 : 1 }}
      >
        <SunIcon size={16} />
      </span>
    </button>
  )
}
