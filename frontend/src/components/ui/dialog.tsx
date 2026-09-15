/**
 * Dialog primitives in the shadcn/ui shape, over Radix.
 *
 * Written here rather than pulled through the shadcn CLI: the CLI wants a
 * `components.json` and its own Tailwind conventions, and all it does is copy
 * this file in. Radix gives the parts that are genuinely hard — focus trapping,
 * scroll locking, `Esc`, and the `aria-modal` wiring a full-screen surface needs.
 */
import * as DialogPrimitive from '@radix-ui/react-dialog'
import type { ComponentPropsWithoutRef, ElementRef } from 'react'
import { forwardRef } from 'react'

export const Dialog = DialogPrimitive.Root
export const DialogTrigger = DialogPrimitive.Trigger
export const DialogClose = DialogPrimitive.Close
export const DialogTitle = DialogPrimitive.Title
export const DialogDescription = DialogPrimitive.Description

export const DialogOverlay = forwardRef<
  ElementRef<typeof DialogPrimitive.Overlay>,
  ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className = '', ...props }, ref) => (
  <DialogPrimitive.Overlay
    ref={ref}
    // `animate-in fade-in-0` were `tailwindcss-animate` utilities and that
    // plugin is not installed, so the dialog had no entrance at all — two dead
    // class names doing nothing. The real keyframes are in `theme.css`, keyed
    // off this attribute so `prefers-reduced-motion` can override them.
    data-sq-overlay
    className={`fixed inset-0 z-50 bg-[var(--color-scrim)] backdrop-blur-[2px] ${className}`}
    {...props}
  />
))
DialogOverlay.displayName = 'DialogOverlay'

/** Full-screen content, inset slightly so the page stays visible behind it. */
export const DialogContent = forwardRef<
  ElementRef<typeof DialogPrimitive.Content>,
  ComponentPropsWithoutRef<typeof DialogPrimitive.Content>
>(({ className = '', children, ...props }, ref) => (
  <DialogPrimitive.Portal>
    <DialogOverlay />
    <DialogPrimitive.Content
      ref={ref}
      // Entrance only. `PipelineDialog` gates its content on `open`, so the
      // element is gone before an exit animation could run; promising one in CSS
      // would be a lie the DOM never gets to tell.
      data-sq-dialog
      className={`fixed inset-3 z-50 flex flex-col overflow-hidden rounded-xl border border-line bg-bg-main shadow-2xl outline-none sm:inset-6 ${className}`}
      {...props}
    >
      {children}
    </DialogPrimitive.Content>
  </DialogPrimitive.Portal>
))
DialogContent.displayName = 'DialogContent'
