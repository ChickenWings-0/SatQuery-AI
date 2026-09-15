/**
 * Popover primitives over Radix, wearing the glass.
 *
 * Every floating surface in the shell — account, notifications, the export
 * menu, the layer switcher's sheet — is this one component, so they share an
 * entrance (`sq-pop-in`, keyed off `data-sq-popover`), the terracotta edge
 * glow, and Radix's focus, `Esc` and outside-click handling.
 */
import * as PopoverPrimitive from '@radix-ui/react-popover'
import type { ComponentPropsWithoutRef, ElementRef } from 'react'
import { forwardRef } from 'react'

export const Popover = PopoverPrimitive.Root
export const PopoverTrigger = PopoverPrimitive.Trigger
export const PopoverAnchor = PopoverPrimitive.Anchor
export const PopoverClose = PopoverPrimitive.Close

export const PopoverContent = forwardRef<
  ElementRef<typeof PopoverPrimitive.Content>,
  ComponentPropsWithoutRef<typeof PopoverPrimitive.Content> & { glow?: boolean }
>(({ className = '', glow = true, sideOffset = 8, collisionPadding = 12, ...props }, ref) => (
  <PopoverPrimitive.Portal>
    <PopoverPrimitive.Content
      ref={ref}
      data-sq-popover
      sideOffset={sideOffset}
      collisionPadding={collisionPadding}
      className={`glass z-50 outline-none ${glow ? 'glass-glow' : ''} ${className}`}
      {...props}
    />
  </PopoverPrimitive.Portal>
))
PopoverContent.displayName = 'PopoverContent'
