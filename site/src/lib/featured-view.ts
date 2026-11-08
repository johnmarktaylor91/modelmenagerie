/**
 * The picture window a featured model's card shows: a hand-chosen, readable part of its own SVG
 * (`FEATURED_VIEWS` in `featured.ts`), drawn by CSS so it stays sharp at any size and needs no new
 * asset. Used by the home panels and the listing cards, at build time and in the browser.
 *
 * Browser-safe: imports only the lists in `featured.ts`.
 */
import { FEATURED_VIEWS } from "@/lib/featured";

/** The SVG at `width` percent of the frame, shifted by `left` and `top` percent of the frame's width. */
export interface PictureWindow {
  width: number;
  left: number;
  top: number;
}

/** An SVG whose size moved by more than this was redrawn since its window was chosen: the window no longer fits. */
const SIZE_TOLERANCE_PX = 2;

function round(value: number): number {
  return Math.round(value * 100) / 100;
}

/**
 * The window for a model's default diagram, or null when the model has none or its SVG changed
 * size since the window was chosen (the card then shows its ordinary picture).
 */
export function featuredWindow(slug: string, variant: { svg_width_px?: number | null; svg_height_px?: number | null } | null | undefined): PictureWindow | null {
  const view = FEATURED_VIEWS[slug];
  const width = variant?.svg_width_px;
  const height = variant?.svg_height_px;
  if (!view || typeof width !== "number" || typeof height !== "number") {
    return null;
  }
  const [expectedWidth, expectedHeight] = view.svg;
  if (Math.abs(width - expectedWidth) > SIZE_TOLERANCE_PX || Math.abs(height - expectedHeight) > SIZE_TOLERANCE_PX) {
    return null;
  }
  const [x, y, w] = view.crop;
  return { width: round((width / w) * 100), left: round((-x / w) * 100), top: round((-y / w) * 100) };
}

/**
 * Inline style placing the SVG in its frame. Percentage margins resolve against the frame's width,
 * so one rule fits any frame size and aspect ratio (the window's top-left corner is pinned).
 */
export function windowStyle(win: PictureWindow): string {
  return `width:${win.width}%;margin-left:${win.left}%;margin-top:${win.top}%`;
}
