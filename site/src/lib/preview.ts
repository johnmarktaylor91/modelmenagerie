/**
 * What counts as a usable preview image and as a heavy diagram. One rule for cards, the home page
 * and the diagram viewer.
 *
 * Browser-safe: no Node imports, so client scripts can share it with build-time pages.
 */

/** SVGs above this size (or flagged monster) load only when the reader asks. */
export const HEAVY_SVG_BYTES = 1_000_000;

/** Thumbnails smaller than this are near blank (a huge graph squeezed to 400 px wide) or "Thumbnail unavailable" images. */
export const MIN_PREVIEW_THUMB_BYTES = 2_000;

/** Return true when a thumbnail of this size is a real preview. */
export function hasUsablePreview(thumbBytes: number): boolean {
  return thumbBytes >= MIN_PREVIEW_THUMB_BYTES;
}

/** True when a variant's SVG should wait for an explicit load (monster flag or byte size). */
export function isHeavyVariant(variant: { monster: boolean; svg: { bytes: number } }): boolean {
  return variant.monster || variant.svg.bytes > HEAVY_SVG_BYTES;
}

/** Cards and home panels show the SVG instead of the thumbnail when it is this small or smaller. */
export const MAX_CARD_SVG_BYTES = 100_000;

interface AssetLike {
  url: string;
  bytes: number;
}

/**
 * The picture a card or home panel shows: the SVG when it is no larger than the thumbnail and
 * small (sharp at any size, and lighter), else the thumbnail. Both draw the whole graph from the top.
 */
export function cardImage(variant: { monster: boolean; svg: AssetLike; thumb: AssetLike }): AssetLike {
  const svgFits = !variant.monster && variant.svg.bytes <= MAX_CARD_SVG_BYTES && variant.svg.bytes <= variant.thumb.bytes;
  return svgFits ? variant.svg : variant.thumb;
}
