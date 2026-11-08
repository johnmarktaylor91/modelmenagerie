/**
 * Addresses of files on the asset host (diagram SVGs, card pictures, PDFs, share images, dataset
 * downloads). Every such address is built here, so the cache key reaches all of them; the rule itself is
 * in `asset-href.mjs`, shared with astro.config.mjs.
 */
import { assetKeyOf, makeAssetHref } from "@/lib/asset-href.mjs";

/** PUBLIC_R2_BASE without a trailing slash; empty for local previews. */
export const ASSET_BASE = ((import.meta.env.PUBLIC_R2_BASE as string | undefined) ?? "").replace(/\/+$/, "");

/** The asset host's origin, for the preconnect hint (no cache key); empty for local previews. */
export const ASSET_ORIGIN = ASSET_BASE ? new URL(ASSET_BASE).origin : "";

/** Public address of a path on the asset host, with the cache key appended. */
export const assetHref: (path: string) => string = makeAssetHref(
  ASSET_BASE,
  import.meta.env.PUBLIC_ASSET_CACHE_KEY as string | undefined,
);

/** Public address of an export asset reference (`/export/assets/<sha256>.<ext>` or a bare key). */
export function assetRefHref(url: string): string {
  const key = assetKeyOf(url);
  if (!key) {
    throw new Error(`Asset reference has no key: ${url}`);
  }
  return assetHref(key);
}
