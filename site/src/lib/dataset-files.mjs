// The catalog files the build serves from /dataset/<name>, shared by astro.config.mjs (which copies
// them) and the Dataset page (which links them). Plain JavaScript so the config can import it.

/** Catalog files served from /dataset/<name> on the site's own origin. */
export const DATASET_FILES = ["catalog.csv", "models.jsonl", "export_manifest.json"];

/** Cloudflare Pages refuses single files above 25 MiB; larger downloads are not copied and get no link. */
export const PAGES_MAX_FILE_BYTES = 25 * 1024 * 1024;

/** How the export names a diagram file; the copy names it by its public address instead. */
const EXPORT_ASSET_PREFIX = "/export/assets/";

/** An export diagram address and its object key. */
const EXPORT_ASSET_ADDRESS = new RegExp(`${EXPORT_ASSET_PREFIX}([A-Za-z0-9._-]+)`, "g");

/**
 * The file's text with every diagram address made public, so a downloaded file links to the diagrams
 * it lists. `assetHref` maps an object key to its public address (`lib/assets.ts`, or `makeAssetHref`
 * from `asset-href.mjs` where the site's modules are not loaded).
 * @param {string} text
 * @param {(key: string) => string} assetHref
 * @returns {string}
 */
export function publicDatasetText(text, assetHref) {
  return text.replace(EXPORT_ASSET_ADDRESS, (_match, key) => assetHref(key));
}
