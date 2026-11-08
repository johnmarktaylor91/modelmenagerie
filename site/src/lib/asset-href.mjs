// The one rule for addresses on the asset host, in plain JavaScript so astro.config.mjs and the dataset
// copy can use it. Site code imports `assetHref` from `lib/assets.ts`, which applies this rule to the
// build's environment.

/**
 * Appended to every address on the asset host (see gotchas.md, "cached not-found answers"). Override with
 * PUBLIC_ASSET_CACHE_KEY; an empty value turns it off.
 */
export const DEFAULT_ASSET_CACHE_KEY = "?v=1";

/**
 * The object key an asset reference points at: the last path segment, without any query or fragment.
 * @param {string} url
 * @returns {string}
 */
export function assetKeyOf(url) {
  const last = String(url ?? "").split(/[?#]/)[0].split("/").filter(Boolean).pop() ?? "";
  return last;
}

/**
 * A function from a path on the asset host (an object key such as `<sha256>.svg`, or `dataset/<v>/<file>`)
 * to its public address. With an asset base the address is `<base>/<path><cache key>`; without one (local
 * previews) it is `/assets/<path>`, served by the dev server, and carries no cache key.
 * @param {string | undefined} base PUBLIC_R2_BASE
 * @param {string | undefined} cacheKey PUBLIC_ASSET_CACHE_KEY; undefined means the default
 * @returns {(path: string) => string}
 */
export function makeAssetHref(base, cacheKey) {
  const root = String(base ?? "").replace(/\/+$/, "");
  const raw = cacheKey ?? DEFAULT_ASSET_CACHE_KEY;
  const query = raw.replace(/^[?&]/, "");
  return (path) => {
    const clean = String(path ?? "").replace(/^\/+/, "");
    if (!clean) {
      return "";
    }
    if (!root) {
      return `/assets/${clean}`;
    }
    if (!query) {
      return `${root}/${clean}`;
    }
    return `${root}/${clean}${clean.includes("?") ? "&" : "?"}${query}`;
  };
}
