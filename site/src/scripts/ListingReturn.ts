/**
 * Remember the last listing view (search, filters, sort, page) so a model page's back link
 * returns to it instead of the unfiltered first page.
 */

const STORAGE_KEY = "menagerie:listing";

/** Only listing addresses are accepted: /models/ or /models/N/, with an optional query. */
const LISTING_PATH = /^\/models\/(?:\d+\/)?(?:\?[^#]*)?$/;

/** Store the current listing address; storage may be unavailable, which only loses the convenience. */
export function rememberListingUrl(): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, location.pathname + location.search);
  } catch {
    // Private mode or blocked storage: the back link keeps its static /models/ target.
  }
}

/** Return the remembered listing address, or null when there is none or it is not a listing page. */
export function rememberedListingUrl(): string | null {
  try {
    const value = sessionStorage.getItem(STORAGE_KEY);
    return value && LISTING_PATH.test(value) ? value : null;
  } catch {
    return null;
  }
}

/** True when the visitor came here from a listing or another model page on this site. */
function cameFromModels(): boolean {
  try {
    const referrer = new URL(document.referrer);
    return referrer.origin === location.origin && referrer.pathname.startsWith("/models/");
  } catch {
    return false;
  }
}

/**
 * Point back links at the remembered listing when the visitor arrived from the listing (or walked
 * Previous and Next from it): "Back to results" for a search or filter, "Back to page N" for a
 * later static page.
 */
export function restoreBackLinks(selector = "[data-listing-back]"): void {
  const target = cameFromModels() ? rememberedListingUrl() : null;
  if (!target || target === "/models/") {
    return;
  }
  document.querySelectorAll<HTMLAnchorElement>(selector).forEach((link) => {
    link.href = target;
    const staticPage = /^\/models\/(\d+)\/$/.exec(target);
    if (target.includes("?")) {
      link.textContent = "Back to results";
    } else if (staticPage) {
      link.textContent = `Back to page ${staticPage[1]}`;
    }
  });
}
