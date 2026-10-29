/**
 * The listing's Cards or List view: `?view=` in the address wins (a shared link shows what its
 * sender saw), else the visitor's choice remembered in localStorage. The page's inline script
 * applies it before first paint; this module wires the toggle buttons.
 * Storage can be missing or throw (private windows, blocked site data): the view then stays
 * Cards for this visit, which is the default anyway.
 */

/** Must match the inline script in pages/models/[...page].astro. */
export const LISTING_VIEW_KEY = "menagerie:listing-view";

export type ListingView = "cards" | "list";

/** The view on screen now, as set on the root element. */
export function currentListingView(): ListingView {
  return document.documentElement.dataset.listingView === "list" ? "list" : "cards";
}

function storeListingView(view: ListingView): void {
  try {
    localStorage.setItem(LISTING_VIEW_KEY, view);
  } catch {
    // Not remembered; the choice still applies until the visitor leaves the page.
  }
}

function applyListingView(root: HTMLElement, view: ListingView): void {
  if (view === "list") {
    document.documentElement.dataset.listingView = "list";
  } else {
    delete document.documentElement.dataset.listingView;
  }
  for (const button of root.querySelectorAll<HTMLButtonElement>("button[data-listing-view]")) {
    button.setAttribute("aria-pressed", String(button.dataset.listingView === view));
  }
}

/** Sync the toggle with the view applied before paint; on click switch, remember and tell `onChange`. */
export function initListingView(root: HTMLElement, onChange: (view: ListingView) => void = () => undefined): void {
  applyListingView(root, currentListingView());
  for (const button of root.querySelectorAll<HTMLButtonElement>("button[data-listing-view]")) {
    button.addEventListener("click", () => {
      const view: ListingView = button.dataset.listingView === "list" ? "list" : "cards";
      applyListingView(root, view);
      storeListingView(view);
      onChange(view);
    });
  }
}
