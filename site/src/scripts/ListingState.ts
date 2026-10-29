/**
 * Listing state kept in the URL: query, facet filters, family, sort, view and page. One parser
 * serves the first load, Back/Forward and links from the header search form (`/models/?q=...`),
 * so a reloaded or shared address shows the same results.
 */
import { FACET_KEYS, SORT_KEYS, type SortKey } from "@/lib/facets";

export { SORT_KEYS, type SortKey };

/** Facet groups plus `family`, an exact family filter linked from model and family pages. */
export const FILTER_KEYS = [...FACET_KEYS, "family"] as const;
export type FilterKey = (typeof FILTER_KEYS)[number];

export type ViewKey = "cards" | "list";

export interface ListingState {
  q: string;
  filters: Record<FilterKey, string[]>;
  /** The sort the visitor chose, or null for the default. */
  sort: SortKey | null;
  /** The view in the address; null leaves it to the visitor's remembered choice. */
  view: ViewKey | null;
  page: number;
}

/** A state with no query, filters or choices. */
export function emptyFilters(): Record<FilterKey, string[]> {
  return Object.fromEntries(FILTER_KEYS.map((key) => [key, [] as string[]])) as Record<FilterKey, string[]>;
}

/** Parse a view parameter; anything else is no choice. */
export function parseView(raw: string | null): ViewKey | null {
  return raw === "list" || raw === "cards" ? raw : null;
}

/** Largest page number accepted from a URL before clamping to the real page count. */
const MAX_PAGE = 1_000_000;

/** Parse a page parameter: anything but a positive integer becomes page 1. */
export function parsePage(raw: string | null): number {
  if (raw === null || !/^\d+$/.test(raw.trim())) {
    return 1;
  }
  const page = Number(raw.trim());
  return Number.isSafeInteger(page) && page >= 1 && page <= MAX_PAGE ? page : 1;
}

/** Parse a sort parameter; unknown values fall back to the default. */
export function parseSort(raw: string | null): SortKey | null {
  return SORT_KEYS.find((key) => key === raw) ?? null;
}

/** The sort used when the visitor has not chosen one: best match while searching, pictures first otherwise. */
export function defaultSort(state: ListingState): SortKey {
  return state.q.trim() ? "match" : "preview";
}

/** The sort actually applied: the visitor's choice, or the default. */
export function effectiveSort(state: ListingState): SortKey {
  if (state.sort === "match" && !state.q.trim()) {
    return "preview";
  }
  return state.sort ?? defaultSort(state);
}

/** Read listing state from URL search parameters. A legacy `zoo` text filter folds into the query. */
export function readListingState(params: URLSearchParams): ListingState {
  const legacy = [params.get("zoo")].filter((value): value is string => Boolean(value?.trim()));
  const q = [params.get("q") ?? "", ...legacy].join(" ").trim();
  const filters = emptyFilters();
  for (const key of FILTER_KEYS) {
    filters[key] = [...new Set(params.getAll(key).map((value) => value.trim()).filter(Boolean))];
  }
  return { q, filters, sort: parseSort(params.get("sort")), view: parseView(params.get("view")), page: parsePage(params.get("page")) };
}

/** Write listing state as URL search parameters, omitting defaults. */
export function writeListingState(state: ListingState): URLSearchParams {
  const params = new URLSearchParams();
  if (state.q.trim()) {
    params.set("q", state.q.trim());
  }
  for (const key of FILTER_KEYS) {
    for (const value of state.filters[key]) {
      params.append(key, value);
    }
  }
  if (state.sort) {
    params.set("sort", state.sort);
  }
  if (state.view === "list") {
    params.set("view", "list");
  }
  if (state.page > 1) {
    params.set("page", String(state.page));
  }
  return params;
}

/** Number of facet values selected (the query is counted separately). */
export function activeFilterCount(state: ListingState): number {
  return FILTER_KEYS.reduce((total, key) => total + state.filters[key].length, 0);
}

/** True when the listing must be computed in the browser rather than shown from a static page. */
export function needsClientResults(state: ListingState): boolean {
  return Boolean(state.q.trim()) || activeFilterCount(state) > 0 || state.sort !== null;
}

/** Clamp a page to the range that exists for `total` results. */
export function clampPage(page: number, total: number, pageSize: number): number {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  return Math.min(Math.max(1, page), pages);
}
