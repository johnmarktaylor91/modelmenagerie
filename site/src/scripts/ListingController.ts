/**
 * Client behaviour for /models/: search, facet filters, sort and paging over the full catalog.
 *
 * The static page's own cards stay on screen until a filtered result is ready; a failed or
 * slow index request shows an error with Retry instead of an endless loading state.
 */
import { matchesFacets } from "@/lib/facets";
import { renderCardHtml } from "@/scripts/CardView";
import { createListingLoader, loadFailureText, type LoadedIndex } from "@/scripts/IndexLoader";
import type { IndexEntry } from "@/scripts/ListingIndex";
import { rangeText, renderPagerHtml } from "@/scripts/ListingPager";
import { rememberListingUrl } from "@/scripts/ListingReturn";
import {
  FILTER_KEYS,
  activeFilterCount,
  clampPage,
  defaultSort,
  effectiveSort,
  emptyFilters,
  needsClientResults,
  parsePage,
  readListingState,
  writeListingState,
  type ListingState,
  type SortKey,
} from "@/scripts/ListingState";
import { initListingView, type ListingView } from "@/scripts/ListingView";
import { compactText } from "@/scripts/CardView";
import { parseQuery, scoreMatch } from "@/scripts/SearchMatch";

const SEARCH_DEBOUNCE_MS = 150;

interface Elements {
  root: HTMLElement;
  search: HTMLInputElement;
  form: HTMLFormElement;
  sort: HTMLSelectElement;
  filters: HTMLDetailsElement;
  filterCount: HTMLElement;
  count: HTMLElement;
  grid: HTMLElement;
  empty: HTMLElement;
  error: HTMLElement;
  errorText: HTMLElement;
  pager: HTMLElement;
  note: HTMLElement;
  family: HTMLElement;
}

function required<T extends Element>(root: ParentNode, selector: string): T {
  const element = root.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Listing element missing: ${selector}`);
  }
  return element;
}

function findElements(root: HTMLElement): Elements {
  return {
    root,
    search: required(root, "#listing-search"),
    form: required(root, "#listing-search-form"),
    sort: required(root, "#listing-sort"),
    filters: required(root, "#listing-filters"),
    filterCount: required(root, "[data-filter-count]"),
    count: required(root, "#listing-count"),
    grid: required(root, "#listing-grid"),
    empty: required(root, "#listing-empty"),
    error: required(root, "#listing-error"),
    errorText: required(root, "#listing-error-text"),
    pager: required(root, "#listing-pager"),
    note: required(root, "#listing-note"),
    family: required(root, "#listing-family"),
  };
}

/** Known numbers in the given direction, unknown last. */
function compareKnown(left: number | null, right: number | null, direction: 1 | -1): number {
  if (left === null || right === null) {
    return Number(left === null) - Number(right === null);
  }
  return direction * (left - right);
}

/** The primary comparison for a sort key; 0 leaves the order to the catalog tiebreak. */
function comparePrimary(sort: SortKey, left: { entry: IndexEntry; score: number }, right: { entry: IndexEntry; score: number }): number {
  const a = left.entry;
  const b = right.entry;
  switch (sort) {
    case "match":
      return left.score - right.score || a.card.name.length - b.card.name.length;
    case "preview":
      return Number(!a.hasPicture) - Number(!b.hasPicture);
    case "ops-desc":
      return compareKnown(a.card.nOps, b.card.nOps, -1);
    case "ops-asc":
      return compareKnown(a.card.nOps, b.card.nOps, 1);
    case "params-desc":
      return compareKnown(a.card.paramCount, b.card.paramCount, -1);
    case "params-asc":
      return compareKnown(a.card.paramCount, b.card.paramCount, 1);
    case "year-desc":
      return compareKnown(a.facets.year, b.facets.year, -1);
    case "year-asc":
      return compareKnown(a.facets.year, b.facets.year, 1);
    case "citations-desc":
      return compareKnown(a.facets.citations, b.facets.citations, -1);
    default:
      return 0;
  }
}

/** Filter, score and sort the full index for a state. */
export function computeResults(index: LoadedIndex, state: ListingState): IndexEntry[] {
  const query = parseQuery(state.q);
  const families = new Set(state.filters.family.map(compactText));
  const scored: Array<{ entry: IndexEntry; score: number }> = [];
  for (const entry of index.entries) {
    if (!matchesFacets(entry.facets, state.filters)) {
      continue;
    }
    if (families.size && !families.has(compactText(entry.familyName))) {
      continue;
    }
    const score = query ? scoreMatch(index.fields[entry.order], query) : 0;
    if (score !== null) {
      scored.push({ entry, score });
    }
  }
  const sort: SortKey = effectiveSort(state);
  scored.sort((left, right) => comparePrimary(sort, left, right) || left.entry.order - right.entry.order);
  return scored.map(({ entry }) => entry);
}

/** Text for a page number that does not exist, or "" when the address asked for none. */
function pageNote(raw: string | null, shown: number): string {
  if (raw === null || raw.trim() === String(shown)) {
    return "";
  }
  return `There is no page ${raw.trim().slice(0, 12)}. Showing page ${shown}.`;
}

/** Wire the listing page. Called once from the page's module script. */
export function initListing(root: HTMLElement): void {
  const els = findElements(root);
  const pageSize = parsePage(root.dataset.pageSize ?? null);
  const staticPage = parsePage(root.dataset.staticPage ?? null);
  const lastPage = parsePage(root.dataset.lastPage ?? null);
  const staticPath = staticPage === 1 ? "/models/" : `/models/${staticPage}/`;
  const loadIndex = createListingLoader((root.dataset.assetBase ?? "").replace(/\/+$/, ""));
  const staticView = { grid: els.grid.innerHTML, count: els.count.textContent ?? "", pager: els.pager.innerHTML };
  const firstParams = new URLSearchParams(location.search);
  let state = readListingState(firstParams);
  let pendingPageNote: string | null = firstParams.get("page");
  let renderToken = 0;

  const urlFor = (next: ListingState): string => {
    const params = writeListingState(next).toString();
    return `/models/${params ? `?${params}` : ""}`;
  };
  const staticUrl = (): string => (state.view === "list" ? `${staticPath}?view=list` : staticPath);

  initListingView(root, (view: ListingView) => {
    state = { ...state, view: view === "list" ? "list" : null };
    history.replaceState(history.state, "", needsClientResults(state) ? urlFor(state) : staticUrl());
  });

  function syncControls(): void {
    els.search.value = state.q;
    for (const input of root.querySelectorAll<HTMLInputElement>("input[data-filter]")) {
      const key = input.dataset.filter as (typeof FILTER_KEYS)[number];
      input.checked = state.filters[key]?.includes(input.value) ?? false;
    }
    els.sort.value = effectiveSort(state);
    const active = activeFilterCount(state);
    els.filterCount.textContent = active ? `(${active})` : "";
    const family = state.filters.family.join(", ");
    els.family.hidden = family === "";
    const label = els.family.querySelector<HTMLElement>("[data-family-name]");
    if (label) {
      label.textContent = family;
    }
  }

  function showNote(text: string): void {
    els.note.textContent = text;
    els.note.hidden = text === "";
  }

  function showError(message: string): void {
    els.errorText.textContent = message;
    els.error.hidden = false;
    els.grid.removeAttribute("aria-busy");
    els.count.textContent = "Search is unavailable right now";
  }

  function restoreStatic(): void {
    renderToken += 1;
    els.grid.innerHTML = staticView.grid;
    els.pager.innerHTML = staticView.pager;
    els.count.textContent = staticView.count;
    els.grid.removeAttribute("aria-busy");
    els.grid.hidden = false;
    els.empty.hidden = true;
    els.error.hidden = true;
    history.replaceState(null, "", staticUrl());
  }

  function renderResults(results: IndexEntry[], push: boolean): void {
    const asked = state.page;
    state = { ...state, page: clampPage(state.page, results.length, pageSize) };
    if (pendingPageNote !== null || asked !== state.page) {
      showNote(pageNote(pendingPageNote ?? String(asked), state.page));
    }
    pendingPageNote = null;
    const pages = Math.max(1, Math.ceil(results.length / pageSize));
    const start = (state.page - 1) * pageSize;
    const slice = results.slice(start, start + pageSize);
    els.grid.innerHTML = slice.map((entry) => renderCardHtml(entry.card)).join("");
    els.grid.hidden = slice.length === 0;
    els.grid.removeAttribute("aria-busy");
    els.empty.hidden = slice.length !== 0;
    els.error.hidden = true;
    els.count.textContent = rangeText(start + 1, start + slice.length, results.length, { one: "match", many: "matches" });
    els.pager.innerHTML = renderPagerHtml(state.page, pages, (page) => urlFor({ ...state, page }));
    const url = urlFor(state);
    if (push) {
      history.pushState(null, "", url);
    } else {
      history.replaceState(null, "", url);
    }
  }

  async function update(push = false): Promise<void> {
    syncControls();
    if (!needsClientResults(state)) {
      restoreStatic();
      return;
    }
    const token = ++renderToken;
    els.grid.setAttribute("aria-busy", "true");
    els.count.textContent = "Loading the full model list...";
    try {
      const index = await loadIndex();
      if (token === renderToken) {
        renderResults(computeResults(index, state), push);
      }
    } catch (error) {
      if (token === renderToken) {
        showError(`${loadFailureText(error)} Retry, or browse the pages below.`);
      }
    }
  }

  function setState(next: Partial<ListingState>, push = false): void {
    state = { ...state, ...next };
    showNote("");
    void update(push);
  }

  let debounce: number | undefined;
  els.search.addEventListener("input", () => {
    window.clearTimeout(debounce);
    debounce = window.setTimeout(() => setState({ q: els.search.value, page: 1 }), SEARCH_DEBOUNCE_MS);
  });
  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    window.clearTimeout(debounce);
    setState({ q: els.search.value, page: 1 });
  });
  for (const input of root.querySelectorAll<HTMLInputElement>("input[data-filter]")) {
    input.addEventListener("change", () => {
      const key = input.dataset.filter as (typeof FILTER_KEYS)[number];
      const values = new Set(state.filters[key]);
      if (input.checked) values.add(input.value);
      else values.delete(input.value);
      setState({ filters: { ...state.filters, [key]: [...values] }, page: 1 });
    });
  }
  els.sort.addEventListener("change", () => {
    const chosen = els.sort.value as SortKey;
    setState({ sort: chosen === defaultSort(state) ? null : chosen, page: 1 });
  });
  for (const button of root.querySelectorAll<HTMLButtonElement>("[data-clear-filters]")) {
    button.addEventListener("click", () => {
      setState({ q: "", filters: emptyFilters(), sort: null, page: 1 });
      els.search.focus();
    });
  }
  required<HTMLButtonElement>(root, "[data-clear-family]").addEventListener("click", () => {
    setState({ filters: { ...state.filters, family: [] }, page: 1 });
  });
  required<HTMLButtonElement>(root, "#listing-retry").addEventListener("click", () => void update());
  els.pager.addEventListener("click", (event) => {
    const link = (event.target as Element).closest<HTMLAnchorElement>("a[data-page]");
    if (!link || !needsClientResults(state) || event.metaKey || event.ctrlKey || event.shiftKey) {
      return;
    }
    event.preventDefault();
    setState({ page: parsePage(link.dataset.page ?? null) }, true);
    els.count.scrollIntoView({ block: "start" });
  });
  // The model page's back link returns to the view the visitor left.
  window.addEventListener("pagehide", rememberListingUrl);
  window.addEventListener("popstate", () => {
    if (location.pathname !== "/models/" && location.pathname !== staticPath) {
      location.reload();
      return;
    }
    state = readListingState(new URLSearchParams(location.search));
    void update();
  });
  const prefetch = () => void loadIndex().catch(() => undefined);
  els.search.addEventListener("focus", prefetch, { once: true });
  els.filters.addEventListener("pointerenter", prefetch, { once: true });

  // First load: a bare ?page=N without filters belongs to the static page N.
  if (!needsClientResults(state) && location.search) {
    if (state.page > 1 && state.page <= lastPage && state.page !== staticPage) {
      location.replace(state.view === "list" ? `/models/${state.page}/?view=list` : `/models/${state.page}/`);
      return;
    }
    showNote(pageNote(pendingPageNote, staticPage));
    pendingPageNote = null;
    history.replaceState(null, "", staticUrl());
  }
  syncControls();
  if (needsClientResults(state)) {
    void update();
  }
}
