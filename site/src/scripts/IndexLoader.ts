/**
 * Fetch the compact model index (and, for the listing, the picture keys) once per page, with a
 * timeout. A failure clears the cached request, so Retry fetches again.
 */
import { decodeModelIndex, parseModelIndex, parseModelThumbs, type IndexEntry, type ModelIndexFile } from "@/scripts/ListingIndex";
import { searchFields, type SearchFields } from "@/scripts/SearchMatch";

export const INDEX_URL = "/models-index.json";
export const THUMBS_URL = "/models-thumbs.json";
const TIMEOUT_MS = 20_000;

export interface LoadedIndex {
  entries: IndexEntry[];
  fields: SearchFields[];
}

async function fetchJson(url: string): Promise<unknown> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      throw new Error(`The model list returned ${response.status}`);
    }
    return await response.json();
  } finally {
    window.clearTimeout(timer);
  }
}

/** Keep one request in flight; forget it when it fails so the next call retries. */
function once<T>(load: () => Promise<T>): () => Promise<T> {
  let pending: Promise<T> | null = null;
  return () => {
    pending ??= load().catch((error: unknown) => {
      pending = null;
      throw error;
    });
    return pending;
  };
}

const loadIndexFile = once(async (): Promise<ModelIndexFile> => parseModelIndex(await fetchJson(INDEX_URL)));

function withFields(entries: IndexEntry[]): LoadedIndex {
  const fields = entries.map((entry) => searchFields(entry.card.name, entry.card.slug, entry.familyName, entry.card.category));
  return { entries, fields };
}

/** The index without pictures: enough for search suggestions. */
export const loadSearchIndex = once(async (): Promise<LoadedIndex> => withFields(decodeModelIndex(await loadIndexFile(), "")));

/** The index with picture URLs, for listing cards. */
export function createListingLoader(assetBase: string): () => Promise<LoadedIndex> {
  return once(async (): Promise<LoadedIndex> => {
    const [file, thumbs] = await Promise.all([loadIndexFile(), fetchJson(THUMBS_URL)]);
    return withFields(decodeModelIndex(file, assetBase, parseModelThumbs(thumbs, file.rows.length)));
  });
}

/** A visitor-facing sentence for a failed load. */
export function loadFailureText(error: unknown): string {
  return error instanceof DOMException && error.name === "AbortError" ? "The model list took too long to load." : "The model list could not be loaded.";
}
