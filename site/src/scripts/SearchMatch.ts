/**
 * Tokenised model search. Text is compared with punctuation, spaces and case removed, so
 * "ResNet-50", "resnet 50" and "resnet50" find the same models.
 */
import { categoryLabel } from "@/lib/labels";
import { compactText } from "@/scripts/CardView";

export interface SearchQuery {
  /** The whole query, compacted: "ResNet-50" becomes "resnet50". */
  whole: string;
  /** Each whitespace-separated word, compacted; every one must match. */
  tokens: string[];
}

export interface SearchFields {
  name: string;
  slug: string;
  family: string;
  /** All fields joined with a separator so a token cannot match across two fields. */
  haystack: string;
}

/** Score bands, best first. */
export const MATCH_EXACT = 0;
export const MATCH_PREFIX = 1;
export const MATCH_NAME = 2;
export const MATCH_OTHER = 3;

/** Parse a visitor's query; returns null for an empty query. */
export function parseQuery(raw: string): SearchQuery | null {
  const tokens = raw.split(/\s+/).map(compactText).filter(Boolean);
  if (tokens.length === 0) {
    return null;
  }
  return { whole: tokens.join(""), tokens };
}

/** Precompute the compacted search fields for one model, once per index load. */
export function searchFields(name: string, slug: string, family: string, category: string): SearchFields {
  const fields = {
    name: compactText(name),
    slug: compactText(slug),
    family: compactText(family),
  };
  const haystack = [fields.name, fields.slug, fields.family, compactText(categoryLabel(category)), compactText(category)].join("|");
  return { ...fields, haystack };
}

/** Return a score band (lower is better), or null when the model does not match. */
export function scoreMatch(fields: SearchFields, query: SearchQuery): number | null {
  const { whole } = query;
  if (fields.name === whole || fields.slug === whole) {
    return MATCH_EXACT;
  }
  if (fields.name.startsWith(whole) || fields.slug.startsWith(whole)) {
    return MATCH_PREFIX;
  }
  if (fields.name.includes(whole) || fields.slug.includes(whole)) {
    return MATCH_NAME;
  }
  if (fields.haystack.includes(whole) || query.tokens.every((token) => fields.haystack.includes(token))) {
    return MATCH_OTHER;
  }
  return null;
}
