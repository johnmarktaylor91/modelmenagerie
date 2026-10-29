/**
 * The listing's facets and sorts, one definition for the build (counts, which groups show) and
 * the browser (filtering). A group shows only when at least two of its values occur, so groups
 * for fields the preview export leaves empty (parameters, year, framework, licence, paper)
 * appear by themselves once the data fills them.
 *
 * Browser-safe: no Node imports.
 */
import {
  PARAM_BUCKETS,
  YEAR_BUCKETS,
  frameworkLabel,
  isPresent,
  isUsefulFacet,
  labelCategoryFacets,
  labelSizeFacets,
  paramBucket,
  sizeBucket,
  yearBucket,
  type LabelledFacet,
} from "@/lib/labels";

/** The per-model values the facets and sorts read; the compact index carries exactly these. */
export interface FacetFields {
  category: string;
  nOps: number | null;
  paramCount: number | null;
  year: number | null;
  /** Framework (`backend`), "" when unknown. */
  backend: string;
  /** SPDX licence name, "" when unknown. */
  licence: string;
  hasPaper: boolean;
  citations: number | null;
}

export const FACET_KEYS = ["category", "size", "params", "year", "framework", "licence", "paper"] as const;
export type FacetKey = (typeof FACET_KEYS)[number];

interface FacetDefinition {
  legend: string;
  /** The value a model files under, "" when it has none (left out of the counts). */
  value: (fields: FacetFields) => string;
  /** Label and order the counted values. */
  arrange: (counts: Array<{ value: string; count: number }>) => LabelledFacet[];
}

function inOrder(order: ReadonlyArray<{ value: string; label: string }>) {
  return (counts: Array<{ value: string; count: number }>): LabelledFacet[] =>
    order.flatMap((bucket) => {
      const found = counts.find((count) => count.value === bucket.value);
      return found ? [{ ...found, label: bucket.label }] : [];
    });
}

function largestFirst(label: (value: string) => string) {
  return (counts: Array<{ value: string; count: number }>): LabelledFacet[] =>
    counts.map((count) => ({ ...count, label: label(count.value) })).sort((left, right) => right.count - left.count || left.label.localeCompare(right.label));
}

export const FACETS: Readonly<Record<FacetKey, FacetDefinition>> = {
  category: { legend: "Task", value: (fields) => fields.category, arrange: labelCategoryFacets },
  size: {
    legend: "Operations",
    value: (fields) => (fields.nOps === null ? "" : sizeBucket(fields.nOps)),
    arrange: (counts) => labelSizeFacets(counts).filter((facet) => facet.value !== "unknown"),
  },
  params: { legend: "Parameters", value: (fields) => paramBucket(fields.paramCount), arrange: inOrder(PARAM_BUCKETS) },
  year: { legend: "Year", value: (fields) => yearBucket(fields.year), arrange: inOrder(YEAR_BUCKETS) },
  framework: { legend: "Framework", value: (fields) => fields.backend, arrange: largestFirst(frameworkLabel) },
  licence: { legend: "Licence", value: (fields) => fields.licence, arrange: largestFirst((value) => value) },
  paper: {
    legend: "Paper",
    value: (fields) => (fields.hasPaper ? "yes" : "no"),
    arrange: inOrder([
      { value: "yes", label: "Has a paper link" },
      { value: "no", label: "No paper link yet" },
    ]),
  },
};

export interface FacetGroup {
  key: FacetKey;
  legend: string;
  facets: LabelledFacet[];
}

/** Count every facet over the catalog and keep the groups that can narrow the results. */
export function facetGroups(models: readonly FacetFields[]): FacetGroup[] {
  return FACET_KEYS.flatMap((key) => {
    const definition = FACETS[key];
    const counts = new Map<string, number>();
    for (const fields of models) {
      const value = definition.value(fields);
      if (value) {
        counts.set(value, (counts.get(value) ?? 0) + 1);
      }
    }
    // "Has a paper" is useful only when some models have one.
    if (key === "paper" && !counts.has("yes")) {
      return [];
    }
    const facets = definition.arrange([...counts].map(([value, count]) => ({ value, count })));
    return isUsefulFacet(facets) ? [{ key, legend: definition.legend, facets }] : [];
  });
}

/** True when a model passes every selected facet (values within a group are alternatives). */
export function matchesFacets(fields: FacetFields, selected: Readonly<Partial<Record<FacetKey, readonly string[]>>>): boolean {
  return FACET_KEYS.every((key) => {
    const wanted = selected[key];
    return !wanted?.length || wanted.includes(FACETS[key].value(fields));
  });
}

export const SORT_KEYS = ["match", "preview", "name", "ops-desc", "ops-asc", "params-desc", "params-asc", "year-desc", "year-asc", "citations-desc"] as const;
export type SortKey = (typeof SORT_KEYS)[number];

/** Sort choices; one whose field no model has stays out of the menu. */
export const SORT_OPTIONS: ReadonlyArray<{ key: SortKey; label: string; needs?: (fields: FacetFields) => boolean }> = [
  { key: "match", label: "Best match" },
  { key: "preview", label: "Pictures first" },
  { key: "name", label: "Name" },
  { key: "ops-desc", label: "Most operations", needs: (fields) => fields.nOps !== null },
  { key: "ops-asc", label: "Fewest operations", needs: (fields) => fields.nOps !== null },
  { key: "params-desc", label: "Most parameters", needs: (fields) => fields.paramCount !== null },
  { key: "params-asc", label: "Fewest parameters", needs: (fields) => fields.paramCount !== null },
  { key: "year-desc", label: "Newest", needs: (fields) => fields.year !== null },
  { key: "year-asc", label: "Oldest", needs: (fields) => fields.year !== null },
  { key: "citations-desc", label: "Most cited", needs: (fields) => fields.citations !== null },
];

/** The sort options worth offering for this catalog. */
export function availableSorts(models: readonly FacetFields[]): Array<{ key: SortKey; label: string }> {
  return SORT_OPTIONS.filter((option) => !option.needs || models.some(option.needs)).map(({ key, label }) => ({ key, label }));
}

/** Facet fields from a full record (build time). */
export function facetFieldsFromRecord(model: {
  category: string;
  n_ops: number | null;
  param_count: number | null;
  year?: number | null;
  backend?: string | null;
  source_license: string;
  paper_url: string | null;
  citations?: { count: number } | null;
}): FacetFields {
  return {
    category: model.category,
    nOps: model.n_ops,
    paramCount: model.param_count,
    year: model.year ?? null,
    backend: isPresent(model.backend) ? model.backend : "",
    licence: isPresent(model.source_license) ? model.source_license : "",
    hasPaper: isPresent(model.paper_url),
    citations: model.citations?.count ?? null,
  };
}
