import fs from "node:fs";
import path from "node:path";

import { hasUsablePreview } from "@/lib/preview";
import type { AssetRef, Funnel, Manifest, ModelRecord, ModelVariant } from "@/types/record";
export { HEAVY_SVG_BYTES, MIN_PREVIEW_THUMB_BYTES, hasUsablePreview, isHeavyVariant } from "@/lib/preview";

const siteDataRoot = path.join(process.cwd(), "site-data");
const modelsJsonlPath = path.join(siteDataRoot, "models.jsonl");
const publicR2Base = (import.meta.env.PUBLIC_R2_BASE as string | undefined)?.replace(/\/+$/, "") ?? "";

let catalogModels: ModelRecord[] | null = null;
let catalogBySlug: Map<string, ModelRecord> | null = null;
let sortedSlugs: readonly string[] | null = null;
let categoryNeighbours: Map<string, { prev: ModelRecord | null; next: ModelRecord | null }> | null = null;
let catalogFacets: CatalogIndex["facets"] | null = null;
let diagramShares: Map<string, { count: number; families: number }> | null = null;
let listingOrder: ModelRecord[] | null = null;
let diagramFocus: Map<string, number> | null = null;

/** A diagram shared by models from at least this many families is a stand-in, not the model's own graph. */
export const STAND_IN_MIN_FAMILIES = 10;

export type {
  AssetRef,
  FamilyRecord,
  Funnel,
  Manifest,
  ModelRecord,
  ModelVariant,
  SizedAssetRef,
  TlspecRef,
  ValidationState,
} from "@/types/record";

export interface CatalogFacet {
  value: string;
  count: number;
}

export interface CatalogIndex {
  models: ModelRecord[];
  facets: {
    categories: CatalogFacet[];
    sizes: CatalogFacet[];
    domains: CatalogFacet[];
    families: CatalogFacet[];
    zoos: CatalogFacet[];
    tiers: CatalogFacet[];
  };
}

/** Return the canonical committed site data root. */
export function getExportRoot(): string {
  return siteDataRoot;
}

/** Read a JSON file from committed site data. */
function readJson<T>(relativePath: string): T {
  return JSON.parse(fs.readFileSync(path.join(siteDataRoot, relativePath), "utf-8")) as T;
}

/** Ensure committed catalog data exists before a page attempts to build. */
export function assertFixtureReady(): void {
  if (!fs.existsSync(modelsJsonlPath)) {
    throw new Error(
      "Committed site data missing. Run `python scripts/sync_site_data.py --export-dir <generated-catalog-dir> --site-data ./site-data` after building the catalog data.",
    );
  }
}

/** Return the asset basename used as the R2 object key. */
function assetKey(ref: AssetRef): string {
  const key = ref.url.split("/").filter(Boolean).pop();
  if (!key) {
    throw new Error(`Asset reference has no key: ${ref.url}`);
  }
  return key;
}

/** Return a build-time public URL for a content-addressed asset. */
function publicAssetUrl(ref: AssetRef): string {
  const key = assetKey(ref);
  return publicR2Base ? `${publicR2Base}/${key}` : `/assets/${key}`;
}

/** Return an asset reference with its URL rewritten for public delivery. */
function withPublicAssetUrl<T extends AssetRef>(ref: T): T {
  return {
    ...ref,
    url: publicAssetUrl(ref),
  };
}

/** Rewrite an optional asset reference; absent and null both stay null. */
function optionalPublicAsset<T extends AssetRef>(ref: T | null | undefined): T | null {
  return ref ? withPublicAssetUrl(ref) : null;
}

/** Return a model variant with public asset URLs. */
function withPublicVariantAssetUrls(variant: ModelVariant): ModelVariant {
  return {
    ...variant,
    svg: withPublicAssetUrl(variant.svg),
    thumb: withPublicAssetUrl(variant.thumb),
    pdf: optionalPublicAsset(variant.pdf),
    thumb_crop: optionalPublicAsset(variant.thumb_crop),
  };
}

/** Return a model record with public asset URLs. Exported for fixture records. */
export function withPublicModelAssetUrls(model: ModelRecord): ModelRecord {
  return {
    ...model,
    variants: model.variants.map(withPublicVariantAssetUrls),
    tlspec: optionalPublicAsset(model.tlspec),
    interactive: optionalPublicAsset(model.interactive),
    netron: optionalPublicAsset(model.netron),
    model_explorer: optionalPublicAsset(model.model_explorer),
    share_image: optionalPublicAsset(model.share_image),
  };
}

/** Load all model records from committed JSON Lines data once. */
function loadCatalogModels(): ModelRecord[] {
  assertFixtureReady();
  if (catalogModels === null) {
    catalogModels = fs
      .readFileSync(modelsJsonlPath, "utf-8")
      .split("\n")
      .filter(Boolean)
      .map((line) => withPublicModelAssetUrls(JSON.parse(line) as ModelRecord));
  }
  return catalogModels;
}

/** Return the slug lookup map for committed JSON Lines data. */
function loadCatalogBySlug(): Map<string, ModelRecord> {
  if (catalogBySlug === null) {
    catalogBySlug = new Map(loadCatalogModels().map((model) => [model.slug, model]));
  }
  return catalogBySlug;
}

/** Return all model slugs, sorted once per build. */
export function listModelSlugs(): string[] {
  if (sortedSlugs === null) {
    sortedSlugs = loadCatalogModels()
      .map((model) => model.slug)
      .sort();
  }
  return [...sortedSlugs];
}

/** Load one model by slug from the in-memory JSON Lines index. */
export function getModelBySlug(slug: string): ModelRecord {
  const model = loadCatalogBySlug().get(slug);
  if (!model) {
    throw new Error(`Model not found: ${slug}`);
  }
  return model;
}

/** Load the complete catalog for aggregate pages and search bootstrap data. */
export function getAllModels(): ModelRecord[] {
  return loadCatalogModels();
}

/** Load funnel counts. */
export function getFunnel(): Funnel {
  assertFixtureReady();
  return readJson<Funnel>("funnel.json");
}

/** Load export manifest metadata. */
export function getManifest(): Manifest {
  assertFixtureReady();
  return readJson<Manifest>("export_manifest.json");
}

/** Load the precomputed catalog index and attach the model list. */
export function getCatalogIndex(): CatalogIndex {
  assertFixtureReady();
  if (catalogFacets === null) {
    catalogFacets = readJson<CatalogIndex["facets"]>("facets.json");
  }
  return {
    models: getAllModels(),
    facets: catalogFacets,
  };
}

/** Return the default variant for a model. Every exported record has one; see `findDefaultVariant` for records that may not. */
export function getDefaultVariant(model: ModelRecord): ModelVariant {
  const variant = findDefaultVariant(model);
  if (!variant) {
    throw new Error(`Model has no diagram variants: ${model.slug}`);
  }
  return variant;
}

/** Return the default variant, or null for a record without any diagram. */
export function findDefaultVariant(model: ModelRecord): ModelVariant | null {
  return model.variants.find((variant) => variant.key === model.default_variant) ?? model.variants[0] ?? null;
}

/** Return one variant by key, falling back to the default variant. */
export function getVariantByKey(model: ModelRecord, key: string | null | undefined): ModelVariant {
  return model.variants.find((variant) => variant.key === key) ?? getDefaultVariant(model);
}

/** Order models for reading: display name, then slug as a stable tiebreak. */
function compareByName(left: ModelRecord, right: ModelRecord): number {
  return left.display_name.localeCompare(right.display_name, "en", { sensitivity: "base" }) || left.slug.localeCompare(right.slug);
}

/** Build every model's previous and next neighbour within its category once per build. */
function loadCategoryNeighbours(): Map<string, { prev: ModelRecord | null; next: ModelRecord | null }> {
  if (categoryNeighbours === null) {
    const byCategory = new Map<string, ModelRecord[]>();
    for (const model of loadCatalogModels()) {
      const group = byCategory.get(model.category) ?? [];
      group.push(model);
      byCategory.set(model.category, group);
    }
    categoryNeighbours = new Map();
    for (const group of byCategory.values()) {
      group.sort(compareByName);
      group.forEach((model, index) => {
        categoryNeighbours?.set(model.slug, { prev: group[index - 1] ?? null, next: group[index + 1] ?? null });
      });
    }
  }
  return categoryNeighbours;
}

/** Return the previous and next models in the same category, in display-name order. */
export function getAdjacentModels(slug: string): { prev: ModelRecord | null; next: ModelRecord | null } {
  return loadCategoryNeighbours().get(slug) ?? { prev: null, next: null };
}

/** Count, per SVG hash, how many models and distinct families use that exact diagram. */
function loadDiagramShares(): Map<string, { count: number; families: number }> {
  if (diagramShares === null) {
    const families = new Map<string, Set<string>>();
    const counts = new Map<string, number>();
    for (const model of loadCatalogModels()) {
      for (const variant of model.variants) {
        const hash = variant.svg.sha256;
        counts.set(hash, (counts.get(hash) ?? 0) + 1);
        const set = families.get(hash) ?? new Set<string>();
        set.add(model.family_normalized || model.slug);
        families.set(hash, set);
      }
    }
    diagramShares = new Map([...counts].map(([hash, count]) => [hash, { count, families: families.get(hash)?.size ?? 0 }]));
  }
  return diagramShares;
}

/**
 * Return how many catalog entries share this variant's exact diagram when it is a stand-in
 * (the same graph drawn for models from many unrelated families), else 0.
 */
export function getStandInShareCount(variant: ModelVariant | null): number {
  if (!variant) {
    return 0;
  }
  const share = loadDiagramShares().get(variant.svg.sha256);
  return share && share.families >= STAND_IN_MIN_FAMILIES ? share.count : 0;
}

/** True when a model's card shows its own picture: a usable thumbnail that is not a stand-in. */
export function hasCardPreview(model: ModelRecord): boolean {
  if (model.own_diagram === false) {
    return false;
  }
  const variant = findDefaultVariant(model);
  return variant !== null && getStandInShareCount(variant) === 0 && (Boolean(variant.thumb_crop) || hasUsablePreview(variant.thumb.bytes));
}

/**
 * The default listing order: models with a picture first, each group in catalog (name) order.
 * The listing's "Pictures first" sort reproduces it in the browser from the index.
 */
export function getListingOrder(): ModelRecord[] {
  if (listingOrder === null) {
    const models = loadCatalogModels();
    listingOrder = [...models.filter(hasCardPreview), ...models.filter((model) => !hasCardPreview(model))];
  }
  return listingOrder;
}

/**
 * Where a wide diagram's drawing starts, as a fraction of its width (see scripts/diagram_focus.py),
 * or null when the viewer should centre it. A missing focus file means every diagram is centred.
 */
export function getDiagramFocus(variant: ModelVariant): number | null {
  if (diagramFocus === null) {
    const file = path.join(getExportRoot(), "diagram_focus.json");
    const data = fs.existsSync(file) ? (JSON.parse(fs.readFileSync(file, "utf-8")) as { focus?: Record<string, number> }) : {};
    diagramFocus = new Map(Object.entries(data.focus ?? {}));
  }
  return diagramFocus.get(variant.svg.sha256) ?? null;
}

/** Plain-word label for a diagram variant, for example "Unrolled, all layers". */
export function variantLabel(variant: ModelVariant): string {
  const mode = variant.vis_mode === "rolled" ? "Rolled" : "Unrolled";
  const collapse = { none: "all layers", auto: "grouped layers", max: "collapsed layers" }[variant.collapse] ?? variant.collapse;
  return `${mode}, ${collapse}`;
}

/** Format large integers compactly for UI metadata. */
export function formatCount(value: number | null): string {
  if (value === null) {
    return "unknown";
  }
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

/** Format a byte count for people, for example 4.4 MB or 95 KB. */
export function formatBytes(bytes: number): string {
  if (bytes >= 1_000_000) {
    return `${(bytes / 1_000_000).toFixed(1)} MB`;
  }
  return `${Math.max(1, Math.round(bytes / 1000))} KB`;
}
