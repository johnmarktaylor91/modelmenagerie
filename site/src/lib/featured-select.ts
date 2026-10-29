/**
 * Picks the home page's pictures from the lists in `featured.ts`: one hero diagram drawn large, and a
 * row of small panels of other well-known architectures, with a deterministic fallback when a slug
 * leaves the catalog.
 */
import type { ModelRecord, ModelVariant } from "@/lib/catalog";
import { FEATURED_SLUGS, HERO_SLUGS } from "@/lib/featured";
import { cardImage, hasUsablePreview, isHeavyVariant } from "@/lib/preview";

/** Hero SVGs above this size fall back to the thumbnail. */
const HERO_MAX_SVG_BYTES = 150_000;

/** Fallback picks stay in a size range whose top-of-graph crop is readable. */
const FALLBACK_MIN_OPS = 20;
const FALLBACK_MAX_OPS = 400;

/**
 * Thumbnails are 400 px wide renders of the whole graph, so a tall graph's file can be several
 * hundred KB while a panel shows only its top. A model whose panel picture (see `cardImage`) is
 * larger than this is skipped so the home page stays light (vit-b-16's is 195 KB); a
 * top-of-graph crop from the renderer would remove the need.
 */
const MAX_FEATURED_IMAGE_BYTES = 100_000;

/** A thumbnail shared by more records than this is a stand-in, not that model's graph. */
const MAX_SHARED_THUMB_USES = 3;

function defaultVariant(model: ModelRecord): ModelVariant | undefined {
  return model.variants.find((variant) => variant.key === model.default_variant) ?? model.variants[0];
}

function thumbUseCounts(models: readonly ModelRecord[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const model of models) {
    const sha = defaultVariant(model)?.thumb.sha256;
    if (sha) {
      counts.set(sha, (counts.get(sha) ?? 0) + 1);
    }
  }
  return counts;
}

/** True when a model can stand on the home page: its own, real, not oversized preview. */
function isShowcaseable(model: ModelRecord, thumbUses: Map<string, number>): boolean {
  const variant = defaultVariant(model);
  if (!variant || model.quarantine || model.monster_graph || isHeavyVariant(variant)) {
    return false;
  }
  return (
    hasUsablePreview(variant.thumb.bytes) &&
    cardImage(variant).bytes <= MAX_FEATURED_IMAGE_BYTES &&
    (thumbUses.get(variant.thumb.sha256) ?? 0) <= MAX_SHARED_THUMB_USES
  );
}

/**
 * Return `count` featured models: curated slugs first, then mid-size models from
 * different categories (catalog order, so the pick is stable between builds). Slugs in
 * `exclude` (the hero) are never picked, so no diagram appears twice on the page.
 */
export function getFeaturedModels(models: readonly ModelRecord[], count = 6, exclude: readonly string[] = []): ModelRecord[] {
  const thumbUses = thumbUseCounts(models);
  const bySlug = new Map(models.map((model) => [model.slug, model]));
  const picked: ModelRecord[] = [];
  for (const slug of FEATURED_SLUGS) {
    const model = bySlug.get(slug);
    if (model && !exclude.includes(slug) && picked.length < count && isShowcaseable(model, thumbUses)) {
      picked.push(model);
    }
  }
  const usedCategories = new Set(picked.map((model) => model.category));
  const fallback = models.filter((model) => {
    const nOps = model.n_ops ?? 0;
    return !picked.includes(model) && !exclude.includes(model.slug) && nOps >= FALLBACK_MIN_OPS && nOps <= FALLBACK_MAX_OPS && isShowcaseable(model, thumbUses);
  });
  for (const model of fallback) {
    if (picked.length >= count) {
      break;
    }
    if (!usedCategories.has(model.category)) {
      picked.push(model);
      usedCategories.add(model.category);
    }
  }
  return picked.slice(0, count);
}

export interface HeroImage {
  model: ModelRecord;
  url: string;
  /** "svg" images are drawn at reading scale; "thumb" images fill the panel width. */
  kind: "svg" | "thumb";
}

/**
 * Pick the hero diagram: a small curated SVG when available, else the first showcaseable
 * curated model's thumbnail. Pick the hero before the panels and exclude it from them.
 */
export function getHeroImage(models: readonly ModelRecord[]): HeroImage | null {
  const bySlug = new Map(models.map((model) => [model.slug, model]));
  for (const slug of HERO_SLUGS) {
    const model = bySlug.get(slug);
    const variant = model ? defaultVariant(model) : undefined;
    if (model && variant && !model.quarantine && variant.svg.bytes <= HERO_MAX_SVG_BYTES) {
      return { model, url: variant.svg.url, kind: "svg" };
    }
  }
  const [first] = getFeaturedModels(models, 1);
  const variant = first ? defaultVariant(first) : undefined;
  return first && variant ? { model: first, url: variant.thumb.url, kind: "thumb" } : null;
}
