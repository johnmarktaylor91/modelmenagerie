/**
 * Families and relatives: the family pages (`/families/<slug>/`) and the model page's
 * "Related models" block. Build-time only.
 */
import { getAllModels, type ModelRecord } from "@/lib/catalog";
import { isPresent } from "@/lib/labels";
import { compactText } from "@/scripts/CardView";

/**
 * A family gets its own page only when it has at least this many members. Pages Free allows
 * 20,000 files and the site keeps 20 percent headroom; on the June 2026 preview data, 342
 * families qualify at 5 (951 at 3, 2,432 at 2). Smaller families are reached through the
 * listing's family filter instead.
 */
export const FAMILY_PAGE_MIN_MEMBERS = 5;

/** How many relatives of each kind the model page shows. */
export const RELATIVES_SHOWN = 6;

export interface Family {
  /** The catalog's family value, used by the listing's `family` filter. */
  name: string;
  slug: string;
  /** Members, smallest first (parameters, then operations, then name). */
  members: ModelRecord[];
  /** A paper every member cites, when they agree on one. */
  paper: { url: string; title: string | null } | null;
  summary: string | null;
}

let families: Map<string, Family> | null = null;
let familiesByName: Map<string, Family> | null = null;
let shapeGroups: Map<string, ModelRecord[]> | null = null;

/** Lower-case words joined by hyphens: "smp unet" becomes "smp-unet". */
export function familySlug(name: string): string {
  return name
    .normalize("NFKD")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** True when a family value names a real group rather than a placeholder. */
export function isRealFamily(name: string | null | undefined): name is string {
  return isPresent(name) && compactText(name) !== "";
}

/** Known numbers ascending, unknown last. */
function compareKnown(left: number | null, right: number | null): number {
  if (left === null || right === null) {
    return Number(left === null) - Number(right === null);
  }
  return left - right;
}

function compareBySize(left: ModelRecord, right: ModelRecord): number {
  return (
    compareKnown(left.param_count, right.param_count) ||
    compareKnown(left.n_ops, right.n_ops) ||
    left.display_name.localeCompare(right.display_name, "en", { sensitivity: "base" }) ||
    left.slug.localeCompare(right.slug)
  );
}

/** The paper all members link to, or null when some differ or none has one. */
function sharedPaper(members: readonly ModelRecord[]): Family["paper"] {
  const urls = new Set(members.map((member) => member.paper_url ?? ""));
  const [url] = [...urls];
  if (urls.size !== 1 || !isPresent(url)) {
    return null;
  }
  const titled = members.find((member) => isPresent(member.paper_title));
  return { url, title: titled?.paper_title ?? null };
}

/** Group a record list into families; exported so fixture pages build families the same way. */
export function groupFamilies(models: readonly ModelRecord[]): Map<string, Family> {
  const groups = new Map<string, ModelRecord[]>();
  for (const model of models) {
    if (isRealFamily(model.family_normalized)) {
      const group = groups.get(model.family_normalized) ?? [];
      group.push(model);
      groups.set(model.family_normalized, group);
    }
  }
  const bySlug = new Map<string, Family>();
  for (const [name, members] of groups) {
    let slug = familySlug(name) || "family";
    for (let suffix = 2; bySlug.has(slug); suffix += 1) {
      slug = `${familySlug(name) || "family"}-${suffix}`;
    }
    members.sort(compareBySize);
    const summary = members.map((member) => member.family_summary).find(isPresent) ?? null;
    bySlug.set(slug, { name, slug, members, paper: sharedPaper(members), summary });
  }
  return bySlug;
}

function loadFamilies(): Map<string, Family> {
  families ??= groupFamilies(getAllModels());
  return families;
}

/** Families that get a page, largest first. */
export function listFamilyPages(): Family[] {
  return [...loadFamilies().values()]
    .filter((family) => family.members.length >= FAMILY_PAGE_MIN_MEMBERS)
    .sort((left, right) => right.members.length - left.members.length || left.name.localeCompare(right.name));
}

/** The family a model belongs to, or null. */
export function getFamilyOf(model: ModelRecord): Family | null {
  familiesByName ??= new Map([...loadFamilies().values()].map((family) => [family.name, family]));
  return familiesByName.get(model.family_normalized) ?? null;
}

/** Where a family is shown: its own page when it has one, else the listing filtered to it. */
export function familyHref(family: Pick<Family, "name" | "slug" | "members">): string {
  return family.members.length >= FAMILY_PAGE_MIN_MEMBERS
    ? `/families/${family.slug}/`
    : `/models/?family=${encodeURIComponent(family.name)}`;
}

function loadShapeGroups(): Map<string, ModelRecord[]> {
  if (shapeGroups === null) {
    shapeGroups = new Map();
    for (const model of getAllModels()) {
      if (isPresent(model.graph_shape_hash)) {
        const group = shapeGroups.get(model.graph_shape_hash) ?? [];
        group.push(model);
        shapeGroups.set(model.graph_shape_hash, group);
      }
    }
  }
  return shapeGroups;
}

/** Members nearest in size to a model, the model itself left out. */
export function familyRelatives(model: ModelRecord, family: Family | null, limit = RELATIVES_SHOWN): ModelRecord[] {
  if (!family) {
    return [];
  }
  const index = family.members.findIndex((member) => member.slug === model.slug);
  const others = family.members.filter((member) => member.slug !== model.slug);
  if (index < 0 || others.length <= limit) {
    return others.slice(0, limit);
  }
  const start = Math.min(Math.max(0, index - Math.floor(limit / 2)), others.length - limit);
  return others.slice(start, start + limit);
}

/** Other models with the same graph shape, outside the model's family (those show as family). */
export function sameShapeModels(model: ModelRecord, limit = RELATIVES_SHOWN): ModelRecord[] {
  if (!isPresent(model.graph_shape_hash)) {
    return [];
  }
  return (loadShapeGroups().get(model.graph_shape_hash) ?? [])
    .filter((other) => other.slug !== model.slug && other.family_normalized !== model.family_normalized)
    .slice(0, limit);
}
