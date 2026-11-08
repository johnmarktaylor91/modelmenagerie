/**
 * Build side of the compact index (`/models-index.json`) and picture keys (`/models-thumbs.json`).
 * Card fields come from the same function the static cards use, so filtered results match them.
 */
import { assetKeyOf } from "@/lib/asset-href.mjs";
import { findDefaultVariant, getAllModels, getListingFeaturedSlugs, getShownModels, getStandInShareCount, hasCardPreview, type ModelRecord } from "@/lib/catalog";
import { recordTier } from "@/lib/shown";
import { availableSorts, facetFieldsFromRecord, facetGroups, type FacetGroup, type SortKey } from "@/lib/facets";
import { familyDisplayName, isRealFamily } from "@/lib/families";
import { cardDataFromRecord, familyChip } from "@/scripts/CardView";
import {
  FLAG_CROP,
  FLAG_PAPER,
  FLAG_PICTURE,
  FLAG_REPLAY,
  FLAG_STRUCTURE,
  FLAG_VERIFIED,
  MODEL_INDEX_VERSION,
  encodeThumbKey,
  slugFromName,
  trimRow,
  type ModelIndexFile,
  type ModelIndexRow,
  type ModelThumbsFile,
} from "@/scripts/ListingIndex";

let built: { index: ModelIndexFile; thumbs: ModelThumbsFile } | null = null;
let listingControls: { groups: FacetGroup[]; sorts: Array<{ key: SortKey; label: string }> } | null = null;


/**
 * A family's chip label as the index stores it: "" for no chip (no readable display name), "=" when the
 * display name is the family value itself, else the display name (`familyChip` decides the rest).
 */
function indexFamilyLabel(name: string): string {
  const label = familyDisplayName(name);
  if (!name || !label || familyChip(name, label, "") === "") {
    return "";
  }
  return label === name ? "=" : label;
}

/** A lookup list that hands out each value's position, adding it on first use. */
function lookup(): { values: string[]; position: (value: string) => number } {
  const values: string[] = [];
  const positions = new Map<string, number>();
  return {
    values,
    position(value: string): number {
      if (!value) {
        return -1;
      }
      let known = positions.get(value);
      if (known === undefined) {
        known = values.push(value) - 1;
        positions.set(value, known);
      }
      return known;
    },
  };
}

/** Build both files in one pass over the catalog (or a fixture list). */
export function buildModelIndex(models: readonly ModelRecord[], featuredSlugs: readonly string[] = []): { index: ModelIndexFile; thumbs: ModelThumbsFile } {
  const categories = lookup();
  const families = lookup();
  const frameworks = lookup();
  const licences = lookup();
  const keys: string[] = [];
  const windows: NonNullable<ModelIndexFile["windows"]> = [];
  const rows = models.map((model, order): ModelIndexRow => {
    const card = cardDataFromRecord(model, getStandInShareCount(findDefaultVariant(model), model) > 0, familyDisplayName(model.family_normalized));
    const facets = facetFieldsFromRecord(model);
    const flags =
      (card.verified ? FLAG_VERIFIED : 0) |
      (facets.hasPaper ? FLAG_PAPER : 0) |
      (hasCardPreview(model) ? FLAG_PICTURE : 0) |
      (card.crop ? FLAG_CROP : 0) |
      (recordTier(model) === "structure" ? FLAG_STRUCTURE : 0) |
      (card.status === "replay" ? FLAG_REPLAY : 0);
    const crop = card.crop ? { width: card.cropWidth, height: card.cropHeight } : null;
    keys.push(card.imageUrl ? encodeThumbKey(assetKeyOf(card.imageUrl), crop) : "");
    if (card.window) {
      windows.push([order, card.window.width, card.window.left, card.window.top]);
    }
    return trimRow([
      slugFromName(card.name) === card.slug ? "" : card.slug,
      card.name,
      categories.position(card.category),
      families.position(isRealFamily(model.family_normalized) ? model.family_normalized : ""),
      card.nOps,
      card.paramCount,
      flags,
      facets.year,
      frameworks.position(facets.backend),
      licences.position(facets.licence),
      facets.citations,
    ]);
  });
  const position = new Map(models.map((model, row) => [model.slug, row]));
  const featured = featuredSlugs.map((slug) => position.get(slug)).filter((row): row is number => row !== undefined);
  return {
    index: { version: MODEL_INDEX_VERSION, categories: categories.values, families: families.values, familyLabels: families.values.map(indexFamilyLabel), frameworks: frameworks.values, licences: licences.values, rows, featured, ...(windows.length ? { windows } : {}) },
    thumbs: { version: MODEL_INDEX_VERSION, keys },
  };
}

/** The catalog's index and picture keys, built once per build. */
export function getModelIndexFiles(): { index: ModelIndexFile; thumbs: ModelThumbsFile } {
  built ??= buildModelIndex(getAllModels(), getListingFeaturedSlugs());
  return built;
}

/** The listing's facet groups and sort choices, counted over the default (validated) listing, once per build. */
export function getListingControls(): { groups: FacetGroup[]; sorts: Array<{ key: SortKey; label: string }> } {
  if (listingControls === null) {
    const fields = getShownModels().map(facetFieldsFromRecord);
    listingControls = { groups: facetGroups(fields), sorts: availableSorts(fields) };
  }
  return listingControls;
}
