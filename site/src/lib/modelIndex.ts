/**
 * Build side of the compact index (`/models-index.json`) and picture keys (`/models-thumbs.json`).
 * Card fields come from the same function the static cards use, so filtered results match them.
 */
import { findDefaultVariant, getAllModels, getStandInShareCount, hasCardPreview, type ModelRecord } from "@/lib/catalog";
import { availableSorts, facetFieldsFromRecord, facetGroups, type FacetGroup, type SortKey } from "@/lib/facets";
import { isRealFamily } from "@/lib/families";
import { cardDataFromRecord } from "@/scripts/CardView";
import {
  FLAG_CROP,
  FLAG_PAPER,
  FLAG_PICTURE,
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

/** Return the content-addressed asset key from a public asset URL ("" stays ""). */
function assetKey(url: string): string {
  return url.split("/").filter(Boolean).pop() ?? "";
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
export function buildModelIndex(models: readonly ModelRecord[]): { index: ModelIndexFile; thumbs: ModelThumbsFile } {
  const categories = lookup();
  const families = lookup();
  const frameworks = lookup();
  const licences = lookup();
  const keys: string[] = [];
  const rows = models.map((model): ModelIndexRow => {
    const card = cardDataFromRecord(model, getStandInShareCount(findDefaultVariant(model)) > 0);
    const facets = facetFieldsFromRecord(model);
    const flags =
      (card.verified ? FLAG_VERIFIED : 0) |
      (facets.hasPaper ? FLAG_PAPER : 0) |
      (hasCardPreview(model) ? FLAG_PICTURE : 0) |
      (card.crop ? FLAG_CROP : 0);
    keys.push(card.imageUrl ? encodeThumbKey(assetKey(card.imageUrl)) : "");
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
  return {
    index: { version: MODEL_INDEX_VERSION, categories: categories.values, families: families.values, frameworks: frameworks.values, licences: licences.values, rows },
    thumbs: { version: MODEL_INDEX_VERSION, keys },
  };
}

/** The catalog's index and picture keys, built once per build. */
export function getModelIndexFiles(): { index: ModelIndexFile; thumbs: ModelThumbsFile } {
  built ??= buildModelIndex(getAllModels());
  return built;
}

/** The listing's facet groups and sort choices for the catalog, computed once per build. */
export function getListingControls(): { groups: FacetGroup[]; sorts: Array<{ key: SortKey; label: string }> } {
  if (listingControls === null) {
    const fields = getAllModels().map(facetFieldsFromRecord);
    listingControls = { groups: facetGroups(fields), sorts: availableSorts(fields) };
  }
  return listingControls;
}
