/**
 * Compact full-catalog index for search, filters and sorts (`/models-index.json`), and the
 * picture keys the listing's cards need (`/models-thumbs.json`), kept apart so the header
 * search loads only the small index.
 *
 * Shared by the build-time endpoints (encode side) and the browser (decode side).
 */
import type { FacetFields } from "@/lib/facets";
import { distinctFamily, type CardData } from "@/scripts/CardView";

export const MODEL_INDEX_VERSION = 3;

/**
 * One row: slug ("" when it is `slugFromName(name)`), display name, category index, family
 * index, operation count, parameter count, flags, year, framework index, licence index,
 * citation count. Trailing null, 0 and -1 values are dropped.
 */
export type ModelIndexRow = [
  slug: string,
  name: string,
  category: number,
  family?: number,
  nOps?: number | null,
  params?: number | null,
  flags?: number,
  year?: number | null,
  framework?: number,
  licence?: number,
  citations?: number | null,
];

export interface ModelIndexFile {
  version: number;
  categories: string[];
  families: string[];
  frameworks: string[];
  licences: string[];
  rows: ModelIndexRow[];
}

/** Picture keys in index row order: "" for none, ".webp" left off. */
export interface ModelThumbsFile {
  version: number;
  keys: string[];
}

export const FLAG_VERIFIED = 1;
export const FLAG_PAPER = 2;
export const FLAG_PICTURE = 4;
/** The picture is a real 4:3 top crop, not the whole-graph strip. */
export const FLAG_CROP = 8;

/** One decoded index entry: card data plus the fields the filters and sorts read. */
export interface IndexEntry {
  card: CardData;
  facets: FacetFields;
  /** The catalog's family value, "" when the record has none (the card shows it only when it adds to the name). */
  familyName: string;
  /** Catalog order, which is also the static pages' name order. */
  order: number;
  hasPicture: boolean;
}

/** The slug a display name gives when it is lower-cased and joined by hyphens. */
export function slugFromName(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Return the thumbnail key stored for an asset key, without the common extension. */
export function encodeThumbKey(key: string): string {
  return key.endsWith(".webp") ? key.slice(0, -".webp".length) : key;
}

function imageUrl(assetBase: string, storedKey: string): string {
  if (!storedKey) {
    return "";
  }
  const key = storedKey.includes(".") ? storedKey : `${storedKey}.webp`;
  return assetBase ? `${assetBase}/${key}` : `/assets/${key}`;
}

/** Drop trailing values that decode to the defaults, so most rows stay short. */
export function trimRow(row: ModelIndexRow): ModelIndexRow {
  const values = [...row] as Array<string | number | null | undefined>;
  while (values.length > 3) {
    const last = values[values.length - 1];
    if (last === null || last === undefined || last === 0 || last === -1 || last === "") {
      values.pop();
    } else {
      break;
    }
  }
  return values as ModelIndexRow;
}

/** Validate the fetched index; throws on anything unexpected so the page can offer a retry. */
export function parseModelIndex(value: unknown): ModelIndexFile {
  const file = value as Partial<ModelIndexFile> | null;
  if (
    !file ||
    file.version !== MODEL_INDEX_VERSION ||
    !Array.isArray(file.categories) ||
    !Array.isArray(file.families) ||
    !Array.isArray(file.frameworks) ||
    !Array.isArray(file.licences) ||
    !Array.isArray(file.rows)
  ) {
    throw new Error("The model list has an unexpected format");
  }
  return file as ModelIndexFile;
}

/** Validate the fetched picture keys against the index they belong to. */
export function parseModelThumbs(value: unknown, rows: number): ModelThumbsFile {
  const file = value as Partial<ModelThumbsFile> | null;
  if (!file || file.version !== MODEL_INDEX_VERSION || !Array.isArray(file.keys) || file.keys.length !== rows) {
    throw new Error("The model pictures list has an unexpected format");
  }
  return file as ModelThumbsFile;
}

/** Decode index rows into entries; picture URLs come from the thumbs file when given. */
export function decodeModelIndex(file: ModelIndexFile, assetBase: string, thumbs: ModelThumbsFile | null = null): IndexEntry[] {
  return file.rows.map((row, order) => {
    const [slug, name, categoryIndex, familyIndex = -1, nOps = null, paramCount = null, flags = 0, year = null, framework = -1, licence = -1, citations = null] = row;
    const category = file.categories[categoryIndex] ?? "";
    const familyName = file.families[familyIndex] ?? "";
    return {
      card: {
        slug: slug || slugFromName(name),
        name,
        category,
        family: distinctFamily(familyName, name),
        nOps,
        paramCount,
        verified: (flags & FLAG_VERIFIED) !== 0,
        imageUrl: thumbs ? imageUrl(assetBase, thumbs.keys[order] ?? "") : "",
        crop: (flags & FLAG_CROP) !== 0,
      },
      facets: {
        category,
        nOps,
        paramCount,
        year,
        backend: file.frameworks[framework] ?? "",
        licence: file.licences[licence] ?? "",
        hasPaper: (flags & FLAG_PAPER) !== 0,
        citations,
      },
      familyName,
      order,
      hasPicture: (flags & FLAG_PICTURE) !== 0,
    };
  });
}
