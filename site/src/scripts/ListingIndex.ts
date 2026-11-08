/**
 * Compact full-catalog index for search, filters and sorts (`/models-index.json`), and the
 * picture keys the listing's cards need (`/models-thumbs.json`), kept apart so the header
 * search loads only the small index.
 *
 * Shared by the build-time endpoints (encode side) and the browser (decode side).
 */
import { assetHref } from "@/lib/assets";
import type { FacetFields } from "@/lib/facets";
import { CROP_HEIGHT, CROP_WIDTH, familyChip, type CardData } from "@/scripts/CardView";

export const MODEL_INDEX_VERSION = 6;

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
  /** Chip labels aligned with `families`: "" for no chip, "=" for the family value itself, else the display name. */
  familyLabels: string[];
  frameworks: string[];
  licences: string[];
  rows: ModelIndexRow[];
  /** Row positions of the featured models, in the order the browse page opens on. */
  featured?: number[];
  /** Featured cards' picture windows on their SVG (`featured-view.ts`): row, width, left, top. */
  windows?: Array<[number, number, number, number]>;
}

/**
 * Picture keys in index row order: "" for none, ".webp" left off. A crop that is not the full
 * 800 by 600 box carries its size after a tilde ("abc123~800x412"), so its card reserves the
 * right height.
 */
export interface ModelThumbsFile {
  version: number;
  keys: string[];
}

export const FLAG_VERIFIED = 1;
export const FLAG_PAPER = 2;
export const FLAG_PICTURE = 4;
/** The picture is a real top crop, not the whole-graph strip. */
export const FLAG_CROP = 8;
/** A structure-only record: it has a page but is listed only when the reader asks for untraced entries. */
export const FLAG_STRUCTURE = 16;
/** Checked by replay on a non-PyTorch framework (the card's "Replay checked" chip). */
export const FLAG_REPLAY = 32;

/** One decoded index entry: card data plus the fields the filters and sorts read. */
export interface IndexEntry {
  card: CardData;
  facets: FacetFields;
  /** The catalog's family value, "" when the record has none (the card shows it only when it adds to the name). */
  familyName: string;
  /** Catalog order, which is also the static pages' name order. */
  order: number;
  hasPicture: boolean;
  /** True for a structure-only record (see `FLAG_STRUCTURE`). */
  structure: boolean;
  /** Position in the featured list, or -1 when the model is not featured. */
  featured: number;
}

/** The slug a display name gives when it is lower-cased and joined by hyphens. */
export function slugFromName(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Return the thumbnail key stored for an asset key, without the common extension, plus a crop's size when it is not 800 by 600. */
export function encodeThumbKey(key: string, crop: { width?: number; height?: number } | null = null): string {
  const base = key.endsWith(".webp") ? key.slice(0, -".webp".length) : key;
  const width = crop?.width ?? CROP_WIDTH;
  const height = crop?.height ?? CROP_HEIGHT;
  return crop && base && (width !== CROP_WIDTH || height !== CROP_HEIGHT) ? `${base}~${width}x${height}` : base;
}

/** Split a stored key into its asset key and an optional crop size. */
export function decodeThumbKey(stored: string): { key: string; width?: number; height?: number } {
  const [key, size = ""] = stored.split("~");
  const match = /^(\d+)x(\d+)$/.exec(size);
  return match ? { key, width: Number(match[1]), height: Number(match[2]) } : { key };
}

function imageUrl(key: string): string {
  if (!key) {
    return "";
  }
  return assetHref(key.includes(".") ? key : `${key}.webp`);
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

/** The display name the index stores for a family position ("=" stands for the family value itself). */
function familyLabel(file: ModelIndexFile, familyIndex: number): string {
  const label = file.familyLabels[familyIndex] ?? "";
  return label === "=" ? (file.families[familyIndex] ?? "") : label;
}

/** Validate the fetched index; throws on anything unexpected so the page can offer a retry. */
export function parseModelIndex(value: unknown): ModelIndexFile {
  const file = value as Partial<ModelIndexFile> | null;
  if (
    !file ||
    file.version !== MODEL_INDEX_VERSION ||
    !Array.isArray(file.categories) ||
    !Array.isArray(file.families) ||
    !Array.isArray(file.familyLabels) ||
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
export function decodeModelIndex(file: ModelIndexFile, thumbs: ModelThumbsFile | null = null): IndexEntry[] {
  const featured = new Map((Array.isArray(file.featured) ? file.featured : []).map((row, rank) => [row, rank]));
  const windows = new Map((Array.isArray(file.windows) ? file.windows : []).map(([row, width, left, top]) => [row, { width, left, top }]));
  return file.rows.map((row, order) => {
    const [slug, name, categoryIndex, familyIndex = -1, nOps = null, paramCount = null, flags = 0, year = null, framework = -1, licence = -1, citations = null] = row;
    const category = file.categories[categoryIndex] ?? "";
    const familyName = file.families[familyIndex] ?? "";
    const crop = (flags & FLAG_CROP) !== 0;
    const thumb = decodeThumbKey(thumbs?.keys[order] ?? "");
    const pictureWindow = thumbs && thumb.key ? windows.get(order) : undefined;
    return {
      card: {
        slug: slug || slugFromName(name),
        name,
        category,
        family: familyChip(familyName, familyLabel(file, familyIndex), name),
        nOps,
        paramCount,
        verified: (flags & FLAG_VERIFIED) !== 0,
        status: (flags & FLAG_STRUCTURE) !== 0 ? "structure" : (flags & FLAG_REPLAY) !== 0 ? "replay" : "",
        imageUrl: thumbs ? imageUrl(thumb.key) : "",
        crop,
        ...(crop ? { cropWidth: thumb.width ?? CROP_WIDTH, cropHeight: thumb.height ?? CROP_HEIGHT } : {}),
        ...(pictureWindow ? { window: pictureWindow } : {}),
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
      structure: (flags & FLAG_STRUCTURE) !== 0,
      featured: featured.get(order) ?? -1,
    };
  });
}
