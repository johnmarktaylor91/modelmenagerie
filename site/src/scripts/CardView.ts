/**
 * The one model card. Build-time pages and the client listing both render cards through
 * `renderCardHtml`, so filtered results look exactly like the static pages.
 *
 * Browser-safe: imports only types and labels.
 */
import type { ModelRecord } from "@/lib/catalog";
import { formatCompact, formatNumber, taskLabel } from "@/lib/labels";
import { featuredWindow, windowStyle, type PictureWindow } from "@/lib/featured-view";
import { cardImage, hasUsablePreview } from "@/lib/preview";
import { recordTier } from "@/lib/shown";

/** Everything a card shows; empty strings and nulls are hidden, never printed. */
export interface CardData {
  slug: string;
  name: string;
  category: string;
  /** Family chip text (`familyChip`), or "" when the family has no readable name or only repeats the model name. */
  family: string;
  nOps: number | null;
  paramCount: number | null;
  verified: boolean;
  /**
   * The exception the card marks with a chip: "structure" (graph structure only, not checked),
   * "replay" (checked by replay on a non-PyTorch framework), or "" for an ordinary checked model,
   * which needs no chip because every card in the default listing is one.
   */
  status: CardStatus;
  /** Public preview picture URL (thumbnail or small SVG, see `cardImage`), or "" when the record has no usable preview. */
  imageUrl: string;
  /** True when the picture is a real top crop (`thumb_crop`, 800 wide, up to 600 tall) rather than the whole-graph strip. */
  crop?: boolean;
  /** The crop's own size, so the image reserves its box before it loads (defaults 800 by 600). */
  cropWidth?: number;
  cropHeight?: number;
  /** A featured model's hand-chosen window on its SVG (`imageUrl` is then the SVG); see `featured-view.ts`. */
  window?: PictureWindow;
}

export type CardStatus = "" | "structure" | "replay";

/** The chip text for each exception. */
export const STATUS_CHIPS: Readonly<Record<Exclude<CardStatus, "">, string>> = {
  structure: "Structure only",
  replay: "Replay checked",
};

/** The card status of a record, by the same flags the site's shown-set rule reads. */
export function cardStatus(model: Pick<ModelRecord, "forward_validated" | "validation_state" | "capture" | "fidelity" | "own_diagram" | "backend">): CardStatus {
  if (recordTier(model) === "structure") {
    return "structure";
  }
  return model.backend && model.backend !== "torch" ? "replay" : "";
}

/** A crop's size when the export does not say: the full 4:3 box. */
export const CROP_WIDTH = 800;
export const CROP_HEIGHT = 600;

/** How many cards at the top of a listing load their picture first (one row on a wide screen). */
export const PRIORITY_CARDS = 4;

export interface CardOptions {
  /** Heading level for the model name: 2 on the listing, 3 under a section heading. */
  headingLevel?: 2 | 3;
  /** True for the first row of a listing: the picture loads eagerly with high fetch priority. */
  priority?: boolean;
}

/** Lowercase and keep only letters and digits, so "ResNet-50" and "resnet50" compare equal. */
export function compactText(value: string): string {
  return value.normalize("NFKD").toLowerCase().replace(/[^a-z0-9]+/g, "");
}

/** Lower-case words joined by hyphens: "smp unet" becomes "smp-unet" (family page slugs). */
export function slugText(value: string): string {
  return value
    .normalize("NFKD")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/**
 * The family chip's text: the display name `families.jsonl` gives the family, only when that name is
 * readable (it differs from the family's slug, so "cnn-classic" gets no chip) and adds to the model name.
 * The model page's family link does not depend on this.
 */
export function familyChip(family: string, displayName: string | null | undefined, modelName: string): string {
  const label = (displayName ?? "").trim();
  if (!family || !label || label === slugText(family)) {
    return "";
  }
  return distinctFamily(label, modelName);
}

/** Return the family only when it adds something the model name does not already say. */
export function distinctFamily(family: string, name: string): string {
  const familyKey = compactText(family);
  if (!familyKey || familyKey === "unknown") {
    return "";
  }
  return compactText(name).includes(familyKey) ? "" : family;
}

/** Escape text for HTML element content and quoted attribute values. */
export function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => {
    switch (character) {
      case "&":
        return "&amp;";
      case "<":
        return "&lt;";
      case ">":
        return "&gt;";
      case '"':
        return "&quot;";
      default:
        return "&#39;";
    }
  });
}

/**
 * Build card data from a full catalog record at build time. `standIn` marks a default diagram that
 * many unrelated models share (see `getStandInShareCount`); like a near-blank thumbnail, it is not
 * shown as this model's preview. `familyDisplayName` is the family's `families.jsonl` display name.
 */
export function cardDataFromRecord(model: ModelRecord, standIn = false, familyDisplayName: string | null = null): CardData {
  const variant = model.variants.find((candidate) => candidate.key === model.default_variant) ?? model.variants[0];
  const ownPicture = variant !== undefined && !standIn && model.own_diagram !== false;
  const view = ownPicture ? featuredWindow(model.slug, variant) : null;
  const crop = ownPicture && !view && variant.thumb_crop ? variant.thumb_crop : null;
  const picture = view ? variant.svg.url : crop ? crop.url : ownPicture && hasUsablePreview(variant.thumb.bytes) ? cardImage(variant).url : "";
  return {
    slug: model.slug,
    name: model.display_name,
    category: model.category,
    family: familyChip(model.family_normalized, familyDisplayName, model.display_name),
    nOps: model.n_ops,
    paramCount: model.param_count,
    verified: model.forward_validated,
    status: cardStatus(model),
    imageUrl: picture,
    crop: crop !== null,
    ...(crop ? { cropWidth: crop.width || CROP_WIDTH, cropHeight: crop.height || CROP_HEIGHT } : {}),
    ...(view ? { window: view } : {}),
  };
}

/**
 * The facts under the name: task, operations, parameters. Unknown numbers are empty strings, so
 * every card has the same three slots (the list view lines them up as columns).
 */
export function cardFacts(data: CardData): { task: string; ops: string; params: string } {
  return {
    task: taskLabel(data.category),
    ops: data.nOps === null ? "" : `${formatNumber(data.nOps)} ${data.nOps === 1 ? "op" : "ops"}`,
    params: data.paramCount === null ? "" : `${formatCompact(data.paramCount)} params`,
  };
}

function factsHtml(data: CardData): string {
  const facts = cardFacts(data);
  // Separators come from CSS, between non-empty slots only.
  return `<p class="model-card__facts"><span class="model-card__fact model-card__fact--task">${escapeHtml(facts.task)}</span><span class="model-card__fact model-card__fact--ops">${escapeHtml(facts.ops)}</span><span class="model-card__fact model-card__fact--params">${escapeHtml(facts.params)}</span></p>`;
}

/**
 * A picture card shows the top of the graph in a fixed 4:3 box (no layout shift while it loads).
 * A real crop is drawn at the box's width from the top, with its own width and height set; when
 * the export has no crop the whole-graph strip is drawn the same way. The first row of a listing
 * loads eagerly with high fetch priority; the rest load lazily. A record without a usable preview
 * gets a smaller text card; its empty frame shows only in the list view. A featured model with a
 * chosen window shows that part of its SVG instead (`featured-view.ts`).
 */
function previewHtml(data: CardData, priority: boolean): string {
  if (!data.imageUrl) {
    return '<div class="model-card__frame model-card__frame--empty" aria-hidden="true"></div>';
  }
  const crop = data.crop ? " model-card__frame--crop" : "";
  const size = data.crop ? ` width="${data.cropWidth ?? CROP_WIDTH}" height="${data.cropHeight ?? CROP_HEIGHT}"` : "";
  const loading = priority ? ' fetchpriority="high"' : ' loading="lazy"';
  if (data.window) {
    // Never lazy: before it loads the picture has no height, so a deep window's negative top margin (VGG-16's is
    // -941%) parks it outside the lazy-load range and it never starts. Windowed cards are a few small SVGs.
    const windowLoading = priority ? ' fetchpriority="high"' : "";
    return `<div class="model-card__frame model-card__frame--window"><img src="${escapeHtml(data.imageUrl)}" alt="" style="${windowStyle(data.window)}"${windowLoading} decoding="async" /></div>`;
  }
  return `<div class="model-card__frame${crop}"><img src="${escapeHtml(data.imageUrl)}" alt=""${size}${loading} decoding="async" /></div>`;
}

function chipsHtml(data: CardData): string {
  const chips: string[] = [];
  if (data.family) {
    chips.push(`<span class="model-card__chip">${escapeHtml(data.family)}</span>`);
  }
  if (data.status) {
    chips.push(`<span class="model-card__chip model-card__chip--${data.status}">${STATUS_CHIPS[data.status]}</span>`);
  }
  return chips.length ? `<div class="model-card__chips">${chips.join("")}</div>` : "";
}

/** Render one card as an HTML string; every value is escaped. */
export function renderCardHtml(data: CardData, options: CardOptions = {}): string {
  const heading = `h${options.headingLevel ?? 2}`;
  const kind = data.imageUrl ? "model-card" : "model-card model-card--text";
  const noPreview = data.imageUrl ? "" : '<p class="model-card__no-preview">Preview not available</p>';
  return `<article class="${kind}" data-model-card>
  <a class="model-card__link" href="/models/${encodeURIComponent(data.slug)}/">
    ${previewHtml(data, options.priority ?? false)}
    <div class="model-card__body">
      <${heading} class="model-card__title">${escapeHtml(data.name)}</${heading}>
      ${factsHtml(data)}
      ${chipsHtml(data)}
      ${noPreview}
    </div>
  </a>
</article>`;
}
