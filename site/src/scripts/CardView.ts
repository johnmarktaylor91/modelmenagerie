/**
 * The one model card. Build-time pages and the client listing both render cards through
 * `renderCardHtml`, so filtered results look exactly like the static pages.
 *
 * Browser-safe: imports only types and labels.
 */
import type { ModelRecord } from "@/lib/catalog";
import { categoryLabel, formatCompact, formatNumber } from "@/lib/labels";
import { cardImage, hasUsablePreview } from "@/lib/preview";

/** Everything a card shows; empty strings and nulls are hidden, never printed. */
export interface CardData {
  slug: string;
  name: string;
  category: string;
  /** Family, or "" when it only repeats the model name. */
  family: string;
  nOps: number | null;
  paramCount: number | null;
  verified: boolean;
  /** Public preview picture URL (thumbnail or small SVG, see `cardImage`), or "" when the record has no usable preview. */
  imageUrl: string;
  /** True when the picture is a real top crop (4:3, `thumb_crop`) rather than the whole-graph strip. */
  crop?: boolean;
}

export interface CardOptions {
  /** Heading level for the model name: 2 on the listing, 3 under a section heading. */
  headingLevel?: 2 | 3;
}

/** Lowercase and keep only letters and digits, so "ResNet-50" and "resnet50" compare equal. */
export function compactText(value: string): string {
  return value.normalize("NFKD").toLowerCase().replace(/[^a-z0-9]+/g, "");
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
 * shown as this model's preview.
 */
export function cardDataFromRecord(model: ModelRecord, standIn = false): CardData {
  const variant = model.variants.find((candidate) => candidate.key === model.default_variant) ?? model.variants[0];
  const ownPicture = variant !== undefined && !standIn && model.own_diagram !== false;
  const crop = ownPicture && variant.thumb_crop ? variant.thumb_crop : null;
  return {
    slug: model.slug,
    name: model.display_name,
    category: model.category,
    family: distinctFamily(model.family_normalized, model.display_name),
    nOps: model.n_ops,
    paramCount: model.param_count,
    verified: model.forward_validated,
    imageUrl: crop ? crop.url : ownPicture && hasUsablePreview(variant.thumb.bytes) ? cardImage(variant).url : "",
    crop: crop !== null,
  };
}

/**
 * The facts under the name: task, operations, parameters. Unknown numbers are empty strings, so
 * every card has the same three slots (the list view lines them up as columns).
 */
export function cardFacts(data: CardData): { task: string; ops: string; params: string } {
  return {
    task: categoryLabel(data.category),
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
 * A real crop fills the box (cover, anchored at the top); until crops exist the whole-graph strip
 * is drawn at the box's width from the top. A record without a usable preview gets a smaller
 * text card; its empty frame shows only in the list view.
 */
function previewHtml(data: CardData): string {
  if (!data.imageUrl) {
    return '<div class="model-card__frame model-card__frame--empty" aria-hidden="true"></div>';
  }
  const crop = data.crop ? " model-card__frame--crop" : "";
  const size = data.crop ? ' width="800" height="600"' : "";
  return `<div class="model-card__frame${crop}"><img src="${escapeHtml(data.imageUrl)}" alt=""${size} loading="lazy" decoding="async" /></div>`;
}

function chipsHtml(data: CardData): string {
  const chips: string[] = [];
  if (data.family) {
    chips.push(`<span class="model-card__chip">${escapeHtml(data.family)}</span>`);
  }
  if (data.verified) {
    chips.push('<span class="model-card__chip model-card__chip--verified">Verified</span>');
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
    ${previewHtml(data)}
    <div class="model-card__body">
      <${heading} class="model-card__title">${escapeHtml(data.name)}</${heading}>
      ${factsHtml(data)}
      ${chipsHtml(data)}
      ${noPreview}
    </div>
  </a>
</article>`;
}
