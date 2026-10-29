/**
 * Share images, written into the build as /og/<name>.png: `default` (home and most pages), `dataset`,
 * and one per task category, which model pages use until each model has its own (plan section 4).
 * One image per task keeps the build far under the hosting file cap; one per model would double it.
 */
import type { APIRoute, GetStaticPaths } from "astro";

import { getAllModels, getCatalogIndex, getModelBySlug, listModelSlugs } from "@/lib/catalog";
import { SHARE_IMAGE_SLUGS } from "@/lib/featured";
import { categoryLabel, formatNumber } from "@/lib/labels";
import { shareImageName } from "@/lib/seo";
import { renderShareImage, type ShareCard } from "@/lib/share-image";

const known = new Set(listModelSlugs());

/** "Pictured: AlexNet" for the model whose diagram a card shows, when it is in the catalog. */
function caption(key: string): string | null {
  const slug = SHARE_IMAGE_SLUGS[key] ?? SHARE_IMAGE_SLUGS.default;
  return slug && known.has(slug) ? `Pictured: ${getModelBySlug(slug).display_name}` : null;
}

function cards(): Map<string, ShareCard> {
  const total = formatNumber(getAllModels().length);
  const result = new Map<string, ShareCard>([
    ["default", { headline: "Computational graphs of neural networks", subline: `${total} architectures, traced and drawn with TorchLens.`, crop: "default", caption: caption("default") }],
    ["dataset", { headline: "The dataset", subline: `One record per model for ${total} models, with every diagram file. CSV and JSON Lines.`, crop: "default", caption: caption("default") }],
  ]);
  for (const facet of getCatalogIndex().facets.categories) {
    const name = shareImageName(facet.value);
    result.set(name, {
      headline: categoryLabel(facet.value),
      subline: `${formatNumber(facet.count)} ${facet.count === 1 ? "architecture" : "architectures"}, traced and drawn with TorchLens.`,
      crop: name,
      caption: caption(facet.value),
    });
  }
  return result;
}

export const getStaticPaths = (() => [...cards().keys()].map((name) => ({ params: { name } }))) satisfies GetStaticPaths;

export const GET: APIRoute = async ({ params }) => {
  const card = cards().get(String(params.name));
  if (!card) {
    return new Response("Not found", { status: 404 });
  }
  const png = await renderShareImage(card);
  return new Response(new Uint8Array(png), { headers: { "Content-Type": "image/png" } });
};
