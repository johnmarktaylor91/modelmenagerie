/**
 * Search and sharing metadata: share images, structured data (JSON-LD) and the per-model metadata
 * a model page passes to the layout. The tags themselves are written by `components/SeoHead.astro`.
 */
import type { AssetRef, ModelRecord } from "@/lib/catalog";
import { categoryLabel } from "@/lib/labels";
import { CONTENT_LICENCE, CREATOR, SITE_NAME, SITE_URL, TORCHLENS } from "@/lib/site";

export const SHARE_IMAGE_WIDTH = 1200;
export const SHARE_IMAGE_HEIGHT = 630;

const publicR2Base = (import.meta.env.PUBLIC_R2_BASE as string | undefined)?.replace(/\/+$/, "") ?? "";

export interface ShareImage {
  url: string;
  alt: string;
  width?: number;
  height?: number;
}

/** File name of a generated share image: `default`, `dataset`, or a task category. */
export function shareImageName(key: string): string {
  return key.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

/** Site-relative path of a generated share image (`src/pages/og/[name].png.ts` writes them). */
export function shareImagePath(key: string): string {
  return `/og/${shareImageName(key)}.png`;
}

export const DEFAULT_SHARE_IMAGE: ShareImage = {
  url: shareImagePath("default"),
  alt: `${SITE_NAME}: computational graphs of neural networks, drawn by TorchLens`,
  width: SHARE_IMAGE_WIDTH,
  height: SHARE_IMAGE_HEIGHT,
};

/** The record's own share image once the export carries one (planned `share_image`), else none. */
function ownShareImage(model: ModelRecord): AssetRef | null {
  const ref = (model as ModelRecord & { share_image?: AssetRef | null }).share_image;
  if (!ref?.url) {
    return null;
  }
  const key = ref.url.split("/").filter(Boolean).pop() ?? "";
  return { ...ref, url: publicR2Base ? `${publicR2Base}/${key}` : `/assets/${key}` };
}

/** Share image for a model page: its own when the export has one, else its task's image. */
export function modelShareImage(model: ModelRecord): ShareImage {
  const own = ownShareImage(model);
  if (own) {
    return { url: own.url, alt: `Computational graph of ${model.display_name}`, width: SHARE_IMAGE_WIDTH, height: SHARE_IMAGE_HEIGHT };
  }
  return {
    url: shareImagePath(model.category),
    alt: `${categoryLabel(model.category)} models in the ${SITE_NAME}`,
    width: SHARE_IMAGE_WIDTH,
    height: SHARE_IMAGE_HEIGHT,
  };
}

const absolute = (pathname: string): string => new URL(pathname, SITE_URL).href;

/** The site as a schema.org WebSite with its search, for the home page. */
export function websiteJsonLd(): Record<string, unknown> {
  return {
    "@context": "https://schema.org",
    "@type": "WebSite",
    name: SITE_NAME,
    url: absolute("/"),
    license: CONTENT_LICENCE.url,
    creator: { "@type": "Person", name: CREATOR.name, url: CREATOR.url },
    potentialAction: {
      "@type": "SearchAction",
      target: { "@type": "EntryPoint", urlTemplate: `${absolute("/models/")}?q={search_term_string}` },
      "query-input": "required name=search_term_string",
    },
  };
}

/** Value of a field the planned export adds, read without depending on the record type having it yet. */
function optionalField<T>(model: ModelRecord, key: string): T | null {
  const value = (model as unknown as Record<string, unknown>)[key];
  return value === undefined || value === null || value === "" || value === "unknown" ? null : (value as T);
}

/**
 * A model page as a schema.org CreativeWork (the diagram, which is this site's work) about a
 * SoftwareSourceCode (the model's code, under its own licence), plus the breadcrumb trail.
 */
export function modelJsonLd(model: ModelRecord): Record<string, unknown>[] {
  const pageUrl = absolute(`/models/${model.slug}/`);
  const sourceUrl = optionalField<string>(model, "source_url");
  const sourceLicence = optionalField<string>(model, "source_license");
  const paperUrl = optionalField<string>(model, "paper_url");
  const code: Record<string, unknown> = {
    "@type": "SoftwareSourceCode",
    name: model.display_name,
    ...(sourceUrl ? { codeRepository: sourceUrl } : {}),
    ...(sourceLicence ? { license: sourceLicence } : {}),
    ...(paperUrl ? { citation: paperUrl } : {}),
    programmingLanguage: "Python",
  };
  return [
    {
      "@context": "https://schema.org",
      "@type": "CreativeWork",
      name: `${model.display_name} computational graph`,
      url: pageUrl,
      image: absolute(modelShareImage(model).url),
      genre: categoryLabel(model.category),
      about: code,
      license: CONTENT_LICENCE.url,
      isPartOf: { "@type": "WebSite", name: SITE_NAME, url: absolute("/") },
      isBasedOn: { "@type": "SoftwareApplication", name: "TorchLens", url: TORCHLENS.repo },
    },
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "Home", item: absolute("/") },
        { "@type": "ListItem", position: 2, name: "Models", item: absolute("/models/") },
        { "@type": "ListItem", position: 3, name: model.display_name, item: pageUrl },
      ],
    },
  ];
}

/** Everything a model page passes to the layout for search and sharing: `<BaseLayout {...modelSeo(model)}>`. */
export function modelSeo(model: ModelRecord): { ogImage: string; ogImageAlt: string; jsonLd: Record<string, unknown>[] } {
  const image = modelShareImage(model);
  return { ogImage: image.url, ogImageAlt: image.alt, jsonLd: modelJsonLd(model) };
}
