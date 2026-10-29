/**
 * Draws a 1200 by 630 share image (the picture a link shows in chat apps and social sites) at build
 * time: the site's name and a headline on paper, beside the top of a real diagram on a white inset.
 * Text is laid out by satori (which turns it into outlines, so no system fonts are needed) and
 * rasterised by resvg. The diagram crops are committed PNGs made by `scripts/share_crops.mjs`.
 */
import fs from "node:fs";
import path from "node:path";
import { Resvg } from "@resvg/resvg-js";
import satori from "satori";

import { SHARE_IMAGE_HEIGHT, SHARE_IMAGE_WIDTH } from "@/lib/seo";

/** Inner size of the diagram inset; `scripts/share_crops.mjs` writes crops at exactly this size. */
export const CROP_WIDTH = 460;
export const CROP_HEIGHT = 582;

const CROPS_DIR = path.join(process.cwd(), "src", "assets", "share");
const FONTS_DIR = path.join(process.cwd(), "node_modules", "@fontsource");

/** The light paper palette from `styles/global.css`. */
const PAPER = { bg: "#faf8f3", surface: "#ffffff", border: "#ddd7ca", text: "#1c1b19", secondary: "#5c574e" };

export interface ShareCard {
  /** Large serif line, for example a task name. */
  headline: string;
  /** One sentence under the headline. */
  subline: string;
  /** Crop file name in `src/assets/share/` without `.png`; the default crop is used when missing. */
  crop: string;
  /** Names the pictured model, for example "Pictured: AlexNet". */
  caption: string | null;
}

let fonts: { name: string; data: Buffer; weight: 400 | 600; style: "normal" }[] | null = null;

function loadFonts() {
  if (fonts === null) {
    const read = (file: string) => fs.readFileSync(path.join(FONTS_DIR, file));
    fonts = [
      { name: "Source Serif 4", data: read("source-serif-4/files/source-serif-4-latin-600-normal.woff"), weight: 600, style: "normal" },
      { name: "Inter", data: read("inter/files/inter-latin-400-normal.woff"), weight: 400, style: "normal" },
      { name: "Inter", data: read("inter/files/inter-latin-600-normal.woff"), weight: 600, style: "normal" },
    ];
  }
  return fonts;
}

/** The crop as a data URL, or null when neither it nor the default crop exists. */
function cropDataUrl(name: string): string | null {
  for (const candidate of [name, "default"]) {
    const file = path.join(CROPS_DIR, `${candidate}.png`);
    if (fs.existsSync(file)) {
      return `data:image/png;base64,${fs.readFileSync(file).toString("base64")}`;
    }
  }
  return null;
}

type Node = { type: string; props: Record<string, unknown> };

function el(type: string, style: Record<string, unknown>, children?: unknown, extra: Record<string, unknown> = {}): Node {
  return { type, props: { style, children, ...extra } };
}

function layout(card: ShareCard): Node {
  const crop = cropDataUrl(card.crop);
  const text = el("div", { display: "flex", flexDirection: "column", position: "absolute", left: 72, top: 64, width: 560 }, [
    el("div", { display: "flex", fontFamily: "Source Serif 4", fontSize: 34, fontWeight: 600, color: PAPER.text }, [
      el("span", { color: PAPER.secondary, marginRight: 10 }, "TorchLens"),
      el("span", {}, "Model Menagerie"),
    ]),
    el(
      "div",
      { display: "flex", marginTop: 72, fontFamily: "Source Serif 4", fontSize: card.headline.length > 28 ? 52 : 64, fontWeight: 600, lineHeight: 1.1, color: PAPER.text },
      card.headline,
    ),
    el("div", { display: "flex", marginTop: 24, fontFamily: "Inter", fontSize: 28, lineHeight: 1.4, color: PAPER.secondary }, card.subline),
  ]);
  const footer = el(
    "div",
    { display: "flex", flexDirection: "column", position: "absolute", left: 72, bottom: 52, width: 560, fontFamily: "Inter" },
    [
      el("div", { display: "flex", fontSize: 26, fontWeight: 600, color: PAPER.text }, "modelmenagerie.ai"),
      card.caption ? el("div", { display: "flex", marginTop: 6, fontSize: 20, color: PAPER.secondary }, card.caption) : null,
    ].filter(Boolean),
  );
  const inset = el(
    "div",
    {
      display: "flex",
      position: "absolute",
      left: 680,
      top: 48,
      width: CROP_WIDTH + 2,
      height: CROP_HEIGHT + 1,
      background: PAPER.surface,
      border: `1px solid ${PAPER.border}`,
      borderBottom: "none",
      borderTopLeftRadius: 14,
      borderTopRightRadius: 14,
      overflow: "hidden",
    },
    crop ? el("img", { width: CROP_WIDTH, height: CROP_HEIGHT }, undefined, { src: crop, width: CROP_WIDTH, height: CROP_HEIGHT }) : undefined,
  );
  return el("div", { display: "flex", position: "relative", width: SHARE_IMAGE_WIDTH, height: SHARE_IMAGE_HEIGHT, background: PAPER.bg }, [text, footer, inset]);
}

/** Render one share image to PNG bytes. */
export async function renderShareImage(card: ShareCard): Promise<Buffer> {
  const svg = await satori(layout(card) as Parameters<typeof satori>[0], {
    width: SHARE_IMAGE_WIDTH,
    height: SHARE_IMAGE_HEIGHT,
    fonts: loadFonts(),
  });
  return new Resvg(svg, { fitTo: { mode: "width", value: SHARE_IMAGE_WIDTH } }).render().asPng();
}
