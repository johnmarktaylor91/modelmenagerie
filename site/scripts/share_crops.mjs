// Draw the diagram crops used in share images: for each entry of SHARE_IMAGE_SLUGS in
// src/lib/featured.ts, fetch that model's default diagram (SVG) and save the top of it, at the
// size the site shows it, as src/assets/share/<name>.png (460 by 582, white paper), starting at the
// topmost node.
//
// usage, on a worker, from site/ (needs the QA harness's packages: cd qa && npm ci):
//   PUBLIC_R2_BASE=https://<asset host> CHROME=/path/to/chrome node scripts/share_crops.mjs
// Node 22.18 or later reads featured.ts directly. Commit the PNGs it writes; the build turns them
// into the share images (src/pages/og/[name].png.ts).
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(new URL("../qa/package.json", import.meta.url));
const { chromium } = require("playwright-core");

const CROP_WIDTH = 460;
const CROP_HEIGHT = 582;
/** Paper kept above the topmost node so it does not touch the inset's edge. */
const TOP_MARGIN = 28;

const siteDir = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const outDir = path.join(siteDir, "src", "assets", "share");
const base = (process.env.PUBLIC_R2_BASE ?? "").replace(/\/+$/, "");
if (!base) {
  throw new Error("Set PUBLIC_R2_BASE to the asset host the site uses.");
}

const { SHARE_IMAGE_SLUGS } = await import(pathToFileURL(path.join(siteDir, "src", "lib", "featured.ts")).href);
const shareImageName = (key) => key.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

const bySlug = new Map();
for (const line of fs.readFileSync(path.join(siteDir, "site-data", "models.jsonl"), "utf8").split("\n")) {
  if (line) {
    const record = JSON.parse(line);
    bySlug.set(record.slug, record);
  }
}

fs.mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({ executablePath: process.env.CHROME || undefined });
const page = await browser.newPage({ viewport: { width: CROP_WIDTH, height: CROP_HEIGHT }, deviceScaleFactor: 1 });
const report = [];
for (const [key, slug] of Object.entries(SHARE_IMAGE_SLUGS)) {
  const record = bySlug.get(slug);
  if (!record) {
    report.push({ key, slug, error: "not in the catalog" });
    continue;
  }
  const variant = record.variants.find((v) => v.key === record.default_variant) ?? record.variants[0];
  const response = await fetch(`${base}/${variant.svg.sha256}.svg`);
  if (!response.ok) {
    report.push({ key, slug, error: `fetch ${response.status}` });
    continue;
  }
  const svg = await response.text();
  // Inline, so the nodes can be measured. The graph's title text (model name, tensor and parameter totals)
  // is removed: the share image names the task itself. The crop starts just above the anchor node, the
  // first output if there is one (a language model's cache outputs can sit higher), else the topmost
  // node, and is centred on it.
  await page.setContent(`<!doctype html><html><body style="margin:0;background:#fff"><div id="d" style="position:absolute;left:0;top:0">${svg}</div></body></html>`);
  const box = await page.evaluate(() => {
    const root = document.querySelector("#d svg");
    for (const title of root.querySelectorAll("g.graph > text")) title.remove();
    const outer = root.getBoundingClientRect();
    const nodes = [...root.querySelectorAll("g.node")].filter((node) => node.getBoundingClientRect().width > 0);
    const label = (node) => node.querySelector("text")?.textContent?.trim() ?? "";
    const topmost = nodes.reduce((best, node) => (best === null || node.getBoundingClientRect().top < best.getBoundingClientRect().top ? node : best), null);
    const anchor = nodes.find((node) => label(node) === "output_1") ?? topmost;
    const r = anchor ? anchor.getBoundingClientRect() : { top: 0, left: 0, width: outer.width };
    return { width: outer.width, height: outer.height, nodeTop: r.top, nodeCentre: r.left + r.width / 2 };
  });
  const left = box.width <= CROP_WIDTH ? -(CROP_WIDTH - box.width) / 2 : Math.min(Math.max(0, box.nodeCentre - CROP_WIDTH / 2), box.width - CROP_WIDTH);
  const top = Math.max(0, box.nodeTop - TOP_MARGIN);
  await page.evaluate(({ left, top, width }) => {
    // Text the crop would cut through at its sides (a wide graph's module labels) is dropped, not halved.
    for (const text of document.querySelectorAll("#d svg text")) {
      const r = text.getBoundingClientRect();
      if (r.width > 0 && (r.left < left || r.right > left + width)) text.remove();
    }
    const holder = document.getElementById("d");
    holder.style.left = `${-left}px`;
    holder.style.top = `${-top}px`;
  }, { left, top, width: CROP_WIDTH });
  const name = key === "default" ? "default" : shareImageName(key);
  const file = path.join(outDir, `${name}.png`);
  await page.screenshot({ path: file, clip: { x: 0, y: 0, width: CROP_WIDTH, height: CROP_HEIGHT } });
  report.push({ key, slug, name, svg_width: Math.round(box.width), svg_height: Math.round(box.height), crop_left: Math.round(left), crop_top: Math.round(top), bytes: fs.statSync(file).size });
}
await browser.close();
console.log(JSON.stringify(report, null, 1));
if (report.some((row) => row.error)) {
  process.exitCode = 1;
}
