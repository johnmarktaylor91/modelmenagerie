// usage: node qa/crawl.mjs BASEURL OUT.json [MAX_PAGES]
// Over HTTP: fetch the main pages, a spread of model pages from the sitemap and whatever they link to
// (breadth first, up to MAX_PAGES pages), then request every internal link and image target found,
// without following redirects. Also runs the copy rules on each crawled page.
import fs from "node:fs";

import { checkCopy, dataNames } from "./lib/copy.mjs";

const [base, out, maxArg] = process.argv.slice(2);
const MAX_PAGES = Number(maxArg ?? 400);
const origin = new URL(base).origin;
// Run from site/, so the catalog names are at hand when checking a deployed copy of this branch.
const exempt = dataNames("site-data/models.jsonl");
const SEEDS = ["/", "/models/", "/models/2/", "/about/", "/dataset/", "/legal/", "/changelog/", "/models/resnet18/", "/models/gpt2/"];

function internalTargets(html, pageUrl) {
  const body = html.replace(/<script\b[\s\S]*?<\/script>/gi, "").replace(/<style\b[\s\S]*?<\/style>/gi, "");
  const found = new Set();
  for (const match of body.matchAll(/\s(?:href|src|content)="([^"]+)"/g)) {
    const raw = match[1].replace(/&amp;/g, "&");
    if (/^(mailto:|tel:|javascript:|data:|#)/.test(raw)) continue;
    let url;
    try {
      url = new URL(raw, pageUrl);
    } catch {
      continue;
    }
    // Absolute links to the canonical host count as internal: map them onto the server under test.
    if (url.origin !== origin && url.host !== "modelmenagerie.ai") continue;
    if (match[0].includes("content=") && !/\.(png|jpe?g|webp)$/i.test(url.pathname)) continue;
    url.hash = "";
    found.add(url.pathname + url.search);
  }
  return [...found];
}

async function get(pathname) {
  try {
    const response = await fetch(origin + pathname, { redirect: "manual", headers: { Accept: "text/html,*/*" } });
    const type = response.headers.get("content-type") ?? "";
    const text = response.status === 200 && type.includes("html") ? await response.text() : (await response.arrayBuffer(), "");
    return { status: response.status, location: response.headers.get("location"), text };
  } catch (error) {
    return { status: 0, location: null, text: "", error: String(error) };
  }
}

const sitemapUrls = [];
const index = await get("/sitemap-index.xml");
const indexXml = index.status === 200 ? await (await fetch(origin + "/sitemap-index.xml")).text() : "";
for (const loc of indexXml.matchAll(/<loc>([^<]+)<\/loc>/g)) {
  const xml = await (await fetch(origin + new URL(loc[1]).pathname)).text();
  for (const m of xml.matchAll(/<loc>([^<]+)<\/loc>/g)) sitemapUrls.push(new URL(m[1]).pathname);
}
const models = sitemapUrls.filter((p) => /^\/models\/[^/]+\/$/.test(p) && !/^\/models\/\d+\/$/.test(p));
const others = sitemapUrls.filter((p) => !models.includes(p) && !/^\/models\/\d+\/$/.test(p));
const stride = Math.max(1, Math.floor(models.length / Math.max(1, MAX_PAGES - 80)));
const queue = [...new Set([...SEEDS, ...others.slice(0, 40)])];
for (let i = 0; i < models.length; i += stride) queue.push(models[i]);

const crawled = new Map();
const targets = new Map();
const copyHits = [];
while (queue.length && crawled.size < MAX_PAGES) {
  const pathname = queue.shift();
  if (crawled.has(pathname)) continue;
  const result = await get(pathname);
  crawled.set(pathname, result.status);
  if (result.status !== 200 || !result.text) continue;
  for (const hit of checkCopy(result.text, exempt)) if (copyHits.length < 200) copyHits.push({ page: pathname, ...hit });
  for (const target of internalTargets(result.text, origin + pathname)) {
    if (!targets.has(target)) targets.set(target, new Set());
    targets.get(target).add(pathname);
    const bare = target.split("?")[0];
    if (!crawled.has(bare) && !queue.includes(bare) && bare.endsWith("/")) queue.push(bare);
  }
}

const checks = [];
const pending = [...targets];
async function worker() {
  while (pending.length) {
    const [target, from] = pending.shift();
    const response = await fetch(origin + target, { redirect: "manual" }).catch(() => null);
    if (response) await response.arrayBuffer();
    checks.push({ target, status: response?.status ?? 0, location: response?.headers.get("location") ?? null, from: [...from].slice(0, 3) });
  }
}
await Promise.all(Array.from({ length: 6 }, worker));
const bad = checks.filter((c) => c.status !== 200);
const summary = {
  base,
  sitemap_urls: sitemapUrls.length,
  sitemap_ok: index.status === 200 && sitemapUrls.length > 0,
  pages_crawled: crawled.size,
  page_failures: [...crawled].filter(([, s]) => s !== 200),
  internal_targets: checks.length,
  non_ok: bad.length,
  redirects: bad.filter((c) => c.status >= 300 && c.status < 400).length,
  not_found: bad.filter((c) => c.status === 404).length,
  bad_sample: bad.slice(0, 40),
  copy_hits: copyHits.length,
  copy_sample: copyHits.slice(0, 40),
};
fs.writeFileSync(out, JSON.stringify({ summary, crawled: [...crawled], checks }, null, 1));
console.log(JSON.stringify({ ...summary, bad_sample: summary.bad_sample.slice(0, 8), copy_sample: summary.copy_sample.slice(0, 8) }, null, 1));
