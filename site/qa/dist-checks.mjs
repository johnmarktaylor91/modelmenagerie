// usage: node qa/dist-checks.mjs DIST OUT.json
// Checks the built site on disk: file count and largest file against the hosting caps, every internal
// link and image in every HTML page resolves to a built file, and the copy rules on every page.
import fs from "node:fs";
import path from "node:path";

import { checkCopy, dataNames } from "./lib/copy.mjs";

const [distArg, out] = process.argv.slice(2);
const dist = path.resolve(distArg);

/** Every file under the build, relative, with its size. */
function walk(dir, files = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, files);
    else files.push({ rel: path.relative(dist, full).split(path.sep).join("/"), bytes: fs.statSync(full).size });
  }
  return files;
}

const files = walk(dist);
const present = new Set(files.map((file) => file.rel));
const largest = files.reduce((a, b) => (b.bytes > a.bytes ? b : a), { rel: "", bytes: 0 });
const htmlFiles = files.filter((file) => file.rel.endsWith(".html"));
const exempt = dataNames(path.join(dist, "..", "site-data", "models.jsonl"));

/** The built file a site path is served from, or null when nothing would answer it. */
function resolves(sitePath) {
  const clean = decodeURIComponent(sitePath.split(/[?#]/)[0]).replace(/^\//, "");
  if (clean === "" || clean.endsWith("/")) return present.has(`${clean}index.html`);
  return present.has(clean);
}

const broken = new Map();
const missingSlash = new Map();
const copyHits = [];
let links = 0;
for (const file of htmlFiles) {
  const html = fs.readFileSync(path.join(dist, file.rel), "utf8");
  const page = `/${file.rel.replace(/index\.html$/, "")}`;
  const body = html.replace(/<script\b[\s\S]*?<\/script>/gi, "");
  for (const match of body.matchAll(/\s(?:href|src)="(\/[^"/][^"]*|\/)"/g)) {
    const target = match[1].replace(/&amp;/g, "&");
    links += 1;
    if (!resolves(target)) {
      if (!broken.has(target)) broken.set(target, page);
    } else if (!target.split(/[?#]/)[0].endsWith("/") && !/\.[a-z0-9]+$/i.test(target.split(/[?#]/)[0])) {
      // A page link without its trailing slash answers with a redirect on Pages.
      if (!missingSlash.has(target)) missingSlash.set(target, page);
    }
  }
  for (const hit of checkCopy(html, exempt)) {
    if (copyHits.length < 200) copyHits.push({ page, ...hit });
  }
}

const copyByRule = {};
for (const hit of copyHits) copyByRule[hit.rule] = (copyByRule[hit.rule] ?? 0) + 1;
const summary = {
  files: files.length,
  html_pages: htmlFiles.length,
  bytes: files.reduce((sum, file) => sum + file.bytes, 0),
  largest_file: largest,
  links_checked: links,
  broken_links: broken.size,
  broken_sample: [...broken].slice(0, 30).map(([target, from]) => ({ target, from })),
  links_without_slash: missingSlash.size,
  without_slash_sample: [...missingSlash].slice(0, 30).map(([target, from]) => ({ target, from })),
  copy_hits: copyHits.length,
  copy_by_rule: copyByRule,
  copy_sample: copyHits.slice(0, 60),
  // Not gated: catalog names that break the copy rules, for the data to be cleaned upstream.
  data_names_with_copy_hits: exempt.length,
  data_names_sample: exempt.slice(0, 40),
};
fs.writeFileSync(out, JSON.stringify(summary, null, 1));
console.log(JSON.stringify({ ...summary, broken_sample: summary.broken_sample.slice(0, 5), copy_sample: summary.copy_sample.slice(0, 10) }, null, 1));
