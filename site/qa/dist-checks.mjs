// usage: node qa/dist-checks.mjs DIST OUT.json
// Checks the built site on disk: file count and largest file against the hosting caps, every internal
// link and image in every HTML page resolves to a built file, and the copy rules on every page.
import fs from "node:fs";
import path from "node:path";

import { checkCopy, dataNames, quotedHits } from "./lib/copy.mjs";
import { tiers } from "./lib/routes.mjs";

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
// Every target in the Pages redirects file must be a built page; a dead target counts as a broken link.
const redirectsFile = path.join(dist, "_redirects");
if (fs.existsSync(redirectsFile)) {
  for (const line of fs.readFileSync(redirectsFile, "utf8").split("\n")) {
    const [source, target] = line.replace(/#.*/, "").trim().split(/\s+/);
    if (source && target?.startsWith("/") && !resolves(target) && !broken.has(target)) broken.set(target, "/_redirects");
  }
}
const missingSlash = new Map();
const copyHits = [];
const quoted = [];
let quotedCount = 0;
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
  for (const hit of quotedHits(html)) {
    quotedCount += 1;
    if (quoted.length < 40) quoted.push({ page, ...hit });
  }
}

/**
 * Counts typed into the source: every count in the export manifest and funnel (100 or more) is
 * looked for, plain and with thousands separators, in the page source. Counts must be read from
 * the data at build time so they never go stale. Comments and all-number arrays (pixel windows
 * such as `crop: [10, 1820, 430]`) are left out: they are layout, and a daily count that happens
 * to equal one is a chance match, not a typed count.
 */
function withoutLayoutNumbers(text) {
  return text
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/(^|[^:])\/\/.*$/gm, "$1")
    .replace(/\[\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*\s*\]/g, " ");
}

function typedCounts() {
  const root = path.join(dist, "..");
  const values = new Set();
  const collect = (value) => {
    if (typeof value === "number" && Number.isInteger(value) && value >= 100) values.add(value);
    else if (value && typeof value === "object") Object.values(value).forEach(collect);
  };
  for (const name of ["export_manifest.json", "funnel.json"]) {
    const file = path.join(root, "site-data", name);
    if (fs.existsSync(file)) collect(JSON.parse(fs.readFileSync(file, "utf8")));
  }
  const years = (value) => value >= 1900 && value <= 2100;
  const forms = [...values].filter((value) => !years(value)).flatMap((value) => [String(value), value.toLocaleString("en-US")]);
  const hits = [];
  const scan = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name !== "fixtures") scan(full);
      } else if (/\.(astro|ts|md|mdx)$/.test(entry.name)) {
        const text = withoutLayoutNumbers(fs.readFileSync(full, "utf8"));
        for (const form of forms) {
          const pattern = new RegExp(`(?<![\\d.,])${form.replace(/[.,]/g, "\\$&")}(?![\\d]|[.,]\\d)`);
          if (pattern.test(text)) hits.push({ file: path.relative(root, full), count: form });
        }
      }
    }
  };
  if (forms.length && fs.existsSync(path.join(root, "src"))) scan(path.join(root, "src"));
  return hits;
}
const typed = typedCounts();

/**
 * Counts printed on the pages against what the build actually lists. The home, listing and dataset
 * validated counts must equal the cards in the default listing; the dataset's structure-only count
 * the structure-only pages; the dataset's "model names in the catalog" the listing total with
 * untraced entries included (pages plus roster rows). The export's funnel must say the same (gated in
 * verdict.mjs): the export holds exactly the shown set, so the manifest, downloads and pages agree.
 */
function countChecks() {
  const root = path.join(dist, "..");
  const { shown, structure } = tiers(path.join(root, "site-data", "models.jsonl"));
  const readDist = (rel) => (fs.existsSync(path.join(dist, rel)) ? fs.readFileSync(path.join(dist, rel), "utf8") : "");
  const rosterJson = readDist("roster-index.json");
  const rosterRows = rosterJson ? JSON.parse(rosterJson).rows.length : 0;
  const indexJson = readDist("models-index.json");
  const indexRows = indexJson ? JSON.parse(indexJson).rows.length : null;
  const listingTotal = shown.length + structure.length + rosterRows;
  const printed = (rel, key) => {
    const match = readDist(rel).match(new RegExp(`data-count="${key}"[^>]*>([\\d,]+)<`));
    return match ? Number(match[1].replace(/,/g, "")) : null;
  };
  const expect = [
    ["home validated", printed("index.html", "validated"), shown.length],
    ["listing validated", printed("models/index.html", "listing"), shown.length],
    ["dataset validated", printed("dataset/index.html", "validated"), shown.length],
    ["search index rows (pages)", indexRows, shown.length + structure.length],
    ...(structure.length ? [["dataset structure only", printed("dataset/index.html", "structure"), structure.length]] : []),
    ...(rosterRows ? [["dataset model names = listing total", printed("dataset/index.html", "roster"), listingTotal]] : []),
  ];
  const mismatches = expect.filter(([, got, want]) => got !== want).map(([name, got, want]) => ({ name, printed: got, expected: want }));
  const funnelFile = path.join(root, "site-data", "funnel.json");
  const funnel = fs.existsSync(funnelFile) ? JSON.parse(fs.readFileSync(funnelFile, "utf8")) : {};
  const funnelDisagreements = [
    ["validated_records", funnel.validated_records, shown.length],
    ["structure_only_records", funnel.structure_only_records, structure.length],
    ...(rosterRows ? [[typeof funnel.catalog_entries === "number" ? "catalog_entries" : "roster_names", funnel.catalog_entries ?? funnel.roster_names, listingTotal]] : []),
  ]
    .filter(([, exported, site]) => typeof exported === "number" && exported !== site)
    .map(([field, exported, site]) => ({ field, exported, site }));
  return { shown: shown.length, structure: structure.length, roster_rows: rosterRows, listing_total: listingTotal, checked: expect.length, mismatches, funnel_disagreements: funnelDisagreements };
}
const counts = countChecks();

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
  typed_counts: typed.length,
  typed_counts_sample: typed.slice(0, 20),
  count_mismatches: counts.mismatches.length,
  counts,
  // Not gated: catalog names that break the copy rules, for the data to be cleaned upstream.
  data_names_with_copy_hits: exempt.length,
  data_names_sample: exempt.slice(0, 40),
  // Not gated: rule hits in text quoted from or written from the paper (marked data-quoted).
  quoted_text_hits: quotedCount,
  quoted_text_sample: quoted,
};
fs.writeFileSync(out, JSON.stringify(summary, null, 1));
console.log(JSON.stringify({ ...summary, broken_sample: summary.broken_sample.slice(0, 5), copy_sample: summary.copy_sample.slice(0, 10) }, null, 1));
