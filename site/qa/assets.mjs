// usage: node qa/assets.mjs DIST OUT.json
// The asset-host references of the built site. Every file address on the asset host (PUBLIC_R2_BASE)
// that a built page or data file names must be one the export names (site-data: models.jsonl, the
// families and the dataset release in export_manifest.json), so nothing points past what the publish
// loop staged and uploaded (scripts/site_publish.py reconciles the export against the upload). Then a
// sample of each kind, and every dataset release file, is asked for on the asset host itself (HEAD,
// with the cache key), and must answer 200. Without PUBLIC_R2_BASE the build names no asset host and
// the check writes nothing (its gates read as skipped).
import fs from "node:fs";
import path from "node:path";

import { makeAssetHref } from "../src/lib/asset-href.mjs";

const [distArg, out] = process.argv.slice(2);
const dist = path.resolve(distArg);
const root = path.join(dist, "..");
const base = (process.env.PUBLIC_R2_BASE ?? "").replace(/\/+$/, "");
const SAMPLE_PER_KIND = Number(process.env.ASSET_HEAD_SAMPLE ?? 8);
const TIMEOUT_MS = 20_000;

if (!base) {
  console.log("no PUBLIC_R2_BASE: the build names no asset host; asset checks skipped");
  process.exit(0);
}

const escaped = base.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
const ADDRESS = new RegExp(`${escaped}/([A-Za-z0-9._/-]+)`, "g");

/** Every file under the build. */
function walk(dir, files = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, files);
    else if (/\.(html|json|xml|js|txt|webmanifest)$/.test(entry.name)) files.push(full);
  }
  return files;
}

/** Keys the built site names on the asset host, each with one page that names it. */
const named = new Map();
for (const file of walk(dist)) {
  const text = fs.readFileSync(file, "utf8");
  for (const match of text.matchAll(ADDRESS)) {
    if (!named.has(match[1])) named.set(match[1], `/${path.relative(dist, file).split(path.sep).join("/")}`);
  }
}

/** Keys the export names: `/export/assets/<key>` anywhere in its data files, plus the release files. */
const exported = new Set();
for (const name of ["models.jsonl", "families.jsonl", "roster-index.jsonl", "diagram_focus.json"]) {
  const file = path.join(root, "site-data", name);
  if (!fs.existsSync(file)) continue;
  for (const match of fs.readFileSync(file, "utf8").matchAll(/\/export\/assets\/([A-Za-z0-9._-]+)/g)) exported.add(match[1]);
}
const manifestFile = path.join(root, "site-data", "export_manifest.json");
const manifest = fs.existsSync(manifestFile) ? JSON.parse(fs.readFileSync(manifestFile, "utf8")) : {};
const release = manifest.dataset_release;
const releaseKeys = (release?.files ?? []).map((item) => item.path ?? `dataset/${release.version}/${item.name}`);
releaseKeys.forEach((key) => exported.add(key));

const unknown = [...named].filter(([key]) => !exported.has(key));

/** A key's kind: its file ending. */
function kind(key) {
  const name = key.split("/").pop();
  for (const ending of [".netron.json", ".model-explorer.json", ".jsonl.gz"]) if (name.endsWith(ending)) return ending.slice(1);
  return name.includes(".") ? name.split(".").pop() : "?";
}

const byKind = new Map();
for (const key of named.keys()) {
  const k = kind(key);
  if (!byKind.has(k)) byKind.set(k, []);
  byKind.get(k).push(key);
}

/** Evenly spaced picks from a sorted list, so the sample is the same on every run of one build. */
function spread(keys, n) {
  const sorted = [...keys].sort();
  if (sorted.length <= n) return sorted;
  return Array.from({ length: n }, (_, i) => sorted[Math.floor((i * sorted.length) / n)]);
}

const href = makeAssetHref(base, process.env.PUBLIC_ASSET_CACHE_KEY);
const sample = [...new Set([...[...byKind.values()].flatMap((keys) => spread(keys, SAMPLE_PER_KIND)), ...releaseKeys])];

async function head(key) {
  const url = href(key);
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const response = await fetch(url, { method: "HEAD", redirect: "manual", signal: AbortSignal.timeout(TIMEOUT_MS) });
      if (response.status < 500) return { key, kind: kind(key), status: response.status, type: response.headers.get("content-type") };
    } catch (error) {
      if (attempt === 2) return { key, kind: kind(key), status: 0, error: String(error).slice(0, 120) };
    }
    await new Promise((resolve) => setTimeout(resolve, 1000 * (attempt + 1)));
  }
  return { key, kind: kind(key), status: 0, error: "server errors" };
}

const results = [];
for (let i = 0; i < sample.length; i += 8) results.push(...(await Promise.all(sample.slice(i, i + 8).map(head))));
const failures = results.filter((row) => row.status !== 200);

const summary = {
  base,
  named_keys: named.size,
  named_by_kind: Object.fromEntries([...byKind].map(([k, keys]) => [k, keys.length]).sort()),
  exported_keys: exported.size,
  not_in_export: unknown.length,
  not_in_export_sample: unknown.slice(0, 20).map(([key, page]) => ({ kind: kind(key), page })),
  release_files: releaseKeys.length,
  head_checked: results.length,
  head_by_kind: Object.fromEntries([...new Set(results.map((r) => r.kind))].sort().map((k) => [k, results.filter((r) => r.kind === k).length])),
  head_failures: failures.length,
  head_failure_sample: failures.slice(0, 20).map((row) => ({ kind: row.kind, status: row.status, error: row.error })),
};
fs.writeFileSync(out, JSON.stringify(summary, null, 1));
console.log(JSON.stringify(summary, null, 1));
