// usage: node qa/verdict.mjs OUTDIR THRESHOLDS.json
// Reads what run.sh wrote in OUTDIR, compares it with the thresholds and writes OUTDIR/verdict.json.
// Prints one line per gate; exits 1 when any gate fails. A gate whose input is absent (for example the
// file checks when checking a deployed site) is reported as skipped, not passed.
import fs from "node:fs";
import path from "node:path";

const [outDir, thresholdsFile] = process.argv.slice(2);
const t = JSON.parse(fs.readFileSync(thresholdsFile, "utf8"));
const read = (name) => {
  const file = path.join(outDir, name);
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : null;
};

const check = read("check.json");
const build = read("build.json");
const dist = read("dist.json");
const crawl = read("crawl.json")?.summary ?? null;
const pages = read("pages.json");
const meta = read("meta.json");
const lighthouse = read("lighthouse/summary.json");

const gates = [];
function gate(name, value, limit, pass, detail = undefined) {
  gates.push({ name, value, limit, status: value === null || value === undefined ? "skipped" : pass ? "pass" : "fail", ...(detail ? { detail } : {}) });
}

gate("astro check errors", check?.errors, `<= ${t.astro_check_errors}`, check && check.exit === 0 && check.errors <= t.astro_check_errors);
gate("build seconds", build?.seconds, `<= ${t.build_seconds}`, build && build.exit === 0 && build.seconds <= t.build_seconds);
const fileLimit = Math.floor(t.file_cap * (1 - t.file_headroom));
gate("built files", dist?.files, `<= ${fileLimit} (cap ${t.file_cap} less ${t.file_headroom * 100}%)`, dist && dist.files <= fileLimit);
gate("largest file bytes", dist?.largest_file?.bytes, `<= ${t.max_file_bytes}`, dist && dist.largest_file.bytes <= t.max_file_bytes, dist?.largest_file?.rel);
gate("broken links in build", dist?.broken_links, `<= ${t.broken_links}`, dist && dist.broken_links <= t.broken_links, dist?.broken_sample?.slice(0, 5));
gate("page links without trailing slash", dist?.links_without_slash, `<= ${t.internal_redirects}`, dist && dist.links_without_slash <= t.internal_redirects, dist?.without_slash_sample?.slice(0, 5));
const copyHits = dist ? dist.copy_hits : crawl?.copy_hits;
gate("copy hits", copyHits, `<= ${t.copy_hits}`, copyHits <= t.copy_hits, (dist?.copy_sample ?? crawl?.copy_sample ?? []).slice(0, 8));
gate("crawl: sitemap", crawl ? (crawl.sitemap_ok ? 1 : 0) : null, "= 1", crawl?.sitemap_ok, crawl ? `${crawl.sitemap_urls} urls` : undefined);
gate("crawl: pages failing", crawl?.page_failures?.length, "= 0", crawl && crawl.page_failures.length === 0, crawl?.page_failures?.slice(0, 5));
gate("crawl: broken targets", crawl?.not_found, `<= ${t.broken_links}`, crawl && crawl.non_ok - crawl.redirects <= t.broken_links, crawl?.bad_sample?.filter((c) => c.status < 300 || c.status >= 400).slice(0, 5));
gate("crawl: internal redirects", crawl?.redirects, `<= ${t.internal_redirects}`, crawl && crawl.redirects <= t.internal_redirects, crawl?.bad_sample?.filter((c) => c.status >= 300 && c.status < 400).slice(0, 5));

if (pages) {
  const overflow = pages.filter((row) => row.overflow);
  gate("sideways scroll (page x width x theme)", overflow.length, `<= ${t.overflow_pages}`, overflow.length <= t.overflow_pages, overflow.slice(0, 6).map((r) => `${r.name} ${r.width} ${r.scheme} ${r.wide.join(" ")}`));
  const errors = pages.filter((row) => row.errors.length > 0 || row.failed.length > 0);
  gate("console errors or failed requests", errors.length, `<= ${t.console_errors}`, errors.length <= t.console_errors, errors.slice(0, 6).map((r) => `${r.name} ${r.width} ${r.scheme}: ${[...r.errors, ...r.failed].slice(0, 2).join("; ")}`));
  const allow = new Set(t.axe_allow);
  const violations = pages.flatMap((row) => (row.axe ?? []).filter((v) => !allow.has(v.id)).map((v) => `${row.name} ${row.width} ${row.scheme} ${v.id} (${v.impact}) ${v.target ?? ""}`));
  gate("axe violations", violations.length, `<= ${t.axe_violations}`, violations.length <= t.axe_violations, violations.slice(0, 10));
  const theme = pages.filter((row) => !row.theme_ok && row.status === 200);
  gate("theme follows the system setting", theme.length, "= 0", theme.length === 0, theme.slice(0, 4).map((r) => `${r.name} ${r.width} ${r.scheme}`));
} else {
  gate("page checks", null, "", false);
}

gate("share previews and structured data", meta?.problems, "= 0", meta?.problems === 0, meta ? Object.entries(meta.routes).filter(([, r]) => r.problems.length).map(([k, r]) => `${k}: ${r.problems.join("; ")}`) : undefined);
if (meta) {
  const heavy = Object.entries(meta.routes).filter(([, r]) => r.image && r.image.bytes > t.share_image.max_bytes);
  gate("share image bytes", Math.max(...Object.values(meta.routes).map((r) => r.image?.bytes ?? 0)), `<= ${t.share_image.max_bytes}`, heavy.length === 0);
}

if (lighthouse) {
  for (const [name, row] of Object.entries(lighthouse)) {
    if (row.error) {
      gate(`lighthouse ${name}`, null, "", false, row.error);
      continue;
    }
    const L = t.lighthouse;
    const misses = [
      row.performance < L.performance && `performance ${row.performance}`,
      row.total_blocking_time_ms > L.total_blocking_time_ms && `TBT ${row.total_blocking_time_ms} ms`,
      row.cumulative_layout_shift > L.cumulative_layout_shift && `CLS ${row.cumulative_layout_shift}`,
      row.accessibility < L.accessibility && `accessibility ${row.accessibility}`,
      row.seo < L.seo && `SEO ${row.seo}`,
    ].filter(Boolean);
    gate(`lighthouse ${name}`, row.performance, `perf >= ${L.performance}, TBT <= ${L.total_blocking_time_ms}, CLS <= ${L.cumulative_layout_shift}, a11y >= ${L.accessibility}, SEO >= ${L.seo}`, misses.length === 0, misses.length ? misses : `a11y ${row.accessibility}, SEO ${row.seo}, TBT ${row.total_blocking_time_ms} ms, CLS ${row.cumulative_layout_shift}, LCP ${row.largest_contentful_paint_s} s, ${row.transfer_kib} KiB`);
  }
} else {
  gate("lighthouse", null, "", false);
}

const failed = gates.filter((g) => g.status === "fail");
const verdict = { green: failed.length === 0, failed: failed.length, skipped: gates.filter((g) => g.status === "skipped").length, gates };
fs.writeFileSync(path.join(outDir, "verdict.json"), JSON.stringify(verdict, null, 1));
for (const g of gates) {
  console.log(`${g.status.padEnd(7)} ${g.name}: ${g.value ?? "-"} (${g.limit})${g.status === "fail" && g.detail ? ` ${JSON.stringify(g.detail).slice(0, 300)}` : ""}`);
}
console.log(verdict.green ? `VERDICT green (${verdict.skipped} skipped)` : `VERDICT red: ${failed.length} failing`);
process.exitCode = verdict.green ? 0 : 1;
