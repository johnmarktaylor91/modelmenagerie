// usage: node qa/lighthouse.mjs BASEURL OUTDIR
// Lighthouse (mobile preset, one run per route) on the key page types; writes each report and a
// summary.json with the scores and timings the verdict gates on. Single runs on a shared worker
// differ by a point or two between runs; treat small changes as noise.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const [base, outDir] = process.argv.slice(2);
fs.mkdirSync(outDir, { recursive: true });
const ROUTES = {
  home: "/",
  listing: "/models/",
  small: "/models/pi-sigma-network/",
  large: "/models/espnet-tts-tacotron2-gst/",
  monster: "/models/nllb-moe/",
  dataset: "/dataset/",
  about: "/about/",
  legal: "/legal/",
  changelog: "/changelog/",
};
const bin = path.join(path.dirname(new URL(import.meta.url).pathname), "node_modules", ".bin", "lighthouse");

const summary = {};
for (const [name, route] of Object.entries(ROUTES)) {
  const file = path.join(outDir, `${name}-mobile.json`);
  try {
    execFileSync(
      bin,
      [base + route, "--chrome-flags=--headless=new --no-sandbox --disable-dev-shm-usage", "--output=json", `--output-path=${file}`, "--quiet"],
      { stdio: "ignore", timeout: 180000, env: { ...process.env, CHROME_PATH: process.env.CHROME_PATH ?? process.env.CHROME } },
    );
    const report = JSON.parse(fs.readFileSync(file, "utf8"));
    const audits = report.audits;
    const score = (key) => Math.round((report.categories[key]?.score ?? 0) * 100);
    const items = audits["network-requests"]?.details?.items ?? [];
    summary[name] = {
      route,
      performance: score("performance"),
      accessibility: score("accessibility"),
      best_practices: score("best-practices"),
      seo: score("seo"),
      total_blocking_time_ms: Math.round(audits["total-blocking-time"].numericValue),
      cumulative_layout_shift: Number(audits["cumulative-layout-shift"].numericValue.toFixed(3)),
      largest_contentful_paint_s: Number((audits["largest-contentful-paint"].numericValue / 1000).toFixed(2)),
      transfer_kib: Math.round(items.reduce((sum, item) => sum + Number(item.transferSize ?? 0), 0) / 1024),
    };
  } catch (error) {
    summary[name] = { route, error: String(error).slice(0, 200) };
  }
  console.log(name, JSON.stringify(summary[name]));
}
fs.writeFileSync(path.join(outDir, "summary.json"), JSON.stringify(summary, null, 1));
