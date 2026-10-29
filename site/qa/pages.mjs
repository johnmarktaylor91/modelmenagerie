// usage: node qa/pages.mjs BASEURL OUTDIR
// Each page type at 390, 800 and 1280 px in light and dark: sideways scroll, console errors, failed
// same-origin requests and a screenshot; axe-core (every impact) at 390 and 1280 px.
import fs from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright-core");
const axeSource = fs.readFileSync(require.resolve("axe-core/axe.min.js"), "utf8");

const [base, outDir] = process.argv.slice(2);
const shots = `${outDir}/shots`;
fs.mkdirSync(shots, { recursive: true });

export const PAGES = {
  home: "/",
  listing: "/models/",
  "listing-2": "/models/2/",
  "model-small": "/models/resnet18/",
  "model-wide": "/models/gpt2/",
  "model-large": "/models/espnet-tts-tacotron2-gst/",
  "model-monster": "/models/nllb-moe/",
  dataset: "/dataset/",
  about: "/about/",
  legal: "/legal/",
  changelog: "/changelog/",
  notfound: "/no-such-page/",
};
const WIDTHS = [
  [390, 844],
  [800, 900],
  [1280, 900],
];
const AXE_WIDTHS = new Set([390, 1280]);

// A family page, when the site has them: the first one in the sitemap.
try {
  const index = await (await fetch(`${base}/sitemap-index.xml`)).text();
  for (const loc of index.matchAll(/<loc>([^<]+)<\/loc>/g)) {
    const xml = await (await fetch(base + new URL(loc[1]).pathname)).text();
    const family = [...xml.matchAll(/<loc>([^<]+\/families\/[^<]+\/)<\/loc>/g)][0];
    if (family) {
      PAGES.family = new URL(family[1]).pathname;
      break;
    }
  }
} catch {
  // No sitemap: check the fixed list only.
}

const browser = await chromium.launch({ executablePath: process.env.CHROME || undefined, args: ["--no-sandbox", "--disable-dev-shm-usage"] });
const rows = [];
for (const [name, pathname] of Object.entries(PAGES)) {
  for (const [width, height] of WIDTHS) {
    for (const scheme of ["light", "dark"]) {
      const context = await browser.newContext({ viewport: { width, height }, colorScheme: scheme, deviceScaleFactor: 1, hasTouch: width < 600, isMobile: width < 600 });
      const page = await context.newPage();
      const errors = [];
      const failed = [];
      page.on("pageerror", (error) => errors.push(String(error)));
      page.on("console", (message) => {
        // The 404 page's own status is reported as a console error; that one is expected.
        const own404 = pathname === PAGES.notfound && /status of 404/.test(message.text()) && message.location()?.url === base + pathname;
        if (message.type() === "error" && !own404) errors.push(message.text());
      });
      page.on("response", (response) => {
        const url = response.url();
        if (url.startsWith(base) && response.status() >= 400 && !url.endsWith(PAGES.notfound)) failed.push(`${response.status()} ${url.slice(base.length)}`);
      });
      const response = await page.goto(base + pathname, { waitUntil: "networkidle", timeout: 60000 }).catch((error) => {
        errors.push(`goto ${error.message}`);
        return null;
      });
      await page.waitForTimeout(400);
      const probe = await page.evaluate(() => {
        const doc = document.documentElement;
        const wide = [...document.querySelectorAll("body *")]
          .filter((el) => el.getBoundingClientRect().right > doc.clientWidth + 1 && getComputedStyle(el).position !== "fixed")
          .slice(0, 3)
          .map((el) => `${el.tagName.toLowerCase()}.${[...el.classList].join(".")}`);
        return { scrollWidth: doc.scrollWidth, clientWidth: doc.clientWidth, theme: doc.dataset.theme ?? null, title: document.title, wide };
      });
      let axe = null;
      if (AXE_WIDTHS.has(width)) {
        await page.addScriptTag({ content: axeSource });
        axe = await page.evaluate(async () => {
          const result = await window.axe.run(document, { resultTypes: ["violations"] });
          return result.violations.map((v) => ({ id: v.id, impact: v.impact, nodes: v.nodes.length, target: v.nodes[0]?.target?.join(" ") }));
        });
      }
      await page.screenshot({ path: `${shots}/${name}-${width}-${scheme}.png` });
      const row = {
        name,
        path: pathname,
        width,
        scheme,
        status: response?.status() ?? 0,
        overflow: probe.scrollWidth > probe.clientWidth,
        ...probe,
        theme_ok: probe.theme === scheme,
        errors,
        failed,
        axe,
      };
      rows.push(row);
      console.log(name, width, scheme, "status", row.status, "overflow", row.overflow, "axe", (axe ?? []).map((v) => `${v.id}:${v.impact}`).join(",") || "-", "errors", errors.length, "failed", failed.length);
      await context.close();
    }
  }
}
await browser.close();
fs.writeFileSync(`${outDir}/pages.json`, JSON.stringify(rows, null, 1));
