import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sitemap from "@astrojs/sitemap";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "astro/config";
import { DATASET_FILES, PAGES_MAX_FILE_BYTES, publicDatasetText } from "./src/lib/dataset-files.mjs";

/** Asset base for diagram files (the hosting build sets it). */
const PUBLIC_R2_BASE = process.env.PUBLIC_R2_BASE ?? "";

/**
 * Copy the committed catalog files into the build so the Dataset page links to same-origin downloads.
 * Diagram addresses inside them are rewritten from the export's relative paths to full public URLs, so a
 * downloaded file links to the diagrams it lists.
 */
function datasetDownloads() {
  return {
    name: "dataset-downloads",
    hooks: {
      "astro:build:done": ({ dir, logger }) => {
        const outDir = path.join(fileURLToPath(dir), "dataset");
        fs.mkdirSync(outDir, { recursive: true });
        for (const name of DATASET_FILES) {
          const source = fileURLToPath(new URL(`./site-data/${name}`, import.meta.url));
          const text = publicDatasetText(fs.readFileSync(source, "utf-8"), PUBLIC_R2_BASE);
          const size = Buffer.byteLength(text);
          if (size > PAGES_MAX_FILE_BYTES) {
            logger.warn(`${name} is ${size} bytes, above the hosting limit; not copied.`);
            continue;
          }
          fs.writeFileSync(path.join(outDir, name), text);
        }
        logger.info(`dataset files copied to ${outDir}`);
      },
    },
  };
}

export default defineConfig({
  site: "https://modelmenagerie.ai",
  output: "static",
  trailingSlash: "always",
  integrations: [
    sitemap({
      // Pages only: share images and the search index are files, and the 404 page is not a page to index.
      filter: (page) => page.endsWith("/") && !/\/404\/?$/.test(new URL(page).pathname),
    }),
    datasetDownloads(),
  ],
  vite: {
    plugins: [tailwindcss()],
    // The share-image rasteriser is a native module; load it from node_modules instead of bundling it.
    ssr: { external: ["@resvg/resvg-js"] },
  },
});
