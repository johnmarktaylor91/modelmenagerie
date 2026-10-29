# Model Menagerie website

Astro static site for Model Menagerie (https://modelmenagerie.ai), the public gallery of neural-network architectures traced with TorchLens. This directory is developed as `site/` inside the Menagerie repo, which also holds the code that finds, runs and renders the models.

The site commits lightweight catalog data in `site-data/` and serves content-addressed diagram assets from Cloudflare R2.

## Run

Install dependencies and run the site:

```bash
npm install
npm run dev
```

Asset URLs are composed from `PUBLIC_R2_BASE`. If it is unset, the site falls back to `/assets/<sha256>.<ext>` for local preview. To preview locally without R2, copy assets from a generated export into Astro public assets:

```bash
mkdir -p public/assets
cp -a export/assets/. public/assets/
```

Type-check and build the static site:

```bash
PUBLIC_R2_BASE=https://assets.modelmenagerie.ai npx astro check
PUBLIC_R2_BASE=https://assets.modelmenagerie.ai npm run build
```

Search runs in the browser over `models-index.json`; there is no separate search index build step.

## Deploy

End-to-end deployment flow:

```bash
python scripts/build_catalog.py \
  --gallery /path/to/menagerie_gallery \
  --manifest /path/to/menagerie_gallery/manifest.tsv \
  --out ./export \
  --jobs 8
python scripts/sync_site_data.py --export-dir ./export --site-data ./site-data
git add site-data
git commit -m "feat: update catalog data"
scripts/upload_assets.sh
git push
```

Cloudflare Pages should be connected to the GitHub repo and configured with:

- Build command: `npm run build`
- Build output directory: `dist`
- Environment variable: `PUBLIC_R2_BASE=https://assets.modelmenagerie.ai` (the bucket's custom domain)
- Environment variable: `PAGES_WRANGLER_MAJOR_VERSION=4`

R2 must expose the uploaded object keys through the public URL used by `PUBLIC_R2_BASE`. Its CORS policy allows `GET` and `HEAD` from https://modelmenagerie.ai, https://www.modelmenagerie.ai and the Pages preview addresses (the viewer's save button fetches the file as a blob).

`scripts/upload_assets.sh` uses `rclone` and is idempotent for content-addressed assets. Either configure an rclone S3 remote named `r2` pointed at the target bucket, or set `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, and `R2_BUCKET`.

For a small local export refresh, keep using `--limit` on `build_catalog.py`, then run `sync_site_data.py`:

```bash
python scripts/build_catalog.py --gallery /path/to/menagerie_gallery --manifest /path/to/menagerie_gallery/manifest.tsv --out ./export --limit 600
python scripts/sync_site_data.py --export-dir ./export --site-data ./site-data
npm run build
```

## Regenerating the Catalog

`scripts/build_catalog.py` is the re-runnable Phase 0 export builder:

```bash
python scripts/build_catalog.py --gallery /path/to/menagerie_gallery --manifest /path/to/menagerie_gallery/manifest.tsv --out ./export --limit 600
```

It reads rendered manifest rows, verifies each SVG exists, copies SVGs into content-addressed assets, generates WebP thumbnails, and emits:

- `models/<slug>.json`
- `models.jsonl`
- `funnel.json`
- `export_manifest.json`
- `facets.json`
- `catalog.csv`
- `assets/<sha256>.svg` and `assets/<sha256>.webp`

`scripts/diagram_focus.py` then records, for every diagram wider than a phone at reading scale, where the
drawing starts across its top (`site-data/diagram_focus.json`); the viewer opens there instead of the middle.

Only aggregate text data is committed:

- `site-data/models.jsonl`
- `site-data/facets.json`
- `site-data/funnel.json`
- `site-data/export_manifest.json`
- `site-data/catalog.csv`
- `site-data/diagram_focus.json`

Do not commit `export/models/*.json` or `export/assets/`.

Use `--limit 600` for a fast local preview. To scale to the full rendered gallery, omit `--limit` or set a higher value:

```bash
python scripts/build_catalog.py --gallery /path/to/menagerie_gallery --manifest /path/to/menagerie_gallery/manifest.tsv --out ./export --jobs 8
```

Full-scale thumbnail generation is intentionally a separate operator-triggered step.

## Data Loading

All catalog access goes through `src/lib/catalog.ts`, which reads only committed `site-data/` files. Pages and components consume typed records from:

- `getAllModels`
- `getModelBySlug`
- `listModelSlugs`
- `getFunnel`
- `getCatalogIndex`
- `getFeaturedModels`

`models.jsonl` is loaded once and indexed in memory by slug. The model schema already includes the Tier-B fields, with honest placeholders such as `forward_validated: false`, `source_license: "unknown"`, `param_count: null`, `tlspec: null`, and exactly one current variant.

## Site shell

- `src/layouts/BaseLayout.astro` holds the header, footer, page metadata (through `components/SeoHead.astro`) and
  the `DEFAULT_THEME` switch at the top (`"system"`, `"light"` or `"dark"`); the preview tag sits in its header
  markup. The theme is set by a small inline script in `<head>` before
  first paint.
- `src/styles/global.css` holds every colour and font as a CSS variable, exposed to Tailwind as `text-secondary`,
  `bg-surface`, `border-line`, `text-accent` and similar. Changing the palette or the heading face is a token edit
  there; do not add hard-coded colours or `style=` attributes.
- `astro.config.mjs` sets `trailingSlash: "always"`; every internal link ends with `/`. Its `dataset-downloads`
  integration copies `site-data/catalog.csv`, `models.jsonl` and `export_manifest.json` into `dist/dataset/` so the
  Dataset page links to same-origin downloads (files above the 25 MiB Pages limit are skipped and get no link).
- `public/_headers` sets one-year immutable caching for `/_astro/*`, a short cache for `models-index.json`, and
  `Content-Disposition: attachment` for the dataset files. HTML keeps the Pages default (always revalidate).
- Share images (1200 x 630) are drawn at build time by `src/pages/og/[name].png.ts` from the crops in
  `src/assets/share/`; `components/SeoHead.astro` writes the search and sharing tags and `lib/seo.ts` the JSON-LD.
  `favicon.svg`, `favicon.ico` and `apple-touch-icon.png` are the icons.
- `qa/run.sh` is the QA harness (docs/site.md, QA harness).

## Content-Addressed Assets

Every emitted SVG and WebP thumbnail has an `AssetRef`:

```ts
interface AssetRef {
  url: string;
  sha256: string;
  bytes: number;
}
```

Generated asset references preserve the digest, byte count, and generated path in the source data. At build time, the site takes the `<sha256>.<ext>` basename and composes a public URL as `${PUBLIC_R2_BASE}/<sha256>.<ext>`. If `PUBLIC_R2_BASE` is unset, URLs fall back to `/assets/<sha256>.<ext>` for local preview.

## Scope

This repo is the separate web project. It does not modify the TorchLens package repo or the read-only rendered source gallery.
