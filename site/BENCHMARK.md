# 10k Benchmark (historical)

This page records a June 2026 benchmark of an earlier design that also built a Pagefind index. The site no longer
uses Pagefind: search runs in the browser over `models-index.json`, so the output is about one file per page. The
current figures are in "Current build" at the end.

Command:

```bash
scripts/benchmark_10k.sh
```

Latest local result, run on 2026-06-22:

| Metric | Result |
| --- | ---: |
| Fixture size | 10,000 models |
| Wall-clock build time | 106 seconds |
| Peak memory | 698,596 KB |
| Total `dist` file count | 20,144 files |
| `dist` size | 209 MB |
| Astro pages built | 10,005 pages |
| Pagefind indexed pages | 10,005 pages |
| Pagefind files | 10,096 files |

The benchmark intentionally measures the static metadata-page build plus Pagefind indexing with `TORCHLENS_SKIP_PUBLIC_FIXTURE=1`, so generated fixture assets are treated as external URLs and are not copied into `dist`. If the 10k build exceeds local memory or time limits, the first mitigation is to keep generated assets outside the Pages deployment and publish only metadata pages plus search artifacts.

Readout: the architecture holds for local 10k static generation on this machine: build time is well under 15 minutes and peak memory is under 1 GB. The output narrowly exceeds Cloudflare Pages Free's 20,000-file cap because Pagefind emits roughly one index fragment per page. Mitigation options are to deploy on Pages Pro with `PAGES_WRANGLER_MAJOR_VERSION=4`, reduce Pagefind shard/file output if supported by the selected Pagefind version, split the catalog into multiple deployments, or move search to a hosted/external index while keeping the static pages.

## Current build

Full catalog build on 2026-10-05 (`npm run build`, 4 CPU cores, review branch `site/review-2026-10-05`):

| Metric | Result |
| --- | ---: |
| Models | 14,291 |
| Astro pages built | 14,593 |
| Total `dist` file count | 14,613 |
| `dist` size | 308 MB (24.6 MB of it the dataset downloads in `dist/dataset/`) |
| Wall-clock build time | 24.4 seconds |
| Peak memory | 643,720 KB |
| Font files emitted | 4 (Inter latin 400 and 600, woff2 and woff) |

The file count stays under Cloudflare Pages Free's 20,000-file cap with 5,387 files of headroom; at about one page
per model plus one listing page per 48 models, the cap is reached near 19,500 models.
