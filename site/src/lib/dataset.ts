/**
 * What the Dataset page offers: the downloadable files with their sizes, and a dictionary of every
 * column, read from the files themselves so the page cannot drift from what people download. The
 * build fails when a column has no description here.
 */
import fs from "node:fs";
import path from "node:path";

import { ASSET_BASE, assetHref } from "@/lib/assets";
import { getExportRoot, getManifest } from "@/lib/catalog";
import { DATASET_FILES, PAGES_MAX_FILE_BYTES, publicDatasetText } from "@/lib/dataset-files.mjs";


export interface DatasetFile {
  file: string;
  label: string;
  /** File name the browser saves it under. */
  saveAs: string;
  format: string;
  href: string;
  bytes: number;
  primary: boolean;
}

const FILE_INFO: Record<string, Omit<DatasetFile, "file" | "href" | "bytes">> = {
  "catalog.csv": { label: "CSV", saveAs: "model-menagerie-catalog.csv", format: "text/csv", primary: true },
  "models.jsonl": { label: "JSON Lines", saveAs: "model-menagerie-models.jsonl", format: "application/jsonl", primary: false },
  "models.jsonl.gz": { label: "JSON Lines (gzip)", saveAs: "model-menagerie-models.jsonl.gz", format: "application/gzip", primary: false },
  "export_manifest.json": { label: "Release summary (JSON)", saveAs: "model-menagerie-release.json", format: "application/json", primary: false },
};

/**
 * The release files the export published on the asset host (``publish release``): today the full
 * records as gzipped JSON Lines, too large for the site's own origin.
 */
interface PublishedRelease {
  version: string;
  files: { name: string; path?: string; bytes: number; format?: string }[];
}

/** The release block of a site-export/1 manifest, when the export carries one and the build has an asset host. */
function publishedRelease(): PublishedRelease | null {
  const release = (getManifest() as unknown as { dataset_release?: PublishedRelease }).dataset_release;
  return release && Array.isArray(release.files) && release.files.length > 0 && ASSET_BASE ? release : null;
}

/**
 * The files to offer, one inventory for the buttons, the Python example and the structured data: the
 * catalog files the build copies to /dataset/ (see astro.config.mjs) that fit under the hosting cap,
 * then the release files on the asset host. A file in both places is offered once, from the site.
 */
export function datasetFiles(): DatasetFile[] {
  const onSite = DATASET_FILES.map((file) => {
    const text = publicDatasetText(fs.readFileSync(path.join(getExportRoot(), file), "utf-8"), assetHref);
    return { file, ...FILE_INFO[file], href: `/dataset/${file}`, bytes: Buffer.byteLength(text) };
  }).filter((item) => item.bytes <= PAGES_MAX_FILE_BYTES);
  const release = publishedRelease();
  const hosted = (release?.files ?? [])
    .filter((item) => !onSite.some((site) => site.file === item.name))
    .map((item) => ({
      file: item.name,
      label: FILE_INFO[item.name]?.label ?? item.name,
      saveAs: `model-menagerie-${release!.version}-${item.name}`,
      format: item.format ?? FILE_INFO[item.name]?.format ?? "application/octet-stream",
      href: assetHref(item.path ?? `dataset/${release!.version}/${item.name}`),
      bytes: item.bytes,
      primary: false,
    }));
  // The summary goes last: it describes the release rather than holding records.
  const ordered = [...onSite.filter((item) => item.file !== "export_manifest.json"), ...hosted, ...onSite.filter((item) => item.file === "export_manifest.json")];
  return ordered;
}

/** An absolute address for a file from `datasetFiles`. */
function absolute(href: string, siteUrl: string): string {
  return new URL(href, siteUrl).href;
}

/**
 * The Dataset page's Python example, built from the same files as its buttons: the CSV, and the full
 * records when a JSON Lines file is on offer; null when there is no CSV to load.
 */
export function pythonExample(files: DatasetFile[], siteUrl: string): string | null {
  const csv = files.find((item) => item.file === "catalog.csv");
  if (!csv) return null;
  const lines = [
    "import pandas as pd",
    "",
    `models = pd.read_csv("${absolute(csv.href, siteUrl)}")`,
    'print(models[["display_name", "category", "n_ops"]].head())',
  ];
  const jsonl = files.find((item) => item.file === "models.jsonl") ?? files.find((item) => item.file === "models.jsonl.gz");
  if (jsonl) {
    const gzip = jsonl.file.endsWith(".gz") ? ', compression="gzip"' : "";
    lines.push("", "# Each model's full record, with diagram file addresses and checksums:", `records = pd.read_json("${absolute(jsonl.href, siteUrl)}", lines=True${gzip})`);
  }
  return lines.join("\n");
}

/** Plain meaning of every column of `catalog.csv`; `models.jsonl` has the same fields plus those below. */
const CSV_COLUMNS: Record<string, string> = {
  stable_id: "Permanent identifier for the model. It stays the same between releases.",
  slug: "Short name used in the model's web address: modelmenagerie.ai/models/<slug>/.",
  display_name: "Model name.",
  category: "Task category, such as vision_classification-backbone. The site shows it as a plain label.",
  family_normalized: "Architecture family the model belongs to.",
  domain: "Broad field of the model, such as vision/segmentation, when recorded.",
  zoo: "Collection the model was found in, such as timm.",
  source_zoo: "Library or repository the model's code comes from, when recorded.",
  source_license: "License of the model's source code as an SPDX identifier, when recorded.",
  source_url: "Address of the model's source code, when recorded.",
  paper_url: "Address of the paper that introduced the model, when recorded.",
  year: "Year the model was published, when recorded.",
  year_source: "Where the year comes from: citation (the paper's citation record) or june_era (an earlier estimate).",
  era: "Period the model belongs to as first cataloged: a year, a range of years, or blank.",
  backend: "Framework the traced code is written in, such as torch.",
  validation_state: "passed when the trace passed its replay check (for PyTorch, a match with the model's own forward pass); structure_only when only the graph's structure was captured.",
  capture: "full when the whole forward pass was captured; structure_only when only the graph's structure was.",
  torchlens_version: "TorchLens version that made the trace, when recorded.",
  render_status: "Whether a diagram was drawn. Every row in this release has one.",
  forward_validated:
    "true when the trace was replayed operation by operation and passed its check.",
  input_tier: "The input the model was traced on: real (an example input from its source), wrapper (generated data of the right shape) or blank when not recorded.",
  quarantine: "true for rows held back from the listing.",
  n_ops: "Number of operations TorchLens recorded in one forward pass.",
  graph_shape_hash: "Fingerprint of the graph's structure. Models with the same value have the same graph shape.",
  param_count: "Number of parameters, when recorded.",
  is_recurrent: "Whether the forward pass runs a module more than once, when recorded.",
  default_variant: "Which drawing of the graph the site shows first, for example unrolled-none.",
  monster_graph: "true for graphs too large to draw on the page; their files are still listed.",
  svg_url: "Address of the diagram (SVG).",
  svg_sha256: "SHA-256 checksum of the diagram file.",
  svg_bytes: "Size of the diagram file in bytes.",
  thumb_url: "Address of the small picture of the whole graph (WebP).",
  thumb_sha256: "SHA-256 checksum of the small picture.",
  thumb_bytes: "Size of the small picture in bytes.",
};

/** Fields only the JSON Lines file has. */
const JSONL_ONLY: Record<string, string> = {
  variants:
    "Every drawing of the graph: unrolled or rolled, with all, grouped or collapsed layers. Each gives its SVG, small picture and PDF with address, checksum and size.",
  source_url: "Address of the model's source code, when recorded.",
  paper_url: "Address of the paper that introduced the model, when recorded.",
  tlspec: "The saved TorchLens trace, when published.",
  recipe_text: "The code that builds the model and its input for tracing, when published.",
  torchlens_version: "TorchLens version that made the trace, when recorded.",
  renderer_version: "Version of the drawing step, when recorded.",
};

export interface ColumnInfo {
  name: string;
  description: string;
  /** Only in models.jsonl. */
  jsonlOnly: boolean;
}

/** The dictionary, in the CSV's own column order. Throws when the CSV has a column not described here. */
export function datasetColumns(): ColumnInfo[] {
  const csv = fs.readFileSync(path.join(getExportRoot(), "catalog.csv"), "utf-8");
  const header = csv.slice(0, csv.indexOf("\n")).trim().split(",");
  const missing = header.filter((name) => !(name in CSV_COLUMNS));
  if (missing.length > 0) {
    throw new Error(`catalog.csv has columns with no description in src/lib/dataset.ts: ${missing.join(", ")}`);
  }
  return [
    ...header.map((name) => ({ name, description: CSV_COLUMNS[name], jsonlOnly: false })),
    ...Object.entries(JSONL_ONLY)
      .filter(([name]) => !header.includes(name))
      .map(([name, description]) => ({ name, description, jsonlOnly: true })),
  ];
}

/**
 * When the export was built, as an ISO date: the manifest's `as_of` once the export records it, else the
 * build date of the June 2026 export the live site runs on (docs/site.md, "Restored export").
 */
export function exportDate(): string | null {
  const manifest = getManifest() as unknown as { as_of?: string; export_version?: string };
  if (manifest.as_of) {
    return manifest.as_of.slice(0, 10);
  }
  return manifest.export_version === "phase0-preview" ? "2026-06-23" : null;
}

/** "23 June 2026" for an ISO date. */
export function formatDate(iso: string): string {
  return new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" }).format(new Date(`${iso}T00:00:00Z`));
}
