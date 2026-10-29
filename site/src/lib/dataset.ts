/**
 * What the Dataset page offers: the downloadable files with their sizes, and a dictionary of every
 * column, read from the files themselves so the page cannot drift from what people download. The
 * build fails when a column has no description here.
 */
import fs from "node:fs";
import path from "node:path";

import { getExportRoot, getManifest } from "@/lib/catalog";
import { DATASET_FILES, PAGES_MAX_FILE_BYTES, publicDatasetText } from "@/lib/dataset-files.mjs";

const publicR2Base = (import.meta.env.PUBLIC_R2_BASE as string | undefined)?.replace(/\/+$/, "") ?? "";

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
  "export_manifest.json": { label: "Release summary (JSON)", saveAs: "model-menagerie-release.json", format: "application/json", primary: false },
};

/** A numbered release published as files under `dataset/<version>/` on the asset host (plan section 4). */
interface PublishedRelease {
  version: string;
  files: { name: string; bytes: number; format?: string }[];
}

/** The release block of a site-export/1 manifest, when the export carries one. */
function publishedRelease(): PublishedRelease | null {
  const release = (getManifest() as unknown as { dataset_release?: PublishedRelease }).dataset_release;
  return release && Array.isArray(release.files) && release.files.length > 0 && publicR2Base ? release : null;
}

/**
 * The files to offer: a published release's files on the asset host when the export names one,
 * otherwise the catalog files the build copies to /dataset/ (see astro.config.mjs).
 */
export function datasetFiles(): DatasetFile[] {
  const release = publishedRelease();
  if (release) {
    return release.files.map((item, index) => ({
      file: item.name,
      label: item.name,
      saveAs: `model-menagerie-${release.version}-${item.name}`,
      format: item.format ?? "application/octet-stream",
      href: `${publicR2Base}/dataset/${release.version}/${item.name}`,
      bytes: item.bytes,
      primary: index === 0,
    }));
  }
  return DATASET_FILES.map((file) => {
    const text = publicDatasetText(fs.readFileSync(path.join(getExportRoot(), file), "utf-8"), publicR2Base);
    return { file, ...FILE_INFO[file], href: `/dataset/${file}`, bytes: Buffer.byteLength(text) };
  }).filter((item) => item.bytes <= PAGES_MAX_FILE_BYTES);
}

/** Plain meaning of every column of `catalog.csv`; `models.jsonl` has the same fields plus those below. */
const CSV_COLUMNS: Record<string, string> = {
  stable_id: "Permanent identifier for the model. It stays the same between releases.",
  slug: "Short name used in the model's web address: modelmenagerie.ai/models/<slug>/.",
  display_name: "Model name.",
  category: "Task category, such as vision_classification-backbone. The site shows it as a plain label.",
  family_normalized: "Architecture family the model belongs to.",
  domain: "Broad field of the model. Not filled in yet.",
  zoo: "Collection the model was found in. Not filled in yet.",
  source_zoo: "Library or repository the model's code comes from, when recorded.",
  source_license: "Licence of the model's source code as an SPDX identifier, when recorded.",
  render_status: "Whether a diagram was drawn. Every row in this release has one.",
  forward_validated:
    "true when the capture was replayed operation by operation and matched the model's own forward pass.",
  input_tier: "The input the model was traced on: real (a genuine example), wrapper (generated data of the right shape) or none recorded.",
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
    ...Object.entries(JSONL_ONLY).map(([name, description]) => ({ name, description, jsonlOnly: true })),
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
