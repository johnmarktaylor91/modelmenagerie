/**
 * What the model page says about a record: the facts a reader wants first, the files that
 * exist, and the details fold. Anything missing is left out, never printed as unknown.
 */
import { formatBytes, type ModelRecord, type ModelVariant } from "@/lib/catalog";
import { formatDate } from "@/lib/dataset";
import { formatCompact, formatNumber, frameworkLabel, isPresent, licenceLabel, sourceUrlLabel, taskLabel, venueLabel, webUrl } from "@/lib/labels";

export interface Fact {
  label: string;
  value: string;
  href?: string;
  title?: string;
  /** The value is quoted from the paper (its title), not the site's own words. */
  quoted?: boolean;
}

export interface FileLink {
  label: string;
  note?: string;
  size: string;
  href: string;
  /** Keys the viewer script uses to repoint SVG and PDF links when the variant changes. */
  asset?: "svg" | "pdf";
  /** File name for the SVG's Save button. */
  saveAs?: string;
}

/** A catalog source name that is a package at a version ("timm 1.0.30", "torchvision 0.21.0"). */
const PACKAGE_AT_VERSION = /^[A-Za-z0-9_.-]+ v?\d[\w.+-]*$/;

/**
 * Where the code comes from. A package keeps its name and version ("timm 1.0.30", without the
 * crawler's parenthetical notes); any other source name is the catalog's own bookkeeping and is
 * never printed: the label is said from the web address instead ("GitHub: owner/repo", "PyPI
 * package", "Hugging Face: org/name"). Null when neither gives a label.
 */
export function sourceLabel(zoo: string | null | undefined, url?: string | null): string | null {
  const name = isPresent(zoo) ? zoo.replace(/\s*\(.*$/, "").trim() : "";
  if (PACKAGE_AT_VERSION.test(name)) {
    return name;
  }
  return sourceUrlLabel(url);
}

/** Plain sentence on how the trace was checked, following the TorchLens docs at the pinned version. */
export function validationSentence(model: ModelRecord): string | null {
  const state = model.validation_state ?? null;
  const preview = isPresent(model.backend) && model.backend !== "torch";
  if (model.capture === "structure_only" || state === "structure_only") {
    return "Structure only: shapes and data types are hypotheses, not measurements.";
  }
  if (state === "passed") {
    return preview
      ? "Replay check passed: each recorded operation was replayed from its inputs. The whole model was not rerun against a known output."
      : "Passed: the trace was replayed operation by operation and matched the model's own output for this input.";
  }
  if (state === "failed") {
    return "The replay check did not pass.";
  }
  if (state === "unverified" || state === "unavailable") {
    return "Not checked yet.";
  }
  if (model.forward_validated) {
    return model.input_tier === "real" ? "Yes, on a real input." : "Yes, on a generated input.";
  }
  return null;
}

/** The short tag beside the name, or null when there is nothing to say. */
export function validationTag(model: ModelRecord): string | null {
  if (model.capture === "structure_only" || model.validation_state === "structure_only") {
    return "Structure only";
  }
  if (model.validation_state === "passed" || (model.validation_state == null && model.forward_validated)) {
    return isPresent(model.backend) && model.backend !== "torch" ? "Replay checked" : "Checked";
  }
  return null;
}

/**
 * "CVPR 2022", "2017, Lecture Notes in Computer Science", "ICLR 2024" or the year alone. The venue
 * is deduplicated and shortened (`venueLabel`); a venue that names a year is shown alone.
 * When the year is the paper's arXiv year (it matches the year in the arXiv id) and the venue names
 * none, the two are not joined as if the year were the venue's: ResNet is on arXiv in 2015 and at
 * CVPR in 2016, so it reads "CVPR (arXiv 2015)", not "CVPR 2015".
 */
export function yearAndVenue(model: Pick<ModelRecord, "year" | "venue"> & Partial<Pick<ModelRecord, "paper_url" | "bibtex">>): string | null {
  const year = model.year ? String(model.year) : "";
  const venue = isPresent(model.venue) ? venueLabel(model.venue) : { label: "", short: false };
  if (!venue.label) {
    return year || null;
  }
  // A venue that names its own year ("CVPR 2025") dates the paper; an earlier preprint year is not added.
  if (!year || /\b(19|20)\d\d\b/.test(venue.label)) {
    return venue.label;
  }
  if (!/arxiv/i.test(venue.label) && arxivYear(model) === year) {
    return `${venue.label} (arXiv ${year})`;
  }
  return venue.short ? `${venue.label} ${year}` : `${year}, ${venue.label}`;
}

/** The year in the paper's new-style arXiv id ("1512.03385" is 2015), from its address or BibTeX; else null. */
function arxivYear(model: Partial<Pick<ModelRecord, "paper_url" | "bibtex">>): string | null {
  const fromUrl = typeof model.paper_url === "string" ? /arxiv\.org\/(?:abs|pdf)\/(\d{2})(\d{2})\.\d{4,5}/i.exec(model.paper_url) : null;
  const fromBibtex = typeof model.bibtex === "string" ? /eprint\s*=\s*\{(\d{2})(\d{2})\.\d{4,5}/i.exec(model.bibtex) : null;
  const match = fromUrl ?? fromBibtex;
  return match ? `20${match[1]}` : null;
}

/** The paper row: its title (quoted) linking to the paper, when the paper has a web address. */
function paperFact(model: ModelRecord): Fact | null {
  const href = webUrl(model.paper_url);
  if (!href) {
    return null;
  }
  return isPresent(model.paper_title) ? { label: "Paper", value: model.paper_title, href, quoted: true } : { label: "Paper", value: "Read the paper", href };
}

/** "1,234 (as of 2 November 2026)" for the paper's citation count, or null when it has none. */
function citationFact(model: ModelRecord): Fact | null {
  const citations = model.citations;
  if (!citations || !Number.isFinite(citations.count) || citations.count < 0) {
    return null;
  }
  const asOf = /^\d{4}-\d{2}-\d{2}/.test(citations.as_of ?? "") ? ` (as of ${formatDate(citations.as_of.slice(0, 10))})` : "";
  return { label: "Cited by", value: `${formatNumber(citations.count)}${asOf}` };
}

/** The Source row: the package or the code's web address, linked when it has one; hidden when neither says anything. */
function sourceFact(model: ModelRecord): Fact | null {
  const label = sourceLabel(model.source_zoo, model.source_url);
  return label ? { label: "Source", value: label, href: webUrl(model.source_url) ?? undefined } : null;
}

/** The facts block, in reading order. `family` links to the family's page or filtered listing. */
export function modelFacts(model: ModelRecord, family: { name: string; href: string } | null): Fact[] {
  const facts: Array<Fact | null> = [
    taskLabel(model.category) ? { label: "Task", value: taskLabel(model.category), href: `/models/?category=${encodeURIComponent(model.category)}` } : null,
    family ? { label: "Family", value: family.name, href: family.href } : null,
    model.param_count !== null
      ? {
          label: "Parameters",
          value: formatCompact(model.param_count),
          title: `${formatNumber(model.param_count)}${model.params_source === "geometry" ? ", estimated from layer shapes" : ""}`,
        }
      : null,
    model.n_ops !== null ? { label: "Operations", value: formatNumber(model.n_ops) } : null,
    isPresent(model.backend) ? { label: "Framework", value: frameworkLabel(model.backend) } : null,
    model.is_recurrent !== null ? { label: "Recurrent", value: model.is_recurrent ? "Yes" : "No" } : null,
    yearAndVenue(model) ? { label: "Published", value: yearAndVenue(model) as string, quoted: isPresent(model.venue) } : null,
    paperFact(model),
    citationFact(model),
    model.authors?.length ? { label: "Authors", value: model.authors.length > 4 ? `${model.authors.slice(0, 3).join(", ")} and others` : model.authors.join(", "), quoted: true } : null,
    sourceFact(model),
    isPresent(model.source_name) ? { label: "Name in source", value: model.source_name } : null,
    isPresent(model.source_license) ? { label: "License", value: licenceLabel(model.source_license) } : null,
    validationSentence(model) ? { label: "Checked", value: validationSentence(model) as string } : null,
  ];
  return facts.filter((fact): fact is Fact => fact !== null);
}

/** Files that exist for the record (and its first variant), each with a readable name and size. */
export function modelFiles(model: ModelRecord, variant: ModelVariant | null): FileLink[] {
  const files: Array<FileLink | null> = [
    variant ? { label: "Diagram, SVG", size: formatBytes(variant.svg.bytes), href: variant.svg.url, asset: "svg", saveAs: `${model.slug}-${variant.key}.svg` } : null,
    variant?.pdf ? { label: "Diagram, PDF", size: formatBytes(variant.pdf.bytes), href: variant.pdf.url, asset: "pdf" } : null,
    model.interactive ? { label: "Interactive viewer", note: "TorchLens HTML, opens in a new tab", size: formatBytes(model.interactive.bytes), href: model.interactive.url } : null,
    model.tlspec
      ? { label: "TorchLens spec", note: model.tlspec.runnable ? "runnable" : undefined, size: formatBytes(model.tlspec.bytes), href: model.tlspec.url }
      : null,
    model.netron ? { label: "Netron graph", note: "for inspection, not runnable", size: formatBytes(model.netron.bytes), href: model.netron.url } : null,
    model.model_explorer ? { label: "Model Explorer graph", size: formatBytes(model.model_explorer.bytes), href: model.model_explorer.url } : null,
  ];
  return files.filter((file): file is FileLink => file !== null);
}

/**
 * How the trace was checked, in words: the pipeline's kind names (``torch-replay``, and
 * ``preview-replay`` for models TorchLens traces through its TensorFlow, JAX, Paddle or MLX backend)
 * are internal; an unknown kind is shown as it is.
 */
export function checkLabel(kind: string, backend?: string | null): string {
  if (kind === "torch-replay") return "Forward pass replayed in PyTorch";
  if (/replay|preview/.test(kind) && isPresent(backend)) return `Forward pass replayed in ${frameworkLabel(backend)}`;
  return kind;
}

/** The Details fold: identifiers, versions and how the trace was made. */
export function modelDetails(model: ModelRecord, variant: ModelVariant | null): Fact[] {
  const details: Array<Fact | null> = [
    isPresent(model.stable_id) ? { label: "Stable ID", value: model.stable_id } : null,
    isPresent(model.graph_shape_hash) ? { label: "Graph hash", value: model.graph_shape_hash } : null,
    isPresent(model.hashes?.skeleton) ? { label: "Skeleton hash", value: model.hashes?.skeleton as string } : null,
    isPresent(model.hashes?.architecture) ? { label: "Architecture hash", value: model.hashes?.architecture as string } : null,
    variant ? { label: "Diagram SHA-256", value: variant.svg.sha256 } : null,
    isPresent(model.torchlens_version) ? { label: "TorchLens version", value: model.torchlens_version } : null,
    isPresent(model.renderer_version) ? { label: "Renderer", value: model.renderer_version } : null,
    isPresent(model.validation_kind) ? { label: "Check", value: checkLabel(model.validation_kind, model.backend) } : null,
    model.fidelity === "minor" && isPresent(model.fidelity_note) ? { label: "Differs from the paper", value: model.fidelity_note } : null,
  ];
  return details.filter((detail): detail is Fact => detail !== null);
}
