/**
 * What the model page says about a record: the facts a reader wants first, the files that
 * exist, and the details fold. Anything missing is left out, never printed as unknown.
 */
import { formatBytes, type ModelRecord, type ModelVariant } from "@/lib/catalog";
import { categoryLabel, formatCompact, formatNumber, frameworkLabel, isPresent } from "@/lib/labels";

export interface Fact {
  label: string;
  value: string;
  href?: string;
  title?: string;
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

/** Plain sentence on how the trace was checked, following the TorchLens docs at the pinned version. */
export function validationSentence(model: ModelRecord): string | null {
  const state = model.validation_state ?? null;
  const preview = isPresent(model.backend) && model.backend !== "torch";
  if (model.capture === "structure_only" || state === "structure_only") {
    return "Structure only: shapes and data types are hypotheses, not measurements.";
  }
  if (state === "passed") {
    return preview
      ? "Preview check passed: each captured op was replayed from its inputs. The model was not rerun against a ground-truth output."
      : "Passed: the capture was replayed op by op against the original forward output.";
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
    return isPresent(model.backend) && model.backend !== "torch" ? "Preview check passed" : "Checked";
  }
  return null;
}

function yearAndVenue(model: ModelRecord): string | null {
  const parts = [model.year ? String(model.year) : null, isPresent(model.venue) ? model.venue : null].filter(Boolean);
  return parts.length ? parts.join(", ") : null;
}

/** The facts block, in reading order. `family` links to the family's page or filtered listing. */
export function modelFacts(model: ModelRecord, family: { name: string; href: string } | null): Fact[] {
  const facts: Array<Fact | null> = [
    isPresent(model.category) ? { label: "Task", value: categoryLabel(model.category), href: `/models/?category=${encodeURIComponent(model.category)}` } : null,
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
    yearAndVenue(model) ? { label: "Published", value: yearAndVenue(model) as string } : null,
    isPresent(model.paper_url) ? { label: "Paper", value: isPresent(model.paper_title) ? model.paper_title : "Read the paper", href: model.paper_url } : null,
    model.authors?.length ? { label: "Authors", value: model.authors.length > 4 ? `${model.authors.slice(0, 3).join(", ")} and others` : model.authors.join(", ") } : null,
    isPresent(model.source_zoo) || isPresent(model.source_url)
      ? { label: "Source", value: isPresent(model.source_zoo) ? model.source_zoo : "Source code", href: isPresent(model.source_url) ? model.source_url : undefined }
      : null,
    isPresent(model.source_license) ? { label: "Licence", value: model.source_license } : null,
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
    isPresent(model.validation_kind) ? { label: "Check", value: model.validation_kind } : null,
    model.citations ? { label: "Citations", value: `${formatNumber(model.citations.count)} (as of ${model.citations.as_of})` } : null,
    model.fidelity === "minor" && isPresent(model.fidelity_note) ? { label: "Differs from the paper", value: model.fidelity_note } : null,
  ];
  return details.filter((detail): detail is Fact => detail !== null);
}
