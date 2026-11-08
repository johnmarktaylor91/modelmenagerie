/**
 * Visitor-facing labels and number formatting for browse pages.
 *
 * Browser-safe: no Node imports, so client scripts can share it with build-time pages.
 */

/** Plain-word labels for the catalog's category slugs; URLs and filters keep the slug. */
export const CATEGORY_LABELS: Readonly<Record<string, string>> = {
  "vision_classification-backbone": "Image classification",
  vision_segmentation: "Segmentation",
  "vision_detection-tracking": "Detection and tracking",
  "generative_diffusion-GAN-flow": "Diffusion and generative",
  audio_speech: "Audio and speech",
  graph_geometric: "Graphs and geometry",
  NLP_LLM_text: "Language",
  recsys_tabular: "Tabular and recommender",
  video_action: "Video",
  "time-series_dynamics": "Time series",
  vision_pose: "Pose",
  "3D_geometry_NeRF_point-cloud": "3D and point clouds",
  scientific_molecular_protein_genomics: "Molecules, proteins and genomics",
  multimodal: "Multimodal",
  OCR_document: "Documents and OCR",
  spiking_neuromorphic: "Spiking networks",
  RL_robotics_control: "Reinforcement learning and control",
  "vision_depth-geometry": "Depth estimation",
  scientific_physics_weather_PDE: "Physics and weather",
  "neuro_scientific-signals": "Neuroscience signals",
  "medical-imaging": "Medical imaging",
  exotic_other: "Other",
  history: "Historical",
};

/** Categories listed after the named ones, whatever their size, in this order: the catch-all "Other" comes last. */
const TRAILING_ORDER = ["history", "exotic_other"];
const trailingRank = (value: string) => {
  const index = TRAILING_ORDER.indexOf(value);
  return index === -1 ? 0 : index + 1;
};

/** Size buckets by recorded operation count; values match the catalog's size facet. */
export const SIZE_LABELS: Readonly<Record<string, string>> = {
  small: "Under 50 operations",
  medium: "50 to 249 operations",
  large: "250 to 999 operations",
  monster: "1,000 operations or more",
  unknown: "Operation count not recorded",
};

const SIZE_ORDER = ["small", "medium", "large", "monster", "unknown"];

/** Parameter-count buckets, smallest first. */
export const PARAM_BUCKETS: ReadonlyArray<{ value: string; label: string; below: number }> = [
  { value: "p-under-1m", label: "Under 1M parameters", below: 1e6 },
  { value: "p-1m-10m", label: "1M to 10M", below: 1e7 },
  { value: "p-10m-100m", label: "10M to 100M", below: 1e8 },
  { value: "p-100m-1b", label: "100M to 1B", below: 1e9 },
  { value: "p-1b-plus", label: "1B or more", below: Infinity },
];

/** Publication-year buckets, newest first. */
export const YEAR_BUCKETS: ReadonlyArray<{ value: string; label: string; from: number }> = [
  { value: "y-2024", label: "2024 and later", from: 2024 },
  { value: "y-2020", label: "2020 to 2023", from: 2020 },
  { value: "y-2015", label: "2015 to 2019", from: 2015 },
  { value: "y-2010", label: "2010 to 2014", from: 2010 },
  { value: "y-older", label: "Before 2010", from: -Infinity },
];

/** Framework names for the record's `backend` value. */
export const FRAMEWORK_LABELS: Readonly<Record<string, string>> = {
  torch: "PyTorch",
  tf: "TensorFlow",
  tensorflow: "TensorFlow",
  jax: "JAX",
  mlx: "MLX",
  paddle: "PaddlePaddle",
  tinygrad: "tinygrad",
};

const integerFormat = new Intl.NumberFormat("en");
const compactFormat = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });

/** Format an integer with thousands separators ("14,291"). */
export function formatNumber(value: number): string {
  return integerFormat.format(value);
}

/** Format a large count compactly ("11.7M"). */
export function formatCompact(value: number): string {
  return compactFormat.format(value);
}

/** Return true when a catalog text field holds a real value rather than a placeholder. */
export function isPresent(value: string | null | undefined): value is string {
  if (value === null || value === undefined) {
    return false;
  }
  const trimmed = value.trim().toLowerCase();
  return trimmed !== "" && trimmed !== "unknown" && trimmed !== "pending";
}

/** The value when it is a web address (http or https), else null: repository paths are not links. */
export function webUrl(value: string | null | undefined): string | null {
  const trimmed = (value ?? "").trim();
  return /^https?:\/\/\S+$/i.test(trimmed) ? trimmed : null;
}

/** Turn an unmapped slug into words: "foo_bar-baz" becomes "Foo bar baz". */
function humanizeSlug(slug: string): string {
  const words = slug.replace(/[_-]+/g, " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : slug;
}

/** Return the plain-word label for a category slug. */
export function categoryLabel(slug: string): string {
  return CATEGORY_LABELS[slug] ?? humanizeSlug(slug);
}

/** The catch-all category the describe step leaves unlabelled; it reads "Other" only in the task filter. */
export const UNSORTED_CATEGORY = "exotic_other";

/**
 * The task as a card, list row, heading or At-a-glance row prints it: "" for the catch-all
 * category, which says nothing about the model (the task filter still lists it as "Other").
 */
export function taskLabel(slug: string | null | undefined): string {
  if (!slug || slug === UNSORTED_CATEGORY) {
    return "";
  }
  return categoryLabel(slug);
}

/** Return the plain-word label for a size bucket. */
export function sizeLabel(bucket: string): string {
  return SIZE_LABELS[bucket] ?? humanizeSlug(bucket);
}

/** Return the parameter bucket for a count, or "" when it is unknown. */
export function paramBucket(count: number | null | undefined): string {
  if (count === null || count === undefined || count < 0) {
    return "";
  }
  return PARAM_BUCKETS.find((bucket) => count < bucket.below)?.value ?? "";
}

/** Return the year bucket for a publication year, or "" when it is unknown. */
export function yearBucket(year: number | null | undefined): string {
  if (year === null || year === undefined || !Number.isFinite(year)) {
    return "";
  }
  return YEAR_BUCKETS.find((bucket) => year >= bucket.from)?.value ?? "";
}

/** Return the plain name of a framework. */
export function frameworkLabel(backend: string): string {
  return FRAMEWORK_LABELS[backend.toLowerCase()] ?? backend;
}

/** Return the size bucket for an operation count; thresholds match the catalog's size facet. */
export function sizeBucket(nOps: number | null): string {
  if (nOps === null) {
    return "unknown";
  }
  if (nOps < 50) {
    return "small";
  }
  if (nOps < 250) {
    return "medium";
  }
  if (nOps < 1000) {
    return "large";
  }
  return "monster";
}

export interface FacetValue {
  value: string;
  count: number;
}

export interface LabelledFacet {
  value: string;
  label: string;
  count: number;
}

/** Categories with labels, largest first, catch-all groups last. */
export function labelCategoryFacets(facets: readonly FacetValue[]): LabelledFacet[] {
  return facets
    .map((facet) => ({ ...facet, label: categoryLabel(facet.value) }))
    .sort(
      (left, right) =>
        trailingRank(left.value) - trailingRank(right.value) ||
        right.count - left.count ||
        left.label.localeCompare(right.label),
    );
}

/** Size buckets with labels, smallest first. */
export function labelSizeFacets(facets: readonly FacetValue[]): LabelledFacet[] {
  const rank = (value: string) => (SIZE_ORDER.includes(value) ? SIZE_ORDER.indexOf(value) : SIZE_ORDER.length);
  return facets
    .map((facet) => ({ ...facet, label: sizeLabel(facet.value) }))
    .sort((left, right) => rank(left.value) - rank(right.value));
}

/** A facet group is worth showing only when it can actually narrow the results. */
export function isUsefulFacet(facets: readonly FacetValue[]): boolean {
  return facets.filter((facet) => facet.count > 0).length >= 2;
}

/**
 * The plain name of a licence identifier. arXiv's distribution licence covers the paper, not
 * code, so it says so; other SPDX identifiers are shown as they are.
 */
export function licenceLabel(spdx: string): string {
  return /^LicenseRef-arXiv/i.test(spdx) ? "arXiv non-exclusive license (paper)" : spdx;
}

/**
 * Short forms for the commonest long venue names, matched on the cleaned venue string. The
 * label replaces the whole string; a venue that names a repository rather than a publication
 * maps to "" and is not printed.
 */
const VENUE_SHORT_FORMS: ReadonlyArray<[RegExp, string]> = [
  [/Computer Vision and Pattern Recognition Workshops|\(CVPRW\)/i, "CVPR Workshops"],
  [/Conference on Computer Vision and Pattern Recognition|\(CVPR\)/i, "CVPR"],
  [/International Conference on Computer Vision Workshops|\(ICCVW\)/i, "ICCV Workshops"],
  [/International Conference on Computer Vision|\(ICCV\)/i, "ICCV"],
  [/European Conference on Computer Vision.*Workshops/i, "ECCV Workshops"],
  [/European Conference on Computer Vision|\(ECCV\)/i, "ECCV"],
  [/^(Advances in )?neural information processing systems$/i, "NeurIPS"],
  [/North American Chapter of the Association for Computational Linguistics/i, "NAACL"],
  [/Annual Meeting of the Association for Computational Linguistics/i, "ACL"],
  [/Empirical Methods in Natural Language Processing/i, "EMNLP"],
  [/AAAI Conference on Artificial Intelligence/i, "AAAI"],
  [/^(Proceedings of the )?International Conference on Machine Learning$/i, "ICML"],
  [/^International Conference on Learning Representations$/i, "ICLR"],
  [/Conference on Artificial Intelligence and Statistics/i, "AISTATS"],
  [/International Conference on Acoustics,? Speech,? and Signal Processing/i, "ICASSP"],
  [/Medical Image Computing and Computer[- ]Assisted Intervention/i, "MICCAI"],
  [/Transactions on Pattern Analysis and Machine Intelligence/i, "IEEE TPAMI"],
  [/^International Journal of Computer Vision$/i, "IJCV"],
  [/Winter Conference on Applications of Computer Vision/i, "WACV"],
  [/^British Machine Vision Conference$/i, "BMVC"],
  [/^arXiv \(Cornell University\)$/i, "arXiv"],
  [/^Lecture notes in computer science$/i, "Lecture Notes in Computer Science"],
  // Repositories that hold a copy of the paper, not where it was published.
  [/^(CaltechAUTHORS|Zenodo)\b/i, ""],
];

/**
 * The venue with repeated segments dropped: catalog venues sometimes join the same name twice
 * ("X/Proceedings - X/Proceedings"). Segments are split on " - " and "/"; a bare "Proceedings"
 * and any segment already seen (ignoring case and a leading "Proceedings") are dropped.
 */
export function dedupeVenue(venue: string): string {
  const pieces = venue.trim().split(/(\s+-\s+|\/)/);
  const seen = new Set<string>();
  let out = "";
  for (let index = 0; index < pieces.length; index += 2) {
    const segment = pieces[index].trim();
    const key = segment.toLowerCase().replace(/^proceedings(\s+of(\s+the)?)?\s*/, "");
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    out += out ? `${pieces[index - 1] ?? " - "}${segment}` : segment;
  }
  return out;
}

/**
 * The venue as the Published row prints it: deduplicated, then shortened when it is one of the
 * commonest long names. `short` is true when the label is a short form (the caller may append
 * the year to it, "CVPR 2022"). "" means the venue is not worth printing.
 */
export function venueLabel(venue: string): { label: string; short: boolean } {
  const clean = dedupeVenue(venue);
  const withoutYear = clean.replace(/^(19|20)\d\d\s+/, "");
  for (const [pattern, label] of VENUE_SHORT_FORMS) {
    if (pattern.test(withoutYear)) {
      return { label, short: label !== "" };
    }
  }
  return { label: clean, short: false };
}

/**
 * Where the code comes from, said from its web address when the catalog's own source name is
 * not a package: "GitHub: owner/repo", "PyPI package", "Hugging Face: org/name", or the host.
 * Null when the address is not a web address.
 */
export function sourceUrlLabel(url: string | null | undefined): string | null {
  const href = webUrl(url);
  if (!href) {
    return null;
  }
  let parsed: URL;
  try {
    parsed = new URL(href);
  } catch {
    return null;
  }
  const host = parsed.hostname.replace(/^www\./, "").toLowerCase();
  const parts = parsed.pathname.split("/").filter(Boolean);
  if (host === "github.com" && parts.length >= 2) {
    return `GitHub: ${parts[0]}/${parts[1].replace(/\.git$/, "")}`;
  }
  if (host === "github.com") {
    return "GitHub";
  }
  if (host === "pypi.org") {
    return "PyPI package";
  }
  if (host === "huggingface.co") {
    const repo = parts[0] === "spaces" || parts[0] === "datasets" ? parts.slice(1, 3) : parts.slice(0, 2);
    return repo.length ? `Hugging Face: ${repo.join("/")}` : "Hugging Face";
  }
  return host || null;
}
