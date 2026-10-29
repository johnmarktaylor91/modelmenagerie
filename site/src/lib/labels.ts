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
  history: "Assorted",
};

/** Catch-all categories listed after the named ones, whatever their size. */
const TRAILING_CATEGORIES = new Set(["exotic_other", "history"]);

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

/** Turn an unmapped slug into words: "foo_bar-baz" becomes "Foo bar baz". */
function humanizeSlug(slug: string): string {
  const words = slug.replace(/[_-]+/g, " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : slug;
}

/** Return the plain-word label for a category slug. */
export function categoryLabel(slug: string): string {
  return CATEGORY_LABELS[slug] ?? humanizeSlug(slug);
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
        Number(TRAILING_CATEGORIES.has(left.value)) - Number(TRAILING_CATEGORIES.has(right.value)) ||
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
