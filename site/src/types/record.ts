/**
 * The site's catalog record: the June 2026 preview export's fields, plus every field the
 * publish stage plans to emit (`site-export/1`, menagerie/pipeline/publish.py) and the site
 * plan's additions. Fields the preview export lacks are optional, so pages compile and render
 * against today's data, an all-empty record and a full one alike. Missing values are hidden on
 * the page, never printed as "unknown".
 *
 * Browser-safe: types only.
 */

export interface AssetRef {
  url: string;
  sha256: string;
  bytes: number;
}

/** A raster asset whose pixel size is known, so the page can reserve its box before it loads. */
export interface SizedAssetRef extends AssetRef {
  width?: number;
  height?: number;
}

export interface TlspecRef extends AssetRef {
  /** True when the spec replays as a runnable model, not only a structure description. */
  runnable?: boolean;
}

export type VisMode = "unrolled" | "rolled";
export type Collapse = "none" | "auto" | "max";

export interface ModelVariant {
  key: string;
  vis_mode: VisMode;
  collapse: Collapse;
  /** Eval or train mode trace; absent in the preview export (eval). */
  mode?: "eval" | "train";
  label: string;
  svg: AssetRef;
  thumb: SizedAssetRef;
  pdf: AssetRef | null;
  monster: boolean;
  /** Top crop of the rendered graph, 800 by 600 WebP (plan section 4, crawler ask 1). */
  thumb_crop?: SizedAssetRef | null;
  /** The SVG's own size, from its attributes at render time (crawler ask 3). */
  svg_width_px?: number | null;
  svg_height_px?: number | null;
}

export type ValidationState = "passed" | "failed" | "unverified" | "unavailable" | "structure_only";
export type Rung = "library" | "vendor" | "port" | "native" | "reimplement" | "classic";
export type Fidelity = "faithful" | "minor" | "fabrication" | "cannot_judge";
export type ListedStatus = "not_yet_traced" | "source_unavailable" | "paper_only";

export interface ModelRecord {
  // Identity and grouping (preview export).
  stable_id: string;
  slug: string;
  display_name: string;
  category: string;
  family_normalized: string;
  domain: string;
  zoo: string;
  source_zoo: string;
  /** SPDX name, or "unknown". */
  source_license: string;
  source_url: string | null;
  paper_url: string | null;
  render_status: "rendered";
  forward_validated: boolean;
  input_tier: "real" | "wrapper" | "none";
  quarantine: boolean;
  n_ops: number | null;
  graph_shape_hash: string;
  param_count: number | null;
  is_recurrent: boolean | null;
  default_variant: string;
  variants: ModelVariant[];
  tlspec: TlspecRef | null;
  recipe_text: string;
  torchlens_version: string;
  renderer_version: string;
  monster_graph: boolean;

  // Validation and capture (publish.py).
  validation_state?: ValidationState | null;
  /** For example torch-replay, tf-preview. */
  validation_kind?: string | null;
  capture?: "full" | "structure_only";
  /** torch, tf, jax, mlx, paddle or tinygrad. */
  backend?: string | null;
  rung?: Rung | null;
  fidelity?: Fidelity | null;
  /** Shown beside a "minor" fidelity verdict. */
  fidelity_note?: string | null;
  hashes?: { skeleton?: string; architecture?: string; model?: string } | null;
  params_source?: "counted" | "geometry" | null;
  trace_stats?: Record<string, number | string | null> | null;

  // Other exports of the trace (publish.py).
  interactive?: AssetRef | null;
  netron?: AssetRef | null;
  model_explorer?: AssetRef | null;

  // Paper and provenance (publish.py).
  paper_title?: string | null;
  year?: number | null;
  year_source?: string | null;
  era?: string | null;
  authors?: string[];
  venue?: string | null;
  bibtex?: string | null;
  citations?: { count: number; as_of: string } | null;
  labs?: string[];
  lab?: string | null;
  field?: string | null;
  output_type?: string | null;

  // Describe pass, withheld when ungrounded (publish.py).
  tagline?: string | null;
  summary_markdown?: string | null;
  key_contribution?: string | null;
  keywords?: string[];

  // Site plan additions (plan section 4, crawler asks 2, 4, 5 and 8).
  share_image?: SizedAssetRef | null;
  /** True when the default variant's SVG is this record's own trace, not a shared stand-in. */
  own_diagram?: boolean | null;
  diagram_shared_with?: number | null;
  listed_status?: ListedStatus | null;
  listed_reason?: string | null;
  family_summary?: string | null;
  aliases?: string[];
}

/** One family of the shown set (`families.jsonl`, plan section 4); optional until the export has it. */
export interface FamilyRecord {
  family: string;
  display_name?: string;
  member_slugs: string[];
  summary?: string | null;
  paper_url?: string | null;
  paper_title?: string | null;
}

/** Public counts. The preview export has the first block; `site-export/1` the dated second block. */
export interface Funnel {
  cataloged: number;
  build_verified: number;
  forward_validated_real: number;
  forward_validated_wrapper: number;
  rendered: number;
  quarantined: number;
  deferred: number;
  predicate_note: string;
  as_of?: string;
  roster_names?: number;
  validated_records?: number;
  structure_only_records?: number;
  distinct_architectures?: number;
  distinct_basis?: string;
}

export interface Manifest {
  schema_version: string;
  export_version: string;
  as_of?: string;
  counts: Record<string, number>;
  torchlens_version: string;
  renderer_version: string;
}
