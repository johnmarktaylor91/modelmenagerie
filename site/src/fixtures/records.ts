/**
 * Fixture records that prove every page type renders with an all-empty record and with a full
 * one (every planned field filled), plus the structure-only and preview-backend cases. Pages
 * under /fixtures/ are built only when MENAGERIE_FIXTURE_PAGES=1, never in the deployed site.
 *
 * The full records borrow resnet18's real diagram assets so pictures load; every other value
 * is made up for the fixture and labelled as such in the name.
 */
import { getAllModels, getModelBySlug, type ModelRecord, type ModelVariant } from "@/lib/catalog";
import { groupFamilies, type Family } from "@/lib/families";

export const FIXTURE_CASES = ["empty", "full", "structure-only", "preview-backend"] as const;
export type FixtureCase = (typeof FIXTURE_CASES)[number];

/** True when this build should include the fixture pages. */
export function fixturePagesEnabled(): boolean {
  return process.env.MENAGERIE_FIXTURE_PAGES === "1";
}

const FIXTURE_FAMILY = "Fixture ResNet";

/** Every required field present and empty; every optional field absent. */
function emptyRecord(): ModelRecord {
  return {
    stable_id: "",
    slug: "fixture-empty",
    display_name: "Fixture: empty record",
    category: "",
    family_normalized: "",
    domain: "",
    zoo: "",
    source_zoo: "",
    source_license: "",
    source_url: null,
    paper_url: null,
    render_status: "rendered",
    forward_validated: false,
    input_tier: "none",
    quarantine: false,
    n_ops: null,
    graph_shape_hash: "",
    param_count: null,
    is_recurrent: null,
    default_variant: "",
    variants: [],
    tlspec: null,
    recipe_text: "",
    torchlens_version: "",
    renderer_version: "",
    monster_graph: false,
  };
}

/** A real variant re-keyed, with the planned crop and SVG dimensions filled in. */
function plannedVariant(source: ModelVariant, key: string, visMode: ModelVariant["vis_mode"], collapse: ModelVariant["collapse"]): ModelVariant {
  return {
    ...source,
    key,
    vis_mode: visMode,
    collapse,
    mode: "eval",
    label: key,
    pdf: { ...source.svg, bytes: 120_000 },
    thumb: { ...source.thumb, width: 400, height: 1210 },
    thumb_crop: { ...source.thumb, width: 800, height: 600 },
    svg_width_px: 412,
    svg_height_px: 1250,
  };
}

function fullRecord(): ModelRecord {
  const base = getModelBySlug("resnet18");
  const variant = base.variants[0];
  const asset = variant.svg;
  return {
    ...base,
    stable_id: "fixture-full",
    slug: "fixture-full",
    display_name: "Fixture: ResNet-18, every field filled",
    category: "vision_classification-backbone",
    family_normalized: FIXTURE_FAMILY,
    domain: "vision/image",
    zoo: "torchvision",
    source_zoo: "torchvision 0.21.0",
    source_license: "BSD-3-Clause",
    source_url: "https://github.com/pytorch/vision",
    paper_url: "https://arxiv.org/abs/1512.03385",
    forward_validated: true,
    input_tier: "real",
    param_count: 11_689_512,
    params_source: "counted",
    is_recurrent: false,
    default_variant: "unrolled-none",
    variants: [
      plannedVariant(variant, "unrolled-none", "unrolled", "none"),
      plannedVariant(variant, "unrolled-auto", "unrolled", "auto"),
      plannedVariant(variant, "rolled-none", "rolled", "none"),
    ],
    tlspec: { ...asset, bytes: 2_400_000, runnable: true },
    recipe_text: "torchvision.models.resnet18(weights=None)",
    torchlens_version: "2.35.3",
    renderer_version: "graphviz 2.43.0",
    validation_state: "passed",
    validation_kind: "torch-replay",
    capture: "full",
    backend: "torch",
    rung: "library",
    fidelity: "faithful",
    hashes: { skeleton: "a".repeat(64), architecture: "b".repeat(64) },
    interactive: { ...asset, bytes: 140_000 },
    netron: { ...asset, bytes: 830_000 },
    model_explorer: { ...asset, bytes: 1_600_000 },
    paper_title: "Deep Residual Learning for Image Recognition",
    year: 2016,
    year_source: "paper",
    era: "2015-2019",
    authors: ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
    venue: "CVPR",
    bibtex: "@inproceedings{he2016deep, title={Deep Residual Learning for Image Recognition}, year={2016}}",
    citations: { count: 250_000, as_of: "2026-10-04" },
    labs: ["Microsoft Research"],
    lab: "Microsoft Research",
    field: "computer vision",
    output_type: "class scores",
    tagline: "An 18-layer image classifier built from residual blocks.",
    summary_markdown:
      "ResNet-18 stacks **residual blocks**: each block adds its input back to its output, so the network learns a correction to the identity.\n\n- Four stages of two blocks each\n- A global average pool before the classifier\n\nThe idea comes from [the ResNet paper](https://arxiv.org/abs/1512.03385).",
    key_contribution: "Skip connections let very deep networks train without degrading.",
    keywords: ["residual", "image classification", "skip connections"],
    share_image: { ...variant.thumb, width: 1200, height: 630 },
    own_diagram: true,
    diagram_shared_with: 0,
    family_summary: "Residual networks of increasing depth, from the 2016 paper that introduced skip connections.",
    aliases: ["ResNet18", "resnet-18"],
  };
}

/** One fixture record by case. */
export function fixtureRecord(kind: FixtureCase): ModelRecord {
  switch (kind) {
    case "empty":
      return emptyRecord();
    case "full":
      return fullRecord();
    case "structure-only":
      return {
        ...fullRecord(),
        slug: "fixture-structure-only",
        display_name: "Fixture: structure-only capture",
        capture: "structure_only",
        validation_state: "structure_only",
        validation_kind: null,
        forward_validated: false,
        param_count: 11_700_000,
        params_source: "geometry",
      };
    case "preview-backend":
      return {
        ...fullRecord(),
        slug: "fixture-preview-backend",
        display_name: "Fixture: JAX preview backend",
        backend: "jax",
        validation_kind: "jax-preview",
        fidelity: "minor",
        fidelity_note: "Uses a learned positional embedding where the paper uses a fixed one.",
        tlspec: null,
        netron: null,
        model_explorer: null,
        variants: [fullRecord().variants[0]],
      };
  }
}

/** The fixture family: the full, structure-only and preview-backend records. */
export function fixtureFamily(): Family {
  const members = (["full", "structure-only", "preview-backend"] as const).map(fixtureRecord);
  const family = [...groupFamilies(members).values()][0];
  if (!family) {
    throw new Error("Fixture family did not form");
  }
  return family;
}

/** A family with no members, the empty case for the family page. */
export function emptyFamily(): Family {
  return { name: "Fixture: empty family", slug: "fixture-empty", members: [], paper: null, summary: null };
}

/** Fixture model pages: slug, record and the family context the page shows. */
export function fixtureModelPages(): Array<{ model: ModelRecord; family: Family | null }> {
  return FIXTURE_CASES.map((kind) => {
    const model = fixtureRecord(kind);
    return { model, family: kind === "empty" ? null : fixtureFamily() };
  });
}

/** A few real catalog records for the relatives block, so those links go to pages that exist. */
export function realNeighbours(count: number): ModelRecord[] {
  return getAllModels().slice(0, count);
}
