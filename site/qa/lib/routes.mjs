// Model pages the harness visits, chosen from the export so they exist in the build under test:
// the preferred slug when the export shows it, else a stand-in by size. Reads `site-data/models.jsonl`
// relative to the working directory (run from site/), or the path given.
import fs from "node:fs";

/** Shown and structure-only records, by the same flags the site reads (src/lib/shown.ts). */
export function tiers(file) {
  const shown = [];
  const structure = [];
  if (!fs.existsSync(file)) return { shown, structure };
  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    if (!line) continue;
    const r = JSON.parse(line);
    if (r.fidelity === "fabrication" || r.own_diagram === false) continue;
    if (r.capture === "structure_only" || r.validation_state === "structure_only") structure.push(r);
    else if (r.validation_state === undefined || (r.forward_validated && r.validation_state === "passed")) shown.push(r);
  }
  return { shown, structure };
}

function nearest(records, ops) {
  return records.filter((r) => typeof r.n_ops === "number").sort((a, b) => Math.abs(a.n_ops - ops) - Math.abs(b.n_ops - ops) || a.slug.localeCompare(b.slug))[0];
}

function widest(records) {
  const ratio = (r) => {
    const v = (r.variants ?? []).find((x) => x.key === r.default_variant) ?? r.variants?.[0];
    const w = r.svg_width_px ?? v?.svg_width_px;
    const h = r.svg_height_px ?? v?.svg_height_px;
    return w && h ? w / h : 0;
  };
  const best = [...records].sort((a, b) => ratio(b) - ratio(a))[0];
  return best && ratio(best) > 0 ? best : null;
}

/** {small, wide, large, monster, structure}: site paths, or null when the export has no candidate. */
export function modelRoutes(file = "site-data/models.jsonl") {
  const { shown, structure } = tiers(file);
  const has = new Set(shown.map((r) => r.slug));
  const pick = (preferred, fallback) => {
    const slug = preferred.find((s) => has.has(s)) ?? fallback?.slug;
    return slug ? `/models/${slug}/` : null;
  };
  const monster = [...shown].sort((a, b) => (b.n_ops ?? 0) - (a.n_ops ?? 0))[0];
  return {
    small: pick(["resnet18", "pi-sigma-network"], nearest(shown, 60)),
    wide: pick(["gpt2"], widest(shown) ?? nearest(shown, 800)),
    large: pick(["espnet-tts-tacotron2-gst"], nearest(shown, 3000)),
    monster: pick(["nllb-moe"], monster),
    structure: structure.length ? `/models/${structure.sort((a, b) => a.slug.localeCompare(b.slug))[0].slug}/` : null,
  };
}
