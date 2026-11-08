/**
 * The public counts, read from the export's manifest and funnel, never typed into a page (the QA
 * harness checks for typed counts). Every count is printed with the date it is true for.
 *
 * The pages must agree with each other, so when the funnel disagrees with what the site actually
 * shows, the shown set wins: the validated count is the number of cards in the default listing,
 * the structure-only count the number of structure-only pages, and the catalog's model names the
 * number of entries the listing reaches with "Include models without a checked diagram" ticked. Each
 * disagreement is logged once at build time, for the export to be corrected upstream.
 */
import { getAllModels, getFunnel, getManifest, getShownModels } from "@/lib/catalog";
import { exportDate, formatDate } from "@/lib/dataset";
import { rosterEntryCount } from "@/lib/roster";
import { recordTier } from "@/lib/shown";

export interface SiteCounts {
  /** Models with a checked diagram (the default listing). */
  validated: number;
  /** Models whose graph was captured but whose forward pass was not checked; null when the export does not say. */
  structureOnly: number | null;
  /** Model names in the catalog, drawn or not; null when the export does not say. */
  rosterNames: number | null;
  /** Distinct architectures among the validated models, null when the export does not say. */
  distinctArchitectures: number | null;
  /** The date the counts are true for (ISO), null when the export does not say. */
  asOf: string | null;
  /** "2 November 2026", or "" when there is no date. */
  asOfText: string;
}

/** One count where the export's own figure and the site's shown set disagree. */
export interface CountDisagreement {
  count: string;
  exported: number;
  shown: number;
}

let counts: SiteCounts | null = null;
let disagreements: CountDisagreement[] = [];

function count(...values: Array<number | null | undefined>): number | null {
  return values.find((value): value is number => typeof value === "number" && value >= 0) ?? null;
}

/** The shown figure, noting when the export said something else. */
function reconcile(name: string, exported: number | null, shown: number): number {
  if (exported !== null && exported !== shown) {
    disagreements.push({ count: name, exported, shown });
  }
  return shown;
}

/** A funnel count newer than the funnel's type (``catalog_entries``). */
function funnelValue(name: string): number | null {
  const value = (getFunnel() as unknown as Record<string, unknown>)[name];
  return typeof value === "number" ? value : null;
}

/** One TorchLens version and how many checked records it traced. */
export interface VersionCount {
  version: string;
  checked: number;
}

/**
 * Checked records per TorchLens version, newest first, from the manifest's
 * ``validated_by_torchlens_version``; empty when the export does not say. Several accepted pins can
 * stand in one release (decisions.md, FORKS item 1 B), so the pages print the split.
 */
export function checkedByVersion(): VersionCount[] {
  const raw = (getManifest().counts as unknown as Record<string, unknown>).validated_by_torchlens_version;
  if (!raw || typeof raw !== "object") return [];
  return Object.entries(raw as Record<string, unknown>)
    .filter((entry): entry is [string, number] => typeof entry[1] === "number" && entry[1] > 0)
    .map(([version, checked]) => ({ version, checked }))
    .sort((a, b) => b.version.localeCompare(a.version, "en", { numeric: true }));
}

/** The counts for this build, read once. */
export function getSiteCounts(): SiteCounts {
  if (counts === null) {
    const funnel = getFunnel();
    const manifest = getManifest();
    const asOf = funnel.as_of?.slice(0, 10) ?? exportDate();
    const pages = getAllModels();
    // Exports without validation fields (early previews) have no structure-only tier to count.
    const hasTiers = pages.some((model) => model.validation_state !== undefined);
    const structurePages = pages.filter((model) => recordTier(model) === "structure").length;
    const rosterRows = rosterEntryCount();
    disagreements = [];
    counts = {
      validated: reconcile("validated", count(funnel.validated_records, manifest.counts.validated), getShownModels().length),
      structureOnly: hasTiers
        ? reconcile("structure only", count(funnel.structure_only_records, manifest.counts.structure_only), structurePages)
        : count(funnel.structure_only_records, manifest.counts.structure_only),
      // Without a roster index the listing cannot reach the other names, so the export's figure stands.
      // The catalog's entries: every model page plus every other roster name (the export's catalog_entries).
      rosterNames: rosterRows > 0
        ? reconcile("model names in the catalog", count(funnelValue("catalog_entries"), funnel.roster_names), pages.length + rosterRows)
        : count(funnelValue("catalog_entries"), funnel.roster_names),
      distinctArchitectures: count(funnel.distinct_architectures),
      asOf,
      asOfText: asOf ? formatDate(asOf) : "",
    };
    for (const row of disagreements) {
      console.warn(`[counts] the export says ${row.exported} ${row.count}; the site shows ${row.shown} and prints that`);
    }
  }
  return counts;
}

/** Where the export's counts and the shown set disagreed in this build (empty when they agree). */
export function getCountDisagreements(): readonly CountDisagreement[] {
  getSiteCounts();
  return disagreements;
}
