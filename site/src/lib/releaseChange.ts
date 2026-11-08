/**
 * The Changes page entry for the current dataset release, dated by the export's `as_of` and
 * worded from its counts, so it never drifts from what the site shows.
 */
import type { Change } from "@/lib/changelog";
import { getAllModels } from "@/lib/catalog";
import { getSiteCounts } from "@/lib/counts";
import { formatNumber } from "@/lib/labels";
import { rosterEntryCount } from "@/lib/roster";
import { datasetVersion, isLaunched } from "@/lib/stage";

/** True when at least one record in the export carries a citation count. */
function hasCitationCounts(): boolean {
  return getAllModels().some((model) => model.citations != null && Number.isFinite(model.citations.count) && model.citations.count >= 0);
}

/** The entry for this release, or null when the export has no date (early exports). */
export function releaseChange(): Change | null {
  const counts = getSiteCounts();
  if (!counts.asOf) {
    return null;
  }
  const version = datasetVersion();
  const items = [
    `${formatNumber(counts.validated)} models have a diagram whose trace was replayed operation by operation and passed its check. Entries whose check did not pass, or whose diagram turned out to belong to a different model, no longer have a diagram page.`,
    counts.structureOnly ? `${formatNumber(counts.structureOnly)} more models have their graph's structure only. Their pages say so, and the model list shows them when you tick "Include models without a checked diagram".` : null,
    rosterEntryCount() > 0
      ? "Every model in the catalog is now listed. Those without a diagram yet appear as text in the search and, when you ask for them, in the model list, with their status and links to the paper and the code."
      : null,
    hasCitationCounts()
      ? "Model pages show a one-line description under the name, the paper's citation count with its date where known, and a BibTeX entry to copy."
      : "Model pages show a one-line description under the name and a BibTeX entry to copy.",
    "Pictures and diagrams keep their place while they load, so the page no longer jumps.",
    version ? `The dataset download is release ${version}.` : null,
  ].filter((item): item is string => item !== null);
  return { date: counts.asOf, title: isLaunched() ? "Launch" : "Checked diagrams", items };
}
