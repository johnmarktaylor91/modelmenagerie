/**
 * Build side of the optional export files that list what the site does not draw:
 * `roster-index.jsonl` (catalogued models with no diagram yet) and `families.jsonl` (family
 * summaries). Both are optional; when absent the roster list is empty and families are grouped
 * from the records alone.
 */
import fs from "node:fs";
import path from "node:path";

import { getExportRoot } from "@/lib/catalog";
import { canonicalFamily, cleanModelName } from "@/lib/display-names";
import type { FamilyRecord } from "@/types/record";
import { encodeRosterIndex, type RosterIndexFile, type RosterRecord } from "@/scripts/RosterIndex";

let rosterFile: RosterIndexFile | null = null;
let familyRecords: Map<string, FamilyRecord> | null = null;

/** Read a JSON Lines file from the site data, or [] when it is absent. */
function readJsonl<T>(name: string): T[] {
  const file = path.join(getExportRoot(), name);
  if (!fs.existsSync(file)) {
    return [];
  }
  return fs
    .readFileSync(file, "utf-8")
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as T);
}

/** The compact roster index, built once per build (empty rows when the export has none). */
export function getRosterIndexFile(): RosterIndexFile {
  rosterFile ??= encodeRosterIndex(
    readJsonl<RosterRecord>("roster-index.jsonl").map((row) => ({
      ...row,
      display_name: typeof row.display_name === "string" ? cleanModelName(row.display_name) : row.display_name,
      family_normalized: canonicalFamily(row.family_normalized),
    })),
  );
  return rosterFile;
}

/** How many roster entries the site lists as text. */
export function rosterEntryCount(): number {
  return getRosterIndexFile().rows.length;
}

/** Family rows from `families.jsonl`, keyed by the catalog's family value (empty when absent). */
export function getFamilyRecords(): Map<string, FamilyRecord> {
  familyRecords ??= new Map(readJsonl<FamilyRecord>("families.jsonl").map((row) => [row.family, row]));
  return familyRecords;
}
