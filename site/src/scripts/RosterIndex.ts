/**
 * The compact list of catalogued models that have no diagram yet (`/roster-index.json`). They are
 * listed as text entries (header search suggestions, and the listing when the reader asks for
 * them) with a status phrase and links to the paper and the code; they get no page of their own.
 *
 * Shared by the build-time endpoint (encode side) and the browser (decode side). Loaded only on
 * request, never on first paint.
 */
import type { ListedStatus } from "@/types/record";
import { compactText, escapeHtml } from "@/scripts/CardView";

export const ROSTER_INDEX_VERSION = 1;

export const LISTED_STATUSES: readonly ListedStatus[] = ["not_yet_traced", "source_unavailable", "paper_only"];

/** The phrase a visitor reads for each status. */
export const STATUS_PHRASES: Record<ListedStatus, string> = {
  not_yet_traced: "Not yet traced",
  source_unavailable: "Source unavailable",
  paper_only: "Paper only",
};

/**
 * One row: display name, status index, family index (-1 when the family only repeats the name),
 * year (null when not known), code link, paper link. Links are shortened: "g:owner/repo" for a
 * GitHub repository, "a:2401.12345" for an arXiv abstract, else the full URL; "" for none.
 */
export type RosterRow = [name: string, status: number, family: number, year: number | null, source: string, paper: string];

export interface RosterIndexFile {
  version: number;
  families: string[];
  rows: RosterRow[];
}

/** One roster row from the export (`roster-index.jsonl`, site-export/1). */
export interface RosterRecord {
  stable_id: string;
  display_name: string;
  family_normalized?: string | null;
  zoo?: string | null;
  source_url?: string | null;
  paper_url?: string | null;
  year?: number | null;
  listed_status: ListedStatus;
  listed_reason?: string | null;
}

/** One decoded entry, ready to search and show. */
export interface RosterEntry {
  name: string;
  status: ListedStatus;
  family: string;
  year: number | null;
  sourceUrl: string;
  paperUrl: string;
}

const GITHUB = "https://github.com/";
const ARXIV_ABS = "https://arxiv.org/abs/";

/** Shorten a link for the index (see `RosterRow`). Only http(s) links are kept. */
export function shortenLink(url: string | null | undefined): string {
  const value = (url ?? "").trim();
  if (!/^https?:\/\//i.test(value)) {
    return "";
  }
  const repo = value.startsWith(GITHUB) ? value.slice(GITHUB.length).replace(/\/+$/, "") : "";
  if (repo && /^[\w.-]+\/[\w.-]+$/.test(repo)) {
    return `g:${repo}`;
  }
  const arxiv = value.startsWith(ARXIV_ABS) ? value.slice(ARXIV_ABS.length) : "";
  if (/^\d{4}\.\d{4,5}(v\d+)?$/.test(arxiv)) {
    return `a:${arxiv}`;
  }
  return value;
}

/** Expand a shortened link; anything that is not http(s) after expansion becomes "". */
export function expandLink(short: string): string {
  if (short.startsWith("g:")) {
    return `${GITHUB}${short.slice(2)}`;
  }
  if (short.startsWith("a:")) {
    return `${ARXIV_ABS}${short.slice(2)}`;
  }
  return /^https?:\/\//i.test(short) ? short : "";
}

/** Build the compact file from export rows, sorted by name. */
export function encodeRosterIndex(records: readonly RosterRecord[]): RosterIndexFile {
  const families: string[] = [];
  const familyPositions = new Map<string, number>();
  const familyIndex = (family: string, name: string): number => {
    const key = compactText(family);
    if (!key || key === "unknown" || compactText(name).includes(key)) {
      return -1;
    }
    let known = familyPositions.get(family);
    if (known === undefined) {
      known = families.push(family) - 1;
      familyPositions.set(family, known);
    }
    return known;
  };
  const sorted = [...records]
    .filter((record) => record.display_name && LISTED_STATUSES.includes(record.listed_status))
    .sort((left, right) => left.display_name.localeCompare(right.display_name, "en", { sensitivity: "base" }));
  const rows = sorted.map(
    (record): RosterRow => [
      record.display_name,
      LISTED_STATUSES.indexOf(record.listed_status),
      familyIndex(record.family_normalized ?? "", record.display_name),
      typeof record.year === "number" && Number.isFinite(record.year) ? record.year : null,
      shortenLink(record.source_url),
      shortenLink(record.paper_url),
    ],
  );
  return { version: ROSTER_INDEX_VERSION, families, rows };
}

/** Validate and decode the fetched file; throws on anything unexpected. */
export function decodeRosterIndex(value: unknown): RosterEntry[] {
  const file = value as Partial<RosterIndexFile> | null;
  if (!file || file.version !== ROSTER_INDEX_VERSION || !Array.isArray(file.families) || !Array.isArray(file.rows)) {
    throw new Error("The list of models without a checked diagram has an unexpected format");
  }
  const families = file.families;
  return file.rows.map(([name, status, family, year, source, paper]) => ({
    name,
    status: LISTED_STATUSES[status] ?? "not_yet_traced",
    family: families[family] ?? "",
    year: year ?? null,
    sourceUrl: expandLink(source ?? ""),
    paperUrl: expandLink(paper ?? ""),
  }));
}

/** The paper and code links of an entry as HTML (empty when it has neither). */
export function rosterLinksHtml(entry: RosterEntry): string {
  const links = [
    entry.paperUrl && `<a class="listed-entry__link" href="${escapeHtml(entry.paperUrl)}" rel="noopener">Paper</a>`,
    entry.sourceUrl && `<a class="listed-entry__link" href="${escapeHtml(entry.sourceUrl)}" rel="noopener">Source</a>`,
  ].filter(Boolean);
  return links.join(" ");
}

/**
 * A text entry for the listing, in the same three fact slots as a model card so the list view's
 * columns line up: status phrase, then family and year, then the paper and code links. It has
 * no page, so the entry itself is not a link.
 */
export function renderRosterEntryHtml(entry: RosterEntry, headingLevel: 2 | 3 = 2): string {
  const heading = `h${headingLevel}`;
  const about = [entry.family, entry.year !== null ? String(entry.year) : ""].filter(Boolean).join(", ");
  return `<article class="model-card model-card--text model-card--listed">
  <div class="model-card__body">
    <${heading} class="model-card__title">${escapeHtml(entry.name)}</${heading}>
    <p class="model-card__facts"><span class="model-card__fact model-card__fact--task">${escapeHtml(STATUS_PHRASES[entry.status])}</span><span class="model-card__fact model-card__fact--ops">${escapeHtml(about)}</span><span class="model-card__fact model-card__fact--params">${rosterLinksHtml(entry)}</span></p>
  </div>
</article>`;
}
