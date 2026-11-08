// The copy check: words and marks that must not appear in what a visitor reads (visible text, alt and
// title text, meta descriptions). One list, used on built HTML and on crawled pages. Model and family
// names come from the catalog data, not from the site's copy: names that trip a rule are taken out
// before checking and reported apart (`dataHits`), for the data to be cleaned, not the pages.
import fs from "node:fs";

/** [name, pattern, why]. Patterns run on visible text with tags removed. */
export const COPY_RULES = [
  ["internal word", /\b(Tier-B|Phase 0|rungs?|funnel|TODO|pending|unknown|site-data|preview honesty|stand-in heuristic)\b/i, "internal vocabulary (plan section 2)"],
  ["internal word", /\bR2\b/, "internal vocabulary: the file host's name"],
  ["em dash", /—|\s--\s/, "house voice: no em dashes"],
  ["arrow joiner", /\s(->|=>|→)\s/, "house voice: no arrows as joiners"],
  ["hype", /\b(revolutionary|cutting-edge|state-of-the-art|seamless(ly)?|game-?changing|groundbreaking|world-class|best-in-class|blazing|effortless(ly)?|unleash|supercharge)\b/i, "plain copy"],
  ["personal address", /@(gmail|googlemail|columbia)\.|\bjt3295\b|johnmarkedwardtaylor/i, "no personal email on the public site; corrections go to the repository's issues"],
  ["markup as text", /<!--|-->/, "an HTML comment escaped into visible text (for example a Cloudflare marker given to set:html as a quoted string)"],
  ["unverifiable", /\b(the (largest|biggest|most complete)|every neural network|all neural networks|complete catalog|definitive)\b/i, "claims the site cannot back"],
];

/** Phrases decided by JMT that the rules would otherwise flag (rulings 2026-10-10). */
const ALLOWED = [/Preview not available/gi];

/** Tags, with quoted attribute values that may hold ">" (model names such as "encoder -> head"). */
const TAG = /<(?:[^>"']|"[^"]*"|'[^']*')*>/g;

/**
 * Catalog names (display and family) that would trip a copy rule, longest first, from a
 * `models.jsonl`; empty when the file is absent.
 */
export function dataNames(modelsJsonl) {
  if (!modelsJsonl || !fs.existsSync(modelsJsonl)) return [];
  const names = new Set();
  for (const line of fs.readFileSync(modelsJsonl, "utf8").split("\n")) {
    if (!line) continue;
    const record = JSON.parse(line);
    for (const name of [record.display_name, record.family_normalized]) {
      if (name && COPY_RULES.some(([, pattern]) => pattern.test(name))) names.add(name);
    }
  }
  return [...names].sort((a, b) => b.length - a.length);
}

/**
 * Elements marked `data-quoted` hold words quoted from the paper or written from it (paper titles,
 * the labelled summary, key idea, keywords): not the site's own copy. They are taken out before the
 * rules run and checked apart (`quotedHits`), for the data to be cleaned upstream.
 */
const QUOTED = /<([a-z]+)\b(?:[^>"']|"[^"]*"|'[^']*')*\sdata-quoted(?:="")?(?:[^>"']|"[^"]*"|'[^']*')*>[\s\S]*?<\/\1>/gi;

/** The quoted regions of a page, as HTML. */
export function quotedRegions(html) {
  return [...html.matchAll(QUOTED)].map((match) => match[0]);
}

/** Visible text of an HTML page plus alt, title and aria-label text and the meta description. */
export function readableText(html) {
  const attrs = [...html.matchAll(/\s(?:alt|title|aria-label)="([^"]*)"/g)].map((m) => m[1]);
  const meta = [...html.matchAll(/<meta\s+(?:name="description"|property="og:(?:title|description)")\s+content="([^"]*)"/g)].map((m) => m[1]);
  const body = html
    .replace(QUOTED, " ")
    .replace(/<script\b[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[\s\S]*?<\/style>/gi, " ")
    .replace(/<pre\b[\s\S]*?<\/pre>/gi, " ")
    .replace(/<code\b[\s\S]*?<\/code>/gi, " ")
    .replace(TAG, " ");
  const text = [body, ...attrs, ...meta].join(" \n ").replace(/&amp;/g, "&").replace(/&#39;|&apos;/g, "'").replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&nbsp;|&#160;/g, " ");
  return ALLOWED.reduce((acc, pattern) => acc.replace(pattern, " "), text);
}

function ruleHits(text) {
  const hits = [];
  for (const [rule, pattern, why] of COPY_RULES) {
    const global = new RegExp(pattern.source, pattern.flags.includes("g") ? pattern.flags : `${pattern.flags}g`);
    for (const match of text.matchAll(global)) {
      const at = match.index ?? 0;
      hits.push({ rule, why, match: match[0].trim(), context: text.slice(Math.max(0, at - 50), at + 50).replace(/\s+/g, " ").trim() });
      if (hits.length > 20) break;
    }
  }
  // A preview build: the header tag says "preview" once on every page and the home page adds one sentence
  // (ruling look.4); more than that on one page is repetition. A launch build (PUBLIC_SITE_STAGE=launch, what
  // production builds) has neither, so any "preview" left is stale preview copy.
  const previews = (text.match(/\bpreview\b/gi) ?? []).length;
  const launch = (process.env.PUBLIC_SITE_STAGE ?? "").trim().toLowerCase() === "launch";
  if (launch ? previews > 0 : previews > 3) {
    const why = launch ? "a launch build says preview nowhere" : "tag plus one sentence only";
    hits.push({ rule: "preview repeated", why, match: `${previews} mentions`, context: "" });
  }
  return hits;
}

/** Rule hits inside the quoted regions of a page (reported, never gated). */
export function quotedHits(html) {
  const text = quotedRegions(html).join(" \n ").replace(TAG, " ").replace(/&amp;/g, "&").replace(/&#39;|&apos;/g, "'").replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">");
  return ruleHits(text).filter((hit) => hit.rule !== "preview repeated");
}

/** Copy hits on one page: [{rule, why, match, context}], after taking out the `exempt` catalog names. */
export function checkCopy(html, exempt = []) {
  let text = readableText(html);
  const first = ruleHits(text);
  if (first.length === 0 || exempt.length === 0) return first;
  for (const name of exempt) {
    if (text.includes(name)) text = text.split(name).join(" ");
  }
  return ruleHits(text);
}
