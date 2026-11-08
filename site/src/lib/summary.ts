/**
 * The one-line label shown with every AI-written summary, on model and family pages (site plan fork 3,
 * JMT 2026-10-10: summaries publish with no human review, always labelled). It names the source the
 * summary was written from (`summary_basis`) and says it was checked only when the record carries
 * `summary_checked`.
 */
const SUMMARY_SOURCES: Record<string, string> = {
  full: "the paper",
  abstract: "the paper's abstract",
  docs: "the model's documentation",
  readme: "the project's README",
};

export function summaryLabel(basis: string | null | undefined, checked: boolean): string {
  const source = basis ? SUMMARY_SOURCES[basis] : undefined;
  const written = source ? `AI-written summary, based on ${source}` : "AI-written summary";
  const check = checked ? (source ? " and checked automatically against it" : ", checked automatically against its sources") : "";
  return `${written}${check}; not reviewed by a person. Corrections welcome.`;
}
