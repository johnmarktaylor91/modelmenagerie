/**
 * Pagination markup shared by the static listing pages and the client listing. Links are real
 * links in both modes; the client intercepts them to page without a reload.
 */
import { formatNumber } from "@/lib/labels";

/** Page numbers to show: first, last and two either side of the current page. */
export function visiblePages(current: number, pages: number): number[] {
  const shown = new Set([1, pages]);
  for (let page = current - 2; page <= current + 2; page += 1) {
    if (page >= 1 && page <= pages) {
      shown.add(page);
    }
  }
  return [...shown].sort((left, right) => left - right);
}

function pageLink(page: number, label: string, href: string, extraClass = ""): string {
  return `<a class="button pager__item${extraClass}" href="${href}" data-page="${page}">${label}</a>`;
}

function inert(label: string, extraClass = ""): string {
  return `<span class="button pager__item${extraClass}" aria-disabled="true">${label}</span>`;
}

/** Render Previous, numbered pages with gaps, a phone-only "Page n of m" line, and Next. */
export function renderPagerHtml(current: number, pages: number, hrefFor: (page: number) => string): string {
  if (pages <= 1) {
    return "";
  }
  const parts: string[] = [];
  parts.push(current > 1 ? pageLink(current - 1, "Previous", hrefFor(current - 1), " pager__step") : inert("Previous", " pager__step"));
  let previous = 0;
  for (const page of visiblePages(current, pages)) {
    if (page - previous > 1) {
      parts.push('<span class="pager__gap" aria-hidden="true">&hellip;</span>');
    }
    parts.push(
      page === current
        ? `<a class="button primary pager__item pager__number" href="${hrefFor(page)}" data-page="${page}" aria-current="page">${formatNumber(page)}</a>`
        : pageLink(page, formatNumber(page), hrefFor(page), " pager__number"),
    );
    previous = page;
  }
  parts.push(`<span class="pager__summary">Page ${formatNumber(current)} of ${formatNumber(pages)}</span>`);
  parts.push(current < pages ? pageLink(current + 1, "Next", hrefFor(current + 1), " pager__step") : inert("Next", " pager__step"));
  return parts.join("");
}

/** "Showing 1 to 48 of 14,291 models", or "No matches". */
export function rangeText(start: number, end: number, total: number, noun: { one: string; many: string }): string {
  if (total === 0) {
    return "No matches";
  }
  return `Showing ${formatNumber(start)} to ${formatNumber(end)} of ${formatNumber(total)} ${total === 1 ? noun.one : noun.many}`;
}
