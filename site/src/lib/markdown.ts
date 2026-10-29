/**
 * A small, safe Markdown subset for generated model summaries: paragraphs, "- " lists, `code`,
 * **bold**, *italic* and [links](https://...). Everything is HTML-escaped first, and only
 * http(s) links survive, so a summary can never inject markup.
 */
import { escapeHtml } from "@/scripts/CardView";

function inline(text: string): string {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a class="textlink" href="$2" rel="noopener">$1</a>');
}

/** Render the subset to HTML; returns "" for empty input. */
export function renderSummaryMarkdown(markdown: string | null | undefined): string {
  if (!markdown?.trim()) {
    return "";
  }
  return markdown
    .replace(/\r\n?/g, "\n")
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean)
    .map((block) => {
      const lines = block.split("\n").map((line) => line.trim());
      if (lines.every((line) => /^[-*] /.test(line))) {
        return `<ul>${lines.map((line) => `<li>${inline(line.slice(2))}</li>`).join("")}</ul>`;
      }
      return `<p>${inline(lines.join(" ").replace(/^#+\s*/, ""))}</p>`;
    })
    .join("");
}
