/**
 * Search suggestions for a `SiteSearch` box (ARIA combobox with a listbox). The compact index
 * loads on first focus; each keystroke ranks it in the browser, so suggestions need no server.
 * Catalogued models with no diagram yet follow as text entries with their status, from a second
 * index fetched once the visitor starts typing.
 */
import { taskLabel } from "@/lib/labels";
import { escapeHtml } from "@/scripts/CardView";
import { loadFailureText, loadRosterIndex, loadSearchIndex, type LoadedIndex, type LoadedRoster } from "@/scripts/IndexLoader";
import type { IndexEntry } from "@/scripts/ListingIndex";
import { STATUS_PHRASES, type RosterEntry } from "@/scripts/RosterIndex";
import { parseQuery, scoreMatch } from "@/scripts/SearchMatch";

export const SUGGESTIONS_SHOWN = 8;
/** At most this many models with no diagram yet follow the suggestions. */
export const ROSTER_SUGGESTIONS_SHOWN = 3;
const DEBOUNCE_MS = 60;

/** The best matches for a query, best first; ties go to the shorter name, then catalog order. */
export function suggest(index: LoadedIndex, raw: string, limit = SUGGESTIONS_SHOWN): IndexEntry[] {
  const query = parseQuery(raw);
  if (!query) {
    return [];
  }
  const scored: Array<{ entry: IndexEntry; score: number }> = [];
  for (const entry of index.entries) {
    const score = scoreMatch(index.fields[entry.order], query);
    if (score !== null) {
      scored.push({ entry, score });
    }
  }
  scored.sort((left, right) => left.score - right.score || left.entry.card.name.length - right.entry.card.name.length || left.entry.order - right.entry.order);
  return scored.slice(0, limit).map(({ entry }) => entry);
}

/**
 * Matches for the query, or when it has none, for the query with its last words dropped one at a
 * time ("resnet 999" falls back to "resnet"). `used` is the query the matches answer.
 */
export function suggestClosest(index: LoadedIndex, raw: string, limit = SUGGESTIONS_SHOWN): { entries: IndexEntry[]; used: string } {
  const words = raw.trim().split(/\s+/).filter(Boolean);
  for (let count = words.length; count >= 1; count -= 1) {
    const used = words.slice(0, count).join(" ");
    const entries = suggest(index, used, limit);
    if (entries.length) {
      return { entries, used };
    }
  }
  return { entries: [], used: raw.trim() };
}

/** The best roster matches for a query (exact and prefix matches first, then shorter names). */
export function suggestRoster(roster: LoadedRoster, raw: string, limit = ROSTER_SUGGESTIONS_SHOWN): RosterEntry[] {
  const query = parseQuery(raw);
  if (!query) {
    return [];
  }
  const scored: Array<{ entry: RosterEntry; score: number; order: number }> = [];
  roster.entries.forEach((entry, order) => {
    const score = scoreMatch(roster.fields[order], query);
    if (score !== null) {
      scored.push({ entry, score, order });
    }
  });
  scored.sort((left, right) => left.score - right.score || left.entry.name.length - right.entry.name.length || left.order - right.order);
  return scored.slice(0, limit).map(({ entry }) => entry);
}

function rosterOptionHtml(id: string, entry: RosterEntry): string {
  const href = `/models/?q=${encodeURIComponent(entry.name)}&untraced=1`;
  const meta = [STATUS_PHRASES[entry.status], entry.family].filter(Boolean).join(" · ");
  return `<li class="site-search__option site-search__option--listed" id="${id}" role="option" aria-selected="false" data-href="${escapeHtml(href)}"><span class="site-search__option-name">${escapeHtml(entry.name)}</span><span class="site-search__option-meta">${escapeHtml(meta)}</span></li>`;
}

function optionHtml(id: string, entry: IndexEntry): string {
  const meta = [taskLabel(entry.card.category), entry.card.family].filter(Boolean).join(" · ");
  return `<li class="site-search__option" id="${id}" role="option" aria-selected="false" data-href="/models/${encodeURIComponent(entry.card.slug)}/"><span class="site-search__option-name">${escapeHtml(entry.card.name)}</span><span class="site-search__option-meta">${escapeHtml(meta)}</span></li>`;
}

function allResultsHtml(id: string, query: string): string {
  const href = `/models/?q=${encodeURIComponent(query)}`;
  return `<li class="site-search__option site-search__option--all" id="${id}" role="option" aria-selected="false" data-href="${escapeHtml(href)}"><span class="site-search__option-name">See all results for "${escapeHtml(query)}"</span></li>`;
}

function part<T extends Element>(form: HTMLFormElement, selector: string): T {
  const node = form.querySelector<T>(selector);
  if (!node) {
    throw new Error(`Search box part missing: ${selector}`);
  }
  return node;
}

/** Wire one search box. */
export function initSiteSearch(form: HTMLFormElement): void {
  const input = part<HTMLInputElement>(form, "input[role='combobox']");
  const popup = part<HTMLElement>(form, "[data-search-popup]");
  const list = part<HTMLUListElement>(form, "[data-search-list]");
  const message = part<HTMLElement>(form, "[data-search-message]");
  const retry = part<HTMLButtonElement>(form, "[data-search-retry]");
  const idBase = list.id;
  let active = -1;
  let token = 0;
  let debounce: number | undefined;

  const options = () => [...list.querySelectorAll<HTMLElement>("[role='option']")];

  function setOpen(open: boolean): void {
    popup.hidden = !open;
    input.setAttribute("aria-expanded", String(open && list.childElementCount > 0));
    if (!open) {
      setActive(-1);
    }
  }

  function setActive(next: number): void {
    const all = options();
    active = all.length ? Math.max(-1, Math.min(next, all.length - 1)) : -1;
    all.forEach((option, position) => option.setAttribute("aria-selected", String(position === active)));
    if (active >= 0) {
      input.setAttribute("aria-activedescendant", all[active].id);
      all[active].scrollIntoView({ block: "nearest" });
    } else {
      input.removeAttribute("aria-activedescendant");
    }
  }

  function showMessage(text: string, canRetry = false): void {
    message.textContent = text;
    retry.hidden = !canRetry;
  }

  async function refresh(): Promise<void> {
    const raw = input.value.trim();
    const mine = ++token;
    if (!raw) {
      list.innerHTML = "";
      showMessage("");
      setOpen(false);
      return;
    }
    let index: LoadedIndex;
    try {
      if (!list.childElementCount) {
        showMessage("Loading the model list...");
        setOpen(true);
      }
      index = await loadSearchIndex();
    } catch (error) {
      if (mine === token) {
        list.innerHTML = "";
        showMessage(`${loadFailureText(error)} Press Enter to search anyway.`, true);
        setOpen(true);
      }
      return;
    }
    if (mine !== token) {
      return;
    }
    const { entries: matches, used } = suggestClosest(index, raw);
    const render = (listed: RosterEntry[]) => {
      const options = [
        ...matches.map((entry, position) => optionHtml(`${idBase}-${position}`, entry)),
        ...listed.map((entry, position) => rosterOptionHtml(`${idBase}-listed-${position}`, entry)),
      ];
      list.innerHTML = options.length ? [...options, allResultsHtml(`${idBase}-all`, matches.length ? used : raw)].join("") : "";
      const message = !options.length ? "No models match. Try a shorter name." : matches.length && used !== raw ? `Nothing matches all of that. Showing matches for "${used}".` : "";
      showMessage(message);
      setOpen(true);
      setActive(-1);
    };
    render([]);
    // Models with no diagram yet: fetched once, on the first keystroke; a failure only drops them.
    const roster = await loadRosterIndex().catch(() => null);
    if (roster && mine === token) {
      const listed = suggestRoster(roster, raw);
      if (listed.length) {
        render(listed);
      }
    }
  }

  function go(option: HTMLElement): void {
    const href = option.dataset.href;
    if (href) {
      window.location.assign(href);
    }
  }

  input.addEventListener("input", () => {
    window.clearTimeout(debounce);
    debounce = window.setTimeout(() => void refresh(), DEBOUNCE_MS);
  });
  input.addEventListener("focus", () => {
    void loadSearchIndex().catch(() => undefined);
    // The list of models with no diagram yet waits for the first keystroke (see refresh).
    if (input.value.trim() && list.childElementCount) {
      setOpen(true);
    }
  });
  input.addEventListener("keydown", (event) => {
    const count = options().length;
    if (event.key === "ArrowDown" && count) {
      event.preventDefault();
      setOpen(true);
      setActive(active + 1 >= count ? 0 : active + 1);
    } else if (event.key === "ArrowUp" && count) {
      event.preventDefault();
      setActive(active <= 0 ? count - 1 : active - 1);
    } else if (event.key === "Escape") {
      if (!popup.hidden) {
        event.preventDefault();
        setOpen(false);
      }
    } else if (event.key === "Enter" && active >= 0 && !popup.hidden) {
      event.preventDefault();
      go(options()[active]);
    }
  });
  // Pointer down, not click, so the choice lands before the input's blur closes the list.
  list.addEventListener("pointerdown", (event) => {
    const option = (event.target as Element).closest<HTMLElement>("[role='option']");
    if (option) {
      event.preventDefault();
      go(option);
    }
  });
  retry.addEventListener("click", () => {
    input.focus();
    void refresh();
  });
  form.addEventListener("focusout", (event) => {
    if (!form.contains(event.relatedTarget as Node | null)) {
      setOpen(false);
    }
  });
}
