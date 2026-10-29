import { DiagramViewer } from "./viewer";
import type { ViewerPayload, ViewerVariant } from "./viewer-types";

/** Read the variant data the page serialised at build time. */
function readPayload(): ViewerPayload | null {
  const node = document.getElementById("viewer-data");
  if (!node?.textContent) {
    return null;
  }
  return JSON.parse(node.textContent) as ViewerPayload;
}

/** Pick the variant named by ?v=, falling back to the model's default for missing or unknown keys. */
function variantFromUrl(payload: ViewerPayload): ViewerVariant {
  const key = new URLSearchParams(window.location.search).get("v");
  const byKey = (wanted: string | null) => payload.variants.find((variant) => variant.key === wanted);
  return byKey(key) ?? byKey(payload.defaultKey) ?? payload.variants[0];
}

/** Point every download link on the page at the selected variant's files. */
function syncDownloads(variant: ViewerVariant): void {
  document.querySelectorAll<HTMLAnchorElement>("[data-asset='svg']").forEach((link) => {
    link.href = variant.svgUrl;
  });
  document.querySelectorAll<HTMLAnchorElement>("[data-asset='svg-save']").forEach((link) => {
    link.href = variant.svgUrl;
    link.download = variant.svgFileName;
  });
  document.querySelectorAll<HTMLElement>("[data-asset-size='svg']").forEach((node) => {
    node.textContent = variant.svgSize;
  });
  document.querySelectorAll<HTMLElement>("[data-asset-group='pdf']").forEach((group) => {
    group.hidden = variant.pdfUrl === null;
  });
  document.querySelectorAll<HTMLAnchorElement>("[data-asset='pdf']").forEach((link) => {
    link.href = variant.pdfUrl ?? "";
  });
  document.querySelectorAll<HTMLElement>("[data-asset-size='pdf']").forEach((node) => {
    node.textContent = variant.pdfSize ?? "";
  });
}

/** Mark the selected variant button and keep the address bar in step without adding history entries. */
function syncVariantButtons(payload: ViewerPayload, variant: ViewerVariant): void {
  document.querySelectorAll<HTMLButtonElement>("[data-variant-key]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.variantKey === variant.key));
  });
  const url = new URL(window.location.href);
  if (variant.key === payload.defaultKey) {
    url.searchParams.delete("v");
  } else {
    url.searchParams.set("v", variant.key);
  }
  if (url.href !== window.location.href) {
    history.replaceState(history.state, "", url);
  }
}

/**
 * Save a cross-origin file under a readable name. Browsers ignore the download attribute on
 * cross-origin links, so fetch it as a blob; if the asset host does not allow that (no CORS),
 * open the file instead and say how to keep it.
 */
async function saveAsset(link: HTMLAnchorElement): Promise<void> {
  const note = document.querySelector<HTMLElement>("[data-save-note]");
  try {
    const response = await fetch(link.href, { mode: "cors" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const blobUrl = URL.createObjectURL(await response.blob());
    const anchor = document.createElement("a");
    anchor.href = blobUrl;
    anchor.download = link.download;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(blobUrl), 60_000);
  } catch {
    window.open(link.href, "_blank", "noopener");
    if (note) {
      note.hidden = false;
    }
  }
}

/** Wire the model page: viewer, variant buttons, downloads. */
function start(): void {
  const payload = readPayload();
  const root = document.querySelector<HTMLElement>("[data-viewer]");
  if (!payload || !root || payload.variants.length === 0) {
    return;
  }
  const viewer = new DiagramViewer(root, payload.modelName);
  const select = (variant: ViewerVariant) => {
    viewer.show(variant);
    syncDownloads(variant);
    syncVariantButtons(payload, variant);
  };
  document.querySelectorAll<HTMLButtonElement>("[data-variant-key]").forEach((button) => {
    button.addEventListener("click", () => {
      const variant = payload.variants.find((candidate) => candidate.key === button.dataset.variantKey);
      if (variant) {
        select(variant);
      }
    });
  });
  document.querySelectorAll<HTMLAnchorElement>("[data-asset='svg-save']").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      void saveAsset(link);
    });
  });
  window.addEventListener("popstate", () => select(variantFromUrl(payload)));
  select(variantFromUrl(payload));
}

start();
