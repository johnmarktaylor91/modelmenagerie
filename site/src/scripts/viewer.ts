import type { ViewMode, ViewerVariant } from "./viewer-types";
import { getViewerElements, type ViewerElements } from "./viewer-elements";

/** The view a diagram opens in. "readable" opens legible at the top; "whole" opens fitted (overview first). */
export const DEFAULT_VIEW_MODE: ViewMode = "readable";

/** Smallest scale the readable view drops to, so labels stay legible on wide graphs (Graphviz 14 pt text at about 11 px). */
const LEGIBLE_MIN_SCALE = 0.6;
const MAX_SCALE = 3;
/**
 * A graph wider than this at readable scale cannot be scanned by scrolling (the content is a needle
 * in a haystack of paper), so it opens on the whole graph, where a click zooms into the spot.
 */
const WIDE_GRAPH_READABLE_PX = 4000;
const ZOOM_STEP = 1.25;
const WHEEL_COMMIT_MS = 200;
const BUTTON_COMMIT_MS = 250;
const DRAG_THRESHOLD_PX = 4;
/** Thumbnails thinner than this, or more elongated than the aspect limit, are slivers and are not shown. */
const MIN_PREVIEW_SIDE_PX = 48;
const MAX_PREVIEW_ASPECT = 25;

type LoadState = "idle" | "loading" | "ready" | "error";

interface Fit {
  whole: number;
  readable: number;
  min: number;
}

interface WheelGesture {
  base: number;
  factor: number;
  clientX: number;
  clientY: number;
}

interface Drag {
  pointerId: number;
  x: number;
  y: number;
  left: number;
  top: number;
  moved: boolean;
}

/** Clamp a number into an inclusive range. */
function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Format a scale as a percentage of the diagram's natural size. */
function formatScale(scale: number): string {
  if (scale >= 0.1) {
    return `${Math.round(scale * 100)}%`;
  }
  return scale >= 0.01 ? `${(scale * 100).toFixed(1)}%` : "<1%";
}

/** Wait for an image element to load and decode; rejects when the request fails. */
async function loadImage(img: HTMLImageElement, url: string): Promise<void> {
  if (img.getAttribute("src") !== url || !img.complete || img.naturalWidth === 0) {
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error(`Image failed: ${url}`));
      if (img.getAttribute("src") === url && img.complete) {
        // A cached failure: reassigning the same URL would not fire onerror again.
        img.removeAttribute("src");
      }
      img.src = url;
    });
  }
  await img.decode().catch(() => undefined);
}

/** A native-scroll diagram viewer: zoom resizes the image, panning is ordinary scrolling. */
export class DiagramViewer {
  private readonly el: ViewerElements;
  private readonly modelName: string;
  private variant: ViewerVariant | null = null;
  private token = 0;
  private state: LoadState = "idle";
  private mode: ViewMode | "custom" = DEFAULT_VIEW_MODE;
  private natural = { width: 0, height: 0 };
  private fit: Fit = { whole: 1, readable: 1, min: 1 };
  private scale = 1;
  private gesture: WheelGesture | null = null;
  private gestureTimer = 0;
  private drag: Drag | null = null;
  private suppressClick = false;
  private expanded = false;

  constructor(root: HTMLElement, modelName: string) {
    this.el = getViewerElements(root);
    this.modelName = modelName;
    this.bindControls();
    this.bindStage();
    new ResizeObserver(() => this.onResize()).observe(this.el.stage);
  }

  /** Show a variant: preview first, then the full diagram (now, or on request when heavy). */
  show(variant: ViewerVariant): void {
    this.token += 1;
    this.variant = variant;
    this.cancelGesture();
    this.el.diagram.hidden = true;
    this.el.diagram.alt = `${this.modelName} architecture diagram`;
    this.el.loadButton.textContent = `Load full diagram (${variant.svgSize})`;
    this.el.standIn.textContent = variant.standInNote;
    this.el.standIn.hidden = variant.standInNote === "";
    this.showPreview(variant.thumbUrl, this.token);
    if (variant.heavy) {
      this.el.diagram.removeAttribute("src");
      this.setState("idle", variant.heavyNote);
    } else {
      void this.load();
    }
  }

  /** Fetch and display the full diagram for the current variant. */
  async load(): Promise<void> {
    const variant = this.variant;
    if (!variant) {
      return;
    }
    const token = this.token;
    this.setState("loading", `Loading diagram (${variant.svgSize})`);
    try {
      await loadImage(this.el.diagram, variant.svgUrl);
    } catch {
      if (token === this.token) {
        this.setState("error", this.errorMessage());
      }
      return;
    }
    if (token !== this.token) {
      return;
    }
    this.natural = {
      width: this.el.diagram.naturalWidth || this.el.stage.clientWidth,
      height: this.el.diagram.naturalHeight || this.el.stage.clientHeight,
    };
    this.el.diagram.hidden = false;
    this.el.preview.hidden = true;
    // Ready first: the stage's padding depends on the state, and the fit depends on the padding.
    this.setState("ready", "");
    this.computeFit();
    this.setMode(this.initialMode());
  }

  private errorMessage(): string {
    return this.el.preview.hidden
      ? `The diagram for ${this.modelName} did not load.`
      : "The full diagram did not load. The picture shown is a small preview.";
  }

  private showPreview(url: string | null, token: number): void {
    const preview = this.el.preview;
    preview.hidden = url === null;
    if (url === null) {
      preview.removeAttribute("src");
      return;
    }
    loadImage(preview, url)
      .then(() => {
        const { naturalWidth: w, naturalHeight: h } = preview;
        const sliver = Math.min(w, h) < MIN_PREVIEW_SIDE_PX || Math.max(w, h) / Math.max(1, Math.min(w, h)) > MAX_PREVIEW_ASPECT;
        if (token === this.token && (sliver || this.state === "ready")) {
          preview.hidden = true;
        }
      })
      .catch(() => {
        if (token === this.token) {
          preview.hidden = true;
          if (this.state === "error") {
            this.setState("error", this.errorMessage());
          }
        }
      });
  }

  private setState(state: LoadState, message: string): void {
    this.state = state;
    this.el.root.dataset.state = state;
    this.el.status.textContent = message;
    this.el.overlay.hidden = state === "ready";
    this.el.loadButton.hidden = state !== "idle";
    this.el.retryButton.hidden = state !== "error";
  }

  private initialMode(): ViewMode {
    const readableWidth = this.natural.width * this.fit.readable;
    return DEFAULT_VIEW_MODE === "readable" && readableWidth > WIDE_GRAPH_READABLE_PX ? "whole" : DEFAULT_VIEW_MODE;
  }

  private computeFit(): void {
    const { width, height } = this.innerSize();
    const fitWidth = width / this.natural.width;
    const whole = Math.min(fitWidth, height / this.natural.height, 1);
    const readable = Math.min(1, Math.max(fitWidth, LEGIBLE_MIN_SCALE));
    this.fit = { whole, readable, min: Math.min(whole, readable) };
  }

  private innerSize(): { width: number; height: number } {
    const style = getComputedStyle(this.el.canvas);
    const padX = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
    const padY = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
    return {
      width: Math.max(1, this.el.stage.clientWidth - padX),
      height: Math.max(1, this.el.stage.clientHeight - padY),
    };
  }

  private applyScale(scale: number): void {
    this.scale = clamp(scale, this.fit.min, MAX_SCALE);
    this.el.diagram.style.width = `${this.natural.width * this.scale}px`;
    this.el.diagram.style.height = `${this.natural.height * this.scale}px`;
    this.syncZoomControls(this.scale);
  }

  private syncZoomControls(scale: number): void {
    this.el.zoomReadout.textContent = formatScale(scale);
    // aria-disabled rather than disabled, so a keyboard user's focus stays on the button at the limit.
    this.el.zoomIn.setAttribute("aria-disabled", String(scale >= MAX_SCALE));
    this.el.zoomOut.setAttribute("aria-disabled", String(scale <= this.fit.min));
  }

  private setMode(mode: ViewMode): void {
    this.cancelGesture();
    this.mode = mode;
    this.applyScale(mode === "whole" ? this.fit.whole : this.fit.readable);
    const stage = this.el.stage;
    stage.scrollTop = 0;
    stage.scrollLeft = this.openingScrollLeft();
    this.syncModeButton();
  }

  /**
   * Horizontal scroll for a fresh view: centred on where the drawing starts across the top of the
   * graph (gpt2's middle is blank paper; its layers run down the left third), else the middle.
   */
  private openingScrollLeft(): number {
    const stage = this.el.stage;
    const overflow = stage.scrollWidth - stage.clientWidth;
    if (overflow <= 0) {
      return 0;
    }
    const focus = this.variant?.focusX;
    if (focus === null || focus === undefined) {
      return overflow / 2;
    }
    // The focus point in the stage's scroll coordinates, whatever the current scroll position.
    const image = this.el.diagram.getBoundingClientRect();
    const box = stage.getBoundingClientRect();
    const focusPx = image.left - box.left - stage.clientLeft + stage.scrollLeft + focus * image.width;
    return clamp(focusPx - stage.clientWidth / 2, 0, overflow);
  }

  private syncModeButton(): void {
    this.el.root.dataset.mode = this.mode;
    this.el.modeButton.textContent = this.mode === "whole" ? "Readable size" : "Whole graph";
  }

  /** Zoom to a scale keeping the diagram point under (clientX, clientY) in place. */
  private zoomAt(scale: number, clientX: number, clientY: number): void {
    const before = this.el.diagram.getBoundingClientRect();
    const u = (clientX - before.left) / Math.max(1, before.width);
    const v = (clientY - before.top) / Math.max(1, before.height);
    this.applyScale(scale);
    const after = this.el.diagram.getBoundingClientRect();
    this.el.stage.scrollLeft += after.left + u * after.width - clientX;
    this.el.stage.scrollTop += after.top + v * after.height - clientY;
    this.mode = Math.abs(this.scale - this.fit.whole) < 1e-6 ? "whole" : "custom";
    this.syncModeButton();
  }

  private zoomBy(factor: number): void {
    const rect = this.el.stage.getBoundingClientRect();
    const current = this.gesture ? this.gesture.base * this.gesture.factor : this.scale;
    this.previewZoom(current * factor, rect.left + rect.width / 2, rect.top + rect.height / 2, BUTTON_COMMIT_MS);
  }

  /**
   * Show a zoom at once as a compositor transform, and resize the image for real only when the
   * input pauses: re-rasterising a large SVG on every wheel tick or click is what makes zoom stutter.
   */
  private previewZoom(target: number, clientX: number, clientY: number, commitMs: number): void {
    if (!this.gesture) {
      const rect = this.el.diagram.getBoundingClientRect();
      const u = clamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
      const v = clamp((clientY - rect.top) / Math.max(1, rect.height), 0, 1);
      this.gesture = { base: this.scale, factor: 1, clientX, clientY };
      this.el.diagram.style.transformOrigin = `${u * 100}% ${v * 100}%`;
      this.el.diagram.style.willChange = "transform";
    }
    const scale = clamp(target, this.fit.min, MAX_SCALE);
    this.gesture.factor = scale / this.gesture.base;
    this.el.diagram.style.transform = `scale(${this.gesture.factor})`;
    this.syncZoomControls(scale);
    window.clearTimeout(this.gestureTimer);
    this.gestureTimer = window.setTimeout(() => this.commitGesture(), commitMs);
  }

  private toggleMode(): void {
    this.setMode(this.mode === "whole" ? "readable" : "whole");
  }

  private toggleExpanded(force?: boolean): void {
    this.expanded = force ?? !this.expanded;
    this.el.root.toggleAttribute("data-expanded", this.expanded);
    document.documentElement.classList.toggle("viewer-expanded", this.expanded);
    this.el.expandButton.textContent = this.expanded ? "Exit full screen" : "Full screen";
    this.el.stage.focus({ preventScroll: true });
  }

  private onResize(): void {
    if (this.state !== "ready") {
      return;
    }
    const stage = this.el.stage;
    const ratioTop = stage.scrollTop / Math.max(1, stage.scrollHeight);
    this.computeFit();
    if (this.mode === "custom") {
      this.applyScale(this.scale);
      return;
    }
    this.applyScale(this.mode === "whole" ? this.fit.whole : this.fit.readable);
    stage.scrollTop = ratioTop * stage.scrollHeight;
    stage.scrollLeft = this.openingScrollLeft();
  }

  private bindControls(): void {
    this.el.zoomIn.addEventListener("click", () => this.zoomBy(ZOOM_STEP));
    this.el.zoomOut.addEventListener("click", () => this.zoomBy(1 / ZOOM_STEP));
    this.el.modeButton.addEventListener("click", () => this.toggleMode());
    this.el.expandButton.addEventListener("click", () => this.toggleExpanded());
    this.el.loadButton.addEventListener("click", () => void this.load());
    this.el.retryButton.addEventListener("click", () => void this.load());
    this.el.root.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && this.expanded) {
        event.preventDefault();
        this.toggleExpanded(false);
      }
    });
  }

  private bindStage(): void {
    const stage = this.el.stage;
    stage.addEventListener("keydown", (event) => this.onKey(event));
    stage.addEventListener("wheel", (event) => this.onWheel(event), { passive: false });
    stage.addEventListener("pointerdown", (event) => this.onPointerDown(event));
    stage.addEventListener("pointermove", (event) => this.onPointerMove(event));
    stage.addEventListener("pointerup", (event) => this.onPointerUp(event));
    stage.addEventListener("pointercancel", (event) => this.onPointerUp(event));
    this.el.diagram.addEventListener("click", (event) => this.onDiagramClick(event));
  }

  private onKey(event: KeyboardEvent): void {
    if (this.state !== "ready" || event.ctrlKey || event.metaKey || event.altKey) {
      return;
    }
    const actions: Record<string, () => void> = {
      "+": () => this.zoomBy(ZOOM_STEP),
      "=": () => this.zoomBy(ZOOM_STEP),
      "-": () => this.zoomBy(1 / ZOOM_STEP),
      _: () => this.zoomBy(1 / ZOOM_STEP),
      "0": () => this.setMode("whole"),
      "1": () => this.setMode("readable"),
    };
    const action = actions[event.key];
    if (action) {
      event.preventDefault();
      action();
    }
  }

  /** Ctrl or Cmd plus wheel (and trackpad pinch) zooms; a plain wheel scrolls as usual. */
  private onWheel(event: WheelEvent): void {
    if (this.state !== "ready" || !(event.ctrlKey || event.metaKey)) {
      return;
    }
    event.preventDefault();
    const pixels = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY;
    const current = this.gesture ? this.gesture.base * this.gesture.factor : this.scale;
    const anchor = this.gesture ?? { clientX: event.clientX, clientY: event.clientY };
    this.previewZoom(current * Math.exp(-pixels * 0.0025), anchor.clientX, anchor.clientY, WHEEL_COMMIT_MS);
  }

  private commitGesture(): void {
    const gesture = this.gesture;
    this.cancelGesture();
    if (gesture) {
      this.zoomAt(gesture.base * gesture.factor, gesture.clientX, gesture.clientY);
    }
  }

  private cancelGesture(): void {
    window.clearTimeout(this.gestureTimer);
    this.gesture = null;
    this.el.diagram.style.transform = "";
    this.el.diagram.style.transformOrigin = "";
    this.el.diagram.style.willChange = "";
  }

  /** Mouse drag pans; touch keeps the browser's own scrolling and pinch. */
  private onPointerDown(event: PointerEvent): void {
    if (this.state !== "ready" || event.pointerType !== "mouse" || event.button !== 0) {
      return;
    }
    const stage = this.el.stage;
    this.drag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, left: stage.scrollLeft, top: stage.scrollTop, moved: false };
    this.suppressClick = false;
  }

  private onPointerMove(event: PointerEvent): void {
    const drag = this.drag;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) {
      return;
    }
    if (!drag.moved) {
      drag.moved = true;
      this.el.stage.setPointerCapture(event.pointerId);
      this.el.root.toggleAttribute("data-dragging", true);
    }
    event.preventDefault();
    this.el.stage.scrollLeft = drag.left - dx;
    this.el.stage.scrollTop = drag.top - dy;
  }

  private onPointerUp(event: PointerEvent): void {
    const drag = this.drag;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    this.suppressClick = drag.moved;
    this.drag = null;
    this.el.root.toggleAttribute("data-dragging", false);
    if (this.el.stage.hasPointerCapture(event.pointerId)) {
      this.el.stage.releasePointerCapture(event.pointerId);
    }
  }

  /** In the whole-graph view, clicking a spot opens the readable view at that spot. */
  private onDiagramClick(event: MouseEvent): void {
    if (this.suppressClick) {
      this.suppressClick = false;
      return;
    }
    if (this.state === "ready" && this.mode === "whole" && this.fit.readable > this.fit.whole) {
      this.zoomAt(this.fit.readable, event.clientX, event.clientY);
      this.mode = "readable";
      this.syncModeButton();
    }
  }
}
