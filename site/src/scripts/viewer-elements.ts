/** The DOM parts of one diagram viewer, found by their data-viewer attributes. */
export interface ViewerElements {
  root: HTMLElement;
  stage: HTMLElement;
  canvas: HTMLElement;
  diagram: HTMLImageElement;
  preview: HTMLImageElement;
  overlay: HTMLElement;
  status: HTMLElement;
  standIn: HTMLElement;
  loadButton: HTMLButtonElement;
  retryButton: HTMLButtonElement;
  zoomIn: HTMLButtonElement;
  zoomOut: HTMLButtonElement;
  zoomReadout: HTMLElement;
  modeButton: HTMLButtonElement;
  expandButton: HTMLButtonElement;
}

/** Find one required element inside the viewer; a missing part is a markup bug, so fail loudly. */
function part<T extends HTMLElement>(root: HTMLElement, name: string): T {
  const found = root.querySelector<T>(`[data-viewer-part="${name}"]`);
  if (!found) {
    throw new Error(`Diagram viewer markup is missing [data-viewer-part="${name}"]`);
  }
  return found;
}

/** Collect every part the viewer controller drives. */
export function getViewerElements(root: HTMLElement): ViewerElements {
  return {
    root,
    stage: part(root, "stage"),
    canvas: part(root, "canvas"),
    diagram: part<HTMLImageElement>(root, "diagram"),
    preview: part<HTMLImageElement>(root, "preview"),
    overlay: part(root, "overlay"),
    status: part(root, "status"),
    standIn: part(root, "stand-in"),
    loadButton: part<HTMLButtonElement>(root, "load"),
    retryButton: part<HTMLButtonElement>(root, "retry"),
    zoomIn: part<HTMLButtonElement>(root, "zoom-in"),
    zoomOut: part<HTMLButtonElement>(root, "zoom-out"),
    zoomReadout: part(root, "zoom-readout"),
    modeButton: part<HTMLButtonElement>(root, "mode"),
    expandButton: part<HTMLButtonElement>(root, "expand"),
  };
}
