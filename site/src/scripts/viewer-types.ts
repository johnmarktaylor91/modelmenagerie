/** How a diagram is framed: legible scale from the top, or the whole graph fitted to the stage. */
export type ViewMode = "readable" | "whole";

/** One diagram variant as the browser needs it; built at page-render time from the catalog record. */
export interface ViewerVariant {
  key: string;
  label: string;
  svgUrl: string;
  svgSize: string;
  /** File name offered when the reader saves the SVG, for example "resnet18-unrolled-none.svg". */
  svgFileName: string;
  /** Small preview image, or null when the thumbnail is too small to be useful. */
  thumbUrl: string | null;
  pdfUrl: string | null;
  pdfSize: string | null;
  /** Load only on request (monster flag or byte size above the site-wide limit). */
  heavy: boolean;
  /** Sentence shown before a heavy diagram is loaded. */
  heavyNote: string;
  /** Sentence warning that this exact diagram is shared by many unrelated entries; empty when not. */
  standInNote: string;
  /**
   * Where the drawing starts across the top of a wide diagram, as a fraction of its width; the
   * readable view opens centred there. Null: open centred on the middle.
   */
  focusX: number | null;
}

/** Everything the model page script needs, serialised into the page as JSON. */
export interface ViewerPayload {
  modelName: string;
  defaultKey: string;
  variants: ViewerVariant[];
}
