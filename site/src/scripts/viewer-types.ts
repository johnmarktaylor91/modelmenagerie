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
  /** Preview image (the top crop when the export has one, else the thumbnail), or null when there is no useful one. */
  thumbUrl: string | null;
  /** The preview's own size when known (a crop's width and height), so it reserves its box before it loads. */
  previewWidth: number | null;
  previewHeight: number | null;
  /** The SVG's own size from the export, or null; used to size the stage and the scale before the SVG loads. */
  svgWidth: number | null;
  svgHeight: number | null;
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
  /**
   * Where the readable view of a wide diagram starts down the page, as a fraction of its height,
   * when the top of the window it opens on is blank paper. Null: open at the top.
   */
  focusY: number | null;
}

/** Everything the model page script needs, serialised into the page as JSON. */
export interface ViewerPayload {
  modelName: string;
  defaultKey: string;
  variants: ViewerVariant[];
}
