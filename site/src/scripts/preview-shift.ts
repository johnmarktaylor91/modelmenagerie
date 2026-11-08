/**
 * Where a phone shows the top-corner crop of a heavy diagram before it loads. The crop is the
 * diagram's top-left corner at its own scale (800 px wide), and a phone stage shows only about 320
 * px of it, so the window is centred on the diagram's focus (where the drawing starts, the same
 * point the loaded view opens on), clamped to the crop. The result is the CSS length for
 * `--preview-shift` (half the crop width minus the focus, in px); null keeps the crop's left edge.
 */
export function previewShift(focusX: number | null, svgWidth: number | null, previewWidth: number | null): string | null {
  if (focusX === null || svgWidth === null || previewWidth === null) return null;
  const focusPx = Math.min(Math.max(focusX * svgWidth, 0), previewWidth);
  return `${Math.round(previewWidth / 2 - focusPx)}px`;
}
