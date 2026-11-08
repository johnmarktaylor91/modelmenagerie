/**
 * Which records the site shows, read from the record's flags (never from image sizes).
 *
 * - "shown": a validated model. It gets a card in the default listing and a page.
 * - "structure": the graph was captured but the forward pass was not checked. It keeps its page
 *   (with the structure-only banner) and is listed only when the reader asks for untraced entries.
 * - "hidden": no card and no page (a fabricated graph, a diagram that is not the model's own, or a
 *   record that did not pass).
 *
 * Records from an export without validation fields (the early preview data) stay "shown".
 * Browser-safe: no Node imports.
 */
import type { ModelRecord } from "@/types/record";

export type RecordTier = "shown" | "structure" | "hidden";

type TierFields = Pick<ModelRecord, "forward_validated" | "validation_state" | "capture" | "fidelity" | "own_diagram">;

/** The tier a record belongs to. */
export function recordTier(model: TierFields): RecordTier {
  if (model.fidelity === "fabrication" || model.own_diagram === false) {
    return "hidden";
  }
  if (model.capture === "structure_only" || model.validation_state === "structure_only") {
    return "structure";
  }
  if (model.validation_state === undefined) {
    return "shown";
  }
  return model.forward_validated && model.validation_state === "passed" ? "shown" : "hidden";
}

/** True when the record gets a page (shown or structure-only). */
export function hasPage(model: TierFields): boolean {
  return recordTier(model) !== "hidden";
}

/** True when the record belongs in the default listing. */
export function isListedByDefault(model: TierFields): boolean {
  return recordTier(model) === "shown";
}
