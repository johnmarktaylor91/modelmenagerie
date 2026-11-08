/**
 * Preview or launch. `PUBLIC_SITE_STAGE=launch` drops the preview tag and the preview sentence
 * and shows the dataset version from the manifest instead; unset (or anything else) keeps the
 * preview label. See docs/site.md, "Launch flag".
 */
import { getManifest } from "@/lib/catalog";
import { releaseVersion } from "@/lib/site";

/** True when the build is the launched site rather than the preview. */
export function isLaunched(): boolean {
  return (import.meta.env.PUBLIC_SITE_STAGE as string | undefined)?.trim().toLowerCase() === "launch";
}

/** The dataset version shown at launch ("v2026-11-02"), or null when the manifest has none. */
export function datasetVersion(): string | null {
  return releaseVersion(getManifest().export_version);
}
