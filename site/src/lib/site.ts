/**
 * Facts about the site itself, used by the About, Dataset, Legal and Changelog pages, the footer and the
 * structured data: names, addresses, licences, citations and contacts. Edit them here, once.
 */

import { CONTACT_EMAIL } from "@/lib/contact";

export const SITE_NAME = "TorchLens Model Menagerie";
export const SITE_URL = "https://modelmenagerie.ai";

export const TORCHLENS = {
  repo: "https://github.com/johnmarktaylor91/torchlens",
  issues: "https://github.com/johnmarktaylor91/torchlens/issues",
  pypi: "https://pypi.org/project/torchlens/",
  readme: "https://github.com/johnmarktaylor91/torchlens#readme",
  backends: "https://github.com/johnmarktaylor91/torchlens/blob/main/docs/backends.md",
  paper: "https://www.nature.com/articles/s41598-023-40807-0",
  doi: "https://doi.org/10.1038/s41598-023-40807-0",
  licence: { name: "Apache License 2.0", spdx: "Apache-2.0", url: "https://www.apache.org/licenses/LICENSE-2.0" },
} as const;

export const GRAPHVIZ_URL = "https://graphviz.org/";

/** Licence of the site's own text, diagrams and dataset (decisions.md, 2026-10-03, item 9). */
export const CONTENT_LICENCE = {
  name: "Creative Commons Attribution 4.0 International",
  short: "CC BY 4.0",
  spdx: "CC-BY-4.0",
  url: "https://creativecommons.org/licenses/by/4.0/",
} as const;

export const CREATOR = { name: "JohnMark Taylor", url: "https://github.com/johnmarktaylor91" } as const;

/** Where corrections go (public), and where requests that should not be public go. */
export const CONTACT = {
  issues: TORCHLENS.issues,
  email: CONTACT_EMAIL,
} as const;

export const TORCHLENS_CITATION =
  "Taylor, J., Kriegeskorte, N. Extracting and visualizing hidden activations and computational graphs of PyTorch models with TorchLens. Sci Rep 13, 14375 (2023). https://doi.org/10.1038/s41598-023-40807-0";

export const TORCHLENS_BIBTEX = `@article{taylor2023torchlens,
  author  = {Taylor, JohnMark and Kriegeskorte, Nikolaus},
  title   = {Extracting and visualizing hidden activations and computational graphs of {PyTorch} models with {TorchLens}},
  journal = {Scientific Reports},
  volume  = {13},
  pages   = {14375},
  year    = {2023},
  doi     = {10.1038/s41598-023-40807-0}
}`;

/** Citation for the site and its dataset; `year` and `version` come from the release being cited. */
export function siteCitation(year: number, version: string | null): string {
  const release = version ? `, release ${version}` : "";
  return `Taylor, J. TorchLens Model Menagerie${release} (${year}). ${SITE_URL}/`;
}

export function siteBibtex(year: number, version: string | null): string {
  const lines = [
    `@misc{menagerie${year},`,
    `  author       = {Taylor, JohnMark},`,
    `  title        = {{TorchLens Model Menagerie}: computational graphs of neural networks},`,
    `  year         = {${year}},`,
    ...(version ? [`  version      = {${version}},`] : []),
    `  howpublished = {\\url{${SITE_URL}/}},`,
    `  note         = {Dataset licensed under CC BY 4.0}`,
    `}`,
  ];
  return lines.join("\n");
}

/** A release version worth printing: numbered releases only, never internal build names. */
export function releaseVersion(exportVersion: string | null | undefined): string | null {
  if (!exportVersion || !/^v?\d/.test(exportVersion)) {
    return null;
  }
  return exportVersion.startsWith("v") ? exportVersion : `v${exportVersion}`;
}
