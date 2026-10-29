/**
 * The Changes page: what changed on the site, newest first, kept by hand here, plus one entry for each
 * dataset release the export records. Add a line when a deploy changes what visitors see; plain words,
 * no internal names.
 */

export interface Change {
  /** ISO date, in Pacific time, of the day the change went live. */
  date: string;
  title: string;
  items: string[];
}

export const SITE_CHANGES: readonly Change[] = [
  {
    date: "2026-10-11",
    title: "Browsing and model pages",
    items: [
      "Filter models by task, family, size and more, and sort them; the address keeps your choices, so a filtered list can be shared.",
      "The search box suggests models as you type.",
      "Model pages lead with the facts, show related models and hide files that do not exist.",
      "New family pages group the models of one architecture family.",
    ],
  },
  {
    date: "2026-10-11",
    title: "About, dataset and licence pages",
    items: [
      "The About page explains how each diagram is traced, checked and drawn, and how to cite the Menagerie.",
      "The Dataset page lists every column of the download, with licence and citation.",
      "New pages for privacy and licences, and for this list of changes.",
      "Links shared in chat apps and social sites show a picture of a real diagram for each task.",
      "Downloaded catalog files now give each diagram's full web address.",
    ],
  },
  {
    date: "2026-10-10",
    title: "New look",
    items: [
      "A light paper look, with a dark mode that keeps diagrams on white paper so they match the downloads.",
      "The home page opens on one large diagram, with six well-known architectures beneath it.",
      "Models are listed as picture cards, with a List view for a compact table.",
      "The diagram viewer opens at a readable size at the top of the graph, with a button to see the whole graph.",
    ],
  },
  {
    date: "2026-06-23",
    title: "First catalog",
    items: ["The first catalog of diagrams, drawn with TorchLens, is published as an early release."],
  },
];
