/**
 * Alt text for diagram pictures, one pattern site-wide: "<model name>, <unrolled|rolled> computational
 * graph", plus the layer grouping when it is not the full drawing ("ResNet-18, unrolled computational
 * graph, grouped layers"). A picture inside a link or card that already names the model gets alt=""
 * instead, so a screen reader does not say the name twice.
 */
interface VariantLike {
  vis_mode: string;
  collapse: string;
}

const GROUPING: Record<string, string> = { auto: "grouped layers", max: "collapsed layers" };

export function diagramAlt(name: string, variant: VariantLike | null | undefined): string {
  const mode = variant?.vis_mode === "rolled" ? "rolled" : "unrolled";
  const grouping = variant ? GROUPING[variant.collapse] : undefined;
  return `${name}, ${mode} computational graph${grouping ? `, ${grouping}` : ""}`;
}
