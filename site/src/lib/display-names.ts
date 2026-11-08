/**
 * How the site names catalog entries and families when the export's own value is an identifier.
 * Display only: the export, the dataset download and each page's "Name in source" keep the raw value.
 * It holds lists and pure functions (no imports), so the browser bundle and plain Node can use it.
 */

/**
 * Families the export spells two ways, folded into one so a family is not listed twice ("mask_rcnn"
 * and "mask r-cnn"). GPT-2 on Hugging Face sits in "GPT" with one GPT-2-based decoder while seven
 * GPT-2 variants form "GPT-2"; they are one family. Key: the export's value; value: the family it joins.
 */
export const FAMILY_MERGES: Readonly<Record<string, string>> = {
  GPT: "GPT-2",
  mask_rcnn: "mask r-cnn",
  "cascade r-cnn": "cascade_rcnn",
  "regnet x": "regnetx",
  "regnet y": "regnety",
  "vision transformer": "vit",
  "wav2vec2/xlsr": "wav2vec 2.0",
  "voxpopuli wav2vec2": "wav2vec 2.0",
  mobilenet_v3: "mobilenetv3",
  "mobilenet v3": "mobilenetv3",
  "mobilenet-v2": "mobilenetv2",
  "swin v2": "swinv2",
};

/** Readable names for families whose export name is an identifier ("yolo_detect"). They win over families.jsonl. */
export const FAMILY_DISPLAY_NAMES: Readonly<Record<string, string>> = {
  aimv2: "AIMv2",
  base_edit_model: "MMagic image restoration",
  cascade_rcnn: "Cascade R-CNN",
  coat: "CoaT",
  "data2vec-audio": "data2vec (audio)",
  deep_rl_cnn: "Deep RL convolutional networks",
  "depth-anything-dpt": "Depth Anything",
  edgenext: "EdgeNeXt",
  effdet_cspresdet: "EfficientDet with CSP backbones",
  efficientdet_tf: "EfficientDet (TensorFlow variants)",
  efficientformer: "EfficientFormer",
  ese_vovnet: "ESE-VoVNet",
  faster_rcnn: "Faster R-CNN",
  fasternet: "FasterNet",
  "fc-gan": "Fully connected GANs",
  fno: "Fourier Neural Operator",
  "fpn-segmentation": "FPN segmentation",
  hardcorenas: "HardCoRe-NAS",
  hdemucs: "Hybrid Demucs",
  hgnetv2: "HGNetV2",
  hiera: "Hiera",
  "kg-embedding": "Knowledge graph embeddings",
  lcnet: "LCNet",
  "learned-compression": "Learned image compression",
  mambaout: "MambaOut",
  "mask r-cnn": "Mask R-CNN",
  mobileone: "MobileOne",
  moe: "Mixture of experts",
  naflexvit: "NaFlexViT",
  nest: "NesT",
  nf_resnet_regnet: "NF-ResNet and NF-RegNet",
  ocr: "OCR models",
  "pan-segmentation": "PAN segmentation",
  parametric_lif_net: "Parametric LIF spiking networks",
  poolformerv2: "PoolFormer V2",
  rdn: "RDN",
  regnetx: "RegNetX",
  regnety: "RegNetY",
  repghostnet: "RepGhostNet",
  resnetrs: "ResNet-RS",
  "retrieval-encoder": "Retrieval encoders",
  rexnet: "ReXNet",
  rexnetr: "ReXNet-R",
  sb3_policy: "Stable-Baselines3 policies",
  "segformer-decoder": "SegFormer decoder",
  selecsls: "SelecSLS",
  "shufflenet v2": "ShuffleNet V2",
  "ssl-head": "Self-supervised learning heads",
  starnet: "StarNet",
  swiftformer: "SwiftFormer",
  timm_test: "timm test models",
  tiny_vit: "TinyViT",
  tinynet: "TinyNet",
  transformer: "Transformer",
  transgan: "TransGAN",
  twins: "Twins",
  upernet: "UPerNet",
  vit: "ViT",
  vitamin: "ViTamin",
  whisper: "Whisper",
  "yolo-nas": "YOLO-NAS",
  "yolo-nas-pose": "YOLO-NAS Pose",
  yolo12: "YOLO12",
  yolo_classify: "YOLO classification",
  yolo_detect: "YOLO detection",
  yolo_obb: "YOLO oriented boxes",
  yolo_pose: "YOLO pose",
  yolo_segment: "YOLO segmentation",
  yolo_semantic: "YOLO semantic segmentation",
};

/** The family a model is listed under: the export's value, folded by `FAMILY_MERGES`. */
export function canonicalFamily<T extends string | null | undefined>(family: T): T | string {
  return typeof family === "string" && Object.prototype.hasOwnProperty.call(FAMILY_MERGES, family) ? FAMILY_MERGES[family] : family;
}

/** The site's own readable name for a family, or null to use families.jsonl or the raw value. */
export function familyDisplayOverride(family: string | null | undefined): string | null {
  return typeof family === "string" && Object.prototype.hasOwnProperty.call(FAMILY_DISPLAY_NAMES, family) ? FAMILY_DISPLAY_NAMES[family] : null;
}

/** Package and toolbox prefixes, written as their projects write them. */
const LIBRARY_PREFIXES: Readonly<Record<string, string>> = {
  mmaction: "MMAction2",
  mmagic: "MMagic",
  mmdet: "MMDetection",
  mmdet3d: "MMDetection3D",
  mmocr: "MMOCR",
  mmpose: "MMPose",
  mmpretrain: "MMPretrain",
  mmrotate: "MMRotate",
  mmseg: "MMSegmentation",
  mmtrack: "MMTracking",
  midas: "MiDaS",
};

const ACRONYMS: Readonly<Record<string, string>> = { asr: "ASR", ctc: "CTC", ocr: "OCR", tts: "TTS", vqa: "VQA", xvector: "x-vector" };

/** One "prefix:rest" identifier with no spaces around the colon and no address in it. */
const PREFIXED_NAME = /^([A-Za-z0-9][\w.-]*):([A-Za-z0-9][^:\s/]*)$/;
/** A task or size suffix: lowercase words joined by hyphens, no digits ("visual-question-answering", "large"). */
const TASK_SUFFIX = /^[a-z]+(?:-[a-z]+)*$/;

/**
 * A catalog name of the form "zoo:name" or "model:task" written for people:
 * "mmpretrain:convnext" becomes "convnext (MMPretrain)", "blip:visual-question-answering" becomes
 * "blip (visual question answering)", "LongSafari:hyenadna-small-32k-seqlen-hf" becomes
 * "hyenadna-small-32k-seqlen-hf (LongSafari)". Any other name is returned unchanged.
 */
export function cleanModelName(name: string): string {
  const match = PREFIXED_NAME.exec(name);
  if (!match) {
    return name;
  }
  const [, prefix, rest] = match;
  const library = LIBRARY_PREFIXES[prefix.toLowerCase()];
  if (library) {
    return `${rest} (${library})`;
  }
  if (TASK_SUFFIX.test(rest)) {
    return `${prefix} (${rest.split("-").map((word) => ACRONYMS[word] ?? word).join(" ")})`;
  }
  return rest.toLowerCase().includes(prefix.toLowerCase()) ? rest : `${rest} (${prefix})`;
}
