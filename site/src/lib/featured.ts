/**
 * The hand-picked models the site shows off, in one file to edit. It holds only lists (no imports), so
 * build scripts can read it with plain Node as well as the site. The rules that apply these lists
 * (skip a slug that left the catalog, has a stand-in or blank picture, or is too heavy) are in
 * `featured-select.ts`; a slug that fails them is passed over silently, so order is preference.
 */

/** Home hero, first that qualifies: its full SVG is drawn at reading scale, so small, narrow graphs. */
export const HERO_SLUGS: readonly string[] = ["alexnet", "vgg16", "resnet18"];

/**
 * Home panels, one per kind of model (language, speech, detection, generative, 3D points, early
 * vision), checked by eye on 2026-10-10. Narrow graphs come first: their tops read as diagrams at
 * panel width, and their pictures are small, so the home page stays light. The rest are fallbacks.
 */
export const FEATURED_SLUGS: readonly string[] = [
  "gpt2",
  "wavenet",
  "tiny-yolov3",
  "dcgan-original",
  "pointnet-classification",
  "lenet-5",
  "mcculloch-pitts-threshold-logic-net",
  "monai-unet",
  "openai-whisper-tiny",
  "vgg16",
  "resnet18",
  "alexnet",
];

/**
 * The diagram pictured in each task's share image (the preview a link shows in chat apps and social
 * sites), by category. Model pages share their task's image until each model has its own. After an
 * edit here, rerun `scripts/share_crops.mjs` on a worker to redraw the pictures (docs/site.md).
 */
export const SHARE_IMAGE_SLUGS: Readonly<Record<string, string>> = {
  default: "alexnet",
  "3D_geometry_NeRF_point-cloud": "pointnet-classification",
  NLP_LLM_text: "gpt2",
  OCR_document: "craft-1303",
  RL_robotics_control: "muzero-resnet",
  audio_speech: "wavenet",
  exotic_other: "capsnet",
  "generative_diffusion-GAN-flow": "dcgan-original",
  graph_geometric: "chebconv",
  history: "mcculloch-pitts-threshold-logic-net",
  "medical-imaging": "monai-unet",
  multimodal: "show-attend-tell",
  "neuro_scientific-signals": "cornet-z",
  recsys_tabular: "deepctr-deepfm",
  scientific_molecular_protein_genomics: "chemprop-d-mpnn",
  scientific_physics_weather_PDE: "fno-2d",
  spiking_neuromorphic: "norse-lif-convnet",
  "time-series_dynamics": "darts-n-beats",
  video_action: "c3d",
  "vision_classification-backbone": "lenet-5",
  "vision_depth-geometry": "manydepth-resnet18",
  "vision_detection-tracking": "tiny-yolov3",
  vision_pose: "openpose-body25",
  vision_segmentation: "cellpose",
};
