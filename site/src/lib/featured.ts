/**
 * The hand-picked models the site shows off, in one file to edit. It holds only lists (no imports), so
 * build scripts can read it with plain Node as well as the site. The rules that apply these lists
 * (skip a slug that left the catalog, has a stand-in or blank picture, or is too heavy) are in
 * `featured-select.ts`; a slug that fails them is passed over silently, so order is preference.
 */

/** Home hero, first that qualifies: its full SVG is drawn at reading scale, so small, narrow graphs. */
export const HERO_SLUGS: readonly string[] = ["alexnet", "vgg16", "resnet18"];

/**
 * Home panels and the browse page's first cards, one per kind of structure (a residual network's
 * stem, a Transformer's encoder feeding its decoder layers, a U-Net skip connection, MobileNetV2's
 * inverted residual, a VGG stage, LeNet-5's convolutions), checked by eye on 2026-11-04 (the U-Net on
 * 2026-11-05: MONAI's BasicUNet, 5.7M parameters, replaced the 2.5K-parameter monai.UNet configuration). Each window
 * is at most about 490 diagram pixels wide, so node labels stay readable on a phone-width card.
 * QA fails on a number here that equals a published count; nudge a window by a pixel or two.
 * Each has a window in `FEATURED_VIEWS`; a slug that drops out is replaced by a fallback pick.
 */
export const FEATURED_SLUGS: readonly string[] = ["resnet18", "transformer-original", "monai-basicunet", "mobilenet-v2", "vgg16", "lenet-5"];

/**
 * The readable part of each featured diagram a card shows, in the SVG's own CSS pixels:
 * `svg` is the drawing's width and height when the window was chosen (a redrawn SVG of another
 * size falls back to the ordinary picture), `crop` the window's left edge, top edge and width.
 * The frame keeps its own aspect ratio, so the window's height follows the frame. About 430 to 800
 * px wide keeps node titles legible at card width. After a change, check the cards at 390 and
 * 1280 px (docs/site.md).
 */
export interface FeaturedView {
  svg: readonly [number, number];
  crop: readonly [number, number, number];
}

export const FEATURED_VIEWS: Readonly<Record<string, FeaturedView>> = {
  // The stem at the input end: 7 by 7 convolution with stride 2, then batch norm.
  resnet18: { svg: [447, 2344], crop: [10, 1820, 430] },
  // Two decoder layers, each fed by the encoder's output from the left.
  "transformer-original": { svg: [793, 3048], crop: [300, 1495, 470] },
  // A skip connection: the concatenation that joins the encoder path back in.
  "monai-basicunet": { svg: [1084, 8055], crop: [200, 4860, 470] },
  // An inverted residual block: its skip connection curves round the block into the add.
  "mobilenet-v2": { svg: [661, 2649], crop: [50, 1620, 470] },
  // Convolution, then max pooling: the repeated VGG stage.
  vgg16: { svg: [493, 6560], crop: [0, 4640, 493] },
  // The first convolution, tanh and average pooling, at the input end.
  "lenet-5": { svg: [453, 2315], crop: [12, 1690, 430] },
};

/**
 * Cleaner names for catalog entries whose own name is a package identifier or a bookkeeping
 * label. The page keeps the catalog's name in its Details ("Name in source").
 */
export const DISPLAY_NAMES: Readonly<Record<string, string>> = {
  "transformer-original": "Transformer",
  "monai-basicunet": "U-Net (MONAI BasicUNet)",
  "monai-unet": "Residual U-Net (MONAI)",
  "gpt-sw3": "GPT-SW3",
  "rnn-lm-elman": "Elman RNN language model",
  "openai-whisper-tiny": "Whisper tiny",
  googlenet: "GoogLeNet",
  "inception-v3": "Inception v3",
  "mobilenet-v2": "MobileNetV2",
  densenet121: "DenseNet-121",
};

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
