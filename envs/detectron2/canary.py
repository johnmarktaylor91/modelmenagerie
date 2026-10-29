"""Detectron2 CPU compiled operation and default model-zoo inference. No weight download."""

import os

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torchlens  # noqa: F401  # Import compatibility is part of the canary.
from detectron2 import model_zoo
from detectron2.layers import nms_rotated
from detectron2.modeling import build_model

torch.set_num_threads(1)
torch.manual_seed(0)
assert nms_rotated(torch.tensor([[5., 5., 10., 10., 0.]]), torch.tensor([.9]), .5).numel() == 1
config = model_zoo.get_config('COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml')
config.MODEL.DEVICE = 'cpu'
config.MODEL.WEIGHTS = ''
# Input-sized task settings; backbone/head structure stays the model-zoo reference.
config.INPUT.MIN_SIZE_TEST = 64
config.INPUT.MAX_SIZE_TEST = 64
model = build_model(config).eval()
with torch.no_grad():
    output = model([{'image': torch.randn(3, 64, 64), 'height': 64, 'width': 64}])
assert 'instances' in output[0]
print('detectron2 canary: imports, rotated CPU NMS, model-zoo forward passed')
