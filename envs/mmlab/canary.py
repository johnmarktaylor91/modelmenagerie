"""MMLab CPU imports, compiled operation and default ResNet-18 forward. No weights fetched."""

import os

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import mmcv  # noqa: F401  # Import compatibility is part of the canary.
import mmdet  # noqa: F401  # Import compatibility is part of the canary.
import mmseg  # noqa: F401  # Import compatibility is part of the canary.
import torchlens  # noqa: F401  # Import compatibility is part of the canary.
from mmcv.ops import nms
from mmdet.models.backbones import ResNet

torch.set_num_threads(1)
torch.manual_seed(0)
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]])
scores = torch.tensor([.9, .8])
assert len(nms(boxes, scores, .5)[1]) == 1
model = ResNet(depth=18)
model.eval()  # MMDetection's ResNet.train() returns None, so eval() cannot be chained.
with torch.no_grad():
    output = model(torch.randn(1, 3, 64, 64))
assert output[-1].shape == (1, 512, 2, 2)
print('mmlab canary: imports, CPU NMS, ResNet-18 forward passed')
