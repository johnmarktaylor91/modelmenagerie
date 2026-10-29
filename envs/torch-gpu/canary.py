"""torch-gpu canary: the promised imports and a CUDA 12.4 build of torch, a CPU forward and trace;
with MENAGERIE_GPU_CANARY=1 (inside the GPU window only) the same forward and trace on the GPU."""

from __future__ import annotations

import importlib
import os
import sys

PROMISED = (
    "torch", "torchvision", "torchlens", "timm", "transformers", "diffusers",
    "segmentation_models_pytorch", "ultralytics", "einops", "PIL", "numpy", "psutil", "yaml",
    "pydantic", "jsonschema", "referencing", "tqdm", "packaging", "setuptools", "ninja",
)


def main() -> int:
    for module in PROMISED:
        importlib.import_module(module)
    import timm
    import torch
    import torchlens as tl

    assert sys.version_info[:2] == (3, 11), sys.version
    assert torch.__version__.startswith("2.6.") and torch.version.cuda == "12.4", (
        torch.__version__, torch.version.cuda)
    devices = ["cpu"] + (["cuda"] if os.environ.get("MENAGERIE_GPU_CANARY") == "1" else [])
    for device in devices:
        torch.manual_seed(0)
        model = timm.create_model("resnet18", pretrained=False).eval().to(device)
        with torch.no_grad():
            out = model(torch.randn(1, 3, 224, 224, device=device))
        assert tuple(out.shape) == (1, 1000), out.shape
        tiny = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()).to(device)
        trace = tl.trace(tiny, torch.randn(2, 4, device=device))
        name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
        print(f"canary {device} ({name}): python {sys.version.split()[0]}, torch "
              f"{torch.__version__} cuda {torch.version.cuda}, torchlens {tl.__version__}, "
              f"resnet18 {tuple(out.shape)}, tiny trace {trace.num_ops} ops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
