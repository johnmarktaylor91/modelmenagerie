"""torch-main canary: import every library the lock promises, then one forward and one trace."""

from __future__ import annotations

import importlib
import sys

PROMISED = (
    "torch", "torchvision", "torchlens", "timm", "transformers", "diffusers",
    "segmentation_models_pytorch", "ultralytics", "einops", "PIL", "numpy", "psutil", "yaml",
    "pydantic", "jsonschema", "referencing", "tqdm", "packaging",
)


def main() -> int:
    for module in PROMISED:
        importlib.import_module(module)
    import timm
    import torch
    import torchlens as tl

    assert sys.version_info[:2] == (3, 11), sys.version
    assert torch.__version__.startswith("2.6.") and torch.version.cuda is None, torch.__version__
    torch.manual_seed(0)
    model = timm.create_model("resnet18", pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert tuple(out.shape) == (1, 1000), out.shape
    trace = tl.trace(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()), torch.randn(2, 4))
    print(f"canary ok: python {sys.version.split()[0]}, torch {torch.__version__}, "
          f"torchlens {tl.__version__}, timm {timm.__version__}, resnet18 {tuple(out.shape)}, "
          f"tiny trace {getattr(trace, 'num_ops', None)} ops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
