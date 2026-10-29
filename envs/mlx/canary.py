"""mlx canary: the promised imports, then a tiny MLX module traced and validated natively on the
Linux CPU backend."""

from __future__ import annotations

import importlib
import sys

PROMISED = ("mlx.core", "mlx.nn", "torch", "torchlens", "numpy", "psutil", "yaml", "pydantic",
            "jsonschema", "referencing", "PIL")


def state(result: object) -> str:
    """A preview validation result's state: a bool is ``passed`` or ``failed``."""

    if isinstance(result, bool):
        return "passed" if result else "failed"
    return str(getattr(result, "state", result))


def main() -> int:
    for module in PROMISED:
        importlib.import_module(module)
    import mlx.core as mx
    import mlx.nn as nn
    import torch
    import torchlens as tl

    assert sys.version_info[:2] == (3, 11), sys.version
    assert torch.__version__.startswith("2.6.") and torch.version.cuda is None, torch.__version__
    mx.random.seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    x = mx.ones((1, 3))
    mx.eval(model(x))  # a never-run module's lazy parameters break capture at mlx 0.26.3
    trace = tl.trace(model, [x], backend="mlx")
    # The worker's validation: the live capture's replay (tl.validate cannot run a TF preview
    # at 2.35.3, and no preview takes random_seed).
    verdict = state(trace.validate_forward_pass([model(x)], validate_metadata=False))
    print(f"canary: python {sys.version.split()[0]}, mlx {mx.__version__}, torch "
          f"{torch.__version__}, torchlens {tl.__version__}, {trace.num_ops} ops, module mode "
          f"{trace.module_identity_mode}, validation {verdict}")
    assert verdict == "passed", verdict
    return 0


if __name__ == "__main__":
    sys.exit(main())
