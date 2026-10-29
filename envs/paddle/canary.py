"""paddle canary: the promised imports, then a tiny dygraph Layer traced and validated natively."""

from __future__ import annotations

import importlib
import sys

PROMISED = ("paddle", "torch", "torchlens", "numpy", "psutil", "yaml", "pydantic", "jsonschema",
            "referencing", "PIL")


def state(result: object) -> str:
    """A preview validation result's state: a bool is ``passed`` or ``failed``."""

    if isinstance(result, bool):
        return "passed" if result else "failed"
    return str(getattr(result, "state", result))


def main() -> int:
    for module in PROMISED:
        importlib.import_module(module)
    import paddle
    import torch
    import torchlens as tl

    assert sys.version_info[:2] == (3, 11), sys.version
    assert torch.__version__.startswith("2.6.") and torch.version.cuda is None, torch.__version__
    assert paddle.__version__.startswith("3.3."), paddle.__version__
    assert not paddle.device.is_compiled_with_cuda(), "the paddle lock must stay CPU-only"
    paddle.seed(0)
    model = paddle.nn.Sequential(paddle.nn.Linear(3, 4), paddle.nn.ReLU(), paddle.nn.Linear(4, 2))
    model.eval()
    x = paddle.ones([1, 3], dtype="float32")
    trace = tl.trace(model, [x], backend="paddle")
    # The worker's validation: the live capture's replay (tl.validate cannot run a TF preview
    # at 2.35.3, and no preview takes random_seed).
    verdict = state(trace.validate_forward_pass([model(x)], validate_metadata=True))
    fn_trace = tl.trace(lambda v: paddle.nn.functional.relu(v) * 2.0, [x], backend="paddle")
    fn_verdict = state(fn_trace.validate_forward_pass([], validate_metadata=True))
    print(f"canary: python {sys.version.split()[0]}, paddle {paddle.__version__}, torch "
          f"{torch.__version__}, torchlens {tl.__version__}, {trace.num_ops} ops, module mode "
          f"{trace.module_identity_mode}, validation {verdict}; function {fn_trace.num_ops} ops, "
          f"module mode {fn_trace.module_identity_mode}, validation {fn_verdict}")
    # At TorchLens 2.35.3 the Paddle coverage oracle fails any op that reads a layer's
    # parameters (functional.linear), so a Layer validates `failed`; the environment and the
    # preview are healthy when the capture runs and a parameter-free function passes.
    assert trace.num_ops > 0 and fn_verdict == "passed", (verdict, fn_verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
