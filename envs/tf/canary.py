"""tf canary: the promised imports, then a tiny Keras 3 model traced and validated natively."""

from __future__ import annotations

import importlib
import sys

PROMISED = ("tensorflow", "keras", "torch", "torchlens", "numpy", "psutil", "yaml", "pydantic",
            "jsonschema", "referencing", "PIL")


def state(result: object) -> str:
    """A preview validation result's state: a bool is ``passed`` or ``failed``."""

    if isinstance(result, bool):
        return "passed" if result else "failed"
    return str(getattr(result, "state", result))


def main() -> int:
    for module in PROMISED:
        importlib.import_module(module)
    import keras
    import tensorflow as tf
    import torch
    import torchlens as tl

    assert sys.version_info[:2] == (3, 11), sys.version
    assert torch.__version__.startswith("2.6.") and torch.version.cuda is None, torch.__version__
    assert int(tf.__version__.split(".")[1]) >= 16 and keras.__version__.startswith("3."), (
        tf.__version__, keras.__version__)
    assert keras.backend.backend() == "tensorflow", keras.backend.backend()
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(4, activation="relu"),
                              keras.layers.Dense(2)])
    x = tf.ones((1, 3), dtype=tf.float32)
    trace = tl.trace(model, [x], backend="tf")
    # The worker's validation: the live capture's replay (tl.validate cannot run a TF preview
    # at 2.35.3, and no preview takes random_seed).
    verdict = state(trace.validate_forward_pass([model(x)], validate_metadata=True))
    status = trace.validation_replay_status
    w, b = tf.constant([[1.0, -1.0], [0.5, 2.0], [-1.5, 0.25]]), tf.constant([0.1, -0.2])
    fn_trace = tl.trace(lambda v: tf.nn.relu(tf.matmul(v, w) + b), [x], backend="tf")
    fn_verdict = state(fn_trace.validate_forward_pass([], validate_metadata=True))
    print(f"canary: python {sys.version.split()[0]}, tensorflow {tf.__version__}, keras "
          f"{keras.__version__}, torch {torch.__version__}, torchlens {tl.__version__}, "
          f"{trace.num_ops} ops, module mode {trace.module_identity_mode}, validation {verdict} "
          f"({status.replayed_node_count} replayed, {status.effect_region_node_count} effect "
          f"regions, {status.failed_node_count} failed); function {fn_trace.num_ops} ops, "
          f"module mode {fn_trace.module_identity_mode}, validation {fn_verdict}")
    # At TorchLens 2.35.3 every variable read (ReadVariableOp) is an unverified effect region, so
    # a Keras model is `unverified` at best: the environment is healthy when nothing fails.
    assert verdict in ("passed", "unverified") and status.failed_node_count == 0, verdict
    assert status.replayed_node_count > 0 and fn_verdict in ("passed", "unverified"), fn_verdict
    return 0


if __name__ == "__main__":
    sys.exit(main())
