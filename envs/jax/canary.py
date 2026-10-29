"""jax canary: the promised imports, then an Equinox, a Flax NNX and a raw-function model traced
and validated natively (``pytree_module`` twice, ``function_root`` once)."""

from __future__ import annotations

import importlib
import sys

PROMISED = ("jax", "jaxlib", "flax", "equinox", "torch", "torchlens", "numpy", "psutil", "yaml",
            "pydantic", "jsonschema", "referencing", "PIL")


def state(result: object) -> str:
    """A preview validation result's state: a bool is ``passed`` or ``failed``."""

    if isinstance(result, bool):
        return "passed" if result else "failed"
    return str(getattr(result, "state", result))


def check(tl: object, name: str, model: object, args: list) -> str:
    trace = tl.trace(model, args, backend="jax")
    # The worker's validation: the live capture's replay (no preview takes random_seed).
    verdict = state(trace.validate_forward_pass([model(*args)], validate_metadata=True))
    print(f"  {name}: {trace.num_ops} ops, module mode {trace.module_identity_mode}, "
          f"validation {verdict}")
    return verdict


def main() -> int:
    for module in PROMISED:
        importlib.import_module(module)
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import torch
    import torchlens as tl
    from flax import nnx

    assert sys.version_info[:2] == (3, 11), sys.version
    assert torch.__version__.startswith("2.6.") and torch.version.cuda is None, torch.__version__
    assert jax.default_backend() == "cpu", jax.default_backend()
    key = jax.random.PRNGKey(0)
    x = jnp.ones((2, 3), dtype=jnp.float32)
    mlp = eqx.nn.MLP(in_size=3, out_size=2, width_size=4, depth=1, key=key)
    linear = nnx.Linear(3, 2, rngs=nnx.Rngs(0))
    params = {"w": jax.random.normal(key, (3, 2)), "b": jnp.zeros((2,))}

    def raw(p: dict, v: jax.Array) -> jax.Array:
        return jnp.tanh(v @ p["w"] + p["b"])

    print(f"canary: python {sys.version.split()[0]}, jax {jax.__version__}, torch "
          f"{torch.__version__}, torchlens {tl.__version__}")
    verdicts = [check(tl, "equinox MLP", mlp, [x[0]]),
                check(tl, "flax nnx Linear", linear, [x]),
                check(tl, "raw function", raw, [params, x])]
    assert verdicts == ["passed"] * 3, verdicts
    return 0


if __name__ == "__main__":
    sys.exit(main())
