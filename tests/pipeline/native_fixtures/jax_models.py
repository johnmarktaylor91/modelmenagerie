"""Small JAX models for the worker's JAX preview tests (test fixtures, no real names): an Equinox
MLP, a Flax NNX block with dropout and a Flax Linen pair (module, variables); and two builds that
break the trace contract: a wrapper object around a Linen pair, and a function closing over its
weights."""

from __future__ import annotations

import jax
import jax.numpy as jnp

VARIANTS = {"m900011": {"model": "eqx_mlp"}, "m900012": {"model": "nnx_dropout"},
            "m900013": {"model": "linen_pair"}}


def build(cfg: dict):
    kind = cfg.get("model", "eqx_mlp")
    if kind == "nnx_dropout":
        from flax import nnx

        class Block(nnx.Module):
            def __init__(self, rngs: nnx.Rngs) -> None:
                self.lin1 = nnx.Linear(8, 16, rngs=rngs)
                self.drop = nnx.Dropout(0.5, rngs=rngs)
                self.lin2 = nnx.Linear(16, 4, rngs=rngs)

            def __call__(self, x: jax.Array) -> jax.Array:
                return self.lin2(self.drop(jax.nn.relu(self.lin1(x))))

        return Block(nnx.Rngs(0))
    if kind == "closed_constants":
        weight = jax.random.normal(jax.random.PRNGKey(0), (8, 4))
        return lambda x: jnp.tanh(x @ weight)  # the weights are a closure, not a parameter
    if kind == "wrapper":
        module, variables = build({"model": "linen_pair"})

        class Policy:  # holds the pair and is called like a model: not what TorchLens takes
            def __call__(self, x: jax.Array) -> jax.Array:
                return module.apply(variables, x)

        return Policy()
    if kind == "linen_pair":
        import flax.linen as nn

        class Dense2(nn.Module):
            @nn.compact
            def __call__(self, x: jax.Array) -> jax.Array:
                return nn.Dense(4)(nn.relu(nn.Dense(16)(x)))

        module = Dense2()
        return module, module.init(jax.random.PRNGKey(0), jnp.ones((1, 8)))
    import equinox as eqx

    return eqx.nn.MLP(in_size=8, out_size=4, width_size=16, depth=2, key=jax.random.PRNGKey(0))
