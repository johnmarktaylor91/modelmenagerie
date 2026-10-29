"""Inputs for verified library models. Dimensions come from their actual configuration.

These builders change the input boundary only, never the model's forward method.
Unsupported interfaces stay needs_adapter:inputs instead of receiving a sentinel.
"""

from __future__ import annotations

import inspect
import hashlib
import json
from typing import Any

from menagerie.pipeline.records import RecordError

NODES = 4
TEXT_LENGTH = 16
IMAGE_SIZE = 32


def preset(recipe: dict[str, Any]) -> str:
    """Distinct module names keep independently built recipes' inputs separate."""

    identity = {'call': recipe['call'], 'args': recipe.get('args') or [],
                'kwargs': recipe.get('kwargs') or {}, 'config': recipe.get('config') or {}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return f'example:menagerie_library_inputs_{key}:example_input'


def _get(config: Any, key: str, default: Any = None) -> Any:
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def _image(config: Any) -> Any:
    from menagerie.pipeline.inputs import contract, make_input

    size = _get(config, 'image_size', IMAGE_SIZE)
    size = (size, size) if isinstance(size, int) else tuple(size)
    channels = _get(config, 'num_channels', 3)
    preset = 'image' if channels in (1, 3, 4) else 'randn'
    return make_input(contract(preset, (1, channels, *size)))[0][0]


def _text(config: Any) -> Any:
    from menagerie.pipeline.inputs import contract, make_input

    vocab = _get(config, 'vocab_size', 257)
    spec = contract('text' if vocab >= 257 else f'randint:{vocab}', (1, TEXT_LENGTH))
    return make_input(spec)[0][0]


def _hf(model: Any) -> dict[str, Any]:
    import torch

    cfg = model.config
    parameters = inspect.signature(model.forward).parameters
    result = {}
    if 'input_ids' in parameters:
        result['input_ids'] = _text(_get(cfg, 'text_config', cfg))
        if 'attention_mask' in parameters:
            result['attention_mask'] = torch.ones_like(result['input_ids'])
    if 'pixel_values' in parameters:
        result['pixel_values'] = _image(_get(cfg, 'vision_config', cfg))
    if _get(cfg, 'is_encoder_decoder', False) and 'decoder_input_ids' in parameters:
        result['decoder_input_ids'] = _text(cfg)
    required = {n for n, p in parameters.items() if p.default is p.empty and
                p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    if not result or required - set(result):
        raise RecordError(f'needs_adapter:inputs: unsupported HF inputs {sorted(required - set(result))}')
    return result


def _graph(model: Any) -> dict[str, Any]:
    import torch

    params = inspect.signature(model.forward).parameters
    features = getattr(model, 'in_channels', None)
    if not isinstance(features, int) or features <= 0:
        raise RecordError('needs_adapter:inputs: graph input feature count is unresolved')
    available = {'x': torch.randn(NODES, features),
                 'edge_index': torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
                 'batch': torch.zeros(NODES, dtype=torch.long),
                 'edge_weight': torch.ones(NODES)}
    required = {n for n, p in params.items() if p.default is p.empty and
                p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    if required - set(available):
        raise RecordError(f'needs_adapter:inputs: unsupported graph inputs {sorted(required)}')
    return {n: v for n, v in available.items() if n in params}


def _diffusion(model: Any) -> dict[str, Any]:
    import torch

    name, cfg = type(model).__name__, model.config
    supported = {'UNet2DModel', 'UNet2DConditionModel', 'AutoencoderKL',
                 'VQModel', 'AutoencoderTiny', 'T2IAdapter'}
    if name not in supported:
        raise RecordError(f'needs_adapter:inputs: unsupported diffusion interface {name}')
    channels = _get(cfg, 'in_channels', 3)
    size = _get(cfg, 'sample_size') or IMAGE_SIZE
    size = (size, size) if isinstance(size, int) else tuple(size)
    result = {'sample': torch.randn(1, channels, *size)}
    if name.startswith('UNet'):
        result['timestep'] = torch.tensor(1)
    if name == 'UNet2DConditionModel':
        dim = _get(cfg, 'cross_attention_dim')
        if not isinstance(dim, int):
            raise RecordError('needs_adapter:inputs: nonuniform cross attention dimensions')
        result['encoder_hidden_states'] = torch.randn(1, TEXT_LENGTH, dim)
    if name == 'T2IAdapter':
        return {'x': result['sample']}
    return result


def _forecast(model: Any) -> tuple[dict[str, Any]]:
    import torch

    # NeuralForecast forward consumes one windows_batch dictionary.
    length, horizon = model.input_size, model.h
    if length <= 0:
        raise RecordError('needs_adapter:inputs: unresolved forecasting window')
    return ({'insample_y': torch.randn(1, length, 1),
             'insample_mask': torch.ones(1, length, 1),
             'futr_exog': torch.randn(1, length + horizon, model.futr_exog_size),
             'hist_exog': torch.randn(1, length, model.hist_exog_size),
             'stat_exog': torch.randn(1, model.stat_exog_size)},)


def _ctr(model: Any) -> tuple[Any]:
    import torch

    features = model.feature_index
    width = max(end for _, end in features.values())
    # Zero is valid for every categorical vocabulary; dense features are seeded.
    value = torch.zeros(2, width)
    return (value,)


def for_model(model: Any, call: str) -> Any:
    """Make seeded example inputs; callers use inputs._example to preserve RNG state."""

    if call.startswith('transformers.'):
        return _hf(model)
    if call.startswith('torch_geometric.'):
        return _graph(model)
    if call.startswith('diffusers.'):
        return _diffusion(model)
    if call.startswith('neuralforecast.'):
        return _forecast(model)
    if call.startswith('deepctr_torch.'):
        return _ctr(model)
    if call.endswith(('raft_small', 'raft_large')):
        image = _image({'image_size': 128})
        return image, image.clone()
    raise RecordError(f'needs_adapter:inputs: no verified builder for {call}')


def example_input(recipe: dict[str, Any]) -> Any:
    """Bound by the library builder. The meta model contributes config, never tensors."""

    from menagerie.pipeline.harvest_build import _builder
    from menagerie.pipeline.meta_build import build_structure

    plain = {k: v for k, v in recipe.items() if k != 'input'}
    # A constructor that reads tensor values gets a sized CPU build (meta_build).
    model = build_structure(_builder(plain, None))
    return for_model(model, recipe['call'])
