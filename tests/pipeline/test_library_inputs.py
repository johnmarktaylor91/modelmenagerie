"""Actual small library forwards and input-boundary checks, with seeded examples."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from menagerie.pipeline import harvest_build, library_inputs, read_json, runner
from menagerie.pipeline.inputs import make_input
from menagerie.pipeline.records import GiveUp, ModelRecord, RecordError
from pathlib import Path


def test_graph_inputs_drive_an_actual_gcn() -> None:
    torch = pytest.importorskip('torch')
    pyg = pytest.importorskip('torch_geometric.nn')
    model = pyg.GCNConv(8, 4)
    kwargs = library_inputs.for_model(model, 'torch_geometric.nn.GCNConv')
    assert model(**kwargs).shape == (4, 4)
    assert kwargs['edge_index'].dtype == torch.long


def test_unresolved_graph_dimensions_and_diffusion_interfaces_stay_typed() -> None:
    class UnknownGraph:
        in_channels = (-1, -1)
        def forward(self, x, edge_index):
            pass
    with pytest.raises(RecordError, match='feature count'):
        library_inputs.for_model(UnknownGraph(), 'torch_geometric.nn.HeteroConv')
    with pytest.raises(RecordError, match='unsupported diffusion'):
        library_inputs.for_model(SimpleNamespace(config={}), 'diffusers.FluxTransformer2DModel')


def test_default_tiny_vae_forward_uses_its_actual_config() -> None:
    torch = pytest.importorskip('torch')
    diffusers = pytest.importorskip('diffusers')
    model = diffusers.AutoencoderTiny().eval()
    inputs = library_inputs.for_model(model, 'diffusers.AutoencoderTiny')
    with torch.no_grad():
        assert model(**inputs).sample.shape == (1, 3, 32, 32)


def test_hf_multimodal_boundary_and_decoder_are_explicit() -> None:
    class Multi:
        config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=100),
                                 vision_config=SimpleNamespace(image_size=32, num_channels=3))
        def forward(self, input_ids=None, pixel_values=None, attention_mask=None):
            pass
    kwargs = library_inputs.for_model(Multi(), 'transformers.CLIPModel')
    assert kwargs['input_ids'].max() < 100
    assert kwargs['pixel_values'].shape == (1, 3, 32, 32)
    class Seq:
        config = SimpleNamespace(vocab_size=100, is_encoder_decoder=True)
        def forward(self, input_ids, decoder_input_ids):
            pass
    assert set(library_inputs.for_model(Seq(), 'transformers.T5Model')) == {
        'input_ids', 'decoder_input_ids'}


def test_actual_default_multimodal_forward() -> None:
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    model = transformers.CLIPModel(transformers.CLIPConfig()).eval()
    kwargs = library_inputs.for_model(model, 'transformers.CLIPModel')
    with torch.no_grad():
        output = model(**kwargs)
    assert output.logits_per_image.shape == (1, 1)


def test_actual_default_encoder_decoder_forward() -> None:
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    model = transformers.T5Model(transformers.T5Config()).eval()
    kwargs = library_inputs.for_model(model, 'transformers.T5Model')
    with torch.no_grad():
        output = model(**kwargs)
    assert output.last_hidden_state.shape == (1, 16, model.config.d_model)


def test_actual_default_two_image_flow_forward() -> None:
    torch = pytest.importorskip('torch')
    flow = pytest.importorskip('torchvision.models.optical_flow')
    model = flow.raft_small(weights=None).eval()
    args = library_inputs.for_model(model, 'torchvision.models.optical_flow.raft_small')
    with torch.no_grad():
        output = model(*args)
    assert output[-1].shape == (1, 2, 128, 128)


def test_forecast_and_ctr_builders_respect_model_input_schemas() -> None:
    forecast = SimpleNamespace(input_size=24, h=12, futr_exog_size=2,
                               hist_exog_size=3, stat_exog_size=1)
    (batch,) = library_inputs.for_model(forecast, 'neuralforecast.models.MLP')
    assert batch['futr_exog'].shape == (1, 36, 2)
    assert batch['hist_exog'].shape == (1, 24, 3)
    (ctr,) = library_inputs.for_model(SimpleNamespace(feature_index={'id': (0, 1), 'score': (1, 3)}),
                                     'deepctr_torch.models.AFN')
    assert ctr.shape == (2, 3) and ctr.sum() == 0


def test_actual_default_forecast_forward() -> None:
    torch = pytest.importorskip('torch')
    models = pytest.importorskip('neuralforecast.models')
    model = models.MLP(h=12, input_size=24).eval()
    args = library_inputs.for_model(model, 'neuralforecast.models.MLP')
    with torch.no_grad():
        assert model(*args).shape[:2] == (1, 12)


def test_actual_default_ctr_forward() -> None:
    torch = pytest.importorskip('torch')
    models = pytest.importorskip('deepctr_torch.models')
    features = pytest.importorskip('deepctr_torch.inputs')
    schema = [features.SparseFeat('user', 8), features.SparseFeat('item', 16)]
    model = models.AFN(linear_feature_columns=schema, dnn_feature_columns=schema).eval()
    args = library_inputs.for_model(model, 'deepctr_torch.models.AFN')
    with torch.no_grad():
        assert model(*args).shape == (2, 1)


def test_bound_library_input_survives_json_recipe_and_restores_rng() -> None:
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from menagerie.pipeline import from_dict
    from menagerie.pipeline.records import Recipe

    # SqueezeBERT's actual defaults fit the lane cap; no shrunken named model.
    raw = dict(kind='library', env='torch-main', backend='torch',
        rev='0' * 12, call='transformers.AutoModel.from_config', config={'model_type': 'squeezebert'},
        input={'kind': 'random-fallback', 'note': 'actual configuration'})
    raw['input']['preset'] = library_inputs.preset(raw)
    recipe = from_dict(Recipe, raw)
    model = harvest_build.construct(recipe).eval()
    state = torch.random.get_rng_state().clone()
    args, kwargs = make_input(recipe.input)
    assert not args and set(kwargs) == {'input_ids', 'attention_mask'}
    assert torch.equal(state, torch.random.get_rng_state())
    with torch.no_grad():
        assert model(**kwargs).last_hidden_state.shape[:2] == (1, 16)


@pytest.mark.parametrize('cls', ['june_code:flagged', 'standin:kwargs'])
def test_flagged_and_standin_records_cannot_request_a_trace(cls: str) -> None:
    fixture = Path(__file__).with_name('fixtures') / 'm8839.model.json'
    model = read_json(ModelRecord, fixture)
    model = replace(model, give_up=GiveUp(cls=cls, reason='unproven', at='2026-10-08T00:00:00Z'))
    with pytest.raises(RecordError, match=cls):
        runner.request_for(model, 'trace')


@pytest.mark.parametrize(('call', 'config'), [
    ('diffusers.AutoencoderKL', {}),
    ('diffusers.T2IAdapter', {}),
    ('transformers.DPRReader', {'model_type': 'dpr'}),
])
def test_accepted_sample_rebuilds_forward_from_their_bound_recipes(call, config) -> None:
    torch = pytest.importorskip('torch')
    pytest.importorskip(call.partition('.')[0])
    from menagerie.pipeline import from_dict
    from menagerie.pipeline.records import Recipe

    raw = dict(kind='library', env='torch-main', backend='torch', rev='0' * 12,
               call=call, config=config,
               input={'kind': 'random-fallback', 'note': 'actual reference configuration'})
    raw['input']['preset'] = library_inputs.preset(raw)
    recipe = from_dict(Recipe, raw)
    model = harvest_build.construct(recipe).eval()
    args, kwargs = make_input(recipe.input)
    with torch.no_grad():
        output = model(*args, **kwargs)
    if call.endswith('AutoencoderKL'):
        assert output.sample.shape == kwargs['sample'].shape
    elif call.endswith('T2IAdapter'):
        assert len(output) == len(model.config.channels)
        assert output[0].shape[1] == model.config.channels[0]
    else:
        assert output.start_logits.shape == kwargs['input_ids'].shape
        assert output.end_logits.shape == kwargs['input_ids'].shape
