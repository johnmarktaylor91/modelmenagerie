"""Meta builds of constructors that read tensor values: sized CPU fallback or a typed refusal."""

import sys
import types

import pytest

from menagerie.pipeline import harvest, library_inputs, meta_build
from menagerie.pipeline.harvest_build import on_device

torch = pytest.importorskip('torch')


class Schedule(torch.nn.Module):
    """Like segmentation_models_pytorch's MiT encoders: a drop-path schedule read by .item()."""

    def __init__(self) -> None:
        super().__init__()
        self.rates = [x.item() for x in torch.linspace(0, 0.3, 4)]
        self.proj = torch.nn.Linear(8, 4)


def test_meta_only_build_fails_and_is_recognised() -> None:
    with pytest.raises(RuntimeError) as caught, on_device('meta'):
        Schedule()
    assert meta_build.is_meta_error(caught.value)


def test_meta_failure_falls_back_to_a_sized_cpu_build() -> None:
    state = torch.random.get_rng_state()
    model = meta_build.build_structure(Schedule)
    assert torch.equal(state, torch.random.get_rng_state())
    assert model.rates == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert model.proj.weight.device.type == 'cpu'
    assert meta_build.probe_gib(Schedule) == pytest.approx(36 * 4 / 2**30)


def test_meta_failure_over_the_declared_memory_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('MENAGERIE_MEMORY_GB', '1')
    monkeypatch.setattr(meta_build, 'CPU_BUILD_FACTOR', 1e12)
    with pytest.raises(meta_build.MetaBuildRefused) as caught:
        meta_build.build_structure(Schedule)
    assert caught.value.cls == 'memory_cap'


def test_example_input_and_weight_size_use_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType('menagerie_test_schedule')
    module.Schedule = Schedule
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(library_inputs, 'for_model', lambda model, call: model.rates)
    recipe = {'kind': 'library', 'call': 'menagerie_test_schedule.Schedule', 'args': [],
              'kwargs': {}, 'config': {}}
    assert library_inputs.example_input(recipe) == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert harvest._weights_gib(recipe, None, strict=True) == pytest.approx(36 * 4 / 2**30)


def test_a_structure_only_build_of_a_value_reading_constructor_is_all_meta(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import from_dict, harvest_build
    from menagerie.pipeline.records import Recipe

    module = types.ModuleType('menagerie_test_schedule')
    module.Schedule = Schedule
    monkeypatch.setitem(sys.modules, module.__name__, module)
    recipe = from_dict(Recipe, dict(
        kind='library', env='torch-main', backend='torch', rev='0' * 12,
        call='menagerie_test_schedule.Schedule',
        input={'preset': 'image', 'kind': 'random-fallback'}))
    state = torch.random.get_rng_state()
    model = harvest_build.construct(recipe, device='meta')
    assert torch.equal(state, torch.random.get_rng_state())
    assert model.rates == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert all(t.is_meta for t in (*model.parameters(), *model.buffers()))
