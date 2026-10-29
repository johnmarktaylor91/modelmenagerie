"""Constructor audit: inspect nested construction and never execute June code."""

import pytest

from menagerie.pipeline import june_audit


@pytest.mark.parametrize('source', [
    'import torch; model=torch.nn.Sequential(torch.nn.Linear(8,4))',
    'import transformers; cfg=transformers.BertConfig(hidden_size=8); '
    'model=transformers.BertModel(cfg)',
    'import transformers; cfg=transformers.BertConfig(); cfg.hidden_size=8; '
    'model=transformers.BertModel(cfg)',
    'import transformers; cfg=transformers.BertConfig(); setattr(cfg,"hidden_size",8); '
    'model=transformers.BertModel(cfg)',
    'import timm; model=timm.create_model("resnet18", pretrained=True)',
    'import timm; model=timm.create_model("resnet18", **sizes)',
    'import torch; model=type("Fake",(torch.nn.Module,),{})(); '
    'f=lambda:torch.nn.Linear(2,2)',
    'exec("raise RuntimeError(\"must never execute\")")',
    'import transformers; cfg=transformers.BertConfig(); '
    'cfg=transformers.PretrainedConfig.from_dict({"hidden_size":8}); '
    'model=transformers.BertModel(cfg)',
    'import timm; model=timm.create_model("resnet18"); del model.fc',
    'import timm; model=timm.create_model("resnet18"); delattr(model,"fc")',
    'import timm,torch; model=timm.create_model("resnet18"); '
    'model.add_module("fc",torch.nn.Identity())',
    'import transformers; cfg=transformers.BertConfig(); '
    'vars(cfg).update(hidden_size=8); model=transformers.BertModel(cfg)',
    'import transformers; cfg=transformers.BertConfig(); '
    'cfg.__setattr__("hidden_size",8); model=transformers.BertModel(cfg)',
    'import timm; model=timm.create_model("resnet18"); model.apply(customize)',
    'import timm; model=timm.create_model("resnet18"); exec("del model.fc")',
    'from torchvision.models import *; model=resnet18(num_classes=2)',
    'from transformers import BertConfig,BertForSequenceClassification; '
    'model=BertForSequenceClassification(BertConfig(num_labels=1))',
    'this is not valid python!'
])
def test_unproven_or_shrunken_construction_is_flagged(source: str) -> None:
    assert not june_audit.audit(source).passed


def test_all_constructors_must_pass_not_only_the_first() -> None:
    result = june_audit.audit('import timm, torch; model=timm.create_model("resnet18", '
                              'pretrained=False); other=torch.nn.Linear(2,2)')
    assert not result.passed and len(result.calls) == 2


def test_default_library_constructor_passes() -> None:
    result = june_audit.audit('import timm; model=timm.create_model("resnet18", pretrained=False)')
    assert result.passed, result.issues


def test_star_import_with_eval_keeps_the_default_constructor() -> None:
    result = june_audit.audit('from torchvision.models import *; model=resnet18().eval()')
    assert result.passed, result.issues
    assert result.calls == ('torchvision.models.resnet18',)


def test_lowercase_extracted_factory_is_audited(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import harvest_build

    def build(config=None, depth=12):
        return None

    monkeypatch.setattr(harvest_build, '_resolve', lambda call: build)
    source = 'import library; model=library.build(depth=2)'
    assert june_audit.extract(source)[0] == 'library.build'
    result = june_audit.audit(source)
    assert not result.passed and 'library.build' in result.calls


def test_nested_lowercase_constructor_cannot_escape_the_audit(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import harvest_build

    def builder(depth=12):
        return None

    monkeypatch.setattr(harvest_build, '_resolve', lambda call: builder)
    source = 'import library; model=library.build(); f=lambda:library.compact(depth=2)'
    result = june_audit.audit(source)
    assert not result.passed and set(result.calls) == {'library.build', 'library.compact'}


@pytest.mark.parametrize('hidden', [
    'M=timm.create_model; other=M("resnet18",num_classes=2)',
    'other=getattr(timm,"create_model")("resnet18",num_classes=2)',
    'factories=[timm.create_model]; other=factories[0]("resnet18",num_classes=2)',
    'other=(lambda: timm.create_model("resnet18"))()',
    'timm=object(); other=timm.create_model("resnet18")',
    'def hidden(timm): return timm.create_model("resnet18")',
    'factory: object = timm.create_model; other=factory("resnet18",num_classes=2)',
    'other=unknown_factory("resnet18",num_classes=2)',
    'other=unknown_module.factory("resnet18",num_classes=2)',
])
def test_unresolved_or_rebound_constructors_fail_closed(hidden: str) -> None:
    source = 'import timm; model=timm.create_model("resnet18")\n' + hidden
    assert not june_audit.audit(source).passed


def _fake_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """``fakelib``: a factory with a width default and a net that takes its own components."""
    import sys
    import types

    lib = types.ModuleType('fakelib')
    neuron = types.ModuleType('fakelib.neuron')

    class IF:
        pass

    class ATan:
        def __init__(self, alpha=2.0):
            self.alpha = alpha

    class Net:
        def __init__(self, spiking_neuron=None, depth=4, **kwargs):
            pass

    def make(width=8):
        return None

    neuron.IF, neuron.ATan = IF, ATan
    lib.neuron, lib.Net, lib.make, lib.DEPTH = neuron, Net, make, 3
    monkeypatch.setitem(sys.modules, 'fakelib', lib)
    monkeypatch.setitem(sys.modules, 'fakelib.neuron', neuron)


def test_a_factory_imported_by_name_is_audited(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_library(monkeypatch)
    shrunk = june_audit.audit('from fakelib import make; other=make(width=2); '
                              'model=make()')
    assert not shrunk.passed and 'fakelib.make: nondefault or unproven argument width' in \
        shrunk.issues
    assert june_audit.audit('from fakelib import make; model=make(width=8)').passed


def test_components_from_the_same_library_are_its_published_choice(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The drafts review's false flag m10216: ``spiking_neuron=neuron.IFNode,
    surrogate_function=surrogate.ATan()``; the nested call is audited as a call of its own."""
    _fake_library(monkeypatch)
    source = ('from fakelib import neuron; from fakelib import Net; '
              'model=Net(spiking_neuron=neuron.IF, surrogate_function=neuron.ATan())')
    result = june_audit.audit(source)
    assert result.passed, result.issues
    assert june_audit.extract(source)[2]['surrogate_function'] == {
        '$call': 'fakelib.neuron.ATan', 'args': [], 'kwargs': {}}
    tuned = source.replace('neuron.ATan()', 'neuron.ATan(alpha=9.0)')
    assert not june_audit.audit(tuned).passed  # the component's own width-like override
    import types
    import sys

    other = types.ModuleType('otherlib')
    other.Neuron = type('Neuron', (), {})
    monkeypatch.setitem(sys.modules, 'otherlib', other)
    foreign = 'import otherlib; from fakelib import Net; model=Net(spiking_neuron=otherlib.Neuron)'
    assert not june_audit.audit(foreign).passed
    assert not june_audit.audit('from fakelib import Net; model=Net(depth=2)').passed


@pytest.mark.parametrize('argument', [
    'spiking_neuron=fakelib.DEPTH',      # a constant where the default is None
    'hidden=fakelib.make(width=8)',      # a helper's result absorbed by **kwargs
])
def test_a_same_library_value_that_is_not_a_class_is_audited(
        monkeypatch: pytest.MonkeyPatch, argument: str) -> None:
    _fake_library(monkeypatch)
    assert not june_audit.audit(f'import fakelib; model=fakelib.Net({argument})').passed


def test_a_library_that_does_not_import_is_an_environment_fact() -> None:
    result = june_audit.audit('from nosuchlib_r3cov import Net; model=Net(width=3)')
    assert not result.passed and result.missing and not result.departures
    mixed = june_audit.audit('import torch; from nosuchlib_r3cov import Net; '
                             'model=torch.nn.Sequential(Net())')
    assert mixed.missing and mixed.departures


@pytest.mark.parametrize('source', [
    "model = AutoModelForCausalLM.from_config(AutoConfig.for_model('gpt2', n_layer=2))",
    'model = GPT2Model(GPT2Config(n_layer=2))',
])
def test_a_name_june_never_imports_is_a_code_fact_not_a_missing_library(source: str) -> None:
    """A June recipe with no import statement names transformers objects bare: the library
    imports here, so the row is a June-code verdict (and may take the registry retry)."""

    result = june_audit.audit(source)
    assert not result.passed and not result.missing
    assert any(june_audit.UNBOUND in issue for issue in result.issues)


@pytest.mark.parametrize('source', [
    'from nosuchlib_r3cov import *; model = Net(width=3)',
    'try:\n    import nosuchlib_r3cov\nexcept ImportError:\n    pass\nmodel = nosuchlib_r3cov.Net(width=3)',
    'if True:\n    from nosuchlib_r3cov import Net\nmodel = Net(width=3)',
])
def test_a_missing_library_imported_by_star_or_below_top_level_stays_missing(source: str) -> None:
    """Only a name no import binds is UNBOUND; a star import or a nested import of a library that
    does not import here is still an environment fact."""

    result = june_audit.audit(source)
    assert not any(june_audit.UNBOUND in issue for issue in result.issues)
    assert result.missing or not result.calls  # audited as missing, or not audited at all (as before)

