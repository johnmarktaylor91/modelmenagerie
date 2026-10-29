"""Faithful library extraction and rebuilds, including frozen real catalog rows."""

from dataclasses import replace

import pytest

import json
import os
from pathlib import Path

from menagerie.pipeline import DataLayout, harvest_tiers, library_adapter
from menagerie.pipeline.harvest import Row
from menagerie.pipeline import read_json, write_json, runner
from menagerie.pipeline.records import GiveUp, ModelRecord, RecordError

DATA = DataLayout.from_env()


def _row(name: str, source: str, zoo: str = 'diffusers') -> Row:
    cat = {'name': name, 'zoo': zoo, 'variant': '', 'family': name,
           'recipe': {'type': 'statement', 'code': source},
           'input': {'kind': 'tensor', 'spec': {'shape': [1, 3, 32, 32], 'dtype': 'float32'}}}
    return Row('m1', name, zoo, '', name, catalog=cat)


def test_default_rebuild_drops_width_and_gets_actual_input() -> None:
    row = _row('AutoencoderTiny', 'from diffusers import AutoencoderTiny; '
               'model=AutoencoderTiny(num_encoder_blocks=(1,1,1,1), '
               'num_decoder_blocks=(1,1,1,1))')
    made = library_adapter.draft(row, DATA, rebuild=True)
    assert made.give_up is None, made.give_up
    assert 'num_encoder_blocks' not in made.recipe['kwargs']
    assert made.recipe['input'].preset.startswith('example:menagerie_library_inputs_')


def test_named_custom_model_never_rebuilds_as_reference_config() -> None:
    row = _row('Geneformer', 'from transformers import BertConfig,BertModel; '
               'model=BertModel(BertConfig(hidden_size=8))', 'transformers')
    made = library_adapter.draft(row, DATA, rebuild=True)
    assert made.give_up[0] == 'standin:no_published_config' and not made.recipe


def test_unproven_june_code_has_no_traceable_recipe() -> None:
    row = _row('TrackFormer', 'import torch; model=torch.nn.Sequential(torch.nn.Linear(8,4))')
    made = library_adapter.draft(row, DATA)
    assert made.give_up[0] == 'june_code:flagged' and not made.recipe


def test_rebuild_cannot_keep_unpublished_required_width() -> None:
    row = _row('Linear', 'import torch; model=torch.nn.Linear(8,4)', 'torch.nn')
    made = library_adapter.draft(row, DATA, rebuild=True)
    assert made.give_up[0] == 'standin:no_published_config'


def test_input_only_raft_adapter_keeps_the_library_model() -> None:
    row = _row('raft_small', 'from torchvision.models.optical_flow import raft_small; '
               'model=raft_small(weights=None)', 'torchvision.models.optical_flow')
    row.catalog['input'] = {'kind': 'none'}
    # The optical-flow correlation pyramid needs at least 128 pixels.
    made = library_adapter.draft(row, DATA)
    assert made.give_up is None, made.give_up


def test_frozen_catalog_small_sample_is_typed_or_has_a_faithful_recipe() -> None:
    # About twenty real residual identities spanning the work lists. No trace runs here.
    selected = {'m3142', 'm10244', 'm1914', 'm8150', 'm1956', 'm1248', 'm4134',
                'm5508', 'm10820', 'm10819', 'm138', 'm5310', 'm569', 'm5155',
                'm7860', 'm2973', 'm24', 'm3938', 'm4646', 'm5690'}
    seen, outcomes = set(), []
    for cat, ros in harvest_tiers.joined(DATA):
        sid = ros['stable_id']
        if sid not in selected:
            continue
        seen.add(sid)
        row = Row(sid, cat['name'], cat['zoo'], cat['variant'], cat['family'], catalog=cat)
        made = library_adapter.draft(row, DATA, rebuild=sid in {'m138', 'm5310', 'm569',
                                                               'm5155', 'm7860', 'm2973'})
        outcomes.append({'id': sid, 'name': row.name, 'class': made.give_up[0] if made.give_up else None,
                         'reason': made.give_up[1] if made.give_up else '',
                         'recipe': made.recipe['call'] if made.recipe else None})
        if made.give_up:
            assert not made.recipe
        else:
            assert made.recipe['kind'] == 'library'
            assert not made.recipe['input'].note.startswith('sentinel')
    assert seen == selected
    if target := os.environ.get('RESIDUAL_SAMPLE_JSON'):
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(outcomes, indent=2) + '\n')


@pytest.mark.parametrize('old_class', ['needs_adapter', 'standin:kwargs'])
def test_failed_replacement_preserves_revision_and_blocks_trace(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old_class: str) -> None:
    data = DataLayout(tmp_path)
    fixture = Path(__file__).with_name('fixtures') / 'm8839.model.json'
    old = replace(read_json(ModelRecord, fixture), give_up=GiveUp(
        cls=old_class, reason='unproven', at='2026-10-08T00:00:00Z'))
    write_json(old, data.model(old.id))
    row = replace(_row('Linear', 'import torch; model=torch.nn.Linear(8,4)'), id=old.id)
    monkeypatch.setattr(harvest_tiers, '_tier_rows', lambda *args: [row])
    monkeypatch.setattr(harvest_tiers, '_owned_elsewhere', lambda *args: False)
    [held] = library_adapter.library_adapter(row.zoo, [old.id], data=data, write=True)
    assert held.recipe.rev == old.recipe.rev
    assert held.give_up.cls == ('june_code:flagged' if old_class == 'needs_adapter' else old_class)
    assert held.give_up.by == ('script' if old_class == 'needs_adapter' else None)
    assert read_json(ModelRecord, data.model(old.id)) == held
    with pytest.raises(RecordError):
        runner.request_for(held, 'trace')


def _isolation_rows(monkeypatch: pytest.MonkeyPatch, raises: dict) -> list:
    rows = [replace(_row(f'Net{i}', 'import torch'), id=f'm{i}') for i in (1, 2, 3)]
    monkeypatch.setattr(harvest_tiers, '_tier_rows', lambda *args: rows)
    monkeypatch.setattr(harvest_tiers, '_owned_elsewhere', lambda *args: False)

    def fake_draft(row, data, *, rebuild=False):
        if row.id in raises:
            raise raises[row.id]
        return harvest_tiers._give_up(row, 'needs_adapter', 'fake', None)

    written = []
    monkeypatch.setattr(library_adapter, 'draft', fake_draft)
    monkeypatch.setattr(library_adapter.harvest, 'write_drafts',
                        lambda drafts, **kw: written.extend(drafts) or list(drafts))
    return written


def test_one_raising_row_is_typed_and_the_others_continue(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    written = _isolation_rows(monkeypatch, {
        'm2': RuntimeError('Tensor.item() cannot be called on meta tensors')})
    out = library_adapter.library_adapter('diffusers', ['m1', 'm2', 'm3'],
                                          data=DataLayout(tmp_path), write=True)
    assert [d.row.id for d in out] == ['m1', 'm2', 'm3'] == [d.row.id for d in written]
    classes = {d.row.id: d.give_up[0] for d in written}
    assert classes == {'m1': 'needs_adapter', 'm2': 'construct_error', 'm3': 'needs_adapter'}
    assert 'meta tensors' in written[1].give_up[1]


def test_a_stopped_loop_still_writes_finished_rows(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    written = _isolation_rows(monkeypatch, {'m3': KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        library_adapter.library_adapter('diffusers', ['m1', 'm2', 'm3'],
                                        data=DataLayout(tmp_path), write=True)
    assert [d.row.id for d in written] == ['m1', 'm2']


@pytest.mark.parametrize('evidence', ['by', 'ledger'])
def test_an_agents_give_up_is_never_redrafted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence: str) -> None:
    """Only a scripted verdict is the script's to redo; an agent's give-up settles its row."""
    from menagerie.pipeline import append_jsonl
    from menagerie.pipeline.records import LedgerRow

    data = DataLayout(tmp_path)
    fixture = Path(__file__).with_name('fixtures') / 'm8839.model.json'
    give = GiveUp(cls='missing_dependency', reason='no wheels', at='2026-10-04T19:10:18Z',
                  by='agent' if evidence == 'by' else None)
    old = replace(read_json(ModelRecord, fixture), recipe=None, give_up=give)
    write_json(old, data.model(old.id))
    if evidence == 'ledger':
        append_jsonl([LedgerRow(id=old.id, stage='forward', at=give.at, host='worker',
                                env='torch-main', recipe_rev='0' * 12, outcome='giveup',
                                cls=give.cls)], data.ledger('2026-10'))
    row = replace(_row('Linear', 'import torch; model=torch.nn.Linear(8,4)'), id=old.id)
    monkeypatch.setattr(harvest_tiers, '_tier_rows', lambda *args: [row])
    monkeypatch.setattr(harvest_tiers, '_owned_elsewhere', lambda *args: False)
    assert library_adapter.library_adapter(row.zoo, [old.id], data=data, write=True) == []
    assert read_json(ModelRecord, data.model(old.id)) == old


@pytest.mark.parametrize(('name', 'zoo', 'parsed', 'same'), [
    ('monai.BasicUNetPlusPlus', 'monai.networks.nets',
     ('monai.networks.nets.BasicUNetPlusPlus', [], {}, {}), True),
    ('recbole.BPR', 'recbole', ('recbole.model.general_recommender.BPR', [], {}, {}), True),
    ('spikingjelly_spiking_vgg19_bn', 'spikingjelly',
     ('spikingjelly.activation_based.model.spiking_vgg.spiking_vgg19_bn', [], {}, {}), True),
    ('ganzoo_dcgan_discriminator', 'ganzoo (pip pytorch-gan-zoo)',
     ('ganzoo.DCGAN_Discriminator', [], {}, {}), True),
    ('qwen2_moe_hf', 'transformers',
     ('transformers.AutoModel.from_config', [], {}, {'model_type': 'qwen2_moe'}), True),
    ('coca_base', 'open_clip-2.32.0', ('open_clip.create_model', ['coca_base'], {}, {}), True),
    ('yoloe-26s-seg', 'ultralytics',
     ('ultralytics.nn.tasks.SegmentationModel', ['yoloe-26s-seg.yaml'], {}, {}), True),
    ('resnet18', 'torchvision', ('torchvision.models.get_model', ['resnet18'], {}, {}), True),
    # still refused: a different model, or a name that only shares a library
    ('coca_base', 'open_clip-2.32.0', ('open_clip.create_model', ['ViT-B-32'], {}, {}), False),
    ('yoloe-26s-seg', 'ultralytics',
     ('ultralytics.nn.tasks.SegmentationModel', ['yolov8n-seg.yaml'], {}, {}), False),
    ('monai.ViT', 'monai.networks.nets', ('monai.networks.nets.UNet', [], {}, {}), False),
    ('qwen2_moe_hf', 'transformers',
     ('transformers.AutoModel.from_config', [], {}, {'model_type': 'qwen2'}), False),
])
def test_identity_knows_zoo_prefixes_suffixes_and_selectors(name: str, zoo: str, parsed: tuple,
                                                          same: bool) -> None:
    row = Row('m1', name, zoo, '', name, catalog={})
    assert library_adapter._identity(row, parsed) is same


def test_a_classics_row_takes_the_classics_path(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _row('Sonata', 'from menagerie.classics.point import build_sonata; '
               'model=build_sonata()', zoo='discovered-pytorch')
    monkeypatch.setattr(harvest_tiers, 'tier_draft', lambda row, data: ('classics path', row.id))
    assert library_adapter.draft(row, DATA) == ('classics path', 'm1')


@pytest.mark.parametrize('source', [
    'from nosuchlib_r3cov import Net; model=Net(width=3)',
    # June code that also departs (a hand-written wrapper): still an environment fact first
    'import torch; from nosuchlib_r3cov import Net; model=torch.nn.Sequential(Net(), torch.nn.ReLU())',
    'import nosuchlib_r3cov as lib; _m = lib.Net(); model = _m.blocks[0]',
])
def test_a_missing_library_is_typed_missing_dependency(source: str) -> None:
    """Crawler 2026-10-09: a library that does not import even in the overlay is never a
    June-code departure; the row is ``missing_dependency`` and reaches 2b."""

    made = library_adapter.draft(_row('Net', source, zoo='nosuchlib'), DATA)
    assert made.give_up[0] == 'missing_dependency' and 'does not import' in made.give_up[1]


def test_a_flagged_row_is_retried_as_its_registry_name(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _row('fuyu', 'import transformers; cfg=transformers.AutoConfig.for_model("fuyu"); '
               'cfg.hidden_size=8; model=transformers.AutoModel.from_config(cfg)',
               zoo='discovered-pytorch')
    seen = []
    monkeypatch.setattr(library_adapter.registry, 'lookup', lambda name, zoo: (
        'transformers.AutoModel.from_config', [], {}, {'model_type': 'fuyu'}, {}))
    monkeypatch.setattr(library_adapter.heads, 'built_class', lambda head, model_type: 'FuyuModel')

    def parsed_draft(row, parsed, data, *, rebuild, identified=False):
        from types import SimpleNamespace

        seen.append((parsed, rebuild, identified))
        return SimpleNamespace(give_up=None, delta={}, row=row)

    monkeypatch.setattr(library_adapter, '_parsed_draft', parsed_draft)
    made = library_adapter.draft(row, DATA)
    assert seen[-1] == (('transformers.AutoModel.from_config', [], {}, {'model_type': 'fuyu'},
                         {}), True, True)
    assert made.give_up is None and made.delta['registry'].startswith('transformers.')
    monkeypatch.setattr(library_adapter.registry, 'lookup', lambda name, zoo: None)
    assert library_adapter.draft(row, DATA).give_up[0] == 'june_code:flagged'


def test_a_zoo_that_needs_extras_drafts_inside_its_overlay(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [replace(_row('Net1', 'import torch'), id='m1')]
    monkeypatch.setattr(harvest_tiers, '_tier_rows', lambda *args: rows)
    monkeypatch.setattr(harvest_tiers, '_missing_extras', lambda rows: ['monai==1.5.2'])
    monkeypatch.delenv(harvest_tiers.OVERLAY_FLAG, raising=False)
    calls = []
    monkeypatch.setattr(library_adapter, '_in_overlay',
                        lambda zoo, names, extras, write: calls.append(extras) or ['overlaid'])
    assert library_adapter.library_adapter('monai', ['m1'], data=DataLayout(tmp_path)) == \
        ['overlaid'] and calls == [['monai==1.5.2']]
    monkeypatch.setenv(harvest_tiers.OVERLAY_FLAG, '1')  # inside the overlay: draft here
    monkeypatch.setattr(harvest_tiers, '_owned_elsewhere', lambda *args: False)
    monkeypatch.setattr(library_adapter, 'draft', lambda row, data, *, rebuild=False:
                        harvest_tiers._give_up(row, 'needs_adapter', 'fake', None))
    monkeypatch.setattr(library_adapter.harvest, 'write_drafts',
                        lambda drafts, **kw: list(drafts))
    out = library_adapter.library_adapter('monai', ['m1'], data=DataLayout(tmp_path))
    assert [d.row.id for d in out] == ['m1'] and calls == [['monai==1.5.2']]


@pytest.mark.parametrize(('value', 'left'), [
    ({'$ref': 'collections.OrderedDict'}, True),   # a class: the library's own component
    ({'$ref': 'math.pi'}, False),                  # a constant is a width-like choice
    ({'$call': 'os.getcwd', 'args': [], 'kwargs': {}}, False),
])
def test_only_a_class_is_left_to_the_caller(value: dict, left: bool) -> None:
    assert library_adapter._left_to_caller(None, value) is left


def _registry_row(monkeypatch: pytest.MonkeyPatch, name: str, zoo: str, model_type: str,
                  built: object) -> tuple[Row, list]:
    row = _row(name, 'import torch; model = torch.nn.Linear(2, 2)', zoo=zoo)  # a stand-in
    monkeypatch.setattr(library_adapter.registry, 'lookup', lambda name, zoo: (
        'transformers.AutoModel.from_config', [], {}, {'model_type': model_type}, {}))

    def built_class(head: str, kind: str) -> str:
        if isinstance(built, Exception):
            raise built
        return built

    monkeypatch.setattr(library_adapter.heads, 'built_class', built_class)
    seen = []

    def parsed_draft(row, parsed, data, *, rebuild, identified=False):
        from types import SimpleNamespace

        seen.append(parsed[0])
        return SimpleNamespace(give_up=None, delta={}, row=row)

    monkeypatch.setattr(library_adapter, '_parsed_draft', parsed_draft)
    return row, seen


@pytest.mark.parametrize(('name', 'zoo', 'call'), [
    ('wav2vec2', 'transformers:AutoModelForCTC', 'transformers.AutoModelForCTC.from_config'),
    ('udop:image-text-to-text', 'base-pytorch-compact',
     'transformers.AutoModelForImageTextToText.from_config'),
    ('udop:base-document-vlm', 'base-pytorch-compact', 'transformers.AutoModel.from_config'),
    ('clvp', 'transformers:ClvpModelForConditionalGeneration',
     'transformers.ClvpModelForConditionalGeneration'),
])
def test_a_registry_default_builds_the_class_the_zoo_names(
        monkeypatch: pytest.MonkeyPatch, name: str, zoo: str, call: str) -> None:
    row, seen = _registry_row(monkeypatch, name, zoo, 'udop', 'BuiltClass')
    made = library_adapter.draft(row, DATA)
    assert seen == [call] and made.give_up is None
    assert made.delta['built_class'] == 'BuiltClass' and made.delta['registry'].startswith(call)


@pytest.mark.parametrize(('zoo', 'name', 'built', 'why'), [
    ('transformers:AutoModelForCTC', 'bert', LookupError('AutoModelForCTC has no model for bert'),
     'AutoModelForCTC has no model for bert'),
    ('transformers:trust_remote_code', 'bert', 'BertModel', 'not a transformers class'),
    ('base-pytorch-compact', 'bert:frontier-segmentation', 'BertModel', 'no known transformers class'),
])
def test_a_registry_default_that_cannot_be_the_named_class_stays_open(
        monkeypatch: pytest.MonkeyPatch, zoo: str, name: str, built: object, why: str) -> None:
    row, seen = _registry_row(monkeypatch, name, zoo, 'bert', built)
    made = library_adapter.draft(row, DATA)
    assert seen == [] and made.give_up[0] in library_adapter.REGISTRY_RETRY
    assert why in made.give_up[1]  # the scripted verdict stands, so an agent is owed the row


def test_a_renamed_auto_class_is_drafted_as_its_successor(monkeypatch: pytest.MonkeyPatch) -> None:
    """transformers 5 has no AutoModelForVision2Seq; a row that names it builds its successor."""

    row, seen = _registry_row(monkeypatch, 'llava:vision-to-text', 'transformers:AutoModelForVision2Seq',
                              'llava', 'LlavaForConditionalGeneration')
    monkeypatch.setattr(library_adapter.heads, 'installed',
                        lambda head: library_adapter.heads.SUCCESSORS.get(head, head))
    made = library_adapter.draft(row, DATA)
    assert seen == ['transformers.AutoModelForImageTextToText.from_config'] and made.give_up is None

