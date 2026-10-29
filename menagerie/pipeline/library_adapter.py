"""Audited library rebuilds. No June statement or wrapper is ever executed."""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, now, read_json, write_json
from menagerie.pipeline import harvest, harvest_tiers as tiers, june_audit
from menagerie.pipeline import registry_names as registry
from menagerie.pipeline import transformers_heads as heads
from menagerie.pipeline.harvest_build import MissingLibrary, component, names_class
from menagerie.pipeline.records import (
    GiveUp, LedgerRow, ModelRecord, RecordError, from_dict,
)

REBUILD = frozenset({'standin:kwargs', 'standin:compact', 'standin:exec'})
RESIDUAL = REBUILD | {'needs_adapter', 'needs_adapter:inputs', 'missing_dependency',
                      'missing_dependency:sdist', 'june_code:flagged',
                      'standin:no_published_config'}
# Calls whose first argument selects the named model from the library registry.
SELECTOR_CALLS = frozenset({'timm.create_model', 'torchvision.models.get_model',
                            'open_clip.create_model'})
# Verdicts after which the row name is looked up in the installed registries.
REGISTRY_RETRY = frozenset({'june_code:flagged', 'standin:no_published_config'})


def _identity(row: Any, parsed: tuple) -> bool:
    """A default is the named model only when the name actually selects that library model."""

    call, args, kwargs, config = parsed[:4]
    # The row name with a zoo or library prefix, a `:<task>` and an `_hf` suffix dropped.
    names = set(registry.candidates(row.name, row.zoo, (call.split('.')[0],)))
    tail = registry.norm(call.rpartition('.')[2])
    if config:
        return (registry.norm(str(config.get('model_type', ''))) in names or
                tail in names and tail not in ('from_config', 'from_pretrained'))
    if call in SELECTOR_CALLS:
        return bool(args) and registry.norm(str(args[0])) in names
    if call.startswith('ultralytics.') and args and str(args[0]).endswith(('.yaml', '.yml')):
        return registry.norm(Path(str(args[0])).stem) in names
    if tail in names or registry.norm(call) in names:
        return True
    name = row.name.casefold().replace('_', '-').partition(':')[0]
    tail = tail.replace('_', '-')
    if call.startswith('neuralforecast.models.') and name == f'neuralforecast {tail}':
        return True
    if call.startswith('deepctr_torch.models.') and name == f'deepctr-torch.{tail}':
        return True
    if call.startswith('segmentation_models_pytorch.'):
        # Both constructor and encoder are selectors, never width/depth overrides.
        encoder = str(kwargs.get('encoder_name', '')).casefold().replace('_', '-')
        return bool(encoder) and tail in name and encoder in name
    return False


def _defaults(parsed: tuple) -> tuple:
    """Drop only optional architecture overrides; refuse unknown/required dimensions."""

    from menagerie.pipeline.harvest_build import _resolve

    call, args, kwargs, config, imported = parsed
    if config:
        return call, [], {}, {'model_type': config['model_type']}, imported
    params = tiers._parameters(_resolve(call))
    bound = tiers._bind(params, args, kwargs)
    kept = {}
    for name, value in bound.items():
        if name in tiers.SELECTORS or (call.startswith("neuralforecast.models.") and
                                      name in ("h", "input_size", "n_series")):
            kept[name] = value
        elif name in tiers.WEIGHT_SWITCHES:
            kept[name] = False if name.startswith('pretrained') else None
        elif name in params and tiers._same(value, params[name].default):
            kept[name] = value
        elif _left_to_caller(params.get(name), value):
            kept[name] = value
        elif name not in params or params[name].default is tiers.EMPTY:
            raise harvest.NoRecipe('standin:no_published_config',
                                   f'{call} needs published values for {name}')
    return call, [], kept, {}, imported


def _left_to_caller(param: Any, value: Any) -> bool:
    """A component the library leaves to the caller (default ``None`` or ``**kwargs``), named
    or built with no arguments of its own: kept, since a rebuild drops only sizes."""

    if param is not None and param.default is not None:
        return False
    return (component(value) and not value.get('args') and not value.get('kwargs')
            and names_class(value.get('$ref') or value.get('$call')))


def _parsed_draft(row: Any, parsed: tuple, data: DataLayout, *, rebuild: bool,
                  identified: bool = False) -> Any:
    call, args, kwargs, config = parsed[:4]
    if not identified and not _identity(row, parsed):
        raise harvest.NoRecipe('standin:no_published_config' if rebuild else 'june_code:flagged',
                               'the library reference config is not verified as this named model')
    if rebuild:
        call, args, kwargs, config = _defaults(parsed)[:4]
        if config:
            # Validate and size the default on meta before dropping June overrides.
            harvest._weights_gib({"kind": "library", "call": call, "config": config},
                                 data, strict=True)
    spec = row.catalog.get('input') or {}
    # Rebuilding the model never reuses June's shrunken tensor or sentinel dimensions.
    draft = harvest._library_draft(row, data, call, args, kwargs, config,
                                  spec if not rebuild else {})
    draft.recipe['extras'] = list(tiers._extras(call))
    draft.same_as_june = not rebuild
    if draft.recipe['input'].note.startswith('sentinel'):
        from menagerie.pipeline import library_inputs
        from menagerie.pipeline.records import RecipeInput

        from menagerie.pipeline.meta_build import MetaBuildRefused

        try:
            library_inputs.example_input(draft.recipe)
        except MetaBuildRefused as exc:
            raise harvest.NoRecipe(exc.cls, exc.reason) from exc
        except (RecordError, AttributeError, KeyError, ValueError, TypeError) as exc:
            raise harvest.NoRecipe('needs_adapter:inputs', str(exc)) from exc
        draft.recipe['input'] = RecipeInput(
            preset=library_inputs.preset(draft.recipe), kind='random-fallback',
            note='library input builder; dimensions from the actual model configuration; '
                 'canonical image/text where the interface supports them')
    return draft


def draft(row: Any, data: DataLayout, *, rebuild: bool = False) -> Any:
    """Return a faithful library draft or a typed verdict. Flagged code never has a recipe.

    A row built from our classics (group B) takes the classics path, whose recipes are
    reviewed. A June-code or stand-in verdict is retried as the row name in the installed
    registries (``registry_names``); a library that does not import is ``missing_dependency``.
    """

    if tiers.group(row.catalog) == 'B':
        return tiers.tier_draft(row, data)
    made = _audited_draft(row, data, rebuild=rebuild)
    if made.give_up and made.give_up[0] in REGISTRY_RETRY:
        named, note = _registry_draft(row, data)
        if named is not None:
            return named
        if note:  # the scripted verdict stands (an agent is owed the row); say why
            return replace(made, give_up=(made.give_up[0], f'{made.give_up[1]}; {note}'))
    return made


def _registry_draft(row: Any, data: DataLayout) -> tuple[Any | None, str]:
    """The registry default of the entry the row name is, or ``None`` and why not.

    A transformers model type is built as the class the row names (``transformers_heads``):
    the zoo's ``AutoModelFor*`` head, never the bare ``AutoModel`` in its place.
    """

    found = registry.lookup(row.name, row.zoo)
    if found is None:
        return None, ''
    built = ''
    if found[0].startswith('transformers.'):
        model_type = found[3]['model_type']
        head, where = heads.named_head(row.name, row.zoo, tiers.code(row.catalog) or '')
        if head is None:
            return None, f'registry model type {model_type}: {where}'
        if (now := heads.installed(head)) != head:
            head, where = now, f'{where}, which this transformers names {now}'
        try:
            built = heads.built_class(head, model_type)
        except LookupError as exc:
            return None, f'registry model type {model_type}: {where}, but {exc}'
        found = (heads.call_for(head), *found[1:])
    try:
        made = _parsed_draft(row, (*found[:4], {}), data, rebuild=True, identified=True)
    except harvest.NoRecipe as exc:
        return None, f'registry {found[0]}: {exc}'
    except Exception as exc:  # the default does not build here; the scripted verdict stands
        return None, f'registry {found[0]} did not build: {type(exc).__name__}: {exc}'[:300]
    made.delta = {**made.delta, 'registry': f'{found[0]} {found[1] or found[3]}',
                  **({'built_class': built} if built else {})}
    return made, ''


def _audited_draft(row: Any, data: DataLayout, *, rebuild: bool) -> Any:
    source = tiers.code(row.catalog)
    try:
        parsed = june_audit.extract(source)
        checked = None if rebuild else june_audit.audit(source)
        if checked is not None and checked.missing:
            # Never a June-code verdict while a library does not import, even in the overlay.
            raise harvest.NoRecipe('missing_dependency', '; '.join(checked.issues),
                                   'an environment where the library imports')
        if parsed is None:
            raise harvest.NoRecipe('standin:no_published_config' if rebuild else 'june_code:flagged',
                                   'no extractable named library constructor')
        if checked is not None and not checked.passed:
            raise harvest.NoRecipe('june_code:flagged', '; '.join(checked.issues) or
                                   'no verifiable library constructors')
        return _parsed_draft(row, parsed, data, rebuild=rebuild)
    except harvest.NoRecipe as exc:
        return tiers._give_up(row, exc.cls, exc.reason, exc.missing)
    except (ImportError, AttributeError, MissingLibrary) as exc:
        return tiers._give_up(row, 'missing_dependency',
                              f'library import failed: {type(exc).__name__}: {str(exc)[:160]}',
                              'an environment where the library imports')
    except (SyntaxError, RecordError, ValueError, TypeError) as exc:
        return tiers._give_up(row, 'june_code:flagged',
                              f'unproven construction: {type(exc).__name__}: {str(exc)[:160]}',
                              'a verified published config and input builder')


def library_adapter(zoo: str, names: Sequence[str], *, data: DataLayout,
                    write: bool = False) -> list[ModelRecord]:
    """Draft explicit residual ids only, preserving active recipes and foreign packets.

    Writes are opt-in; this does not queue or run any model. Recipes use fresh
    revisions through harvest.write_drafts, so old stand-in forwards never carry.
    A row whose drafting raises gets that typed result and the rest continue; finished
    drafts are written even when the loop itself stops. A zoo that needs libraries this
    interpreter lacks runs again inside their overlay (``MENAGERIE_ENVS``), as ``harvest tier``
    does, so a missing library is never read as a June-code departure.
    """

    if not names:
        raise RecordError('library-adapter needs explicit residual model ids')
    rows = tiers._tier_rows(zoo, names, data)
    extras = tiers._missing_extras(rows)
    if extras and not os.environ.get(tiers.OVERLAY_FLAG):
        overlaid = _in_overlay(zoo, names, extras, write=write)
        if overlaid is not None:
            return overlaid
    drafts, preserved = [], []
    try:
        for row in rows:
            try:
                kind, made = _one_row(row, data, write)
            except Exception as exc:  # one row's failure never stops the others
                kind, made = _row_failure(row, exc, data, write)
            if kind == 'held':
                preserved.append(made)
            elif kind == 'draft':
                drafts.append(made)
    finally:
        written = _write_by_family(drafts, data, write)
    return preserved + written


def _in_overlay(zoo: str, names: Sequence[str], extras: Sequence[str], *,
                write: bool) -> list[ModelRecord] | None:
    """The zoo's drafts run inside the overlay with ``extras``; ``None`` when the overlay
    will not install (its rows are then drafted here, and a library that does not import is
    typed ``missing_dependency``)."""

    from menagerie.pipeline import envs
    from menagerie.pipeline.harvest import ENV

    root = os.environ.get('MENAGERIE_ENVS')
    if not root:
        raise RecordError(f'zoo {zoo!r} needs {list(extras)}: set MENAGERIE_ENVS for the overlay')
    try:
        python = envs.overlay(ENV, extras, root=Path(root))
    except envs.ExtrasError as exc:
        print(f'library-adapter: zoo {zoo!r}: overlay {list(extras)} will not install: {exc}'[:400],
              file=sys.stderr)
        return None
    argv = [str(python), '-m', 'menagerie.pipeline', 'implement', 'library-adapter', zoo,
            *names, *(['--write'] if write else [])]
    done = subprocess.run(argv, capture_output=True, text=True, check=False,
                          env={**os.environ, tiers.OVERLAY_FLAG: '1'})
    sys.stderr.write(done.stderr[-4000:])
    if done.returncode != 0:
        raise RecordError(f'library-adapter {zoo!r} in its overlay exited {done.returncode}')
    return [from_dict(ModelRecord, json.loads(line)) for line in done.stdout.splitlines()
            if line.startswith('{')]


def _one_row(row: Any, data: DataLayout, write: bool) -> tuple[str, Any]:
    """One residual row: ``('skip', None)``, ``('held', record)`` or ``('draft', draft)``."""

    if tiers._owned_elsewhere(row, data):
        return 'skip', None
    old = _residual_record(row.id, data)
    if old is False:
        return 'skip', None
    rebuild = bool(old and old.give_up and old.give_up.cls in REBUILD)
    made = draft(row, data, rebuild=rebuild or tiers.group(row.catalog) == 'X')
    return _keep_or_draft(old, made, data, write)


def _keep_or_draft(old: ModelRecord | None, made: Any, data: DataLayout,
                   write: bool) -> tuple[str, Any]:
    if old and old.recipe and made.give_up:
        # Keep the typed revision identifiable when no replacement was established.
        held = _held_record(old, made)
        if write and held != old:
            write_json(held, data.model(old.id))
        return 'held', held
    return 'draft', made


def _row_failure(row: Any, exc: Exception, data: DataLayout, write: bool) -> tuple[str, Any]:
    """The typed result of a row whose drafting raised; an active record is never touched."""

    cls = 'memory_cap' if isinstance(exc, MemoryError) else 'construct_error'
    print(f'library-adapter: {row.id} ({row.name}): {cls}: {type(exc).__name__}: {exc}'[:400],
          file=sys.stderr)
    made = tiers._give_up(row, cls, f'drafting raised {type(exc).__name__}: {str(exc)[:160]}',
                          'a draft that builds')
    try:
        old = _residual_record(row.id, data)
    except Exception as again:  # an unreadable record is left as it is
        print(f'library-adapter: {row.id}: record unreadable, left as is: {again}'[:400],
              file=sys.stderr)
        return 'skip', None
    if old is False:
        return 'skip', None
    return _keep_or_draft(old, made, data, write)


def _write_by_family(drafts: list[Any], data: DataLayout, write: bool) -> list[ModelRecord]:
    """``harvest.write_drafts`` one family at a time, so one family's error keeps the others."""

    families: dict[str, list[Any]] = {}
    for made in drafts:
        families.setdefault(harvest._family_id(made.row.family), []).append(made)
    records = []
    for family, members in sorted(families.items()):
        try:
            records += harvest.write_drafts(members, data=data, write=write)
        except Exception as exc:  # retry the members alone; a failing one gets its typed result
            print(f'library-adapter: family {family} write failed, members alone: {exc}'[:400],
                  file=sys.stderr)
            records += [r for made in members for r in _write_alone(made, data, write)]
    return records


def _write_alone(made: Any, data: DataLayout, write: bool) -> list[ModelRecord]:
    try:
        return harvest.write_drafts([made], data=data, write=write)
    except Exception as exc:
        if made.give_up:
            print(f'library-adapter: {made.row.id}: typed result not written: {exc}'[:400],
                  file=sys.stderr)
            return []
        kind, typed = _row_failure(made.row, exc, data, write)
        if kind == 'held':
            return [typed]
        return _write_alone(typed, data, write) if kind == 'draft' else []


def _residual_record(model_id: str, data: DataLayout) -> ModelRecord | None | bool:
    """Return an eligible prior record, or False when an active record must be kept.

    An agent's give-up is kept too: only a scripted verdict is the script's to redo.
    """
    path = data.model(model_id)
    if not path.exists():
        return None
    old = read_json(ModelRecord, path)
    if not old.give_up or old.give_up.cls not in RESIDUAL:
        return False
    if _agent_typed(old, data):
        return False
    return old


def _agent_typed(record: ModelRecord, data: DataLayout) -> bool:
    from menagerie.pipeline import drive

    if record.give_up.by is not None:
        return record.give_up.by == 'agent'
    return drive.agent_tried(record, _giveup_rows(str(data.root)).get(record.id, ()))


@functools.lru_cache(maxsize=4)
def _giveup_rows(root: str) -> dict[str, list[LedgerRow]]:
    """The ledger's give-up rows by model id (a residual run appends none)."""
    from menagerie.pipeline import ledger

    found: dict[str, list[LedgerRow]] = {}
    for row in ledger.read_all(data=DataLayout(Path(root))):
        if isinstance(row, LedgerRow) and row.outcome == 'giveup':
            found.setdefault(row.id, []).append(row)
    return found


def _held_record(old: ModelRecord, made: Any) -> ModelRecord:
    # A substitute retains its original typed revision until a verified replacement exists.
    if old.give_up.cls.startswith('standin:') or made.give_up[0] != 'june_code:flagged':
        return old
    same = old.give_up.cls == 'june_code:flagged'
    at, by = (old.give_up.at, old.give_up.by) if same else (now(), 'script')
    return replace(old, give_up=GiveUp(cls='june_code:flagged', reason=made.give_up[1], at=at,
                                     missing=made.missing, by=by))
