"""Fail-closed AST audit of June construction, without executing its statements.

Only extracted library calls are reused. Wrappers, monkey patches and setup code
are never copied into an adapter. Every constructor and configuration mutation
contributes to the verdict, including constructors hidden inside lambdas.
"""

from __future__ import annotations

import ast
import inspect
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from menagerie.pipeline import harvest_tiers as tiers


# The issue a constructor gets when its library does not import here: an environment fact,
# typed `missing_dependency` by the caller, never a June-code departure.
MISSING = 'library does not import here'
UNBOUND = 'unbound name: the June code does not import it'


@dataclass(frozen=True)
class Audit:
    calls: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return bool(self.calls) and not self.issues

    @property
    def missing(self) -> tuple[str, ...]:
        """The issues that only say a library would not import."""
        return tuple(i for i in self.issues if MISSING in i)

    @property
    def departures(self) -> tuple[str, ...]:
        """The issues about the code itself."""
        return tuple(i for i in self.issues if MISSING not in i)

def _aliases(tree: ast.Module) -> dict[str, str]:
    names = {}
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for alias in node.names:
            if alias.name != '*':
                local, qualified = _import_name(node, alias)
                names[local] = qualified
    return names

def _import_roots(tree: ast.Module) -> frozenset[str] | None:
    """Every name and module any import binds, at any depth (``try``/``if`` included); ``None``
    when a star import could bind any name."""

    roots: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for alias in node.names:
            if alias.name == '*':
                return None
            local, qualified = _import_name(node, alias)
            roots.update((local, qualified))
    return frozenset(roots)

def _rooted(call: str, roots: frozenset[str] | None) -> bool:
    """Whether some import (or a star import, ``None``) can have bound the call's name."""

    return roots is None or any(call == r or call.startswith(r + '.') for r in roots)

def _import_name(node: ast.Import | ast.ImportFrom, alias: ast.alias) -> tuple[str, str]:
    if isinstance(node, ast.ImportFrom):
        return alias.asname or alias.name, f'{node.module}.{alias.name}'
    head = alias.name.split('.')[0]
    return (alias.asname, alias.name) if alias.asname else (head, head)

def _qualified(node: ast.expr, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _qualified(node.value, aliases)
        return f'{base}.{node.attr}' if base else ''
    return ''

def _constructor(call: str) -> bool:
    tail = call.rpartition('.')[2]
    library_factory = call.startswith(('torchvision.models.', 'timm.create_model',
                                     'segmentation_models_pytorch.'))
    return bool(tail and (library_factory or tail[0].isupper() or tail in (
        'create_model', 'get_model', 'for_model', 'from_config', 'from_pretrained')))

def extract(source: str) -> tuple | None:
    """Inline config bindings for parsing, never executing or mutating the AST input."""

    tree = ast.parse(source.strip())
    configs = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Call) for t in n.targets
               if isinstance(t, ast.Name) and _config_node(n.value)}

    class Inline(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            return configs.get(node.id, node) if isinstance(node.ctx, ast.Load) else node

    return tiers.parse_model_call(ast.unparse(Inline().visit(tree)))

def _config_node(node: ast.expr) -> bool:
    return isinstance(node, ast.Call) and (
        ast.unparse(node.func).endswith('Config') or
        ast.unparse(node.func).endswith('AutoConfig.for_model'))

def audit(source: str) -> Audit:
    """Inspect all constructor arguments against installed defaults, or refuse.

    Required task dimensions may configure template operators. Other required
    architectural values need published-config evidence and are held for 2b.
    """

    try:
        tree = ast.parse(source.strip())
    except SyntaxError as exc:
        return Audit((), (f'invalid June syntax at line {exc.lineno}',))
    aliases = _aliases(tree)
    modules = tuple(dict.fromkeys([*aliases.values(), *tiers._star_modules(tree)]))
    configs = {t.id for n in tree.body if isinstance(n, ast.Assign)
               and _config_node(n.value) for t in n.targets if isinstance(t, ast.Name)}
    selected = extract(source)
    selected_call = selected[0] if selected else ''
    calls, issues = [], _source_issues(tree, configs, aliases)
    roots = _import_roots(tree)
    for node, call in _resolved_calls(tree, aliases):
        if not (_constructor(call) or call == selected_call or _imported_call(call, modules)):
            continue
        calls.append(call)
        issues.extend(_check(node, call, aliases, configs, roots))
    return Audit(tuple(calls), tuple(dict.fromkeys(issues)))

def _resolved_calls(tree: ast.Module, aliases: dict[str, str]) -> Iterator[tuple[ast.Call, str]]:
    """Resolve imported names once, excluding local and call-result receivers."""
    local, stars = tiers._bound_names(tree), tiers._star_modules(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        raw = _qualified(node.func, aliases)
        if not raw:
            continue
        call = tiers._qualify(raw, local, stars)
        if call is not None:
            yield node, call

def _imported_call(call: str, modules: tuple[str, ...]) -> bool:
    # Tensor input construction is separate from model construction; everything else
    # exported by an imported library must prove its arguments, including lambdas.
    if call.startswith('torch.') and not call.startswith('torch.nn.'):
        return False
    # `from m import f; f(...)` resolves to an imported name itself: it is audited too.
    return any(call == module or call.startswith(module + '.') for module in modules)

def _source_issues(tree: ast.Module, configs: set[str], aliases: dict[str, str]) -> list[str]:
    """Reject mutable setup and dynamic execution independently of call extraction."""

    writes = Counter(n.id for n in ast.walk(tree)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    issues = [f'configuration binding reassigned: {name}'
              for name in configs if writes[name] != 1]
    issues.extend(_call_issues(tree, aliases, set(writes)))
    for node in ast.walk(tree):
        if _object_write(node):
            issues.append(f'configuration or model mutation at line {node.lineno}')
        if isinstance(node, ast.Call) and _mutates(node, set(writes)):
            issues.append(f'unproven mutation or dynamic execution at line {node.lineno}')
    return issues


def _call_issues(tree: ast.Module, aliases: dict[str, str], writes: set[str]) -> list[str]:
    """Imported identity must survive binding changes and every call-site lookup."""

    shadowed = writes | {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    issues = [f'import binding reassigned or shadowed: {name}'
              for name in shadowed.intersection(aliases)]
    local = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    modules = (*aliases.values(), *tiers._star_modules(tree))
    resolved = {id(n) for n, call in _resolved_calls(tree, aliases)
                if _import_identity(call, modules)}
    return issues + [f'unresolved call target at line {node.lineno}'
                     for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and _unresolved(node, local, resolved)]

def _import_identity(call: str, modules: tuple[str, ...]) -> bool:
    return any(call == module or call.startswith(module + '.') for module in modules)

def _unresolved(node: ast.Call, local: set[str], resolved: set[int]) -> bool:
    """Local aliases and computed receivers cannot establish constructor identity."""

    if isinstance(node.func, ast.Name):
        return id(node) not in resolved and node.func.id not in local
    if isinstance(node.func, ast.Attribute):
        if node.func.attr == 'eval' and not node.args and not node.keywords:
            return False
        return id(node) not in resolved
    return True

def _object_write(node: ast.AST) -> bool:
    if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
        return False
    targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
    return any(isinstance(t, (ast.Attribute, ast.Subscript)) for t in targets)

def _mutates(node: ast.Call, bound: set[str]) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id in ('exec', 'eval', 'setattr', 'delattr', 'getattr', '__import__') and not _wrapper_binding(node)
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr in ('import_module', 'update', 'add_module', 'register_module', 'register_parameter',
                          '__setattr__', '__delattr__', '__setitem__', '__delitem__'):
        return True
    # A call on a constructed object can change architecture even without an assignment.
    return (isinstance(node.func.value, ast.Name) and node.func.value.id in bound
            and node.func.attr != 'eval')

def _wrapper_binding(node: ast.Call) -> bool:
    return (len(node.args) == 3 and isinstance(node.args[0], ast.Name)
            and node.args[0].id == 'self' and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in ('m', 'core', 'model'))

def _values(node: ast.Call, aliases: dict[str, str]) -> tuple[list, dict]:
    """Literal arguments; an imported object as ``$ref``; a call of an imported name as
    ``$call`` (audited as a call of its own); ``dict(k=v)`` and list or tuple displays."""

    if any(k.arg is None for k in node.keywords):
        raise ValueError('unresolved **kwargs')
    return ([_argument(a, aliases) for a in node.args],
            {k.arg: _argument(k.value, aliases) for k in node.keywords})


def _imported(ref: str, aliases: dict[str, str]) -> bool:
    return bool(ref) and '.' in ref and any(
        ref == module or ref.startswith(module + '.') for module in aliases.values())


def _argument(value: ast.expr, aliases: dict[str, str]) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError):
        pass
    if isinstance(value, (ast.List, ast.Tuple)):
        return [_argument(e, aliases) for e in value.elts]
    if isinstance(value, ast.Call) and not any(k.arg is None for k in value.keywords):
        return _call_argument(value, aliases)
    ref = _qualified(value, aliases)
    if _imported(ref, aliases):
        return {'$ref': ref}
    raise ValueError('nonliteral constructor argument')


def _call_argument(value: ast.Call, aliases: dict[str, str]) -> Any:
    keywords = {k.arg: _argument(k.value, aliases) for k in value.keywords}
    if isinstance(value.func, ast.Name) and value.func.id == 'dict' and 'dict' not in aliases:
        if value.args:
            raise ValueError('dict(...) with positional arguments')
        return keywords
    ref = _qualified(value.func, aliases)
    if not _imported(ref, aliases):
        raise ValueError('nonliteral constructor argument')
    return {'$call': ref, 'args': [_argument(a, aliases) for a in value.args], 'kwargs': keywords}


def _check(node: ast.Call, call: str, aliases: dict[str, str], configs: set[str],
           roots: frozenset[str] | None) -> list[str]:
    from menagerie.pipeline.harvest_build import MissingLibrary, _resolve

    if call.endswith('.from_pretrained'):
        return [f'{call}: model weight/config download is not an offline named config']
    if call.startswith('torch.nn.') and not call.endswith('Module'):
        return [f'{call}: handwritten network requires a fidelity review']
    if _config_argument(node, configs):
        # The separate AST visit audits the config constructor and all binding mutations.
        return []
    try:
        args, kwargs = _values(node, aliases)
        fn = _resolve(call)
        params = tiers._parameters(fn)
        if not params and (args or kwargs):
            return [f'{call}: no inspectable defaults']
        bound = tiers._bind(params, args, kwargs)
        bad = [name for name, value in bound.items()
               if not _default(params.get(name), name, value, call)]
        return [f'{call}: nondefault or unproven argument {name}' for name in bad]
    except (ImportError, MissingLibrary) as exc:
        if not _rooted(call, roots):
            # A bare name June's code never imports (`AutoConfig.for_model(...)` with no import)
            # is a property of the code, not a library missing from this environment.
            return [f'{call}: {UNBOUND}']
        return [f'{call}: {MISSING} ({type(exc).__name__})']
    except (AttributeError, ValueError, TypeError) as exc:
        return [f'{call}: cannot verify constructor ({type(exc).__name__})']

def _config_argument(node: ast.Call, configs: set[str]) -> bool:
    if len(node.args) != 1 or node.keywords:
        return False
    value = node.args[0]
    return _config_node(value) or isinstance(value, ast.Name) and value.id in configs

def _default(param: inspect.Parameter | None, name: str, value: Any, call: str) -> bool:
    if name in tiers.WEIGHT_SWITCHES:
        return value in (None, False, '')  # no enabled pretrained switch survives
    if name in tiers.SELECTORS or name == 'model_type':
        return True
    if call.startswith('neuralforecast.models.') and name in ('h', 'input_size', 'n_series'):
        return True  # dataset horizon and window dimensions, not network widths
    if _own_component(param, value, call):
        return True
    return param is not None and tiers._same(value, param.default)


def _own_component(param: inspect.Parameter | None, value: Any, call: str) -> bool:
    """A component the library leaves to the caller (default ``None``, or passed through its
    ``**kwargs``), chosen from the same library: ``spiking_neuron=neuron.IFNode``. A width or
    depth is a literal, never a component; a ``$call``'s own arguments are audited as a call."""

    from menagerie.pipeline.harvest_build import CALL, REF, component, names_class

    if not component(value) or (param is not None and param.default is not None):
        return False
    target = value.get(REF) or value.get(CALL) or ''
    return target.split('.')[0] == call.split('.')[0] and names_class(target)
