"""Tests for closed recipes and typed adapters."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from menagerie.crawler.identity import compute_recipe_revision, hash_bytes
from menagerie.crawler.recipe import (
    DeclarativeRecipe,
    RecipeError,
    load_declarative_recipe,
    load_recipe,
    load_typed_adapter,
    reject_opaque_recipe,
    validate_pretrained_disable_fields,
    validate_pretrained_disposition,
)


def test_opaque_eval_and_exec_recipes_are_rejected() -> None:
    """Legacy executable strings never enter the Slice B recipe path."""

    with pytest.raises(RecipeError):
        reject_opaque_recipe({"type": "expression", "code": "eval('bad')"})
    with pytest.raises(RecipeError):
        reject_opaque_recipe({"exec": "model = bad()"})


def test_declarative_r1_loads_direct_constructor() -> None:
    """A closed R1 recipe imports a direct symbol and applies JSON kwargs."""

    loaded = load_declarative_recipe(
        {
            "distribution": "torch",
            "version": torch.__version__,
            "module": "torch.nn",
            "symbol": "Linear",
            "kwargs": {"in_features": 4, "out_features": 2},
            "pretrained_disable_fields": [],
        }
    )
    model = loaded.build_model()

    assert isinstance(model, torch.nn.Linear)
    assert model.in_features == 4
    assert model.out_features == 2


@pytest.mark.smoke
def test_asserted_pretrained_fields_absent_loads_a_config_only_constructor() -> None:
    """The checked absence assertion is the R1 path for nothing-to-disable shapes.

    ``torch.nn.Linear`` stands in for every ``XForCausalLM(config)``, GNN layer,
    and SNN module whose constructor exposes no pretrained-capable keyword: the
    assertion is verified against the real signature and the build proceeds.
    """

    loaded = load_declarative_recipe(
        {
            "distribution": "torch",
            "version": torch.__version__,
            "module": "torch.nn",
            "symbol": "Linear",
            "kwargs": {"in_features": 4, "out_features": 2},
            "pretrained_disable_fields": [],
            "pretrained_fields_absent": True,
        }
    )
    assert isinstance(loaded.build_model(), torch.nn.Linear)


@pytest.mark.smoke
def test_false_absence_assertion_is_refused_by_the_real_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asserting absence over a constructor that exposes ``pretrained`` refuses.

    The assertion is a checked fact, not a recorded one: the load-time signature
    scan proves it false and names the remedy, so the assertion can never paper
    over a real pretrained flag.
    """

    class FakePretrainedNet(torch.nn.Module):
        def __init__(self, pretrained: bool = False) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    monkeypatch.setattr(torch.nn, "FakePretrainedNet", FakePretrainedNet, raising=False)
    with pytest.raises(RecipeError, match="contradicted by the pinned constructor signature"):
        load_declarative_recipe(
            {
                "distribution": "torch",
                "version": torch.__version__,
                "module": "torch.nn",
                "symbol": "FakePretrainedNet",
                "kwargs": {},
                "pretrained_disable_fields": [],
                "pretrained_fields_absent": True,
            }
        )


@pytest.mark.smoke
def test_absence_assertion_contradicting_disable_fields_is_refused() -> None:
    """Declaring both an absence and a disable list is a contradiction."""

    with pytest.raises(RecipeError, match="contradicts"):
        DeclarativeRecipe.from_mapping(
            {
                "distribution": "torch",
                "version": torch.__version__,
                "module": "torch.nn",
                "symbol": "Linear",
                "kwargs": {"weights": None},
                "pretrained_disable_fields": ["weights"],
                "pretrained_fields_absent": True,
            }
        )


@pytest.mark.smoke
def test_pretrained_capable_kwargs_key_cannot_ride_through_unlisted() -> None:
    """A known pretrained-asset key never rides through the declaration unlisted.

    Two distinct refusals close the historical dodge where listing a harmless
    field satisfied the non-empty rule while ``weights`` carried an enabling
    value unlisted: an ENABLING value on a known key is refused outright as the
    safety hazard it is, and even a DISABLING value on a known key is refused
    until the key is declared in ``pretrained_disable_fields``, so the disable
    list stays a complete account of the constructor's known pretrained surface.
    """

    with pytest.raises(RecipeError, match="leave known pretrained keywords enabled"):
        validate_pretrained_disposition(
            {"weights": "IMAGENET1K_V1", "progress": False}, ["progress"], fields_absent=False
        )
    with pytest.raises(RecipeError, match="pretrained-capable keys"):
        validate_pretrained_disposition(
            {"weights": None, "progress": False}, ["progress"], fields_absent=False
        )
    # The complete disabling declaration for the same key remains satisfiable.
    validate_pretrained_disposition(
        {"weights": None, "progress": False}, ["weights"], fields_absent=False
    )


@pytest.mark.smoke
def test_enabled_pretrained_flag_still_refuses_even_when_listed() -> None:
    """The original tripwire fires unchanged: a listed field must disable."""

    with pytest.raises(RecipeError, match="does not carry a disabling value"):
        validate_pretrained_disable_fields({"pretrained": True}, ["pretrained"])


@pytest.mark.smoke
def test_absence_assertion_is_omitted_from_default_recipe_payloads() -> None:
    """Historical recipes keep their exact recipe-revision hash preimage."""

    recipe = DeclarativeRecipe.from_mapping(
        {
            "distribution": "torch",
            "version": torch.__version__,
            "module": "torch.nn",
            "symbol": "Linear",
            "kwargs": {"in_features": 4, "out_features": 2},
            "pretrained_disable_fields": [],
        }
    )
    assert "pretrained_fields_absent" not in recipe.to_dict()
    asserted = DeclarativeRecipe.from_mapping(
        {**recipe.to_dict(), "pretrained_fields_absent": True}
    )
    assert asserted.to_dict()["pretrained_fields_absent"] is True


@pytest.mark.smoke
def test_m4334_var_keyword_pretrained_disable_fields_are_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact m4334 recipe shape passes validation and builds.

    Rung-11 census defect class 3: eleven proven-runnable timm models
    (m4334 dla60 first ran verified in rung 9b) terminalized on
    ``pretrained disable fields are not explicit constructor parameters:
    ['pretrained_cfg']`` -- but timm entrypoints are
    ``dla60(pretrained=False, **kwargs)`` and DO honor ``pretrained_cfg``
    through ``build_model_with_cfg``. A disable field satisfied via the
    entrypoint's ``**kwargs`` is accepted; its disabling VALUE stays enforced.
    """

    def dla60(pretrained: bool = False, **kwargs: object) -> torch.nn.Linear:
        assert pretrained is False
        assert kwargs.get("pretrained_cfg") is None
        return torch.nn.Linear(4, 2)

    monkeypatch.setattr(torch.nn, "dla60", dla60, raising=False)
    loaded = load_declarative_recipe(
        {
            "distribution": "torch",
            "version": torch.__version__,
            "module": "torch.nn",
            "symbol": "dla60",
            "kwargs": {"pretrained": False, "pretrained_cfg": None},
            "pretrained_disable_fields": ["pretrained", "pretrained_cfg"],
        }
    )
    assert isinstance(loaded.build_model(), torch.nn.Linear)


@pytest.mark.smoke
def test_unforwardable_disable_field_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disable field the constructor can neither name nor absorb stays refused."""

    def rigid_net(pretrained: bool = False) -> torch.nn.Linear:
        return torch.nn.Linear(4, 2)

    monkeypatch.setattr(torch.nn, "rigid_net", rigid_net, raising=False)
    with pytest.raises(RecipeError, match=r"accepts no \*\*kwargs to forward them"):
        load_declarative_recipe(
            {
                "distribution": "torch",
                "version": torch.__version__,
                "module": "torch.nn",
                "symbol": "rigid_net",
                "kwargs": {"pretrained": False, "pretrained_cfg": None},
                "pretrained_disable_fields": ["pretrained", "pretrained_cfg"],
            }
        )


@pytest.mark.smoke
def test_var_keyword_acceptance_still_enforces_disabling_values() -> None:
    """**kwargs forwarding never loosens the disabling-VALUE tripwire."""

    with pytest.raises(RecipeError, match="does not carry a disabling value"):
        load_declarative_recipe(
            {
                "distribution": "torch",
                "version": torch.__version__,
                "module": "torch.nn",
                "symbol": "Linear",
                "kwargs": {"in_features": 4, "out_features": 2, "pretrained_cfg": {"url": "x"}},
                "pretrained_disable_fields": ["pretrained_cfg"],
            }
        )


def test_typed_adapter_loads_required_functions(tmp_path: Path) -> None:
    """A statically safe typed module supplies both required typed entry points."""

    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import torch

def build_model() -> object:
    return torch.nn.Linear(4, 2)

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return ((torch.randn(1, 4, generator=generator, device=device),), {})
""",
        encoding="utf-8",
    )

    loaded = load_typed_adapter(adapter)
    args, kwargs = loaded.make_dummy_call(3, "cpu") if loaded.make_dummy_call else ((), {})

    assert isinstance(loaded.build_model(), torch.nn.Linear)
    assert tuple(getattr(args[0], "shape")) == (1, 4)
    assert kwargs == {}


_SIBLING_ADAPTER = """from __future__ import annotations
import flow_model
import torch

def build_model() -> object:
    return flow_model.build()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return ((torch.randn(1, 4, generator=generator, device=device),), {})
"""

_SIBLING_MODULE = """import torch

def build():
    return torch.nn.Linear(4, 2)
"""


def _write_sibling_closure(tmp_path: Path) -> tuple[Path, Path]:
    """Stage the m11584 shape: an adapter importing a manifest sibling."""

    adapter = tmp_path / "adapter.py"
    adapter.write_text(_SIBLING_ADAPTER, encoding="utf-8")
    sibling = tmp_path / "flow_model.py"
    sibling.write_text(_SIBLING_MODULE, encoding="utf-8")
    return adapter, sibling


def test_typed_adapter_imports_manifest_sibling_module(tmp_path: Path) -> None:
    """An adapter importing a staged sibling loads under the manifest root.

    This is the exact m11584 (zuko_gf) shape from the rung-11 census:
    ``flow_model.py`` sat digest-verified in the bound code manifest beside
    the adapter, yet ``import flow_model`` died ``ModuleNotFoundError``
    because the loader made no importable code root.
    """

    adapter, _sibling = _write_sibling_closure(tmp_path)
    loaded = load_typed_adapter(adapter, manifest_root=tmp_path)

    assert isinstance(loaded.build_model(), torch.nn.Linear)
    assert str(tmp_path) not in sys.path, "the import root must not outlive the exec window"
    assert "flow_model" not in sys.modules, "staged siblings must not stay cached"


def test_typed_adapter_imports_manifest_package_tree(tmp_path: Path) -> None:
    """A vendored package tree (recbole shape) is importable under the root."""

    package = tmp_path / "vendored"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "nets.py").write_text(_SIBLING_MODULE, encoding="utf-8")
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
from vendored.nets import build
import torch

def build_model() -> object:
    return build()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return ((torch.randn(1, 4, generator=generator, device=device),), {})
""",
        encoding="utf-8",
    )

    loaded = load_typed_adapter(adapter, manifest_root=tmp_path)

    assert isinstance(loaded.build_model(), torch.nn.Linear)
    assert str(tmp_path) not in sys.path
    assert "vendored" not in sys.modules
    assert "vendored.nets" not in sys.modules


def test_typed_adapter_without_manifest_root_stays_fail_closed(tmp_path: Path) -> None:
    """Without a declared manifest the sibling import still fails closed."""

    adapter, _sibling = _write_sibling_closure(tmp_path)
    with pytest.raises(ModuleNotFoundError):
        load_typed_adapter(adapter)
    assert str(tmp_path) not in sys.path


def test_typed_adapter_import_failure_restores_sys_path(tmp_path: Path) -> None:
    """A failing adapter exec never leaks the import root or cached modules."""

    adapter = tmp_path / "adapter.py"
    adapter.write_text("import flow_model\nimport missing_module_xyz\n", encoding="utf-8")
    (tmp_path / "flow_model.py").write_text(_SIBLING_MODULE, encoding="utf-8")
    with pytest.raises(ModuleNotFoundError, match="missing_module_xyz"):
        load_typed_adapter(adapter, manifest_root=tmp_path)
    assert str(tmp_path) not in sys.path
    assert "flow_model" not in sys.modules


def test_adapter_functions_can_issue_lazy_stdlib_imports(tmp_path: Path) -> None:
    """The adapter module dict carries ``__builtins__`` like a real module.

    ``FunctionType(code, module.__dict__)()`` never inserts ``__builtins__``
    into the globals dict, so a C-level lazy import issued FROM an adapter
    function frame died ``KeyError: '__builtins__'`` -- the live m5445 case:
    ``datetime.timetuple()`` imports ``time`` through the calling frame's
    builtins inside ``make_dummy_call``.
    """

    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
from datetime import datetime
import torch

def build_model() -> object:
    return torch.nn.Linear(4, 2)

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    day = datetime.fromisoformat('2016-07-01T00:00:00').timetuple().tm_yday
    generator = torch.Generator(device=device).manual_seed(seed + day)
    return ((torch.randn(1, 4, generator=generator, device=device),), {})
""",
        encoding="utf-8",
    )

    loaded = load_typed_adapter(adapter)
    assert loaded.make_dummy_call is not None
    args, kwargs = loaded.make_dummy_call(3, "cpu")
    assert tuple(getattr(args[0], "shape")) == (1, 4)
    assert kwargs == {}


def test_load_recipe_derives_manifest_root_from_code_manifest(tmp_path: Path) -> None:
    """The worker-request mapping alone makes the staged closure importable."""

    adapter, sibling = _write_sibling_closure(tmp_path)
    loaded = load_recipe(
        {
            "kind": "typed-adapter",
            "path": str(adapter),
            "adapter_sha256": hash_bytes(adapter.read_bytes()),
            "code_manifest": [
                {"path": str(sibling), "sha256": hash_bytes(sibling.read_bytes())},
                {"path": str(adapter), "sha256": hash_bytes(adapter.read_bytes())},
            ],
        }
    )
    assert isinstance(loaded.build_model(), torch.nn.Linear)


def test_mapping_recipe_revision_pin_survives_a_permissive_caller(tmp_path: Path) -> None:
    """A stale mapping-level pin refuses even when the caller drops its own pin.

    The worker's permissive retry re-loads with ``expected_recipe_revision=None``
    (rung-11 census defect class 4 masked every R2 root cause behind it). The
    admission-bound pin rides the recipe mapping, so the retry cannot shed it.
    """

    adapter, sibling = _write_sibling_closure(tmp_path)
    mapping = {
        "kind": "typed-adapter",
        "path": str(adapter),
        "adapter_sha256": hash_bytes(adapter.read_bytes()),
        "recipe_revision": "sha256:" + "0" * 64,
        "code_manifest": [
            {"path": str(sibling), "sha256": hash_bytes(sibling.read_bytes())},
        ],
    }
    with pytest.raises(RecipeError, match="recipe revision mismatch"):
        load_recipe(mapping, source_identity="sha256:" + "a" * 64)
    with pytest.raises(RecipeError, match="recipe revision mismatch"):
        load_recipe(
            mapping,
            source_identity="sha256:" + "a" * 64,
            expected_recipe_revision=None,
        )


def test_mapping_recipe_revision_pin_accepts_the_admission_derivation(
    tmp_path: Path,
) -> None:
    """The exact adapter-flavor pin bound at admission loads cleanly."""

    adapter, sibling = _write_sibling_closure(tmp_path)
    source_identity = "sha256:" + "a" * 64
    pin = compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": adapter.name},
        source_identity,
        adapter_bytes=adapter.read_bytes(),
    )
    loaded = load_recipe(
        {
            "kind": "typed-adapter",
            "path": str(adapter),
            "adapter_sha256": hash_bytes(adapter.read_bytes()),
            "recipe_revision": pin,
            "code_manifest": [
                {"path": str(sibling), "sha256": hash_bytes(sibling.read_bytes())},
            ],
        },
        source_identity=source_identity,
    )
    assert loaded.recipe_revision == pin
    assert isinstance(loaded.build_model(), torch.nn.Linear)


def test_preexisting_module_cannot_shadow_manifest_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker-startup module with a colliding name never satisfies the manifest.

    Sol wave-8 finding 3 (HIGH), executable counterexample replayed: Python
    resolves imports from ``sys.modules`` BEFORE the prepended manifest root,
    so a preloaded generic name (``flow_model``) silently bound the receipt to
    bytes the manifest never declared. The owned name is displaced for the
    window, the digest-verified sibling wins, and the foreign module is
    restored intact afterwards.
    """

    import types

    poison = types.ModuleType("flow_model")
    poison.ORIGIN = "preexisting"
    monkeypatch.setitem(sys.modules, "flow_model", poison)

    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import flow_model
import torch

def build_model() -> object:
    assert flow_model.ORIGIN == "manifest", flow_model.ORIGIN
    return flow_model.build()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return ((torch.randn(1, 4, generator=generator, device=device),), {})
""",
        encoding="utf-8",
    )
    (tmp_path / "flow_model.py").write_text(
        'import torch\nORIGIN = "manifest"\n\ndef build():\n    return torch.nn.Linear(4, 2)\n',
        encoding="utf-8",
    )

    loaded = load_typed_adapter(adapter, manifest_root=tmp_path)

    assert isinstance(loaded.build_model(), torch.nn.Linear)
    assert sys.modules["flow_model"] is poison, (
        "the displaced foreign module must be restored intact after the window"
    )
    assert getattr(sys.modules["flow_model"], "ORIGIN") == "preexisting"


def test_owned_name_resolving_outside_the_root_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest-owned name never falls through to foreign ``sys.path`` bytes.

    The manifest declares ``flow_model.py`` but the staged file is absent. The
    window's own finder claims every owned name, so the import dies typed on
    the declared member's bytes -- the foreign ``sys.path`` copy is never even
    consulted, let alone silently bound.
    """

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "flow_model.py").write_text('ORIGIN = "foreign"\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(foreign))

    root = tmp_path / "model"
    root.mkdir()
    adapter = root / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import flow_model
import torch

def build_model() -> object:
    return torch.nn.Linear(4, 2)

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return ((torch.randn(1, 4, generator=generator, device=device),), {})
""",
        encoding="utf-8",
    )

    with pytest.raises(
        RecipeError, match="staged manifest member for module 'flow_model' is unreadable"
    ):
        load_typed_adapter(
            adapter,
            manifest_root=root,
            manifest_member_paths=[root / "flow_model.py"],
        )
    assert "flow_model" not in sys.modules, "the poisoned binding must not stay cached"
    assert str(root) not in sys.path


def test_lazy_adapter_imports_resolve_manifest_siblings(tmp_path: Path) -> None:
    """Imports issued lazily inside adapter entry points resolve the manifest.

    Sol wave-8 functionality edge: an import first executed inside
    ``build_model``/``make_dummy_call`` runs AFTER the exec window closed and
    died ``ModuleNotFoundError``. The window now re-opens around every entry
    point call, and the privately cached sibling is reused (module-level code
    executes exactly once across calls).
    """

    marker = tmp_path / "exec-count.txt"
    (tmp_path / "flow_model.py").write_text(
        "import pathlib\n"
        "import torch\n"
        f"pathlib.Path({str(marker)!r}).open('a').write('x')\n"
        "def build():\n"
        "    return torch.nn.Linear(4, 2)\n",
        encoding="utf-8",
    )
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import torch

def build_model() -> object:
    import flow_model
    return flow_model.build()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    import flow_model
    assert callable(flow_model.build)
    generator = torch.Generator(device=device).manual_seed(seed)
    return ((torch.randn(1, 4, generator=generator, device=device),), {})
""",
        encoding="utf-8",
    )

    loaded = load_typed_adapter(adapter, manifest_root=tmp_path)

    assert isinstance(loaded.build_model(), torch.nn.Linear)
    assert loaded.make_dummy_call is not None
    args, _kwargs = loaded.make_dummy_call(3, "cpu")
    assert tuple(getattr(args[0], "shape")) == (1, 4)
    assert marker.read_text() == "x", (
        "the cached sibling must be reused across calls, never re-executed"
    )
    assert "flow_model" not in sys.modules
    assert str(tmp_path) not in sys.path


def test_manifest_modules_never_leak_between_adapters(tmp_path: Path) -> None:
    """Two same-named siblings from different closures stay fully isolated."""

    def stage(name: str, origin: str) -> Path:
        root = tmp_path / name
        root.mkdir()
        (root / "flow_model.py").write_text(
            f'import torch\nORIGIN = "{origin}"\n\n'
            "def build():\n    return torch.nn.Linear(4, 2)\n",
            encoding="utf-8",
        )
        adapter = root / "adapter.py"
        adapter.write_text(
            f"""from __future__ import annotations
import flow_model
import torch

def build_model() -> object:
    assert flow_model.ORIGIN == "{origin}", flow_model.ORIGIN
    return flow_model.build()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return ((torch.randn(1, 4, generator=generator, device=device),), {{}})
""",
            encoding="utf-8",
        )
        return adapter

    loaded_alpha = load_typed_adapter(stage("alpha", "alpha"), manifest_root=tmp_path / "alpha")
    loaded_beta = load_typed_adapter(stage("beta", "beta"), manifest_root=tmp_path / "beta")

    assert isinstance(loaded_alpha.build_model(), torch.nn.Linear)
    assert isinstance(loaded_beta.build_model(), torch.nn.Linear)
    assert isinstance(loaded_alpha.build_model(), torch.nn.Linear)
    assert "flow_model" not in sys.modules


def test_forged_meta_path_loader_cannot_substitute_manifest_bytes(tmp_path: Path) -> None:
    """A foreign import hook forging __file__/origin never wins an owned name.

    Sol wave-8 R2 finding 4 (HIGH), probe replayed: a ``sys.meta_path`` finder
    for manifest-owned ``flow_model`` executed IN-MEMORY foreign bytes while
    pointing ``__file__`` and the spec origin at the real staged file, and the
    old post-hoc metadata check accepted it. Resolution is now CONTROLLED: the
    window's own finder sits first and loads the declared member's
    digest-verified bytes, so the forger is never consulted.
    """

    import importlib.machinery

    staged = tmp_path / "flow_model.py"
    staged.write_text(
        'import torch\nORIGIN = "manifest"\n\ndef build():\n    return torch.nn.Linear(4, 2)\n',
        encoding="utf-8",
    )
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import flow_model
import torch

def build_model() -> object:
    assert flow_model.ORIGIN == "manifest", flow_model.ORIGIN
    return flow_model.build()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return ((torch.randn(1, 4, generator=generator, device=device),), {})
""",
        encoding="utf-8",
    )

    class ForgingLoader:
        def create_module(self, spec: object) -> None:
            return None

        def exec_module(self, module: object) -> None:
            module.__file__ = str(staged)  # forged provenance
            module.ORIGIN = "foreign-loader-bytes"  # type: ignore[attr-defined]

    class ForgingFinder:
        consulted = 0

        def find_spec(self, fullname: str, path: object = None, target: object = None) -> object:
            if fullname != "flow_model":
                return None
            ForgingFinder.consulted += 1
            spec = importlib.machinery.ModuleSpec("flow_model", ForgingLoader(), origin=str(staged))
            spec.has_location = True
            return spec

    forger = ForgingFinder()
    sys.meta_path.insert(0, forger)
    try:
        loaded = load_typed_adapter(adapter, manifest_root=tmp_path)
        assert isinstance(loaded.build_model(), torch.nn.Linear)
    finally:
        sys.meta_path.remove(forger)
    assert ForgingFinder.consulted == 0, (
        "the manifest-owned finder must preempt the forger entirely"
    )
    assert "flow_model" not in sys.modules


def test_undeclared_member_of_an_owned_package_is_refused(tmp_path: Path) -> None:
    """An owned name with no declared member behind it is a typed refusal.

    ``vendored/nets.py`` exists ON DISK but the manifest never declared it;
    deferring to ambient machinery would hand an owned name to whatever
    answers first, so the window's finder refuses instead.
    """

    package = tmp_path / "vendored"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "nets.py").write_text(_SIBLING_MODULE, encoding="utf-8")
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
from vendored.nets import build
import torch

def build_model() -> object:
    return build()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return ((torch.randn(1, 4, generator=generator, device=device),), {})
""",
        encoding="utf-8",
    )

    with pytest.raises(
        RecipeError,
        match="manifest-owned module 'vendored.nets' is not a declared member",
    ):
        load_typed_adapter(
            adapter,
            manifest_root=tmp_path,
            manifest_member_paths=[package / "__init__.py"],
        )
    assert "vendored" not in sys.modules
    assert "vendored.nets" not in sys.modules


def test_member_digest_drift_is_refused_at_import(tmp_path: Path) -> None:
    """The window's loader re-hashes each member against its declared digest."""

    adapter, sibling = _write_sibling_closure(tmp_path)
    with pytest.raises(RecipeError, match="drifted from its declared digest"):
        load_recipe(
            {
                "kind": "typed-adapter",
                "path": str(adapter),
                "adapter_sha256": hash_bytes(adapter.read_bytes()),
                "code_manifest": [
                    {"path": str(sibling), "sha256": "sha256:" + "0" * 64},
                    {"path": str(adapter), "sha256": hash_bytes(adapter.read_bytes())},
                ],
            }
        )
    assert "flow_model" not in sys.modules


def test_interleaved_adapter_windows_refuse_and_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second context opening mid-window is refused; global state stays whole.

    Sol wave-8 R2 finding 4, secondary probe: A-enter/B-enter/A-exit/B-exit
    left A's manifest module in global ``sys.modules``. The transaction is now
    serialized: same-thread interleaving refuses typed at B's enter, and the
    displaced foreign module is restored intact when A closes.
    """

    import types

    from menagerie.crawler.recipe import _ManifestImportContext, _manifest_owned_names

    poison = types.ModuleType("flow_model")
    poison.ORIGIN = "preexisting"
    monkeypatch.setitem(sys.modules, "flow_model", poison)

    def _context(name: str) -> _ManifestImportContext:
        root = tmp_path / name
        root.mkdir()
        member = root / "flow_model.py"
        member.write_text(f'ORIGIN = "{name}"\n', encoding="utf-8")
        return _ManifestImportContext(
            root,
            _manifest_owned_names(root, root / "adapter.py", [member]),
            f"adapter_{name}",
            [(member, None)],
        )

    context_a = _context("alpha")
    context_b = _context("beta")
    with context_a:
        with pytest.raises(RecipeError, match="do not interleave"):
            context_b.__enter__()
        # A's window is still coherent after B's refusal.
        import flow_model  # noqa: F401 -- resolved by A's finder

        assert sys.modules["flow_model"].ORIGIN == "alpha"
    assert sys.modules["flow_model"] is poison, "the foreign module must be restored"
    # The lock is released: both contexts reopen cleanly afterwards.
    with context_b:
        pass
    with context_a:
        pass


def test_adapter_windows_serialize_across_threads(tmp_path: Path) -> None:
    """A second thread's window blocks until the open window closes."""

    import threading
    import time

    from menagerie.crawler.recipe import _ManifestImportContext

    def _context(name: str) -> _ManifestImportContext:
        root = tmp_path / name
        root.mkdir()
        member = root / f"member_{name}.py"
        member.write_text("VALUE = 1\n", encoding="utf-8")
        return _ManifestImportContext(
            root, frozenset({f"member_{name}"}), f"adapter_{name}", [(member, None)]
        )

    context_a = _context("alpha")
    context_b = _context("beta")
    entered_b = threading.Event()

    def open_b() -> None:
        with context_b:
            entered_b.set()

    thread = threading.Thread(target=open_b)
    with context_a:
        thread.start()
        time.sleep(0.2)
        assert not entered_b.is_set(), "B must wait for A's window to close"
    thread.join(timeout=5)
    assert entered_b.is_set()
