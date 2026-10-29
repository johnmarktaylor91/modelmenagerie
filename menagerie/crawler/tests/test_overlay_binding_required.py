"""EN-8: an overlay manifest without its conda-export binding refuses everywhere.

T3-T15 made the binding line (``# conda-resolved-export-sha256:``) typed grammar,
but every guard tolerated its ABSENCE -- a legacy unbound manifest silently pairs
with ANY conda solve, which is exactly the another-rotation's-residue hole the
binding exists to close. This suite pins the flip to absent-binding-REFUSES at
all three guard sites:

* ``envs.compute_environment_generation`` -- an unbound overlay forms NO generation;
* ``env_lifecycle.SequentialEnvironmentLifecycle._apply_pip_overlay`` -- an
  unbound overlay refuses to install;
* ``env_lifecycle.materialized_environment_generation`` -- an unbound overlay
  proves no materialized identity.

The ONLY tolerated unbound manifests are the two committed legacy families
(graph and speech, opus D4 spot-check), enumerated by EXACT byte digest in
``envs.LEGACY_UNBOUND_OVERLAY_MANIFESTS``. The enumeration is SELF-RETIRING:
this suite also asserts each legacy digest still matches a currently-committed
manifest byte-for-byte, so the EN-8 rotation that republishes bound manifests
makes that assertion fail until the enumeration is deleted in the same change.
A fresh unbound manifest can never match an exact digest, so nothing new can
launder through the carve-out.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from menagerie.crawler.env_lifecycle import (
    EnvironmentOverlayError,
    SequentialEnvironmentLifecycle,
    materialized_environment_generation,
)
from menagerie.crawler.envs import (
    DEFAULT_ENVS_ROOT,
    LEGACY_UNBOUND_OVERLAY_MANIFESTS,
    compute_environment_generation,
    load_environment_registry,
    pip_overlay_export_binding,
    render_pip_overlay_manifest,
)
from menagerie.crawler.identity import hash_bytes

pytestmark = pytest.mark.smoke

_ROWS = ({"name": "demo-pkg", "version": "1.0.0", "sha256": "sha256:" + "ab" * 32},)


def _unbound_manifest() -> bytes:
    """A fresh pinned manifest carrying NO conda-export binding line."""

    bound = render_pip_overlay_manifest(_ROWS, conda_export_sha256="sha256:" + "cd" * 32)
    lines = [
        line
        for line in bound.decode("utf-8").splitlines()
        if not line.startswith("# conda-resolved-export-sha256:")
    ]
    content = ("\n".join(lines) + "\n").encode()
    assert pip_overlay_export_binding(content) is None
    return content


def _overlay(manifest_bytes: bytes, tmp_path: Path) -> SimpleNamespace:
    manifest_path = tmp_path / "osx-arm64.pip-overlay.txt"
    manifest_path.write_bytes(manifest_bytes)
    return SimpleNamespace(
        status="pinned",
        manifest_bytes=manifest_bytes,
        manifest_path=manifest_path,
        export_binding=pip_overlay_export_binding(manifest_bytes),
        rows=lambda: [dict(row) for row in _ROWS],
    )


def _lock(export_bytes: bytes) -> SimpleNamespace:
    return SimpleNamespace(
        status="locked",
        target="osx-arm64",
        lock_bytes=b"https://conda.example/x.conda#" + b"1" * 64 + b"\n",
        export_bytes=export_bytes,
        declared_export_hash=hash_bytes(export_bytes),
    )


_EXPORT = b'{"packages": []}\n'
_PROBES = SimpleNamespace(imports=("a",), export_checks=(), source_build=())


# ---------------------------------------------------------------------------
# The committed state and the self-retiring enumeration
# ---------------------------------------------------------------------------


def test_committed_overlays_are_bound_or_exactly_the_legacy_two() -> None:
    """Every committed overlay is bound, except the two enumerated legacies."""

    registry = load_environment_registry(DEFAULT_ENVS_ROOT, target="osx-arm64")
    unbound: dict[str, str] = {}
    for name, intent in sorted(registry.intents.items()):
        overlay = getattr(intent, "pip_overlay", None)
        if overlay is None or overlay.manifest_bytes is None:
            continue
        if overlay.export_binding is None:
            unbound[name] = hash_bytes(overlay.manifest_bytes)
    assert set(unbound.values()) <= LEGACY_UNBOUND_OVERLAY_MANIFESTS, unbound
    # SELF-RETIRING: each enumerated digest must still name a committed
    # manifest. The EN-8 rotation republishes BOUND manifests for graph and
    # speech; the moment it lands, this assertion fails until the enumeration
    # is deleted in the same reviewed change.
    assert set(unbound.values()) == LEGACY_UNBOUND_OVERLAY_MANIFESTS, (
        "a legacy digest no longer matches any committed unbound manifest; "
        "delete it from LEGACY_UNBOUND_OVERLAY_MANIFESTS in this same change"
    )


# ---------------------------------------------------------------------------
# envs.compute_environment_generation
# ---------------------------------------------------------------------------


def test_fresh_unbound_overlay_forms_no_generation(tmp_path: Path) -> None:
    overlay = _overlay(_unbound_manifest(), tmp_path)
    assert (
        compute_environment_generation("demo", _lock(_EXPORT), _PROBES, pip_overlay=overlay) is None
    )


def test_bound_overlay_forms_a_generation(tmp_path: Path) -> None:
    lock = _lock(_EXPORT)
    bound = render_pip_overlay_manifest(_ROWS, conda_export_sha256=lock.declared_export_hash)
    overlay = _overlay(bound, tmp_path)
    assert compute_environment_generation("demo", lock, _PROBES, pip_overlay=overlay) is not None


def test_legacy_enumerated_manifest_still_forms_a_generation() -> None:
    """The two committed legacy families keep working until their rotation."""

    registry = load_environment_registry(DEFAULT_ENVS_ROOT, target="osx-arm64")
    for name in ("graph", "speech"):
        intent = registry.intents[name]
        assert intent.pip_overlay is not None
        assert intent.pip_overlay.export_binding is None  # unbound today, by record
        generation = compute_environment_generation(
            name, intent.lock, intent.probes, pip_overlay=intent.pip_overlay
        )
        assert generation is not None, name


# ---------------------------------------------------------------------------
# env_lifecycle._apply_pip_overlay
# ---------------------------------------------------------------------------


def _lifecycle_apply(overlay: SimpleNamespace) -> None:
    intent = SimpleNamespace(name="demo", pip_overlay=overlay, lock=_lock(_EXPORT))
    backend = SimpleNamespace(
        overlay=lambda prefix, manifest_path: {
            "manifest_sha256": hash_bytes(overlay.manifest_bytes),
            "packages": [dict(row) for row in _ROWS],
        }
    )
    host = SimpleNamespace(_backend=backend)
    SequentialEnvironmentLifecycle._apply_pip_overlay(host, intent, Path("/nonexistent-prefix"))


def test_lifecycle_refuses_a_fresh_unbound_overlay(tmp_path: Path) -> None:
    with pytest.raises(EnvironmentOverlayError, match="binding|unbound"):
        _lifecycle_apply(_overlay(_unbound_manifest(), tmp_path))


def test_lifecycle_installs_a_bound_overlay(tmp_path: Path) -> None:
    bound = render_pip_overlay_manifest(_ROWS, conda_export_sha256=hash_bytes(_EXPORT))
    _lifecycle_apply(_overlay(bound, tmp_path))


# ---------------------------------------------------------------------------
# env_lifecycle.materialized_environment_generation
# ---------------------------------------------------------------------------


def test_materialized_identity_refuses_a_fresh_unbound_overlay(tmp_path: Path) -> None:
    from menagerie.crawler.env_lifecycle import IntentProbes, ProbeResult
    from menagerie.crawler.tests.rotation_transaction_support import solve_fixture

    fixture = solve_fixture("overlaybind")
    overlay = _overlay(_unbound_manifest(), tmp_path)
    probes = IntentProbes(imports=("a", "b", "c"), export_checks=(), source_build=())
    intent = SimpleNamespace(
        name="demo",
        pip_overlay=overlay,
        lock=SimpleNamespace(target="osx-arm64", status="locked"),
        probes=probes,
    )
    inventory = materialized_environment_generation  # keep the call readable
    with pytest.raises(Exception, match="binding|unbound"):
        inventory(
            intent,
            lock_bytes=fixture.lock_bytes,
            export_bytes=fixture.export_bytes,
            package_bytes=fixture.export_bytes,
            probe_results=tuple(
                ProbeResult(name, True, "ok") for name in ("import:a", "import:b", "import:c")
            ),
            python_version="3.12.0",
            compiler_identity="clang",
            sdk_identity="macos",
        )
