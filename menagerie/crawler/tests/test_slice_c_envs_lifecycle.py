"""Slice C environment registry and sequential lifecycle tests."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pytest

from menagerie.crawler.env_lifecycle import (
    ArtifactReceipt,
    DiskRecoveryError,
    EnvironmentCleanupError,
    EnvironmentExactnessError,
    EnvironmentProbeError,
    PerIntentGenerationEffort,
    ProbeResult,
    SequentialEnvironmentLifecycle,
    SolveResult,
    _require_lock_inventory_match,
    canonical_probe_receipt_bytes,
    expected_probe_names,
    parse_exact_lock,
    parse_probe_receipt_bytes,
    parse_resolved_export,
)
from menagerie.crawler.envs import DEFAULT_ENVS_ROOT, IntentProbes, load_environment_registry
from menagerie.crawler.effort import CapFailureRecord, EffortCapExceeded, EffortTracker, StageCap
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.tools.rotate_setup_artifacts import (
    ROTATION_PROVENANCE_SCHEMA_VERSION,
)

_ARTIFACT_BYTES = b"exact conda artifact"
_ARTIFACT_SHA256 = hash_bytes(_ARTIFACT_BYTES)
_ARTIFACT_URL = "https://conda.example.test/python.conda"
_LOCK_BYTES = f"{_ARTIFACT_URL}#{_ARTIFACT_SHA256.removeprefix('sha256:')}\n".encode()
_EXPORT_BYTES = (
    b'{"packages":[{"build":"h1_0","name":"python","sha256":"'
    + _ARTIFACT_SHA256.encode()
    + b'","url":"https://conda.example.test/python.conda","version":"3.11"}]}\n'
)


class FakeEnvironmentBackend:
    """Record lifecycle calls without invoking a real package solver."""

    def __init__(self, events: list[str], *, elapsed_seconds: float = 1.0) -> None:
        """Configure event storage and measured fake solve duration."""

        self.events = events
        self.elapsed_seconds = elapsed_seconds

    def solve(self, environment_file: Path, target: str) -> SolveResult:
        """Return deterministic exact artifact bytes."""

        assert environment_file.is_file()
        self.events.append(f"solve:{target}")
        return SolveResult(
            _LOCK_BYTES,
            _EXPORT_BYTES,
            self.elapsed_seconds,
            40,
            (ArtifactReceipt(_ARTIFACT_URL, _ARTIFACT_SHA256),),
        )

    def create(self, lock_file: Path, prefix: Path) -> bytes:
        """Materialize the prefix for real, so the residue tripwire is armed.

        T5-10: a fake that never creates the prefix makes the post-teardown
        residue check pass VACUOUSLY in every lifecycle test. The fake now
        writes real filesystem state, so a teardown that fails to remove it
        is caught by the same tripwire production relies on.
        """

        assert lock_file.read_bytes() == _LOCK_BYTES
        self.events.append("create")
        prefix.mkdir(parents=True, exist_ok=True)
        (prefix / "conda-meta").mkdir(exist_ok=True)
        (prefix / "conda-meta" / "history").write_bytes(b"fake-create\n")
        return _EXPORT_BYTES

    def probe(self, prefix: Path, probes: IntentProbes) -> Sequence[ProbeResult]:
        """Return successful results for all declared import canaries."""

        self.events.append("probe")
        return tuple(ProbeResult(name, True, "ok") for name in expected_probe_names(probes))

    def remove(self, prefix: Path) -> None:
        """Tear the materialized prefix down for real."""

        self.events.append("remove")
        shutil.rmtree(prefix, ignore_errors=True)


def _copy_env_specs(tmp_path: Path) -> Path:
    """Copy intent inputs so lifecycle outputs never modify the source tree."""

    target = tmp_path / "envs"
    shutil.copytree(DEFAULT_ENVS_ROOT, target)
    return target


def _swap_in_fake_family(root: Path, name: str) -> None:
    """Commit the fake backend's lock family as one intent's build input.

    The committed family is the lane's BUILD INPUT (committed locks are
    inputs, not outputs), so lifecycle tests must commit the exact bytes the
    fake backend creates and inventories from. The spec's declared
    dependencies are narrowed to match, keeping the family ``locked`` rather
    than ``superseded``.
    """

    spec_path = root / name / "environment.yml"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["dependencies"] = ["python"]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    lock_dir = root / name / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / "osx-arm64.lock").write_bytes(_LOCK_BYTES)
    (lock_dir / "osx-arm64.resolved.json").write_bytes(_EXPORT_BYTES)
    (lock_dir / "osx-arm64.resolved.sha256").write_text(
        f"{hash_bytes(_EXPORT_BYTES)}\n", encoding="utf-8"
    )
    # The committed probe receipt is a lock-family member and a BUILD INPUT:
    # runtime compares its observed all-ok fake receipts byte-for-byte against
    # it. Intents without a shipped family (e.g. oddballs) previously lacked
    # it, so their fake-family runs died at the receipt read.
    probes = load_environment_registry(root).intents[name].probes
    (lock_dir / "osx-arm64.probes.json").write_bytes(
        canonical_probe_receipt_bytes(
            tuple(ProbeResult(probe, True, "ok") for probe in expected_probe_names(probes))
        )
    )


#: Per-target on-host binding for rotation provenance. A rotation lock is
#: solved ON the target platform (the honest inverse of the release locks'
#: attested cross-solve), so the recorded host must BE the target.
_ROTATION_TARGET_HOSTS = {
    "osx-arm64": ("Darwin", "arm64"),
    "linux-64": ("Linux", "x86_64"),
}


def _rotation_lock_family_errors(
    env_root: Path,
    lock_path: Path,
    family: dict[str, Path],
    provenance: dict[str, object],
    *,
    lock_bytes: bytes,
    resolved_bytes: bytes,
    declared_resolved_hash: str,
) -> tuple[str, ...]:
    """Return violations in one committed per-intent ROTATION lock family.

    Rotation families are the committed setup artifacts a quarantine/park
    identity hashes (``envs/<intent>/locks/<target>.*``). They are produced by
    ``tools/rotate_setup_artifacts.py`` through the production environment
    operator, on the target host itself, with every probe passing against a
    really created prefix. This validator keeps the anti-fabrication property
    of the release-lock gate for that second committed flavor: a hand-authored
    family cannot satisfy the live spec/probe-contract hash bindings, and a
    family whose spec or probe contract has since changed reads as STALE and
    must be re-rotated.

    Parameters
    ----------
    env_root, lock_path, family, provenance:
        Registry root, the committed lock, its sibling family paths, and the
        parsed provenance object.
    lock_bytes, resolved_bytes, declared_resolved_hash:
        Pre-read family bytes shared with the release-lock validator.

    Returns
    -------
    tuple[str, ...]
        Located anti-fabrication diagnostics, empty for a fully bound family.
    """

    errors: list[str] = []
    target = provenance.get("target")
    intent_name = provenance.get("intent")
    if (
        not isinstance(target, str)
        or lock_path.name != f"{target}.lock"
        or lock_path.parent.name != "locks"
        or lock_path.parent.parent.parent != env_root
    ):
        errors.append(f"{lock_path}:rotation-path-shape-invalid")
        return tuple(errors)
    if not isinstance(intent_name, str) or intent_name != lock_path.parent.parent.name:
        errors.append(f"{lock_path}:rotation-intent-not-bound")
        return tuple(errors)
    if provenance.get("lock_path") != (
        f"menagerie/crawler/envs/{lock_path.relative_to(env_root).as_posix()}"
    ):
        errors.append(f"{lock_path}:lock-path-not-bound")
    if provenance.get("lock_sha256") != hash_bytes(lock_bytes):
        errors.append(f"{lock_path}:lock-hash-mismatch")
    if declared_resolved_hash != hash_bytes(resolved_bytes):
        errors.append(f"{lock_path}:resolved-hash-file-mismatch")
    if provenance.get("resolved_export_sha256") != declared_resolved_hash:
        errors.append(f"{lock_path}:resolved-provenance-mismatch")
    if provenance.get("resolved_export_path") != (
        f"menagerie/crawler/envs/{family['resolved'].relative_to(env_root).as_posix()}"
    ):
        errors.append(f"{lock_path}:resolved-path-not-bound")
    try:
        _require_lock_inventory_match(lock_bytes, resolved_bytes)
    except EnvironmentExactnessError:
        errors.append(f"{lock_path}:lock-inventory-mismatch")
    # The live spec binding is the staleness tripwire the rung-12 census asked
    # for in reverse: editing environment.yml AFTER a rotation invalidates the
    # committed family (spec-hash mismatch) instead of silently shipping
    # artifacts that no longer realize the declared spec.
    spec_path = lock_path.parent.parent / "environment.yml"
    if (
        provenance.get("spec_path")
        != (f"menagerie/crawler/envs/{spec_path.relative_to(env_root).as_posix()}")
        or not spec_path.is_file()
        or provenance.get("spec_sha256") != hash_bytes(spec_path.read_bytes())
    ):
        errors.append(f"{lock_path}:rotation-spec-stale-or-unbound")
    # The probe-contract staleness authority is the receipt re-validation
    # below, which parses the committed receipt against THIS intent's CURRENT
    # declared probes. The provenance's whole-file probes.yml hash is
    # deliberately NOT checked: probes.yml is shared, so a sibling intent's
    # probe edit would false-invalidate every rotated family while this
    # intent's own contract -- the thing the family actually realizes -- is
    # unchanged. A per-intent contract change still fails the re-validation.
    # Pip-overlay binding. A declared overlay makes the pinned manifest a
    # FIFTH identity-bearing committed member, so it carries the same
    # anti-fabrication obligations as the conda family: live spec binding
    # (editing pip-overlay.in after a rotation reads STALE), a bound manifest
    # digest, and the closed pinned+hashed grammar. An intent WITHOUT a
    # declared overlay must carry neither the member nor the provenance block.
    overlay_spec = lock_path.parent.parent / "pip-overlay.in"
    overlay_manifest = lock_path.with_name(f"{target}.pip-overlay.txt")
    overlay_provenance = provenance.get("pip_overlay")
    if overlay_spec.is_file():
        if not overlay_manifest.is_file():
            errors.append(f"{lock_path}:rotation-overlay-manifest-missing")
        elif not isinstance(overlay_provenance, dict):
            errors.append(f"{lock_path}:rotation-overlay-unattested")
        else:
            manifest_bytes = overlay_manifest.read_bytes()
            if overlay_provenance.get("manifest_path") != (
                f"menagerie/crawler/envs/{overlay_manifest.relative_to(env_root).as_posix()}"
            ) or overlay_provenance.get("manifest_sha256") != hash_bytes(manifest_bytes):
                errors.append(f"{lock_path}:rotation-overlay-manifest-not-bound")
            if overlay_provenance.get("spec_path") != (
                f"menagerie/crawler/envs/{overlay_spec.relative_to(env_root).as_posix()}"
            ) or overlay_provenance.get("spec_sha256") != hash_bytes(overlay_spec.read_bytes()):
                errors.append(f"{lock_path}:rotation-overlay-spec-stale-or-unbound")
            try:
                registry = load_environment_registry(env_root, target=target)
                overlay = registry.intents[intent_name].pip_overlay
                if overlay is None or overlay.status != "pinned":
                    errors.append(f"{lock_path}:rotation-overlay-not-pinned")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{lock_path}:rotation-overlay-invalid:{type(exc).__name__}")
    else:
        if overlay_manifest.is_file():
            errors.append(f"{lock_path}:rotation-overlay-manifest-undeclared")
        if overlay_provenance is not None:
            errors.append(f"{lock_path}:rotation-overlay-block-undeclared")
    # Rotation families REQUIRE a real probe receipt: there is no hosted-CI
    # attestation escape hatch for an on-host rotation, and the receipt must
    # re-validate against the CURRENTLY declared probe contract.
    if not family["probes"].is_file():
        errors.append(f"{lock_path}:rotation-probe-receipt-missing")
    else:
        receipt_bytes = family["probes"].read_bytes()
        if provenance.get("probe_receipt_path") != (
            f"menagerie/crawler/envs/{family['probes'].relative_to(env_root).as_posix()}"
        ):
            errors.append(f"{lock_path}:probe-receipt-path-not-bound")
        if provenance.get("probe_receipt_sha256") != hash_bytes(receipt_bytes):
            errors.append(f"{lock_path}:probe-receipt-hash-mismatch")
        try:
            registry = load_environment_registry(env_root, target=target)
            probes = registry.intents[intent_name].probes
            parse_probe_receipt_bytes(probes, receipt_bytes)
        except (KeyError, EnvironmentProbeError, Exception) as exc:  # noqa: BLE001
            errors.append(f"{lock_path}:rotation-probe-receipt-invalid:{type(exc).__name__}")
    solver = provenance.get("solver")
    solver_command = solver.get("command") if isinstance(solver, dict) else None
    if not isinstance(solver_command, list) or solver_command[-2:] != [
        "-m",
        "menagerie.crawler.operator_environment",
    ]:
        errors.append(f"{lock_path}:rotation-solver-command-not-bound")
    host = provenance.get("host")
    expected_host = _ROTATION_TARGET_HOSTS.get(target)
    if (
        expected_host is None
        or not isinstance(host, dict)
        or (host.get("system"), host.get("machine")) != expected_host
        or host.get("on_target_solve") is not True
    ):
        errors.append(f"{lock_path}:rotation-not-solved-on-target")
    clean_create = provenance.get("clean_create")
    if not isinstance(clean_create, dict) or clean_create != {
        "validated": True,
        "inventory_matches_resolved_export": True,
    }:
        errors.append(f"{lock_path}:clean-create-unattested")
    artifact_validation = provenance.get("artifact_hash_validation")
    if (
        not isinstance(artifact_validation, dict)
        or artifact_validation.get("algorithm") != "sha256"
        or artifact_validation.get("verified") is not True
    ):
        errors.append(f"{lock_path}:artifact-hashes-unattested")
    return tuple(errors)


def _release_lock_provenance_errors(env_root: Path) -> tuple[str, ...]:
    """Return violations in committed release-lock provenance.

    Parameters
    ----------
    env_root:
        Environment registry root containing optional committed release locks.

    Returns
    -------
    tuple[str, ...]
        Located anti-fabrication diagnostics, empty only for fully attested lock families.
    """

    generated_outputs = {
        path
        for path in env_root.rglob("*")
        if path.is_file()
        and (
            path.suffix == ".lock"
            or path.name.endswith(".resolved.json")
            or path.name.endswith(".resolved.sha256")
        )
    }
    accounted_outputs: set[Path] = set()
    errors: list[str] = []
    for lock_path in sorted(env_root.rglob("*.lock")):
        family = {
            "resolved": lock_path.with_suffix(".resolved.json"),
            "resolved_hash": lock_path.with_suffix(".resolved.sha256"),
            "provenance": lock_path.with_suffix(".provenance.json"),
            "probes": lock_path.with_suffix(".probes.json"),
        }
        accounted_outputs.add(lock_path)
        accounted_outputs.update(
            path for name, path in family.items() if name in {"resolved", "resolved_hash"}
        )
        missing = [name for name, path in family.items() if not path.is_file()]
        if missing:
            errors.append(f"{lock_path}:missing-{','.join(sorted(missing))}")
            continue
        try:
            lock_bytes = lock_path.read_bytes()
            resolved_bytes = family["resolved"].read_bytes()
            parse_exact_lock(lock_bytes)
            if parse_resolved_export(resolved_bytes) != resolved_bytes:
                errors.append(f"{lock_path}:noncanonical-resolved-export")
            declared_resolved_hash = family["resolved_hash"].read_text(encoding="utf-8").strip()
            provenance = json.loads(family["provenance"].read_bytes())
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{lock_path}:invalid-lock-family:{type(exc).__name__}")
            continue
        if not isinstance(provenance, dict):
            errors.append(f"{lock_path}:provenance-not-object")
            continue
        if provenance.get("schema_version") == ROTATION_PROVENANCE_SCHEMA_VERSION:
            errors.extend(
                _rotation_lock_family_errors(
                    env_root,
                    lock_path,
                    family,
                    provenance,
                    lock_bytes=lock_bytes,
                    resolved_bytes=resolved_bytes,
                    declared_resolved_hash=declared_resolved_hash,
                )
            )
            continue
        target = provenance.get("target")
        spec_path = provenance.get("spec_path")
        probe_contract_path = provenance.get("probe_contract_path")
        if not isinstance(target, str) or not lock_path.name.endswith(f"-{target}.lock"):
            errors.append(f"{lock_path}:target-not-bound-to-lock")
        if provenance.get("schema_version") != "menagerie.crawler.release-lock-provenance.v1":
            errors.append(f"{lock_path}:invalid-provenance-schema")
        if provenance.get("lock_path") != (
            f"menagerie/crawler/envs/{lock_path.relative_to(env_root).as_posix()}"
        ):
            errors.append(f"{lock_path}:lock-path-not-bound")
        if provenance.get("lock_sha256") != hash_bytes(lock_bytes):
            errors.append(f"{lock_path}:lock-hash-mismatch")
        if declared_resolved_hash != hash_bytes(resolved_bytes):
            errors.append(f"{lock_path}:resolved-hash-file-mismatch")
        if provenance.get("resolved_export_sha256") != declared_resolved_hash:
            errors.append(f"{lock_path}:resolved-provenance-mismatch")
        if provenance.get("resolved_export_path") != (
            f"menagerie/crawler/envs/{family['resolved'].relative_to(env_root).as_posix()}"
        ):
            errors.append(f"{lock_path}:resolved-path-not-bound")
        if family["probes"].is_file():
            if provenance.get("probe_receipt_path") != (
                f"menagerie/crawler/envs/{family['probes'].relative_to(env_root).as_posix()}"
            ):
                errors.append(f"{lock_path}:probe-receipt-path-not-bound")
            if provenance.get("probe_receipt_sha256") != hash_bytes(family["probes"].read_bytes()):
                errors.append(f"{lock_path}:probe-receipt-hash-mismatch")
        elif provenance.get("probe_observation") != {
            "committed_on_linux": False,
            "producer": "hosted macOS release CI attestation",
        }:
            errors.append(f"{lock_path}:hosted-probe-observation-not-bound")
        for label, declared_path, digest_key in (
            ("spec", spec_path, "spec_sha256"),
            ("probe-contract", probe_contract_path, "probe_contract_sha256"),
        ):
            expected_parent = Path("menagerie/crawler/envs/specs")
            if not isinstance(declared_path, str) or Path(declared_path).parent != expected_parent:
                errors.append(f"{lock_path}:{label}-path-outside-registry")
                continue
            local_path = env_root / "specs" / Path(declared_path).name
            if not local_path.is_file() or provenance.get(digest_key) != hash_bytes(
                local_path.read_bytes()
            ):
                errors.append(f"{lock_path}:{label}-hash-mismatch")
        solver = provenance.get("solver")
        solver_command = solver.get("command") if isinstance(solver, dict) else None
        if (
            not isinstance(solver_command, list)
            or spec_path not in solver_command
            or (
                solver_command[:3] != ["conda", "env", "create"]
                # Validate the conda-lock invocation signature ("<python> -m conda_lock")
                # without pinning the solver's absolute interpreter path (host-specific).
                and solver_command[1:3] != ["-m", "conda_lock"]
            )
        ):
            errors.append(f"{lock_path}:solver-command-not-bound")
        clean_create = provenance.get("clean_create")
        artifact_validation = provenance.get("artifact_hash_validation")
        if not isinstance(clean_create, dict):
            errors.append(f"{lock_path}:clean-create-unattested")
        elif target == "linux-64" and clean_create.get("validated") is not True:
            errors.append(f"{lock_path}:clean-create-unattested")
        elif target == "osx-arm64" and clean_create.get("validation_host") != (
            "hosted macOS release CI"
        ):
            errors.append(f"{lock_path}:clean-create-unattested")
        if (
            not isinstance(artifact_validation, dict)
            or artifact_validation.get("algorithm") != "sha256"
            or artifact_validation.get("verified") is not True
        ):
            errors.append(f"{lock_path}:artifact-hashes-unattested")
        target_host = {"linux-64": ("Linux", "x86_64", "__linux")}.get(target)
        host = provenance.get("host")
        virtual_packages = provenance.get("virtual_packages")
        if target == "linux-64" and (
            target_host is None
            or not isinstance(host, dict)
            or (host.get("system"), host.get("machine")) != target_host[:2]
            or not isinstance(virtual_packages, list)
            or not any(
                isinstance(row, list) and row and row[0] == target_host[2]
                for row in virtual_packages
            )
        ):
            errors.append(f"{lock_path}:solve-host-not-on-target")
        if target == "osx-arm64" and (
            not isinstance(host, dict)
            or host.get("system") != "Linux"
            or "virtual_package_spec_path" not in provenance
            or not isinstance(virtual_packages, list)
            or not any(
                isinstance(row, list) and row and row[0] == "__osx" for row in virtual_packages
            )
            or provenance.get("virtual_package_spec_sha256")
            != hash_bytes(
                (env_root / "specs" / "round21-release.virtual-packages.yml").read_bytes()
            )
        ):
            errors.append(f"{lock_path}:cross-solve-not-bound")
    for path in sorted(generated_outputs - accounted_outputs):
        errors.append(f"{path}:orphan-hand-authored-output")
    return tuple(errors)


#: Intents whose four setup artifacts are committed by the wave-9 rotation
#: (tools/rotate_setup_artifacts.py). The rung-12 census proved that changing
#: only ``environment.yml`` is NOT a rotation: the quarantine/park identity
#: hashes these committed artifacts, so they must exist -- and change -- for a
#: promised rotation to release environment-rebuild parks.
ROTATED_INTENTS = frozenset(
    {
        "core",
        "graph",
        "audio",
        "tabular-ts",
        "recsys",
        "detect-misc",
        "legacy-torch",
        "mmlab",
        "oddballs",
        "paddle-arm64",
        "speech",
    }
)

#: Intents whose committed setup family includes the FIFTH identity member --
#: the pinned pip-overlay manifest -- and therefore a pinned overlay.
OVERLAY_INTENTS = frozenset({"graph", "speech"})


def test_registry_loads_all_intents_with_rotated_setup_artifacts() -> None:
    """All thirteen intents load; rotated intents carry locked artifacts."""

    registry = load_environment_registry()
    assert len(registry.intents) == 13
    assert registry.small_set_target
    assert registry.hard_cap is None
    assert [phase.value for phase in registry.phase_order] == ["pytorch", "native-tail"]
    for intent in registry.intents.values():
        assert len(intent.probes.imports) >= 3
        assert intent.dependencies
        if intent.name in ROTATED_INTENTS:
            assert intent.lock_status == "locked"
            assert intent.generation is not None
        else:
            assert intent.lock_status == "unlocked"
            assert intent.generation is None
        if intent.name in OVERLAY_INTENTS:
            assert intent.pip_overlay is not None
            assert intent.pip_overlay.status == "pinned"
        else:
            assert intent.pip_overlay is None
        assert (DEFAULT_ENVS_ROOT / intent.name / "environment.yml").is_file()


def test_generation_is_stable_and_changes_with_exact_lock(tmp_path: Path) -> None:
    """Identical target artifacts hash stably and a changed lock is a new generation."""

    root = _copy_env_specs(tmp_path)
    lock_dir = root / "core" / "locks"
    # The shipped tree carries core's rotated setup artifacts; this test
    # overwrites the family with its own synthetic lock generation.
    lock_dir.mkdir(exist_ok=True)
    export = b'{"python":"3.11","torch":"2.4"}\n'
    (lock_dir / "osx-arm64.lock").write_bytes(b"url hash-one\n")
    (lock_dir / "osx-arm64.resolved.json").write_bytes(export)
    (lock_dir / "osx-arm64.resolved.sha256").write_text(f"{hash_bytes(export)}\n", encoding="utf-8")
    first = load_environment_registry(root).intents["core"].generation
    second = load_environment_registry(root).intents["core"].generation
    assert first == second
    (lock_dir / "osx-arm64.lock").write_bytes(b"url hash-two\n")
    assert load_environment_registry(root).intents["core"].generation != first


def test_repository_contains_no_hand_authored_lock_or_hash_outputs(tmp_path: Path) -> None:
    """Only solve-provenance-bound exact lock families may be committed."""

    assert _release_lock_provenance_errors(DEFAULT_ENVS_ROOT) == ()
    assert (DEFAULT_ENVS_ROOT / "locks" / "README.md").is_file()
    copied_root = _copy_env_specs(tmp_path)
    fabricated_lock = copied_root / "locks" / "hand-authored-linux-64.lock"
    fabricated_lock.write_text("@EXPLICIT\nhttps://example.test/fake#sha256=00\n", encoding="utf-8")
    fabricated_lock.with_suffix(".resolved.sha256").write_text(
        "sha256:0000000000000000000000000000000000000000000000000000000000000000\n",
        encoding="utf-8",
    )
    assert any(
        "hand-authored-linux-64.lock:missing-" in error
        for error in _release_lock_provenance_errors(copied_root)
    )


def test_rotation_family_goes_stale_when_the_spec_changes_after_rotation(
    tmp_path: Path,
) -> None:
    """Editing environment.yml AFTER a rotation invalidates the committed family.

    This is the rung-12 failure inverted: the census proved a spec edit is not
    a rotation; this gate proves a rotation does not survive a later spec edit.
    Either direction, artifacts and spec must be re-proven together.
    """

    copied_root = _copy_env_specs(tmp_path)
    assert _release_lock_provenance_errors(copied_root) == ()
    spec = copied_root / "core" / "environment.yml"
    spec.write_bytes(spec.read_bytes() + b"\n")
    errors = _release_lock_provenance_errors(copied_root)
    assert any("rotation-spec-stale-or-unbound" in error for error in errors)

    # A tampered probe receipt is caught by its provenance hash binding.
    receipt = copied_root / "graph" / "locks" / "osx-arm64.probes.json"
    if receipt.is_file():
        receipt.write_bytes(receipt.read_bytes() + b"\n")
        errors = _release_lock_provenance_errors(copied_root)
        assert any(
            "graph/locks/osx-arm64.lock:probe-receipt-hash-mismatch" in error for error in errors
        )


def test_lifecycle_orders_create_probe_use_and_teardown(tmp_path: Path) -> None:
    """One committed environment builds sequentially and checks disk after removal.

    The committed lock family is the BUILD INPUT: no solve runs, and the
    committed members stay byte-identical through the whole lifecycle.
    """

    root = _copy_env_specs(tmp_path)
    _swap_in_fake_family(root, "core")
    intent = load_environment_registry(root).intents["core"]
    events: list[str] = []
    disk_values = iter((1000, 700, 1000))
    lifecycle = SequentialEnvironmentLifecycle(
        FakeEnvironmentBackend(events),
        EffortTracker({"environment": StageCap(attempts=1, seconds=5, bytes=100)}),
        env_root=tmp_path / "active",
        disk_free=lambda _path: next(disk_values),
        minimum_free_bytes=0,
    )
    observed_probes: list[tuple[ProbeResult, ...]] = []

    def use(_prefix: Path, probe_results: tuple[ProbeResult, ...]) -> None:
        """Record the exact successful canary observations passed to the driver."""

        observed_probes.append(probe_results)
        events.append("use")

    result = lifecycle.run(intent, use=use)
    assert events == ["create", "probe", "use", "remove"]
    assert result.disk_recovery_checked
    assert result.disk_after_teardown == result.disk_before
    assert observed_probes == [result.probe_results]
    assert intent.lock.lock_path.read_bytes() == _LOCK_BYTES
    assert intent.lock.export_hash_path.read_text(encoding="utf-8").startswith("sha256:")


class CreateFailingBackend(FakeEnvironmentBackend):
    """Fail every create while ``fail`` is set, then behave like the fake."""

    def __init__(self, events: list[str]) -> None:
        """Start in the failing state."""

        super().__init__(events)
        self.fail = True

    def create(self, lock_file: Path, prefix: Path) -> bytes:
        """Raise a genuine build failure until told to behave."""

        if self.fail:
            self.events.append("create-failed")
            raise EnvironmentExactnessError("synthetic build failure")
        return super().create(lock_file, prefix)


def test_environment_effort_cap_is_independent_per_intent_generation(tmp_path: Path) -> None:
    """Exhausting one intent via genuine failures cannot touch another's budget.

    BL-9a sharpened the cap's contract: only genuine failed turns consume it
    (a successful turn resets), so exhaustion is driven here by real build
    failures -- and one intent's exhaustion still never bleeds into another
    intent-generation's independent budget.
    """

    root = _copy_env_specs(tmp_path)
    _swap_in_fake_family(root, "core")
    # oddballs, not graph: the conda-only FakeEnvironmentBackend cannot serve
    # graph's declared pip overlay (a declared overlay fails closed without
    # backend support, by design). Its fake family makes it a buildable input.
    _swap_in_fake_family(root, "oddballs")
    registry = load_environment_registry(root)
    core = registry.intents["core"]
    oddballs = registry.intents["oddballs"]
    effort = PerIntentGenerationEffort(StageCap(attempts=1, seconds=5, bytes=100))
    backend = CreateFailingBackend([])
    lifecycle = SequentialEnvironmentLifecycle(
        backend,
        effort,
        env_root=tmp_path / "active",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )

    with pytest.raises(EnvironmentExactnessError):
        lifecycle.run(core, use=lambda _prefix, _probes: None)
    assert effort(core).usage("environment").attempts == 1
    with pytest.raises(EffortCapExceeded):
        lifecycle.run(core, use=lambda _prefix, _probes: None)

    # core's exhaustion consumed nothing of oddballs' independent budget: its
    # turn builds normally, and the successful turn leaves the budget intact.
    backend.fail = False
    lifecycle.run(oddballs, use=lambda _prefix, _probes: None)
    assert effort(oddballs).usage("environment").attempts == 0


def test_cap_exceeding_build_is_typed_recorded_and_not_materialized(tmp_path: Path) -> None:
    """A build attempt over cap stops before any create and rewrites nothing."""

    root = _copy_env_specs(tmp_path)
    _swap_in_fake_family(root, "core")
    intent = load_environment_registry(root).intents["core"]
    committed_lock_bytes = intent.lock.lock_path.read_bytes()
    events: list[str] = []
    recorded: list[CapFailureRecord] = []
    tracker = EffortTracker(
        {"environment": StageCap(attempts=1, seconds=100)}, recorder=recorded.append
    )
    lifecycle = SequentialEnvironmentLifecycle(
        CreateFailingBackend(events),
        tracker,
        env_root=tmp_path / "active",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )
    # A genuine failed turn consumes the single-attempt budget (BL-9a: only
    # failures consume; a success would have reset it).
    with pytest.raises(EnvironmentExactnessError):
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    events.clear()
    with pytest.raises(EffortCapExceeded):
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    assert recorded[0].actual_stage == "environment"
    assert recorded[0].metric == "attempts"
    assert events == []
    assert intent.lock.lock_path.read_bytes() == committed_lock_bytes


def test_failed_teardown_releases_lifecycle_for_a_later_retry(tmp_path: Path) -> None:
    """A remove exception cannot leave the sequential lifecycle permanently busy."""

    class RemoveFailingBackend(FakeEnvironmentBackend):
        """Fail only the first teardown after otherwise successful use."""

        def __init__(self, events: list[str]) -> None:
            """Initialize one-shot teardown failure state."""

            super().__init__(events)
            self.remove_calls = 0

        def remove(self, prefix: Path) -> None:
            """Raise on the first removal and succeed on the next."""

            self.remove_calls += 1
            self.events.append("remove")
            if self.remove_calls == 1:
                raise DiskRecoveryError("synthetic teardown failure")
            shutil.rmtree(prefix, ignore_errors=True)

    root = _copy_env_specs(tmp_path)
    _swap_in_fake_family(root, "core")
    intent = load_environment_registry(root).intents["core"]
    events: list[str] = []
    lifecycle = SequentialEnvironmentLifecycle(
        RemoveFailingBackend(events),
        EffortTracker({"environment": StageCap(attempts=2, seconds=10, bytes=200)}),
        env_root=tmp_path / "active",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )

    with pytest.raises(EnvironmentCleanupError, match="synthetic teardown"):
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))

    assert events.count("use") == 2


def test_ambient_disk_consumption_is_not_cleanup_residue(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A free-space delta with the prefix verifiably gone is reported, not raised.

    Rung 11: ledger appends, diagnostics, and unrelated processes move the
    ``statvfs`` measurement during a leg. Reading that ambient delta as an
    incomplete cleanup quarantined a healthy environment generation and fanned
    out over 21 models. The residue authority is the materialized prefix, not
    a df delta.
    """

    root = _copy_env_specs(tmp_path)
    _swap_in_fake_family(root, "core")
    intent = load_environment_registry(root).intents["core"]
    events: list[str] = []
    disk_values = iter((1000, 700, 400))
    lifecycle = SequentialEnvironmentLifecycle(
        FakeEnvironmentBackend(events),
        EffortTracker({"environment": StageCap(attempts=1, seconds=5, bytes=100)}),
        env_root=tmp_path / "active",
        disk_free=lambda _path: next(disk_values),
        minimum_free_bytes=0,
        recovery_tolerance_bytes=100,
    )

    with caplog.at_level("WARNING", logger="menagerie.crawler.env_lifecycle"):
        result = lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))

    assert events == ["create", "probe", "use", "remove"]
    assert result.disk_recovery_checked
    assert result.disk_after_teardown == 400
    assert result.ambient_disk_delta_bytes == 600
    assert any("ambient disk consumption" in record.message for record in caplog.records)


def test_fake_backend_materializes_the_prefix_so_the_residue_tripwire_arms(
    tmp_path: Path,
) -> None:
    """The shared fake writes real state; the residue check is never vacuous.

    T5-10: the previous fake never created the prefix, so the post-teardown
    residue tripwire passed vacuously in every lifecycle test. The fake now
    materializes and removes real filesystem state, exactly like production.
    """

    root = _copy_env_specs(tmp_path)
    _swap_in_fake_family(root, "core")
    intent = load_environment_registry(root).intents["core"]
    events: list[str] = []
    lifecycle = SequentialEnvironmentLifecycle(
        FakeEnvironmentBackend(events),
        EffortTracker({"environment": StageCap(attempts=1, seconds=5, bytes=100)}),
        env_root=tmp_path / "active",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )
    seen: list[bool] = []

    def use(prefix: Path, _probes: object) -> None:
        seen.append(prefix.is_dir())
        events.append("use")

    lifecycle.run(intent, use=use)
    assert seen == [True]
    # The teardown genuinely removed the materialized prefix.
    assert not any((tmp_path / "active").glob("core-*"))


def test_teardown_residue_is_a_disk_recovery_error(tmp_path: Path) -> None:
    """A prefix surviving a claimed removal is real residue and still quarantines."""

    class ResidueLeavingBackend(FakeEnvironmentBackend):
        """Materialize the prefix on create and leave it behind on remove."""

        def __init__(self, events: list[str]) -> None:
            """Initialize one-shot residue state."""

            super().__init__(events)
            self.leave_residue = True

        def create(self, lock_file: Path, prefix: Path) -> bytes:
            """Create the prefix directory exactly like the real backend."""

            prefix.mkdir(parents=True, exist_ok=True)
            (prefix / "leftover.bin").write_bytes(b"residue")
            return super().create(lock_file, prefix)

        def remove(self, prefix: Path) -> None:
            """Claim success while leaving bytes behind on the first cycle."""

            self.events.append("remove")
            if not self.leave_residue:
                shutil.rmtree(prefix, ignore_errors=True)

    root = _copy_env_specs(tmp_path)
    _swap_in_fake_family(root, "core")
    intent = load_environment_registry(root).intents["core"]
    events: list[str] = []
    backend = ResidueLeavingBackend(events)
    lifecycle = SequentialEnvironmentLifecycle(
        backend,
        EffortTracker({"environment": StageCap(attempts=2, seconds=10, bytes=200)}),
        env_root=tmp_path / "active",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )

    with pytest.raises(DiskRecoveryError, match="residue"):
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))

    # The failed teardown released the lifecycle; a clean cycle succeeds.
    for leftover in (tmp_path / "active").iterdir():
        shutil.rmtree(leftover, ignore_errors=True)
    backend.leave_residue = False
    result = lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    assert result.disk_recovery_checked
    assert events.count("use") == 2


def test_surviving_dedicated_state_is_residue_not_ambient(tmp_path: Path) -> None:
    """A falsely successful remove leaving a dedicated root behind stays loud.

    Rung-11 review finding 4: the backend contract owns the prefix AND its
    dedicated scratch/cache/registration, but the teardown check verified only
    the prefix, so real backend-owned residue outside it read as ambient disk
    consumption. The lifecycle now absence-checks every root in the backend's
    create-time owned-path receipt.
    """

    class DedicatedStateBackend(FakeEnvironmentBackend):
        """Enumerate a dedicated cache root and leave it behind on remove."""

        def __init__(self, events: list[str], state_root: Path) -> None:
            """Initialize dedicated-state layout and residue mode."""

            super().__init__(events)
            self.state_root = state_root
            self.leave_cache_residue = True

        def create(self, lock_file: Path, prefix: Path) -> bytes:
            """Materialize the prefix plus its dedicated cache and registration."""

            prefix.mkdir(parents=True, exist_ok=True)
            cache = self.state_root / "package-caches" / prefix.name
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "artifact.conda").write_bytes(b"cached bytes")
            registration = self.state_root / "prefix-registrations" / f"{prefix.name}.json"
            registration.parent.mkdir(parents=True, exist_ok=True)
            registration.write_text("{}", encoding="utf-8")
            return super().create(lock_file, prefix)

        def owned_paths(self, prefix: Path) -> tuple[Path, ...]:
            """Return the complete generation-owned inventory."""

            return (
                prefix,
                self.state_root / "package-caches" / prefix.name,
                self.state_root / "prefix-registrations" / f"{prefix.name}.json",
            )

        def remove(self, prefix: Path) -> None:
            """Report success while leaving the dedicated cache behind once."""

            self.events.append("remove")
            shutil.rmtree(prefix, ignore_errors=True)
            registration = self.state_root / "prefix-registrations" / f"{prefix.name}.json"
            registration.unlink(missing_ok=True)
            if not self.leave_cache_residue:
                shutil.rmtree(self.state_root / "package-caches" / prefix.name, True)

    root = _copy_env_specs(tmp_path)
    _swap_in_fake_family(root, "core")
    intent = load_environment_registry(root).intents["core"]
    events: list[str] = []
    backend = DedicatedStateBackend(events, tmp_path / "operator-state")
    lifecycle = SequentialEnvironmentLifecycle(
        backend,
        EffortTracker({"environment": StageCap(attempts=2, seconds=10, bytes=200)}),
        env_root=tmp_path / "active",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )

    with pytest.raises(DiskRecoveryError, match="package-caches"):
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))

    # A complete removal of every owned root passes the same authority.
    for leftover in (tmp_path / "operator-state" / "package-caches").iterdir():
        shutil.rmtree(leftover, ignore_errors=True)
    backend.leave_cache_residue = False
    result = lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    assert result.disk_recovery_checked
    assert events.count("use") == 2


def test_lifecycle_rejects_empty_probe_contract_and_inventory_mismatch(tmp_path: Path) -> None:
    """Neither zero probe receipts nor a drifted created inventory are admissible."""

    root = _copy_env_specs(tmp_path)
    _swap_in_fake_family(root, "core")
    base_intent = load_environment_registry(root).intents["core"]
    empty_intent = replace(base_intent, probes=IntentProbes((), (), ()))
    lifecycle = SequentialEnvironmentLifecycle(
        FakeEnvironmentBackend([]),
        EffortTracker({"environment": StageCap(attempts=1, seconds=5, bytes=100)}),
        env_root=tmp_path / "empty-probes",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )
    with pytest.raises(EnvironmentProbeError, match="empty"):
        lifecycle.run(empty_intent, use=lambda _prefix, _probes: None)

    class DriftedInventoryBackend(FakeEnvironmentBackend):
        """Create a prefix whose installed inventory differs from the committed export."""

        def create(self, lock_file: Path, prefix: Path) -> bytes:
            """Report inventory bytes the committed resolved export never declared."""

            del lock_file, prefix
            self.events.append("create")
            return _EXPORT_BYTES.replace(b"3.11", b"3.12")

    lifecycle = SequentialEnvironmentLifecycle(
        DriftedInventoryBackend([]),
        EffortTracker({"environment": StageCap(attempts=1, seconds=5, bytes=100)}),
        env_root=tmp_path / "drifted-inventory",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )
    with pytest.raises(EnvironmentExactnessError, match="differs from the declared"):
        lifecycle.run(base_intent, use=lambda _prefix, _probes: None)
