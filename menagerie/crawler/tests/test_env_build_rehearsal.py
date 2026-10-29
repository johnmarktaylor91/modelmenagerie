"""EN-4 -- build-only rehearsal receipts: minted only on green, refused when absent.

Six of eleven locked intents were never built by the 200a driver, so their first
production build would have been mid-leg, on paid time. The rehearsal tool runs the
production lifecycle transaction per intent and mints a receipt bound to
(intent, generation, target, machine); ``evaluate_rehearsal_receipts`` is the
preflight-arm predicate that refuses routed intents without one. Chaos arms pinned
here: a crash mid-create (the kill -9 shape) never leaves a receipt and the re-run
converges, and the rehearsal succeeds from a bare parent environment (the launchd
wake shape).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import pytest

from menagerie.crawler.constants import EnvironmentPhase
from menagerie.crawler.env_lifecycle import (
    EnvironmentLifecycleError,
    IntentProbes,
    ProbeResult,
    SequentialEnvironmentLifecycle,
    canonical_probe_receipt_bytes,
    expected_probe_names,
)
from menagerie.crawler.effort import EffortTracker, StageCap
from menagerie.crawler.envs import (
    EnvironmentIntent,
    LockArtifacts,
    compute_environment_generation,
)
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.tools.env_build_rehearsal import (
    REHEARSAL_RECEIPT_VERSION,
    evaluate_rehearsal_receipts,
    rehearse_intents,
)

_MACHINE = "test-host:arm64"


def _family_bytes(tag: str) -> tuple[bytes, bytes]:
    artifact_sha256 = hash_bytes(f"exact conda artifact {tag}".encode())
    url = f"https://conda.example.test/python-{tag}.conda"
    lock = f"{url}#{artifact_sha256.removeprefix('sha256:')}\n".encode()
    export = (
        b'{"packages":[{"build":"h1_0","name":"python","sha256":"'
        + artifact_sha256.encode()
        + b'","url":"'
        + url.encode()
        + b'","version":"3.11"}]}\n'
    )
    return lock, export


_LOCK_BYTES, _EXPORT_BYTES = _family_bytes("v1")


def _locked_intent(tmp_path: Path, name: str = "demo", tag: str = "v1") -> EnvironmentIntent:
    """Build one locked intent with a committed family and a proven generation."""

    lock_bytes, export_bytes = _family_bytes(tag)
    lock_dir = tmp_path / name / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / "osx-arm64.lock").write_bytes(lock_bytes)
    (lock_dir / "osx-arm64.resolved.json").write_bytes(export_bytes)
    (lock_dir / "osx-arm64.resolved.sha256").write_text(
        f"{hash_bytes(export_bytes)}\n", encoding="utf-8"
    )
    probes = IntentProbes(imports=("a", "b", "c"), export_checks=(), source_build=())
    receipt = canonical_probe_receipt_bytes(
        tuple(ProbeResult(probe, True, "ok") for probe in expected_probe_names(probes))
    )
    (lock_dir / "osx-arm64.probes.json").write_bytes(receipt)
    lock = LockArtifacts(
        target="osx-arm64",
        lock_path=lock_dir / "osx-arm64.lock",
        export_path=lock_dir / "osx-arm64.resolved.json",
        export_hash_path=lock_dir / "osx-arm64.resolved.sha256",
        lock_bytes=lock_bytes,
        export_bytes=export_bytes,
        declared_export_hash=hash_bytes(export_bytes),
        declared_dependencies=("python",),
    )
    generation = compute_environment_generation(name, lock, probes, pip_overlay=None)
    assert generation is not None
    return EnvironmentIntent(
        name=name,
        phase=EnvironmentPhase("pytorch"),
        framework="pytorch",
        description="test",
        split_guidance="test",
        channels=("conda-forge",),
        dependencies=("python",),
        probes=probes,
        lock=lock,
        generation=generation,
    )


class _GreenBackend:
    """Follow the production contract without any real solver or prefix."""

    def __init__(self) -> None:
        self.creates = 0

    def solve(self, environment_file: Path, target: str):  # pragma: no cover - unused
        raise AssertionError("a rehearsal never solves")

    def create(self, lock_file: Path, prefix: Path) -> bytes:
        self.creates += 1
        return _EXPORT_BYTES

    def probe(self, prefix: Path, probes: IntentProbes) -> Sequence[ProbeResult]:
        return tuple(ProbeResult(name, True, "ok") for name in expected_probe_names(probes))

    def remove(self, prefix: Path) -> None:
        return None


class _DiesMidCreateOnceBackend(_GreenBackend):
    """First create dies like a killed process; later creates succeed."""

    def __init__(self) -> None:
        super().__init__()
        self.died = False

    def create(self, lock_file: Path, prefix: Path) -> bytes:
        if not self.died:
            self.died = True
            raise EnvironmentLifecycleError("simulated mid-create death (kill -9 shape)")
        return super().create(lock_file, prefix)


def _factory(tmp_path: Path, backend: Any):
    def build() -> SequentialEnvironmentLifecycle:
        return SequentialEnvironmentLifecycle(
            backend,
            EffortTracker({"environment": StageCap(attempts=2, seconds=60, bytes=10_000)}),
            env_root=tmp_path / "active",
            disk_free=lambda _path: 10_000,
            minimum_free_bytes=0,
        )

    return build


@pytest.mark.smoke
def test_green_rehearsal_mints_a_bound_receipt_and_arms_the_predicate(
    tmp_path: Path,
) -> None:
    intent = _locked_intent(tmp_path)
    receipts_root = tmp_path / "receipts"
    outcome = rehearse_intents(
        [intent],
        lifecycle_factory=_factory(tmp_path, _GreenBackend()),
        receipts_root=receipts_root,
        machine=_MACHINE,
    )
    assert outcome["all_rehearsed"] is True
    receipt_path = Path(outcome["outcomes"][0]["receipt"])
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == REHEARSAL_RECEIPT_VERSION
    assert payload["intent"] == "demo"
    assert payload["generation"] == intent.generation
    assert payload["machine"] == _MACHINE
    assert payload["lifecycle"]["export_sha256"] == hash_bytes(_EXPORT_BYTES)

    verdict = evaluate_rehearsal_receipts(
        [intent], receipts_root=receipts_root, machine=_MACHINE
    )
    assert verdict["armed"] is True
    assert verdict["refusals"] == []


@pytest.mark.smoke
def test_missing_receipt_refuses_the_routed_intent(tmp_path: Path) -> None:
    intent = _locked_intent(tmp_path)
    verdict = evaluate_rehearsal_receipts(
        [intent], receipts_root=tmp_path / "receipts", machine=_MACHINE
    )
    assert verdict["armed"] is False
    assert verdict["refusals"][0]["reason"] == "missing-or-invalid-receipt"


@pytest.mark.smoke
def test_receipt_for_another_generation_or_machine_is_stale(tmp_path: Path) -> None:
    """A rotation or a different host must re-rehearse; old receipts never carry."""

    intent = _locked_intent(tmp_path)
    receipts_root = tmp_path / "receipts"
    rehearse_intents(
        [intent],
        lifecycle_factory=_factory(tmp_path, _GreenBackend()),
        receipts_root=receipts_root,
        machine=_MACHINE,
    )
    # Rotate the family: new lock bytes => new generation, same receipt on disk.
    rotated = _locked_intent(tmp_path / "rotated", name="demo", tag="v2")
    assert rotated.generation != intent.generation
    verdict = evaluate_rehearsal_receipts(
        [rotated], receipts_root=receipts_root, machine=_MACHINE
    )
    assert verdict["armed"] is False
    assert verdict["refusals"][0]["reason"] == "stale-receipt"
    # A different machine binding never reads this machine's receipt.
    other = evaluate_rehearsal_receipts(
        [intent], receipts_root=receipts_root, machine="other-host:arm64"
    )
    assert other["armed"] is False
    assert other["refusals"][0]["reason"] == "missing-or-invalid-receipt"


@pytest.mark.smoke
def test_tampered_receipt_bytes_are_refused(tmp_path: Path) -> None:
    intent = _locked_intent(tmp_path)
    receipts_root = tmp_path / "receipts"
    outcome = rehearse_intents(
        [intent],
        lifecycle_factory=_factory(tmp_path, _GreenBackend()),
        receipts_root=receipts_root,
        machine=_MACHINE,
    )
    receipt_path = Path(outcome["outcomes"][0]["receipt"])
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["lifecycle"]["export_sha256"] = "sha256:" + "0" * 64
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    verdict = evaluate_rehearsal_receipts(
        [intent], receipts_root=receipts_root, machine=_MACHINE
    )
    assert verdict["armed"] is False
    assert verdict["refusals"][0]["reason"] == "missing-or-invalid-receipt"


@pytest.mark.smoke
def test_unlocked_intent_is_refused_never_skipped(tmp_path: Path) -> None:
    intent = _locked_intent(tmp_path)
    lockless = EnvironmentIntent(
        name="lockless",
        phase=intent.phase,
        framework="pytorch",
        description="test",
        split_guidance="test",
        channels=("conda-forge",),
        dependencies=("python",),
        probes=intent.probes,
        lock=LockArtifacts(
            target="osx-arm64",
            lock_path=tmp_path / "none.lock",
            export_path=tmp_path / "none.json",
            export_hash_path=tmp_path / "none.sha256",
            lock_bytes=None,
            export_bytes=None,
            declared_export_hash=None,
        ),
        generation=None,
    )
    outcome = rehearse_intents(
        [lockless],
        lifecycle_factory=_factory(tmp_path, _GreenBackend()),
        receipts_root=tmp_path / "receipts",
        machine=_MACHINE,
    )
    assert outcome["all_rehearsed"] is False
    assert outcome["outcomes"][0]["status"] == "refused"
    verdict = evaluate_rehearsal_receipts(
        [lockless], receipts_root=tmp_path / "receipts", machine=_MACHINE
    )
    assert verdict["armed"] is False
    assert verdict["refusals"][0]["reason"] == "not-locked"


@pytest.mark.smoke
def test_chaos_kill_mid_create_leaves_no_receipt_and_the_rerun_converges(
    tmp_path: Path,
) -> None:
    """The kill -9 arm: a receipt exists only for a fully green transaction."""

    intent = _locked_intent(tmp_path)
    receipts_root = tmp_path / "receipts"
    backend = _DiesMidCreateOnceBackend()
    factory = _factory(tmp_path, backend)
    first = rehearse_intents(
        [intent], lifecycle_factory=factory, receipts_root=receipts_root, machine=_MACHINE
    )
    assert first["all_rehearsed"] is False
    assert first["outcomes"][0]["status"] == "failed"
    assert list(receipts_root.glob("*.json")) == []
    refused = evaluate_rehearsal_receipts(
        [intent], receipts_root=receipts_root, machine=_MACHINE
    )
    assert refused["armed"] is False

    second = rehearse_intents(
        [intent], lifecycle_factory=factory, receipts_root=receipts_root, machine=_MACHINE
    )
    assert second["all_rehearsed"] is True
    healed = evaluate_rehearsal_receipts(
        [intent], receipts_root=receipts_root, machine=_MACHINE
    )
    assert healed["armed"] is True


@pytest.mark.smoke
def test_chaos_bare_environment_rehearsal_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launchd-wake arm: no inherited environment is load-bearing."""

    intent = _locked_intent(tmp_path)
    for key in list(os.environ):
        if key not in ("PATH", "HOME"):
            monkeypatch.delenv(key, raising=False)
    outcome = rehearse_intents(
        [intent],
        lifecycle_factory=_factory(tmp_path, _GreenBackend()),
        receipts_root=tmp_path / "receipts",
        machine=_MACHINE,
    )
    assert outcome["all_rehearsed"] is True
