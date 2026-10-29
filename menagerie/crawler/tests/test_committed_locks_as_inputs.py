"""Committed locks are INPUTS: the env lane never overwrites a proven family.

C-ENV-LOCKS (T3-T1 3-lab + T1-04 S2-CONFIRMED + T1-09, riders T1-16/T3-T15).
The defect family these tests pin:

* the launch lifecycle solved fresh (or stale-cache-first) and rewrote the
  committed lock family BEFORE create/probe, overwriting proven artifacts with
  unproven bytes (T3-T1);
* the campaign solve cache keyed on names-only currency silently served
  pre-rotation bytes, reverting a completed rotation on the next leg (T1-04);
* rotation publication was unserialized: a crash between member writes left a
  mixed-identity family whose accidental new identity released parks against
  an environment that was never proven (T1-09);
* the materialized environment generation omitted the pip-overlay manifest, so
  an overlay-only rotation left runs reading "current" (T1-16);
* the overlay manifest carried no binding to the conda export it was compiled
  against, so a pip closure could install into a different conda solve
  (T3-T15).

The fix inverts the data flow: the committed, provenance-bound lock family is
the authoritative BUILD INPUT; solving is a rotation-only operation; rotation
invalidates/republishes the campaign solve cache and journals its publication;
the launch preflight asserts store-mount, CAS completeness, and refuses
lockless families.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest


from menagerie.crawler.constants import EnvironmentPhase
from menagerie.crawler.env_lifecycle import (
    ArtifactReceipt,
    CommittedLockFamilyError,
    EnvironmentOverlayError,
    EnvironmentProbeError,
    EnvironmentSolveError,
    IntentProbes,
    ProbeResult,
    SequentialEnvironmentLifecycle,
    SolveResult,
    canonical_probe_receipt_bytes,
    expected_probe_names,
    materialized_environment_generation,
)
from menagerie.crawler.effort import EffortTracker, StageCap
from menagerie.crawler.envs import (
    EnvironmentIntent,
    EnvironmentSpecError,
    LockArtifacts,
    PipOverlay,
    compute_environment_generation,
    load_environment_registry,
    parse_pip_overlay_manifest,
    pip_overlay_export_binding,
    publication_journal_path,
    render_pip_overlay_manifest,
)
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.operator_environment import EnvironmentOperator
from menagerie.crawler.tools.launch_preflight import (
    STORE_MOUNT_UNAVAILABLE,
    evaluate_cas_completeness,
    evaluate_family_readiness,
    evaluate_store_mount,
    main as preflight_main,
)
from menagerie.crawler.tools.rotate_setup_artifacts import (
    publish_rotation_family,
    republish_campaign_solve_cache,
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


def _committed_family(
    lock_dir: Path, *, lock: bytes = _LOCK_BYTES, export: bytes = _EXPORT_BYTES
) -> None:
    """Write one complete committed lock family below a lock directory."""

    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / "osx-arm64.lock").write_bytes(lock)
    (lock_dir / "osx-arm64.resolved.json").write_bytes(export)
    (lock_dir / "osx-arm64.resolved.sha256").write_text(f"{hash_bytes(export)}\n", encoding="utf-8")


def _intent(
    tmp_path: Path,
    *,
    committed: bool = True,
    dependencies: tuple[str, ...] = ("python",),
    pip_overlay: PipOverlay | None = None,
) -> EnvironmentIntent:
    """Build one directly constructed intent whose family lives under tmp."""

    lock_dir = tmp_path / "demo" / "locks"
    if committed:
        _committed_family(lock_dir)
        probes = IntentProbes(imports=("a", "b", "c"), export_checks=(), source_build=())
        receipt = canonical_probe_receipt_bytes(
            tuple(ProbeResult(name, True, "ok") for name in expected_probe_names(probes))
        )
        (lock_dir / "osx-arm64.probes.json").write_bytes(receipt)
    lock = LockArtifacts(
        target="osx-arm64",
        lock_path=lock_dir / "osx-arm64.lock",
        export_path=lock_dir / "osx-arm64.resolved.json",
        export_hash_path=lock_dir / "osx-arm64.resolved.sha256",
        lock_bytes=_LOCK_BYTES if committed else None,
        export_bytes=_EXPORT_BYTES if committed else None,
        declared_export_hash=hash_bytes(_EXPORT_BYTES) if committed else None,
        declared_dependencies=dependencies,
    )
    return EnvironmentIntent(
        name="demo",
        phase=EnvironmentPhase("pytorch"),
        framework="pytorch",
        description="test",
        split_guidance="test",
        channels=("conda-forge",),
        dependencies=dependencies,
        probes=IntentProbes(imports=("a", "b", "c"), export_checks=(), source_build=()),
        lock=lock,
        generation=None,
        pip_overlay=pip_overlay,
    )


class _RecordingBackend:
    """Record every lifecycle action; refuse none; never a real solver."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.create_lock_paths: list[Path] = []

    def solve(self, environment_file: Path, target: str) -> SolveResult:
        self.events.append(f"solve:{target}")
        return SolveResult(
            _LOCK_BYTES,
            _EXPORT_BYTES,
            1.0,
            40,
            (ArtifactReceipt(_ARTIFACT_URL, _ARTIFACT_SHA256),),
        )

    def create(self, lock_file: Path, prefix: Path) -> bytes:
        self.events.append("create")
        self.create_lock_paths.append(lock_file)
        return _EXPORT_BYTES

    def probe(self, prefix: Path, probes: IntentProbes) -> Sequence[ProbeResult]:
        self.events.append("probe")
        return tuple(ProbeResult(name, True, "ok") for name in expected_probe_names(probes))

    def remove(self, prefix: Path) -> None:
        self.events.append("remove")


def _lifecycle(tmp_path: Path, backend: Any) -> SequentialEnvironmentLifecycle:
    return SequentialEnvironmentLifecycle(
        backend,
        EffortTracker({"environment": StageCap(attempts=2, seconds=50, bytes=1000)}),
        env_root=tmp_path / "active",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )


# --- (1) the committed family is the build input -------------------------------


def test_launch_builds_from_the_committed_family_and_never_solves(tmp_path: Path) -> None:
    """A locked intent creates directly from the committed lock; no solve runs."""

    intent = _intent(tmp_path)
    events: list[str] = []
    backend = _RecordingBackend(events)
    result = _lifecycle(tmp_path, backend).run(intent, use=lambda _p, _r: events.append("use"))
    assert events == ["create", "probe", "use", "remove"]
    assert backend.create_lock_paths == [intent.lock.lock_path]
    assert result.export_sha256 == hash_bytes(_EXPORT_BYTES)


def test_launch_leaves_the_committed_family_byte_identical(tmp_path: Path) -> None:
    """The proven committed lock family is never rewritten by a launch."""

    intent = _intent(tmp_path)
    lock_dir = intent.lock.lock_path.parent
    before = {
        path.name: path.read_bytes()
        for path in (
            intent.lock.lock_path,
            intent.lock.export_path,
            intent.lock.export_hash_path,
            lock_dir / "osx-arm64.probes.json",
        )
    }
    _lifecycle(tmp_path, _RecordingBackend([])).run(intent, use=lambda _p, _r: None)
    after = {
        path.name: path.read_bytes()
        for path in (
            intent.lock.lock_path,
            intent.lock.export_path,
            intent.lock.export_hash_path,
            lock_dir / "osx-arm64.probes.json",
        )
    }
    assert after == before


def test_an_already_canonical_receipt_is_never_rewritten_by_a_launch(
    tmp_path: Path,
) -> None:
    """A launch whose observed probes match the committed receipt writes NOTHING.

    X4: the committed receipt is bound by the rotation provenance, so an
    unnecessary re-attestation de-binds it and the next launch preflight refuses
    ``unstable-probe-receipt``/``mixed-family``. Observed for real: three
    consecutive rehearsal builds each dirtied the committed checkout even though
    every probe passed. With a host-independent canonical detail the observed
    bytes equal the committed ones, and the lane must leave the family entirely
    alone -- including the member's mtime, which is what a checkout diff sees.
    """

    intent = _intent(tmp_path)
    receipt_path = intent.lock.lock_path.with_name("osx-arm64.probes.json")

    established = receipt_path.read_bytes()
    stat_before = receipt_path.stat()

    # The launch observes the SAME passing probes.
    _lifecycle(tmp_path, _RecordingBackend([])).run(intent, use=lambda _p, _r: None)

    assert receipt_path.read_bytes() == established
    assert receipt_path.stat().st_mtime_ns == stat_before.st_mtime_ns
    assert receipt_path.stat().st_ino == stat_before.st_ino


def test_committed_family_survives_a_probe_failure_untouched(tmp_path: Path) -> None:
    """A failing build can never leave unproven bytes in the committed family."""

    class ProbeFailingBackend(_RecordingBackend):
        def probe(self, prefix: Path, probes: IntentProbes) -> Sequence[ProbeResult]:
            self.events.append("probe")
            return tuple(ProbeResult(name, False, "boom") for name in expected_probe_names(probes))

    intent = _intent(tmp_path)
    receipt_path = intent.lock.lock_path.with_name("osx-arm64.probes.json")
    receipt_before = receipt_path.read_bytes()
    before = {
        path.name: path.read_bytes()
        for path in (
            intent.lock.lock_path,
            intent.lock.export_path,
            intent.lock.export_hash_path,
        )
    }
    with pytest.raises(Exception):
        _lifecycle(tmp_path, ProbeFailingBackend([])).run(intent, use=lambda _p, _r: None)
    after = {
        path.name: path.read_bytes()
        for path in (
            intent.lock.lock_path,
            intent.lock.export_path,
            intent.lock.export_hash_path,
        )
    }
    assert after == before
    assert receipt_path.read_bytes() == receipt_before


@pytest.mark.smoke
def test_runtime_probe_drift_cannot_write_a_sealed_generation(tmp_path: Path) -> None:
    """A runtime observation mismatch refuses without mutating setup authority."""

    class DriftedProbeBackend(_RecordingBackend):
        """Return validly shaped but noncanonical probe observations."""

        def probe(self, prefix: Path, probes: IntentProbes) -> Sequence[ProbeResult]:
            """Return one changed detail for every declared passing probe."""

            self.events.append("probe")
            return tuple(
                ProbeResult(name, True, "host-specific drift")
                for name in expected_probe_names(probes)
            )

    intent = _intent(tmp_path)
    lock_dir = intent.lock.lock_path.parent
    before = {path: path.read_bytes() for path in sorted(lock_dir.iterdir())}

    with pytest.raises(EnvironmentProbeError, match="rotate"):
        _lifecycle(tmp_path, DriftedProbeBackend([])).run(intent, use=lambda _p, _r: None)

    after = {path: path.read_bytes() for path in sorted(lock_dir.iterdir())}
    assert after == before


def test_lockless_family_is_refused_typed_before_any_backend_action(tmp_path: Path) -> None:
    """No committed family means REFUSE: solving is a rotation-only operation."""

    intent = _intent(tmp_path, committed=False)
    events: list[str] = []
    with pytest.raises(CommittedLockFamilyError, match="rotation"):
        _lifecycle(tmp_path, _RecordingBackend(events)).run(intent, use=lambda _p, _r: None)
    assert events == []
    # The refusal stays inside the typed solve arm of the driver transition
    # table (never the build-failed catch-all).
    assert issubclass(CommittedLockFamilyError, EnvironmentSolveError)


def test_superseded_family_is_refused_not_silently_resolved(tmp_path: Path) -> None:
    """A spec that outgrew its committed family demands a rotation, not a solve."""

    intent = _intent(tmp_path, dependencies=("python", "newly-declared-package"))
    assert intent.lock.status == "superseded"
    events: list[str] = []
    with pytest.raises(CommittedLockFamilyError, match="superseded"):
        _lifecycle(tmp_path, _RecordingBackend(events)).run(intent, use=lambda _p, _r: None)
    assert events == []


# --- (2) solve-cache currency includes the committed family (T1-04) ------------


class _CacheFakeRunner:
    """Serve conda-lock/create/probe argv calls without real tools."""

    def __init__(self, package_rows: Sequence[Mapping[str, Any]]) -> None:
        self.package_rows = list(package_rows)
        self.solve_calls = 0

    def __call__(self, command: Sequence[str], environment: Mapping[str, str], timeout: float):
        import yaml

        from menagerie.crawler.operator_environment import CommandObservation

        argv = list(command)
        if "lock" in argv:
            self.solve_calls += 1
            lock_path = Path(argv[argv.index("--lockfile") + 1])
            lock_path.write_text(yaml.safe_dump({"package": self.package_rows}), encoding="utf-8")
            return CommandObservation(0, "solved", "")
        return CommandObservation(0, "", "")


def _cache_operator(tmp_path: Path, runner: _CacheFakeRunner) -> EnvironmentOperator:
    return EnvironmentOperator(
        state_root=(tmp_path / "operator-state").resolve(),
        conda_lock_command=("/fake/conda-lock",),
        conda_command=("/fake/conda",),
        mamba_command=("/fake/mamba",),
        runner=runner,
        sleeper=lambda _s: None,
    )


def _cache_environment(tmp_path: Path) -> Path:
    path = tmp_path / "envs" / "demo" / "environment.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "demo",
                "channels": ["conda-forge"],
                "dependencies": ["demo-package=1.2.3"],
            }
        ),
        encoding="utf-8",
    )
    return path.resolve()


def _cache_package_row(tmp_path: Path, content: bytes) -> dict[str, Any]:
    artifact = tmp_path / "demo-package-1.2.3-py311h123_0.conda"
    artifact.write_bytes(content)
    return {
        "name": "demo-package",
        "version": "1.2.3",
        "manager": "conda",
        "platform": "osx-arm64",
        "url": artifact.resolve().as_uri(),
        "hash": {"sha256": hash_bytes(content).removeprefix("sha256:")},
    }


def test_solve_cache_predating_a_rotation_is_never_served(tmp_path: Path) -> None:
    """A cached solve whose lock differs from the committed family is stale.

    This is the S2-confirmed T1-04 gun: the Aug-11 graph rotation published a
    new committed family while the campaign solve cache kept its pre-rotation
    bytes; names-only currency served the stale cache and the next leg
    reverted the rotation. Committed-family identity now supersedes the cache.
    """

    runner = _CacheFakeRunner([_cache_package_row(tmp_path, b"artifact bytes")])
    operator = _cache_operator(tmp_path, runner)
    environment = _cache_environment(tmp_path)
    first = operator.solve(environment, "osx-arm64")
    assert first["cache_hit"] is False

    # A rotation commits a DIFFERENT family next to the specification.
    committed_dir = environment.parent / "locks"
    committed_dir.mkdir(parents=True, exist_ok=True)
    (committed_dir / "osx-arm64.lock").write_bytes(
        b"https://conda.example.test/rotated.conda#" + b"ab" * 32 + b"\n"
    )

    second = operator.solve(environment, "osx-arm64")
    assert second["cache_hit"] is False
    assert runner.solve_calls == 2


def test_solve_cache_matching_the_committed_family_still_serves(tmp_path: Path) -> None:
    """Cache currency is unchanged when the cache IS the committed family."""

    runner = _CacheFakeRunner([_cache_package_row(tmp_path, b"artifact bytes")])
    operator = _cache_operator(tmp_path, runner)
    environment = _cache_environment(tmp_path)
    first = operator.solve(environment, "osx-arm64")
    committed_dir = environment.parent / "locks"
    committed_dir.mkdir(parents=True, exist_ok=True)
    (committed_dir / "osx-arm64.lock").write_bytes(Path(first["lock_path"]).read_bytes())
    second = operator.solve(environment, "osx-arm64")
    assert second["cache_hit"] is True
    assert runner.solve_calls == 1


# --- (3) rotation publication is serialized (T1-09) -----------------------------


def _family_payload() -> dict[str, bytes]:
    receipt = (
        b'{"probes":[{"detail":"ok","name":"import:a","passed":true},'
        b'{"detail":"ok","name":"import:b","passed":true},'
        b'{"detail":"ok","name":"import:c","passed":true}]}\n'
    )
    return {
        "osx-arm64.lock": _LOCK_BYTES,
        "osx-arm64.resolved.json": _EXPORT_BYTES,
        "osx-arm64.resolved.sha256": f"{hash_bytes(_EXPORT_BYTES)}\n".encode(),
        "osx-arm64.probes.json": receipt,
        "osx-arm64.provenance.json": b'{"schema_version":"x"}\n',
    }


def test_publication_writes_a_journal_first_and_removes_it_last(tmp_path: Path) -> None:
    """The publication transaction is journaled: torn windows are detectable."""

    lock_dir = tmp_path / "demo" / "locks"
    lock_dir.mkdir(parents=True)
    journal = publication_journal_path(lock_dir, "osx-arm64")
    observed: list[tuple[str, bool]] = []

    payload = _family_payload()
    publish_rotation_family(
        lock_dir,
        "osx-arm64",
        payload,
        observer=lambda name: observed.append((name, journal.is_file())),
    )
    # Every member write happened under an existing journal...
    assert observed and all(present for _name, present in observed)
    # ...and the journal is gone once the family is fully published.
    assert not journal.is_file()
    for name, content in payload.items():
        assert (lock_dir / name).read_bytes() == content


def test_a_torn_publication_reads_as_publishing_and_is_refused(tmp_path: Path) -> None:
    """A legacy torn publication (journal, no manifest) refuses everywhere.

    The pre-manifest publisher (T1-09) wrote members in place under a journal;
    a crash between member writes left the journal over a mixed flat family.
    The manifest-era publisher never produces this state (its commit point is
    the atomic manifest switch, pinned in test_rotation_transaction.py), but a
    tree last written by the OLD code -- or one whose manifest was lost while
    the journal survived -- must keep failing closed exactly as before.
    """

    envs_root = _mini_envs_root(tmp_path, lockless_extra=False)
    lock_dir = envs_root / "demo" / "locks"
    journal = publication_journal_path(lock_dir, "osx-arm64")

    # The legacy torn state: one member already carries next-family bytes,
    # the journal is live, and no family manifest exists.
    (lock_dir / "osx-arm64.lock").write_bytes(_LOCK_BYTES + b"# next family\n")
    journal.write_bytes(b'{"schema_version":"legacy","members":{}}\n')

    # The registry loader reads the torn family as mid-publication, not locked.
    intent = load_environment_registry(envs_root, target="osx-arm64").intents["demo"]
    assert intent.lock.publication_pending
    assert intent.lock.status == "publishing"
    assert intent.generation is None

    # And the env lane refuses to build from it.
    events: list[str] = []
    with pytest.raises(CommittedLockFamilyError, match="publishing"):
        _lifecycle(tmp_path, _RecordingBackend(events)).run(intent, use=lambda _p, _r: None)
    assert events == []


# --- (4) rotation invalidates/republishes the campaign solve cache -------------


def _rotation_state_root(tmp_path: Path) -> Path:
    """Build a rotation scratch root holding the fresh solve and its CAS."""

    root = tmp_path / "rotation-state"
    solve_root = root / "solves" / "demo" / "osx-arm64"
    solve_root.mkdir(parents=True)
    (solve_root / "explicit.lock").write_bytes(_LOCK_BYTES)
    (solve_root / "resolved.json").write_bytes(_EXPORT_BYTES)
    (solve_root / "conda-lock.yml").write_bytes(b"package: []\n")
    cas = root / "artifact-cas" / "sha256"
    cas.mkdir(parents=True)
    (cas / _ARTIFACT_SHA256.removeprefix("sha256:")).write_bytes(_ARTIFACT_BYTES)
    return root


def test_rotation_republishes_the_campaign_solve_cache(tmp_path: Path) -> None:
    """After rotation the campaign cache serves exactly the committed bytes."""

    rotation_root = _rotation_state_root(tmp_path)
    campaign_root = tmp_path / "campaign-state"
    stale_solve = campaign_root / "solves" / "demo" / "osx-arm64"
    stale_solve.mkdir(parents=True)
    (stale_solve / "explicit.lock").write_bytes(b"stale pre-rotation lock\n")
    (stale_solve / "resolved.json").write_bytes(b'{"packages":[]}\n')

    report = republish_campaign_solve_cache(
        campaign_root,
        rotation_state_root=rotation_root,
        intent="demo",
        target="osx-arm64",
        lock_bytes=_LOCK_BYTES,
        export_bytes=_EXPORT_BYTES,
    )
    assert (stale_solve / "explicit.lock").read_bytes() == _LOCK_BYTES
    assert (stale_solve / "resolved.json").read_bytes() == _EXPORT_BYTES
    cas_object = (
        campaign_root / "artifact-cas" / "sha256" / _ARTIFACT_SHA256.removeprefix("sha256:")
    )
    assert cas_object.read_bytes() == _ARTIFACT_BYTES
    assert report["republished_artifacts"] >= 1


def test_rotation_republication_refuses_a_corrupt_rotation_artifact(tmp_path: Path) -> None:
    """A rotation CAS object failing its digest never lands in the campaign CAS."""

    rotation_root = _rotation_state_root(tmp_path)
    corrupt = rotation_root / "artifact-cas" / "sha256" / _ARTIFACT_SHA256.removeprefix("sha256:")
    corrupt.write_bytes(b"corrupted bytes")
    campaign_root = tmp_path / "campaign-state"
    with pytest.raises(Exception):
        republish_campaign_solve_cache(
            campaign_root,
            rotation_state_root=rotation_root,
            intent="demo",
            target="osx-arm64",
            lock_bytes=_LOCK_BYTES,
            export_bytes=_EXPORT_BYTES,
        )
    stale_solve = campaign_root / "solves" / "demo" / "osx-arm64"
    assert not (stale_solve / "explicit.lock").is_file()


# --- (5) rider T1-16: the materialized generation includes the overlay ---------


_OVERLAY_WHEEL_SHA = "ab" * 32
# EN-8: absent conda-export bindings now refuse everywhere, so the fixture
# manifests carry the current-law binding for the export they pair with.
_MANIFEST = (
    "# Pinned pip overlay compiled by tools/rotate_setup_artifacts.py.\n"
    f"# conda-resolved-export-sha256: {hash_bytes(_EXPORT_BYTES)}\n"
    f"demo-wheel==1.0 \\\n    --hash=sha256:{_OVERLAY_WHEEL_SHA}\n"
).encode()


def _pinned_overlay(tmp_path: Path, manifest: bytes) -> PipOverlay:
    intent_root = tmp_path / "demo"
    intent_root.mkdir(parents=True, exist_ok=True)
    declared = intent_root / "pip-overlay.in"
    declared.write_text("demo-wheel\n", encoding="utf-8")
    manifest_path = intent_root / "locks" / "osx-arm64.pip-overlay.txt"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest)
    return PipOverlay(
        declared_path=declared,
        declared=("demo-wheel",),
        manifest_path=manifest_path,
        manifest_bytes=manifest,
    )


def _materialized_generation(intent: EnvironmentIntent) -> str:
    probes = tuple(ProbeResult(name, True, "ok") for name in expected_probe_names(intent.probes))
    return materialized_environment_generation(
        intent,
        lock_bytes=_LOCK_BYTES,
        export_bytes=_EXPORT_BYTES,
        package_bytes=_EXPORT_BYTES,
        python_version="3.11",
        compiler_identity="clang",
        sdk_identity="sdk",
        probe_results=probes,
    )


def test_materialized_generation_tracks_the_overlay_manifest(tmp_path: Path) -> None:
    """An overlay-only rotation changes the MATERIALIZED generation (T1-16)."""

    first = _materialized_generation(
        _intent(tmp_path, pip_overlay=_pinned_overlay(tmp_path, _MANIFEST))
    )
    rotated_manifest = _MANIFEST.replace(b"==1.0", b"==1.1")
    second = _materialized_generation(
        _intent(tmp_path, pip_overlay=_pinned_overlay(tmp_path, rotated_manifest))
    )
    assert first != second


def test_materialized_generation_without_overlay_is_unchanged(tmp_path: Path) -> None:
    """Intents without a declared overlay keep their historical generation."""

    plain = _intent(tmp_path)
    assert _materialized_generation(plain) == _materialized_generation(plain)
    overlaid = _intent(tmp_path, pip_overlay=_pinned_overlay(tmp_path, _MANIFEST))
    assert _materialized_generation(plain) != _materialized_generation(overlaid)


# --- (6) rider T3-T15: the overlay manifest binds its conda export -------------


def test_rendered_manifest_carries_and_roundtrips_the_export_binding() -> None:
    """Rotation-rendered manifests bind the conda export they compiled against."""

    rows = ({"name": "demo-wheel", "version": "1.0", "sha256": f"sha256:{_OVERLAY_WHEEL_SHA}"},)
    export_digest = hash_bytes(_EXPORT_BYTES)
    content = render_pip_overlay_manifest(rows, conda_export_sha256=export_digest)
    assert pip_overlay_export_binding(content) == export_digest
    assert tuple(dict(row) for row in parse_pip_overlay_manifest(content)) == tuple(
        dict(row) for row in rows
    )


def test_malformed_or_duplicate_export_binding_is_refused() -> None:
    """The binding line is typed grammar, never a comment that can rot."""

    with pytest.raises(EnvironmentSpecError):
        parse_pip_overlay_manifest(b"# conda-resolved-export-sha256: not-a-digest\n" + _MANIFEST)
    digest = hash_bytes(_EXPORT_BYTES)
    line = f"# conda-resolved-export-sha256: {digest}\n".encode()
    with pytest.raises(EnvironmentSpecError):
        parse_pip_overlay_manifest(line + line + _MANIFEST)


def test_overlay_bound_to_a_different_export_refuses_to_apply(tmp_path: Path) -> None:
    """A pip closure compiled against another conda solve never installs."""

    foreign_binding = render_pip_overlay_manifest(
        ({"name": "demo-wheel", "version": "1.0", "sha256": f"sha256:{_OVERLAY_WHEEL_SHA}"},),
        conda_export_sha256=hash_bytes(b"a different conda export"),
    )
    overlay = _pinned_overlay(tmp_path, foreign_binding)
    intent = _intent(tmp_path, pip_overlay=overlay)

    class OverlayBackend(_RecordingBackend):
        def overlay(self, prefix: Path, manifest_path: Path) -> Mapping[str, Any]:
            self.events.append("overlay")
            return {
                "manifest_sha256": hash_bytes(foreign_binding),
                "packages": [dict(row) for row in overlay.rows()],
            }

    events: list[str] = []
    with pytest.raises(EnvironmentOverlayError, match="different conda"):
        _lifecycle(tmp_path, OverlayBackend(events)).run(intent, use=lambda _p, _r: None)
    assert "overlay" not in events

    # The committed-side generation refuses the drifted pairing too.
    assert (
        compute_environment_generation(intent.name, intent.lock, intent.probes, pip_overlay=overlay)
        is None
    )


def test_matching_export_binding_applies_and_generates(tmp_path: Path) -> None:
    """A manifest bound to THIS committed export runs and hashes normally."""

    bound = render_pip_overlay_manifest(
        ({"name": "demo-wheel", "version": "1.0", "sha256": f"sha256:{_OVERLAY_WHEEL_SHA}"},),
        conda_export_sha256=hash_bytes(_EXPORT_BYTES),
    )
    overlay = _pinned_overlay(tmp_path, bound)
    intent = _intent(tmp_path, pip_overlay=overlay)

    class OverlayBackend(_RecordingBackend):
        def overlay(self, prefix: Path, manifest_path: Path) -> Mapping[str, Any]:
            self.events.append("overlay")
            return {
                "manifest_sha256": hash_bytes(bound),
                "packages": [dict(row) for row in overlay.rows()],
            }

    events: list[str] = []
    _lifecycle(tmp_path, OverlayBackend(events)).run(intent, use=lambda _p, _r: None)
    assert events == ["create", "overlay", "probe", "remove"]
    assert (
        compute_environment_generation(intent.name, intent.lock, intent.probes, pip_overlay=overlay)
        is not None
    )


# --- (7) launch preflight asserts ----------------------------------------------


def _mini_envs_root(tmp_path: Path, *, lockless_extra: bool = True) -> Path:
    """Write one minimal loadable registry with a committed demo family."""

    root = tmp_path / "mini-envs"
    intents: dict[str, Any] = {
        "demo": {
            "phase": "pytorch",
            "framework": "pytorch",
            "description": "demo",
            "split_guidance": "demo",
        }
    }
    probe_intents: dict[str, Any] = {"demo": {"imports": ["a", "b", "c"], "export_checks": []}}
    if lockless_extra:
        intents["bare"] = {
            "phase": "pytorch",
            "framework": "pytorch",
            "description": "bare",
            "split_guidance": "bare",
        }
        probe_intents["bare"] = {"imports": ["a", "b", "c"], "export_checks": []}
    root.mkdir(parents=True)
    (root / "registry.yml").write_text(
        json.dumps(
            {
                "intents": intents,
                "phase_order": ["pytorch", "native-tail"],
                "small_set_target": True,
                "hard_cap": None,
                "split_guidance": "x",
            }
        ),
        encoding="utf-8",
    )
    (root / "probes.yml").write_text(json.dumps({"intents": probe_intents}), encoding="utf-8")
    for name in intents:
        intent_dir = root / name
        intent_dir.mkdir()
        (intent_dir / "environment.yml").write_text(
            json.dumps({"name": name, "channels": ["conda-forge"], "dependencies": ["python"]}),
            encoding="utf-8",
        )
    _committed_family(root / "demo" / "locks")
    # The probe receipt is the fourth identity member of a COMPLETE family,
    # and the committed provenance binds every member so the preflight family
    # validation reads the fixture as a genuine rotation product.
    payload = _family_payload()
    (root / "demo" / "locks" / "osx-arm64.probes.json").write_bytes(
        payload["osx-arm64.probes.json"]
    )
    (root / "demo" / "locks" / "osx-arm64.provenance.json").write_bytes(
        _demo_provenance_bytes(root, payload)
    )
    return root


def _demo_provenance_bytes(envs_root: Path, payload: dict[str, bytes]) -> bytes:
    """Author one provenance record binding the fixture demo family exactly."""

    from menagerie.crawler.identity import canonical_json_bytes
    from menagerie.crawler.tools.rotate_setup_artifacts import (
        ROTATION_PROVENANCE_SCHEMA_VERSION,
    )

    provenance = {
        "schema_version": ROTATION_PROVENANCE_SCHEMA_VERSION,
        "intent": "demo",
        "target": "osx-arm64",
        "lock_path": "menagerie/crawler/envs/demo/locks/osx-arm64.lock",
        "lock_sha256": hash_bytes(payload["osx-arm64.lock"]),
        "resolved_export_path": "menagerie/crawler/envs/demo/locks/osx-arm64.resolved.json",
        "resolved_export_sha256": hash_bytes(payload["osx-arm64.resolved.json"]),
        "probe_receipt_path": "menagerie/crawler/envs/demo/locks/osx-arm64.probes.json",
        "probe_receipt_sha256": hash_bytes(payload["osx-arm64.probes.json"]),
        "spec_path": "menagerie/crawler/envs/demo/environment.yml",
        "spec_sha256": hash_bytes((envs_root / "demo" / "environment.yml").read_bytes()),
        "probe_contract_path": "menagerie/crawler/envs/probes.yml",
        "probe_contract_sha256": hash_bytes((envs_root / "probes.yml").read_bytes()),
        "solver": {"command": ["fixture"]},
        "host": {"system": "Darwin", "machine": "arm64", "on_target_solve": True},
        "clean_create": {"validated": True, "inventory_matches_resolved_export": True},
        "artifact_hash_validation": {"algorithm": "sha256", "verified": True},
    }
    return canonical_json_bytes(provenance) + b"\n"


def test_preflight_store_mount_assert_refuses_a_dangling_env_root(tmp_path: Path) -> None:
    """An env_root that resolves nowhere refuses before any build could start."""

    dangling = tmp_path / "env-root-link"
    dangling.symlink_to(tmp_path / "vanished-volume")
    report = evaluate_store_mount(dangling)
    assert report["ok"] is False
    assert STORE_MOUNT_UNAVAILABLE in str(report)

    healthy = tmp_path / "envs-store"
    healthy.mkdir()
    assert evaluate_store_mount(healthy)["ok"] is True


def test_preflight_cas_receipt_refuses_missing_or_corrupt_artifacts(tmp_path: Path) -> None:
    """Every committed-family artifact must be present and digest-verified."""

    envs_root = _mini_envs_root(tmp_path, lockless_extra=False)
    state_root = tmp_path / "operator-state"
    report = evaluate_cas_completeness(
        envs_root=envs_root, state_root=state_root, target="osx-arm64"
    )
    assert report["ok"] is False
    assert report["missing"]

    cas = state_root / "artifact-cas" / "sha256"
    cas.mkdir(parents=True)
    entry = cas / _ARTIFACT_SHA256.removeprefix("sha256:")
    entry.write_bytes(b"wrong bytes")
    report = evaluate_cas_completeness(
        envs_root=envs_root, state_root=state_root, target="osx-arm64"
    )
    assert report["ok"] is False

    entry.write_bytes(_ARTIFACT_BYTES)
    report = evaluate_cas_completeness(
        envs_root=envs_root, state_root=state_root, target="osx-arm64"
    )
    assert report["ok"] is True
    assert report["verified_artifacts"] == 1


def test_preflight_refuses_a_lockless_family_unless_explicitly_allowed(tmp_path: Path) -> None:
    """A routable intent without a committed family refuses the launch."""

    envs_root = _mini_envs_root(tmp_path)
    report = evaluate_family_readiness(envs_root=envs_root, target="osx-arm64", allow_lockless=())
    assert report["ok"] is False
    assert any(row["intent"] == "bare" for row in report["refused"])

    allowed = evaluate_family_readiness(
        envs_root=envs_root, target="osx-arm64", allow_lockless=("bare",)
    )
    assert allowed["ok"] is True


def test_preflight_main_wires_the_new_asserts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI refuses on each new assert and passes when all are satisfied."""

    envs_root = _mini_envs_root(tmp_path)
    records_root = tmp_path / "records"
    (records_root / "models").mkdir(parents=True)
    env_root = tmp_path / "env-store"
    env_root.mkdir()
    state_root = tmp_path / "operator-state"
    cas = state_root / "artifact-cas" / "sha256"
    cas.mkdir(parents=True)
    (cas / _ARTIFACT_SHA256.removeprefix("sha256:")).write_bytes(_ARTIFACT_BYTES)
    # The CAS receipt only means something for the root the operator resolves,
    # which is this variable (X4's state-root-unbound assert). A real launch
    # exports it; so does this fixture.
    monkeypatch.setenv("MENAGERIE_ENVIRONMENT_STATE_ROOT", str(state_root))

    base = [
        "--records-root",
        str(records_root),
        "--envs-root",
        str(envs_root),
        "--env-root",
        str(env_root),
        "--state-root",
        str(state_root),
        # The real free space of the host running the suite is not this test's
        # subject; a zero floor probes nothing, exactly as in the lane fixtures.
        "--disk-floor-bytes",
        "0",
        # W2 (Wave-2 batch): this locks fixture renders no gate verdicts, so
        # the checker-calibration XOR arm requires the explicit declaration --
        # the same fixture spelling the launch-preflight suite uses. The
        # lockless refusal below stays the sole differentiator.
        "--renders-no-gate-verdicts",
    ]
    # Lockless "bare" refuses without an explicit allowance.
    assert preflight_main(base) == 2
    passing = [*base, "--allow-lockless", "bare"]
    assert preflight_main(passing) == 0

    # A dangling env root refuses.
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "gone")
    refused = [
        *passing[:4],
        "--env-root",
        str(dangling),
        "--state-root",
        str(state_root),
        "--allow-lockless",
        "bare",
    ]
    assert preflight_main(refused) == 2

    # A missing CAS artifact refuses.
    empty_state = tmp_path / "empty-state"
    refused_cas = [
        *passing[:4],
        "--env-root",
        str(env_root),
        "--state-root",
        str(empty_state),
        "--allow-lockless",
        "bare",
    ]
    assert preflight_main(refused_cas) == 2


def test_publication_journal_is_invisible_to_the_family_glob(tmp_path: Path) -> None:
    """The journal never reads as a hand-authored committed output."""

    lock_dir = tmp_path / "demo" / "locks"
    lock_dir.mkdir(parents=True)
    journal = publication_journal_path(lock_dir, "osx-arm64")
    assert journal.name.startswith(".")
    assert not journal.name.endswith((".lock", ".resolved.json", ".resolved.sha256"))
