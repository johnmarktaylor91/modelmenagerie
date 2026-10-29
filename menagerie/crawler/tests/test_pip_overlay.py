"""Controlled pip-overlay lane tests: closed grammar, identity, lifecycle order.

The rung-12 censuses converged on this lane's contract: requirements resolved
ONCE to exact pinned versions with sha256 hashes, wheels stored and replayed
offline, installation into the conda prefix as a separate lifecycle step, an
IDENTITY-BEARING pinned manifest (quarantine/park release semantics unchanged),
and teardown/residue accounting over overlay artifacts. These tests pin each
property structurally; the committed audio/graph manifests pin them against the
real registry.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from menagerie.crawler.driver_admission import (
    _deferral_row_fact_still_holds,
    _intent_setup_artifact_identity,
    _intent_setup_artifact_state,
    _routed_environment_packages,
)
from menagerie.crawler.effort import EffortTracker, StageCap
from menagerie.crawler.env_lifecycle import (
    ArtifactReceipt,
    canonical_probe_receipt_bytes,
    EnvironmentOverlayError,
    IntentProbes,
    ProbeResult,
    SequentialEnvironmentLifecycle,
    SolveResult,
    expected_probe_names,
)
from menagerie.crawler.environment_rebuild import (
    append_rebuild_deferral_row,
    build_rebuild_deferral_row,
    rebuild_deferral_path,
)
from menagerie.crawler.environment_coverage import CoverageBasis, find_covering_intents
from menagerie.crawler.envs import (
    EnvironmentIntent,
    EnvironmentSpecError,
    LockArtifacts,
    PipOverlay,
    compute_environment_generation,
    load_environment_registry,
    overlay_package_rows,
    parse_pip_overlay_manifest,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.recipe import resolve_environment_artifact_digest

_SHA = "a" * 64
_SHA_B = "b" * 64
#: ONE shared fixture lock family -- conda artifact, explicit lock line, and
#: resolved export -- used by every section (generation, lifecycle) so the
#: manifest's EN-8 binding provably names THIS family's export at every guard
#: site (the pre-fix module shadowed _EXPORT_BYTES with a second definition,
#: leaving _MANIFEST bound to an export no fixture lock actually carried).
_ARTIFACT_BYTES = b"exact conda artifact"
_ARTIFACT_SHA256 = hash_bytes(_ARTIFACT_BYTES)
_ARTIFACT_URL = "https://conda.example.test/python.conda"
_LOCK_BYTES = f"{_ARTIFACT_URL}#{_ARTIFACT_SHA256.removeprefix('sha256:')}\n".encode()
_EXPORT_BYTES = (
    b'{"packages":[{"build":"h1_0","name":"python","sha256":"'
    + _ARTIFACT_SHA256.encode()
    + b'","url":"https://conda.example.test/python.conda","version":"3.11"}]}\n'
)
_MANIFEST = (
    "# pinned overlay\n"
    # EN-8 (Wave-2 batch): a pinned overlay must carry its conda-export
    # binding -- absent-binding refuses at every guard site now, and the
    # binding must name the committed family's resolved export.
    f"# conda-resolved-export-sha256: {hash_bytes(_EXPORT_BYTES)}\n"
    f"speechbrain==1.0.3 \\\n    --hash=sha256:{_SHA}\n"
    f"hyperpyyaml==1.2.2 \\\n    --hash=sha256:{_SHA_B}\n"
).encode()


def test_manifest_grammar_parses_pinned_hashed_rows() -> None:
    """The closed grammar yields exact name/version/sha256 rows in file order."""

    rows = parse_pip_overlay_manifest(_MANIFEST)
    assert [dict(row) for row in rows] == [
        {"name": "speechbrain", "version": "1.0.3", "sha256": f"sha256:{_SHA}"},
        {"name": "hyperpyyaml", "version": "1.2.2", "sha256": f"sha256:{_SHA_B}"},
    ]


@pytest.mark.parametrize(
    "line",
    (
        "speechbrain",  # unpinned
        "speechbrain>=1.0",  # non-exact specifier
        "speechbrain==1.0.3",  # pinned but unhashed
        f"speechbrain==1.0.3 --hash=md5:{'0' * 32}",  # wrong algorithm
        f"speechbrain[extras]==1.0.3 --hash=sha256:{_SHA}",  # extras
        f"speechbrain==1.0.3; python_version<'3.12' --hash=sha256:{_SHA}",  # marker
        "--index-url https://evil.example/simple",  # option line
        f"speechbrain @ https://evil.example/sb.whl --hash=sha256:{_SHA}",  # URL
        f"-e ./speechbrain --hash=sha256:{_SHA}",  # editable
        f"speechbrain==1.0.3 --hash=sha256:{_SHA} --hash=sha256:{_SHA_B}",  # two hashes
    ),
)
def test_manifest_grammar_refuses_everything_outside_the_closed_form(line: str) -> None:
    """No ambient, unpinned, unhashed, or redirected install can ride in."""

    with pytest.raises(EnvironmentSpecError):
        parse_pip_overlay_manifest(line.encode())


def test_manifest_refuses_duplicate_and_empty_pins() -> None:
    """A repeated canonical name and an empty manifest are both refused."""

    duplicate = (
        f"speechbrain==1.0.3 --hash=sha256:{_SHA}\nSpeechBrain==1.0.4 --hash=sha256:{_SHA_B}\n"
    ).encode()
    with pytest.raises(EnvironmentSpecError, match="more than once"):
        parse_pip_overlay_manifest(duplicate)
    with pytest.raises(EnvironmentSpecError, match="no packages"):
        parse_pip_overlay_manifest(b"# nothing pinned\n")


def _overlay(tmp_path: Path, *, manifest: bytes | None = _MANIFEST) -> PipOverlay:
    """Build one declared overlay whose files live in a tmp registry layout."""

    declared_path = tmp_path / "pip-overlay.in"
    declared_path.write_text("speechbrain\nhyperpyyaml\n", encoding="utf-8")
    manifest_path = tmp_path / "locks" / "osx-arm64.pip-overlay.txt"
    if manifest is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(manifest)
    return PipOverlay(
        declared_path=declared_path,
        declared=("speechbrain", "hyperpyyaml"),
        manifest_path=manifest_path,
        manifest_bytes=manifest,
    )


def test_overlay_status_ladder(tmp_path: Path) -> None:
    """unpinned -> invalid -> superseded -> pinned are all typed, distinct states."""

    assert _overlay(tmp_path, manifest=None).status == "unpinned"
    assert _overlay(tmp_path, manifest=b"garbage\n").status == "invalid"
    partial = f"speechbrain==1.0.3 --hash=sha256:{_SHA}\n".encode()
    superseded = _overlay(tmp_path, manifest=partial)
    assert superseded.status == "superseded"
    assert superseded.missing_declared_requirement() == "hyperpyyaml"
    assert _overlay(tmp_path).status == "pinned"


def test_overlay_rows_flow_only_from_a_pinned_manifest(tmp_path: Path) -> None:
    """Stale or absent manifests contribute nothing to the inventory surface."""

    intent = SimpleNamespace(pip_overlay=_overlay(tmp_path))
    assert [dict(row) for row in overlay_package_rows(intent)] == [  # type: ignore[arg-type]
        {"name": "speechbrain", "version": "1.0.3", "sha256": f"sha256:{_SHA}"},
        {"name": "hyperpyyaml", "version": "1.2.2", "sha256": f"sha256:{_SHA_B}"},
    ]
    assert overlay_package_rows(SimpleNamespace(pip_overlay=None)) == ()  # type: ignore[arg-type]
    assert (
        overlay_package_rows(
            SimpleNamespace(pip_overlay=_overlay(tmp_path, manifest=None))  # type: ignore[arg-type]
        )
        == ()
    )
    assert overlay_package_rows(SimpleNamespace()) == ()  # type: ignore[arg-type]


def test_digest_resolution_derives_the_wheel_hash_from_overlay_rows(
    tmp_path: Path,
) -> None:
    """An overlay-provided distribution yields its pinned wheel digest for R1s."""

    rows = list(overlay_package_rows(SimpleNamespace(pip_overlay=_overlay(tmp_path))))  # type: ignore[arg-type]
    digest = resolve_environment_artifact_digest(rows, distribution="speechbrain", version="1.0.3")
    assert digest == f"sha256:{_SHA}"


# --- identity: the manifest is the FIFTH committed member ---------------------


def _fake_lock_family(tmp_path: Path) -> SimpleNamespace:
    """Write a four-member setup family and return a duck-typed intent."""

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / "osx-arm64.lock").write_bytes(b"url#hash\n")
    (lock_dir / "osx-arm64.resolved.json").write_bytes(b'{"packages": []}\n')
    (lock_dir / "osx-arm64.resolved.sha256").write_bytes(b"sha256:00\n")
    (lock_dir / "osx-arm64.probes.json").write_bytes(b"[]\n")
    return SimpleNamespace(
        name="audio",
        lock=SimpleNamespace(
            lock_path=lock_dir / "osx-arm64.lock",
            export_path=lock_dir / "osx-arm64.resolved.json",
            export_hash_path=lock_dir / "osx-arm64.resolved.sha256",
            target="osx-arm64",
        ),
    )


def test_identity_without_overlay_keeps_the_historical_four_member_vector(
    tmp_path: Path,
) -> None:
    """Leg-8 continuity: non-overlay intents hash the exact same state as before."""

    intent = _fake_lock_family(tmp_path)
    state = _intent_setup_artifact_state(intent)  # type: ignore[arg-type]
    assert sorted(state) == [
        "osx-arm64.lock",
        "osx-arm64.probes.json",
        "osx-arm64.resolved.json",
        "osx-arm64.resolved.sha256",
    ]
    explicit_none = SimpleNamespace(**vars(intent), pip_overlay=None)
    assert _intent_setup_artifact_state(explicit_none) == state  # type: ignore[arg-type]
    assert _intent_setup_artifact_identity(explicit_none) == (  # type: ignore[arg-type]
        _intent_setup_artifact_identity(intent)  # type: ignore[arg-type]
    )


def test_overlay_manifest_joins_and_rotates_the_identity(tmp_path: Path) -> None:
    """Declaring, pinning, and re-pinning the overlay each move the identity."""

    plain = _fake_lock_family(tmp_path)
    plain_identity = _intent_setup_artifact_identity(plain)  # type: ignore[arg-type]
    declared = SimpleNamespace(**vars(plain), pip_overlay=_overlay(tmp_path, manifest=None))
    state = _intent_setup_artifact_state(declared)  # type: ignore[arg-type]
    assert state["osx-arm64.pip-overlay.txt"] == {"state": "absent"}
    declared_identity = _intent_setup_artifact_identity(declared)  # type: ignore[arg-type]
    assert declared_identity != plain_identity

    manifest_path = tmp_path / "locks" / "osx-arm64.pip-overlay.txt"
    manifest_path.write_bytes(_MANIFEST)
    pinned_identity = _intent_setup_artifact_identity(declared)  # type: ignore[arg-type]
    assert pinned_identity != declared_identity
    manifest_path.write_bytes(_MANIFEST + b"# rotated\n")
    assert _intent_setup_artifact_identity(declared) != pinned_identity  # type: ignore[arg-type]


def test_overlay_rotation_releases_a_park_exactly_like_a_conda_rotation(
    tmp_path: Path,
) -> None:
    """The rebuild-park claimant sees an overlay rotation as a released fact."""

    intent = SimpleNamespace(
        **vars(_fake_lock_family(tmp_path)),
        pip_overlay=_overlay(tmp_path),
    )
    identity = _intent_setup_artifact_identity(intent)  # type: ignore[arg-type]
    state = _intent_setup_artifact_state(intent)  # type: ignore[arg-type]
    records = tmp_path / "records"
    (records / "models").mkdir(parents=True)
    row = build_rebuild_deferral_row(
        stable_id="m_overlay",
        work_id="work-m_overlay",
        name="Overlay",
        campaign_id="campaign-overlay",
        run_id="run-1",
        machine_id="machine-1",
        created_at="2026-08-10T00:00:00Z",
        quarantine_event_id="environment-cleanup-overlay1",
        intent="audio",
        target="osx-arm64",
        artifact_identity=identity,
        artifact_state=state,
        env_generation="sha256:" + "22" * 32,
        failure_type="menagerie.crawler.env_lifecycle.DiskRecoveryError",
    )
    append_rebuild_deferral_row(rebuild_deferral_path(records), row)
    driver = SimpleNamespace(
        registry=SimpleNamespace(intents={"audio": intent}),
        paths=SimpleNamespace(
            ledgers=SimpleNamespace(models=records / "models" / "current-shard.jsonl")
        ),
    )
    item = SimpleNamespace(route=SimpleNamespace(intent="audio"))
    assert _deferral_row_fact_still_holds(driver, item, dict(row)) is True  # type: ignore[arg-type]
    # Rotating ONLY the overlay manifest releases the park.
    (tmp_path / "locks" / "osx-arm64.pip-overlay.txt").write_bytes(_MANIFEST + b"# rotated\n")
    assert _deferral_row_fact_still_holds(driver, item, dict(row)) is False  # type: ignore[arg-type]


# --- generation participation --------------------------------------------------


def _locked_artifacts(tmp_path: Path) -> LockArtifacts:
    """Build one complete locked artifact family on disk."""

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    export = _EXPORT_BYTES
    lock_bytes = _LOCK_BYTES
    (lock_dir / "osx-arm64.lock").write_bytes(lock_bytes)
    (lock_dir / "osx-arm64.resolved.json").write_bytes(export)
    (lock_dir / "osx-arm64.resolved.sha256").write_text(f"{hash_bytes(export)}\n", encoding="utf-8")
    return LockArtifacts(
        target="osx-arm64",
        lock_path=lock_dir / "osx-arm64.lock",
        export_path=lock_dir / "osx-arm64.resolved.json",
        export_hash_path=lock_dir / "osx-arm64.resolved.sha256",
        lock_bytes=lock_bytes,
        export_bytes=export,
        declared_export_hash=hash_bytes(export),
    )


def test_generation_payload_without_overlay_is_bit_identical_to_history(
    tmp_path: Path,
) -> None:
    """Existing intent generations are untouched by the overlay feature."""

    lock = _locked_artifacts(tmp_path)
    probes = IntentProbes(imports=("a", "b", "c"), export_checks=(), source_build=())
    legacy_payload = stable_hash(
        {
            "intent": "core",
            "target": "osx-arm64",
            "lock_sha256": hash_bytes(lock.lock_bytes or b""),
            "resolved_export_sha256": lock.declared_export_hash,
            "probes": {"imports": ("a", "b", "c"), "export_checks": [], "source_build": []},
        }
    )
    assert compute_environment_generation("core", lock, probes) == legacy_payload
    assert compute_environment_generation("core", lock, probes, pip_overlay=None) == legacy_payload


def test_generation_requires_and_tracks_the_pinned_overlay(tmp_path: Path) -> None:
    """An unpinned overlay yields no generation; a re-pin moves the generation."""

    lock = _locked_artifacts(tmp_path)
    probes = IntentProbes(imports=("a", "b", "c"), export_checks=(), source_build=())
    unpinned = _overlay(tmp_path, manifest=None)
    assert compute_environment_generation("audio", lock, probes, pip_overlay=unpinned) is None
    pinned = _overlay(tmp_path)
    first = compute_environment_generation("audio", lock, probes, pip_overlay=pinned)
    assert first is not None
    repinned = replace(pinned, manifest_bytes=_MANIFEST + b"# rotated\n")
    assert compute_environment_generation("audio", lock, probes, pip_overlay=repinned) != first


# --- lifecycle ordering and fail-closed overlay step ---------------------------
# (The lifecycle fixtures build from the SAME module-level lock family above;
# a section-local _EXPORT_BYTES redefinition would silently unbind _MANIFEST.)


class _OverlayFakeBackend:
    """Record lifecycle calls, including the overlay step, without real tools."""

    def __init__(self, events: list[str], *, receipt: Mapping[str, Any] | None) -> None:
        self.events = events
        self._receipt = receipt

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
        return _EXPORT_BYTES

    def overlay(self, prefix: Path, manifest_path: Path) -> Mapping[str, Any]:
        self.events.append("overlay")
        assert manifest_path.is_file()
        if self._receipt is None:
            raise AssertionError("overlay receipt not configured")
        return self._receipt

    def probe(self, prefix: Path, probes: IntentProbes) -> Sequence[ProbeResult]:
        self.events.append("probe")
        return tuple(ProbeResult(name, True, "ok") for name in expected_probe_names(probes))

    def remove(self, prefix: Path) -> None:
        self.events.append("remove")


def _overlay_intent(tmp_path: Path, *, manifest: bytes | None = _MANIFEST) -> EnvironmentIntent:
    """Build one real EnvironmentIntent whose overlay files live under tmp.

    The intent carries a COMMITTED locked family: the lifecycle builds from
    committed inputs and refuses a lockless intent before the overlay step
    could even be reached.
    """

    lock_dir = tmp_path / "intent" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / "osx-arm64.lock").write_bytes(_LOCK_BYTES)
    (lock_dir / "osx-arm64.resolved.json").write_bytes(_EXPORT_BYTES)
    (lock_dir / "osx-arm64.resolved.sha256").write_text(
        f"{hash_bytes(_EXPORT_BYTES)}\n", encoding="utf-8"
    )
    probes = IntentProbes(imports=("a", "b", "c"), export_checks=(), source_build=())
    (lock_dir / "osx-arm64.probes.json").write_bytes(
        canonical_probe_receipt_bytes(
            tuple(ProbeResult(name, True, "ok") for name in expected_probe_names(probes))
        )
    )
    overlay = _overlay(tmp_path / "intent", manifest=manifest)
    return EnvironmentIntent(
        name="audio",
        phase=next(iter(load_environment_registry().phase_order)),
        framework="pytorch",
        description="test",
        split_guidance="test",
        channels=("conda-forge",),
        dependencies=("python",),
        probes=probes,
        lock=LockArtifacts(
            target="osx-arm64",
            lock_path=lock_dir / "osx-arm64.lock",
            export_path=lock_dir / "osx-arm64.resolved.json",
            export_hash_path=lock_dir / "osx-arm64.resolved.sha256",
            lock_bytes=_LOCK_BYTES,
            export_bytes=_EXPORT_BYTES,
            declared_export_hash=hash_bytes(_EXPORT_BYTES),
            declared_dependencies=("python",),
        ),
        generation=None,
        pip_overlay=overlay,
    )


def _lifecycle(tmp_path: Path, backend: _OverlayFakeBackend) -> SequentialEnvironmentLifecycle:
    """Build one deterministic lifecycle around the overlay fake."""

    return SequentialEnvironmentLifecycle(
        backend,
        EffortTracker({"environment": StageCap(attempts=1, seconds=5, bytes=100)}),
        env_root=tmp_path / "active",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )


def test_lifecycle_overlays_between_create_and_probe(tmp_path: Path) -> None:
    """The overlay step runs after creation and before every probe."""

    intent = _overlay_intent(tmp_path)
    receipt = {
        "manifest_sha256": hash_bytes(_MANIFEST),
        "packages": [dict(row) for row in parse_pip_overlay_manifest(_MANIFEST)],
    }
    events: list[str] = []
    lifecycle = _lifecycle(tmp_path, _OverlayFakeBackend(events, receipt=receipt))
    lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    assert events == ["create", "overlay", "probe", "use", "remove"]


class _OverlaylessFakeBackend(_OverlayFakeBackend):
    """A conda-only backend that predates the overlay action."""

    overlay = None  # type: ignore[assignment]


def test_declared_overlay_fails_closed_without_backend_support(tmp_path: Path) -> None:
    """A backend lacking the overlay action refuses an overlay intent outright."""

    intent = _overlay_intent(tmp_path)
    events: list[str] = []
    lifecycle = _lifecycle(tmp_path, _OverlaylessFakeBackend(events, receipt=None))
    with pytest.raises(EnvironmentOverlayError, match="no overlay action"):
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    # The prefix is still torn down and no probe or use ever ran.
    assert events == ["create", "remove"]


def test_unpinned_overlay_refuses_to_run(tmp_path: Path) -> None:
    """A declared-but-unpinned overlay never silently runs conda-only."""

    intent = _overlay_intent(tmp_path, manifest=None)
    events: list[str] = []
    receipt = {"manifest_sha256": hash_bytes(b""), "packages": []}
    lifecycle = _lifecycle(tmp_path, _OverlayFakeBackend(events, receipt=receipt))
    with pytest.raises(EnvironmentOverlayError, match="unpinned"):
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    assert "overlay" not in events and "use" not in events


def test_overlay_receipt_mismatch_is_a_typed_refusal(tmp_path: Path) -> None:
    """A backend receipt attesting different bytes or rows fails the run."""

    intent = _overlay_intent(tmp_path)
    events: list[str] = []
    wrong = {
        "manifest_sha256": hash_bytes(b"other"),
        "packages": [dict(row) for row in parse_pip_overlay_manifest(_MANIFEST)],
    }
    lifecycle = _lifecycle(tmp_path, _OverlayFakeBackend(events, receipt=wrong))
    with pytest.raises(EnvironmentOverlayError, match="different manifest"):
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    assert "use" not in events


def test_shipped_overlays_cover_speechbrain_and_pykeen() -> None:
    """The committed speech/graph overlay manifests serve the pip-only libraries.

    This is the census's 50-row leverage target made real: the covering-intent
    search proves coverage from the LOCKED inventory (conda export + pinned
    overlay rows), so speechbrain and pykeen models admit instead of parking as
    `deferred:needs-environment-coverage`.
    """

    registry = load_environment_registry()
    speech = find_covering_intents("speechbrain", registry)
    assert [item.intent for item in speech] == ["speech"]
    assert speech[0].basis is CoverageBasis.LOCKED_INVENTORY
    pykeen = find_covering_intents("pykeen", registry)
    assert [item.intent for item in pykeen] == ["graph"]
    assert pykeen[0].basis is CoverageBasis.LOCKED_INVENTORY


def test_routed_inventory_appends_pinned_overlay_rows(tmp_path: Path) -> None:
    """Coverage/disclosure/digest consumers see overlay rows behind a locked lane."""

    lock = _locked_artifacts(tmp_path)
    intent = SimpleNamespace(lock=lock, pip_overlay=_overlay(tmp_path))
    registry = SimpleNamespace(intents={"audio": intent})
    rows = _routed_environment_packages(registry, "audio")  # type: ignore[arg-type]
    names = [str(row["name"]) for row in rows]
    assert "speechbrain" in names and "hyperpyyaml" in names
    # An unlocked conda lane still contributes nothing, overlay or not.
    unlocked = SimpleNamespace(
        lock=replace(lock, lock_bytes=None, export_bytes=None, declared_export_hash=None),
        pip_overlay=_overlay(tmp_path),
    )
    assert (
        _routed_environment_packages(
            SimpleNamespace(intents={"audio": unlocked}),
            "audio",  # type: ignore[arg-type]
        )
        == ()
    )
