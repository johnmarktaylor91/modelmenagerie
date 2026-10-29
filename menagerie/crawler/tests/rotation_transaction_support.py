"""Shared fixtures for the rotation-transaction tests and their subprocesses.

C-ROTATION-TXN (T3-T7 + T5-11 + T3-T6 residual). Everything here is
deterministic from a short tag so the in-process tests, the SIGKILL child
driver, and the scripted CLI operator all reconstruct the exact same solve
bytes without any side channel.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from menagerie.crawler.env_lifecycle import (
    ArtifactReceipt,
    IntentProbes,
    ProbeResult,
    SolveResult,
    canonical_probe_receipt_bytes,
    expected_probe_names,
    parse_resolved_export,
)
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash

TARGET = "osx-arm64"
INTENT = "demo"

#: Ordered publication cuts for the all-or-nothing matrix: each is one SIGKILL
#: injection point in the child driver. ``old`` cuts must leave the committed
#: OLD family authoritative; ``new`` cuts must leave the complete,
#: provenance-valid NEW family authoritative.
OLD_FAMILY_CUTS = (
    "after-solve-commit",
    "after-probe-proof",
    "before-generation-stage",
    "before-manifest-switch",
)
NEW_FAMILY_CUTS = (
    f"before-flat:{TARGET}.lock",
    f"before-flat:{TARGET}.probes.json",
    f"before-flat:{TARGET}.provenance.json",
    f"before-flat:{TARGET}.resolved.json",
    f"before-flat:{TARGET}.resolved.sha256",
    "before-journal-clear",
    "after-publish",
)


@dataclass(frozen=True)
class SolveFixture:
    """One deterministic fake solve: artifact, lock, export, and receipt."""

    tag: str
    artifact_bytes: bytes
    artifact_sha256: str
    url: str
    lock_bytes: bytes
    export_bytes: bytes
    receipt_bytes: bytes


def solve_fixture(
    tag: str, *, probe_names: Sequence[str] = ("import:a", "import:b", "import:c")
) -> SolveFixture:
    """Build the deterministic solve outputs for one tag.

    Parameters
    ----------
    tag:
        Short identity-bearing suffix; distinct tags yield distinct families.
    probe_names:
        Exact receipt names, defaulting to the mini registry's three imports.
    """

    artifact_bytes = f"conda artifact {tag}".encode()
    artifact_sha256 = hash_bytes(artifact_bytes)
    url = f"https://conda.example.test/python-{tag}.conda"
    lock_bytes = f"{url}#{artifact_sha256.removeprefix('sha256:')}\n".encode()
    export_bytes = (
        canonical_json_bytes(
            {
                "packages": [
                    {
                        "name": "python",
                        "version": "3.11",
                        "build": f"h{tag}_0",
                        "url": url,
                        "sha256": artifact_sha256,
                    }
                ]
            }
        )
        + b"\n"
    )
    receipt_bytes = canonical_probe_receipt_bytes(
        tuple(ProbeResult(name, True, "ok") for name in probe_names)
    )
    return SolveFixture(
        tag=tag,
        artifact_bytes=artifact_bytes,
        artifact_sha256=artifact_sha256,
        url=url,
        lock_bytes=lock_bytes,
        export_bytes=export_bytes,
        receipt_bytes=receipt_bytes,
    )


def family_members(fixture: SolveFixture, provenance: dict) -> dict[str, bytes]:
    """Return the exact publication member map for one fixture solve."""

    return {
        f"{TARGET}.lock": fixture.lock_bytes,
        f"{TARGET}.resolved.json": fixture.export_bytes,
        f"{TARGET}.resolved.sha256": f"{hash_bytes(fixture.export_bytes)}\n".encode(),
        f"{TARGET}.probes.json": fixture.receipt_bytes,
        f"{TARGET}.provenance.json": canonical_json_bytes(provenance) + b"\n",
    }


def setup_artifact_identity(fixture: SolveFixture) -> str:
    """Compute the complete-family setup-artifact identity for one fixture.

    Mirrors ``driver_admission._intent_setup_artifact_identity`` for a
    complete four-member family without a pip overlay.
    """

    export_hash_bytes = f"{hash_bytes(fixture.export_bytes)}\n".encode()
    return stable_hash(
        {
            f"{TARGET}.lock": hash_bytes(fixture.lock_bytes),
            f"{TARGET}.resolved.json": hash_bytes(fixture.export_bytes),
            f"{TARGET}.resolved.sha256": hash_bytes(export_hash_bytes),
            f"{TARGET}.probes.json": hash_bytes(fixture.receipt_bytes),
        }
    )


def rotation_provenance(envs_root: Path, intent: str, fixture: SolveFixture) -> dict:
    """Author one consistent rotation provenance record for a fixture family."""

    from menagerie.crawler.tools.rotate_setup_artifacts import (
        ROTATION_PROVENANCE_SCHEMA_VERSION,
    )

    prefix = f"menagerie/crawler/envs/{intent}/locks"
    return {
        "schema_version": ROTATION_PROVENANCE_SCHEMA_VERSION,
        "intent": intent,
        "target": TARGET,
        "lock_path": f"{prefix}/{TARGET}.lock",
        "lock_sha256": hash_bytes(fixture.lock_bytes),
        "resolved_export_path": f"{prefix}/{TARGET}.resolved.json",
        "resolved_export_sha256": hash_bytes(fixture.export_bytes),
        "probe_receipt_path": f"{prefix}/{TARGET}.probes.json",
        "probe_receipt_sha256": hash_bytes(fixture.receipt_bytes),
        "spec_path": f"menagerie/crawler/envs/{intent}/environment.yml",
        "spec_sha256": hash_bytes((envs_root / intent / "environment.yml").read_bytes()),
        "probe_contract_path": "menagerie/crawler/envs/probes.yml",
        "probe_contract_sha256": hash_bytes((envs_root / "probes.yml").read_bytes()),
        "solver": {"command": ["scripted-test-operator"]},
        "host": {"system": "Darwin", "machine": "arm64", "on_target_solve": True},
        "clean_create": {
            "validated": True,
            "inventory_matches_resolved_export": True,
        },
        "artifact_hash_validation": {"algorithm": "sha256", "verified": True},
    }


def write_attested_family(envs_root: Path, intent: str, fixture: SolveFixture) -> None:
    """Write one complete, provenance-consistent flat family for an intent."""

    lock_dir = envs_root / intent / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    members = family_members(fixture, rotation_provenance(envs_root, intent, fixture))
    for name, content in members.items():
        (lock_dir / name).write_bytes(content)


def mini_envs_root(tmp_path: Path, *, old: SolveFixture) -> Path:
    """Build one minimal registry root with a single attested ``demo`` intent."""

    root = tmp_path / "mini-envs"
    root.mkdir(parents=True)
    (root / "registry.yml").write_text(
        json.dumps(
            {
                "intents": {
                    INTENT: {
                        "phase": "pytorch",
                        "framework": "pytorch",
                        "description": "demo",
                        "split_guidance": "demo",
                    }
                },
                "phase_order": ["pytorch", "native-tail"],
                "small_set_target": True,
                "hard_cap": None,
                "split_guidance": "x",
            }
        ),
        encoding="utf-8",
    )
    (root / "probes.yml").write_text(
        json.dumps({"intents": {INTENT: {"imports": ["a", "b", "c"], "export_checks": []}}}),
        encoding="utf-8",
    )
    intent_dir = root / INTENT
    intent_dir.mkdir()
    (intent_dir / "environment.yml").write_text(
        json.dumps({"name": INTENT, "channels": ["conda-forge"], "dependencies": ["python"]}),
        encoding="utf-8",
    )
    write_attested_family(root, INTENT, old)
    return root


def populate_solve_commit(state_root: Path, intent: str, fixture: SolveFixture) -> None:
    """Mimic the operator's solve-commit into one rotation/campaign state root."""

    solve_root = state_root / "solves" / intent / TARGET
    solve_root.mkdir(parents=True, exist_ok=True)
    (solve_root / "explicit.lock").write_bytes(fixture.lock_bytes)
    (solve_root / "resolved.json").write_bytes(fixture.export_bytes)
    unified = f"# scripted unified lock {fixture.tag}\n".encode()
    (solve_root / "conda-lock.yml").write_bytes(unified)
    manifest = {
        "protocol_version": "menagerie.crawler.environment-operator.v1",
        "intent": intent,
        "target": TARGET,
        "lock_sha256": hash_bytes(fixture.lock_bytes),
        "resolved_export_sha256": hash_bytes(fixture.export_bytes),
        "conda_lock_sha256": hash_bytes(unified),
        "artifacts": [
            {
                "url": fixture.url,
                "path": str(cas_artifact_path(state_root, fixture)),
                "sha256": fixture.artifact_sha256,
            }
        ],
    }
    (solve_root / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    cas_entry = cas_artifact_path(state_root, fixture)
    cas_entry.parent.mkdir(parents=True, exist_ok=True)
    cas_entry.write_bytes(fixture.artifact_bytes)


def cas_artifact_path(state_root: Path, fixture: SolveFixture) -> Path:
    """Return the CAS path of the fixture artifact below one state root."""

    return state_root / "artifact-cas" / "sha256" / fixture.artifact_sha256.removeprefix("sha256:")


class ScriptedRotationBackend:
    """In-process environment backend replaying one deterministic solve.

    The backend mimics the production operator's observable side effects
    (solve-commit into the state root, CAS materialization) so the rotation
    transaction around it is exercised against realistic on-disk state.
    """

    def __init__(
        self,
        state_root: Path,
        fixture: SolveFixture,
        *,
        fail_probe: bool = False,
        after_solve_commit: Optional[Callable[[], None]] = None,
        after_probe_proof: Optional[Callable[[], None]] = None,
    ) -> None:
        self.state_root = state_root
        self.fixture = fixture
        self.fail_probe = fail_probe
        self.after_solve_commit = after_solve_commit
        self.after_probe_proof = after_probe_proof

    def solve(self, environment_file: Path, target: str) -> SolveResult:
        populate_solve_commit(self.state_root, environment_file.parent.name, self.fixture)
        if self.after_solve_commit is not None:
            self.after_solve_commit()
        return SolveResult(
            self.fixture.lock_bytes,
            self.fixture.export_bytes,
            1.0,
            len(self.fixture.artifact_bytes),
            (ArtifactReceipt(self.fixture.url, self.fixture.artifact_sha256),),
        )

    def create(self, lock_file: Path, prefix: Path) -> bytes:
        prefix.mkdir(parents=True, exist_ok=True)
        return parse_resolved_export(self.fixture.export_bytes)

    def probe(self, prefix: Path, probes: IntentProbes) -> Sequence[ProbeResult]:
        if self.fail_probe:
            return tuple(
                ProbeResult(name, False, "scripted failure")
                for name in expected_probe_names(probes)
            )
        results = tuple(ProbeResult(name, True, "ok") for name in expected_probe_names(probes))
        if self.after_probe_proof is not None:
            self.after_probe_proof()
        return results

    def remove(self, prefix: Path) -> None:
        shutil.rmtree(prefix, ignore_errors=True)
