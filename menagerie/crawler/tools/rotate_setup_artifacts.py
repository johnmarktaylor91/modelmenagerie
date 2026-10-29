"""Rotate one intent's committed setup artifacts through the canonical env lane.

A quarantined environment generation (``environment-cleanup-quarantined``) holds
every parked work item while the TYPED committed setup-artifact identity -- the
hash over the four files ``envs/<intent>/locks/<target>.lock`` /
``.resolved.json`` / ``.resolved.sha256`` / ``.probes.json`` -- is unchanged and
no clearance event names the quarantine. Rung 12 proved the failure mode this
tool exists to close: wave 9 edited ``environment.yml`` specs and CALLED that a
rotation, but an environment-spec edit is not a setup-artifact rotation, so the
core identity stayed at the quarantined value and all ~43 rebuild parks were
re-claimed by the deferral valve before env/worker/reducer on every subsequent
leg. Worse, the parks themselves kept the env lane from ever running for the
intent, so nothing would ever rewrite the artifacts: a deadlock only an
explicit rotation transaction breaks.

The transaction per intent is the artifact-producing half of
``SequentialEnvironmentLifecycle.run`` against the SAME production operator
wrapper the driver uses (``python -m menagerie.crawler.operator_environment``):

1. solve ``environment.yml`` on-target (fresh state root: no stale solve cache),
2. verify materialized artifact digests against the exact lock (the backend
   adapter enforces this),
3. create the prefix from the staged lock, require the installed inventory to
   equal the declared resolved export byte-for-byte,
4. run every declared probe, require all to pass, canonicalize the receipt,
5. tear the prefix down and absence-check every generation-owned root,
6. only then publish the whole family plus its provenance binding as ONE
   atomic generation switch: the complete family is staged into an immutable
   ``locks/generations/<digest>/`` directory and committed by a single atomic
   rename of the ``<target>.family.json`` manifest, so every crash cut leaves
   the complete old family XOR the complete proven new family (the flat
   committed members are then materialized from the generation; a crash there
   is completed by ``--complete-publication``),
7. REFUSE before publication if the setup-artifact identity would not change,
   and
8. invalidate and republish every named campaign solve cache from the
   rotation outcome (``--campaign-state-root``), closing the
   stale-cache-reverts-the-rotation hole.

Publication happens strictly after the probes pass, so a failed rotation can
never leave a half-rotated family whose accidental new identity would release
parks against an environment that was never proven. Rotation is the ONLY
solving operation: the launch lane builds FROM the committed family and
refuses an intent without one.

The old/new identity pair is printed for every intent so a launch log carries
the rotation evidence, and the provenance file binds spec, probe contract,
solver command, host, and artifact digests (the committed-lock anti-fabrication
gate requires exactly that binding).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from menagerie.crawler.driver_admission import (
    CommandEnvironmentBackend,
    _intent_setup_artifact_identity,
)
from menagerie.crawler.env_lifecycle import (
    DiskFree,
    DiskSpaceAuthority,
    EnvironmentBackend,
    _atomic_write,
    _backend_owned_paths,
    _require_lock_inventory_match,
    canonical_probe_receipt_bytes,
    disk_free_bytes,
    parse_exact_lock,
    parse_resolved_export,
    validate_probe_receipts,
)
from menagerie.crawler.execution_lock import (
    DEFAULT_EXECUTION_LOCK_TIMEOUT_SECONDS,
    acquire_global_execution_flock,
    release_global_execution_flock,
)
from menagerie.crawler.envs import (
    DEFAULT_ENVS_ROOT,
    DEFAULT_ENVIRONMENT_DISK_FLOOR_BYTES,
    FAMILY_MANIFEST_SCHEMA_VERSION,
    GENERATIONS_DIR_NAME,
    EnvironmentSpecError,
    _flat_family_has_valid_provenance,
    family_manifest_path,
    load_environment_registry,
    parse_pip_overlay_manifest,
    probe_declaration_digest,
    publication_journal_path,
    render_pip_overlay_manifest,
    resolve_family_manifest,
)
from menagerie.crawler.identity import (
    atomic_replace_bytes,
    canonical_json_bytes,
    fsync_directory,
    hash_bytes,
    stable_hash,
)
from menagerie.crawler.operator_environment import (
    PROTOCOL_VERSION as OPERATOR_PROTOCOL_VERSION,
    _cas_path,
    _exclusive_file_lock,
    _stream_sha256,
    _wheel_cas_entry,
    _wheel_cas_file,
)

ROTATION_PROVENANCE_SCHEMA_VERSION = "menagerie.crawler.rotation-lock-provenance.v1"

#: The env lane refuses to build below 30 GiB; a rotation creates exactly one
#: prefix at a time and tears it down, so it carries the same default floor and
#: an explicit override for constrained operator hosts.
DEFAULT_MINIMUM_FREE_BYTES = DEFAULT_ENVIRONMENT_DISK_FLOOR_BYTES


class RotationError(RuntimeError):
    """Raised when a setup-artifact rotation cannot be completed exactly."""


ROTATION_PUBLICATION_JOURNAL_SCHEMA_VERSION = "menagerie.crawler.rotation-publication-journal.v1"

#: Sentinel distinguishing "compare-and-swap not requested" (legacy callers)
#: from "the rotation observed NO committed manifest" (``None``).
_CAS_UNSET = object()


def rotation_publication_lock_path(lock_dir: Path, target: str) -> Path:
    """Return the per-intent, per-target publication lock path (EN-7c).

    Gitignored runtime state exactly like the family manifest: the lock
    serializes concurrent publications on ONE host, where sol's reproduced
    OSError 66 collision lives.
    """

    return lock_dir / f".{target}.publication.lock"


def rotate_intent_write_destinations(
    *,
    envs_root: Path,
    intent: str,
    state_root: Path,
    env_root: Path,
    campaign_state_roots: Sequence[Path] = (),
) -> tuple[tuple[str, Path], ...]:
    """Enumerate every filesystem destination one rotation writes (EN-7a).

    The rotation writes to five independent places that can live on five
    different filesystems: the operator state root (solve cache + artifact
    CAS), the transient prefix root, the process temporary directory (overlay
    resolution scratch), the committed lock directory (publication), and each
    campaign state root (republication). A floor check on any single one says
    nothing about the others -- rung GO-day cost: rotations died on the
    unmeasured root mid-flight.

    Returns
    -------
    tuple[tuple[str, Path], ...]
        ``(role, path)`` pairs in authority order for
        :class:`menagerie.crawler.env_lifecycle.DiskSpaceAuthority`.
    """

    destinations: list[tuple[str, Path]] = [
        ("state-root", state_root),
        ("env-root", env_root),
        ("tmp-root", Path(tempfile.gettempdir())),
        ("publication", envs_root / intent / "locks"),
    ]
    destinations.extend(("campaign-state", root) for root in campaign_state_roots)
    return tuple(destinations)


def _require_disk_floor(
    destinations: Sequence[tuple[str, Path]],
    *,
    minimum_free_bytes: int,
    disk_free: DiskFree,
    stage: str,
) -> None:
    """Refuse typed when ANY measured write destination is below the floor.

    A destination that cannot be measured fails CLOSED: an unmeasurable root
    is indistinguishable from a full one for a transaction that must never
    half-publish.
    """

    snapshot = DiskSpaceAuthority(destinations, disk_free=disk_free).snapshot()
    for observation in snapshot.destinations:
        if observation.free_bytes is None:
            raise RotationError(
                f"rotation {stage} disk check could not measure the "
                f"{observation.role} destination {observation.destination}: "
                f"{observation.error}"
            )
        if observation.free_bytes < minimum_free_bytes:
            raise RotationError(
                f"rotation {stage} disk check: only {observation.free_bytes} bytes are "
                f"free below the {observation.role} destination {observation.destination}; "
                f"the floor is {minimum_free_bytes} bytes -- no publication proceeds on "
                "any single-root breach"
            )


def _refuse_whitespace_solve_roots(state_root: Path, env_root: Path) -> None:
    """Refuse solve roots whose paths carry whitespace (EN-7b).

    conda/mamba tooling breaks on whitespace paths (the space-in-volume-name
    shape: ``/Volumes/Devmini SSD``); it cost 3 of 4 rotation GO-day attempts.
    The refusal happens BEFORE any backend work so the failure is one typed
    sentence instead of a mid-solve stack.
    """

    for role, root in (("--state-root", state_root), ("--env-root", env_root)):
        if re.search(r"\s", str(root)):
            raise RotationError(
                f"rotation {role} {str(root)!r} contains whitespace; conda tooling "
                "breaks on whitespace paths (the space-in-volume-name shape) -- "
                "pass a whitespace-free root (an internal scratch volume, not a "
                "display-named external)"
            )


def publish_rotation_family(
    lock_dir: Path,
    target: str,
    members: "dict[str, bytes]",
    *,
    observer: "Optional[Callable[[str], None]]" = None,
    phase_observer: "Optional[Callable[[str], None]]" = None,
    expected_prior_generation: "object" = _CAS_UNSET,
) -> None:
    """Publish one committed lock family as an atomic generation switch.

    T3-T7 (supersedes the T1-09 journal-only window): the setup-artifact
    identity hashes the on-disk members, so a crash between member writes
    leaves a MIXED family whose accidental new identity would release parks
    against an environment that was never proven. The transaction here makes
    the whole-family switch atomic:

    1. the COMPLETE family is staged into an immutable per-generation
       directory (``generations/<digest>/``), fsynced, and never mutated;
    2. ONE atomic rename of the family manifest
       (``<target>.family.json``) commits the publication -- the registry
       loader reads the manifest-named generation as the family authority
       from that instant, so every cut leaves the complete old family XOR the
       complete proven new family, never a mix;
    3. the flat committed member files are then materialized from the
       generation under the publication journal; a crash in this phase leaves
       the family fully readable from its generation while the journal and
       the digest comparison expose the pending materialization, completed
       idempotently by :func:`complete_family_publication`.

    Parameters
    ----------
    lock_dir, target:
        Lock directory of the intent and the exact conda-lock platform.
    members:
        Exact bytes for every family member file, keyed by file name.
    observer:
        Optional per-member hook invoked before each flat member write;
        exists for crash-window tests and never affects publication.
    phase_observer:
        Optional hook invoked before each transaction phase
        (``generation-stage``, ``manifest-switch``, ``journal-clear``);
        exists for crash-window tests and never affects publication.
    expected_prior_generation:
        EN-7c compare-and-swap: the family-manifest generation this rotation
        observed before solving (``None`` when no manifest existed). When
        given, the switch refuses typed if the committed manifest meanwhile
        moved to any THIRD generation; an identical winner converges. Legacy
        callers that omit it keep the unguarded switch.

    Raises
    ------
    RotationError
        If an existing generation directory for these exact bytes fails its
        digest verification (immutable evidence was tampered with); if a
        rename collision lands on divergent bytes; or if the compare-and-swap
        observes a third family.
    """

    lock_dir.mkdir(parents=True, exist_ok=True)
    member_digests = {name: hash_bytes(content) for name, content in members.items()}
    generation = stable_hash({"lock_family_generation": member_digests}).removeprefix("sha256:")
    generations_root = lock_dir / GENERATIONS_DIR_NAME
    generation_dir = generations_root / generation

    def _verify_existing_generation() -> None:
        """Require the existing generation directory to match these exact bytes."""

        for name, content in sorted(members.items()):
            existing = generation_dir / name
            if not existing.is_file() or existing.read_bytes() != content:
                raise RotationError(
                    f"generation directory {generation_dir} exists but member "
                    f"{name} does not match its immutable bytes"
                )

    # EN-7c: the whole publication runs under the per-intent publication lock,
    # so two same-host rotations serialize instead of colliding mid-transaction.
    with _exclusive_file_lock(rotation_publication_lock_path(lock_dir, target)):
        if expected_prior_generation is not _CAS_UNSET:
            # Compare-and-swap over the manifest authority: the switch is lawful
            # only against the family this rotation actually observed (or an
            # identical winner). A THIRD family means a concurrent rotation
            # proved and published something else -- overwriting it would
            # revert a proven publication.
            manifest_path = family_manifest_path(lock_dir, target)
            committed: Optional[str] = None
            if manifest_path.is_file():
                try:
                    committed_value = json.loads(manifest_path.read_bytes()).get("generation")
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RotationError(
                        f"committed family manifest {manifest_path} is unreadable; "
                        "refusing to switch over torn authority"
                    ) from exc
                committed = str(committed_value) if committed_value is not None else None
            if committed is not None and committed not in (generation, expected_prior_generation):
                raise RotationError(
                    f"another rotation published family generation {committed} for "
                    f"target {target!r} since this rotation observed "
                    f"{expected_prior_generation!r}; compare-and-swap refuses the "
                    "switch -- re-run the rotation against the current family"
                )

        if phase_observer is not None:
            phase_observer("generation-stage")
        if generation_dir.is_dir():
            # Idempotent republication of the exact same bytes: the generation is
            # immutable evidence, so it must still verify -- anything else is
            # tampering, never something to silently rewrite.
            _verify_existing_generation()
        else:
            staging = generations_root / f".stage-{generation}-{os.getpid()}"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            for name, content in sorted(members.items()):
                _atomic_write(staging / name, content)
            fsync_directory(staging)
            try:
                os.rename(staging, generation_dir)
            except OSError:
                # Sol's reproduced OSError 66: an identical winner published
                # this exact generation between the existence check and the
                # rename. Identical bytes converge; anything else refuses.
                shutil.rmtree(staging, ignore_errors=True)
                if not generation_dir.is_dir():
                    raise
                _verify_existing_generation()
            fsync_directory(generations_root)

        if phase_observer is not None:
            phase_observer("manifest-switch")
        manifest = {
            "schema_version": FAMILY_MANIFEST_SCHEMA_VERSION,
            "target": target,
            "generation": generation,
            "members": member_digests,
        }
        _atomic_write(
            family_manifest_path(lock_dir, target), canonical_json_bytes(manifest) + b"\n"
        )
        fsync_directory(lock_dir)

        # The publication is committed. Materialize the git-visible flat members
        # from the generation, journaled exactly as before so a pre-manifest
        # reader (or an operator who lost the manifest) still sees the window.
        journal = publication_journal_path(lock_dir, target)
        entry = {
            "schema_version": ROTATION_PUBLICATION_JOURNAL_SCHEMA_VERSION,
            "target": target,
            "members": member_digests,
        }
        _atomic_write(journal, canonical_json_bytes(entry) + b"\n")
        fsync_directory(lock_dir)
        for name in sorted(members):
            if observer is not None:
                observer(name)
            _atomic_write(lock_dir / name, members[name])
        if phase_observer is not None:
            phase_observer("journal-clear")
        journal.unlink()
        fsync_directory(lock_dir)


def complete_family_publication(lock_dir: Path, target: str) -> "dict[str, object]":
    """Complete one committed publication's flat materialization idempotently.

    Roll-forward only: the family manifest is the authority, so this never
    changes what the registry reads -- it re-materializes any flat member
    that is missing or differs from the manifest digest, byte-for-byte from
    the immutable generation directory, and clears a leftover publication
    journal. Safe to run any number of times.

    Parameters
    ----------
    lock_dir, target:
        Lock directory of the intent and the exact conda-lock platform.

    Returns
    -------
    dict[str, object]
        ``{"completed": True, "generation": ..., "materialized_members": [...]}``.

    Raises
    ------
    RotationError
        If no committed family manifest exists or it fails validation; a
        family that cannot prove its generation has nothing to roll forward.
    """

    if not family_manifest_path(lock_dir, target).is_file():
        raise RotationError(
            f"no family manifest below {lock_dir} for target {target!r}; only a "
            "committed publication can be completed"
        )
    try:
        resolved = resolve_family_manifest(lock_dir, target)
    except EnvironmentSpecError as exc:
        raise RotationError(
            f"family manifest below {lock_dir} fails validation and cannot be rolled forward: {exc}"
        ) from exc
    flat_disagrees = any(
        not (lock_dir / name).is_file()
        or hash_bytes((lock_dir / name).read_bytes()) != digest
        for name, digest in resolved["members"].items()
    )
    if (
        not publication_journal_path(lock_dir, target).is_file()
        and flat_disagrees
        and _flat_family_has_valid_provenance(lock_dir, target)
    ):
        # Host-local manifests are intentionally gitignored. A complete,
        # provenance-bound flat family can therefore be newer after a pull
        # from another rotation host. Discard only the stale local pointer;
        # rolling it forward would overwrite committed newer authority.
        family_manifest_path(lock_dir, target).unlink()
        publication_journal_path(lock_dir, target).unlink(missing_ok=True)
        fsync_directory(lock_dir)
        return {
            "completed": False,
            "stale_manifest_discarded": True,
            "generation": resolved["generation"],
            "materialized_members": [],
        }
    generation_dir: Path = resolved["generation_dir"]
    materialized: list[str] = []
    for name, digest in sorted(resolved["members"].items()):
        flat = lock_dir / name
        if flat.is_file() and hash_bytes(flat.read_bytes()) == digest:
            continue
        _atomic_write(flat, (generation_dir / name).read_bytes())
        materialized.append(name)
    publication_journal_path(lock_dir, target).unlink(missing_ok=True)
    fsync_directory(lock_dir)
    return {
        "completed": True,
        "generation": resolved["generation"],
        "materialized_members": materialized,
    }


def republish_campaign_solve_cache(
    campaign_state_root: Path,
    *,
    rotation_state_root: Path,
    intent: str,
    target: str,
    lock_bytes: bytes,
    export_bytes: bytes,
    overlay_manifest_bytes: Optional[bytes] = None,
) -> "dict[str, object]":
    """Invalidate and republish one campaign solve cache from a rotation.

    T1-04 (S2-confirmed): the campaign driver's persistent solve cache lives
    under its own state root, which a rotation never touched -- so the cache
    kept serving pre-rotation bytes that passed names-only currency and the
    next leg reverted the rotation. This publishes the rotation's outcome INTO
    the campaign cache: stale solve members are removed first (invalidation is
    never conditional on the republication completing), every lock artifact is
    copied digest-verified from the rotation CAS into the campaign CAS, and
    the solve members are re-written byte-identical to the committed family,
    manifest last.

    Parameters
    ----------
    campaign_state_root:
        The campaign's environment-operator state root
        (``.crawl-local/environment-operator``).
    rotation_state_root:
        The rotation's scratch state root holding the fresh solve and CAS.
    intent, target:
        Intent name and exact conda-lock platform.
    lock_bytes, export_bytes:
        The exact bytes the rotation committed.
    overlay_manifest_bytes:
        Optional committed pip-overlay manifest whose pinned wheels are also
        republished into the campaign wheel CAS.

    Returns
    -------
    dict[str, object]
        Republication receipt: solve root, artifact and wheel counts.

    Raises
    ------
    RotationError
        If a rotation-side artifact is missing or fails its digest; the stale
        campaign cache stays invalidated so it can never serve pre-rotation
        bytes.
    """

    solve_root = campaign_state_root / "solves" / intent / target
    solve_root.mkdir(parents=True, exist_ok=True)
    receipts = parse_exact_lock(lock_bytes)
    republished_artifacts = 0
    republished_wheels = 0
    with _exclusive_file_lock(solve_root / ".solve.lock"):
        # Invalidate FIRST: a partial republication must read as no cache at
        # all, never as the stale pre-rotation solve.
        for name in ("manifest.json", "explicit.lock", "resolved.json", "conda-lock.yml"):
            (solve_root / name).unlink(missing_ok=True)
        artifacts: list[dict[str, str]] = []
        for receipt in receipts:
            source = _cas_path(rotation_state_root, receipt.sha256)
            if not source.is_file() or _stream_sha256(source) != receipt.sha256:
                raise RotationError(
                    f"rotation CAS artifact is missing or corrupt for {receipt.url}"
                )
            destination = _cas_path(campaign_state_root, receipt.sha256)
            if not destination.is_file() or _stream_sha256(destination) != receipt.sha256:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(
                    f".{destination.name}.{os.getpid()}.republish.tmp"
                )
                shutil.copyfile(source, temporary)
                if _stream_sha256(temporary) != receipt.sha256:
                    temporary.unlink(missing_ok=True)
                    raise RotationError(
                        f"republished CAS artifact changed in transit for {receipt.url}"
                    )
                os.replace(temporary, destination)
                fsync_directory(destination.parent)
                republished_artifacts += 1
            artifacts.append(
                {"url": receipt.url, "path": str(destination), "sha256": receipt.sha256}
            )
        if overlay_manifest_bytes is not None:
            campaign_wheels = campaign_state_root / "wheel-cas"
            rotation_wheels = rotation_state_root / "wheel-cas"
            for row in parse_pip_overlay_manifest(overlay_manifest_bytes):
                digest = str(row["sha256"])
                source_file = _wheel_cas_file(rotation_wheels, digest)
                if source_file is None:
                    raise RotationError(
                        f"rotation wheel CAS lacks the pinned overlay artifact for {row['name']}"
                    )
                entry = _wheel_cas_entry(campaign_wheels, digest)
                if _wheel_cas_file(campaign_wheels, digest) is not None:
                    continue
                staging = entry.with_name(f"{entry.name}.republish-{os.getpid()}")
                staging.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, staging / source_file.name)
                os.rename(staging, entry)
                republished_wheels += 1
        atomic_replace_bytes(solve_root / "explicit.lock", lock_bytes)
        atomic_replace_bytes(solve_root / "resolved.json", export_bytes)
        rotation_solve_root = rotation_state_root / "solves" / intent / target
        rotation_unified = rotation_solve_root / "conda-lock.yml"
        if rotation_unified.is_file():
            unified_bytes = rotation_unified.read_bytes()
            declared_unified = _declared_unified_lock_digest(rotation_solve_root / "manifest.json")
            if declared_unified is not None and hash_bytes(unified_bytes) != declared_unified:
                raise RotationError(
                    f"rotation unified lock {rotation_unified} does not match the "
                    "solve manifest's conda_lock_sha256 binding; the rotation solve "
                    "cache is torn and must not be republished"
                )
            atomic_replace_bytes(solve_root / "conda-lock.yml", unified_bytes)
        manifest = {
            "protocol_version": OPERATOR_PROTOCOL_VERSION,
            "intent": intent,
            "target": target,
            "lock_sha256": hash_bytes(lock_bytes),
            "resolved_export_sha256": hash_bytes(export_bytes),
            "artifacts": artifacts,
        }
        atomic_replace_bytes(solve_root / "manifest.json", canonical_json_bytes(manifest) + b"\n")
    return {
        "solve_root": str(solve_root),
        "republished_artifacts": republished_artifacts,
        "republished_wheels": republished_wheels,
    }


def _declared_unified_lock_digest(manifest_path: Path) -> Optional[str]:
    """Read the solve manifest's unified-lock digest binding, if declared.

    Pre-binding solve manifests (no ``conda_lock_sha256`` field) return
    ``None``: the unified lock is a convenience copy, so an older cache is
    not refused -- only a manifest that DECLARES the binding and contradicts
    it is torn evidence.
    """

    if not manifest_path.is_file():
        return None
    try:
        value = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    declared = value.get("conda_lock_sha256") if isinstance(value, dict) else None
    return declared if isinstance(declared, str) else None


def _default_environment_command() -> tuple[str, ...]:
    """Return the canonical operator wrapper command for this interpreter."""

    configured = os.environ.get("MENAGERIE_ENVIRONMENT_COMMAND")
    if configured:
        return tuple(configured.split())
    return (sys.executable, "-m", "menagerie.crawler.operator_environment")


def _repo_relative(path: Path, envs_root: Path) -> str:
    """Return the provenance-bound repository-relative spelling of one path."""

    return f"menagerie/crawler/envs/{path.relative_to(envs_root).as_posix()}"


def _compile_pip_overlay(
    declared_path: Path,
    prefix: Path,
    *,
    declared_inventory: bytes,
    conda_export_sha256: str,
) -> bytes:
    """Resolve one pip-overlay declaration ONCE into a pinned, hashed manifest.

    Resolution runs through the freshly created prefix's own interpreter
    (``pip install --dry-run --report``), so pip sees every conda-provided
    distribution as already installed and the manifest pins ONLY what conda
    lacks -- the overlay adds, it never rewrites. Wheels only
    (``--only-binary=:all:``): a dependency without a wheel for this platform
    fails the rotation loudly instead of smuggling in a source build.

    Parameters
    ----------
    declared_path:
        The intent's ``pip-overlay.in`` direct-requirements declaration.
    prefix:
        Freshly created conda prefix.
    declared_inventory:
        The intent's canonical resolved-export bytes, used to refuse any
        resolution that would shadow a conda-provided package.

    Returns
    -------
    bytes
        Canonical manifest bytes in the closed
        ``name==version --hash=sha256:...`` grammar, sorted by canonical name.

    Raises
    ------
    RotationError
        If resolution fails, an item lacks a sha256, or an item shadows a
        conda-provided distribution.
    """

    interpreter = prefix / "bin" / "python"
    if not interpreter.is_file():
        raise RotationError(
            f"prefix interpreter missing below {prefix}; overlay intents must "
            "declare 'pip' among their conda dependencies"
        )
    with tempfile.TemporaryDirectory(prefix=".overlay-report-") as scratch:
        report_path = Path(scratch) / "report.json"
        completed = subprocess.run(
            [
                str(interpreter),
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--quiet",
                "--only-binary",
                ":all:",
                "--report",
                str(report_path),
                "-r",
                str(declared_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RotationError(
                f"pip overlay resolution failed for {declared_path}: {completed.stderr[-1500:]}"
            )
        report = json.loads(report_path.read_bytes())
    conda_names = {
        re.sub(r"[-_.]+", "-", str(row.get("name"))).casefold()
        for row in json.loads(declared_inventory).get("packages", [])
        if isinstance(row, dict)
    }
    rows: list[tuple[str, str, str]] = []
    for item in report.get("install", []):
        metadata = item.get("metadata") or {}
        name = str(metadata.get("name") or "")
        version = str(metadata.get("version") or "")
        hashes = ((item.get("download_info") or {}).get("archive_info") or {}).get("hashes") or {}
        sha256 = str(hashes.get("sha256") or "")
        if not name or not version or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise RotationError(
                "pip overlay resolution produced an item without a pinned sha256 "
                f"wheel: {name or '<unnamed>'}"
            )
        canonical = re.sub(r"[-_.]+", "-", name).casefold()
        if canonical in conda_names:
            raise RotationError(
                f"pip overlay resolution wants to install {name!r}, which the conda "
                "solve already provides; resolve the version conflict in the conda "
                "spec or the overlay declaration instead of shadowing"
            )
        rows.append((name, version, sha256))
    if not rows:
        raise RotationError(f"pip overlay resolution for {declared_path} pinned no packages")
    # The committed manifest is rendered by the grammar owner and binds the
    # conda resolved export the closure was compiled against (T3-T15), so a
    # manifest can never silently install into a different conda solve. The
    # renderer round-trips through the same closed grammar the registry, the
    # operator, and the anti-fabrication gate all parse.
    return render_pip_overlay_manifest(
        tuple(
            {"name": name, "version": version, "sha256": f"sha256:{sha256}"}
            for name, version, sha256 in rows
        ),
        conda_export_sha256=conda_export_sha256,
    )


def rotate_intent(
    name: str,
    *,
    envs_root: Path,
    target: str,
    environment_command: Sequence[str],
    state_root: Path,
    env_root: Path,
    minimum_free_bytes: int,
    backend: Optional[EnvironmentBackend] = None,
    campaign_state_roots: Sequence[Path] = (),
    disk_free: DiskFree = disk_free_bytes,
    execution_lock_path: Optional[Path] = None,
    execution_lock_timeout_seconds: float = DEFAULT_EXECUTION_LOCK_TIMEOUT_SECONDS,
) -> tuple[str, str]:
    """Rotate one intent's committed setup artifacts and return identities.

    Parameters
    ----------
    name:
        Environment intent name.
    envs_root:
        Environment registry root whose ``<name>/locks/`` family is rotated.
    target:
        Exact conda-lock platform.
    environment_command:
        Argv prefix of the canonical environment operator wrapper.
    state_root:
        Fresh absolute operator state root (solve cache, CAS, registrations).
    env_root:
        Absolute directory receiving the transient verification prefix.
    minimum_free_bytes:
        Refuse to start below this free-byte floor, exactly like the env lane.
    backend:
        Optional injected environment backend; the production default wraps
        ``environment_command`` with the rotation state root.
    campaign_state_roots:
        Campaign environment-operator state roots whose solve caches are
        invalidated and republished from this rotation (T1-04): without this,
        a campaign cache keeps its pre-rotation bytes and the next leg would
        have reverted the rotation.
    disk_free:
        Free-byte measurement, injectable for deterministic tests (EN-7a).
    execution_lock_path:
        Override of the account-global execution flock path; production uses
        the one path the campaign driver's worker supervisor holds during a
        leg, so a rotation can never run concurrent with one (EN-7d).
    execution_lock_timeout_seconds:
        How long to wait for a running leg before refusing typed.

    Returns
    -------
    tuple[str, str]
        ``(old_identity, new_identity)`` -- guaranteed distinct.

    Raises
    ------
    RotationError
        If any lane step fails or the identity did not change.
    """

    # EN-7b: refuse whitespace solve roots BEFORE any lock or backend work.
    _refuse_whitespace_solve_roots(state_root, env_root)
    registry = load_environment_registry(envs_root, target=target)
    if name not in registry.intents:
        raise RotationError(f"unknown environment intent: {name!r}")
    intent = registry.intents[name]
    old_identity = _intent_setup_artifact_identity(intent)
    # T3-T6 residual: --state-root is documented fresh but was never enforced.
    # A reused solve cache can serve a stale name-complete solve, blessing it
    # into a fresh provenance; the rotation must re-solve from scratch.
    stale_solve_cache = state_root / "solves" / name / target
    if stale_solve_cache.exists():
        raise RotationError(
            f"rotation state root {state_root} already holds a solve cache at "
            f"{stale_solve_cache}; a rotation must re-solve from scratch -- pass "
            "a FRESH --state-root"
        )
    env_root.mkdir(parents=True, exist_ok=True)
    # EN-7a: every write destination is measured independently, not just the
    # prefix root; a breach anywhere refuses before any work starts.
    write_destinations = rotate_intent_write_destinations(
        envs_root=envs_root,
        intent=name,
        state_root=state_root,
        env_root=env_root,
        campaign_state_roots=campaign_state_roots,
    )
    _require_disk_floor(
        write_destinations,
        minimum_free_bytes=minimum_free_bytes,
        disk_free=disk_free,
        stage="admission",
    )
    # EN-7c: record the family authority this rotation is rotating FROM, so the
    # publication's compare-and-swap can refuse a concurrent third family.
    observed_manifest = family_manifest_path(envs_root / name / "locks", target)
    observed_prior_generation: Optional[str] = None
    if observed_manifest.is_file():
        try:
            observed_value = json.loads(observed_manifest.read_bytes()).get("generation")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RotationError(
                f"committed family manifest {observed_manifest} is unreadable; complete "
                "or repair the pending publication before rotating"
            ) from exc
        observed_prior_generation = (
            str(observed_value) if observed_value is not None else None
        )
    # EN-7d: a rotation is exclusive with a running leg -- it holds the same
    # account-global execution flock the campaign driver's worker supervisor
    # holds, bounded by an explicit deadline.
    execution_descriptor = acquire_global_execution_flock(
        execution_lock_path,
        timeout_seconds=execution_lock_timeout_seconds,
        holder_metadata={"holder": "rotate_setup_artifacts", "intent": name, "target": target},
    )
    try:
        return _rotate_intent_locked(
            name,
            registry=registry,
            intent=intent,
            old_identity=old_identity,
            envs_root=envs_root,
            target=target,
            environment_command=environment_command,
            state_root=state_root,
            env_root=env_root,
            minimum_free_bytes=minimum_free_bytes,
            backend=backend,
            campaign_state_roots=campaign_state_roots,
            disk_free=disk_free,
            write_destinations=write_destinations,
            observed_prior_generation=observed_prior_generation,
        )
    finally:
        release_global_execution_flock(execution_descriptor)


def _rotate_intent_locked(
    name: str,
    *,
    registry: Any,
    intent: Any,
    old_identity: str,
    envs_root: Path,
    target: str,
    environment_command: Sequence[str],
    state_root: Path,
    env_root: Path,
    minimum_free_bytes: int,
    backend: Optional[EnvironmentBackend],
    campaign_state_roots: Sequence[Path],
    disk_free: DiskFree,
    write_destinations: Sequence[tuple[str, Path]],
    observed_prior_generation: Optional[str],
) -> tuple[str, str]:
    """Run one rotation's solve-through-republication under the execution flock."""

    if backend is None:
        backend = CommandEnvironmentBackend((*environment_command, "--state-root", str(state_root)))
    environment_file = envs_root / name / "environment.yml"
    solved = backend.solve(environment_file, target)
    parse_exact_lock(solved.lock_bytes)
    declared_inventory = parse_resolved_export(solved.resolved_export_bytes)
    _require_lock_inventory_match(solved.lock_bytes, declared_inventory)
    export_hash = hash_bytes(solved.resolved_export_bytes)

    # Stage the lock outside the committed family: creation reads it, and a
    # failed create/probe must not leave a half-rotated committed family whose
    # accidental new identity would release parks without a proven environment.
    staging = env_root / f".rotation-staging-{name}"
    staging.mkdir(parents=True, exist_ok=True)
    staged_lock = staging / f"{target}.lock"
    _atomic_write(staged_lock, solved.lock_bytes)
    prefix = env_root / f"{name}-{export_hash[7:19]}"
    owned_paths: tuple[Path, ...] = (prefix,)
    overlay_manifest_bytes: Optional[bytes] = None
    try:
        installed_inventory = backend.create(staged_lock, prefix)
        if installed_inventory != declared_inventory:
            raise RotationError(
                "created-prefix package inventory differs from the declared "
                f"resolved export for intent {name!r}"
            )
        if intent.pip_overlay is not None:
            overlay_manifest_bytes = _compile_pip_overlay(
                intent.pip_overlay.declared_path,
                prefix,
                declared_inventory=declared_inventory,
                conda_export_sha256=export_hash,
            )
            staged_manifest = staging / f"{target}.pip-overlay.txt"
            _atomic_write(staged_manifest, overlay_manifest_bytes)
            backend.overlay(prefix, staged_manifest)
        owned_paths = _backend_owned_paths(backend, prefix)
        # Probe proof is validated HERE, not delegated to backend discipline:
        # publication must be impossible without one successful receipt per
        # declared probe, whatever backend produced the observations.
        probe_results = validate_probe_receipts(
            intent.probes, tuple(backend.probe(prefix, intent.probes))
        )
        receipt_bytes = canonical_probe_receipt_bytes(probe_results)
    finally:
        backend.remove(prefix)
        shutil.rmtree(staging, ignore_errors=True)
    residue = tuple(path for path in owned_paths if path.exists() or path.is_symlink())
    if residue:
        listed = ", ".join(str(path) for path in residue)
        raise RotationError(f"rotation teardown left environment residue at {listed}")

    # The publication targets the FLAT committed lock directory. Never derive
    # it from intent.lock.lock_path: under a manifest authority those paths
    # point into the immutable generation directory, which is evidence, not a
    # publication target.
    lock_dir = envs_root / name / "locks"
    receipt_path = lock_dir / f"{target}.probes.json"
    provenance = {
        "schema_version": ROTATION_PROVENANCE_SCHEMA_VERSION,
        "intent": name,
        "target": target,
        "lock_path": _repo_relative(lock_dir / f"{target}.lock", envs_root),
        "lock_sha256": hash_bytes(solved.lock_bytes),
        "resolved_export_path": _repo_relative(lock_dir / f"{target}.resolved.json", envs_root),
        "resolved_export_sha256": export_hash,
        "probe_receipt_path": _repo_relative(receipt_path, envs_root),
        "probe_receipt_sha256": hash_bytes(receipt_bytes),
        "probe_declaration_sha256": probe_declaration_digest(name, intent.probes),
        "spec_path": _repo_relative(environment_file, envs_root),
        "spec_sha256": hash_bytes(environment_file.read_bytes()),
        "probe_contract_path": "menagerie/crawler/envs/probes.yml",
        "probe_contract_sha256": hash_bytes((envs_root / "probes.yml").read_bytes()),
        "solver": {"command": list(environment_command)},
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "on_target_solve": True,
        },
        "clean_create": {
            "validated": True,
            "inventory_matches_resolved_export": True,
        },
        "artifact_hash_validation": {"algorithm": "sha256", "verified": True},
    }
    if overlay_manifest_bytes is not None and intent.pip_overlay is not None:
        provenance["pip_overlay"] = {
            "manifest_path": _repo_relative(lock_dir / f"{target}.pip-overlay.txt", envs_root),
            "manifest_sha256": hash_bytes(overlay_manifest_bytes),
            "spec_path": _repo_relative(intent.pip_overlay.declared_path, envs_root),
            "spec_sha256": hash_bytes(intent.pip_overlay.declared_path.read_bytes()),
            "resolver": "pip install --dry-run --report (prefix interpreter)",
            "installed_offline": True,
            "only_binary": True,
        }

    # Publication is one journaled serial transaction (T1-09): the journal
    # marks the family ``publishing`` from before the first member write until
    # after the last, so a crash can never leave a silently mixed identity
    # that would release parks against an unproven environment.
    members: dict[str, bytes] = {
        f"{target}.lock": solved.lock_bytes,
        f"{target}.resolved.json": solved.resolved_export_bytes,
        f"{target}.resolved.sha256": f"{export_hash}\n".encode(),
        receipt_path.name: receipt_bytes,
        f"{target}.provenance.json": canonical_json_bytes(provenance) + b"\n",
    }
    if overlay_manifest_bytes is not None:
        members[f"{target}.pip-overlay.txt"] = overlay_manifest_bytes

    # Refuse an unchanged identity BEFORE publication: the prospective
    # complete-family identity is computable from the exact member bytes, so
    # a no-op solve never rewrites the committed family at all. (The
    # provenance member is deliberately outside the identity, exactly like
    # the persisted setup-artifact state vector.)
    prospective_identity = stable_hash(
        {
            member_name: hash_bytes(content)
            for member_name, content in members.items()
            if not member_name.endswith(".provenance.json")
        }
    )
    if prospective_identity == old_identity:
        raise RotationError(
            f"rotation did not rotate: intent {name!r} setup-artifact identity is "
            f"still {old_identity} (an unchanged solve regenerated byte-identical "
            "artifacts; the quarantined generation needs an explicit clearance "
            "event instead)"
        )
    # EN-7a: re-measure every write destination immediately before publication --
    # no single-root breach that appeared mid-rotation may half-publish.
    _require_disk_floor(
        write_destinations,
        minimum_free_bytes=minimum_free_bytes,
        disk_free=disk_free,
        stage="pre-publication",
    )
    publish_rotation_family(
        lock_dir, target, members, expected_prior_generation=observed_prior_generation
    )

    rotated = load_environment_registry(envs_root, target=target).intents[name]
    new_identity = _intent_setup_artifact_identity(rotated)
    if new_identity == old_identity:
        raise RotationError(
            f"rotation did not rotate: intent {name!r} setup-artifact identity is "
            f"still {old_identity} (an unchanged solve regenerated byte-identical "
            "artifacts; the quarantined generation needs an explicit clearance "
            "event instead)"
        )
    for campaign_state_root in campaign_state_roots:
        try:
            republish_campaign_solve_cache(
                campaign_state_root,
                rotation_state_root=state_root,
                intent=name,
                target=target,
                lock_bytes=solved.lock_bytes,
                export_bytes=solved.resolved_export_bytes,
                overlay_manifest_bytes=overlay_manifest_bytes,
            )
        except Exception as exc:
            raise RotationError(
                f"intent {name!r} rotated and its committed family is proven, but "
                f"republication into the campaign solve cache at "
                f"{campaign_state_root} failed: {exc}; the stale cache was "
                "invalidated, so the launch cannot serve pre-rotation bytes -- "
                "re-run the republication or wipe "
                f"{campaign_state_root / 'solves' / name / target} before launch"
            ) from exc
    return old_identity, new_identity


def main(argv: Sequence[str] | None = None) -> int:
    """Rotate the committed setup artifacts for one or more intents."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intent",
        action="append",
        required=True,
        help="environment intent to rotate; repeatable",
    )
    parser.add_argument("--envs-root", type=Path, default=DEFAULT_ENVS_ROOT)
    parser.add_argument("--target", default="osx-arm64")
    parser.add_argument(
        "--state-root",
        type=Path,
        help="fresh ABSOLUTE operator scratch root (solve cache + artifact CAS)",
    )
    parser.add_argument(
        "--env-root",
        type=Path,
        help="ABSOLUTE directory receiving the transient verification prefix",
    )
    parser.add_argument(
        "--complete-publication",
        action="store_true",
        help=(
            "complete a committed publication's flat materialization from its "
            "family manifest (roll-forward after a crash between flat member "
            "writes); no solve runs"
        ),
    )
    parser.add_argument("--minimum-free-bytes", type=int, default=DEFAULT_MINIMUM_FREE_BYTES)
    parser.add_argument(
        "--campaign-state-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "campaign environment-operator state root (.crawl-local/"
            "environment-operator) whose solve cache is invalidated and "
            "republished from this rotation; repeatable"
        ),
    )
    parser.add_argument(
        "--purge-state-between-intents",
        action="store_true",
        help="delete the operator scratch root after each intent to bound disk",
    )
    parser.add_argument(
        "--execution-lock-timeout-seconds",
        type=float,
        default=DEFAULT_EXECUTION_LOCK_TIMEOUT_SECONDS,
        help=(
            "bounded wait for the account-global execution flock (a rotation is "
            "exclusive with a running campaign leg; EN-7d)"
        ),
    )
    parser.add_argument(
        "--execution-lock-path",
        type=Path,
        help=(
            "override of the account-global execution flock path (tests and "
            "sandboxed operators only; the default is the one lock the campaign "
            "driver holds during a leg)"
        ),
    )
    args = parser.parse_args(argv)
    if any(not root.is_absolute() for root in args.campaign_state_root):
        parser.error("--campaign-state-root must be absolute")
    if args.complete_publication:
        failures = 0
        for name in args.intent:
            lock_dir = args.envs_root / name / "locks"
            try:
                report = complete_family_publication(lock_dir, args.target)
            except Exception as exc:  # noqa: BLE001 -- operator-facing per-intent report
                failures += 1
                print(json.dumps({"intent": name, "completed": False, "error": str(exc)}))
            else:
                print(json.dumps({"intent": name, **report}))
        return 1 if failures else 0
    if args.state_root is None or args.env_root is None:
        parser.error("--state-root and --env-root are required for a rotation")
    if not args.state_root.is_absolute() or not args.env_root.is_absolute():
        parser.error("--state-root and --env-root must be absolute")
    environment_command = _default_environment_command()
    failures = 0
    for name in args.intent:
        state_root = args.state_root / name
        state_root.mkdir(parents=True, exist_ok=True)
        try:
            old_identity, new_identity = rotate_intent(
                name,
                envs_root=args.envs_root,
                target=args.target,
                environment_command=environment_command,
                state_root=state_root,
                env_root=args.env_root,
                minimum_free_bytes=args.minimum_free_bytes,
                campaign_state_roots=tuple(args.campaign_state_root),
                execution_lock_path=args.execution_lock_path,
                execution_lock_timeout_seconds=args.execution_lock_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 -- operator-facing per-intent report
            failures += 1
            print(json.dumps({"intent": name, "rotated": False, "error": str(exc)}))
        else:
            print(
                json.dumps(
                    {
                        "intent": name,
                        "rotated": True,
                        "old_identity": old_identity,
                        "new_identity": new_identity,
                    }
                )
            )
        finally:
            if args.purge_state_between_intents:
                shutil.rmtree(state_root, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
