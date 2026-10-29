"""Small shared repository-inspection helpers for crawler acceptance tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Container, Mapping

from menagerie.crawler.identity import hash_bytes

#: Directory that a RELEASE solve output may ship from. Rotation families ship from the
#: per-intent ``envs/<intent>/locks`` directories instead (see the rotation schema below);
#: a ``.lock`` or resolved export anywhere else under ``envs/`` is a stray runtime or
#: hand-authored artifact by location alone.
_LOCK_FAMILY_ROOT = ("menagerie", "crawler", "envs", "locks")

#: File kinds that are literally solver output. ``.provenance.json``/``.probes.json`` are
#: attestation members, not solve outputs, so they are checked as evidence rather than
#: scanned as suspects. The pinned pip-overlay manifest is a solve output too: it is the
#: identity-bearing product of the overlay resolution and must never ship unattested.
_SOLVE_OUTPUT_SUFFIXES = (".lock", ".resolved.json", ".resolved.sha256", ".pip-overlay.txt")

_PROVENANCE_SCHEMA = "menagerie.crawler.release-lock-provenance.v1"

#: Per-intent setup rotation families (tools/rotate_setup_artifacts.py): solved ON the
#: target host through the production environment operator, probe receipts from a really
#: created prefix, published only after every probe passes.
_ROTATION_PROVENANCE_SCHEMA = "menagerie.crawler.rotation-lock-provenance.v1"


def repository_root() -> Path:
    """Return the checked-out repository root containing this acceptance package."""

    return Path(__file__).resolve().parents[2]


def tracked_paths(repo_root: Path) -> tuple[Path, ...]:
    """Return every Git-tracked path without consulting ignored runtime files."""

    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return tuple(Path(value.decode("utf-8")) for value in completed.stdout.split(b"\0") if value)


def _solve_output_family(path: Path) -> tuple[Path, str] | None:
    """Return the lock-family directory and stem of one tracked solve output.

    Parameters
    ----------
    path:
        Repository-relative tracked path.

    Returns
    -------
    tuple[pathlib.Path, str] | None
        Family directory and stem, or ``None`` when the path is not a solve output.
    """

    if path.parts[:3] != ("menagerie", "crawler", "envs"):
        return None
    for suffix in _SOLVE_OUTPUT_SUFFIXES:
        if path.name.endswith(suffix):
            return path.parent, path.name[: -len(suffix)]
    return None


def _declared_digest_matches(
    repo_root: Path,
    tracked: Container[Path],
    provenance: Mapping[str, Any],
    path_key: str,
    digest_key: str,
    expected_path: Path | None = None,
) -> bool:
    """Check one provenance row names a tracked file whose bytes hash to its digest.

    Parameters
    ----------
    repo_root, tracked:
        Checked-out root and the tracked-path membership set.
    provenance:
        Decoded release-lock provenance record.
    path_key, digest_key:
        Provenance keys naming the attested file and its declared SHA-256.
    expected_path:
        Repository-relative path the provenance must name, when the family fixes it.

    Returns
    -------
    bool
        Whether the named file is tracked and hashes to exactly the declared digest.
    """

    declared = provenance.get(path_key)
    if not isinstance(declared, str) or not declared:
        return False
    relative = Path(declared)
    if expected_path is not None and relative != expected_path:
        return False
    if relative not in tracked:
        return False
    absolute = repo_root / relative
    if not absolute.is_file():
        return False
    return provenance.get(digest_key) == hash_bytes(absolute.read_bytes())


def _clean_create_is_attested(provenance: Mapping[str, Any]) -> bool:
    """Check the provenance claims a real clean-create validation of the lock.

    Parameters
    ----------
    provenance:
        Decoded release-lock provenance record.

    Returns
    -------
    bool
        Whether the record claims either a validated clean create or the exact host
        that performed one. A record with no clean-create claim attests nothing.
    """

    clean_create = provenance.get("clean_create")
    if not isinstance(clean_create, dict):
        return False
    if clean_create.get("validated") is True:
        return True
    host = clean_create.get("validation_host")
    return isinstance(host, str) and bool(host)


def _release_family_is_attested(repo_root: Path, tracked: Container[Path], family: Path) -> bool:
    """Check one committed lock family carries complete, byte-binding provenance.

    Parameters
    ----------
    repo_root, tracked:
        Checked-out root and the tracked-path membership set.
    family:
        Repository-relative lock path whose sibling family members are checked.

    Returns
    -------
    bool
        Whether every family member ships and the committed provenance binds each of
        them, the release spec, and the probe contract by exact SHA-256.
    """

    export = family.with_suffix(".resolved.json")
    export_hash = family.with_suffix(".resolved.sha256")
    provenance_path = family.with_suffix(".provenance.json")
    probes = family.with_suffix(".probes.json")
    members = (family, export, export_hash, provenance_path, probes)
    if any(member not in tracked or not (repo_root / member).is_file() for member in members):
        return False
    try:
        provenance = json.loads((repo_root / provenance_path).read_bytes())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(provenance, dict):
        return False
    if provenance.get("schema_version") == _ROTATION_PROVENANCE_SCHEMA:
        return _rotation_family_is_attested(
            repo_root,
            tracked,
            family,
            provenance,
            export=export,
            export_hash=export_hash,
            probes=probes,
        )
    if family.parent.parts != _LOCK_FAMILY_ROOT:
        return False
    if provenance.get("schema_version") != _PROVENANCE_SCHEMA:
        return False
    declared_export_hash = (repo_root / export_hash).read_text(encoding="utf-8").strip()
    if declared_export_hash != hash_bytes((repo_root / export).read_bytes()):
        return False
    if provenance.get("resolved_export_sha256") != declared_export_hash:
        return False
    rows = (
        ("lock_path", "lock_sha256", family),
        ("resolved_export_path", "resolved_export_sha256", export),
        ("probe_receipt_path", "probe_receipt_sha256", probes),
        # The spec and probe contract are the solve inputs; binding them by digest is what
        # makes the committed output traceable to a solve rather than to an author.
        ("spec_path", "spec_sha256", None),
        ("probe_contract_path", "probe_contract_sha256", None),
    )
    if not all(
        _declared_digest_matches(repo_root, tracked, provenance, path_key, digest_key, expected)
        for path_key, digest_key, expected in rows
    ):
        return False
    return _clean_create_is_attested(provenance)


def _rotation_family_is_attested(
    repo_root: Path,
    tracked: Container[Path],
    family: Path,
    provenance: Mapping[str, Any],
    *,
    export: Path,
    export_hash: Path,
    probes: Path,
) -> bool:
    """Check one per-intent ROTATION family carries complete, byte-binding provenance.

    Rotation families live in ``envs/<intent>/locks`` and are produced by
    ``tools/rotate_setup_artifacts.py`` on the target host through the production
    environment operator. The binding obligations mirror the release families --
    lock, resolved export, probe receipt, and the solve input by exact SHA-256 --
    with the intent's own ``environment.yml`` as the spec (a post-rotation spec
    edit therefore reads as unattested and forces a re-rotation). A declared pip
    overlay adds its pinned manifest as one more bound member; the manifest may
    never ship without the declaration, nor the declaration without the manifest.

    The shared ``probes.yml`` is deliberately NOT hash-bound here: it spans every
    intent, so a sibling intent's probe edit would false-invalidate this family
    while its OWN contract -- which the committed receipt is separately validated
    against in the environment test suite -- is unchanged.
    """

    parts = family.parent.parts
    if (
        len(parts) != 5
        or parts[:3] != ("menagerie", "crawler", "envs")
        or parts[4] != "locks"
        or family.name != f"{provenance.get('target')}.lock"
    ):
        return False
    intent_name = parts[3]
    if provenance.get("intent") != intent_name:
        return False
    spec = Path("menagerie", "crawler", "envs", intent_name, "environment.yml")
    rows = [
        ("lock_path", "lock_sha256", family),
        ("resolved_export_path", "resolved_export_sha256", export),
        ("probe_receipt_path", "probe_receipt_sha256", probes),
        ("spec_path", "spec_sha256", spec),
    ]
    declared_export_hash = (repo_root / export_hash).read_text(encoding="utf-8").strip()
    if declared_export_hash != hash_bytes((repo_root / export).read_bytes()):
        return False
    if provenance.get("resolved_export_sha256") != declared_export_hash:
        return False
    if not all(
        _declared_digest_matches(repo_root, tracked, provenance, path_key, digest_key, expected)
        for path_key, digest_key, expected in rows
    ):
        return False
    overlay_spec = Path("menagerie", "crawler", "envs", intent_name, "pip-overlay.in")
    overlay_manifest = family.with_suffix(".pip-overlay.txt")
    overlay = provenance.get("pip_overlay")
    if overlay_spec in tracked:
        if not isinstance(overlay, Mapping):
            return False
        if not all(
            _declared_digest_matches(repo_root, tracked, overlay, path_key, digest_key, expected)
            for path_key, digest_key, expected in (
                ("manifest_path", "manifest_sha256", overlay_manifest),
                ("spec_path", "spec_sha256", overlay_spec),
            )
        ):
            return False
    elif overlay is not None or overlay_manifest in tracked:
        return False
    host = provenance.get("host")
    if not isinstance(host, Mapping) or host.get("on_target_solve") is not True:
        return False
    return _clean_create_is_attested(provenance)


def fabricated_crawler_locks(repo_root: Path) -> tuple[Path, ...]:
    """Return tracked crawler solve outputs that no committed provenance attests.

    A solve output is *fabricated* when nothing proves a solver produced it: it was
    hand-authored, guessed, or copied out of a runtime solve. The release proof, by
    contrast, must consume fixed bytes -- a lock re-solved at proof time attests nothing
    about what shipped -- so the genuine target-solved release families are committed on
    purpose, each with a provenance record binding its lock, resolved export, probe
    receipt, release spec, and probe contract by exact SHA-256.

    The boundary is therefore *which* solve outputs may ship, not whether any may. A
    tracked solve output passes only as a complete member of such an attested family
    living in ``envs/locks``; a stray lock, an orphaned or incomplete family, a resolved
    export whose declared hash does not match its bytes, and any provenance that does not
    bind the committed bytes are all returned here.

    Parameters
    ----------
    repo_root:
        Checked-out repository root.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Every unattested tracked solve output, in sorted repository-relative order.
    """

    tracked = frozenset(tracked_paths(repo_root))
    attested: dict[Path, bool] = {}
    fabricated = []
    for path in sorted(tracked):
        located = _solve_output_family(path)
        if located is None:
            continue
        directory, stem = located
        family = directory / f"{stem}.lock"
        if family not in attested:
            attested[family] = _release_family_is_attested(repo_root, tracked, family)
        if not attested[family]:
            fabricated.append(path)
    return tuple(fabricated)
