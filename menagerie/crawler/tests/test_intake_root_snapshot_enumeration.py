"""Canonical intake-root enumeration: a wrapper directory is not a snapshot claim.

Regression suite for the live pilot blocker found rehearsing MASTER_DOCKET OPS-4 on a
copy of the campaign records: OPS-2 copied the legacy ``intake_head50/`` tree into
``records/intake/`` with its snapshot one level DOWN
(``records/intake/intake_head50/intake-73b7f47d76c13df8048a/``), so
``records/intake/intake_head50`` carries no ``manifest.json`` of its own. Two callers
then disagreed about what the intake root contains:

* ``build_canonical_requeue_grant`` enumerated ``iterdir()`` and called
  ``load_intake_snapshot`` on EVERY child directory, so the manifest-less wrapper
  raised ``canonical intake snapshot is invalid`` -- bricking every operator grant and
  void on the campaign, OPS-4 included; and
* ``_derive_candidate_paths`` globbed ``*/manifest.json``, so it silently SKIPPED the
  nested snapshot it should have verified.

The contract these tests pin: a directory is a snapshot CLAIM exactly when it carries a
``manifest.json``, every claim is verified, and a claim that does not load still raises.
Skipping a manifest-less wrapper weakens nothing -- it never claimed to be a snapshot --
while the nested snapshot beneath it becomes visible to BOTH callers.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from menagerie.crawler.checkpoint import (
    build_canonical_requeue_grant,
    canonical_intake_snapshot_roots,
    canonical_requeue_grants_path,
)
from menagerie.crawler.intake import create_intake_snapshot
from menagerie.crawler.reducer import default_ledger_paths

pytestmark = pytest.mark.smoke


def _snapshot(tmp_path: Path, output_root: Path, stable_ids: tuple[str, ...]) -> Path:
    """Materialize one real intake snapshot below ``output_root``.

    Parameters
    ----------
    tmp_path:
        Test-owned temporary directory for the source catalogs.
    output_root:
        Directory the immutable snapshot is written under.
    stable_ids:
        Model identities the snapshot should carry.

    Returns
    -------
    tuple[pathlib.Path, tuple[str, ...]]
        Snapshot root directory and the stable IDs intake actually assigned (they are
        content-derived, never the catalog's own ``stable_id`` field).
    """

    sources = tmp_path / f"sources-{output_root.name}-{len(stable_ids)}-{stable_ids[0]}"
    sources.mkdir(parents=True, exist_ok=True)
    master = sources / "master_catalog.jsonl"
    master.write_text(
        "".join(
            json.dumps(
                {
                    "name": stable_id,
                    "zoo": "test",
                    "constructor_call": "torch.nn.Identity()",
                    "input_shape": [1, 1],
                    "input_dtype": "float32",
                    "family": "test",
                    "domain": "test",
                    "era": "test",
                    "notes": "",
                    "stable_id": stable_id,
                }
            )
            + "\n"
            for stable_id in stable_ids
        )
    )
    deferred = sources / "deferred.jsonl"
    deferred.write_text("")
    snapshot = create_intake_snapshot(
        master_catalog=master,
        deferred_catalog=deferred,
        output_root=output_root,
    )
    return Path(snapshot.root), tuple(item.stable_id for item in snapshot.items)


def _records_root(tmp_path: Path) -> Path:
    """Return a records root carrying an intake directory and a grant ledger."""

    records_root = tmp_path / "records"
    (records_root / "intake").mkdir(parents=True, exist_ok=True)
    (records_root / "operational").mkdir(parents=True, exist_ok=True)
    (records_root / "models").mkdir(parents=True, exist_ok=True)
    return records_root


def _tamper(snapshot_root: Path) -> None:
    """Break one snapshot's byte-level integrity so its loader must refuse.

    The manifest pins each source file's digest, so appending to a source is the
    smallest tamper ``load_intake_snapshot`` is contractually required to catch.
    """

    source = snapshot_root / "sources" / "master_catalog.jsonl"
    source.write_bytes(source.read_bytes() + b'{"tampered": true}\n')


def test_wrapper_directory_without_a_manifest_is_not_a_snapshot_claim(tmp_path: Path) -> None:
    """A manifest-less wrapper is skipped; the snapshot beneath it is still enumerated."""

    records_root = _records_root(tmp_path)
    intake_root = records_root / "intake"
    canonical, _ = _snapshot(tmp_path, intake_root, ("m1", "m2"))

    # The OPS-2 shape: a legacy snapshot copied WITH its id-named directory inside a
    # human-named wrapper, so the wrapper itself carries no manifest.
    wrapper = intake_root / "intake_head50"
    wrapper.mkdir()
    nested, _ = _snapshot(tmp_path, wrapper, ("m3",))
    assert not (wrapper / "manifest.json").exists()

    roots = canonical_intake_snapshot_roots(intake_root)
    assert canonical in roots
    assert nested in roots
    assert wrapper not in roots


def test_requeue_grant_survives_a_legacy_wrapper_and_sees_its_nested_members(
    tmp_path: Path,
) -> None:
    """OPS-4's exact shape: grant a model that lives ONLY in the nested snapshot."""

    records_root = _records_root(tmp_path)
    intake_root = records_root / "intake"
    _snapshot(tmp_path, intake_root, ("m1", "m2"))
    wrapper = intake_root / "intake_head50"
    wrapper.mkdir()
    _, nested_ids = _snapshot(tmp_path, wrapper, ("m2336",))
    nested_only = nested_ids[0]

    grants_path = canonical_requeue_grants_path(default_ledger_paths(records_root).models)
    grant = build_canonical_requeue_grant(
        grants_path,
        stable_id=nested_only,
        stage="runner",
        reason="OPS-4 rehearsal: pre-issue the stage-matched grant",
        attempts=1,
        granted_by="test",
    )
    assert grant["stable_id"] == nested_only
    assert grant["stage"] == "runner"


def test_a_model_absent_from_every_snapshot_still_refuses(tmp_path: Path) -> None:
    """The membership tripwire is unchanged: an out-of-intake grant still raises."""

    records_root = _records_root(tmp_path)
    intake_root = records_root / "intake"
    _snapshot(tmp_path, intake_root, ("m1", "m2"))
    wrapper = intake_root / "intake_head50"
    wrapper.mkdir()
    _snapshot(tmp_path, wrapper, ("m2336",))

    grants_path = canonical_requeue_grants_path(default_ledger_paths(records_root).models)
    with pytest.raises(ValueError, match="absent from canonical intake snapshots"):
        build_canonical_requeue_grant(
            grants_path,
            stable_id="m_not_in_any_snapshot",
            stage="runner",
            reason="must refuse",
            attempts=1,
            granted_by="test",
        )


def test_a_tampered_snapshot_claim_still_raises(tmp_path: Path) -> None:
    """A directory that CLAIMS to be a snapshot and does not load stays fail-closed."""

    records_root = _records_root(tmp_path)
    intake_root = records_root / "intake"
    canonical, ids = _snapshot(tmp_path, intake_root, ("m1", "m2"))
    _tamper(canonical)

    grants_path = canonical_requeue_grants_path(default_ledger_paths(records_root).models)
    with pytest.raises(ValueError, match="canonical intake snapshot is invalid"):
        build_canonical_requeue_grant(
            grants_path,
            stable_id=ids[0],
            stage="runner",
            reason="tampered snapshot must refuse",
            attempts=1,
            granted_by="test",
        )


def test_a_tampered_NESTED_snapshot_claim_also_raises(tmp_path: Path) -> None:
    """The nested claim is verified too -- the glob site used to skip it silently."""

    records_root = _records_root(tmp_path)
    intake_root = records_root / "intake"
    _, ids = _snapshot(tmp_path, intake_root, ("m1", "m2"))
    wrapper = intake_root / "intake_head50"
    wrapper.mkdir()
    nested, _ = _snapshot(tmp_path, wrapper, ("m2336",))
    _tamper(nested)
    manifest = json.loads((nested / "manifest.json").read_text())
    manifest["item_count"] = 999
    (nested / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")

    grants_path = canonical_requeue_grants_path(default_ledger_paths(records_root).models)
    with pytest.raises(ValueError, match="canonical intake snapshot is invalid"):
        build_canonical_requeue_grant(
            grants_path,
            stable_id="m1",
            stage="runner",
            reason="tampered nested snapshot must refuse",
            attempts=1,
            granted_by="test",
        )


def test_enumeration_ignores_files_and_empty_directories(tmp_path: Path) -> None:
    """`registered-rungs.jsonl` and an empty directory are neither snapshots nor errors."""

    records_root = _records_root(tmp_path)
    intake_root = records_root / "intake"
    canonical, _ = _snapshot(tmp_path, intake_root, ("m1",))
    (intake_root / "registered-rungs.jsonl").write_text("")
    (intake_root / "empty").mkdir()

    assert canonical_intake_snapshot_roots(intake_root) == (canonical,)


def test_enumeration_is_bounded_and_does_not_descend_into_snapshot_internals(
    tmp_path: Path,
) -> None:
    """A snapshot's own `sources/` subtree is never re-enumerated as a snapshot."""

    records_root = _records_root(tmp_path)
    intake_root = records_root / "intake"
    canonical, _ = _snapshot(tmp_path, intake_root, ("m1",))
    assert (canonical / "sources").is_dir()

    roots = canonical_intake_snapshot_roots(intake_root)
    assert roots == (canonical,)
    assert all("sources" not in root.parts for root in roots)


def test_missing_intake_directory_enumerates_nothing(tmp_path: Path) -> None:
    """An absent intake root is empty, not an error."""

    assert canonical_intake_snapshot_roots(tmp_path / "nope") == ()


def test_nested_snapshot_shadowing_a_top_level_copy_is_reported_once(tmp_path: Path) -> None:
    """The same snapshot present at both levels contributes one root, not two.

    The pilot really does hold `records/intake/intake_step100/` and
    `records/intake/intake-b72d4f2ff0db9705d336/` with byte-identical manifests, so the
    duplicate must be collapsed by identity rather than counted twice.
    """

    records_root = _records_root(tmp_path)
    intake_root = records_root / "intake"
    canonical, _ = _snapshot(tmp_path, intake_root, ("m1", "m2"))
    duplicate = intake_root / "intake_step100"
    shutil.copytree(canonical, duplicate)

    roots = canonical_intake_snapshot_roots(intake_root)
    manifests = {
        json.loads((root / "manifest.json").read_text())["snapshot_sha256"] for root in roots
    }
    assert len(manifests) == 1
    assert len(roots) == 2  # both claims exist and both are verified
