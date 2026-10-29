"""Tripwire tests for the RC-2 rotation tooling.

The old operator registration surface is gone with the ceremony: there is no
``declare`` command and no way to type a positive class. What remains is the
seed importer (the one lawful source of ``declared`` rows), the predicate
replay probe (rows measured on the root they govern), the closure
acknowledgement the smoke corpus enforces, golden-replay classification for
the checker prompt, and disclosure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.rotation_registry import (
    append_rotation_registration,
    build_rotation_registration,
    rotation_registry_path,
)
from menagerie.crawler.rotations import CODE_AXES, load_rotations
from menagerie.crawler.tools.register_authority_rotation import (
    ACKNOWLEDGE_COMMAND,
    _live_identities,
    load_kernel_closure_declaration,
    main,
    measured_rows_from_probe,
    undeclared_code_axes,
    write_kernel_closure_declaration,
)

pytestmark = pytest.mark.smoke


def test_print_live_covers_every_registrable_axis(capsys) -> None:
    from menagerie.crawler.rotation_registry import ROTATION_AXES

    assert main(["print-live"]) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert set(payload) == set(ROTATION_AXES)
    assert all(value.startswith("sha256:") for value in payload.values())


def test_there_is_no_operator_declaration_command() -> None:
    with pytest.raises(SystemExit):
        main(["--records-root", "/nonexistent", "declare", "--axis", "award_closure_identity"])


def test_seed_imports_the_frozen_registry_once_and_list_discloses_it(tmp_path: Path, capsys) -> None:
    registry = rotation_registry_path(tmp_path)
    a, b = ("sha256:" + ch * 64 for ch in "ab")
    append_rotation_registration(
        registry,
        build_rotation_registration(
            axis="award_closure_identity",
            from_identity=a,
            to_identity=b,
            rotation_class="soundness",
            rationale="fixture",
            registered_by="operator-test",
            created_at="2026-08-20T00:00:00Z",
        ),
    )
    assert main(["--records-root", str(tmp_path), "seed", "--tree-sha", "deadbeef"]) == 0
    assert json.loads(capsys.readouterr().out.strip()) == {
        "registry_rows": 1, "appended": 1, "already_present": 0,
    }
    assert main(["--records-root", str(tmp_path), "seed", "--tree-sha", "deadbeef"]) == 0
    assert json.loads(capsys.readouterr().out.strip())["appended"] == 0
    assert main(["--records-root", str(tmp_path), "list"]) == 0
    listing = json.loads(capsys.readouterr().out.strip())
    assert listing["edge_classes"] == [
        {"axis": "award_closure_identity", "from": a, "to": b, "class": "soundness"}
    ]
    (row,) = listing["rotations"]
    assert row["measured_by"] == "declared" and row["chain_sha256"]


def test_measured_rows_come_from_a_non_empty_probe_only() -> None:
    from menagerie.crawler.rotations import RotationLedgerError

    live = _live_identities()
    moved = {"award_closure_identity": ["sha256:" + "1" * 64]}
    empty = {"heads": 0, "awards": 0, "flips": [], "class": None}
    with pytest.raises(RotationLedgerError, match="measured no class"):
        measured_rows_from_probe(empty, moved=moved, live=live, tree_sha="t", reason="r", at="2026-09-01T00:00:00Z")
    report = {"heads": 2, "awards": 1, "flips": [{"stable_id": "m_x"}], "class": "soundness"}
    (row,) = measured_rows_from_probe(report, moved=moved, live=live, tree_sha="t", reason="r", at="2026-09-01T00:00:00Z")
    assert row["class"] == "soundness" and row["measured_by"] == "predicate_replay"
    assert row["to"] == live["award_closure_identity"]
    assert "1 flips" in row["reason"]


def test_acknowledge_closure_writes_only_moved_axes_and_the_tripwire_reads_it(tmp_path: Path) -> None:
    declaration = tmp_path / "kernel_closure_declaration.json"
    live = _live_identities()
    stale_live = dict(live)
    stale_live["award_closure_identity"] = "sha256:" + "0" * 64
    write_kernel_closure_declaration(declaration, live=stale_live, tree_sha="old", reason="first", at="2026-09-01T00:00:00Z")
    moved = undeclared_code_axes(load_kernel_closure_declaration(declaration), live)
    assert set(moved) == {"award_closure_identity"}
    assert moved["award_closure_identity"] == ("sha256:" + "0" * 64, live["award_closure_identity"])
    probe_report = {"heads": 3, "awards": 1, "flips": [], "changed_input_stale": [], "class": "refinement", "records_root": "/r", "tree_sha": "new"}
    document = write_kernel_closure_declaration(
        declaration, live=live, tree_sha="new", reason="second", probe_report=probe_report, at="2026-09-02T00:00:00Z"
    )
    assert set(document["axes"]) == set(CODE_AXES)
    assert document["axes"]["award_closure_identity"]["tree_sha"] == "new"
    assert document["axes"]["award_closure_identity"]["measurement"]["class"] == "refinement"
    assert document["axes"]["terminal_rule_identity"]["tree_sha"] == "old"
    assert undeclared_code_axes(load_kernel_closure_declaration(declaration), live) == {}
    assert "acknowledge-closure" in ACKNOWLEDGE_COMMAND


def test_measure_golden_refuses_author_axes(tmp_path: Path, capsys) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"goldens": [{"name": "g", "kind": "metadata_batch", "rotation": {"ran": True, "projection": {}}, "same_epoch": {"projections": [{}]}}]}))
    code = main([
        "--records-root", str(tmp_path), "measure-golden", "--report", str(report),
        "--axis", "author_prompt_identity", "--from-identity", "sha256:" + "1" * 64,
    ])
    assert code == 1
    assert "does not exercise author_prompt_identity" in capsys.readouterr().err
    assert not (tmp_path / "control" / "rotations.jsonl").exists()
    assert load_rotations(tmp_path) == ()
