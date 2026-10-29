"""Tests for the verification ledger JSONL export (Slice A compat fix).

The June validation pipeline's verification ledger is an append-only SQLite
database (``menagerie.ledger``) that is gitignored and machine-local. This
module provides a small, read-only export of the ledger's honest current rows
(``current_verification_real``) to JSONL, so a run's results can be committed
or handed to ``site/scripts/build_catalog.py --ledger`` without shipping
SQLite. It changes nothing about how ``validate_menagerie.py`` writes rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.export_ledger_jsonl import dump_ledger_jsonl
from menagerie.ledger import VerificationRun, append_verification_run, connect


def _run(stable_id: str, status: str, forward_pass: int | None, metadata_ok: int | None, **overrides: object) -> VerificationRun:
    base = dict(
        stable_id=stable_id,
        recipe_revision_sha256="rev-" + stable_id,
        name=stable_id,
        zoo="timm",
        variant="",
        scope="forward",
        status=status,
        forward_pass=forward_pass,
        backward_pass=None,
        backward_na_reason="scope=forward",
        metadata_ok=metadata_ok,
        n_ops=42 if status == "passed" else None,
        graph_shape_hash="shape-" + stable_id if status == "passed" else None,
        svg_sha256=None,
        torchlens_version="2.34.1",
        torch_version="2.14.1",
        python_version="3.13.12",
        device_requested="cpu",
        device_actual="cpu",
        env_hash="base",
        lock_hash="base-no-lock",
        torchlens_source_hash="src-abc",
        input_scale=1.0,
        runner_host="worker-c",
        started_at="2026-10-01T00:00:00Z",
        finished_at="2026-10-01T00:00:01Z",
        duration_sec=1.0,
    )
    base.update(overrides)
    return VerificationRun(**base)


def test_dump_ledger_jsonl_writes_one_line_per_current_row(tmp_path: Path) -> None:
    db_path = tmp_path / "verification.db"
    conn = connect(db_path)
    append_verification_run(conn, _run("m1", "passed", 1, 1))
    append_verification_run(conn, _run("m2", "failed", 0, None))
    conn.close()

    out_path = tmp_path / "verification.jsonl"
    written = dump_ledger_jsonl(db_path, out_path)

    assert written == 2
    lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    by_id = {row["stable_id"]: row for row in lines}
    assert by_id["m1"]["status"] == "passed"
    assert by_id["m1"]["forward_pass"] == 1
    assert by_id["m1"]["metadata_ok"] == 1
    assert by_id["m1"]["torchlens_version"] == "2.34.1"
    assert by_id["m2"]["status"] == "failed"


def test_dump_ledger_jsonl_keeps_only_latest_row_per_stable_id(tmp_path: Path) -> None:
    db_path = tmp_path / "verification.db"
    conn = connect(db_path)
    append_verification_run(conn, _run("m1", "failed", 0, None, finished_at="2026-10-01T00:00:00Z"))
    append_verification_run(conn, _run("m1", "passed", 1, 1, finished_at="2026-10-01T00:05:00Z"))
    conn.close()

    out_path = tmp_path / "verification.jsonl"
    written = dump_ledger_jsonl(db_path, out_path)

    assert written == 1
    (row,) = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert row["status"] == "passed"


def test_dump_ledger_jsonl_missing_db_raises_file_not_found(tmp_path: Path) -> None:
    """A missing or mistyped --db path must error, not silently export zero rows.

    ``menagerie.ledger.connect`` initializes a fresh, empty database at any
    path it is given, so without this check a typo'd ``db_path`` would
    silently produce a legitimate-looking empty export instead of a clear
    failure.
    """
    missing_db = tmp_path / "does-not-exist.db"
    out_path = tmp_path / "verification.jsonl"

    with pytest.raises(FileNotFoundError):
        dump_ledger_jsonl(missing_db, out_path)

    assert not missing_db.exists()
    assert not out_path.exists()
