"""Tests for the ledger join added to build_catalog.py (Slice A compat fix).

Run directly: pytest site/scripts/test_build_catalog_ledger.py
(not part of the root ``tests/`` suite collected by pyproject's testpaths).
"""

from __future__ import annotations

import json
from pathlib import Path

import build_catalog as bc


def _write_ledger(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_ledger_keeps_only_passed_rows(tmp_path: Path) -> None:
    ledger_path = tmp_path / "verification.jsonl"
    _write_ledger(
        ledger_path,
        [
            {
                "stable_id": "m1",
                "recipe_revision_sha256": "aaa",
                "status": "passed",
                "forward_pass": 1,
                "metadata_ok": 1,
                "n_ops": 42,
                "torchlens_version": "2.34.1",
            },
            {
                "stable_id": "m2",
                "recipe_revision_sha256": "bbb",
                "status": "failed",
                "forward_pass": 0,
                "metadata_ok": None,
                "n_ops": None,
                "torchlens_version": "2.34.1",
            },
        ],
    )
    entries = bc.load_ledger(ledger_path)
    assert set(entries) == {("m1", "aaa")}
    assert entries[("m1", "aaa")]["n_ops"] == 42


def test_load_ledger_none_path_is_empty() -> None:
    assert bc.load_ledger(None) == {}


def test_build_model_uses_ledger_when_row_matches(tmp_path: Path) -> None:
    gallery = tmp_path / "gallery" / "cat"
    gallery.mkdir(parents=True)
    svg = gallery / "m1.svg"
    svg.write_text("<svg></svg>", encoding="utf-8")
    out_root = tmp_path / "out"
    (out_root / "assets").mkdir(parents=True)
    (out_root / "models").mkdir(parents=True)

    row = bc.ManifestRow(
        name="m1",
        model_id="1",
        n_nodes=10,
        render_path=svg,
        graph_shape_hash="shape1",
        category="cat",
        stable_id="m1",
        recipe_revision_sha256="aaa",
    )
    ledger = {
        ("m1", "aaa"): {
            "n_ops": 42,
            "torchlens_version": "2.34.1",
        }
    }
    built = bc.build_model(row, "m1", "m1-tok", out_root, ledger)
    assert built.record["forward_validated"] is True
    assert built.record["torchlens_version"] == "2.34.1"
    assert built.record["n_ops"] == 42


def test_build_model_ledger_n_ops_zero_is_preserved(tmp_path: Path) -> None:
    """A ledger n_ops of 0 must not fall back to the render-only node count.

    ``build_model`` used to pick the ledger's n_ops with ``ledger_row.get(...)
    or row.n_nodes``, so a real but falsy ``0`` would be silently replaced by
    the preview render's node count instead of the ledger's honest value.
    """
    gallery = tmp_path / "gallery" / "cat"
    gallery.mkdir(parents=True)
    svg = gallery / "m3.svg"
    svg.write_text("<svg></svg>", encoding="utf-8")
    out_root = tmp_path / "out"
    (out_root / "assets").mkdir(parents=True)
    (out_root / "models").mkdir(parents=True)

    row = bc.ManifestRow(
        name="m3",
        model_id="3",
        n_nodes=10,
        render_path=svg,
        graph_shape_hash="shape3",
        category="cat",
        stable_id="m3",
        recipe_revision_sha256="ccc",
    )
    ledger = {
        ("m3", "ccc"): {
            "n_ops": 0,
            "torchlens_version": "2.34.1",
        }
    }
    built = bc.build_model(row, "m3", "m3-tok", out_root, ledger)
    assert built.record["forward_validated"] is True
    assert built.record["n_ops"] == 0


def test_build_model_without_ledger_match_stays_unvalidated(tmp_path: Path) -> None:
    gallery = tmp_path / "gallery" / "cat"
    gallery.mkdir(parents=True)
    svg = gallery / "m2.svg"
    svg.write_text("<svg></svg>", encoding="utf-8")
    out_root = tmp_path / "out"
    (out_root / "assets").mkdir(parents=True)
    (out_root / "models").mkdir(parents=True)

    row = bc.ManifestRow(
        name="m2",
        model_id="2",
        n_nodes=11,
        render_path=svg,
        graph_shape_hash="shape2",
        category="cat",
        stable_id="m2",
        recipe_revision_sha256="bbb",
    )
    built = bc.build_model(row, "m2", "m2-tok", out_root, {})
    assert built.record["forward_validated"] is False
    assert built.record["torchlens_version"] == "unknown"
    assert built.record["n_ops"] == 11
