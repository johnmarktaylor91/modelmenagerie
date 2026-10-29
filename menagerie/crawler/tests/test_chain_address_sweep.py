"""The pre-cutover sweep is a real measurement, not a green rubber stamp.

The sweep's clean report is a relaunch precondition (it converts "no on-disk row was
stamped by an older rule" from an assumption into a checked fact), so the sweep
itself needs to be provably non-vacuous: it must report CLEAN on honest ledgers and
report the exact row on every divergence class it exists to find.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.identity import canonical_json_bytes, stable_hash, stamp_row_digest
from menagerie.crawler.ledger_chain import link_row
from menagerie.crawler.tools.chain_address_sweep import (
    chain_bearing_ledgers,
    ladder_content_digest,
    main,
    sweep_ledger,
    sweep_records_root,
)

pytestmark = pytest.mark.smoke

_SCOPE = "paid.sessions"


def _honest_ledger(path: Path, rows: int = 3) -> list[dict[str, object]]:
    """Write one honestly stamped, honestly linked ledger and return its rows."""

    linked: list[dict[str, object]] = []
    payload = b""
    for index in range(rows):
        row = stamp_row_digest(
            {
                "schema_version": "menagerie.crawler.example.v1",
                "event": "example",
                "index": index,
                "observed": {"wall_seconds": 60.0 * index},
            }
        )
        chained = link_row(row, scope=_SCOPE, existing_rows=list(linked))
        linked.append(chained)
        payload += canonical_json_bytes(chained) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return linked


def test_an_honest_ledger_sweeps_clean_and_both_heads_agree(tmp_path: Path) -> None:
    """Transparency, measured on bytes: the two rules give one head."""

    ledger = tmp_path / "spend.jsonl"
    _honest_ledger(ledger)
    sweep = sweep_ledger(_SCOPE, ledger)
    assert sweep.rows == 3
    assert sweep.clean
    assert sweep.ladder_head == sweep.derived_head
    assert sweep.as_json()["heads_agree"] is True


def test_the_sweep_names_a_hijacked_row(tmp_path: Path) -> None:
    """The PSLRE5 shape: a foreign digest field that HELD the honest address."""

    ledger = tmp_path / "spend.jsonl"
    rows = _honest_ledger(ledger)
    victim = dict(rows[1])
    stolen = str(victim["row_sha256"])
    edited = {
        **{key: value for key, value in victim.items() if key != "row_sha256"},
        "observed": {"wall_seconds": 0.0},
        "close_sha256": stolen,
    }
    edited["row_sha256"] = stable_hash(
        {key: value for key, value in edited.items() if key not in {"chain", "row_sha256"}}
    )
    lines = ledger.read_bytes().splitlines(keepends=True)
    lines[1] = canonical_json_bytes(edited) + b"\n"
    ledger.write_bytes(b"".join(lines))

    sweep = sweep_ledger(_SCOPE, ledger)
    assert not sweep.clean
    assert sweep.ladder_head != sweep.derived_head
    assert [item["row"] for item in sweep.address_divergences] == [1]
    # BOTH declared digests refuse: the stolen ``close_sha256`` is not the derived
    # address, and the forger's own "honest" re-stamp hashed the injected field into
    # the body, so it is not the derived address either.
    assert {item["field"] for item in sweep.fused_refusals} == {
        "close_sha256",
        "row_sha256",
    }
    # The ladder's answer for that row IS the stolen address -- which is exactly
    # the defect, reproduced by the tool that has to be able to see it.
    assert ladder_content_digest(edited) == stolen


def test_the_sweep_names_a_stale_digest_and_a_padded_envelope(tmp_path: Path) -> None:
    """A row stamped by a DIFFERENT rule, and an envelope carrying an extra key."""

    ledger = tmp_path / "spend.jsonl"
    rows = _honest_ledger(ledger)
    lines = ledger.read_bytes().splitlines(keepends=True)

    stale = dict(rows[0])
    stale["row_sha256"] = stable_hash({"stamped": "by an older rule"})
    lines[0] = canonical_json_bytes(stale) + b"\n"

    padded = dict(rows[2])
    padded["chain"] = {**dict(padded["chain"]), "note": "laundering channel"}  # type: ignore[arg-type, call-overload]
    lines[2] = canonical_json_bytes(padded) + b"\n"
    ledger.write_bytes(b"".join(lines))

    sweep = sweep_ledger(_SCOPE, ledger)
    assert not sweep.clean
    assert [item["row"] for item in sweep.fused_refusals] == [0]
    assert [item["row"] for item in sweep.envelope_refusals] == [2]
    assert [item["unexpected_keys"] for item in sweep.envelope_refusals] == [["note"]]


def test_the_sweep_names_a_nonconforming_envelope_value(tmp_path: Path) -> None:
    """PSLRE6 D6-5: address-invariant ``seq`` padding is surfaced before relaunch too.

    The verifier now refuses an envelope whose VALUES are not machinery-shaped. The
    sweep exists so a relaunch meets no refusal it has not already named, so the same
    condition is reported here rather than discovered on the first read.
    """

    ledger = tmp_path / "spend.jsonl"
    rows = _honest_ledger(ledger)
    lines = ledger.read_bytes().splitlines(keepends=True)
    padded = dict(rows[1])
    padded["chain"] = {**dict(padded["chain"]), "seq": 1.0}  # type: ignore[arg-type, call-overload]
    lines[1] = canonical_json_bytes(padded) + b"\n"
    ledger.write_bytes(b"".join(lines))

    sweep = sweep_ledger(_SCOPE, ledger)
    assert not sweep.clean
    assert [item["row"] for item in sweep.envelope_refusals] == [1]
    assert sweep.envelope_refusals[0]["nonconforming_values"] == [
        "seq must be a non-negative integer row index"
    ]


def test_the_sweep_names_a_pre_existing_legacy_settlement_row(tmp_path: Path) -> None:
    """PSLRE6 acct F4: the sweep surfaces the brick the money gate would refuse on.

    ``_admit_settlement_schema`` refuses a legacy settlement schema on this ledger
    unconditionally and with zero producer input -- correct, and what makes forging the
    cutover declaration worthless. Its cost is that a row already on disk under
    ``paid-session-settlement.v1``/``v2``/``v3`` refuses the whole money authority on
    its first read: a BRICK, never a forgery, but one an operator would otherwise meet
    for the first time at relaunch with a message about a cutover rather than a row.
    """

    from menagerie.crawler.paid_session_accounting import LEGACY_SETTLEMENT_SCHEMAS

    ledger = tmp_path / "spend.jsonl"
    rows = _honest_ledger(ledger)
    assert sweep_ledger(_SCOPE, ledger).legacy_settlements == ()

    legacy = sorted(LEGACY_SETTLEMENT_SCHEMAS)[0]
    body = {
        "schema_version": legacy,
        "event": "session_settled",
        "spend_id": "paid-spend-legacy",
        "observed": {"wall_seconds": 1.0},
    }
    row = stamp_row_digest(body)
    chained = link_row(row, scope=_SCOPE, existing_rows=list(rows))
    ledger.write_bytes(ledger.read_bytes() + canonical_json_bytes(chained) + b"\n")

    sweep = sweep_ledger(_SCOPE, ledger)
    # The row is honestly stamped and honestly linked -- the ADDRESS is fine.
    assert sweep.ladder_head == sweep.derived_head
    assert sweep.fused_refusals == () and sweep.address_divergences == ()
    # ...and the sweep still refuses to call the ledger clean.
    assert not sweep.clean
    assert [item["row"] for item in sweep.legacy_settlements] == [len(rows)]
    assert sweep.legacy_settlements[0]["schema_version"] == legacy
    assert "BRICKED" in str(sweep.legacy_settlements[0]["consequence"])

    # The arm is scope-precise: an identical row on a non-money scope is not its concern.
    assert sweep_ledger("meta.closes", ledger).legacy_settlements == ()


def test_the_sweep_reports_a_malformed_line_without_raising(tmp_path: Path) -> None:
    """A read-only sweep over live bytes may never die on one bad line."""

    ledger = tmp_path / "spend.jsonl"
    _honest_ledger(ledger)
    ledger.write_bytes(ledger.read_bytes() + b"{not json\n")
    sweep = sweep_ledger(_SCOPE, ledger)
    assert not sweep.clean
    assert sweep.unreadable and "not valid JSON" in sweep.unreadable[0]


def test_the_sweep_covers_the_whole_chain_universe(tmp_path: Path) -> None:
    """Every scope the Track A verifier walks is in the sweep's universe."""

    scopes = {scope for scope, _path in chain_bearing_ledgers(tmp_path / "records")}
    assert {"meta.leg-events", "meta.closes", "meta.registry", "paid.sessions"} <= scopes
    assert len(scopes) >= 20, sorted(scopes)
    # An empty root is CLEAN, not an error: nothing on disk cannot have moved.
    assert all(sweep.clean for sweep in sweep_records_root(tmp_path / "records"))


def test_the_cli_exits_nonzero_on_divergence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The relaunch precondition is an exit code and a machine-readable report."""

    root = tmp_path / "records"
    _honest_ledger(root / "operational" / "spend.jsonl")
    assert main([str(root)]) == 0
    assert "CHAIN_ADDRESS_SWEEP_CLEAN" in capsys.readouterr().out

    ledger = root / "operational" / "spend.jsonl"
    lines = ledger.read_bytes().splitlines(keepends=True)
    broken = json.loads(lines[0])
    broken["row_sha256"] = stable_hash({"different": "rule"})
    lines[0] = canonical_json_bytes(broken) + b"\n"
    ledger.write_bytes(b"".join(lines))

    assert main([str(root), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["clean"] is False
    # v2 adds the envelope-VALUE and LEGACY-SETTLEMENT arms to every ledger row.
    assert report["schema_version"] == "menagerie.crawler.chain-address-sweep.v2"
    assert all("legacy_settlements" in item for item in report["ledgers"])
    paid = [item for item in report["ledgers"] if item["scope"] == "paid.sessions"]
    assert paid and paid[0]["fused_refusals"]
