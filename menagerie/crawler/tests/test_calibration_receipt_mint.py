"""W2: the calibration receipt is TOOL-MINTED and recompute-verified.

The receipt's measured numbers derive from the campaign's own ledgers -- the
gates ledger (rendered sessions + walls, reused rows and off-production-width
metadata rows excluded), the paid-session spend ledger (the bought-session
denominator; sol P2-12 denominator honesty), and the wrapper telemetry
(completed-attempt p99 + output tokens, censored attempts never pooled) --
plus the B1-XV2 blind-replay matrix report, whose aggregates are re-derived
from per-golden raw rows and whose bytes are digest-bound into the receipt.
The verifier recomputes EVERYTHING from the same rows and refuses divergence
(fable C2), so a hand-edited receipt cannot survive its own validator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from menagerie.crawler.checker_calibration import (
    CALIBRATION_MIN_SESSIONS,
    CheckerCalibrationInvalidError,
    load_checker_calibration_receipt,
)
from menagerie.crawler.checker_calibration_mint import (
    CheckerCalibrationDivergenceError,
    derive_blind_replay_block,
    gather_lane_measurements,
    load_blind_replay_report,
    mint_calibration_receipt,
    verify_calibration_receipt,
)
from menagerie.crawler.constants import METADATA_BATCH_MAX, GateKind
from menagerie.crawler.paid_session_accounting import BudgetPolicy
from menagerie.crawler.paid_session_lifecycle import (
    PaidSessionAuthority,
    ProviderReconciliation,
)

pytestmark = pytest.mark.smoke

MODEL = "gpt-5.6-sol"
NOW = "2026-08-24T12:00:00Z"


def _gate_row(
    kind: str,
    *,
    wall: float = 100.0,
    model: str = MODEL,
    batch_size: Optional[int] = None,
    reused: bool = False,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
) -> dict[str, Any]:
    minutes = int(wall // 60)
    seconds = wall - 60 * minutes
    row: dict[str, Any] = {
        "gate_kind": kind,
        "batch_size": (
            batch_size
            if batch_size is not None
            else (METADATA_BATCH_MAX if kind == "metadata_batch" else 1)
        ),
        "checker": {
            "provider": "openai",
            "model": model,
            "version": "current",
            "prompt_sha256": "sha256:" + "a" * 64,
            "started_at": "2026-08-24T10:00:00Z",
            "finished_at": f"2026-08-24T10:{minutes:02d}:{seconds:06.3f}Z",
        },
        "items": [],
    }
    if reused:
        row["verdict_source"] = "reused"
    if started_at is not None:
        row["checker"]["started_at"] = started_at
    if finished_at is not None:
        row["checker"]["finished_at"] = finished_at
    return row


def _write_gates(records_root: Path, rows: list[dict[str, Any]]) -> None:
    path = records_root / "gates" / "current-shard.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _buy_sessions(
    records_root: Path, counts: dict[str, int], *, genesis_at: Optional[str] = None
) -> None:
    clock = {"now": genesis_at if genesis_at is not None else NOW}
    authority = PaidSessionAuthority(
        records_root / "operational" / "spend.jsonl",
        runtime_root=records_root.parent / ".crawl-local",
        clock=lambda: clock["now"],
    )
    if genesis_at is not None:
        # Anchor the ledger at a cutover genesis: sessions rendered before
        # this instant settled on a predecessor ledger, not here.
        legacy = records_root / "operational" / "legacy-events.jsonl"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.touch()
        authority.initialize_legacy_prefix(legacy)
        clock["now"] = NOW
    policy = BudgetPolicy(
        quota_sessions=1_000,
        quota_wall_seconds=1_000_000.0,
        runaway_sessions=1_000,
        runaway_wall_seconds=1_000_000.0,
        reservation_wall_seconds=100.0,
    )
    for kind, count in counts.items():
        for index in range(count):
            reservation = authority.reserve(
                scope={
                    "campaign_id": "c1-mech",
                    "generation_id": "generation-open",
                    "lane": "checker",
                    "dispatch_kind": kind,
                    "work_generations": [["m_calib", f"work-{kind}-{index}"]],
                    "gate_kind": kind,
                    "checker_identity": {
                        "provider": "openai",
                        "model": MODEL,
                        "version": "current",
                        "prompt_sha256": "sha256:" + "a" * 64,
                    },
                    "run_id": "run-calibration",
                    "machine_id": "host-calibration",
                },
                policy=policy,
            )
            authority.settle(
                reservation,
                provider_query=lambda key: ProviderReconciliation(
                    accepted=True,
                    observed_wall_seconds=90.0,
                    receipt_sha256="sha256:" + "c" * 64,
                    adapter_identity=None,
                ),
                custody_outcome="exited",
                machine_wall_seconds=90.0,
                local_outcome="completed",
            )


def _write_telemetry(work_root: Path, records: list[dict[str, Any]]) -> None:
    path = work_root / "checker" / "metadata-1" / "operator-telemetry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def _attempt(
    kind: str, *, duration: float = 80.0, tokens: float = 12_000, timed_out: bool = False
) -> dict[str, Any]:
    return {
        "event": "codex-attempt",
        "attempt": 1,
        "classification": "wall" if timed_out else "success",
        "timed_out": timed_out,
        "duration_seconds": duration,
        "gate_kind": kind,
        "batch_size": METADATA_BATCH_MAX if kind == "metadata_batch" else 1,
        "item_count": METADATA_BATCH_MAX if kind == "metadata_batch" else 1,
        "output_tokens": tokens,
        "input_tokens": 40_000,
    }


def _accept_projection(stable_id: str) -> dict[str, Any]:
    return {stable_id: {"verdict": "accurate", "integrity": "accurate", "rung_check": "accurate"}}


def _replay_report(path: Path, *, goldens: int = 9, disagreement: bool = False) -> Path:
    """A v2 raw-facts report: per-draw rows, projections, attestation rows."""

    degenerate = METADATA_BATCH_MAX == 1
    rows = []
    for index in range(goldens):
        stable_id = f"m_{index}"
        projections = [_accept_projection(stable_id) for _ in range(3)]
        if disagreement and index == 0:
            projections[1][stable_id]["verdict"] = "inaccurate"
        rotated = _accept_projection(stable_id)
        if index == 1:
            rotated[stable_id]["rung_check"] = "cannot-verify"
        rows.append(
            {
                "name": f"golden_{index}",
                "kind": "metadata_batch",
                "width": METADATA_BATCH_MAX,
                "request": None,
                "same_epoch": {
                    "replays": len(projections),
                    "projections": projections,
                    "disagreement": any(p != projections[0] for p in projections[1:]),
                },
                "rotation": {
                    "ran": True,
                    "projection": rotated,
                    "disagreement": rotated != projections[0],
                },
                "width_invariance": {
                    "widths": [1] if degenerate else sorted({1, METADATA_BATCH_MAX}),
                    "disagreement": False,
                    "degenerate_at_production_width": degenerate,
                    "folded_into_same_epoch": degenerate,
                },
                "draws": [
                    {
                        "ask": f"golden_{index}:se{replay + 1}",
                        "mode": "same_epoch",
                        "projection": projection,
                        "latency_seconds": 80.0,
                        "integrity_dispute_entries": 0,
                        "licensed": False,
                    }
                    for replay, projection in enumerate(projections)
                ]
                + [
                    {
                        "ask": f"golden_{index}:rot",
                        "mode": "rotation",
                        "projection": rotated,
                        "latency_seconds": 80.0,
                        "integrity_dispute_entries": 0,
                        "licensed": False,
                    }
                ],
                "locator_attestation": {
                    "rows": [
                        {
                            "schema": "text-line-locator.v1",
                            "evidence_id": f"ev-{index}",
                            "source_id": "doc",
                            "source_content_sha256": "sha256:" + "a" * 64,
                            "excerpt_text_sha256": "sha256:" + "b" * 64,
                            "locator": "doc.md line 3",
                            "form": "line",
                            "basis": "one-based-inclusive-universal-newline-v1",
                            "equivalence": "unicode-whitespace-run-v1",
                            "start_line": 3,
                            "end_line": 3,
                            "status": "valid",
                            "detail": None,
                        }
                    ]
                },
            }
        )
    report = {
        "schema_version": "menagerie.crawler.blind-replay-report.v2",
        "checker_model": MODEL,
        "captured_at": NOW,
        "production_batch_width": METADATA_BATCH_MAX,
        "goldens": rows,
        "latency_seconds_at_production_width": [80.0, 90.0, 120.0],
        "authority_conflicts": [],
    }
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def campaign(tmp_path: Path) -> dict[str, Any]:
    """A measured campaign tree: 20/5/5 rendered, +2/+1/+0 refused buys."""

    records_root = tmp_path / "records"
    work_root = tmp_path / ".crawl-local" / "work"
    rows = (
        [_gate_row("metadata_batch") for _ in range(19)]
        + [_gate_row("metadata_batch", wall=200.0)]
        # Excluded rows: reused replay, off-production width, foreign model.
        + [_gate_row("metadata_batch", reused=True)]
        + [_gate_row("metadata_batch", batch_size=METADATA_BATCH_MAX + 4)]
        + [_gate_row("metadata_batch", model="gpt-5.6-terra")]
        + [_gate_row("fidelity") for _ in range(5)]
        + [_gate_row("terminal_disposition") for _ in range(5)]
    )
    _write_gates(records_root, rows)
    _buy_sessions(
        records_root,
        {"metadata_batch": 22, "fidelity": 6, "terminal_disposition": 5},
    )
    telemetry = (
        [_attempt("metadata_batch") for _ in range(20)]
        + [_attempt("metadata_batch", duration=1_500.0, timed_out=True)]
        + [_attempt("fidelity", tokens=5_000) for _ in range(5)]
        + [_attempt("terminal_disposition", duration=140.0, tokens=3_000) for _ in range(5)]
    )
    _write_telemetry(work_root, telemetry)
    report = _replay_report(tmp_path / "replay-report.json")
    return {"records_root": records_root, "work_root": work_root, "report": report}


def _mint(campaign: dict[str, Any], out: Path, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "work_roots": (campaign["work_root"],),
        "checker_model": MODEL,
        "blind_replay_report_path": campaign["report"],
        "captured_at": NOW,
        "source": "test calibration leg",
        "next_rung_models": 200,
        "projected_wall_hours": 60.0,
        "wall_budget_hours": 120.0,
        "projected_sessions": 800,
        "session_budget": 2_000,
        "projected_dollars": 120.0,
        "dollar_budget": 300.0,
        "out_path": out,
    }
    kwargs.update(overrides)
    return mint_calibration_receipt(campaign["records_root"], **kwargs)


def test_measurements_derive_from_the_ledgers(campaign: dict[str, Any]) -> None:
    lanes = gather_lane_measurements(
        campaign["records_root"], work_roots=(campaign["work_root"],), checker_model=MODEL
    )
    metadata = lanes["metadata_batch"]
    # 20 rendered at production width: the reused replay, the off-width row,
    # and the terra row are all excluded from the sample.
    assert metadata["sessions"] == 20
    assert metadata["sessions_bought"] == 22
    assert metadata["refusal_rate"] == pytest.approx(2 / 22)
    assert metadata["latency_p99_seconds"] == pytest.approx(200.0)
    assert metadata["latency_p95_seconds"] == pytest.approx(100.0)
    # The censored 1500s attempt never pools into the completed p99.
    assert metadata["attempt_latency_p99_seconds"] == pytest.approx(80.0)
    assert metadata["attempt_samples"] == 20
    assert metadata["attempt_censoring_rate"] == pytest.approx(1 / 21)
    assert metadata["output_tokens_p99"] == pytest.approx(12_000)
    assert lanes["fidelity"]["sessions"] == 5
    assert lanes["fidelity"]["sessions_bought"] == 6
    assert lanes["terminal_disposition"]["refusal_rate"] == 0.0


def test_mint_writes_a_receipt_the_activation_loader_accepts(
    campaign: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "receipt.json"
    receipt = _mint(campaign, out)
    assert out.is_file()
    assert load_checker_calibration_receipt(out) == receipt
    replay = derive_blind_replay_block(
        load_blind_replay_report(campaign["report"]), report_path=campaign["report"]
    )
    assert receipt["blind_replay"] == replay
    assert receipt["blind_replay"]["same_epoch_disagreements"] == 0
    assert receipt["blind_replay"]["rotation_disagreements"] == 1
    assert receipt["blind_replay"]["p99_at_production_width_seconds"] == pytest.approx(120.0)
    # Verification recomputes from the same ledgers and agrees.
    verified = verify_calibration_receipt(
        out,
        campaign["records_root"],
        work_roots=(campaign["work_root"],),
        blind_replay_report_path=campaign["report"],
    )
    assert verified == receipt


def test_mint_is_fail_closed_no_broken_receipt_survives_on_disk(
    campaign: dict[str, Any], tmp_path: Path
) -> None:
    # A disagreeing blind-replay matrix refuses the mint AND removes the file.
    bad_report = _replay_report(tmp_path / "bad-report.json", disagreement=True)
    out = tmp_path / "refused.json"
    with pytest.raises(CheckerCalibrationInvalidError, match="disagreement"):
        _mint(campaign, out, blind_replay_report_path=bad_report)
    assert not out.exists(), "a refused receipt must never survive on disk"


def test_under_floor_measurement_refuses_the_mint(campaign: dict[str, Any], tmp_path: Path) -> None:
    # Strip the gates ledger below the metadata floor: the mint's own
    # round-trip through the activation loader refuses.
    records_root = campaign["records_root"]
    floor = CALIBRATION_MIN_SESSIONS[GateKind.METADATA_BATCH.value]
    rows = (
        [_gate_row("metadata_batch") for _ in range(floor - 1)]
        + [_gate_row("fidelity") for _ in range(5)]
        + [_gate_row("terminal_disposition") for _ in range(5)]
    )
    _write_gates(records_root, rows)
    out = tmp_path / "under-floor.json"
    with pytest.raises(CheckerCalibrationInvalidError, match="floor"):
        _mint(campaign, out)
    assert not out.exists()


def test_verify_refuses_a_hand_edited_receipt(campaign: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "receipt.json"
    _mint(campaign, out)
    doctored = json.loads(out.read_text(encoding="utf-8"))
    doctored["lanes"]["metadata_batch"]["refusal_rate"] = 0.0
    out.write_text(json.dumps(doctored), encoding="utf-8")
    with pytest.raises(
        CheckerCalibrationDivergenceError, match="lanes.metadata_batch.refusal_rate"
    ):
        verify_calibration_receipt(
            out, campaign["records_root"], work_roots=(campaign["work_root"],)
        )


def test_verify_binds_the_replay_report_bytes(campaign: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "receipt.json"
    _mint(campaign, out)
    # Rewriting the report -- even semantically identically -- moves the byte
    # digest, so a swapped report refuses.
    report = json.loads(campaign["report"].read_text(encoding="utf-8"))
    campaign["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    with pytest.raises(CheckerCalibrationDivergenceError, match="report_sha256"):
        verify_calibration_receipt(
            out,
            campaign["records_root"],
            work_roots=(campaign["work_root"],),
            blind_replay_report_path=campaign["report"],
        )


def test_more_rendered_than_bought_is_a_ledger_inconsistency(
    campaign: dict[str, Any],
) -> None:
    records_root = campaign["records_root"]
    rows = (
        [_gate_row("metadata_batch") for _ in range(25)]
        + [_gate_row("fidelity") for _ in range(5)]
        + [_gate_row("terminal_disposition") for _ in range(5)]
    )
    _write_gates(records_root, rows)
    with pytest.raises(CheckerCalibrationInvalidError, match="denominator"):
        gather_lane_measurements(
            records_root, work_roots=(campaign["work_root"],), checker_model=MODEL
        )


GENESIS_AT = "2026-08-20T09:50:19Z"
PRE_GENESIS_STARTED = "2026-08-18T10:00:00Z"
PRE_GENESIS_FINISHED = "2026-08-18T10:05:00Z"


def _epoch_campaign(
    tmp_path: Path,
    *,
    pre_genesis_fidelity: int = 33,
    post_genesis_fidelity: int = 4,
    undated_fidelity: int = 0,
    fidelity_buys: int = 4,
) -> dict[str, Any]:
    """A campaign whose spend ledger begins at a cutover/rollover genesis.

    The default is the pilot's exact 2026-08-26 refusal shape: 37 fidelity
    verdicts rendered all-time, 33 of them before the ledger's genesis anchor
    (settled on a predecessor ledger), and only 4 post-genesis fidelity buys.
    """

    records_root = tmp_path / "records"
    work_root = tmp_path / ".crawl-local" / "work"
    rows = (
        [_gate_row("metadata_batch") for _ in range(20)]
        + [
            _gate_row(
                "fidelity",
                started_at=PRE_GENESIS_STARTED,
                finished_at=PRE_GENESIS_FINISHED,
            )
            for _ in range(pre_genesis_fidelity)
        ]
        + [_gate_row("fidelity") for _ in range(post_genesis_fidelity)]
        + [
            _gate_row("fidelity", started_at="", finished_at="")
            for _ in range(undated_fidelity)
        ]
        + [_gate_row("terminal_disposition") for _ in range(5)]
    )
    _write_gates(records_root, rows)
    _buy_sessions(
        records_root,
        {"metadata_batch": 22, "fidelity": fidelity_buys, "terminal_disposition": 5},
        genesis_at=GENESIS_AT,
    )
    telemetry = (
        [_attempt("metadata_batch") for _ in range(20)]
        + [_attempt("fidelity", tokens=5_000) for _ in range(5)]
        + [_attempt("terminal_disposition", duration=140.0, tokens=3_000) for _ in range(5)]
    )
    _write_telemetry(work_root, telemetry)
    report = _replay_report(tmp_path / "replay-report.json")
    return {"records_root": records_root, "work_root": work_root, "report": report}


def test_pre_genesis_rendered_artifacts_are_disclosed_never_counted(tmp_path: Path) -> None:
    # The pilot's 2026-08-26 structural blocker: the measurement window must be
    # epoch-scoped to the paid ledger's genesis. Pre-genesis rendered artifacts
    # have no rows on THIS ledger by construction -- they are disclosed in
    # pre_genesis_rendered, never counted against the bought denominator.
    campaign = _epoch_campaign(tmp_path)
    lanes = gather_lane_measurements(
        campaign["records_root"], work_roots=(campaign["work_root"],), checker_model=MODEL
    )
    fidelity = lanes["fidelity"]
    assert fidelity["sessions"] == 4
    assert fidelity["sessions_bought"] == 4
    assert fidelity["pre_genesis_rendered"] == 33
    assert fidelity["refusal_rate"] == 0.0
    assert lanes["metadata_batch"]["pre_genesis_rendered"] == 0
    assert lanes["terminal_disposition"]["pre_genesis_rendered"] == 0


def test_a_genuinely_inconsistent_post_genesis_ledger_still_refuses(tmp_path: Path) -> None:
    # Epoch scoping must never disarm the tripwire: more POST-genesis rendered
    # verdicts than bought sessions is still a ledger inconsistency.
    campaign = _epoch_campaign(tmp_path, post_genesis_fidelity=6, fidelity_buys=4)
    with pytest.raises(CheckerCalibrationInvalidError, match="denominator"):
        gather_lane_measurements(
            campaign["records_root"], work_roots=(campaign["work_root"],), checker_model=MODEL
        )


def test_an_undated_gate_row_counts_inside_the_measurement_window(tmp_path: Path) -> None:
    # A rendered row the window cannot place is counted IN-window: it can only
    # trigger the inconsistency refusal, never escape it through the epoch.
    campaign = _epoch_campaign(
        tmp_path, post_genesis_fidelity=4, undated_fidelity=1, fidelity_buys=4
    )
    with pytest.raises(CheckerCalibrationInvalidError, match="denominator"):
        gather_lane_measurements(
            campaign["records_root"], work_roots=(campaign["work_root"],), checker_model=MODEL
        )


def test_a_genesis_anchor_without_a_timestamp_refuses_typed() -> None:
    from menagerie.crawler.checker_calibration_mint import _paid_ledger_genesis

    with pytest.raises(CheckerCalibrationInvalidError, match="genesis"):
        _paid_ledger_genesis(({"event": "accounting_rollover", "created_at": "not-a-time"},))


def test_matrix_sessions_outside_gatekind_never_inflate_a_lane_denominator(
    campaign: dict[str, Any],
) -> None:
    # The blind-replay matrix buys wrapper sessions under its own dispatch/gate
    # kind OUTSIDE GateKind; they must never inflate a measured lane's bought
    # denominator (which would fabricate refusals the lane never had).
    _buy_sessions(campaign["records_root"], {"blind-replay-matrix": 3})
    lanes = gather_lane_measurements(
        campaign["records_root"], work_roots=(campaign["work_root"],), checker_model=MODEL
    )
    assert lanes["metadata_batch"]["sessions_bought"] == 22
    assert lanes["fidelity"]["sessions_bought"] == 6
    assert lanes["terminal_disposition"]["sessions_bought"] == 5


def test_the_epoch_disclosure_is_recompute_verified(tmp_path: Path) -> None:
    # pre_genesis_rendered is a measured field like any other: the receipt
    # carries it, and a hand-edited disclosure refuses at verify.
    campaign = _epoch_campaign(tmp_path, post_genesis_fidelity=5, fidelity_buys=6)
    out = tmp_path / "receipt.json"
    receipt = _mint(campaign, out)
    assert receipt["lanes"]["fidelity"]["pre_genesis_rendered"] == 33
    verify_calibration_receipt(
        out, campaign["records_root"], work_roots=(campaign["work_root"],)
    )
    doctored = json.loads(out.read_text(encoding="utf-8"))
    doctored["lanes"]["fidelity"]["pre_genesis_rendered"] = 0
    out.write_text(json.dumps(doctored), encoding="utf-8")
    with pytest.raises(
        CheckerCalibrationDivergenceError, match="lanes.fidelity.pre_genesis_rendered"
    ):
        verify_calibration_receipt(
            out, campaign["records_root"], work_roots=(campaign["work_root"],)
        )


def test_metadata_bought_denominator_is_width_scoped(campaign: dict[str, Any]) -> None:
    """Sessions reserved at another batch width bought a DIFFERENT lane.

    The width rotation (Opus adjudication 5.3 rule 1) leaves the retired
    width's measurement buys on the same durable root; without denominator
    scoping they would fabricate a refusal rate the production lane never had
    (25 retired width-5 buys against ~21 width-1 verdicts reads as 0.57). The
    scoping mirrors the rendered numerator exactly -- and a reservation whose
    width cannot be derived still counts, so stripping the member list can
    never hide a dead session.
    """

    records_root = campaign["records_root"]
    baseline = gather_lane_measurements(
        records_root, work_roots=(campaign["work_root"],), checker_model=MODEL
    )["metadata_batch"]["sessions_bought"]
    authority = PaidSessionAuthority(
        records_root / "operational" / "spend.jsonl",
        runtime_root=records_root.parent / ".crawl-local",
        clock=lambda: NOW,
    )
    policy = BudgetPolicy(
        quota_sessions=1_000,
        quota_wall_seconds=1_000_000.0,
        runaway_sessions=1_000,
        runaway_wall_seconds=1_000_000.0,
        reservation_wall_seconds=100.0,
    )

    def buy(members: Any) -> None:
        reservation = authority.reserve(
            scope={
                "campaign_id": "c1-mech",
                "generation_id": "generation-open",
                "lane": "checker",
                "dispatch_kind": "metadata_batch",
                "work_generations": members,
                "gate_kind": "metadata_batch",
                "checker_identity": {
                    "provider": "openai",
                    "model": MODEL,
                    "version": "current",
                    "prompt_sha256": "sha256:" + "a" * 64,
                },
                "run_id": "run-calibration",
                "machine_id": "host-calibration",
            },
            policy=policy,
        )
        authority.settle(
            reservation,
            provider_query=lambda key: ProviderReconciliation(
                accepted=True,
                observed_wall_seconds=90.0,
                receipt_sha256="sha256:" + "c" * 64,
                adapter_identity=None,
            ),
            custody_outcome="exited",
            machine_wall_seconds=90.0,
            local_outcome="completed",
        )

    # Two off-width buys (a retired wider lane) and one production-width buy.
    off_width = [[f"m_off_{index}", f"work-off-{index}"] for index in range(METADATA_BATCH_MAX + 4)]
    buy(off_width)
    buy(off_width)
    buy([["m_on", "work-on"]] * METADATA_BATCH_MAX)
    lanes = gather_lane_measurements(
        records_root, work_roots=(campaign["work_root"],), checker_model=MODEL
    )
    assert lanes["metadata_batch"]["sessions_bought"] == baseline + 1


def test_metadata_attempt_tail_is_width_scoped(campaign: dict[str, Any]) -> None:
    """Off-width attempt telemetry never pools into the production tail.

    A record with no readable batch_size still pools in -- an unknown-width
    sample can only lengthen the reported tail, never shorten it.
    """

    extra = [
        # Off-width: a much slower retired-lane attempt that would otherwise
        # dominate the p99.
        dict(_attempt("metadata_batch", duration=1_400.0), batch_size=METADATA_BATCH_MAX + 4),
        # Unknown width: counts, fail-closed.
        {
            key: value
            for key, value in _attempt("metadata_batch", duration=95.0).items()
            if key != "batch_size"
        },
    ]
    path = campaign["work_root"] / "checker" / "metadata-extra" / "operator-telemetry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in extra), encoding="utf-8")
    lanes = gather_lane_measurements(
        campaign["records_root"], work_roots=(campaign["work_root"],), checker_model=MODEL
    )
    metadata = lanes["metadata_batch"]
    # 20 production-width completions + the unknown-width one; the 1400s
    # off-width sample is excluded, so the completed p99 stays at the
    # production lane's own tail.
    assert metadata["attempt_samples"] == 21
    assert metadata["attempt_latency_p99_seconds"] == pytest.approx(95.0)
