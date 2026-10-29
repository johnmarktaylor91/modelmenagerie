"""W19 operator work-queue tooling: ONE queue derivation, parks, aging, shadow census.

The census JMT received was wrong in COMPOSITION: it conflated queue MEMBERS
(stable_ids whose head is review-held) with queue ROWS (live park/grant rows),
reporting 12 members as 14 rows. The operator dossier's DELTA-1 pinned the
consequence: a hand-copied "42-head review queue" went stale while the live
ledger held 55. These tests pin the fix: queue membership is RE-DERIVED at
execution by ``derive_operator_work_queue()`` (never hand-copied), members and
rows are counted distinctly, park aging is derived from an explicit ``as_of``
(never wall-clock), the park dossier golden covers the 200a nine, and the
B2-X-V30 shadow operator census recomputes a close's withheld set THREE
independent ways and CATCHES a reported set that silently drops a member.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from menagerie.crawler.close_object import build_close_object, close_object_path
from menagerie.crawler.grant_parks import (
    AWARD_PARK_DISPOSITION,
    GRANT_PENDING_DISPOSITION,
    build_grant_park_row,
    grant_park_path,
)
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.leg_generation import CLOSE_SOURCE_NAMES
from menagerie.crawler.partition_proof import (
    PARTITION_PROOF_SCHEMA_VERSION,
    write_partition_proof,
)
from menagerie.crawler.refusal_parks import (
    REFUSAL_PARK_DISPOSITION,
    build_refusal_park_row,
    refusal_park_path,
)
from menagerie.crawler.tools.operator_work_queue import (
    OperatorWorkQueueError,
    derive_operator_work_queue,
    main,
    park_aging_census,
    park_dossier,
    shadow_operator_census,
)

pytestmark = pytest.mark.smoke

AS_OF = "2026-08-25T00:00:00Z"
CAMPAIGN = "camp-1"


def _revision(seed: str) -> str:
    return "sha256:" + (seed * 64)[:64]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _head_row(
    stable_id: str,
    *,
    revision: str,
    code: str = "failed:author",
    kind: str = "failed",
    review_required: bool = False,
    requested_at: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "stable_id": stable_id,
        "record_revision": revision,
        "record_seq": 1,
        "parent_revision": None,
        "status": {
            "kind": kind,
            "code": code,
            "human_review": {
                "required": review_required,
                "reason": None,
                "queue": "crawler-human-review" if review_required else None,
                "requested_at": requested_at,
            },
        },
    }


def _grant_park(
    stable_id: str,
    *,
    work_id: str,
    blocked_parent_revision: str,
    created_at: str = "2026-08-20T00:00:00Z",
    blocked_stage: str = "env-generation",
    disposition: str = GRANT_PENDING_DISPOSITION,
    campaign_id: str = CAMPAIGN,
) -> dict[str, Any]:
    return build_grant_park_row(
        stable_id=stable_id,
        work_id=work_id,
        name=stable_id,
        campaign_id=campaign_id,
        run_id="run-1",
        machine_id="machine-1",
        created_at=created_at,
        blocked_parent_revision=blocked_parent_revision,
        blocked_stage=blocked_stage,
        pending_model={
            "stable_id": stable_id,
            "status": {"code": "runs", "reason_code": None, "stage": None},
        },
        disposition=disposition,
    )


def _refusal_park(
    stable_id: str,
    *,
    work_id: str,
    blocked_parent_revision: Optional[str],
    created_at: str = "2026-08-20T00:00:00Z",
    campaign_id: str = CAMPAIGN,
) -> dict[str, Any]:
    return build_refusal_park_row(
        stable_id=stable_id,
        work_id=work_id,
        name=stable_id,
        campaign_id=campaign_id,
        run_id="run-1",
        machine_id="machine-1",
        created_at=created_at,
        refusal_class="EnvGenerationError",
        record_error_type="EnvGenerationError",
        record_error="recording refused",
        fallback_error_type=None,
        fallback_error=None,
        intended_status_code="failed:env-generation",
        intended_reason_code=None,
        blocked_parent_revision=blocked_parent_revision,
        candidate_sha256=None,
    )


def _grant_row(grant_id: str, stable_id: str, *, stage: str = "author") -> dict[str, Any]:
    return {
        "grant_id": grant_id,
        "stable_id": stable_id,
        "stage": stage,
        "reason": "operator adjudicated",
        "attempts": 1,
        "granted_by": "operator",
        "created_at": "2026-08-21T00:00:00Z",
    }


def _grants_path(records_root: Path) -> Path:
    return records_root / "operational" / "requeue-grants.jsonl"


def _queue_root(tmp_path: Path) -> Path:
    """The DELTA-1 composition fixture: 4 members carrying 5 live rows.

    * m1 -- review-held, one live grant, TWO live grant-park rows (3 rows).
    * m2 -- review-held, one live grant, one RELEASED park row (1 row).
    * m3 -- review-held, one live refusal park, one VOIDED grant (1 row).
    * m4 -- review-held, one CONSUMED grant (0 rows).
    * m5 -- not review-held: never a member.
    """

    records_root = tmp_path / "records"
    _write_jsonl(
        records_root / "models.jsonl",
        [
            _head_row(
                "m1",
                revision=_revision("1"),
                review_required=True,
                requested_at="2026-08-19T00:00:00Z",
            ),
            _head_row("m2", revision=_revision("2"), review_required=True),
            _head_row("m3", revision=_revision("3"), review_required=True),
            _head_row("m4", revision=_revision("4"), review_required=True),
            _head_row("m5", revision=_revision("5")),
        ],
    )
    _write_jsonl(
        grant_park_path(records_root),
        [
            _grant_park("m1", work_id="w-m1-a", blocked_parent_revision=_revision("1")),
            _grant_park("m1", work_id="w-m1-b", blocked_parent_revision=_revision("1")),
            # Blocked parent differs from m2's landed head: RELEASED, never counted.
            _grant_park("m2", work_id="w-m2", blocked_parent_revision=_revision("9")),
        ],
    )
    _write_jsonl(
        refusal_park_path(records_root),
        [_refusal_park("m3", work_id="w-m3", blocked_parent_revision=_revision("3"))],
    )
    _write_jsonl(
        _grants_path(records_root),
        [
            _grant_row("grant-m1", "m1"),
            _grant_row("grant-m2", "m2"),
            _grant_row("grant-m3", "m3"),
            {"void_of": "grant-m3", "reason": "withdrawn", "voided_by": "operator"},
            _grant_row("grant-m4", "m4"),
        ],
    )
    _write_jsonl(
        records_root / "operational" / "events.jsonl",
        [{"event_kind": "requeue-grant-consumed", "details": {"grant_id": "grant-m4"}}],
    )
    return records_root


# ---------------------------------------------------------------------------
# derive_operator_work_queue -- the DELTA-1 members-vs-rows tripwire
# ---------------------------------------------------------------------------


def test_queue_counts_members_and_rows_distinctly(tmp_path: Path) -> None:
    records_root = _queue_root(tmp_path)
    payload = derive_operator_work_queue(records_root, as_of=AS_OF)
    assert payload["members_total"] == 4
    assert payload["rows_total"] == 5
    assert payload["rows_total"] > payload["members_total"]
    assert payload["with_live_grants"] == 2
    assert payload["without_live_grants"] == 2
    assert [member["stable_id"] for member in payload["members"]] == ["m1", "m2", "m3", "m4"]

    by_id = {member["stable_id"]: member for member in payload["members"]}
    assert by_id["m1"]["live_grant_ids"] == ["grant-m1"]
    assert len(by_id["m1"]["park_row_shas"]) == 2
    assert by_id["m1"]["rows"] == 3
    assert by_id["m1"]["requested_at"] == "2026-08-19T00:00:00Z"
    assert by_id["m1"]["status_code"] == "failed:author"
    # m2's park row was RELEASED by its landed head: never a queue row.
    assert by_id["m2"] == {
        "stable_id": "m2",
        "status_code": "failed:author",
        "requested_at": None,
        "live_grant_ids": ["grant-m2"],
        "park_row_shas": [],
        "rows": 1,
    }


def test_queue_excludes_voided_and_consumed_grants(tmp_path: Path) -> None:
    records_root = _queue_root(tmp_path)
    payload = derive_operator_work_queue(records_root, as_of=AS_OF)
    by_id = {member["stable_id"]: member for member in payload["members"]}
    # grant-m3 was voided and grant-m4 was consumed: neither is live.
    assert by_id["m3"]["live_grant_ids"] == []
    assert by_id["m4"]["live_grant_ids"] == []
    assert by_id["m4"]["park_row_shas"] == []
    assert by_id["m4"]["rows"] == 0
    # m3 still holds its live refusal park row.
    assert by_id["m3"]["rows"] == 1
    # m5 (not review-held) is not a member anywhere in the payload.
    assert "m5" not in by_id


# ---------------------------------------------------------------------------
# park_aging_census
# ---------------------------------------------------------------------------


def test_aging_census_buckets_sla_and_oldest(tmp_path: Path) -> None:
    records_root = tmp_path / "records"
    _write_jsonl(records_root / "models.jsonl", [])
    parks = [
        _grant_park(
            "a1",
            work_id="w-a1",
            blocked_parent_revision=_revision("a"),
            created_at="2026-08-24T12:00:00Z",  # 0.5 days
        ),
        _grant_park(
            "a2",
            work_id="w-a2",
            blocked_parent_revision=_revision("b"),
            created_at="2026-08-23T00:00:00Z",  # 2 days
        ),
        _grant_park(
            "a3",
            work_id="w-a3",
            blocked_parent_revision=_revision("c"),
            created_at="2026-08-20T00:00:00Z",  # 5 days
        ),
    ]
    _write_jsonl(grant_park_path(records_root), parks)
    oldest = _refusal_park(
        "a4",
        work_id="w-a4",
        blocked_parent_revision=None,
        created_at="2026-08-10T00:00:00Z",  # 15 days
    )
    _write_jsonl(refusal_park_path(records_root), [oldest])

    payload = park_aging_census(records_root, as_of=AS_OF, sla_days=3.0)
    assert payload["as_of"] == AS_OF
    assert payload["sla_days"] == 3.0
    assert payload["rows_total"] == 4
    assert payload["count_by_age"] == {"<1d": 1, "1-3d": 1, "3-7d": 1, ">=7d": 1}
    assert [row["stable_id"] for row in payload["sla_breaches"]] == ["a4", "a3"]
    assert payload["oldest"] == {
        "stable_id": "a4",
        "family": "park.refusal",
        "row_sha256": oldest["row_sha256"],
        "created_at": "2026-08-10T00:00:00Z",
        "age_days": 15.0,
    }
    ages = {row["stable_id"]: row["age_days"] for row in payload["rows"]}
    assert ages == {"a1": 0.5, "a2": 2.0, "a3": 5.0, "a4": 15.0}


def test_aging_census_refuses_a_wall_clock_shaped_as_of(tmp_path: Path) -> None:
    records_root = tmp_path / "records"
    _write_jsonl(records_root / "models.jsonl", [])
    with pytest.raises(OperatorWorkQueueError, match="as_of"):
        park_aging_census(records_root, as_of="not-a-timestamp", sla_days=3.0)


# ---------------------------------------------------------------------------
# park_dossier -- the 200a nine golden
# ---------------------------------------------------------------------------

THE_200A_NINE_ENV = ("m10517", "m11578", "m8830", "m8838", "m8887", "m9819", "m9911")
THE_200A_NINE_AWARD = ("m9610", "m634")


def test_dossier_golden_over_the_200a_nine(tmp_path: Path) -> None:
    """Full-structure golden: 7 env-generation grant parks + 2 award parks."""

    records_root = tmp_path / "records"
    _write_jsonl(records_root / "models.jsonl", [])
    rows: dict[str, dict[str, Any]] = {}
    for stable_id in THE_200A_NINE_ENV:
        rows[stable_id] = _grant_park(
            stable_id,
            work_id=f"w-{stable_id}",
            blocked_parent_revision=_revision("e"),
            created_at="2026-08-20T00:00:00Z",
            blocked_stage="env-generation",
            campaign_id="camp-200a",
        )
    for stable_id in THE_200A_NINE_AWARD:
        rows[stable_id] = _grant_park(
            stable_id,
            work_id=f"w-{stable_id}",
            blocked_parent_revision=_revision("f"),
            created_at="2026-08-10T00:00:00Z",
            blocked_stage="accuracy-gate",
            disposition=AWARD_PARK_DISPOSITION,
            campaign_id="camp-200a",
        )
    _write_jsonl(
        grant_park_path(records_root),
        [rows[stable_id] for stable_id in (*THE_200A_NINE_ENV, *THE_200A_NINE_AWARD)],
    )

    nine = sorted((*THE_200A_NINE_ENV, *THE_200A_NINE_AWARD))
    payload = park_dossier(records_root, stable_ids=nine, as_of=AS_OF)

    def _member(stable_id: str) -> dict[str, Any]:
        award = stable_id in THE_200A_NINE_AWARD
        return {
            "stable_id": stable_id,
            "status_code": None,
            "requested_at": None,
            "live_grants": [],
            "parks": [
                {
                    "family": "park.grant",
                    "disposition": AWARD_PARK_DISPOSITION
                    if award
                    else GRANT_PENDING_DISPOSITION,
                    "refusal_class": None,
                    "created_at": "2026-08-10T00:00:00Z" if award else "2026-08-20T00:00:00Z",
                    "age_days": 15.0 if award else 5.0,
                    "row_sha256": rows[stable_id]["row_sha256"],
                }
            ],
        }

    assert payload == {
        "as_of": AS_OF,
        "members_total": 9,
        "members": [_member(stable_id) for stable_id in nine],
    }


def test_dossier_defaults_to_queue_members(tmp_path: Path) -> None:
    records_root = _queue_root(tmp_path)
    payload = park_dossier(records_root, as_of=AS_OF)
    assert [member["stable_id"] for member in payload["members"]] == ["m1", "m2", "m3", "m4"]
    by_id = {member["stable_id"]: member for member in payload["members"]}
    assert by_id["m3"]["parks"][0]["family"] == "park.refusal"
    assert by_id["m3"]["parks"][0]["disposition"] == REFUSAL_PARK_DISPOSITION
    assert by_id["m3"]["parks"][0]["refusal_class"] == "EnvGenerationError"
    assert [grant["grant_id"] for grant in by_id["m1"]["live_grants"]] == ["grant-m1"]


# ---------------------------------------------------------------------------
# shadow_operator_census (B2-X-V30)
# ---------------------------------------------------------------------------


def _shadow_root(tmp_path: Path, *, reported: list[str]) -> tuple[Path, str]:
    """Records root whose ledgers withhold mA and mB, close reporting ``reported``."""

    records_root = tmp_path / "records"
    _write_jsonl(records_root / "models.jsonl", [])
    _write_jsonl(
        grant_park_path(records_root),
        [_grant_park("mA", work_id="w-mA", blocked_parent_revision=_revision("a"))],
    )
    _write_jsonl(
        refusal_park_path(records_root),
        [_refusal_park("mB", work_id="w-mB", blocked_parent_revision=None)],
    )
    proof_body = {
        "schema_version": PARTITION_PROOF_SCHEMA_VERSION,
        "campaign_id": CAMPAIGN,
        "intake_snapshot_id": "intake-1",
        "generation_id": "gen-1",
        "members": {
            "mA": {"disposition": "withheld"},
            "mB": {"disposition": "withheld"},
        },
    }
    write_partition_proof(records_root, proof_body)
    proof_sha = stable_hash(proof_body)
    close = build_close_object(
        campaign_id=CAMPAIGN,
        intake_snapshot_id="intake-1",
        intake_snapshot_sha256=_revision("c"),
        phase=None,
        close_kind="backfill",
        generation_id="gen-1",
        leg_open_row_sha256=None,
        sources={
            name: {"ledger_sha256": _revision("d"), "chain_head": _revision("d")}
            for name in CLOSE_SOURCE_NAMES
        },
        partition_proof_sha256=proof_sha,
        predecessor_close_sha256=None,
        run_id="run-1",
        machine_id="machine-1",
        closed_at="2026-08-24T00:00:00Z",
        completion_status="complete",
        partition_arms={
            "valid": True,
            "intake": 2,
            "missing": [],
            "extra": [],
            "duplicates": [],
            "invalid": [],
            "grant_pending": [],
            "refusal_parked": [],
            "buckets": {},
            "machine_deferred": {},
            "withheld_here": reported,
        },
        currency_census={
            "stale_by_axis": {},
            "award_runs_to_non_runs": [],
            "award_runs_to_runs": [],
        },
        authority_identities={"checker": "identity-1"},
    )
    _write_jsonl(close_object_path(records_root), [close])
    return records_root, str(close["close_sha256"])


def test_shadow_census_consistent_when_reported_matches_all_three(tmp_path: Path) -> None:
    records_root, close_id = _shadow_root(tmp_path, reported=["mA", "mB"])
    payload = shadow_operator_census(records_root, campaign_id=CAMPAIGN, close_id=close_id)
    assert payload["status"] == "consistent"
    assert payload["reported"] == ["mA", "mB"]
    assert payload["sources"] == {
        "status_derived": ["mA", "mB"],
        "driver_state": ["mA", "mB"],
        "close_proof": ["mA", "mB"],
    }
    assert payload["divergences"] == []
    for diff in payload["diffs"].values():
        assert diff == {"missing_from_reported": [], "extra_in_reported": []}


def test_shadow_census_catches_a_silently_dropped_member(tmp_path: Path) -> None:
    """MUTATION proof: the reported set drops mB; every view names the drop."""

    records_root, close_id = _shadow_root(tmp_path, reported=["mA"])
    payload = shadow_operator_census(records_root, campaign_id=CAMPAIGN, close_id=close_id)
    assert payload["status"] == "divergent"
    for source in ("status_derived", "driver_state", "close_proof"):
        assert payload["diffs"][f"{source}_vs_reported"]["missing_from_reported"] == ["mB"]
    assert any("mB" in divergence for divergence in payload["divergences"])


def test_shadow_census_fails_typed_on_a_missing_close(tmp_path: Path) -> None:
    records_root, _close_id = _shadow_root(tmp_path, reported=["mA", "mB"])
    with pytest.raises(OperatorWorkQueueError, match="close"):
        shadow_operator_census(
            records_root, campaign_id=CAMPAIGN, close_id="sha256:" + "0" * 64
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_queue_prints_json_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records_root = _queue_root(tmp_path)
    exit_code = main(["queue", "--records-root", str(records_root), "--as-of", AS_OF])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["members_total"] == 4
    assert payload["rows_total"] == 5


def test_cli_shadow_divergent_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records_root, close_id = _shadow_root(tmp_path, reported=["mA"])
    exit_code = main(
        [
            "shadow",
            "--records-root",
            str(records_root),
            "--campaign",
            CAMPAIGN,
            "--close-id",
            close_id,
        ]
    )
    assert exit_code != 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "divergent"
