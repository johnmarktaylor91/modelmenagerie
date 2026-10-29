"""Forward mechanics of the ladder between one green rung and the next (X5).

Everything here was found by rehearsing a REAL step-200 composition against a
copy of the committed campaign records, then fixed. The pins:

* the staging tools stay runnable in a plain repo checkout: ``stratify_rung``
  used to import ``driver_progress`` for one six-line pure helper, which drags
  the reducer and numpy in behind it, so half the documented two-command
  staging sequence could not run where the other half could;
* the empty-stratum refusal's own printed remedy must actually work: the
  refusal says "acknowledge explicitly", and following that instruction has to
  converge instead of earning a second refusal about share sums;
* an acknowledgement is a CLAIM about the records state and is proven, never
  trusted: acknowledging a live stratum refuses;
* the oddball/lockless salting set is exactly the on-disk lockless intent set,
  so an intent gaining or losing a committed lock cannot silently change which
  surprise classes get salted; and
* the rung-boundary tally is phase-split, because a ``pytorch`` campaign's
  partition really does carry native-framework rows (c1-mech's 141
  ``paddle-arm64`` / ERNIE-class rows), so a staged rung owes TWO legs and an
  unsplit tally reads that as models gone missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

from menagerie.crawler import authority as authority_module
from menagerie.crawler.environment_coverage import (
    COVERAGE_DEFERRAL_DISPOSITION,
    COVERAGE_DEFERRAL_SCHEMA_VERSION,
    coverage_deferral_path,
)
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.intake import IntakeSnapshot, load_intake_snapshot
from menagerie.crawler.partitioner import emit_campaign_partitions, load_campaign_bindings
from menagerie.crawler.rung_staging import CARRIED_TIP_STRATUM

pytestmark = pytest.mark.smoke

_CAMPAIGN = "c1-mech"
_TARGET = "osx-arm64"

#: A roster whose c1-mech rows deliberately span both scheduler phases: the
#: ``discovered-paddle`` rows derive ``native-tail`` inside a ``pytorch``
#: campaign, exactly like the committed ERNIE / PaddleOCR rows.
_ROSTER_ROWS: tuple[dict[str, Any], ...] = (
    {"name": "conv-done", "zoo": "recbole-zoo", "variant": ""},
    {"name": "rec-shim", "zoo": "recbole-zoo", "variant": ""},
    {"name": "rec-fresh", "zoo": "recbole-zoo", "variant": ""},
    # Post-graduation (2026-08-19) lockless-routed rows: a ``flax`` marker ->
    # ``jax-flax-arm64``, a ``keras`` marker -> ``tf-keras-arm64``, both staying
    # in c1-mech exactly like the ``discovered-paddle`` rows below.
    {"name": "odd-repo", "zoo": "discovered-flax", "variant": ""},
    {"name": "mm-model", "zoo": "keras-hub", "variant": ""},
    # A LOCKED-family row (``open-mmlab`` -> ``mmlab``, pytorch phase): the
    # valve-parked prior-rung member must stay in the pytorch leg's accounting,
    # and post-graduation mmlab is exactly a locked family, not a salted one.
    {"name": "parked-model", "zoo": "open-mmlab", "variant": ""},
    {"name": "cuda-model", "zoo": "timm", "variant": ""},
    {"name": "refused-model", "zoo": "timm", "variant": ""},
    {"name": "hard-import", "zoo": "timm", "variant": ""},
    {"name": "hard-author", "zoo": "timm", "variant": ""},
    {"name": "nova-a1", "zoo": "zoo-nova-1", "variant": ""},
    {"name": "nova-b1", "zoo": "zoo-nova-2", "variant": ""},
    {"name": "ERNIE-Doc", "zoo": "discovered-paddle", "variant": ""},
    {"name": "ERNIE-Gram", "zoo": "discovered-paddle", "variant": ""},
    # Fresh c1-mech bulk so a size-10 rung has room after tail selection.
    *({"name": f"timm-bulk-{index}", "zoo": "timm", "variant": ""} for index in range(12)),
    # Filler rows so every frozen campaign has a non-empty partition.
    {"name": "disco-one", "zoo": "discovered-pytorch", "variant": ""},
    {"name": "classic-one", "zoo": "unregistered-classics-pytorch", "variant": ""},
    {"name": "native-one", "zoo": "discovered-keras", "variant": ""},
)

_PRIOR_RUNG_NAMES = (
    "conv-done",
    "rec-shim",
    "cuda-model",
    "refused-model",
    "hard-import",
    "hard-author",
    "parked-model",
    "ERNIE-Doc",
)

_TEST_SHARES = {
    "ladder-bulk": 0.4,
    "r2-vendored": 0.1,
    "oddball-intent": 0.1,
    "cuda-deferred": 0.1,
    "content-refusal": 0.1,
    "unsampled-zoo": 0.1,
    "hard-tail": 0.1,
}


def _records_root(repo_root: Path) -> Path:
    """Return the campaign records root for one repository."""

    return repo_root / "menagerie" / "crawler" / "records"


def _root_snapshot(repo_root: Path) -> IntakeSnapshot:
    """Load the c1-mech bound full-partition intake snapshot."""

    bindings = load_campaign_bindings(_records_root(repo_root) / "campaigns.json")
    binding = next(b for b in bindings if b.spec.campaign_id == _CAMPAIGN)
    return load_intake_snapshot(_records_root(repo_root) / binding.intake_path)


def _ids_by_name(repo_root: Path) -> dict[str, str]:
    """Map fixture model names to their intake stable ids."""

    return {item.name: item.stable_id for item in _root_snapshot(repo_root).items}


def _head_record(
    stable_id: str,
    status_code: str,
    *,
    rung: Optional[str] = None,
    author_sessions_used: int = 0,
    gate_rounds_used: int = 0,
) -> dict[str, Any]:
    """Build one minimal current model revision carrying a budget block.

    The status carries BOTH halves of the closed pair. The stratifier and the
    rung report key on ``code``, but the AW-2 award-authority door keys on
    ``kind``, so a head carrying only the code is an award holder no door can
    see -- the guard then passes vacuously over this whole fixture (the C-2
    sibling fix, applied here 2026-08-19).
    """

    body: dict[str, Any] = {
        "stable_id": stable_id,
        "record_seq": 1,
        "parent_revision": None,
        "status": {"kind": status_code.split(":", 1)[0], "code": status_code},
        "source_resolution": {"rung": rung or "NO_RUNG_SELECTED", "attempted_rungs": []},
        "budget": {
            "author_sessions_used": author_sessions_used,
            "author_sessions_max": 3,
            "gate_rounds_used": gate_rounds_used,
            "run_revisions_used": 0,
            "explicit_grants": [],
        },
    }
    body["record_revision"] = stable_hash(body)
    return body


def _refusal_attempt(stable_id: str) -> dict[str, Any]:
    """Build one attempt row carrying the TK1 refusal classification."""

    return {
        "stable_id": stable_id,
        "attempt_id": f"attempt-refused-{stable_id}",
        "stage": "author",
        "started_at": "2026-08-11T00:00:00Z",
        "finished_at": "2026-08-11T00:00:30Z",
        "error": {
            "stage": "author",
            "reason_code": "author-session-failure",
            "failure_class": "content-refusal",
            "failure_class_basis": "session-diagnostic",
            "refusal_excerpt": "I can't help with building this model.",
        },
    }


def _valve_row(stable_id: str, *, valid: bool = True) -> dict[str, Any]:
    """Build one coverage valve row, optionally with a broken content address."""

    body: dict[str, Any] = {
        "schema_version": COVERAGE_DEFERRAL_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "disposition": COVERAGE_DEFERRAL_DISPOSITION,
        "recheck_hint": "re-run this model routed to the 'tf-keras-arm64' environment intent",
    }
    body["row_sha256"] = stable_hash(body) if valid else "sha256:" + "0" * 64
    return body


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write canonical JSONL rows, creating parents."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows))


@pytest.fixture()
def ladder_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Materialize a campaign repository holding one registered prior rung.

    Returns
    -------
    pathlib.Path
        Repository root whose records hold the emitted partitions, a registered
        ``step-100`` rung spanning both phases, model heads with budgets, one
        refusal attempt, and one accountable coverage valve row.
    """

    from menagerie.crawler.rung_staging import CURRENCY_CARVE_OUT_ATTRIBUTE, stage_rung

    repo_root = tmp_path / "repo"
    records_root = _records_root(repo_root)
    roster_path = tmp_path / "roster.jsonl"
    roster_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in _ROSTER_ROWS))
    emit_campaign_partitions(roster_path, records_root)
    monkeypatch.setattr(authority_module, CURRENCY_CARVE_OUT_ATTRIBUTE, True, raising=False)
    ids = _ids_by_name(repo_root)
    # AW-2 fails CLOSED on an absent models ledger: the empty shard is the
    # explicit statement that no awards exist yet when the prior rung stages.
    _write_jsonl(records_root / "models" / "current-shard.jsonl", [])
    stage_rung(
        records_root,
        campaign_id=_CAMPAIGN,
        stable_ids=tuple(ids[name] for name in _PRIOR_RUNG_NAMES),
        count=None,
        derived_from_snapshot_id=None,
        label="step-100",
        registered_by="pytest",
        created_at="2026-08-11T00:00:00Z",
    )
    _write_jsonl(
        records_root / "models" / "current-shard.jsonl",
        [
            _head_record(ids["conv-done"], "runs", rung="R2_VENDOR", author_sessions_used=1),
            _head_record(
                ids["rec-shim"], "failed:import", rung="R2_VENDOR", author_sessions_used=3
            ),
            _head_record(ids["cuda-model"], "deferred:needs-cuda", author_sessions_used=1),
            _head_record(
                ids["refused-model"], "failed:author", author_sessions_used=2, gate_rounds_used=1
            ),
            _head_record(ids["hard-import"], "failed:import", author_sessions_used=1),
            _head_record(ids["hard-author"], "failed:author", author_sessions_used=2),
        ],
    )
    _write_jsonl(
        records_root / "attempts" / "local.jsonl", [_refusal_attempt(ids["refused-model"])]
    )
    _write_jsonl(coverage_deferral_path(records_root), [_valve_row(ids["parked-model"])])
    return repo_root


# ---------------------------------------------------------------------------
# Staging tools stay runnable in a plain checkout.
# ---------------------------------------------------------------------------


def test_framework_derivation_has_exactly_one_authority() -> None:
    """The driver's phase helper delegates to the import-light constant helper.

    Since BL-13 the derivation returns a CANONICAL framework spelling rather than the
    matched marker, so ``keras`` resolves to ``tensorflow`` and ``flax`` to ``jax``. The
    marker WAS returned verbatim as if it were a framework, which is how ``"keras"`` came to
    route to the TF lane while ``standard_inputs._convert_array`` raised on it and
    ``worker._seed_frameworks`` silently seeded nothing for it -- one producer string, three
    behaviours (AREA_BACKENDS_opus B1). One spelling per framework is also what keeps a
    spelling rotation out of the execution identity.
    """

    from menagerie.crawler.constants import (
        NATIVE_FRAMEWORK_MARKERS,
        framework_from_markers,
    )
    from menagerie.crawler.driver_progress import _framework_from_intake, _framework_phase
    from menagerie.crawler.intake import IntakeItem

    def item(name: str, zoo: str) -> IntakeItem:
        return IntakeItem(
            stable_id="m1",
            name=name,
            zoo=zoo,
            variant="",
            discovery_source="test",
            legacy_row_sha256="sha256:" + "0" * 64,
            preserved_legacy_flags=(),
        )

    canonical_for_marker = {
        "tensorflow": "tensorflow",
        "keras": "tensorflow",
        "jax": "jax",
        "flax": "jax",
        "paddle": "paddle",
    }
    assert set(canonical_for_marker) == set(NATIVE_FRAMEWORK_MARKERS)
    assert framework_from_markers("timm", "resnet") == "pytorch"
    for marker, canonical in canonical_for_marker.items():
        assert framework_from_markers(f"discovered-{marker}", "model") == canonical
        assert _framework_from_intake(item("model", f"discovered-{marker}")) == canonical
        assert _framework_phase(item("model", f"discovered-{marker}")) == "native-tail"
    assert _framework_phase(item("resnet", "timm")) == "pytorch"
    # The row's own zoo declaration outranks a framework name inside the MODEL name: the
    # measured 13% false-positive class (paddleseg_*/paddledet_* torch reimplementations in
    # zoo unregistered-classics-pytorch) must stay in the torch lane.
    assert framework_from_markers("unregistered-classics-pytorch", "paddleseg_stfpm") == "pytorch"
    assert _framework_phase(item("paddleseg_stfpm", "unregistered-classics-pytorch")) == "pytorch"


@pytest.mark.parametrize("tool", ["stratify_rung", "stage_rung", "rung_report"])
def test_staging_tools_import_without_the_fat_runtime(tool: str) -> None:
    """The read-only rung tools import with no scientific runtime installed.

    ``stratify_rung`` imported ``driver_progress`` for one pure string helper,
    and through it the reducer and ``numpy``, so half of the documented
    ``stratify_rung`` -> ``stage_rung`` sequence could not run in a checkout
    where the other half ran fine. Pinned as the property that actually matters
    -- a clean import with the heavy runtime BLOCKED -- rather than as a static
    graph shape, because the graph legitimately contains function-scoped heavy
    imports that never execute on these paths.
    """

    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    program = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return None\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'numpy', 'torch', 'scipy'}:\n"
        "            raise ImportError('blocked heavy runtime: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        f"import menagerie.crawler.tools.{tool} as tool\n"
        "assert tool.build_parser() is not None\n"
        "print('ok')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(repo_root), "PATH": "/usr/bin:/bin"},
    )
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), (
        f"tools.{tool} needs the fat runtime to import (move the shared helper "
        f"down, never widen the tool's dependencies):\n{done.stderr}"
    )


def test_operational_ledger_path_has_one_authority(tmp_path: Path) -> None:
    """The records-root and model-ledger spellings resolve to the same file."""

    from menagerie.crawler.checkpoint import canonical_operational_ledger_path
    from menagerie.crawler.recordio import operational_events_path

    records_root = tmp_path / "records"
    models_path = records_root / "models" / "current-shard.jsonl"
    models_path.parent.mkdir(parents=True)
    models_path.write_bytes(b"")
    assert canonical_operational_ledger_path(models_path) == operational_events_path(records_root)


# ---------------------------------------------------------------------------
# Empty-stratum acknowledgement: the printed remedy has to converge.
# ---------------------------------------------------------------------------


def _stratify(repo_root: Path, **kwargs: Any) -> Any:
    """Run the stratifier with the fixture's integer-friendly shares."""

    from menagerie.crawler.rung_stratification import stratify_rung_members

    return stratify_rung_members(
        _records_root(repo_root),
        campaign_id=_CAMPAIGN,
        size=kwargs.pop("size", 10),
        seed=kwargs.pop("seed", 7),
        shares=dict(_TEST_SHARES),
        **kwargs,
    )


def test_following_the_empty_stratum_remedy_converges(ladder_repo: Path) -> None:
    """Each refusal names a flag that works; the loop terminates and discloses."""

    from menagerie.crawler.rung_stratification import EmptyStratumError

    records_root = _records_root(ladder_repo)
    # Empty every head-derived tail stratum, leaving the refusal ladder to walk.
    _write_jsonl(records_root / "models" / "current-shard.jsonl", [])
    _write_jsonl(records_root / "attempts" / "local.jsonl", [])
    acknowledged: list[str] = []
    for _round in range(len(_TEST_SHARES) + 1):
        try:
            plan = _stratify(ladder_repo, acknowledged_empty_strata=tuple(acknowledged))
        except EmptyStratumError as exc:
            assert f"--acknowledge-empty-stratum {exc.stratum}" in str(exc)
            assert "must still sum to 1.0" in str(exc)
            acknowledged.append(exc.stratum)
            continue
        break
    else:  # pragma: no cover - a non-converging remedy is the bug this pins
        pytest.fail("the empty-stratum remedy did not converge")
    assert acknowledged, "the fixture must exercise at least one acknowledgement"
    assert set(plan.acknowledged_empty_strata) == set(acknowledged)
    assert plan.to_report()["acknowledged_empty_strata"] == list(plan.acknowledged_empty_strata)
    assert sum(plan.shares.values()) == pytest.approx(1.0)
    for name in acknowledged:
        assert plan.shares[name] == 0.0
        assert name not in {member.stratum for member in plan.members}
    # The reclaimed share buys COMPOSED width; the carried lineage tip (C-2) is
    # cumulative membership on top of it, not part of what --size bought.
    assert sum(1 for member in plan.members if member.stratum != CARRIED_TIP_STRATUM) == 10


def test_the_composer_discloses_the_carried_awards_authority(ladder_repo: Path) -> None:
    """A carried runs holder arrives with the exact authority the door demands.

    The fixture's ``conv-done`` head holds a ``runs`` award, so a cumulative
    successor carries it (C-2) and the AW-2 door demands an authority naming
    its exact current revision. The composer's disclosure is derived with the
    DOOR's own predicate, which keys on ``status.kind`` -- not
    ``status.code`` -- so a fixture head carrying only the code makes the
    disclosure read ``{}`` and the door see ZERO award holders, and every
    composition test in this module passes with the guard silent. Pins the
    disclosure NON-EMPTY, which is what fails when the fixture goes blind.
    """

    from menagerie.crawler.rung_staging import required_award_authorities

    ids = _ids_by_name(ladder_repo)
    plan = _stratify(ladder_repo)
    member_ids = tuple(member.stable_id for member in plan.members)
    assert ids["conv-done"] in member_ids, "the carried tip must contain the award holder"
    demanded = required_award_authorities(_records_root(ladder_repo), member_ids)
    assert set(demanded) == {ids["conv-done"]}, (
        "the AW-2 demand went silent over a runs head: the door reads status.kind"
    )
    assert plan.award_authorities_required == demanded, (
        "the composer's disclosure must be the door's own demand, never a hand tally"
    )


def test_acknowledging_a_live_stratum_refuses(ladder_repo: Path) -> None:
    """An acknowledgement is proven against the records state, never trusted."""

    from menagerie.crawler.rung_stratification import StratumNotEmptyError

    with pytest.raises(StratumNotEmptyError) as excinfo:
        _stratify(ladder_repo, acknowledged_empty_strata=("cuda-deferred",))
    assert excinfo.value.stratum == "cuda-deferred"
    assert excinfo.value.pool_size >= 1


def test_ladder_bulk_is_not_acknowledgeable(ladder_repo: Path) -> None:
    """An exhausted partition is a rung-sizing refusal, not a vanished class."""

    from menagerie.crawler.rung_stratification import StratificationError

    with pytest.raises(StratificationError, match="rung-sizing refusal"):
        _stratify(ladder_repo, acknowledged_empty_strata=("ladder-bulk",))
    with pytest.raises(StratificationError, match="acknowledge"):
        _stratify(ladder_repo, acknowledged_empty_strata=("no-such-stratum",))


def test_cli_exposes_the_acknowledgement_flag(ladder_repo: Path, tmp_path: Path) -> None:
    """``--acknowledge-empty-stratum`` reaches the composer and the report."""

    from menagerie.crawler.tools.stratify_rung import main

    records_root = _records_root(ladder_repo)
    _write_jsonl(records_root / "models" / "current-shard.jsonl", [])
    _write_jsonl(records_root / "attempts" / "local.jsonl", [])
    report_path = tmp_path / "report.json"
    exit_code = main(
        [
            "--repo-root",
            str(ladder_repo),
            "--campaign",
            _CAMPAIGN,
            "--size",
            "10",
            "--seed",
            "7",
            "--out-members",
            str(tmp_path / "members.txt"),
            "--out-report",
            str(report_path),
            "--acknowledge-empty-stratum",
            "r2-vendored",
            "--acknowledge-empty-stratum",
            "cuda-deferred",
            "--acknowledge-empty-stratum",
            "content-refusal",
            "--acknowledge-empty-stratum",
            "hard-tail",
        ]
    )
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["acknowledged_empty_strata"] == [
        "r2-vendored",
        "cuda-deferred",
        "content-refusal",
        "hard-tail",
    ]
    # Default shares: bulk 0.70 reclaims four 0.05 tail shares.
    assert report["shares"]["ladder-bulk"] == pytest.approx(0.90)
    assert sum(report["shares"].values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The salted oddball set is the on-disk lockless intent set.
# ---------------------------------------------------------------------------


def test_salted_oddball_intents_match_the_lockless_intents_on_disk() -> None:
    """A lock landing or vanishing must not silently change what gets salted.

    ``DEFAULT_ODDBALL_INTENTS`` is a hand-maintained literal that duplicates a
    fact the repository already carries: whether the intent has committed
    setup artifacts. ``paddle-arm64`` is the interesting case -- a native-tail
    intent that DOES have a lock, so it is deliberately not in the salted set.
    """

    from menagerie.crawler.rung_stratification import DEFAULT_ODDBALL_INTENTS

    envs_root = Path(__file__).resolve().parents[1] / "envs"
    registry = json.loads((envs_root / "registry.yml").read_text(encoding="utf-8"))
    lockless = {
        intent
        for intent in registry["intents"]
        if not (envs_root / intent / "locks" / f"{_TARGET}.lock").is_file()
    }
    assert set(DEFAULT_ODDBALL_INTENTS) == lockless, (
        "the salted oddball/lockless set drifted from the committed setup "
        "artifacts; re-adjudicate the salting, do not just edit the literal"
    )
    assert "paddle-arm64" not in DEFAULT_ODDBALL_INTENTS


# ---------------------------------------------------------------------------
# The rung-boundary report.
# ---------------------------------------------------------------------------


def test_rung_report_splits_phases_and_names_the_legs_owed(ladder_repo: Path) -> None:
    """A pytorch campaign's rung carrying native rows owes two legs, plainly."""

    from menagerie.crawler.tools.rung_report import build_rung_report

    report = build_rung_report(_records_root(ladder_repo), campaign_id=_CAMPAIGN, label="step-100")
    assert report["rung"]["label"] == "step-100"
    assert report["rung"]["member_count"] == len(_PRIOR_RUNG_NAMES)
    assert set(report["phases"]) == {"pytorch", "native-tail"}
    native = report["phases"]["native-tail"]
    assert native["members"] == 1, "the ERNIE-class row is scoped to the native leg"
    assert native["closable"] is False and native["unaccounted"] == 1
    pytorch = report["phases"]["pytorch"]
    assert pytorch["converted"] == 1
    assert pytorch["valve_claimed"] == 1
    assert pytorch["by_status"] == {
        "deferred:needs-cuda": 1,
        "failed:author": 2,
        "failed:import": 2,
        "runs": 1,
    }
    assert pytorch["closable"] is True
    assert report["legs_owed"] == ["native-tail"]


def test_rung_report_totals_the_spend_the_rung_actually_bought(ladder_repo: Path) -> None:
    """Author sessions, gate rounds, and attempt counts come from rung records."""

    from menagerie.crawler.tools.rung_report import build_rung_report

    spend = build_rung_report(_records_root(ladder_repo), campaign_id=_CAMPAIGN, label="step-100")[
        "spend"
    ]
    assert spend["author_sessions_used"] == 10
    assert spend["gate_rounds_used"] == 1
    assert spend["heads_without_budget"] == 0
    assert spend["attempts_by_stage"] == {"author": 1}
    assert spend["attempt_wall_seconds_by_stage"]["author"] == {
        "count": 1,
        "total_seconds": 30.0,
    }
    assert spend["checker"] == {"sessions_bought": 0, "verdicts_reused": 0}


def test_rung_report_segregates_a_rejected_valve_row(ladder_repo: Path) -> None:
    """A row the completion partition rejects accounts for nothing here either."""

    from menagerie.crawler.tools.rung_report import build_rung_report

    records_root = _records_root(ladder_repo)
    ids = _ids_by_name(ladder_repo)
    _write_jsonl(
        coverage_deferral_path(records_root),
        [_valve_row(ids["parked-model"], valid=False)],
    )
    report = build_rung_report(records_root, campaign_id=_CAMPAIGN, label="step-100")
    coverage = report["valves"]["environment-coverage"]
    assert coverage["accountable_members"] == 0
    assert coverage["rejected_rows"] == 1
    assert coverage["rejected_detail"][0]["stable_id"] == ids["parked-model"]
    pytorch = report["phases"]["pytorch"]
    assert pytorch["valve_claimed"] == 0
    assert pytorch["closable"] is False, "an unaccounted member must block the close"


def test_rung_certification_refuses_heterogeneous_config_digests(
    ladder_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D-63 (PM-1): two rung-invariant configs in one rung refuse certification.

    200a ran legs 23-43 at a width its ledger did not record; with the
    rung-invariant digest attested per leg, a rung whose legs disagree is a
    typed certification refusal, and pre-fix history rows are disclosed
    rather than counted heterogeneous.
    """

    from menagerie.crawler.recordio import operational_events_path
    from menagerie.crawler.tools.rung_report import build_rung_report, main

    records_root = _records_root(ladder_repo)
    operational_path = operational_events_path(records_root)

    def _leg_config_event(at: str, digest: object) -> dict[str, Any]:
        details: dict[str, Any] = {"campaign_id": _CAMPAIGN}
        if digest is not None:
            details["rung_config_digest"] = digest
        return {
            "event_kind": "leg-config",
            "created_at": at,
            "details": details,
        }

    existing = (
        operational_path.read_text(encoding="utf-8") if operational_path.is_file() else ""
    )
    rows = [
        # Pre-registration leg: outside the rung epoch, never counted.
        _leg_config_event("2026-08-10T00:00:00Z", "sha256:" + "a" * 64),
        # Pre-fix history inside the epoch: disclosed, not heterogeneous.
        _leg_config_event("2026-08-12T00:00:00Z", None),
        _leg_config_event("2026-08-12T01:00:00Z", "sha256:" + "b" * 64),
    ]
    operational_path.parent.mkdir(parents=True, exist_ok=True)
    operational_path.write_text(
        existing + "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    homogeneous = build_rung_report(records_root, campaign_id=_CAMPAIGN, label="step-100")
    census = homogeneous["config_homogeneity"]
    assert census["homogeneous"] is True
    assert census["pre_fix_legs"] == 1
    assert census["digests"] == {"sha256:" + "b" * 64: 1}

    with operational_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_leg_config_event("2026-08-12T02:00:00Z", "sha256:" + "c" * 64)) + "\n"
        )
    heterogeneous = build_rung_report(records_root, campaign_id=_CAMPAIGN, label="step-100")
    assert heterogeneous["config_homogeneity"]["homogeneous"] is False

    # The CLI is the certification surface: heterogeneity exits nonzero.
    exit_code = main(
        ["--repo-root", str(ladder_repo), "--campaign", _CAMPAIGN, "--label", "step-100"]
    )
    capsys.readouterr()
    assert exit_code == 2


def test_rung_report_refuses_unknown_and_unverifiable_rungs(ladder_repo: Path) -> None:
    """Selector and membership-verification refusals are typed and specific."""

    from menagerie.crawler.rung_staging import load_rung_registrations
    from menagerie.crawler.tools.rung_report import RungReportError, build_rung_report

    records_root = _records_root(ladder_repo)
    with pytest.raises(RungReportError, match="no c1-mech rung matches"):
        build_rung_report(records_root, campaign_id=_CAMPAIGN, label="step-999")
    with pytest.raises(RungReportError, match="exactly one of --label or --snapshot-id"):
        build_rung_report(records_root, campaign_id=_CAMPAIGN)
    registration = next(iter(load_rung_registrations(records_root)))
    items_path = records_root / "intake" / registration.snapshot_id / "items.jsonl"
    kept = items_path.read_bytes().splitlines(keepends=True)[:-1]
    items_path.write_bytes(b"".join(kept))
    with pytest.raises(RungReportError, match="unverifiable membership|cannot be loaded"):
        build_rung_report(records_root, campaign_id=_CAMPAIGN, label="step-100")


def test_rung_report_cli_writes_the_report(ladder_repo: Path, tmp_path: Path) -> None:
    """The tool prints and optionally writes the canonical report JSON."""

    from menagerie.crawler.tools.rung_report import RUNG_REPORT_SCHEMA_VERSION, main

    output = tmp_path / "rung.json"
    exit_code = main(
        [
            "--repo-root",
            str(ladder_repo),
            "--campaign",
            _CAMPAIGN,
            "--label",
            "step-100",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == RUNG_REPORT_SCHEMA_VERSION
    assert report["records_state"]["models_ledger_sha256"].startswith("sha256:")
    assert main(["--repo-root", str(ladder_repo), "--campaign", _CAMPAIGN, "--label", "nope"]) == 2
