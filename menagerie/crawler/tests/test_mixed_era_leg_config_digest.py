"""AB-30 matrix: leg-config-digest-identity seam (D-63, lane w2-scheduler).

The Wave-2 batch teaches every NEW leg-config event five fields --
``author_concurrency_source``, ``checker_concurrency_source``,
``repo_tree_sha``, ``repo_tree_dirty``, and ``rung_config_digest`` -- and
makes ``rung_config_digest`` a derived config identity over exactly the
closed ``RUNG_INVARIANT_CONFIG_FIELDS``. Rung certification
(``tools/rung_report.py``, schema v1 -> v2 with ``config_homogeneity``)
REFUSES a rung whose post-era legs disagree on that digest, while PRE-era
legs (events without ``repo_tree_sha`` / ``rung_config_digest``) are
era-scoped: DISCLOSED under ``pre_fix_legs``, never counted heterogeneous,
never bricking certification of pre-fix history.

This seam is declared ``owed-by-batch-lane`` (owed_by D-63) in
``wave2_batch_declarations.json`` with no frozen fixture of its own, so the
prior-era artifact is constructed here per the seam's own definition -- a
LEG_CONFIG operational event row WITHOUT the five D-63 fields, byte-shaped
like the 200a-era emitter's output -- and read through the REAL certification
reader (``build_rung_report`` + the CLI ``main``), never a reimplementation.

Refusal surface: the library-level reader PARSES a heterogeneous rung and
discloses ``homogeneous: false`` in the report (the section is present either
way); the certification REFUSAL is the documented CLI exit-2 contract, and
both halves are asserted here.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    InvocationOrigin,
)
from menagerie.crawler.driver import RUNG_INVARIANT_CONFIG_FIELDS
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.recordio import JsonlLedger, operational_events_path, scan_jsonl
from menagerie.crawler.tests.mixed_era import assert_era_outcome, classify_reader

# House precedent (test_leg_config_event.py): sibling test-module helpers are
# the REAL fixture producers for these seams -- the ladder repository's roster
# and record writers, and the fully fake deterministic driver whose
# ``_record_leg_config`` is the one production emitter of leg-config events.
from menagerie.crawler.tests.test_rung_boundary_protocol import (
    _CAMPAIGN,
    _ROSTER_ROWS,
    _ids_by_name,
    _records_root,
    _write_jsonl,
)
from menagerie.crawler.tests.test_slice_f_driver import _driver, _paths, _snapshot
from menagerie.crawler.tools.rung_report import (
    RUNG_REPORT_SCHEMA_VERSION,
    build_rung_report,
    main,
)

pytestmark = pytest.mark.smoke

_READER_ID = "rung-certification-report"
#: The ladder fixture registers step-100 at this instant; the census scopes
#: leg-config events to the rung epoch by comparing against it.
_REGISTRATION_AT = "2026-08-11T00:00:00Z"

#: The five fields D-63 adds to every NEW leg-config event. Their ABSENCE is
#: what makes a row prior-era for this seam.
_D63_FIELDS = (
    "author_concurrency_source",
    "checker_concurrency_source",
    "repo_tree_sha",
    "repo_tree_dirty",
    "rung_config_digest",
)


@pytest.fixture()
def ladder_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Materialize a campaign repository with one registered step-100 rung.

    The same REAL staging machinery the ladder-repo fixture in
    ``test_rung_boundary_protocol`` uses (partition emission + ``stage_rung``
    over the shared roster), scoped down to exactly what the certification
    census needs: a verifiable registered rung and an empty models shard.
    """

    from menagerie.crawler import authority as authority_module
    from menagerie.crawler.identity import canonical_json_bytes
    from menagerie.crawler.partitioner import emit_campaign_partitions
    from menagerie.crawler.rung_staging import CURRENCY_CARVE_OUT_ATTRIBUTE, stage_rung

    repo_root = tmp_path / "repo"
    records_root = _records_root(repo_root)
    roster_path = tmp_path / "roster.jsonl"
    roster_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in _ROSTER_ROWS))
    emit_campaign_partitions(roster_path, records_root)
    monkeypatch.setattr(authority_module, CURRENCY_CARVE_OUT_ATTRIBUTE, True, raising=False)
    ids = _ids_by_name(repo_root)
    # AW-2 fails CLOSED on an absent models ledger: the empty shard states
    # that no awards exist yet when the rung stages.
    _write_jsonl(records_root / "models" / "current-shard.jsonl", [])
    stage_rung(
        records_root,
        campaign_id=_CAMPAIGN,
        stable_ids=(ids["cuda-model"], ids["refused-model"]),
        count=None,
        derived_from_snapshot_id=None,
        label="step-100",
        registered_by="pytest",
        created_at=_REGISTRATION_AT,
    )
    return repo_root


def _prior_era_details() -> dict[str, Any]:
    """One 200a-era leg-config details block: full config, no D-63 fields."""

    return {
        "campaign_id": _CAMPAIGN,
        "scheduler": "waves",
        "author_concurrency": 12,
        "checker_concurrency": 4,
        "author_model": "fable",
        "checker_model": "gpt-5.6-sol",
        "only_unsettled": False,
        "config_digest": "sha256:" + "d" * 64,
    }


def _post_era_details(rung_config_digest: str) -> dict[str, Any]:
    """One post-D-63 leg-config details block carrying the five new fields."""

    return {
        **_prior_era_details(),
        "author_concurrency_source": "default",
        "checker_concurrency_source": "default",
        "repo_tree_sha": "a" * 40,
        "repo_tree_dirty": False,
        "rung_config_digest": rung_config_digest,
    }


def _leg_config_event(created_at: str, details: Mapping[str, Any]) -> dict[str, Any]:
    return {"event_kind": "leg-config", "created_at": created_at, "details": dict(details)}


def _append_events(records_root: Path, rows: list[Mapping[str, Any]]) -> None:
    """Append raw rows to the operational ledger, preserving existing bytes."""

    path = operational_events_path(records_root)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        existing + "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _certify(records_root: Path) -> tuple[str, Any]:
    """Run the REAL library-level certification reader, classified."""

    return classify_reader(
        lambda: build_rung_report(records_root, campaign_id=_CAMPAIGN, label="step-100")
    )


def _cli_exit(ladder_repo: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    """Run the certification CLI surface; return (exit_code, stderr)."""

    code = main(["--repo-root", str(ladder_repo), "--campaign", _CAMPAIGN, "--label", "step-100"])
    captured = capsys.readouterr()
    return code, captured.err


def _emit_leg_config(driver: Any, paths: Any) -> Mapping[str, Any]:
    """Append one leg-config event through the REAL production emitter."""

    with JsonlLedger(paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
        driver._record_leg_config(operational)  # noqa: SLF001 -- the production emitter under test
    events = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("event_kind") == "leg-config"
    ]
    return events[-1]["details"]


# ---------------------------------------------------------------------------
# Cell 1: a prior-era LEG_CONFIG row parses era-scoped.
# ---------------------------------------------------------------------------


def test_prior_era_leg_config_event_parses_era_scoped(
    ladder_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pre-D-63 row is disclosed under ``pre_fix_legs``, never a refusal.

    Two era boundaries in one artifact: a row BEFORE the rung registration is
    outside the rung epoch entirely (not even a leg of this rung), and a row
    inside the epoch WITHOUT ``repo_tree_sha``/``rung_config_digest`` is
    pre-fix history -- disclosed, counted, and never heterogeneous.
    Certification of pre-fix history is not bricked: the CLI exits 0.
    """

    records_root = _records_root(ladder_repo)
    _append_events(
        records_root,
        [
            # Outside the rung epoch: never counted, digest or not.
            _leg_config_event(
                "2026-08-01T00:00:00Z", _post_era_details("sha256:" + "9" * 64)
            ),
            # Inside the epoch, pre-fix shape: the seam's prior-era artifact.
            _leg_config_event("2026-08-12T00:00:00Z", _prior_era_details()),
        ],
    )

    outcome = _certify(records_root)
    assert_era_outcome(reader_id=_READER_ID, outcome=outcome, prebatch_expected="parsed")
    report = outcome[1]
    assert report["schema_version"] == RUNG_REPORT_SCHEMA_VERSION
    assert report["config_homogeneity"] == {
        "digests": {},
        "pre_fix_legs": 1,
        "legs": 1,
        "homogeneous": True,
    }

    exit_code, stderr = _cli_exit(ladder_repo, capsys)
    assert exit_code == 0, f"pre-fix history bricked certification: {stderr}"


# ---------------------------------------------------------------------------
# Cell 2: homogeneous post-era digests certify, census recorded.
# ---------------------------------------------------------------------------


def test_post_era_homogeneous_digests_certify_with_census_recorded(
    ladder_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Legs agreeing on one rung-invariant digest pass certification."""

    records_root = _records_root(ladder_repo)
    digest = "sha256:" + "b" * 64
    _append_events(
        records_root,
        [
            _leg_config_event("2026-08-12T00:00:00Z", _post_era_details(digest)),
            _leg_config_event("2026-08-12T01:00:00Z", _post_era_details(digest)),
        ],
    )

    outcome = _certify(records_root)
    assert_era_outcome(reader_id=_READER_ID, outcome=outcome, prebatch_expected="parsed")
    census = outcome[1]["config_homogeneity"]
    assert census == {
        "digests": {digest: 2},
        "pre_fix_legs": 0,
        "legs": 2,
        "homogeneous": True,
    }

    exit_code, stderr = _cli_exit(ladder_repo, capsys)
    assert exit_code == 0, f"a homogeneous rung must certify: {stderr}"


# ---------------------------------------------------------------------------
# Cell 3: heterogeneous post-era digests refuse certification.
# ---------------------------------------------------------------------------


def test_post_era_heterogeneous_digests_refuse_certification(
    ladder_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two rung-invariant configs in one rung are a certification refusal.

    The library-level reader exposes the refusal as the DISCLOSED
    ``homogeneous: false`` census (the report builds either way, so the
    heterogeneity is auditable), and the certification surface -- the CLI --
    refuses with the documented exit-2 contract naming both digests. A
    pre-fix row alongside them stays disclosed and never masks or joins
    the heterogeneity.
    """

    records_root = _records_root(ladder_repo)
    digest_one = "sha256:" + "b" * 64
    digest_two = "sha256:" + "c" * 64
    _append_events(
        records_root,
        [
            _leg_config_event("2026-08-12T00:00:00Z", _post_era_details(digest_one)),
            _leg_config_event("2026-08-12T01:00:00Z", _post_era_details(digest_two)),
            _leg_config_event("2026-08-12T02:00:00Z", _prior_era_details()),
        ],
    )

    outcome = _certify(records_root)
    assert outcome[0] == "parsed", "the census itself must stay readable for the audit"
    census = outcome[1]["config_homogeneity"]
    assert census["homogeneous"] is False
    assert census["digests"] == {digest_one: 1, digest_two: 1}
    assert census["pre_fix_legs"] == 1

    exit_code, stderr = _cli_exit(ladder_repo, capsys)
    assert exit_code == 2, "heterogeneous rung digests must refuse certification"
    assert "heterogeneous rung_config_digest" in stderr
    assert digest_one in stderr and digest_two in stderr


# ---------------------------------------------------------------------------
# Cell 4: the digest is a pure function of the closed invariant fields.
# ---------------------------------------------------------------------------


def test_rung_config_digest_is_pure_over_the_closed_invariant_fields(
    tmp_path: Path,
) -> None:
    """Same rung-invariant inputs -> same digest; a changed field moves it.

    Exercised through the REAL producer (``driver._record_leg_config`` via the
    schema-validating ledger writer), and pinned to its declared derivation:
    the digest is ``stable_hash`` over EXACTLY the closed
    ``RUNG_INVARIANT_CONFIG_FIELDS`` projection of the emitted details --
    nothing more, nothing less.
    """

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)

    first = _emit_leg_config(driver, paths)
    second = _emit_leg_config(driver, paths)
    assert first["rung_config_digest"] == second["rung_config_digest"], (
        "the digest must be a deterministic function of the config, "
        "never per-emission entropy"
    )
    # The declared derivation, recomputed from the emitted row itself.
    assert first["rung_config_digest"] == stable_hash(
        {field: first[field] for field in RUNG_INVARIANT_CONFIG_FIELDS}
    )

    # A rung-invariant axis moves the digest.
    driver.config = replace(
        driver.config, checker_concurrency=driver.config.checker_concurrency + 1
    )
    widened = _emit_leg_config(driver, paths)
    assert widened["rung_config_digest"] != first["rung_config_digest"]
    # Moving it BACK restores the identity -- pure function of the closed
    # fields, not emission history. The restored leg differs on a PER-LEG
    # axis (a wake resume) so it is a genuinely new ledger occurrence
    # (identical re-emissions dedup under the frozen test clock), yet its
    # rung-invariant identity equals the first leg's.
    driver.config = replace(
        driver.config,
        checker_concurrency=driver.config.checker_concurrency - 1,
        invocation_origin=InvocationOrigin.WAKE_CALLBACK,
        wake_episode_id="wake-episode-restore",
    )
    restored = _emit_leg_config(driver, paths)
    assert restored["config_digest"] != first["config_digest"]
    assert restored["rung_config_digest"] == first["rung_config_digest"]


def test_new_leg_config_fields_ship_and_the_invariant_set_is_closed(
    tmp_path: Path,
) -> None:
    """Every new leg emits the five D-63 fields; the closed set stays closed.

    The seam's declared change in one census: a fresh production emission
    carries all five new fields, the tree identity is IN the closed
    rung-invariant set (two trees can never certify one rung), and the width
    PROVENANCE sources are deliberately OUT of it (an explicit override
    landing on the same width is homogeneous; a different width is caught by
    the width axis itself).
    """

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    details = _emit_leg_config(_driver(tmp_path, snapshot), paths)

    for field in _D63_FIELDS:
        assert field in details, f"new legs must emit {field!r}"
    assert details["rung_config_digest"].startswith("sha256:")

    assert "repo_tree_sha" in RUNG_INVARIANT_CONFIG_FIELDS
    assert "author_concurrency_source" not in RUNG_INVARIANT_CONFIG_FIELDS
    assert "checker_concurrency_source" not in RUNG_INVARIANT_CONFIG_FIELDS
    assert "repo_tree_dirty" not in RUNG_INVARIANT_CONFIG_FIELDS
    # Every closed field is present on the emitted row, so the digest
    # projection in the purity test above is total, never defaulted.
    missing = [field for field in RUNG_INVARIANT_CONFIG_FIELDS if field not in details]
    assert not missing, f"closed invariant fields absent from the emission: {missing}"
