"""Session-death injection rehearsal: fail a paid author session AFTER reserving.

Launch-night lesson (2026-08-19): the acceptance dry run was the only end-to-end
rehearsal vehicle, and it structurally could not fail a paid session after
reserving -- the one failure injection raised before any reservation existed. The
reserve -> custody bind -> die -> retry composition therefore first executed in
production, where the retry ladder's reservation-reuse bug bricked the step-200
spend ledger with a duplicate-custody row. This module drives the real subprocess
dry-run composition with ``MENAGERIE_DRY_RUN_INJECT_SESSION_DEATH`` and asserts
the post-fix custody contract end to end: the dead attempt's reservations custody
bind exactly once each and settle, the retry buys a NEW attempt head with a fresh
spend id and provider idempotency key, the models still complete, and the spend
ledger folds clean (no duplicate custody, no orphan reservation).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Mapping, cast

import pytest

from menagerie.crawler import driver_admission as driver_admission_module
from menagerie.crawler.authority import build_authority_context
from menagerie.crawler.cli import EXIT_OK, _persisted_environment_generations
from menagerie.crawler.driver import DriverConfig
from menagerie.crawler.driver_contracts import WorkItem
from menagerie.crawler.paid_session_lifecycle import (
    BudgetPolicy,
    PaidSessionAuthority,
    ProviderReconciliation,
    Reservation,
    load_paid_session_rows,
)
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.reducer import materialize_current
from menagerie.crawler.status import funnel_counts
from menagerie.crawler.tests.conftest import RealEnvironmentFixture
from menagerie.crawler.tests.dry_run_support import (
    DRY_RUN_ITEMS,
    SessionDeathAuthor,
    SessionDeathPlan,
    create_dry_run_snapshot,
    dry_run_paths,
    parse_session_death_plan,
)

#: One death per targeted intake name keeps the rehearsal deterministic under the
#: author wave's concurrency: each family's first-dispatched item consumes its own
#: budget, its retry cannot (the budget is spent), and the two deaths land on two
#: distinct stable ids.
_INJECTED_DEATHS = "DryRunMLP:1,DryRunStableConvNet:1"
_EXPECTED_DEATHS = 2


def _dry_run_command(
    repo_root: Path,
    campaign_root: Path,
    environment_prefix: Path,
    *,
    inject_session_death: str,
) -> subprocess.CompletedProcess[str]:
    """Run one documented dry-run ``run`` command with the session-death injection.

    Parameters
    ----------
    repo_root, campaign_root, environment_prefix:
        Checked-out source, disposable campaign, and strictly bound real
        environment roots.
    inject_session_death:
        Verbatim ``MENAGERIE_DRY_RUN_INJECT_SESSION_DEATH`` value for the child.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Captured real CLI process result.
    """

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root)
    environment["MENAGERIE_RELEASE_GATE"] = "1"
    environment["MENAGERIE_DRY_RUN_INJECT_SESSION_DEATH"] = inject_session_death
    return subprocess.run(
        [
            str(environment_prefix / "bin" / "python"),
            "-B",
            "-m",
            "menagerie.crawler",
            "--repo-root",
            str(repo_root),
            "run",
            "--dry-run",
            "--dry-run-root",
            str(campaign_root),
            "--dry-run-environment-prefix",
            str(environment_prefix),
            "--review-checkpoint-at",
            "0",
            "--progress-milestones",
            "",
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )


def _json_output(completed: subprocess.CompletedProcess[str]) -> Mapping[str, Any]:
    """Decode the final structured stdout line from one CLI subprocess."""

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, completed.stderr
    value = json.loads(lines[-1])
    assert isinstance(value, dict)
    return value


def _current_projection(campaign_root: Path) -> Mapping[str, Mapping[str, Any]]:
    """Materialize the authenticated current projection for one dry-run campaign."""

    snapshot = create_dry_run_snapshot(campaign_root)
    paths = dry_run_paths(campaign_root, snapshot)
    config = DriverConfig(
        target="osx-arm64",
        run_id="run-dry-rehearsal",
        author_model="fake-claude",
        author_version="dry-run",
        checker_model="fake-codex",
        checker_version="dry-run",
    )
    context = build_authority_context(
        active_intake_snapshot_id=snapshot.snapshot_id,
        active_intake_snapshot_sha256=snapshot.snapshot_sha256,
        intake_rows=(item.to_dict() for item in snapshot.items),
        author_model=config.author_model,
        author_version=config.author_version,
        checker_model=config.checker_model,
        checker_version=config.checker_version,
    )
    attempts = tuple(scan_jsonl(paths.ledgers.attempts))
    context = replace(
        context,
        environment_generations=_persisted_environment_generations(attempts),
    )
    return materialize_current(paths.ledgers, context=context)


class _InjectionLadderRig(driver_admission_module.AdmissionEnvironmentMixin):
    """The exact production seams one author-lane retry ladder call touches.

    ``_retry_infrastructure_call``, ``_settle_paid_session``, and the paid
    dispatch scope are inherited VERBATIM; only admission plumbing that reads
    driver configuration is replaced with a real reservation against a real
    money authority. Deliberately self-contained (no import from the
    retry-custody suite, whose module-level imports name post-fix constants):
    this module must still import against a locally reverted pre-fix tree so
    the RED reproduction fails on the duplicate-custody refusal, never on an
    ``ImportError``.
    """

    def __init__(self, authority: PaidSessionAuthority) -> None:
        """Bind one real money authority behind the real settlement method."""

        self._authority = authority
        self.waited: list[float] = []
        self.dependencies = SimpleNamespace(
            boundary_hook=lambda *_args: None,
            provider_reconciler=lambda key: ProviderReconciliation(
                True, None, f"sha256:receipt-{key[-12:]}"
            ),
            monotonic=time.monotonic,
        )

    def _check_shutdown(self, *_args: object, **_kwargs: object) -> None:
        """No shutdown is pending in this rig."""

    def _paid_session_authority(self) -> PaidSessionAuthority:
        """Return the one real money authority."""

        return self._authority

    def _admit_author_spend(
        self, item: Any, _operational: Any, attempt_group: str | None = None
    ) -> Reservation:
        """Reserve one author guard session exactly as production admission does."""

        return self._authority.reserve(
            scope={
                "campaign_id": "c1-mech",
                "generation_id": "generation-open",
                "lane": "author",
                "dispatch_kind": "author_source",
                "attempt_group": attempt_group,
                "work_generations": ((item.stable_id, item.active_work_id),),
                "gate_kind": None,
                "checker_identity": None,
                "run_id": "run-one",
                "machine_id": "host-one",
            },
            policy=BudgetPolicy(
                quota_sessions=100,
                quota_wall_seconds=100_000.0,
                runaway_sessions=100,
                runaway_wall_seconds=100_000.0,
                reservation_wall_seconds=100.0,
            ),
        )

    def _wait_before_infrastructure_retry(self, seconds: float) -> None:
        """Record the ladder's wait instead of sleeping."""

        self.waited.append(seconds)

    def _author_no_publication_class(self, _exc: BaseException) -> bool:
        """Keep exhaustion wrapping out of this rig."""

        return False

    def _note_author_lane_publication(self, _stable_id: str) -> None:
        """No lane-health bookkeeping in this rig."""


def test_session_death_injection_dies_on_the_classified_transport_path(
    tmp_path: Path,
) -> None:
    """The injected death takes the real ladder path: settle, retry, fresh head.

    This ladder-level rehearsal is the in-lane RED->GREEN vehicle: it drives
    the injection's exact seams (budget consumption, then the two real paid
    guard subprocesses with the death classified by ``classify_author_exit``)
    through the verbatim production ``_retry_infrastructure_call`` against a
    real anchored spend ledger, with no OS-sandboxed forward required. Against
    a pre-fix tree the second guard rebinds the first reservation's custody
    and this test fails on the launch-night ``duplicate custody`` refusal.
    """

    authority = PaidSessionAuthority(
        tmp_path / "records" / "operational" / "spend.jsonl",
        runtime_root=tmp_path / ".crawl-local",
        clock=lambda: "2026-08-20T03:45:26Z",
    )
    rig = _InjectionLadderRig(authority)
    item = cast(
        WorkItem,
        SimpleNamespace(
            stable_id="m_injected_death",
            active_work_id="work-1",
            granted_attempts=None,
            intake=SimpleNamespace(name="m_injected_death"),
        ),
    )
    author = SessionDeathAuthor(parse_session_death_plan(f"{item.stable_id}:1"))

    def operation() -> str:
        """One author attempt through the injection seams, minus the artifact."""

        if author._consume_death(item):  # noqa: SLF001 -- the seam under test
            author._die_in_author_guard_session(item)  # noqa: SLF001
        return "authored"

    result = rig._retry_infrastructure_call(  # noqa: SLF001
        operation,
        admission=("author", item),
        operational=cast(JsonlLedger, SimpleNamespace(records=())),
    )
    assert result == "authored"
    assert rig.waited, "the injected death must go through the ladder's bounded backoff"

    rows = load_paid_session_rows(authority.path)
    events = [(str(row["event"]), str(row.get("spend_id", ""))) for row in rows]
    spend_ids = [spend for event, spend in events if event == "session_reserved"]
    assert len(spend_ids) == 3 == len(set(spend_ids)), (
        "expected source guard, dying author guard, and retry reservations"
    )
    source_guard, author_guard, retry = spend_ids
    assert events == [
        ("session_reserved", source_guard),
        ("session_custody_bound", source_guard),
        ("session_custody_released", source_guard),
        ("session_settled", source_guard),
        ("session_reserved", author_guard),
        ("session_custody_bound", author_guard),
        ("session_custody_released", author_guard),
        ("session_settled", author_guard),
        ("session_reserved", retry),
        ("session_settled", retry),
    ], f"observed {events}"
    indexed = {(str(row["event"]), str(row.get("spend_id", ""))): row for row in rows}
    assert indexed[("session_settled", author_guard)]["local_outcome"] == "failed"
    assert indexed[("session_settled", author_guard)]["custody_outcome"] == "exited-error"
    assert indexed[("session_settled", source_guard)]["local_outcome"] == "completed"
    assert indexed[("session_settled", retry)]["local_outcome"] == "completed"
    # One attempt, two guard sessions, one durable group; the retry heads its own.
    assert indexed[("session_reserved", author_guard)]["attempt_group"] == source_guard
    assert indexed[("session_reserved", source_guard)]["attempt_group"] is None
    assert indexed[("session_reserved", retry)]["attempt_group"] is None
    provider_keys = {
        str(indexed[("session_reserved", spend)]["provider_idempotency_key"])
        for spend in spend_ids
    }
    assert len(provider_keys) == 3, "every reservation must mint its own provider key"


def test_session_death_plan_parses_counts_and_keyed_budgets() -> None:
    """Both documented injection spellings parse into the same plan shape."""

    assert parse_session_death_plan("2") == SessionDeathPlan(global_budget=2)
    assert parse_session_death_plan(" DryRunMLP:1 , m3353:2 ") == SessionDeathPlan(
        global_budget=None,
        keyed_budgets={"DryRunMLP": 1, "m3353": 2},
    )


def test_session_death_plan_refuses_malformed_budgets() -> None:
    """An injection the composition would silently ignore is refused loudly."""

    for value in ("", "  ", "0", "-1", "DryRunMLP", "DryRunMLP:", ":1", "DryRunMLP:0", "a:1,a:1"):
        with pytest.raises(ValueError):
            parse_session_death_plan(value)


def test_injected_session_death_settles_and_retries_on_a_fresh_reservation(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """The dry run survives paid author-session deaths through the real ladder.

    For each injected death the spend ledger must show the full post-fix custody
    sequence: the attempt's source-request guard heads the attempt group and
    settles completed; the author guard joins the group, binds custody exactly
    once, releases it, and settles as the failed attempt; the retry buys a NEW
    attempt head with a different spend id and provider idempotency key and
    settles completed. Every reservation settles (no orphan), no spend id binds
    custody twice, and acceptance still passes over all ten items.
    """

    repo_root = Path(__file__).resolve().parents[3]
    campaign_root = tmp_path / "dry-run-campaign"
    completed = _dry_run_command(
        repo_root,
        campaign_root,
        real_environment_fixture.prefix,
        inject_session_death=_INJECTED_DEATHS,
    )
    combined = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == EXIT_OK, combined[-4000:]
    assert "duplicate custody" not in combined
    output = _json_output(completed)
    assert output["status"] == "complete"
    assert output["acceptance"]["status"] == "passed"

    current = _current_projection(campaign_root)
    assert len(current) == len(DRY_RUN_ITEMS)
    assert all(model["status"]["code"] == "runs" for model in current.values())
    assert output["funnel"] == funnel_counts(current)
    assert funnel_counts(current) == {
        "framework:pytorch": len(DRY_RUN_ITEMS),
        "metadata:accepted": len(DRY_RUN_ITEMS),
        "mode:eval": len(DRY_RUN_ITEMS),
        "mode:train": len(DRY_RUN_ITEMS),
        "models:total": len(DRY_RUN_ITEMS),
        "rung:R3_PORT": len(DRY_RUN_ITEMS),
        "status:runs": len(DRY_RUN_ITEMS),
    }

    snapshot = create_dry_run_snapshot(campaign_root)
    paths = dry_run_paths(campaign_root, snapshot)
    # The verifying loader IS an assertion: it folds every row under the anchored
    # chain and refuses exactly the duplicate-custody / orphan states this
    # rehearsal exists to catch.
    rows = load_paid_session_rows(paths.paid_session_ledger)
    reserved = {
        str(row["spend_id"]): row for row in rows if row["event"] == "session_reserved"
    }
    settled = {
        str(row["spend_id"]): row for row in rows if row["event"] == "session_settled"
    }
    bound = [str(row["spend_id"]) for row in rows if row["event"] == "session_custody_bound"]
    released = [
        str(row["spend_id"]) for row in rows if row["event"] == "session_custody_released"
    ]

    assert set(reserved) == set(settled), "every reservation must settle -- no orphan spend"
    assert len(bound) == len(set(bound)), "no spend id may bind custody twice"
    assert sorted(bound) == sorted(released), "every bound custody must release"
    provider_keys = [str(row["provider_idempotency_key"]) for row in reserved.values()]
    assert len(provider_keys) == len(set(provider_keys)), (
        "every reservation must mint its own provider idempotency key"
    )

    def _stable_id(spend_id: str) -> str:
        """Return the one work-generation stable id an AUTHOR reservation bought.

        Only the author lane reserves per-model; a checker gate reservation
        legitimately carries the whole batch's work generations, so the
        single-generation shape is asserted for author rows alone.
        """

        row = reserved[spend_id]
        assert row["lane"] == "author", (
            f"single-generation lookup applied to a {row['lane']} reservation"
        )
        generations = row["axis"]["work_generations"]
        assert len(generations) == 1
        return str(generations[0]["stable_id"])

    # Injected deaths are AUTHOR guard sessions; a checker session drained at
    # the review-checkpoint pause also settles local_outcome=failed with its
    # whole batch's work generations, so the death census scopes to the lane.
    failed = sorted(
        spend_id
        for spend_id, row in settled.items()
        if row["local_outcome"] == "failed"
        and reserved[spend_id]["lane"] == "author"
    )
    assert len(failed) == _EXPECTED_DEATHS, (
        f"expected exactly {_EXPECTED_DEATHS} injected session deaths; "
        f"observed failed settlements {failed}"
    )
    assert len({_stable_id(spend_id) for spend_id in failed}) == _EXPECTED_DEATHS, (
        "keyed death budgets must land on distinct items"
    )

    for spend_id in failed:
        row = reserved[spend_id]
        stable_id = _stable_id(spend_id)
        # Reservation of the failed attempt: custody bound once, released, settled.
        assert bound.count(spend_id) == 1, f"dead author guard {spend_id} must bind custody once"
        assert released.count(spend_id) == 1
        assert settled[spend_id]["custody_outcome"] == "exited-error"
        # The dead author guard durably JOINED the attempt headed by its own
        # source-request guard -- one attempt, two guard sessions, two spend ids.
        head = row["attempt_group"]
        assert isinstance(head, str) and head != spend_id
        head_row = reserved[head]
        assert head_row["attempt_group"] is None
        assert _stable_id(head) == stable_id
        assert bound.count(head) == 1 and released.count(head) == 1
        assert settled[head]["local_outcome"] == "completed"
        # The retry bought a NEW attempt head: different spend id, settled clean.
        retry_heads = [
            candidate
            for candidate, candidate_row in reserved.items()
            if candidate not in (spend_id, head)
            and candidate_row["lane"] == "author"
            and candidate_row["attempt_group"] is None
            and _stable_id(candidate) == stable_id
        ]
        assert retry_heads, f"no fresh retry reservation for {stable_id}"
        assert all(
            settled[candidate]["local_outcome"] == "completed" for candidate in retry_heads
        ), f"the retry attempt for {stable_id} must settle completed"
