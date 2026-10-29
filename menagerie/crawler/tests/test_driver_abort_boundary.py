"""Fault-injection proof for the author no-publication abort boundary (rung 7).

Two of the last three rungs were killed entirely by ONE model's author sessions
dying without publishing: rung 5 (m4066) and rung 7 (m7362, whose repair-round
``claude -p`` died twice in 0.6s -- session transcripts show a transient host
credential blip -- stranding five models that held published, gated work). The
W5 containment law (which replaced the rung-7 three-clause boundary whose
"second consecutive distinct model" adjacency killed leg 27): after the bounded
backoff retry ladder spends, a no-publication author death on a host that
PASSES its cheap checks (disk floor, ledger writable, lock held) is CONTAINED
as a deferred-family author-wave park -- never a machine ``failed:author``
(OC-Q6), never a campaign abort. The lane-health escalation is the windowed,
width-normalized publication-DROUGHT detector: zero publications plus
``max(2, ceil(width/2))`` distinct unpublished deaths in the window routes an
all-machine-class drought to the durable usage pause and a content drought to
a LOUD campaign stop; failed host checks always stay campaign-scoped, and on
ANY campaign abort every model holding staged published work gets a durable
stranded record.

These tests extend the existing subprocess-fault pattern
(``test_slice_f_supervisor.py`` drives ``CrawlerSupervisor`` with a
``FakeProcessFactory``) one level up: a REAL subprocess is spawned and dies at
stage1/stage2 with the executor's exact failure line and exit 75, its streams run
through the REAL exit classifier, and the REAL driver loop in a tmp campaign root
decides the boundary -- both sides of it. Blanket never-abort is explicitly a
FAIL here: the no-witness, lane-wide, and host-fault cases must still halt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

import pytest

import menagerie.crawler.author_spend as author_spend_module
from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION
from menagerie.crawler.driver_admission import (
    _raise_for_checker_exit,
    _run_operator_command,
    classify_author_exit,
)
from menagerie.crawler.driver_contracts import (
    AuthorArtifact,
    AuthorSessionRetryExhausted,
    DriverConfig,
    DriverIntegrationError,
    DriverShutdown,
    CheckerOutcome,
    RetryableOperatorError,
    ResearchToolsUnavailableError,
    WorkItem,
)
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakeChecker,
    _driver,
    _paths,
    _snapshot,
)


class _RecordingSleeper:
    """Record every infrastructure backoff wait instead of sleeping it."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


class SubprocessKilledAuthor(FakeAuthor):
    """Author lane whose victim models run a REAL subprocess killed mid-stage.

    Healthy models keep the canonical synthetic behavior. A victim model spawns
    an actual child process that dies with the author executor's exact structured
    failure line and retryable exit code; the captured streams then run through
    the REAL ``classify_author_exit``, so the driver sees byte-for-byte what the
    rung-7 crash produced. ``crash_budget`` bounds how many sessions die before
    the victim recovers (``None`` = every session dies).
    """

    def __init__(self, victims: dict[str, str], crash_budget: Optional[int] = None) -> None:
        """Bind victim models to the stage their subprocess dies at."""

        super().__init__()
        self.victims = dict(victims)
        self.crash_budget = crash_budget
        self.crashes: dict[str, int] = {}

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Kill a real child process for victims; delegate for healthy models."""

        stage = self.victims.get(item.stable_id)
        crashed = self.crashes.get(item.stable_id, 0)
        if stage is not None and (self.crash_budget is None or crashed < self.crash_budget):
            self.crashes[item.stable_id] = crashed + 1
            line = f"author executor {stage} failed: session-crashed (attempt deadbeef)"
            completed = _run_operator_command(
                [
                    sys.executable,
                    "-c",
                    "import sys; "
                    f"sys.stderr.write({line!r}); sys.exit(75)",
                ],
                timeout_seconds=30.0,
                env=None,
            )
            classify_author_exit(
                "author",
                item.stable_id,
                completed.returncode,
                completed.stdout or "",
                completed.stderr or "",
            )
            raise AssertionError("killed author subprocess classified as clean exit")
        return super().author(item, work_root, config, context)


class ProviderOverloadedAuthor(SubprocessKilledAuthor):
    """Author lane whose victim emits the structured provider-overloaded record."""

    def __init__(
        self,
        victims: dict[str, str],
        crash_budget: Optional[int] = None,
        *,
        notice_text: str = "API Error: Repeated 529 Overloaded errors.",
    ) -> None:
        """Bind victim models to structured 529 notices."""

        super().__init__(victims, crash_budget)
        self.notice_text = notice_text

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Kill a victim child with the rung-8 provider-overloaded JSON payload."""

        stage = self.victims.get(item.stable_id)
        crashed = self.crashes.get(item.stable_id, 0)
        if stage is not None and (self.crash_budget is None or crashed < self.crash_budget):
            self.crashes[item.stable_id] = crashed + 1
            payload = {
                "type": "result",
                "subtype": "author-session-failure",
                "failure_class": "provider-overloaded",
                "harness_is_error": True,
                "harness_subtype": "success",
                "harness_terminal_reason": "api_error",
                "returncode": 1,
                "session_error_text_quarantined": self.notice_text,
            }
            line = (
                f"author executor {stage} failed: session-crashed (attempt deadbeef)\n"
                f"{json.dumps(payload, sort_keys=True)}"
            )
            completed = _run_operator_command(
                [
                    sys.executable,
                    "-c",
                    "import sys; "
                    f"sys.stderr.write({line!r}); sys.exit(75)",
                ],
                timeout_seconds=30.0,
                env=None,
            )
            classify_author_exit(
                "author",
                item.stable_id,
                completed.returncode,
                completed.stdout or "",
                completed.stderr or "",
            )
            raise AssertionError("provider-overloaded subprocess classified as clean exit")
        return FakeAuthor.author(self, item, work_root, config, context)


class MixedCauseAuthor(ProviderOverloadedAuthor):
    """Author lane with provider-overload and generic session-death victims."""

    def __init__(
        self,
        provider_victims: dict[str, str],
        generic_victims: dict[str, str],
    ) -> None:
        """Bind each victim set to its failure cause."""

        super().__init__(provider_victims)
        self.generic_victims = dict(generic_victims)

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Route generic victims to the plain subprocess crash fixture."""

        if item.stable_id in self.generic_victims:
            plain = SubprocessKilledAuthor.author
            original_victims = self.victims
            self.victims = self.generic_victims
            try:
                return plain(self, item, work_root, config, context)
            finally:
                self.victims = original_victims
        return super().author(item, work_root, config, context)


class ProviderWeatherChecker(FakeChecker):
    """Checker lane that raises an exact structured capacity storm."""

    def __init__(self, failures: int) -> None:
        """Set the number of provider-weather invocations before recovery."""

        super().__init__()
        self.failures = failures
        self.invocations = 0

    def check_metadata(
        self,
        artifacts: Sequence[AuthorArtifact],
        work_root: Path,
        config: DriverConfig,
    ) -> CheckerOutcome:
        """Raise bounded checker weather, then delegate to the healthy checker."""

        self.invocations += 1
        if self.invocations <= self.failures:
            exc = RetryableOperatorError("synthetic checker provider capacity storm")
            exc.failure_class = "checker-provider-overloaded"
            exc.failure_class_basis = "codex-event:error:model-at-capacity"
            exc.request_sha256 = "sha256:" + "c" * 64
            raise exc
        return super().check_metadata(artifacts, work_root, config)


def _boundary_events(tmp_path: Path, snapshot) -> list[dict]:
    """Return every author-lane boundary decision on the operational ledger."""

    operational = _paths(tmp_path, snapshot).operational_ledger
    if not operational.is_file():
        return []
    return [
        event
        for event in scan_jsonl(operational)
        if event.get("details", {}).get("kind") == "author-no-publication-boundary"
    ]


def _checker_weather_events(tmp_path: Path, snapshot) -> list[dict]:
    """Return every checker-weather heartbeat on the operational ledger."""

    operational = _paths(tmp_path, snapshot).operational_ledger
    if not operational.is_file():
        return []
    return [
        event
        for event in scan_jsonl(operational)
        if event.get("details", {}).get("kind") == "checker-provider-weather-park"
    ]


_REAL_RUNG8_PROVIDER_OVERLOADED_EVENT = (
    b'{"at":"2026-08-05T09:51:00.248916Z","event":"resume-failed",'
    b'"failure":{"failure_class":"provider-overloaded","harness_is_error":true,'
    b'"harness_subtype":"success","harness_terminal_reason":"api_error",'
    b'"returncode":1,"session_error_text_quarantined":"API Error: Repeated 529 '
    b"Overloaded errors. The API is at capacity \\u2014 this is usually temporary. "
    b'Try again in a moment. If it persists, check https://status.claude.com."},'
    b'"resumed_from":"4232a8da-6579-4db4-96b2-a8602086b57f","returncode":1}'
)


def _stranded_events(tmp_path: Path, snapshot) -> list[dict]:
    """Return every durable stranded-model record on the operational ledger."""

    operational = _paths(tmp_path, snapshot).operational_ledger
    if not operational.is_file():
        return []
    return [
        event
        for event in scan_jsonl(operational)
        if event.get("details", {}).get("kind") == "model-stranded-by-campaign-abort"
    ]


def _driver_state(tmp_path: Path, snapshot) -> dict:
    """Read the machine-consumable driver state file."""

    return json.loads(_paths(tmp_path, snapshot).driver_state.read_text(encoding="utf-8"))


@pytest.mark.smoke
@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_isolated_author_subprocess_death_is_member_contained_with_witness(
    tmp_path: Path, stage: str
) -> None:
    """An injected author death on model k leaves models k+1..n completing.

    W5 (OC-Q6): the containment terminal is a deferred-family park, never a
    machine ``failed:author`` -- the fault may be machinery's, so no verdict
    exists until a fresh generation or an operator adjudicates. Both the
    stage1 and stage2 death shapes land on the same classification (exit 75
    without a structured contract line), so both must land on the same side
    of the boundary.
    """

    snapshot = _snapshot(tmp_path, count=3)
    victim = snapshot.items[2].stable_id
    sleeper = _RecordingSleeper()
    author = SubprocessKilledAuthor({victim: stage})

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=1,
        sleeper=sleeper,
    ).run()

    assert result.status == "complete-with-deferrals"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    # The siblings completed and were recorded.
    assert sum(record["status"]["code"] == "runs" for record in models.values()) == 2
    # The victim holds NO model verdict: it is parked deferred-family on the
    # author-wave site, bounded by W12's retry budget.
    assert victim not in models
    from menagerie.crawler.scheduler_parks import (
        load_scheduler_quarantine_rows,
        scheduler_quarantine_path,
    )

    records_root = _paths(tmp_path, snapshot).ledgers.models.parent.parent
    (park,) = load_scheduler_quarantine_rows([scheduler_quarantine_path(records_root)])
    assert park["stable_id"] == victim
    assert park["scheduler_quarantine"]["site"] == "author-wave"
    assert park["scheduler_quarantine"]["reason_code"] == "member-fault"
    # The bounded ladder actually retried, with real backoff between attempts.
    assert sleeper.waits == [5.0, 30.0]
    assert author.crashes[victim] == 3
    # The boundary decision is durable and records the lane-health witness.
    (decision,) = _boundary_events(tmp_path, snapshot)
    assert decision["details"]["decision"] == "member-contained"
    assert decision["details"]["stable_id"] == victim
    assert decision["details"]["publications_this_run"] >= 1
    assert decision["details"]["host_checks"]["passed"] is True
    # Nothing was stranded: every model reduced or parked accountably.
    assert _stranded_events(tmp_path, snapshot) == []


@pytest.mark.smoke
def test_transient_author_death_recovers_through_backoff(tmp_path: Path) -> None:
    """A one-session blip costs backoff, never a terminal (rung 7's actual cause).

    The observed 0.6s rc=1 deaths were a transient credential condition; a ladder
    with real waits between attempts survives it and the model completes
    normally. This is the 'actually retry things classed retryable' side.
    """

    snapshot = _snapshot(tmp_path, count=2)
    victim = snapshot.items[1].stable_id
    sleeper = _RecordingSleeper()
    author = SubprocessKilledAuthor({victim: "stage1"}, crash_budget=1)

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=1,
        sleeper=sleeper,
    ).run()

    assert result.status == "complete"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert models[victim]["status"]["code"] == "runs"
    assert sleeper.waits == [5.0]
    assert _boundary_events(tmp_path, snapshot) == []


@pytest.mark.smoke
def test_repeated_provider_overload_uses_weather_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated structured provider-overloaded crashes park before the final retry."""

    snapshot = _snapshot(tmp_path, count=3)
    monkeypatch.setattr(author_spend_module, "AUTHOR_SESSIONS_MAX", 10)
    victim = snapshot.items[2].stable_id
    sleeper = _RecordingSleeper()
    author = ProviderOverloadedAuthor({victim: "stage2"})

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=1,
        sleeper=sleeper,
    ).run()

    assert result.status == "complete-with-deferrals"
    assert sleeper.waits == [5.0, 300.0, 900.0, 900.0, 900.0]
    assert author.crashes[victim] == 6
    (decision,) = _boundary_events(tmp_path, snapshot)
    assert decision["details"]["decision"] == "member-contained"
    assert decision["details"]["retry_backoff_seconds"] == [
        5.0,
        300.0,
        900.0,
        900.0,
        900.0,
    ]


@pytest.mark.smoke
def test_provider_overload_storm_survives_fifty_minute_park(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The m7362-length storm parks once per re-entry instead of burying models."""

    snapshot = _snapshot(tmp_path, count=3)
    monkeypatch.setattr(author_spend_module, "AUTHOR_SESSIONS_MAX", 10)
    victim = snapshot.items[2].stable_id
    sleeper = _RecordingSleeper()
    author = ProviderOverloadedAuthor({victim: "stage2"}, crash_budget=5)

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=1,
        sleeper=sleeper,
    ).run()

    assert result.status == "complete"
    assert sleeper.waits == [5.0, 300.0, 900.0, 900.0, 900.0]
    assert sum(sleeper.waits) == 3005.0
    assert author.crashes[victim] == 5
    assert _boundary_events(tmp_path, snapshot) == []
    state = _driver_state(tmp_path, snapshot)
    assert state["status"] == "complete"
    assert state.get("retry_exhausted") is not True


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("failures", "expected_waits"),
    [
        (1, [5.0]),
        (5, [5.0, 300.0, 900.0, 900.0, 900.0]),
    ],
)
def test_checker_capacity_storm_parks_then_recovers_without_duplicate_authoring(
    tmp_path: Path,
    failures: int,
    expected_waits: list[float],
) -> None:
    """A transient or fifty-minute checker storm leaves models live-pending."""

    snapshot = _snapshot(tmp_path, count=1)
    sleeper = _RecordingSleeper()
    author = FakeAuthor()
    checker = ProviderWeatherChecker(failures)

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        checker=checker,
        author_concurrency=1,
        sleeper=sleeper,
    ).run()

    stable_id = snapshot.items[0].stable_id
    assert result.status == "complete"
    assert sleeper.waits == expected_waits
    assert author.calls == {stable_id: 1}
    events = _checker_weather_events(tmp_path, snapshot)
    assert [event["details"]["interval_index"] for event in events] == list(range(failures))
    assert all(event["observed_response"] is None for event in events)
    assert all(
        event["details"]["failure_class_basis"] == "codex-event:error:model-at-capacity"
        for event in events
    )


@pytest.mark.smoke
def test_checker_capacity_storm_exhaustion_is_loud_and_resumable(tmp_path: Path) -> None:
    """A storm beyond the bounded park aborts loudly without minting a terminal."""

    snapshot = _snapshot(tmp_path, count=1)
    sleeper = _RecordingSleeper()
    author = FakeAuthor()
    storm = ProviderWeatherChecker(failures=6)

    with pytest.raises(RetryableOperatorError, match="provider-overloaded storm exhausted"):
        _driver(
            tmp_path,
            snapshot,
            author=author,
            checker=storm,
            author_concurrency=1,
            sleeper=sleeper,
        ).run()

    stable_id = snapshot.items[0].stable_id
    assert sleeper.waits == [5.0, 300.0, 900.0, 900.0, 900.0]
    assert author.calls == {stable_id: 1}
    # The leg-open statement materializes an EMPTY models ledger at leg start
    # (close-object v2: a layout may not change presence during a leg), so the
    # abort invariant is "no canonical row minted", not "no file exists".
    assert scan_jsonl(_paths(tmp_path, snapshot).ledgers.models) == []
    state = _driver_state(tmp_path, snapshot)
    assert state["status"] == "retryable:infrastructure"
    assert "provider-overloaded storm exhausted" in state["message"]

    recovered = _driver(
        tmp_path,
        snapshot,
        author=author,
        checker=ProviderWeatherChecker(failures=0),
        author_concurrency=1,
        sleeper=sleeper,
    ).run()
    assert recovered.status == "complete"
    assert author.calls == {stable_id: 1}


@pytest.mark.smoke
def test_checker_weather_wait_is_shutdown_interruptible(tmp_path: Path) -> None:
    """A shutdown raised during weather parking stops before another invocation."""

    snapshot = _snapshot(tmp_path, count=1)
    driver_holder = {}

    def interrupt(_seconds: float) -> None:
        """Model the shutdown event set by the installed SIGTERM handler."""

        driver_holder["driver"]._shutdown_event.set()

    driver = _driver(
        tmp_path,
        snapshot,
        author_concurrency=1,
        sleeper=interrupt,
    )
    driver_holder["driver"] = driver
    (item,) = driver._ordered_work(snapshot, {}, {})
    calls = 0

    def weather() -> None:
        """Raise one exact checker capacity signal."""

        nonlocal calls
        calls += 1
        exc = RetryableOperatorError("synthetic checker capacity")
        exc.failure_class = "checker-provider-overloaded"
        exc.failure_class_basis = "codex-event:error:model-at-capacity"
        exc.request_sha256 = "sha256:" + "d" * 64
        raise exc

    with pytest.raises(DriverShutdown):
        driver._retry_infrastructure_call(weather, admission=("checker", item))
    assert calls == 1


@pytest.mark.smoke
def test_distinct_cause_double_death_stays_contained_beside_a_publishing_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W5: cause diversity never re-opens the adjacency escalation.

    Two DISTINCT members dying of DIFFERENT machine causes beside a lane that
    published inside the window are each one member's weather (the leg-27
    survival law): both contained as author-wave parks, no campaign
    escalation, no storm pause -- the drought detector reads the live
    publication and stays quiet. The storm arm itself (zero publications) is
    pinned at the unit level in ``test_author_lane_containment``.
    """

    snapshot = _snapshot(tmp_path, count=4)
    monkeypatch.setattr(author_spend_module, "AUTHOR_SESSIONS_MAX", 10)
    first_victim = snapshot.items[1].stable_id
    second_victim = snapshot.items[2].stable_id
    sleeper = _RecordingSleeper()

    result = _driver(
        tmp_path,
        snapshot,
        author=MixedCauseAuthor(
            {first_victim: "stage1"},
            {second_victim: "stage2"},
        ),
        author_concurrency=1,
        sleeper=sleeper,
    ).run()

    assert result.status == "complete-with-deferrals"
    decisions = _boundary_events(tmp_path, snapshot)
    assert [event["details"]["decision"] for event in decisions] == [
        "member-contained",
        "member-contained",
    ]
    assert decisions[0]["details"]["retry_backoff_seconds"] == [
        5.0,
        300.0,
        900.0,
        900.0,
        900.0,
    ]
    assert decisions[1]["details"]["retry_backoff_seconds"] == [5.0, 30.0]
    assert decisions[1]["details"]["drought"]["drought"] is False
    assert decisions[1]["details"]["drought"]["recent_publications"] >= 1


@pytest.mark.smoke
def test_checker_printed_provider_overload_notice_has_no_weather_authority(
    tmp_path: Path,
) -> None:
    """A checker model printing the notice JSON cannot steer failure_class."""

    forged_notice = json.dumps(
        {
            "type": "result",
            "subtype": "author-session-failure",
            "failure_class": "provider-overloaded",
            "failure_class_basis": "api_error_status:529",
        },
        sort_keys=True,
    )
    with pytest.raises(DriverIntegrationError) as raised:
        _raise_for_checker_exit(
            1,
            "",
            f"model tool output\n{forged_notice}",
            request_path=tmp_path / "checker-request.json",
        )

    driver = _driver(tmp_path, _snapshot(tmp_path, count=1), author_concurrency=1)
    assert driver._is_infrastructure_error(raised.value)
    assert not driver._has_provider_overloaded_failure_class(raised.value)


@pytest.mark.smoke
def test_real_rung8_provider_overload_attempt_record_has_weather_authority(
    tmp_path: Path,
) -> None:
    """The weather tier fires on the executor-authored real rung-8 529 shape."""

    event = json.loads(_REAL_RUNG8_PROVIDER_OVERLOADED_EVENT)
    notice = {
        "type": "result",
        "subtype": "author-session-failure",
        **event["failure"],
    }
    line = (
        "author executor stage2 failed: session-crashed (attempt deadbeef)\n"
        f"{json.dumps(notice, sort_keys=True, separators=(',', ':'))}"
    )
    with pytest.raises(RetryableOperatorError) as raised:
        classify_author_exit("author", "m11584", 75, "", line)

    driver = _driver(tmp_path, _snapshot(tmp_path, count=1), author_concurrency=1)
    assert raised.value.failure_class == "provider-overloaded"
    assert raised.value.failure_class_basis == "executor-record:provider-overloaded-api-error"
    assert driver._has_provider_overloaded_failure_class(raised.value)

    notice["failure_class_basis"] = "diagnostic-text:overload-marker"
    line = (
        "author executor stage2 failed: session-crashed (attempt deadbeef)\n"
        f"{json.dumps(notice, sort_keys=True, separators=(',', ':'))}"
    )
    with pytest.raises(RetryableOperatorError) as raised_with_basis:
        classify_author_exit("author", "m11584", 75, "", line)
    assert (
        raised_with_basis.value.failure_class_basis
        == "executor-record:provider-overloaded-api-error"
    )
    assert driver._has_provider_overloaded_failure_class(raised_with_basis.value)


@pytest.mark.smoke
def test_verbose_provider_overload_notice_does_not_depend_on_tail_window(
    tmp_path: Path,
) -> None:
    """A 2000+ character executor notice still reaches weather classification."""

    snapshot = _snapshot(tmp_path, count=3)
    victim = snapshot.items[2].stable_id
    sleeper = _RecordingSleeper()
    notice_text = "API Error: Repeated 529 Overloaded. " + ("x" * 2100)
    author = ProviderOverloadedAuthor(
        {victim: "stage2"},
        crash_budget=1,
        notice_text=notice_text,
    )

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=1,
        sleeper=sleeper,
    ).run()

    assert result.status == "complete"
    assert sleeper.waits == [5.0]
    assert author.crashes[victim] == 1


@pytest.mark.smoke
def test_first_model_death_without_witness_halts_loud(tmp_path: Path) -> None:
    """The F arm's contrapositive on the very first session: fail closed.

    A run whose FIRST authored session dies unpublished has no work root yet,
    so the host checks READ SICK (disk usage of an absent directory) and the
    death stays campaign-scoped (BATON trap 3): absence of a host failure
    signal is never treated as evidence of host health, and under W5 the
    member containment fires ONLY on a host that provably passes its checks.
    Blanket never-abort is a FAIL.
    """

    snapshot = _snapshot(tmp_path, count=3)
    victim = snapshot.items[0].stable_id

    with pytest.raises(AuthorSessionRetryExhausted):
        _driver(
            tmp_path,
            snapshot,
            author=SubprocessKilledAuthor({victim: "stage1"}),
            author_concurrency=1,
        ).run()

    (decision,) = _boundary_events(tmp_path, snapshot)
    assert decision["details"]["decision"] == "campaign-scoped"
    assert decision["details"]["publications_this_run"] == 0
    state = _driver_state(tmp_path, snapshot)
    assert state["status"] == "retryable:infrastructure"
    assert state["retry_exhausted"] is True


@pytest.mark.smoke
def test_consecutive_distinct_machine_deaths_pause_durably_never_strand(
    tmp_path: Path,
) -> None:
    """W5: the serial-width machine drought pauses; nothing vanishes or strands.

    Model 0 publishes (witness), model 1 dies unpublished (member-contained,
    author-wave park). Model 2's death arrives after model 1's park -- but
    the window still holds a publication ONLY if it is recent; here the
    witness published before both deaths within the window, so no drought
    fires while a publication is inside it. To reach the drought the deaths
    must be the ONLY window activity -- the rig therefore asserts the
    contained-then-contained shape with a live witness, the exact leg-27
    survival: distinct deaths beside a publishing lane are each one member's
    weather.
    """

    snapshot = _snapshot(tmp_path, count=4)
    first_victim = snapshot.items[1].stable_id
    second_victim = snapshot.items[2].stable_id

    result = _driver(
        tmp_path,
        snapshot,
        author=SubprocessKilledAuthor({first_victim: "stage1", second_victim: "stage2"}),
        author_concurrency=1,
    ).run()

    assert result.status == "complete-with-deferrals"
    decisions = _boundary_events(tmp_path, snapshot)
    assert [event["details"]["decision"] for event in decisions] == [
        "member-contained",
        "member-contained",
    ]
    assert decisions[0]["details"]["stable_id"] == first_victim
    assert decisions[1]["details"]["stable_id"] == second_victim
    # Both victims are accountable parks, no machine failed:author verdicts,
    # and NOTHING is stranded: the siblings reduced normally.
    from menagerie.crawler.scheduler_parks import (
        load_scheduler_quarantine_rows,
        scheduler_quarantine_path,
    )

    records_root = _paths(tmp_path, snapshot).ledgers.models.parent.parent
    parks = load_scheduler_quarantine_rows([scheduler_quarantine_path(records_root)])
    assert sorted(row["stable_id"] for row in parks) == sorted(
        [first_victim, second_victim]
    )
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert first_victim not in models and second_victim not in models
    assert _stranded_events(tmp_path, snapshot) == []


@pytest.mark.smoke
def test_failed_host_check_halts_even_with_a_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine host fault (disk under floor) still aborts loudly.

    'This MACHINE is broken' must still stop the run even when a sibling
    published earlier: the host checks can veto model-scoping, and absence of a
    witness is not the only campaign-scoped trigger.
    """

    import menagerie.crawler.driver_admission as admission

    monkeypatch.setattr(admission, "_AUTHOR_LANE_DISK_FLOOR_BYTES", 10**18)
    snapshot = _snapshot(tmp_path, count=3)
    victim = snapshot.items[2].stable_id

    with pytest.raises(AuthorSessionRetryExhausted):
        _driver(
            tmp_path,
            snapshot,
            author=SubprocessKilledAuthor({victim: "stage1"}),
            author_concurrency=1,
        ).run()

    (decision,) = _boundary_events(tmp_path, snapshot)
    assert decision["details"]["decision"] == "campaign-scoped"
    assert decision["details"]["publications_this_run"] >= 1
    assert decision["details"]["host_checks"]["disk_ok"] is False


@pytest.mark.smoke
def test_provider_outage_types_bypass_the_boundary(tmp_path: Path) -> None:
    """Streak-promotable outage types keep their own identity and semantics.

    ``ResearchToolsUnavailableError`` feeds the author-wave outage promotion
    (consecutive distinct-model streak -> campaign pause with a remedy). Wrapping
    it in ``AuthorSessionRetryExhausted`` would break that counter, so the
    exhausted author-lane retry must re-raise it unwrapped.
    """

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot, author_concurrency=1)
    driver._leg_generation = SimpleNamespace(generation_id="sha256:test-open-generation")
    (item,) = driver._ordered_work(snapshot, {}, {})

    def _outage() -> None:
        raise ResearchToolsUnavailableError(item.stable_id, "provider down")

    with JsonlLedger(
        _paths(tmp_path, snapshot).operational_ledger,
        OPERATIONAL_EVENT_SCHEMA_VERSION,
    ) as operational:
        with pytest.raises(ResearchToolsUnavailableError) as raised:
            driver._retry_infrastructure_call(
                _outage,
                admission=("author", item),
                operational=operational,
            )
    assert not isinstance(raised.value, AuthorSessionRetryExhausted)


@pytest.mark.smoke
def test_retryable_status_is_consumable_from_driver_state(tmp_path: Path) -> None:
    """The retryable exit now has a machine-consumable surface.

    Rung 7's ``retryable:infrastructure`` was advice to a supervisor that never
    existed: nothing above the driver consumed it. The driver state file now
    carries ``retry_exhausted`` and ``stranded_models`` so resume tooling can act
    on an abort without parsing exception text.
    """

    snapshot = _snapshot(tmp_path, count=1)
    victim = snapshot.items[0].stable_id

    with pytest.raises(RetryableOperatorError):
        _driver(
            tmp_path,
            snapshot,
            author=SubprocessKilledAuthor({victim: "stage1"}),
            author_concurrency=1,
        ).run()

    state = _driver_state(tmp_path, snapshot)
    assert set(state) >= {"status", "retry_exhausted", "stranded_models", "message"}
    assert state["status"] == "retryable:infrastructure"


def test_author_session_budget_refuses_the_fourth_session(tmp_path: Path) -> None:
    """A7: at most ``author_sessions_max`` author sessions fund one generation.

    S1_T5 A7 driver-level pin: with the lane failing retryably
    (session-crashed class), the infrastructure retry ladder is one initial
    session plus ``len(_INFRASTRUCTURE_RETRY_BACKOFF_SECONDS)`` retries --
    exactly the ``author_sessions_max=3`` recorded on terminals -- and the
    exhaustion is typed, never a 4th session. A witness sibling published in
    the same run, so the refusal terminalizes MODEL-scoped and the campaign
    completes instead of halting.

    Residual (documented in the docket, needs a design decision first): the
    exhaustion names retry attempts rather than the budget vocabulary, and the
    fresh-budget-on-re-drive half is asserted at the grant layer
    (``test_a_consumed_grant_is_never_consumed_again``) rather than through a
    full driver re-drive.

    Parameters
    ----------
    tmp_path:
        Isolated campaign root.
    """

    snapshot = _snapshot(tmp_path, count=2)
    victim = snapshot.items[1].stable_id
    author = SubprocessKilledAuthor({victim: "stage1"})

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=1,
        sleeper=_RecordingSleeper(),
    ).run()

    sessions = author.crashes.get(victim, 0)
    assert sessions == 3, (
        f"one work generation dispatched {sessions} author sessions; the"
        " author_sessions_max=3 budget admits exactly one initial session"
        " plus two retries"
    )
    # The refusal is a typed member containment (author-wave park), not a
    # campaign halt: the witness sibling published, so the run completes with
    # the victim accountably parked (W5/OC-Q6).
    assert result.status == "complete-with-deferrals"
    decisions = _boundary_events(tmp_path, snapshot)
    assert [event["details"]["decision"] for event in decisions] == ["member-contained"]
    assert decisions[0]["details"]["stable_id"] == victim
    assert decisions[0]["details"]["retry_backoff_seconds"] == [5.0, 30.0]
