"""Typed env-failure routing: host conditions park, never terminalize.

The pre-use environment failure arm used to end in ``(Exception,
"environment", "build-failed")`` -- a catch-all that minted intent-wide MODEL
terminals for pure host/infra conditions (BusyError, ExactnessError, bare
OSError, overlay and store failures). Mutation-verified (T5-2), 3-lab (T3-T2),
and live: 19/63 pilot heads were ``failed:environment``, and one
EnvironmentExactnessError superseded 14 ``runs`` revisions / 11 released
awards in 124 seconds, grant-free, with zero events (T2-08).

The routing registry is now CLOSED and deliberate:

* every environment-lifecycle exception class has an explicit route;
* host/infra classes route to intent-scoped machine PARKS (the existing
  valve idioms), never to model terminals;
* the catch-all is GONE -- an unrouted exception is a loud typed refusal,
  never a silent world-fact;
* a ``runs`` revision can never be superseded by a non-award terminal
  without a grant or a typed scar (the T2-08 reducer guard).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from menagerie.crawler.authority import AuthorityDerivationError
from menagerie.crawler.checkpoint import canonical_operational_ledger_path
from menagerie.crawler.driver_admission import _deferral_row_fact_still_holds
from menagerie.crawler.driver_contracts import DriverIntegrationError
from menagerie.crawler.driver_progress import (
    _ENVIRONMENT_FAILURE_ROUTES,
    _environment_failure,
)
from menagerie.crawler.effort import EffortCapExceeded
from menagerie.crawler.env_lifecycle import (
    CommittedLockFamilyError,
    DiskFloorError,
    DiskRecoveryError,
    EnvironmentBackendContractError,
    EnvironmentBackendUnavailableError,
    EnvironmentBusyError,
    EnvironmentCleanupError,
    EnvironmentExactnessError,
    EnvironmentLifecycleError,
    EnvironmentOverlayError,
    EnvironmentProbeError,
    EnvironmentSolveError,
)
from menagerie.crawler.environment_rebuild import (
    REBUILD_DEFERRAL_DISPOSITION,
    load_rebuild_deferral_rows,
    rebuild_deferral_path,
)
from menagerie.crawler.envs import EnvironmentIntent
from menagerie.crawler.policy import SandboxUnavailableError
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import (
    AWARD_SUPERSESSION_SCAR_PREFIX,
    AwardSupersessionError,
    ReductionError,
)
from menagerie.crawler.tests.test_reducer import CanonicalReducer
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.status import machine_deferral_rejection
from menagerie.crawler.tests.conftest import (
    bind_terminal_attempts,
    make_attempt,
    make_failed_attempt,
    make_gate,
    make_model,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeEnvironments,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke


def _walk_subclasses(root: type) -> set[type]:
    """Return every (transitively) known subclass of one exception root."""

    found: set[type] = set()
    frontier = [root]
    while frontier:
        current = frontier.pop()
        for subclass in current.__subclasses__():
            if subclass not in found:
                found.add(subclass)
                frontier.append(subclass)
    return found


def test_every_environment_lifecycle_exception_has_a_deliberate_routing() -> None:
    """The registry is closed: every lifecycle class routes, nothing catch-alls.

    T5-2's mutation probe showed the old table's last row ``(Exception,
    "environment", "build-failed")`` swallowed BusyError, ExactnessError,
    CleanupError, OverlayError, bare OSError, and RuntimeError into MODEL
    terminals. The registry may never regrow that row, and every concrete
    lifecycle exception -- including ones added later -- must resolve through
    an explicit entry.
    """

    # 1. The catch-all is gone: no registry row types Exception or BaseException.
    for route in _ENVIRONMENT_FAILURE_ROUTES:
        assert route.exception_type not in (Exception, BaseException), (
            f"the environment failure registry regrew a catch-all row for {route.exception_type!r}"
        )

    # 2. Every known lifecycle exception class resolves to a deliberate route.
    lifecycle_classes = {EnvironmentLifecycleError, *_walk_subclasses(EnvironmentLifecycleError)}
    for exception_class in lifecycle_classes:
        if exception_class is EnvironmentLifecycleError:
            # The abstract base itself is never raised; concrete classes route.
            continue
        if exception_class is DiskFloorError:
            error: Exception = DiskFloorError(
                "floor", floor_bytes=2, free_bytes=1, env_root="/tmp/envs"
            )
        else:
            error = exception_class("synthetic")
        route = _environment_failure(error)
        assert route.kind in {
            "park-intent-setup",
            "park-host-condition",
            "park-disk-floor",
            "terminal",
            "propagate",
        }, f"{exception_class.__name__} resolved to unknown route kind {route.kind!r}"

    # 3. The named non-lifecycle classes the arm can observe also route.
    assert _environment_failure(AuthorityDerivationError("authority")).kind == ("park-intent-setup")
    assert _environment_failure(_effort_cap()).kind == "park-intent-setup"
    assert _environment_failure(OSError("SSD not mounted")).kind == "park-host-condition"
    assert _environment_failure(DriverIntegrationError("harness bug")).kind == "propagate"

    # 4. An UNROUTED exception is a loud typed refusal, never a silent terminal.
    for unrouted in (RuntimeError("build"), ValueError("boom"), KeyError("k")):
        with pytest.raises(DriverIntegrationError, match="deliberate"):
            _environment_failure(unrouted)


def _effort_cap() -> EffortCapExceeded:
    """Build one representative environment effort-cap failure."""

    try:
        from menagerie.crawler.effort import EffortTracker, StageCap

        tracker = EffortTracker({"environment": StageCap(attempts=1)})
        tracker.consume("environment", attempts=1)
        tracker.consume("environment", attempts=1)
    except EffortCapExceeded as exc:
        return exc
    raise AssertionError("effort cap did not trip")


@pytest.mark.parametrize(
    ("error", "expected_kind"),
    [
        (EnvironmentBusyError("second environment requested"), "park-host-condition"),
        (EnvironmentCleanupError("removal failed"), "park-host-condition"),
        (DiskRecoveryError("teardown left residue"), "park-host-condition"),
        (OSError("[Errno 2] store SSD not mounted"), "park-host-condition"),
        (EnvironmentBackendUnavailableError("wrapper exit 75"), "park-host-condition"),
        (EnvironmentExactnessError("inventory drift"), "park-intent-setup"),
        (EnvironmentSolveError("solve failed"), "park-intent-setup"),
        (CommittedLockFamilyError("no proven family"), "park-intent-setup"),
        (EnvironmentProbeError("import canary failed"), "park-intent-setup"),
        (EnvironmentOverlayError("overlay refused"), "park-intent-setup"),
        (EnvironmentBackendContractError("wrapper exit 64"), "park-intent-setup"),
    ],
)
def test_host_and_setup_conditions_route_to_parks(error: Exception, expected_kind: str) -> None:
    """Every host/infra class routes to a machine park, never a model terminal."""

    route = _environment_failure(error)
    assert route.kind == expected_kind
    assert route.stage is None and route.reason_code is None


def test_disk_floor_and_sandbox_keep_their_deliberate_routes() -> None:
    """The floor parks; the typed fail-closed sandbox signal stays a terminal."""

    floor = DiskFloorError("floor", floor_bytes=2, free_bytes=1, env_root="/tmp/envs")
    assert _environment_failure(floor).kind == "park-disk-floor"
    sandbox = _environment_failure(SandboxUnavailableError("sandbox"))
    assert sandbox.kind == "terminal"
    assert (sandbox.stage, sandbox.reason_code) == ("policy", "sandbox-unavailable-v1")


class _FailingEnvironments(FakeEnvironments):
    """Raise one configured pre-use failure for every environment build."""

    def __init__(self, root: Path, error: Exception) -> None:
        """Store the fake root and the exception every build raises."""

        super().__init__(root)
        self.error = error

    def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
        """Fail before any model callback begins."""

        del intent, use
        raise self.error


_HOST_CONDITIONS = (
    EnvironmentBusyError("environment already active (synthetic)"),
    EnvironmentExactnessError("created-prefix inventory drift (synthetic)"),
    OSError("[Errno 6] store SSD not mounted (synthetic)"),
    EnvironmentOverlayError("pip overlay refused (synthetic)"),
    DiskRecoveryError("teardown left environment residue (synthetic)"),
    EnvironmentCleanupError("environment removal failed (synthetic)"),
    EnvironmentProbeError("environment probes failed (synthetic)"),
    CommittedLockFamilyError("no proven committed family (synthetic)"),
    EnvironmentBackendUnavailableError("environment wrapper exited 75 (synthetic)"),
)


@pytest.mark.parametrize("error", _HOST_CONDITIONS, ids=lambda error: type(error).__name__)
def test_a_host_condition_never_mints_a_model_terminal(tmp_path: Path, error: Exception) -> None:
    """Machine rows through the real driver: parks, claims, ZERO terminals."""

    snapshot = _snapshot(tmp_path, count=2)
    paths = _paths(tmp_path, snapshot)
    result = _driver(
        tmp_path,
        snapshot,
        environments=_FailingEnvironments(tmp_path / "fake-envs", error),
    ).run()

    assert result.status == "complete-with-deferrals"
    # THE fix: a host/infra condition mints no model terminal, ever.
    assert scan_jsonl(paths.ledgers.models) == []
    records_root = paths.ledgers.models.parent.parent
    rows = load_rebuild_deferral_rows([rebuild_deferral_path(records_root)])
    assert {row["stable_id"] for row in rows} == {item.stable_id for item in snapshot.items}
    for row in rows:
        assert row["disposition"] == REBUILD_DEFERRAL_DISPOSITION
        quarantine = row["quarantine"]
        assert quarantine["failure_type"] == (
            f"{type(error).__module__}.{type(error).__qualname__}"
        )
        assert quarantine["condition_scope"] in {"host-condition", "intent-setup"}
        # The row is honest about what happened: no cleanup-quarantine language
        # for a pre-use failure, and the remedy names a re-drive.
        assert "no build result was recorded" in row["recheck_hint"]
        assert machine_deferral_rejection(str(row["stable_id"]), row) is None
    events = scan_jsonl(canonical_operational_ledger_path(paths.ledgers.models))
    park_events = [
        event
        for event in events
        if event.get("details", {}).get("disposition") == "environment-setup-failure-parked"
    ]
    assert len(park_events) >= 1
    assert {event["details"]["failure_type"] for event in park_events} == {
        f"{type(error).__module__}.{type(error).__qualname__}"
    }


def test_a_transient_host_condition_recovers_on_the_next_run(tmp_path: Path) -> None:
    """A host park releases: the next run with a healthy host completes runs."""

    class _OneShotBusyEnvironments(FakeEnvironments):
        """Fail every build of the first driver run, then behave."""

        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.fail = True

        def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
            if self.fail:
                raise EnvironmentBusyError("environment already active (synthetic)")
            return super().run(intent, use=use)

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    environments = _OneShotBusyEnvironments(tmp_path / "fake-envs")
    first = _driver(tmp_path, snapshot, environments=environments).run()
    assert first.status == "complete-with-deferrals"
    assert scan_jsonl(paths.ledgers.models) == []

    environments.fail = False
    second = _driver(tmp_path, snapshot, environments=environments).run()
    assert second.status == "complete"
    models = scan_jsonl(paths.ledgers.models)
    assert [record["status"]["code"] for record in models] == ["runs"]


def test_an_unrouted_exception_is_a_loud_refusal_not_a_terminal(tmp_path: Path) -> None:
    """The removed catch-all: an unknown exception refuses the RUN, types intact."""

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    with pytest.raises(DriverIntegrationError, match="deliberate"):
        _driver(
            tmp_path,
            snapshot,
            environments=_FailingEnvironments(
                tmp_path / "fake-envs", RuntimeError("synthetic unknown failure")
            ),
        ).run()
    # No silent world-fact was minted on the way down.
    assert scan_jsonl(paths.ledgers.models) == []
    records_root = paths.ledgers.models.parent.parent
    assert load_rebuild_deferral_rows([rebuild_deferral_path(records_root)]) == ()


def test_park_scopes_govern_the_admission_fact_recheck(tmp_path: Path) -> None:
    """Host-scope rows always release; setup-scope rows hold on identity."""

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    _driver(
        tmp_path,
        snapshot,
        environments=_FailingEnvironments(
            tmp_path / "fake-envs",
            EnvironmentBusyError("environment already active (synthetic)"),
        ),
    ).run()
    records_root = paths.ledgers.models.parent.parent
    rows = load_rebuild_deferral_rows([rebuild_deferral_path(records_root)])
    assert len(rows) == 1
    host_row = rows[0]
    assert host_row["quarantine"]["condition_scope"] == "host-condition"

    driver = SimpleNamespace(
        config=SimpleNamespace(machine_id="machine-test"),
        paths=SimpleNamespace(ledgers=SimpleNamespace(models=paths.ledgers.models)),
    )
    item = SimpleNamespace(route=SimpleNamespace(intent=str(host_row["quarantine"]["intent"])))
    # A transient host condition is never provably still true: the row can
    # never re-claim the model, and the live valve re-derives it if it recurs.
    assert _deferral_row_fact_still_holds(driver, item, host_row) is False  # type: ignore[arg-type]


def _stub_wrapper(tmp_path: Path, *, exit_code: int, stderr: str = "") -> list[str]:
    """Write one argv-only environment-wrapper stub with a fixed exit."""

    script = tmp_path / "stub_env_wrapper.py"
    script.write_text(
        f"import sys\nsys.stderr.write({stderr!r})\nsys.exit({exit_code})\n",
        encoding="utf-8",
    )
    import sys

    return [sys.executable, str(script)]


def test_operator_exit_typing_survives_the_backend_boundary(tmp_path: Path) -> None:
    """Exit 75 stays transient infra, exit 64 stays permanent; never collapsed.

    T3-T2 facet (a): the command adapter used to fold every nonzero exit into
    one ``DriverIntegrationError``, which the catch-all then terminalized. The
    operator's typed split (sysexits 75 = retryable infrastructure, 64 =
    permanent rejection) now crosses the subprocess boundary as two distinct
    typed exceptions with two distinct routes.
    """

    from menagerie.crawler.driver_admission import CommandEnvironmentBackend

    transient = CommandEnvironmentBackend(
        _stub_wrapper(tmp_path, exit_code=75, stderr='{"status": "retryable"}')
    )
    with pytest.raises(EnvironmentBackendUnavailableError):
        transient.remove(tmp_path / "prefix")
    assert (
        _environment_failure(EnvironmentBackendUnavailableError("wrapper exit 75")).kind
        == "park-host-condition"
    )

    permanent = CommandEnvironmentBackend(
        _stub_wrapper(tmp_path, exit_code=64, stderr='{"status": "permanent"}')
    )
    with pytest.raises(EnvironmentBackendContractError):
        permanent.remove(tmp_path / "prefix")
    assert (
        _environment_failure(EnvironmentBackendContractError("wrapper exit 64")).kind
        == "park-intent-setup"
    )

    # A wrapper-side ENOSPC (which exits 64 with the errno detail) is a DISK
    # condition: it surfaces as OSError(ENOSPC) so the lifecycle's floor
    # conversion parks it behind the disk-floor valve, not the setup park.
    import errno

    exhausted = CommandEnvironmentBackend(
        _stub_wrapper(
            tmp_path,
            exit_code=64,
            stderr='{"error": "[Errno 28] No space left on device", "status": "permanent"}',
        )
    )
    with pytest.raises(OSError) as refused:
        exhausted.remove(tmp_path / "prefix")
    assert refused.value.errno == errno.ENOSPC

    # Any OTHER exit stays the loud integration refusal it always was.
    broken = CommandEnvironmentBackend(_stub_wrapper(tmp_path, exit_code=2))
    with pytest.raises(DriverIntegrationError):
        broken.remove(tmp_path / "prefix")


def _runs_head(paths: LedgerPaths, reducer: CanonicalReducer) -> dict:
    """Append one valid released runs revision and return the persisted head."""

    del paths
    reducer.append_gate(make_gate(["m_example"]))
    reducer.append_attempt(make_attempt())
    return reducer.append_model(make_model(accepted=True)).record


def _grantless_failure_over(persisted: dict) -> tuple[dict, dict]:
    """Return an otherwise-valid grant-free failure superseding one head.

    The failure evidence is a NEW work generation (its own ``work_id``), like
    the live T2-08 wipe: the re-drive that minted the terminal was a fresh
    dispatch, not more evidence inside the awarded work. Keeping the awarded
    work's attempt stream untouched also keeps the runs revision
    dependency-current, so the projection view the guard assertion reads
    still serves it (an attempt appended INTO the awarded work would move its
    ``status_proof_identity`` and legitimately stale the award for re-drive).
    """

    failed_attempt = make_failed_attempt(attempt_id="attempt-2")
    failed_attempt["ledger_seq"] = 2
    failed_attempt["work_id"] = "work-m_example-redrive"
    candidate = bind_terminal_attempts(
        make_model(status_code="failed:source", attempt_id="attempt-2"),
        [failed_attempt],
    )
    candidate["parent_revision"] = persisted["record_revision"]
    candidate["status"]["supersedes_revision"] = persisted["record_revision"]
    candidate["record_seq"] = int(persisted["record_seq"]) + 1
    return candidate, failed_attempt


def test_a_runs_revision_is_never_superseded_grantless_by_a_non_award(
    tmp_path: Path,
) -> None:
    """The T2-08 guard: award supersession needs a grant or a typed scar.

    Live instance (2026-08-08): one EnvironmentExactnessError superseded 14
    ``runs`` revisions / 11 released awards in 124s with zero grants, zero
    scars, zero events. The reducer now refuses that append outright.
    """

    paths = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = _runs_head(paths, reducer)
        assert persisted["status"]["kind"] == "runs"

        candidate, failed_attempt = _grantless_failure_over(persisted)
        reducer.append_attempt(failed_attempt)
        with pytest.raises(AwardSupersessionError) as refused:
            reducer.append_model(candidate)
        assert isinstance(refused.value, ReductionError)
        # The refusal is a refusal: the award still governs the head. Read
        # defensively so a future read-path change FAILS here instead of
        # KeyError-ing (a guard that errors is a guard a cleanup can delete),
        # and pin the canonical models tail directly -- the ledger HEAD is the
        # authority surface the supersession guard consults, independent of
        # whatever view `current_records` becomes.
        head_view = reducer.current_records.get("m_example")
        assert head_view is not None, (
            "runs award missing from the current view after a refused supersession"
        )
        assert head_view["status"]["kind"] == "runs"
        tail = [row for row in scan_jsonl(paths.models) if row.get("stable_id") == "m_example"][-1]
        assert tail["record_revision"] == persisted["record_revision"]
        assert tail["status"]["kind"] == "runs"

        # AW-1 narrowing (AUDIT100): a GENERIC new grant is retry permission,
        # never downgrade permission -- it releases nothing here (three
        # verified awards died through the old any-superset escape).
        granted = deepcopy(candidate)
        granted["budget"]["explicit_grants"] = [
            *granted["budget"].get("explicit_grants", []),
            "sha256:" + "c" * 64,
        ]
        assert reducer.award_blocked_supersession(granted) is not None

        # A candidate-carried scar is producer input and releases nothing at
        # live admission either (resume-4 destroyed three verified awards
        # through exactly this arm). The one live release is a durable grant
        # naming the exact award revision -- pinned with its full fixture in
        # test_award_custody.py; historical scarred rows stay replay-acceptable
        # (the m5273 chain regression, same module).
        scarred = deepcopy(candidate)
        scarred["scar_history"] = [
            *scarred.get("scar_history", []),
            f"{AWARD_SUPERSESSION_SCAR_PREFIX}{persisted['record_revision']}",
        ]
        with pytest.raises(AwardSupersessionError):
            reducer.append_model(scarred)


def test_award_pre_check_matches_the_append_rule(tmp_path: Path) -> None:
    """``award_blocked_supersession`` mirrors the append tripwire exactly.

    The driver plans its award park (and the deliberate scar receipt when a
    naming grant is consumed) through this pre-check, consulting the ledger
    HEAD -- a projection-stale award is still an award -- so the two
    predicates may never disagree. AW-1 narrowed both together: no
    candidate-supplied field (a generic grant superset or a self-stamped
    scar) releases either one at live admission.
    """

    paths = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = _runs_head(paths, reducer)
        candidate, _failed_attempt = _grantless_failure_over(persisted)
        blocked = reducer.award_blocked_supersession(candidate)
        assert blocked is not None
        assert blocked["record_revision"] == persisted["record_revision"]
        # A generic NEW explicit grant releases neither predicate (retry
        # permission is not downgrade permission).
        granted = deepcopy(candidate)
        granted["budget"]["explicit_grants"] = ["sha256:" + "c" * 64]
        assert reducer.award_blocked_supersession(granted) is not None
        # A candidate-carried scar releases nothing at live admission --
        # producer input can never waive the producer's tripwire.
        scarred = deepcopy(candidate)
        scarred["scar_history"] = [
            f"{AWARD_SUPERSESSION_SCAR_PREFIX}{persisted['record_revision']}"
        ]
        assert reducer.award_blocked_supersession(scarred) is not None
        # A scar naming the WRONG revision certainly releases nothing.
        misscarred = deepcopy(candidate)
        misscarred["scar_history"] = [AWARD_SUPERSESSION_SCAR_PREFIX + "sha256:" + "0" * 64]
        assert reducer.award_blocked_supersession(misscarred) is not None


def test_award_guard_does_not_touch_award_to_award_supersession(tmp_path: Path) -> None:
    """runs -> runs supersession (a fresh re-award) stays untouched."""

    paths = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = _runs_head(paths, reducer)
        successor = make_model(accepted=True)
        successor["parent_revision"] = persisted["record_revision"]
        successor["status"]["supersedes_revision"] = persisted["record_revision"]
        successor["record_seq"] = int(persisted["record_seq"]) + 1
        successor["notes"] = "fresh re-award under the current dependency tree"
        result = reducer.append_model(successor)
        assert result.record["status"]["kind"] == "runs"


def test_the_driver_never_stamps_award_supersession_scars_automatically(
    tmp_path: Path,
) -> None:
    """Award-stability stopgap (2026-08-14, behavioral since AW-1): resume-4's
    re-adjudication used the driver's automatic scar stamp to replace three
    verified runs awards with failures. The automatic stamp stays retired: the
    stamping authority is the durable award-naming grant alone, so with no
    grant (or a generic grant) the driver finds nothing to stamp, the append
    proceeds unstamped, and the reducer's tripwire refuses it. Only a durable
    grant naming the exact award revision returns a stamping authority."""

    from menagerie.crawler.checkpoint import canonical_requeue_grants_path
    from menagerie.crawler.driver import _award_naming_grant_id
    from menagerie.crawler.recordio import canonical_json_bytes

    paths = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = _runs_head(paths, reducer)
        candidate, _failed_attempt = _grantless_failure_over(persisted)
        # No grant at all: nothing to stamp.
        assert _award_naming_grant_id(paths, candidate, persisted) is None
        # A generic new grant with a durable row: still nothing to stamp.
        generic = deepcopy(candidate)
        generic["budget"]["explicit_grants"] = ["sha256:" + "c" * 64]
        grants_path = canonical_requeue_grants_path(paths.models)
        grants_path.parent.mkdir(parents=True, exist_ok=True)
        with grants_path.open("ab") as handle:
            handle.write(
                canonical_json_bytes({"grant_id": "sha256:" + "c" * 64, "stable_id": "m_example"})
                + b"\n"
            )
        assert _award_naming_grant_id(paths, generic, persisted) is None
        # A durable grant naming the EXACT award revision is the one stamping
        # authority (the scar becomes the receipt of that decision).
        with grants_path.open("ab") as handle:
            handle.write(
                canonical_json_bytes(
                    {
                        "grant_id": "sha256:" + "d" * 64,
                        "stable_id": "m_example",
                        "blocked_award_revision": str(persisted["record_revision"]),
                    }
                )
                + b"\n"
            )
        named = deepcopy(candidate)
        named["budget"]["explicit_grants"] = ["sha256:" + "d" * 64]
        assert _award_naming_grant_id(paths, named, persisted) == "sha256:" + "d" * 64
        # And the unstamped, ungranted append still refuses at the reducer.
        with pytest.raises(ReductionError):
            reducer.append_model(candidate)


def test_award_protection_refusals_do_not_advance_the_unrecordable_streak() -> None:
    """Three protected awards in a row tripped the recording circuit breaker
    (m634/m7362/m9430, 2026-08-15) while the guard worked perfectly. An
    AwardSupersessionError is a policy refusal, never a broken recording path:
    it must not advance the generic streak whether or not its park landed --
    the park-append edge is owned by the separate typed award-park breaker."""

    from menagerie.crawler.driver import CrawlerDriver

    class _Ledger:
        def __init__(self) -> None:
            self.records: list[dict] = []

        def append(self, event) -> None:  # noqa: ANN001
            self.records.append(dict(event))

        def iter_records(self):  # noqa: ANN201
            """Iterate the test-double's operational rows."""

            return iter(self.records)

    duck = SimpleNamespace(
        config=SimpleNamespace(run_id="run-test", machine_id="machine-test"),
        dependencies=SimpleNamespace(clock=lambda: "2026-08-16T00:00:00Z"),
        _unrecordable_diagnostics={},
        _unrecordable_streak=[],
    )
    item = SimpleNamespace(stable_id="m_example", active_work_id="work-1")
    operational = _Ledger()
    refusal = AwardSupersessionError(
        stable_id="m_example", blocked_parent_revision="sha256:" + "a" * 64
    )
    # Parked or not, the award refusal never advances the generic streak.
    for parked in ({"row_sha256": "sha256:" + "b" * 64}, None):
        CrawlerDriver._report_unrecordable_terminal(
            duck,
            item,
            "failed:source",
            None,
            refusal,
            None,
            operational,
            recorded=False,
            parked=parked,
        )
        assert duck._unrecordable_streak == []
    # An unexplained recording failure still advances it.
    CrawlerDriver._report_unrecordable_terminal(
        duck,
        item,
        "failed:source",
        None,
        RuntimeError("disk gremlins"),
        None,
        operational,
        recorded=False,
    )
    assert duck._unrecordable_streak == ["m_example"]
