"""Pipelined per-model scheduler: execute-first triage before any checker spend (M3).

The wave driver holds every model in an intake-wide phase barrier: authors,
then metadata/fidelity gates for the WHOLE wave, then environment work. The
measured consequence (mini-rung, 2026-08-12) was ~15h of checker wall against
3.4 minutes of model execution, with machine-dead models buying full metadata
waves before their import ever ran. This module replaces the barrier -- and
only the barrier -- with a per-model DAG driven from a round-based commit loop
on the single writer thread:

    author -> deterministic valves -> TRIAGE-EXEC (full cold/mode matrix,
    ``award_run=False``) -> [machine death => gate-free terminal] ->
    metadata batches (the calibrated width, all full batches flushed in one pooled
    call) -> [fidelity] -> award from the RECORDED triage attempts
    (no env re-entry)

Everything the loop dispatches is an existing, unmodified driver seam:
``_ensure_authors`` (author pool), ``_admit_within_host_capacity``,
``_ensure_pending_run_anchors``, ``_run_environment_work`` (one env-intent
turn, quarantine/disk-floor/busy routing included), ``_ensure_gates`` (one
cap-width batch, repair ladder and fidelity lane included), and
``_award_recorded_run`` / the in-turn award for repaired identities. No proof
rule moves: gates remain mandatory before any ``runs`` award; the only change
is WHEN sessions are purchased and WHAT overlaps in time.

Scheduler state is derived and disposable: every predicate reads the canonical
ledgers and current records, so a crash or resume rebuilds the same labels
(cutover/rollback = the same binary in the other mode; records are
scheduler-agnostic).

Concurrency doctrine (T5-5): this module runs entirely ON the writer thread.
Session concurrency lives inside the author/checker pools each seam already
owns; pool threads only spawn/wait/capture. Every ledger append, reducer
mutation, artifact transaction, terminal routing, and driver-state write
happens here, on this thread, through the same code paths the wave driver
uses. The env lane stays a host singleton (T3-T8): env turns are sequential
by construction because this loop is single-threaded.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from menagerie.crawler.artifact_transactions import ArtifactTransitionError
from menagerie.crawler.constants import METADATA_BATCH_MAX
from menagerie.crawler.driver_contracts import (
    MEMBER_CONTAINMENT_PASSTHROUGH,
    AuthorUsagePause,
    DriverIntegrationError,
    JsonObject,
    WorkItem,
)
from menagerie.crawler.driver_models import (
    _usable_family_representative,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from menagerie.crawler.driver import CrawlerDriver
    from menagerie.crawler.driver_contracts import AuthorArtifact
    from menagerie.crawler.recordio import JsonlLedger
    from menagerie.crawler.reducer import CanonicalReducer

#: Fraction of an intent's live routed models that must be exec-ready before a
#: non-sweep env turn fires (D1a 3.3 turn trigger).
DEFAULT_ENV_TURN_READINESS_FRACTION = 0.25

#: Oldest-waiter trigger: an intent with any model exec-ready this long gets a
#: turn regardless of the readiness fraction (starvation guard, D1a R5).
DEFAULT_ENV_TURN_OLDEST_WAITER_SECONDS = 1800.0

#: Author-lane waves bought per scheduler round, in units of
#: ``config.author_concurrency``. The author turn is a POOLED BARRIER
#: (``_ensure_authors`` commits its whole cohort before returning), so feeding
#: it every ready model at once turns the "pipelined" leg into a pure author
#: queue at rung scale: at 200 members no leg survived the multi-hour step-1
#: barrier, downstream env turns and gate batches never ran, and authored
#: artifacts piled up ungated (step-200, 2026-08-20). One concurrency-wave per
#: round keeps the pool saturated while returning control to the round loop at
#: the earliest point the pool would need refilling, so execution, gating, and
#: awards interleave with authoring -- the scheduler's actual design intent.
AUTHOR_ROUND_SLICE_WAVES = 1

#: Author attempts (paid ladder heads) one model may run per scheduler round
#: (ACAP). The pooled author turn is a barrier, so pre-cap a pathological
#: model's WHOLE multi-attempt ladder -- infra retries, genuine failures, and
#: their backoffs, 7+ attempts observed -- resolved inside one pooled call, in
#: one pool slot: the round could not return until the slowest ladder finished,
#: and the release wave ran 1-2 wide for hours against ~65 ready models while
#: the same few members re-reserved sessions every 2-30 minutes. With the cap,
#: a model that neither resolves nor terminalizes within its in-round budget is
#: returned UNRESOLVED (typed, uncommitted, nothing appended) and stays
#: author-ready, so its remaining attempts pipeline across later rounds while
#: other ready models use the slots. Accounting is unchanged: the cross-round
#: continuation resumes the same ladder counters exactly where they stopped and
#: every attempt buys its admission only when it actually runs -- only WHEN
#: attempts run moves, never how many or what they cost. ``None`` disables the
#: cap (the historical resolve-inside-one-call barrier).
AUTHOR_ATTEMPTS_PER_ROUND: Optional[int] = 2


@dataclass
class _LegState:
    """Run-scoped, derived-and-disposable scheduler bookkeeping.

    Nothing here is authority: every field is recomputable from the canonical
    ledgers plus the in-run driver stashes, and a resume rebuilds it from
    scratch. It exists to make this leg's dispatch decisions (author-once,
    triage-once-per-identity, gate-once-per-identity) explicit and auditable.
    """

    #: Models whose author lane ran this leg (waves author at most once per leg).
    #: ACAP: a model returned unresolved by the per-round attempt cap is
    #: discarded from this set again -- it re-enters ``author_ready`` and boards
    #: a later round of the SAME leg until its ladder resolves or exhausts.
    authored_ids: set[str] = field(default_factory=set)
    #: stable_id -> rounds this leg in which its author ladder deferred (ACAP).
    #: Progress-token input: a deferring round ran real paid attempts, so the
    #: no-progress tripwire must never read an actively laddering member as
    #: provably unroutable.
    author_deferrals: dict[str, int] = field(default_factory=dict)
    #: Models past the host-capacity valve with a staged artifact.
    admitted_ids: set[str] = field(default_factory=set)
    #: stable_id -> recipe_revision that the triage turn actually executed.
    triaged_identity: dict[str, str] = field(default_factory=dict)
    #: stable_id -> recipe_revision that completed a gate round (metadata +
    #: fidelity lane) this leg. Repairs inside the round update the artifact,
    #: so the recorded identity is the post-round one.
    gate_resolved_identity: dict[str, str] = field(default_factory=dict)
    #: Models awarded (or award-attempted into a terminal) this leg.
    awarded_ids: set[str] = field(default_factory=set)
    #: Arrival order of the metadata-gate accumulator (FIFO, stable_id).
    gate_queue: list[str] = field(default_factory=list)
    #: intent name -> env turns this leg (triage + award).
    turns_used: dict[str, int] = field(default_factory=dict)
    #: intent name -> monotonic instant its oldest exec-ready waiter appeared.
    first_ready_at: dict[str, float] = field(default_factory=dict)
    #: Baseline RAW-head revision per model at leg start (re-drive
    #: settlement: a re-driven model settles only once a NEW revision lands).
    baseline_revisions: dict[str, Optional[str]] = field(default_factory=dict)


def run_pipelined_leg(
    driver: "CrawlerDriver",
    work: Sequence[WorkItem],
    reducer: "CanonicalReducer",
    operational: "JsonlLedger",
    state: JsonObject,
) -> Optional[str]:
    """Drive one leg through the per-model pipelined scheduler.

    Called from ``CrawlerDriver._run_locked`` in place of the phase-wave loop;
    the surrounding machinery (grant consumption before, partition close
    after, the ``DriverPaused``/``DriverShutdown`` arms) is shared with the
    wave driver unchanged.

    Parameters
    ----------
    driver:
        The locked driver (writer thread).
    work:
        ``_ordered_work`` output for the whole intake snapshot.
    reducer, operational, state:
        Locked canonical reducer, operational ledger, and driver state.

    Returns
    -------
    str | None
        Usage-pause reason, or ``None`` once every scheduled model settled.
    """

    if driver.config.only_status is not None:
        raise DriverIntegrationError(
            "--only-status re-validation is not supported by the pipelined "
            "scheduler; run it under --scheduler waves (same binary, same records)"
        )
    from menagerie.crawler.driver import _pending_metadata_readmission

    # WS-1: selection happened ONCE, in the driver, through the one policy
    # (`_select_work`): `work` holds exactly the plan-admitted members, and
    # this leg re-derives nothing. Settlement reads RAW model-ledger heads --
    # the same authority the reducer's award guard consults -- never the
    # dependency-currency projection ("a stale award is still an award").
    # The award-currentness re-validation lane is DELETED: it re-authored
    # dependency-stale runs awards each leg (10-model routing livelock,
    # step-100 finishing leg); re-validation under authority rotation is
    # AW-2's recorded-basis matrix, reached only through named modes.

    def record_of(stable_id: str) -> Optional[Mapping[str, object]]:
        return reducer.raw_head_view.get(stable_id)

    # WS-2 composition note: host-capability re-opening is a SELECTION fact
    # consumed by the one policy (`work_selection.work_selection_predicate`
    # via `host_capability_head_reopens`), so a re-opened head arrives here
    # already plan-admitted; this leg re-derives nothing.
    scheduled = tuple(work)

    leg = _LegState()
    for item in scheduled:
        record = record_of(item.stable_id)
        leg.baseline_revisions[item.stable_id] = (
            str(record["record_revision"]) if record is not None else None
        )

    artifacts: dict[str, "AuthorArtifact"] = {}
    scheduled_by_id = {item.stable_id: item for item in scheduled}

    def deferral_claimed(stable_id: str) -> bool:
        return stable_id in driver._machine_deferral_rows  # noqa: SLF001

    def parked(stable_id: str) -> bool:
        return stable_id in driver._checker_lane_parked_ids  # noqa: SLF001

    def pending_runs_readmission(stable_id: str) -> bool:
        record = record_of(stable_id)
        return record is not None and _pending_metadata_readmission(record)

    def settled(item: WorkItem) -> bool:
        stable_id = item.stable_id
        if deferral_claimed(stable_id) or parked(stable_id):
            return True
        record = record_of(stable_id)
        if record is None:
            return False
        if _pending_metadata_readmission(record):
            # Pending-metadata runs record: settled once its catch-up gate
            # round completed this leg (the record legitimately stays `runs`).
            return stable_id in leg.gate_resolved_identity
        baseline = leg.baseline_revisions.get(stable_id)
        if baseline is not None:
            # A member admitted WITH a raw head (granted, named re-drive, or a
            # host-capability head re-opened because this host can now run it)
            # settles only when its generation produced a NEW revision
            # (terminal or runs); the old record does not count.
            return str(record["record_revision"]) != baseline
        return True

    def current_identity(stable_id: str) -> Optional[str]:
        artifact = artifacts.get(stable_id)
        if artifact is None:
            return None
        revision = artifact.proposal.get("recipe_revision")
        return str(revision) if revision is not None else None

    def representative_settled(item: WorkItem) -> bool:
        representative = scheduled_by_id.get(item.family_representative_id)
        if representative is None:
            # Not scheduled this leg: its record is already whatever it will be.
            return True
        return settled(representative)

    def author_ready(item: WorkItem) -> bool:
        if item.stable_id in leg.authored_ids or settled(item):
            return False
        if item.is_family_variant:
            # The F7 family edge: a variant derives from a USABLE representative
            # (accepted AND executed, `_usable_family_representative`). A settled
            # representative that is NOT usable (a terminal failure) releases the
            # variant into the author lane anyway -- wave parity: the lane owns
            # the unusable-representative routing.
            return representative_settled(item) or _usable_family_representative(
                record_of(item.family_representative_id),
                item.family_representative_id,
            )
        return True

    def exec_ready_triage(item: WorkItem) -> bool:
        stable_id = item.stable_id
        if settled(item) or stable_id not in leg.admitted_ids:
            return False
        identity = current_identity(stable_id)
        if identity is None:
            return False
        if pending_runs_readmission(stable_id):
            # Catch-up gates only; the recorded run needs no re-execution.
            return False
        return leg.triaged_identity.get(stable_id) != identity

    def gate_ready(item: WorkItem) -> bool:
        stable_id = item.stable_id
        if settled(item) or stable_id not in leg.admitted_ids:
            return False
        identity = current_identity(stable_id)
        if identity is None:
            return False
        if leg.gate_resolved_identity.get(stable_id) == identity:
            return False
        if pending_runs_readmission(stable_id):
            return True
        return leg.triaged_identity.get(stable_id) == identity

    def award_ready(item: WorkItem) -> bool:
        stable_id = item.stable_id
        if settled(item) or stable_id in leg.awarded_ids:
            return False
        identity = current_identity(stable_id)
        if identity is None or pending_runs_readmission(stable_id):
            return False
        if leg.gate_resolved_identity.get(stable_id) != identity:
            return False
        record = record_of(stable_id)
        if record is not None and not item.requeue_active:
            # A terminal landed during the gate round (accuracy cap, fidelity)
            # blocks the award turn -- but a member admitted WITH a raw head
            # (named re-drive) proceeds while its head is still the leg-start
            # BASELINE: that old record is exactly what the re-drive
            # supersedes, not a mid-leg landing (COMPOSE: mirrors the wave
            # loop's mid-wave settlement carve).
            baseline = leg.baseline_revisions.get(stable_id)
            if baseline is None or str(record["record_revision"]) != baseline:
                return False
        return True

    def exec_ready_award(item: WorkItem) -> bool:
        # Gate-resolved under the current identity, but the recorded triage
        # attempts belong to an older identity (a gate-round repair): the
        # wave-equivalent execute-and-award turn runs it.
        return award_ready(item) and leg.triaged_identity.get(item.stable_id) != current_identity(
            item.stable_id
        )

    def ledger_growth(path) -> int:
        # Byte growth is an exact durable-progress signal for append-only
        # flock+fsync ledgers, and O(1) per round where a full scan_jsonl
        # re-read is O(rows) -- the round loop runs O(models/batch) times, so
        # scanning made the leg quadratic in ledger size at rung scale.
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0

    def progress_token() -> tuple:
        return (
            ledger_growth(driver.paths.ledgers.models),
            ledger_growth(driver.paths.ledgers.attempts),
            ledger_growth(driver.paths.ledgers.gates),
            len(artifacts),
            len(leg.authored_ids),
            dict(leg.author_deferrals),
            len(leg.admitted_ids),
            dict(leg.triaged_identity),
            dict(leg.gate_resolved_identity),
            set(leg.awarded_ids),
            dict(leg.turns_used),
            len(driver._machine_deferral_rows),  # noqa: SLF001
            len(driver._checker_lane_parked_ids),  # noqa: SLF001
        )

    def run_env_turn(
        intent_name: str,
        items: Sequence[WorkItem],
        *,
        award_run: bool,
    ) -> Optional[str]:
        leg.turns_used[intent_name] = leg.turns_used.get(intent_name, 0) + 1
        # The oldest-waiter clock is NOT popped here: a served turn that does
        # not drain the intent's ready set must keep the clock running for the
        # survivors. Leg 24 (2026-08-22, first line out of the new deferral
        # observability): nine ready models read "waited=0s" against a
        # threshold of thirteen forever, because every served turn erased the
        # clock and the next evaluation re-seeded it to NOW -- the 1800s
        # starvation guard could only ever fire once per leg. The clock now
        # clears exclusively where readiness genuinely empties (step 2's
        # evaluation prunes intents with no ready items).
        pause = driver._run_environment_work(  # noqa: SLF001
            tuple(items),
            artifacts,
            reducer,
            operational,
            state,
            award_run=award_run,
        )
        if not award_run:
            for item in items:
                identity = current_identity(item.stable_id)
                if identity is not None:
                    # Post-turn identity: an in-turn repair re-gated and
                    # re-executed its replacement inside the same call.
                    leg.triaged_identity[item.stable_id] = identity
        else:
            for item in items:
                leg.awarded_ids.add(item.stable_id)
        return pause

    def flush_gate_batch(batch_ids: Sequence[str]) -> Optional[str]:
        items = tuple(scheduled_by_id[stable_id] for stable_id in batch_ids)
        pause = driver._ensure_gates(  # noqa: SLF001
            items,
            artifacts,
            reducer,
            operational,
            state,
        )
        for stable_id in batch_ids:
            identity = current_identity(stable_id)
            if identity is not None:
                leg.gate_resolved_identity[stable_id] = identity
        return pause

    readiness_fraction = DEFAULT_ENV_TURN_READINESS_FRACTION
    oldest_waiter_seconds = DEFAULT_ENV_TURN_OLDEST_WAITER_SECONDS
    sweep = False
    while True:
        driver._check_shutdown("scheduler-round")  # noqa: SLF001
        # RC-1 paced start: once 20 new terminals exist under a new
        # configuration identity, the census row is emitted and the author
        # width cap lifts (a no-op for leg invocations and lifted crawls).
        driver._paced_start_tick()  # noqa: SLF001
        before = progress_token()

        # -- 1. Author lane: one concurrency-wave of ready models per round.
        # The pooled call is a barrier, so the round-loop slice (not the pool)
        # is what keeps authoring interleaved with env turns and gate flushes;
        # unsliced ready models stay author_ready and board the next round.
        # The slice is FRESH-FIRST: a retrying model re-enters author_ready
        # after every deferred attempt, and a fixed work-order slice re-seated
        # the same front-of-order ladder models every round while their backoff
        # sleeps held slots -- 68 fresh models queued behind ~10 retriers at
        # 4-6 live sessions (step-200 leg 21, 2026-08-21). Never-deferred
        # models lead every round in work order; deferred ladders take the
        # remaining slots in work order and inherit the lead once the fresh
        # pool drains. Unlike the earlier by-round rotation (whose offset
        # scaled with the lane width), the order never depends on
        # author_concurrency, so the pinned scheduler invariant holds: lane
        # configuration changes no canonical byte (burst-vs-serial parity,
        # test_lane_weather_scheduler_composition).
        author_slice = max(1, int(driver.config.author_concurrency)) * AUTHOR_ROUND_SLICE_WAVES
        all_ready = tuple(item for item in scheduled if author_ready(item))
        fresh_ready = tuple(
            item for item in all_ready if item.stable_id not in leg.author_deferrals
        )
        retrying_ready = tuple(
            item for item in all_ready if item.stable_id in leg.author_deferrals
        )
        ready_authors = (fresh_ready + retrying_ready)[:author_slice]
        if ready_authors:
            for item in ready_authors:
                leg.authored_ids.add(item.stable_id)
            try:
                artifacts.update(
                    driver._ensure_authors(  # noqa: SLF001
                        ready_authors,
                        reducer,
                        operational,
                        state,
                        attempts_per_round=AUTHOR_ATTEMPTS_PER_ROUND,
                    )
                )
            except AuthorUsagePause as pause:
                return pause.reason
            for stable_id in driver._author_wave_deferred_ids:  # noqa: SLF001
                # ACAP: a deferred ladder is unresolved, not authored -- the
                # model re-enters author_ready and boards a later round's
                # slice with its continuation intact.
                leg.authored_ids.discard(stable_id)
                leg.author_deferrals[stable_id] = leg.author_deferrals.get(stable_id, 0) + 1
            staged = tuple(item for item in ready_authors if item.stable_id in artifacts)
            admitted = driver._admit_within_host_capacity(  # noqa: SLF001
                staged, artifacts, operational, reducer=reducer, state=state
            )
            driver._ensure_pending_run_anchors(  # noqa: SLF001
                admitted, artifacts, reducer, operational, state
            )
            for item in admitted:
                if item.stable_id in artifacts:
                    leg.admitted_ids.add(item.stable_id)

        # -- 2. Env turns: triage (execute-first) then award-mode re-execution.
        for award_mode in (False, True):
            ready_predicate = exec_ready_award if award_mode else exec_ready_triage
            by_intent = defaultdict(list)
            for item in scheduled:
                if ready_predicate(item):
                    by_intent[item.route.intent].append(item)
            now = driver.dependencies.monotonic()
            for stale_intent in [
                name for name in leg.first_ready_at if name not in by_intent
            ]:
                # Readiness genuinely emptied for this intent: the continuous-
                # readiness window ended, so the oldest-waiter clock resets.
                leg.first_ready_at.pop(stale_intent, None)
            for intent_name in by_intent:
                leg.first_ready_at.setdefault(intent_name, now)
            for intent_name in driver._ordered_intents(by_intent):  # noqa: SLF001
                items = by_intent[intent_name]
                if not sweep and not award_mode and leg.turns_used.get(intent_name, 0) > 0:
                    # Batching heuristic for steady state ONLY. The intent's FIRST
                    # turn of the leg fires whenever anything is ready: the
                    # fraction threshold counts the whole live population
                    # (un-authored included), so at rung scale a staged backlog
                    # of 5-10 models can never reach it, and `first_ready_at`
                    # is leg-local state, so every restart re-armed the
                    # oldest-waiter clock -- across legs 7-9 (step-200,
                    # 2026-08-20/21) no env turn EVER fired while authored
                    # models exhausted upstream. One guaranteed turn per intent
                    # per leg bounds the spin-up cost and starves nothing.
                    live = sum(
                        1
                        for candidate in scheduled
                        if candidate.route.intent == intent_name and not settled(candidate)
                    )
                    threshold = max(1, int(readiness_fraction * max(1, live)))
                    waited = now - leg.first_ready_at.get(intent_name, now)
                    if len(items) < threshold and waited < oldest_waiter_seconds:
                        driver._note_env_turn_deferred(  # noqa: SLF001
                            intent_name,
                            ready=len(items),
                            threshold=threshold,
                            waited_seconds=waited,
                        )
                        continue
                pause = run_env_turn(intent_name, items, award_run=award_mode)
                if pause is not None:
                    return pause

        # -- 3. Metadata batcher: cap-width eager, short only with the tail proof.
        for item in scheduled:
            if gate_ready(item) and item.stable_id not in leg.gate_queue:
                leg.gate_queue.append(item.stable_id)
        pending_gate_ids = [
            stable_id for stable_id in leg.gate_queue if gate_ready(scheduled_by_id[stable_id])
        ]
        # Whole-flush (SD-14): hand every FULL batch to one _ensure_gates call
        # so the checker pool can settle them concurrently. The cap-width batch
        # composition is unchanged -- _run_gate_rounds chunks the prefix at
        # METADATA_BATCH_MAX exactly as before -- only the call granularity
        # (and therefore the pool occupancy) widens. The 1..4 remainder stays
        # behind the final-tail proof below.
        full_prefix = (len(pending_gate_ids) // METADATA_BATCH_MAX) * METADATA_BATCH_MAX
        if full_prefix:
            batch, pending_gate_ids = (
                pending_gate_ids[:full_prefix],
                pending_gate_ids[full_prefix:],
            )
            pause = flush_gate_batch(batch)
            if pause is not None:
                return pause
        if pending_gate_ids:
            # Final-tail proof: no model outside the accumulator can become
            # metadata-gate-ready this leg WITHOUT this flush. In the
            # round-structured loop that is decidable exactly: nothing is in
            # flight between rounds, so an arrival requires a model that can
            # still author or execute. A family variant whose unsettled
            # representative sits INSIDE the accumulator is not a blocker --
            # its only path to readiness passes through the flush itself
            # (rep gate -> rep award -> variant author), and refusing to
            # flush would deadlock the chain.
            in_accumulator = set(pending_gate_ids)

            def blocks_short_flush(item: WorkItem) -> bool:
                if settled(item) or item.stable_id in in_accumulator:
                    return False
                if author_ready(item) or exec_ready_triage(item):
                    return True
                if item.stable_id not in leg.authored_ids:
                    if item.is_family_variant and not representative_settled(item):
                        return item.family_representative_id not in in_accumulator
                    return True
                return False

            future_arrivals_possible = any(blocks_short_flush(item) for item in scheduled)
            if not future_arrivals_possible:
                pause = flush_gate_batch(tuple(pending_gate_ids))
                if pause is not None:
                    return pause

        # -- 4. Awards from recorded triage attempts (writer act, no env).
        for item in scheduled:
            if not award_ready(item) or exec_ready_award(item):
                continue
            try:
                awarded, pause = driver._award_recorded_run(  # noqa: SLF001
                    item,
                    artifacts[item.stable_id],
                    reducer,
                    operational,
                    state,
                )
            except MEMBER_CONTAINMENT_PASSTHROUGH:
                raise
            except ArtifactTransitionError as exc:
                # The artifact-transaction lifecycle is a CLOSED sequence
                # (staged -> authorized -> reconstruction -> final), so a
                # transition refusal at award time means the model's CACHED
                # chain lawfully cannot serve this award -- m12646's upgrade
                # re-drive reused its failed-era private-committed transaction
                # and the award's publication event was its fifth transition.
                # That is the solve-cache's designed cache-miss, not a
                # scheduler fault: invalidate loudly and author fresh (a new
                # result mints a new transaction that takes the publication
                # route), instead of quarantining the member every leg.
                driver._report_identity_stale_artifact_authority(  # noqa: SLF001
                    item,
                    None,
                    exc,
                    operational,
                    transaction_id=None,
                    recorded_basis_available=False,
                )
                # W9: the wave artifact map held the invalidated authority;
                # leaving it would let a later step this wave consume the
                # exact entry the retirement row just declared dead (the
                # bare-continue with no later check).
                artifacts.pop(item.stable_id, None)
                continue
            except Exception as exc:  # noqa: BLE001 -- BL-8 (a): blast radius is the member
                # Deliberately NOT ``member_quarantine_reason``: an award-ready
                # member holds in-window artifact rows, so a routing marker
                # here would contradict its custody window and refuse the
                # close (RULING W7 clause 3). The post-commit corpus-conflict
                # class (pilot m5717) is disclosed inside
                # ``_promote_award_corpus`` and never reaches this arm.
                driver._record_scheduler_quarantine(  # noqa: SLF001
                    item,
                    site="award",
                    reason_code="member-fault",
                    fault=exc,
                    operational=operational,
                )
                continue
            if awarded:
                leg.awarded_ids.add(item.stable_id)
                if pause is not None:
                    return pause
                continue
            if pause is not None:
                return pause
            # BL-8 (a): the unroutable dead end. Not exec-ready-award (the
            # identity never rotated), gate-resolved, yet the recorded triage
            # attempts cannot satisfy the execution policy -- typically a
            # parked non-model execution fault. No routing step exists for
            # this member this leg; before this arm the round went flat and
            # the tripwire killed the WHOLE leg on one member's dead end.
            driver._record_scheduler_quarantine(  # noqa: SLF001
                item,
                site="award",
                reason_code="award-unroutable",
                fault=(
                    "recorded triage attempts cannot satisfy the execution policy "
                    "under the unchanged proposal identity; no env re-entry is "
                    "routable (execute-first triage recorded a non-awardable matrix)"
                ),
                operational=operational,
            )

        # -- Exit / escalation / tripwire.
        unsettled = tuple(item for item in scheduled if not settled(item))
        if not unsettled:
            return None
        if progress_token() != before:
            sweep = False
            continue
        if not sweep:
            # Escalation round: leg-end sweep semantics. Threshold-blocked env
            # turns fire unconditionally and, once they add nothing, the
            # final-tail batch proof above becomes satisfiable.
            sweep = True
            continue
        # BL-8 (b): the no-progress tripwire QUARANTINES the stuck members, it
        # does not kill the leg. Every unsettled member after a flat round plus
        # a flat escalation sweep is provably unroutable this leg; each gets a
        # durable typed park (partition-accounted, boundedly retried per W12:
        # released while retry budget remains, then operator-drained) and
        # the loop continues, so one wedged member can no longer void the
        # progress of everyone else scheduled beside it.
        for item in unsettled:
            driver._record_scheduler_quarantine(  # noqa: SLF001
                item,
                site="scheduler-round",
                reason_code="no-durable-routing-progress",
                fault=(
                    "no durable routing progress across one full round plus the "
                    "escalation sweep; member isolated so the leg continues"
                ),
                operational=operational,
            )
