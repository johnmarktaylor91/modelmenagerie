"""LC-11 (D-23): generation-span + continuation-budget loop alarms, read-side.

RULING W7 clause 14 ships this REGARDLESS of the deferred segments design:
today one leg generation's evidence window spans the whole 43-leg rung, so a
generation that stays open too long, or a member whose routing continuation
keeps re-manifesting, is invisible until a close tries to adjudicate it.
Both alarms are pure derivations over durable ledgers -- no daemon, no state,
no authority. An alarm can never improve, settle, or license anything; it
exists so the operator reads the loop in minutes instead of at rung end.

* ``generation_span_alarm``: leg-events ledger, open ``leg_open`` rows vs
  closes -- a generation open past the span threshold or joined by too many
  process invocations alarms.
* ``continuation_budget_alarm``: the OPEN-5 arm -- per (member, work) charged
  parks since the last authenticated restoring boundary, through the exact
  budget derivation the close's continuation disclosure uses (ONE
  implementation). A member at the recidivism limit is HELD, operator-owed
  debt: disclosed at close, alarmed here in-flight ("a third marker buys
  nothing").

WS9's monitor lane owns the cadence/wiring; ``tools/continuation_alarm``
gives it the one-invocation surface (exit 0 quiet / 2 alarm / 3 broken).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

JsonObject = dict[str, Any]

#: Campaign legs turn over in hours-to-days (the same cadence rationale as
#: the W31 grant-aging threshold). A generation open past three days has
#: outlived any lawful leg cadence and is far more likely a silent loop.
LC11_GENERATION_SPAN_ALARM_HOURS: float = 72.0

#: A typed pause/resume adds one run row; the DX-23 acceptance shape
#: (kill/resume x3) is four. Six joined invocations on ONE generation is
#: comfortably past lawful churn and reads as a restart loop.
LC11_GENERATION_SPAN_ALARM_RUNS: int = 6


def _parse_stamp(value: Any) -> Optional[datetime]:
    """Parse one UTC ledger timestamp; unparseable reads as ``None``."""

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=timezone.utc)


def generation_span_alarm(
    records_root: Path,
    *,
    now: datetime,
    max_span_hours: float = LC11_GENERATION_SPAN_ALARM_HOURS,
    max_runs: int = LC11_GENERATION_SPAN_ALARM_RUNS,
) -> JsonObject:
    """Alarm on leg generations open too long or joined too many times.

    Read-side only: ``leg_open`` rows without a naming close, their
    ``opened_at`` age against ``now``, and their joined ``leg_run`` count.
    An unparseable ``opened_at`` on an OPEN generation alarms (an undatable
    open window is at least as suspicious as an old one -- never hidden).

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    now:
        Timezone-aware instant spans are measured against.
    max_span_hours, max_runs:
        Alarm thresholds; the defaults are the LC-11 constants.

    Returns
    -------
    dict[str, Any]
        ``{"open_generations": [...], "alarms": [...], "status": ...}`` where
        status is ``"alarm"`` or ``"quiet"``.
    """

    from menagerie.crawler.ledger_chain import decode_ledger  # noqa: PLC0415 -- lazy: import-light
    from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: leg_generation is heavy
        _closed_generation_ids,
        leg_events_path,
    )

    path = leg_events_path(records_root)
    if not path.is_file():
        return {"open_generations": [], "alarms": [], "status": "quiet"}
    events = decode_ledger(path)
    closed = _closed_generation_ids(records_root)
    open_rows = [
        row
        for row in events
        if row.get("event") == "leg_open" and str(row.get("generation_id")) not in closed
    ]
    runs_by_generation: dict[str, int] = {}
    for row in events:
        if row.get("event") == "leg_run":
            generation_id = str(row.get("generation_id"))
            runs_by_generation[generation_id] = runs_by_generation.get(generation_id, 0) + 1
    open_generations: list[JsonObject] = []
    alarms: list[JsonObject] = []
    for row in open_rows:
        generation_id = str(row.get("generation_id"))
        opened = _parse_stamp(row.get("opened_at"))
        span_hours = (now - opened).total_seconds() / 3600 if opened is not None else None
        # The opening run also joins the generation: 1 + later leg_run rows.
        runs = 1 + runs_by_generation.get(generation_id, 0)
        entry: JsonObject = {
            "generation_id": generation_id,
            "campaign_id": row.get("campaign_id"),
            "opened_at": row.get("opened_at"),
            "span_hours": round(span_hours, 3) if span_hours is not None else None,
            "runs": runs,
        }
        open_generations.append(entry)
        reasons: list[str] = []
        if span_hours is None:
            reasons.append("opened_at unparseable: an undatable open window alarms")
        elif span_hours > max_span_hours:
            reasons.append(
                f"open {entry['span_hours']}h > {max_span_hours}h span threshold"
            )
        if runs > max_runs:
            reasons.append(f"{runs} joined invocations > {max_runs} run threshold")
        if reasons:
            alarms.append({**entry, "reasons": reasons})
    return {
        "open_generations": open_generations,
        "alarms": alarms,
        "status": "alarm" if alarms else "quiet",
    }


def continuation_budget_alarm(records_root: Path) -> JsonObject:
    """OPEN-5's alarm arm: HELD and limit-adjacent continuation budgets.

    Per (stable_id, work_id) with a LIVE scheduler-quarantine row: charged
    parks since the last authenticated restoring boundary, through the SAME
    derivation the close's continuing-member disclosure uses (PD-11's
    ``budget_restoring_boundaries`` + the one charged-event key). A member
    at or past ``SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT`` is HELD -- the close
    still completes (RULING W7 clause 7); the debt is operator-owed and this
    alarm is how it reads in-flight instead of at rung end.

    Returns
    -------
    dict[str, Any]
        ``{"budgets": [...], "alarms": [...], "status": ...}``.
    """

    from menagerie.crawler.close_object import _charged_park_events  # noqa: PLC2701 -- the ONE charged-event derivation
    from menagerie.crawler.scheduler_parks import (  # noqa: PLC0415 -- lazy: import-light
        SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT,
        budget_restoring_boundaries,
        load_scheduler_quarantine_ledger,
        scheduler_quarantine_path,
    )
    from menagerie.crawler.scheduler_parks import _park_event_key  # noqa: PLC2701 -- the ONE charged-event key

    ledger = load_scheduler_quarantine_ledger([scheduler_quarantine_path(records_root)])
    if not ledger:
        return {"budgets": [], "alarms": [], "status": "quiet"}
    events = _charged_park_events(records_root)
    boundaries = budget_restoring_boundaries(ledger, events)
    # Per-ROW liveness (RULING W7 clause 9): a park is live until a void names
    # its exact row. "Released" in the budget sense means budget REMAINS for a
    # retry -- a released marker is still a live continuation, exactly what
    # this alarm watches.
    voided = {str(row.get("void_of")) for row in ledger if "void_of" in row}
    live_keys: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in ledger:
        if "void_of" in row or str(row.get("row_sha256")) in voided:
            continue
        live_keys[(str(row.get("stable_id")), str(row.get("work_id")))] = row
    budgets: list[JsonObject] = []
    alarms: list[JsonObject] = []
    for key in sorted(live_keys):
        stable_id, work_id = key
        boundary = boundaries.get(key, "")
        charged = 0
        for event in events:
            event_key = _park_event_key(event)
            if event_key is not None and event_key[:2] == key and event_key[2] > boundary:
                charged += 1
        row = live_keys[key]
        entry: JsonObject = {
            "stable_id": stable_id,
            "work_id": work_id,
            "reason_code": row.get("reason_code"),
            "charged_parks": charged,
            "limit": SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT,
            "held": charged >= SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT,
        }
        budgets.append(entry)
        if entry["held"]:
            alarms.append(
                {
                    **entry,
                    "reasons": [
                        f"charged {charged} >= recidivism limit "
                        f"{SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT}: member HELD as "
                        "operator-owed debt; a further marker buys nothing "
                        "(OPEN-5, RULING W7 clause 7)"
                    ],
                }
            )
    return {
        "budgets": budgets,
        "alarms": alarms,
        "status": "alarm" if alarms else "quiet",
    }
