"""Durable sprint-pause authority: the operator HOLD sentinel (D-62).

200a's outage window had no machine-readable "stop" that the crawler itself
consulted: pauses lived in /tmp sentinels the orchestrator's tooling read and
the crawler never did, so a launch, a fired wake, or a paid admission could
proceed straight through an operator hold. This module is the crawler-side
CHECK (OC-Q3: crawler admission owns the check; the orchestrator owns the
sentinel): one durable JSON sentinel under the canonical records root --
outside /tmp AND outside the teardown-swept runtime root -- read fail-closed
before leg start, before a fired wake resumes, and before every paid
author/checker admission.

AMBIGUITY FAILS CLOSED: an unreadable, non-JSON, wrong-format, field-drifted,
or timestamp-invalid sentinel HOLDS exactly like a valid active one -- the one
state that never holds is the sentinel's clean ABSENCE. An expiry is honored
only through its typed ``default_action`` ("resume" clears; "stay-paused"
holds), so an expiry-less hold is legal but must be cleared by the operator
(the 2026-08-23 5h46m outage happened inside an expiry-less /tmp hold nothing
audited).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

#: Format tag of the one supported sentinel schema.
SPRINT_PAUSE_FORMAT = "menagerie.crawler.sprint-pause.v1"

#: Exact top-level field set; drift in either direction is ambiguity (HELD).
SPRINT_PAUSE_FIELDS = frozenset(
    {"format", "reason", "paused_by", "created_at", "expires_at", "default_action"}
)

#: Closed vocabulary for what an EXPIRED sentinel means.
SPRINT_PAUSE_DEFAULT_ACTIONS = frozenset({"resume", "stay-paused"})


def sprint_pause_path(records_root: Path) -> Path:
    """Return the sentinel's canonical location under the records root."""

    return records_root / "control" / "sprint-pause.json"


@dataclass(frozen=True)
class SprintPauseDecision:
    """One fail-closed evaluation of the sprint-pause sentinel."""

    held: bool
    #: Closed decision class: ``absent`` / ``active`` / ``expired-resume`` /
    #: ``expired-stay-paused`` / ``ambiguous``.
    decision: str
    #: Operator-facing detail (the sentinel's reason, or the ambiguity).
    detail: Optional[str] = None


def _parse_utc(value: object) -> datetime:
    """Parse one strict RFC 3339 UTC timestamp or raise ``ValueError``."""

    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"not an RFC 3339 UTC timestamp: {value!r}")
    parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"not exact UTC: {value!r}")
    return parsed


def evaluate_sprint_pause(records_root: Path, *, now: str) -> SprintPauseDecision:
    """Evaluate the HOLD sentinel fail-closed against one explicit clock.

    Parameters
    ----------
    records_root:
        Canonical records root (the durable, non-teardown-swept tree).
    now:
        The caller's RFC 3339 UTC clock reading -- explicit so every admission
        site evaluates the same instant it records.

    Returns
    -------
    SprintPauseDecision
        ``held=False`` ONLY for a cleanly absent sentinel or a validly
        expired one whose ``default_action`` is ``resume``; every ambiguity
        holds.
    """

    path = sprint_pause_path(records_root)
    try:
        exists = path.exists()
    except OSError as exc:
        return SprintPauseDecision(True, "ambiguous", f"sentinel stat failed: {exc}")
    if not exists:
        return SprintPauseDecision(False, "absent")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return SprintPauseDecision(True, "ambiguous", f"sentinel unreadable: {exc}")
    if not isinstance(payload, dict):
        return SprintPauseDecision(True, "ambiguous", "sentinel is not a JSON object")
    if payload.get("format") != SPRINT_PAUSE_FORMAT:
        return SprintPauseDecision(
            True, "ambiguous", f"unknown sentinel format {payload.get('format')!r}"
        )
    if set(payload) != SPRINT_PAUSE_FIELDS:
        return SprintPauseDecision(
            True,
            "ambiguous",
            f"sentinel field drift (expected exactly {sorted(SPRINT_PAUSE_FIELDS)})",
        )
    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        return SprintPauseDecision(True, "ambiguous", "sentinel reason must be nonempty")
    try:
        _parse_utc(payload["created_at"])
        current = _parse_utc(now)
        expires_at = (
            None if payload["expires_at"] is None else _parse_utc(payload["expires_at"])
        )
    except ValueError as exc:
        return SprintPauseDecision(True, "ambiguous", f"sentinel timestamp invalid: {exc}")
    default_action = payload["default_action"]
    if default_action not in SPRINT_PAUSE_DEFAULT_ACTIONS:
        return SprintPauseDecision(
            True, "ambiguous", f"unknown default_action {default_action!r}"
        )
    if expires_at is None or current < expires_at:
        return SprintPauseDecision(True, "active", reason)
    if default_action == "resume":
        return SprintPauseDecision(False, "expired-resume", reason)
    return SprintPauseDecision(True, "expired-stay-paused", reason)
