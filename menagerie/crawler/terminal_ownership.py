"""Standing per-leg close-proof clause: no non-model-owned terminal head (DX-1/RE-1).

BL-12(e)/C-16 made the ownership rule a classification fact; DX-1 makes it a
STRUCTURAL one. The mint sink (``driver._terminalize``) refuses any ``failed:*``
revision whose classified owner is the HOST or the harness MACHINERY, so this
clause can only fire on a mint-sink BYPASS -- a writer that reached the models
ledger without the sink. It runs on EVERY leg close over exactly the rows this
leg appended (the leg-generation window), so a bypass is caught in the same leg
that minted it, and pre-DX-1 history (the 200a false-terminal cohort) can never
brick a close it did not write.

The scan judges heads by the closed per-reason owner map
(:data:`menagerie.crawler.driver_receipts.WORKER_FAILURE_REASON_OWNERS`): a head
row carries no per-attempt owner stamp, and the map alone is the census
authority for which reasons may never read as model verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from menagerie.crawler.driver_receipts import (
    NEVER_MODEL_TERMINAL_OWNERS,
    WORKER_FAILURE_REASON_OWNERS,
)
from menagerie.crawler.leg_generation import (
    MODELS_SOURCE_NAMES,
    census_source_bytes,
    close_ledger_sources,
)
from menagerie.crawler.models import JsonObject

__all__ = [
    "NonModelTerminalHeadError",
    "non_model_owned_failed_heads",
    "non_model_owned_failed_heads_in_window",
    "pre_model_code_runner_head_findings",
    "unverified_resource_head_findings",
]

#: The attempts-layout slots, in authority order (sharded production first).
_ATTEMPTS_SOURCE_NAMES: tuple[str, ...] = ("attempts.local", "attempts.flat")

#: The runner reasons whose model ownership keys on constructor evidence
#: (DX-2/RE-2): with no proof model code ran, these are machinery parks, so a
#: ``failed:runner`` head carrying one over constructor-free worker attempts is
#: a classification bypass.
_PRE_MODEL_CODE_RUNNER_REASONS = frozenset({"missing-receipt", "protocol-violation"})


class NonModelTerminalHeadError(ValueError):
    """A leg window contains a failed head whose owner forbids model terminals."""


def non_model_owned_failed_heads(
    rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
) -> tuple[JsonObject, ...]:
    """Return every failed head whose reason is host- or machinery-owned.

    Parameters
    ----------
    rows:
        Decoded model-ledger rows to judge.

    Returns
    -------
    tuple[dict[str, Any], ...]
        One finding per offending head: stable id, revision, status code,
        reason code, and the owner the closed map assigns.
    """

    findings: list[JsonObject] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        status = row.get("status")
        if not isinstance(status, Mapping) or status.get("kind") != "failed":
            continue
        reason = status.get("reason_code")
        owner = WORKER_FAILURE_REASON_OWNERS.get(str(reason)) if reason is not None else None
        if owner in NEVER_MODEL_TERMINAL_OWNERS:
            findings.append(
                {
                    "stable_id": row.get("stable_id"),
                    "record_revision": row.get("record_revision"),
                    "status_code": status.get("code"),
                    "reason_code": reason,
                    "owner": owner,
                }
            )
    return tuple(findings)


def non_model_owned_failed_heads_in_window(
    records_root: Path,
    *,
    baselines: Mapping[str, Any],
) -> tuple[JsonObject, ...]:
    """Scan the leg-generation window of every models source for the clause.

    Parameters
    ----------
    records_root:
        Canonical records root whose model sources are scanned.
    baselines:
        The leg-open per-source baselines (``LegGeneration.baselines``) anchoring
        the half-open evidence window this leg appended.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Findings over exactly the rows appended in this leg's window, each
        additionally naming its source and row index.
    """

    sources = close_ledger_sources(records_root)
    findings: list[JsonObject] = []
    for source_name in MODELS_SOURCE_NAMES:
        path = sources[source_name]
        if not path.is_file():
            continue
        floor = int(dict(baselines.get(source_name) or {}).get("rows") or 0)
        lines = [
            line for line in census_source_bytes(path).split(b"\n") if line.strip()
        ]
        for offset, payload in enumerate(lines[floor:]):
            try:
                row = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(row, Mapping):
                continue
            for finding in non_model_owned_failed_heads([row]):
                findings.append(
                    {**finding, "source": source_name, "row_index": floor + offset}
                )
    return tuple(findings)


def _window_rows(
    sources: Mapping[str, Path],
    source_names: tuple[str, ...],
    baselines: Mapping[str, Any],
) -> list[tuple[str, int, Mapping[str, Any]]]:
    """Decode every window row of the named sources, with source and index.

    Parameters
    ----------
    sources:
        The closed close-source path map.
    source_names:
        Which sources to read.
    baselines:
        Per-source leg-open row floors; a missing floor reads the whole source.

    Returns
    -------
    list[tuple[str, int, Mapping[str, Any]]]
        ``(source_name, row_index, row)`` per decodable window row.
    """

    rows: list[tuple[str, int, Mapping[str, Any]]] = []
    for source_name in source_names:
        path = sources[source_name]
        if not path.is_file():
            continue
        floor = int(dict(baselines.get(source_name) or {}).get("rows") or 0)
        lines = [line for line in census_source_bytes(path).split(b"\n") if line.strip()]
        for offset, payload in enumerate(lines[floor:]):
            try:
                row = json.loads(payload)
            except ValueError:
                continue
            if isinstance(row, Mapping):
                rows.append((source_name, floor + offset, row))
    return rows


def unverified_resource_head_findings(
    records_root: Path,
    *,
    baselines: Mapping[str, Any],
) -> tuple[JsonObject, ...]:
    """DX-6 clause: no ``failed:resource`` head may stand on an RSS observation.

    An RSS-cap kill is capacity EVIDENCE: an all-RSS generation defers through
    the observed-capacity valve (``deferred:needs-more-memory`` row), an
    incomplete observation parks as machinery, and a mixed generation settles
    on its non-RSS failure. A ``failed:resource``/``rss-cap`` head minted in
    this leg's window is therefore always a valve bypass.

    Parameters
    ----------
    records_root:
        Canonical records root.
    baselines:
        The leg-open per-source baselines anchoring this leg's append window.

    Returns
    -------
    tuple[dict[str, Any], ...]
        One finding per offending in-window head.
    """

    sources = close_ledger_sources(records_root)
    findings: list[JsonObject] = []
    for source_name, row_index, row in _window_rows(sources, MODELS_SOURCE_NAMES, baselines):
        status = row.get("status")
        if not isinstance(status, Mapping) or status.get("kind") != "failed":
            continue
        if str(status.get("code", "")) != "failed:resource":
            continue
        if status.get("reason_code") not in {"rss-cap", "blocked-by-sibling"}:
            continue
        findings.append(
            {
                "stable_id": row.get("stable_id"),
                "record_revision": row.get("record_revision"),
                "status_code": status.get("code"),
                "reason_code": status.get("reason_code"),
                "owner": "machinery",
                "source": source_name,
                "row_index": row_index,
            }
        )
    return tuple(findings)


def pre_model_code_runner_head_findings(
    records_root: Path,
    *,
    baselines: Mapping[str, Any],
) -> tuple[JsonObject, ...]:
    """DX-2 clause: no ``failed:runner`` head over constructor-free worker deaths.

    A ``failed:runner`` head whose reason is ``missing-receipt`` or
    ``protocol-violation`` may only stand when some bound WORKER attempt proves
    model code ran (``worker_receipt.constructor_started``). Heads bound only to
    driver-lane adjudication attempts (no worker was ever spawned) are exempt --
    those record author/protocol verdicts, not worker deaths.

    Parameters
    ----------
    records_root:
        Canonical records root.
    baselines:
        The leg-open per-source baselines anchoring this leg's append window.

    Returns
    -------
    tuple[dict[str, Any], ...]
        One finding per offending in-window head.
    """

    sources = close_ledger_sources(records_root)
    heads = [
        (source_name, row_index, row, status)
        for source_name, row_index, row in _window_rows(
            sources, MODELS_SOURCE_NAMES, baselines
        )
        if isinstance(status := row.get("status"), Mapping)
        and status.get("kind") == "failed"
        and str(status.get("code", "")) == "failed:runner"
        and status.get("reason_code") in _PRE_MODEL_CODE_RUNNER_REASONS
    ]
    if not heads:
        return ()
    # Full-source attempt join: the head's decisive attempts may precede this
    # leg's window (a resumed generation), so the lookup is never windowed.
    attempts: dict[str, tuple[str, bool]] = {}
    for _source, _index, row in _window_rows(
        sources, _ATTEMPTS_SOURCE_NAMES, {name: {"rows": 0} for name in _ATTEMPTS_SOURCE_NAMES}
    ):
        attempt_id = row.get("attempt_id")
        if not isinstance(attempt_id, str):
            continue
        receipt = row.get("worker_receipt")
        constructor_started = bool(
            receipt.get("constructor_started") if isinstance(receipt, Mapping) else False
        )
        attempts[attempt_id] = (str(row.get("actor")), constructor_started)
    findings: list[JsonObject] = []
    for source_name, row_index, row, status in heads:
        attempt_ids = status.get("attempt_ids")
        bound = [
            attempts[attempt_id]
            for attempt_id in (attempt_ids if isinstance(attempt_ids, list) else [])
            if isinstance(attempt_id, str) and attempt_id in attempts
        ]
        worker_bound = [entry for entry in bound if entry[0] == "worker"]
        if worker_bound and not any(started for _actor, started in worker_bound):
            findings.append(
                {
                    "stable_id": row.get("stable_id"),
                    "record_revision": row.get("record_revision"),
                    "status_code": status.get("code"),
                    "reason_code": status.get("reason_code"),
                    "owner": "machinery",
                    "source": source_name,
                    "row_index": row_index,
                    "worker_attempts": len(worker_bound),
                }
            )
    return tuple(findings)
