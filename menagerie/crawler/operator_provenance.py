"""Operator-row time and identity authentication (D-49/D-53, WS7 close half).

The step-200 audit found the operator-facing rows unauthenticated in time:
0 of 208 requeue grants carried any timestamp, 2 operator voids were dated in
the FUTURE, and operator identities were free text (11 spellings). A row
whose ``created_at`` postdates the close that counts it is a fabricated
audit trail, and nothing refused it.

This module is the ONE shared authority for both halves:

* :func:`parse_operator_timestamp` / :func:`assert_operator_timestamp` --
  the strict UTC parse and the non-future check every write site and the
  close share, so "authenticated in time" means one thing everywhere.
* :func:`post_dated_window_rows` -- the close-side sweep: every row appended
  IN THE CLOSING LEG'S WINDOW to a withholding/park ledger (park rows AND
  their void rows live in the same ledgers) must carry a parseable,
  non-post-dated ``created_at`` if it carries one at all. The close refuses
  on findings.

Era honesty (PD-13: the closed rung is never edited, corrections are
forward-only): rows BELOW the window floor are history -- the two known
future-dated 200a voids are corrected by APPENDED notes and disclosed by the
W42 close census, never re-refused by later legs. A row WITHOUT ``created_at``
is tolerated in any era (the grant schema only gains ``created_at`` in its
v2 wave); a row that CARRIES one inside the live window must parse and must
not postdate the close instant.

Identity (D-53 fold): the closed permitted-identity list is a signed
operator-packet item, so this module ships the census mechanism, not a
guess at the list -- :func:`operator_identity_census` derives the exact
per-identity row counts a close disclosure or an audit needs, recomputed
from rows on every call, never from a persisted summary.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

#: Row fields that record WHO performed an operator-attributable act, in the
#: order they are consulted. Free text today (11 spellings across the pilot);
#: the signed closed vocabulary (D-53 / §6 OP-4) will consume this same
#: census seam when it lands.
OPERATOR_IDENTITY_FIELDS: tuple[str, ...] = ("voided_by", "registered_by", "granted_by")

#: Upper bound handed to the clamped window read; ledgers clamp to their own
#: row count, so this only needs to exceed any real ledger.
_ALL_ROWS = 2**62


class OperatorProvenanceError(ValueError):
    """An operator-attributable row fails time or provenance authentication."""


def parse_operator_timestamp(value: Any, *, source: str = "row") -> datetime:
    """Parse one ``created_at`` as a strict UTC instant, or refuse typed.

    Parameters
    ----------
    value:
        Candidate timestamp value.
    source:
        Human-readable origin for the refusal text.

    Returns
    -------
    datetime
        Timezone-aware UTC instant.

    Raises
    ------
    OperatorProvenanceError
        If the value is not a parseable timestamp. A naive timestamp is read
        as UTC (the ledger convention); an offset-carrying one is normalized.
    """

    if not isinstance(value, str) or not value.strip():
        raise OperatorProvenanceError(
            f"{source} carries no parseable created_at: {value!r}"
        )
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise OperatorProvenanceError(
            f"{source} carries an unparseable created_at: {value!r}"
        ) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def assert_operator_timestamp(
    created_at: Any,
    *,
    as_of: Any,
    source: str = "row",
) -> datetime:
    """Refuse a ``created_at`` that postdates the instant counting it.

    The shared non-future check (D-49): a row dated after the act that reads
    it is a fabricated audit trail. Equality is permitted -- a write and a
    close under one coarse clock may legitimately share an instant.

    Parameters
    ----------
    created_at:
        The row's recorded decision time.
    as_of:
        The counting instant (a close's ``closed_at``, or "now" at a write
        site).
    source:
        Human-readable origin for the refusal text.

    Returns
    -------
    datetime
        The parsed, authenticated UTC instant.

    Raises
    ------
    OperatorProvenanceError
        If either instant is unparseable or ``created_at`` is in the future.
    """

    moment = parse_operator_timestamp(created_at, source=source)
    bound = parse_operator_timestamp(as_of, source=f"{source} as_of")
    if moment > bound:
        raise OperatorProvenanceError(
            f"{source} is post-dated: created_at {created_at!r} is after "
            f"{as_of!r} -- a row dated in the future of the act that counts "
            "it is a fabricated audit trail and must be corrected by an "
            "APPENDED note, never counted"
        )
    return moment


def post_dated_window_rows(
    records_root: Path,
    *,
    baselines: Mapping[str, Any],
    as_of: str,
) -> tuple[str, ...]:
    """Sweep the closing leg's window for time-fabricated withholding rows.

    Scans every row appended at or above each withholding source's window
    floor (the leg-open baseline row count) -- park rows, valve rows, and the
    operator VOID rows that share those ledgers -- and returns one finding
    per row whose ``created_at`` is present but unparseable or post-dated
    relative to ``as_of``. Rows below the floor are prior-rung history and
    are deliberately not re-adjudicated (PD-13 forward-only corrections);
    rows without any ``created_at`` are tolerated in every era.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    baselines:
        Leg-open source identities (``generation.baselines``): per-source
        ``{"rows": N}`` floors.
    as_of:
        The close instant the rows are counted at.

    Returns
    -------
    tuple[str, ...]
        Refusal findings naming ``source[row_index]``; empty when the window
        is clean.
    """

    from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: leg_generation is heavy
        WITHHOLDING_SOURCE_NAMES,
        close_ledger_sources,
        source_row_window,
    )

    sources = close_ledger_sources(records_root)
    findings: list[str] = []
    for name in WITHHOLDING_SOURCE_NAMES:
        path = sources[name]
        floor = int(dict(baselines.get(name) or {}).get("rows") or 0)
        for offset, raw in enumerate(source_row_window(path, floor, _ALL_ROWS)):
            try:
                row = json.loads(raw)
            except ValueError:
                continue  # undecodable bytes are the torn-tail machinery's finding
            if not isinstance(row, Mapping):
                continue
            created_at = row.get("created_at")
            if created_at is None:
                continue
            label = f"{name}[{floor + offset}]"
            try:
                assert_operator_timestamp(created_at, as_of=as_of, source=label)
            except OperatorProvenanceError as exc:
                findings.append(str(exc))
    return tuple(findings)


def operator_identity_census(
    rows: Any,
) -> dict[str, dict[str, int]]:
    """Count operator identities per field across an iterable of ledger rows.

    D-53's disclosure seam: DERIVED per-identity counts, recomputed from the
    rows on every call, for the close census and for the audit that will
    ratify the signed closed identity vocabulary. Free-text identities are
    counted verbatim -- the census discloses the spelling drift instead of
    normalizing it away.

    Parameters
    ----------
    rows:
        Ledger rows (mappings; non-mappings ignored).

    Returns
    -------
    dict[str, dict[str, int]]
        Identity field -> identity spelling -> row count; only fields that
        occur are present.
    """

    census: dict[str, dict[str, int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for field in OPERATOR_IDENTITY_FIELDS:
            value: Optional[Any] = row.get(field)
            if value is None:
                continue
            spellings = census.setdefault(field, {})
            spelling = str(value)
            spellings[spelling] = spellings.get(spelling, 0) + 1
    return {field: dict(sorted(values.items())) for field, values in sorted(census.items())}


def _in_window_withholding_rows(
    records_root: Path, *, baselines: Mapping[str, Any]
) -> list[tuple[str, int, Mapping[str, Any]]]:
    """Return every decodable in-window withholding/park/void row, cited."""

    from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: leg_generation is heavy
        WITHHOLDING_SOURCE_NAMES,
        close_ledger_sources,
        source_row_window,
    )

    sources = close_ledger_sources(records_root)
    rows: list[tuple[str, int, Mapping[str, Any]]] = []
    for name in WITHHOLDING_SOURCE_NAMES:
        path = sources[name]
        floor = int(dict(baselines.get(name) or {}).get("rows") or 0)
        for offset, raw in enumerate(source_row_window(path, floor, _ALL_ROWS)):
            try:
                row = json.loads(raw)
            except ValueError:
                continue  # undecodable bytes are the torn-tail machinery's finding
            if isinstance(row, Mapping):
                rows.append((name, floor + offset, row))
    return rows


def operator_act_census(
    records_root: Path, *, baselines: Mapping[str, Any]
) -> dict[str, Any]:
    """W42: the close's operator-act census over the closing leg's window.

    Every operator act this leg -- park/valve appends and the VOID rows that
    share those ledgers -- disclosed with the derived per-identity spelling
    counts (D-53's seam). Disclosure only: the census can never improve a
    close; the void-vs-live-grant refusal is the separate
    :func:`void_only_with_live_grant_findings` arm.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    baselines:
        Leg-open source identities (``generation.baselines``).

    Returns
    -------
    dict[str, Any]
        ``{"rows", "appends", "voids", "voided_row_sha256s", "identities"}``.
    """

    rows = _in_window_withholding_rows(records_root, baselines=baselines)
    voids = [(name, index, row) for name, index, row in rows if "void_of" in row]
    return {
        "rows": len(rows),
        "appends": len(rows) - len(voids),
        "voids": len(voids),
        "voided_row_sha256s": sorted(str(row["void_of"]) for _n, _i, row in voids),
        "identities": operator_identity_census(row for _n, _i, row in rows),
    }


def void_only_with_live_grant_findings(
    records_root: Path,
    *,
    baselines: Mapping[str, Any],
    as_of: str,
) -> tuple[str, ...]:
    """W42/OS-10: an in-window void-only member with a LIVE grant refuses.

    A live requeue grant is an open operator authorization for a member's
    re-drive. A leg whose only in-window operator act on that member is a
    VOID -- no grant consumption -- serviced the member through the
    receipt-less lane while the authorized lane sat open: exactly the
    operator-act shape W42 exists to make impossible to do silently (D-50
    forbids receipt-less intervention; this refuses the close that would
    launder one). Member-scoped, so the sanctioned drain (void a park over a
    member with NO live grant, or consume the grant itself) never trips it.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    baselines:
        Leg-open source identities (``generation.baselines``).
    as_of:
        The close instant (grant liveness is measured at the close).

    Returns
    -------
    tuple[str, ...]
        One finding per void-only member with a live grant; empty when clean.
    """

    from menagerie.crawler.checkpoint import (  # noqa: PLC0415 -- lazy: checkpoint imports are heavy
        canonical_operational_ledger_path,
        canonical_requeue_grants_path,
    )
    from menagerie.crawler.constants import OperationalEventKind  # noqa: PLC0415 -- lazy: import-light
    from menagerie.crawler.recordio import scan_jsonl  # noqa: PLC0415 -- lazy: import-light

    models_path = records_root / "models" / "current-shard.jsonl"
    grants_path = canonical_requeue_grants_path(models_path)
    grant_rows = (
        tuple(scan_jsonl(grants_path, validate=False)) if grants_path.is_file() else ()
    )
    if not grant_rows:
        return ()
    operational_path = canonical_operational_ledger_path(models_path)
    events = tuple(scan_jsonl(operational_path)) if operational_path.exists() else ()
    voided_grants = {str(row["void_of"]) for row in grant_rows if "void_of" in row}
    consumed_grants = {
        str(dict(event.get("details") or {}).get("grant_id"))
        for event in events
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    }
    live_by_member: dict[str, list[str]] = {}
    for row in grant_rows:
        if "void_of" in row:
            continue
        grant_id = str(row.get("grant_id"))
        if grant_id in voided_grants or grant_id in consumed_grants:
            continue
        live_by_member.setdefault(str(row.get("stable_id")), []).append(grant_id)
    if not live_by_member:
        return ()
    window_rows = _in_window_withholding_rows(records_root, baselines=baselines)
    by_sha = {
        str(row.get("row_sha256")): str(row.get("stable_id"))
        for _n, _i, row in window_rows
        if row.get("row_sha256") is not None and row.get("stable_id") is not None
    }
    findings: list[str] = []
    seen: set[str] = set()
    for name, index, row in window_rows:
        if "void_of" not in row:
            continue
        member = row.get("stable_id") or by_sha.get(str(row["void_of"]))
        if member is None or str(member) in seen:
            continue
        member = str(member)
        grants = live_by_member.get(member)
        if not grants:
            continue
        seen.add(member)
        findings.append(
            f"{member}: the only in-window operator act is a VOID "
            f"({name}[{index}] of {row['void_of']}) while grant(s) "
            f"{sorted(grants)} stay LIVE at {as_of} -- consume or void the "
            "GRANT itself; a receipt-less void beside an open authorization "
            "never closes (W42/OS-10, D-50)"
        )
    return tuple(findings)


__all__ = [
    "OPERATOR_IDENTITY_FIELDS",
    "OperatorProvenanceError",
    "assert_operator_timestamp",
    "operator_act_census",
    "operator_identity_census",
    "parse_operator_timestamp",
    "post_dated_window_rows",
    "void_only_with_live_grant_findings",
]
