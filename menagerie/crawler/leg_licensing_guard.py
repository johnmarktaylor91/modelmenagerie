"""SD-2: leg open refuses on outstanding licensing debt / stale rotated axes.

R9's failure shape is latency: identity axes rotate in the tree, heads mint
under the old identities, and the refusal fires at rung END -- the worst
possible moment, after every leg's spend. R2's is the same shape one level up:
a campaign whose newest close cannot license a successor keeps opening legs,
and the debt is discovered only when staging refuses. DX-24's shadow-close
probe ALERTS on the in-flight classes but deliberately cannot refuse anything.

This guard is the refusal arm, at the one place it is cheap: LEG START,
before any admission or spend. It derives everything from durable evidence
through the EXISTING authorities -- the accounting-integrity verdict is
:func:`menagerie.crawler.close_object.close_accounting_refusals` (the exact
multiset discipline the ``stage_rung`` gate enforces, DX-22's class) over the
campaign's newest close, and axis staleness is
:func:`menagerie.crawler.tools.rotation_diff.stale_head_census` (the W43
census) over the rotation registry. No second implementation of either
predicate exists here. Deliberately NARROW: a stale attestation, a deferral,
or an unresolved verdict on the newest close is a lawful disclosed mid-rung
state the next leg exists to resolve -- only machinery lies (a census that
does not add up) and unregistered/drifted rotation axes are debt.

ONE authority for ``award_closure_identity`` (F51 item 11, D-17 class). The
registry's heads are registered from ``register_authority_rotation
--print-live``, i.e. the shipped reducer-policy identity of the RUNNING code.
This census used to compare those heads against a natively hashed source-tree
closure -- the same axis name in two spaces, stale forever, acknowledged "on
precedent" five times while a REAL rotation on the axis was invisible inside
the unchanged digest. The census now derives the axis through the register
tool's own live-identity function, and the debt digest binds every stale axis
as the (registered head, live identity) PAIR, so a real move on an
already-stale axis changes the digest and re-arms the refusal.

The explicit-ack arm: a refusal names the census's content digest, and a
DURABLE operator acknowledgment (``control/licensing-debt-ack.json``) naming
that EXACT digest lets the leg open as a deliberate continuation. The digest
is recomputed from evidence at every open, so an ack can never be minted in
advance of debt it has not seen: any change in the debt -- a new finding, a
new stale axis, a moved stale axis -- changes the digest and the refusal
re-arms. Coverage is DISCLOSED, never silently partial: an axis whose current
identity the caller cannot compute at leg start reads ``not-computed`` in the
census payload, exactly like the ``STOP_CLASS_EVALUATOR_COVERAGE`` doctrine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

from menagerie.crawler.identity import canonical_json_bytes, durable_fsync, stable_hash

#: Durable operator acknowledgment naming the exact debt census digest.
LICENSING_DEBT_ACK_RELATIVE_PATH = ("control", "licensing-debt-ack.json")

#: Append-only acknowledgment history (P5 fable-F5/F19, sol-11): the sentinel
#: above is REPLACED per ack (only the newest gates a leg open), which
#: destroyed the disclosure trail -- by the fourth acknowledgment the earlier
#: three survived only as prose. Every acknowledgment now ALSO appends here,
#: digest-stamped, before the sentinel moves; the ledger is history, never a
#: gate, so it can never widen what an ack licenses.
LICENSING_DEBT_ACK_LEDGER_RELATIVE_PATH = ("control", "licensing-debt-acks.jsonl")

#: v2 (item 11): the digest preimage binds ``stale_axis_heads`` -- each stale
#: axis as its (registered head, live identity) pair -- instead of the bare
#: ``stale_axes`` list, and the award-closure axis is always live-derived.
LICENSING_DEBT_CENSUS_SCHEMA_VERSION = "menagerie.crawler.licensing-debt-census.v2"
LICENSING_DEBT_ACK_SCHEMA_VERSION = "menagerie.crawler.licensing-debt-ack.v1"

JsonObject = dict[str, Any]


class LegLicensingDebtError(RuntimeError):
    """Leg open refused: outstanding licensing debt or stale rotated axes.

    Carries the full census so callers can render the refusal and the exact
    acknowledgment the operator would have to mint to proceed deliberately.
    """

    def __init__(self, message: str, census: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.census = dict(census)


def licensing_debt_ack_path(records_root: Path) -> Path:
    """Return the durable acknowledgment sentinel below one records root."""

    return records_root.joinpath(*LICENSING_DEBT_ACK_RELATIVE_PATH)


def licensing_debt_ack_ledger_path(records_root: Path) -> Path:
    """Return the append-only acknowledgment history below one records root."""

    return records_root.joinpath(*LICENSING_DEBT_ACK_LEDGER_RELATIVE_PATH)


def load_licensing_debt_acks(records_root: Path) -> tuple[JsonObject, ...]:
    """Load and validate the complete acknowledgment history, fail-closed.

    Every row recomputes its ``ack_sha256`` stamp; a malformed or tampered
    row refuses the whole read (the history is audit evidence -- ambiguity is
    never silently dropped). A missing ledger is empty: acks predating the
    ledger survive only in the sentinel and operator prose, which is exactly
    the disclosure gap this ledger closes going forward.
    """

    path = licensing_debt_ack_ledger_path(records_root)
    if not path.is_file():
        return ()
    from menagerie.crawler.recordio import read_with_tail_recovery  # noqa: PLC0415 -- leaf import, matches the census's evidenced-tail-recovery arm

    def _decode(data: bytes) -> tuple[JsonObject, ...]:
        rows: list[JsonObject] = []
        for index, line in enumerate(data.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"licensing-debt ack ledger line {index} is not JSON: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"licensing-debt ack ledger line {index} is not an object")
            if payload.get("schema_version") != LICENSING_DEBT_ACK_SCHEMA_VERSION:
                raise ValueError(
                    f"licensing-debt ack ledger line {index} declares unknown schema "
                    f"{payload.get('schema_version')!r}"
                )
            stamped = payload.pop("ack_sha256", None)
            if stamped != stable_hash(payload):
                raise ValueError(
                    f"licensing-debt ack ledger line {index} fails its digest stamp"
                )
            rows.append({**payload, "ack_sha256": stamped})
        return tuple(rows)

    data = read_with_tail_recovery(
        path, prefix_check=lambda retained: _decode(retained.read_bytes())
    )
    return _decode(data)


def _live_award_closure_identity(campaign_id: str) -> str:
    """Derive ``award_closure_identity`` through the register tool's authority.

    The ONE derivation (item 11): the exact function ``register_authority_
    rotation --print-live`` prints, so the census compares the registry's
    heads against the space they were registered in.
    """

    from menagerie.crawler.tools.register_authority_rotation import (  # noqa: PLC0415 -- lazy: heavy imports
        live_authority_identities,
    )

    return str(live_authority_identities(campaign_id)["award_closure_identity"])


def licensing_debt_preimage(census: Mapping[str, Any]) -> JsonObject:
    """Return the exact content ``debt_sha256`` hashes.

    Item 11: ``stale_axis_heads`` binds each stale axis as its (registered
    head, live identity) pair, so a REAL move on an already-stale axis is a
    new debt with a new digest, never hidden inside an unchanged one.
    """

    return {
        "schema_version": census["schema_version"],
        "campaign_id": census["campaign_id"],
        "close_sha256": census["close_sha256"],
        "licensing_findings": list(census["licensing_findings"]),
        "stale_axis_heads": dict(census["stale_axis_heads"]),
    }


def licensing_debt_census(
    records_root: Path,
    *,
    campaign_id: str,
    current_identities: Optional[Mapping[str, str]] = None,
    repo_root: Optional[Path] = None,
) -> JsonObject:
    """Derive the leg-start licensing-debt census from durable evidence.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id:
        Campaign whose newest close is adjudicated.
    current_identities:
        Rotation-axis identities the caller can compute at leg start, if any.
        Axes the caller cannot compute read ``not-computed`` in the census --
        disclosed partial coverage, never a silent pass or a false refusal.
        ``award_closure_identity`` is derived here through the register tool's
        live-identity authority when the caller does not supply it.
    repo_root:
        Accepted for signature compatibility; NO LONGER consulted (item 11):
        the award-closure axis is never hashed natively from a source tree.

    Returns
    -------
    dict[str, Any]
        The census: licensing findings over the newest close, the W43
        stale-axis census (when a rotation registry exists) with each stale
        axis's (registered head, live identity) pair, and the content digest
        an acknowledgment must name.
    """

    del repo_root  # item 11: one derivation; the native tree hash is retired here
    from menagerie.crawler.close_object import (  # noqa: PLC0415 -- lazy: keeps leg-open import cost flat
        close_accounting_refusals,
        close_object_path,
        load_close_objects,
    )

    close_path = close_object_path(records_root)
    newest: Optional[Mapping[str, Any]] = None
    if close_path.is_file():
        for row in load_close_objects([close_path]):
            if str(row.get("campaign_id")) == campaign_id:
                newest = row
    licensing_findings: tuple[str, ...] = ()
    if newest is not None:
        # The NARROW accounting-integrity class only (DX-22's multiset
        # discipline): a lying census is machinery debt in any leg. A stale
        # attestation, a deferral, or an unresolved verdict is a lawful
        # disclosed mid-rung state the NEXT leg exists to resolve -- refusing
        # on those would be the close-blocking arm PD-1 adjudicated against
        # (the zero-intervention target).
        licensing_findings = close_accounting_refusals(newest)

    from menagerie.crawler.rotation_registry import (  # noqa: PLC0415 -- lazy, as above
        rotation_registry_path,
    )

    axis_census: JsonObject
    stale_axes: list[str] = []
    stale_axis_heads: dict[str, JsonObject] = {}
    if not rotation_registry_path(records_root).is_file():
        # No rotation registry: this root never adopted rotation discipline,
        # so there is nothing registered to be stale against. Disclosed, not
        # adjudicated -- a fresh fixture root must not read as eight stale
        # axes.
        axis_census = {"status": "no-registry", "stale_axes": [], "axes": []}
    else:
        from menagerie.crawler.tools.rotation_diff import (  # noqa: PLC0415 -- lazy, as above
            RotationDiffError,
            stale_head_census,
        )

        identities = dict(current_identities or {})
        try:
            if "award_closure_identity" not in identities:
                identities["award_closure_identity"] = _live_award_closure_identity(campaign_id)
            # The ONE census implementation (W43) over the caller's identities
            # plus the live award-closure identity in the registry's own
            # space; anything neither can compute is DISCLOSED as
            # not-computed, never silently passed or falsely refused.
            axis_census = stale_head_census(None, records_root, identities)
        except (RotationDiffError, ImportError, KeyError, OSError, ValueError) as exc:
            # Ambiguity fails closed (the D-62 rule): a census that cannot
            # run is debt, not a pass.
            axis_census = {
                "status": "census-error",
                "stale_axes": [],
                "axes": [],
                "error": str(exc),
            }
            licensing_findings = (
                *licensing_findings,
                f"the stale-axis census could not run: {exc}",
            )
        stale_axes = [str(axis) for axis in axis_census.get("stale_axes", [])]
        stale_axis_heads = {
            str(row["axis"]): {
                "newest_registered_head": row.get("newest_registered_head"),
                "live_identity": row.get("current_identity"),
                "status": row.get("status"),
            }
            for row in axis_census.get("axes", [])
            if str(row.get("axis")) in stale_axes
        }

    # Acked-debt disclosure (P5 fable-F19): the full acknowledgment history
    # rides every census so an attestation embedding it discloses exactly how
    # many debts were deliberately acknowledged, and which. Disclosure only:
    # deliberately OUTSIDE debt_sha256 (an ack must never change the debt it
    # acknowledges) and never consulted by the gate.
    try:
        ack_history: JsonObject = {
            "count": 0,
            "acknowledged_debt_sha256s": [],
        }
        for ack_row in load_licensing_debt_acks(records_root):
            ack_history["count"] = int(ack_history["count"]) + 1
            ack_history["acknowledged_debt_sha256s"].append(ack_row["debt_sha256"])
        # P5 round-2 R2-5 / sol-11: an ack performed BEFORE the append-first
        # writer landed gates the leg while the history ledger reads zero --
        # the exact misleading trail this arm was built to close. When the
        # GATING sentinel names a digest the ledger does not hold, the census
        # says so instead of reporting "zero acknowledgments" beside a gate
        # that is passing on one. The remedy is one idempotent re-ack of the
        # same digest through the writer (which backfills the canonical row).
        sentinel = read_licensing_debt_ack(records_root)
        if sentinel is not None and sentinel.get("debt_sha256") not in set(
            ack_history["acknowledged_debt_sha256s"]
        ):
            ack_history["history_incomplete"] = True
            ack_history["pre_ledger_gating_ack"] = {
                "debt_sha256": sentinel.get("debt_sha256"),
                "acked_at": sentinel.get("acked_at"),
                "acked_by": sentinel.get("acked_by"),
                "remedy": (
                    "re-execute this exact acknowledgment through the append-first "
                    "writer: python -m menagerie.crawler.tools.licensing_debt ack "
                    f"--debt-sha256 {sentinel.get('debt_sha256')} --reason "
                    "'canonical-history backfill of the pre-ledger ack'"
                ),
            }
    except ValueError as exc:
        ack_history = {"count": None, "error": str(exc)}

    census: JsonObject = {
        "schema_version": LICENSING_DEBT_CENSUS_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "close_sha256": str(newest["close_sha256"]) if newest is not None else None,
        "licensing_findings": list(licensing_findings),
        "ack_history": ack_history,
        "stale_axes": stale_axes,
        # Item 11: per-axis (registered head, live identity) binding -- a real
        # move on an already-stale axis is a NEW debt with a new digest.
        "stale_axis_heads": stale_axis_heads,
        "axis_census": axis_census,
    }
    census["debt_sha256"] = stable_hash(licensing_debt_preimage(census))
    census["in_debt"] = bool(licensing_findings or stale_axes)
    return census


def read_licensing_debt_ack(records_root: Path) -> Optional[JsonObject]:
    """Load the durable acknowledgment, or ``None`` when absent.

    An unreadable or malformed acknowledgment reads as ABSENT: ambiguity
    fails closed toward the refusal, never toward a pass.
    """

    path = licensing_debt_ack_path(records_root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != LICENSING_DEBT_ACK_SCHEMA_VERSION:
        return None
    return payload


def acknowledge_licensing_debt(
    records_root: Path,
    *,
    debt_sha256: str,
    reason: str,
    acked_at: str,
    acked_by: str,
) -> JsonObject:
    """Durably acknowledge ONE exact debt census, by content digest.

    The acknowledgment is scoped: it names the digest of the census the
    operator actually read. Debt that changes in any way re-arms the refusal,
    so this can never become a standing waiver.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    debt_sha256:
        The ``debt_sha256`` of the census being acknowledged.
    reason:
        Non-empty operator rationale.
    acked_at, acked_by:
        Provenance.

    Returns
    -------
    dict[str, Any]
        The persisted acknowledgment payload.
    """

    if not reason.strip():
        raise ValueError("licensing-debt acknowledgment requires a non-empty reason")
    if not str(debt_sha256).startswith("sha256:"):
        raise ValueError("licensing-debt acknowledgment must name a sha256: debt digest")
    payload: JsonObject = {
        "schema_version": LICENSING_DEBT_ACK_SCHEMA_VERSION,
        "debt_sha256": str(debt_sha256),
        "reason": reason,
        "acked_at": acked_at,
        "acked_by": acked_by,
    }
    path = licensing_debt_ack_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # History FIRST (P5 fable-F5/F19, sol-11): the append-only ledger row
    # lands before the sentinel moves, so a crash between the two leaves a
    # disclosed-but-not-gating ack (fail-closed), never a gating ack with no
    # durable history. The stamp binds the exact payload the sentinel gets.
    ledger_row = {**payload, "ack_sha256": stable_hash(payload)}
    ledger = licensing_debt_ack_ledger_path(records_root)
    with ledger.open("ab") as handle:
        handle.write(canonical_json_bytes(ledger_row) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    scratch = path.with_suffix(".json.tmp")
    with scratch.open("wb") as handle:
        handle.write(canonical_json_bytes(payload) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    os.replace(scratch, path)
    return payload


def assert_leg_open_licensed(
    records_root: Path,
    *,
    campaign_id: str,
    current_identities: Optional[Mapping[str, str]] = None,
    repo_root: Optional[Path] = None,
) -> JsonObject:
    """Refuse a leg open over unacknowledged licensing debt (SD-2).

    Returns the census on a clean or deliberately acknowledged open; raises
    :class:`LegLicensingDebtError` otherwise. Disposition-negative: this can
    only refuse a leg start, never upgrade, license, or publish anything.
    """

    census = licensing_debt_census(
        records_root,
        campaign_id=campaign_id,
        current_identities=current_identities,
        repo_root=repo_root,
    )
    if not census["in_debt"]:
        census["acknowledged"] = False
        return census
    ack = read_licensing_debt_ack(records_root)
    if ack is not None and ack.get("debt_sha256") == census["debt_sha256"]:
        census["acknowledged"] = True
        census["ack"] = ack
        return census
    findings = "; ".join(
        (
            *census["licensing_findings"],
            *(
                f"stale rotated axis: {axis} (registered head "
                f"{census['stale_axis_heads'].get(axis, {}).get('newest_registered_head')}, "
                f"live {census['stale_axis_heads'].get(axis, {}).get('live_identity')})"
                for axis in census["stale_axes"]
            ),
        )
    )
    raise LegLicensingDebtError(
        f"leg open refuses on outstanding licensing debt for campaign "
        f"{campaign_id}: {findings}. Resolve the debt, or continue "
        "DELIBERATELY by acknowledging this exact census: python -m "
        "menagerie.crawler.tools.licensing_debt ack --records-root "
        f"{records_root} --debt-sha256 {census['debt_sha256']} --reason '<why>'. "
        "The digest is recomputed from evidence at every leg open; any change "
        "in the debt re-arms this refusal.",
        census,
    )
