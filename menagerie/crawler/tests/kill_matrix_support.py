"""Fork-run fault-injection harness for the uncatchable-kill durability matrix.

The W5 merge gate (docket C-W5-MERGE, finding T5-4) requires proving that a
driver killed with SIGKILL -- no handlers, no ``finally`` blocks, no buffer
flushes -- inside the atomic award-commit windows, during partition close, or
mid-ledger-append leaves durable state from which one production resume either
reconstructs the campaign exactly or refuses loudly.  Nothing here monkeypatches
the process under test from the outside: every injection is installed inside a
forked child immediately before ``CrawlerDriver.run`` and dies with it.

The assert pattern is the serial no-kill oracle: the identical campaign is run
to completion at the SAME absolute path first (award identities embed campaign
paths, so a sibling directory is not comparable), its durable projection is
captured, the directory is recreated, and the killed-then-resumed campaign must
converge to that exact projection.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import signal
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.leg_exit_vocabulary import LegExitClass
from menagerie.crawler.recordio import scan_jsonl

CHILD_OUTCOME_MARKER = "kill-matrix-child-outcome.json"

#: Event fields that legitimately vary across byte-identical replays: the
#: publication-authorized event stamps a wall-clock ``created_at``, which
#: cascades into its own digest and event identity.  Every other field must
#: reproduce exactly for the oracle comparison to hold.
_VOLATILE_EVENT_FIELDS = (
    "created_at",
    "payload_sha256",
    "artifact_event_id",
    "predecessor_event_id",
)

_FINAL_EVENT_KINDS = ("published", "private-committed")

#: WS-1 clause 7: EVERY leg start emits a ``work-selection-census`` and a
#: ``projection-staleness`` disclosure.  A killed-then-resumed campaign runs
#: more legs than its serial oracle, so its operational ledger legitimately
#: carries extra disclosure pairs (identical re-emissions dedupe through the
#: idempotent event identity; only a leg whose selection view differs appends
#: new rows).  The disclosures are compared leg-aware, never excluded.
_LEG_SELECTION_DISCLOSURE_KINDS = ("work-selection-census", "projection-staleness")

#: W6/OB-M12 (Wave-2 batch): every configured leg's end is mirrored as ONE
#: ``leg-outcome`` operational event, and the W6 boot reconciler closes a
#: kill-9 orphaned leg span ``orphaned_uncontained`` (synthesized XOR orphaned)
#: before a new leg opens.  A killed-then-resumed campaign therefore has a
#: genuinely different leg topology than its serial no-kill oracle: it lawfully
#: carries one extra leg-outcome per extra leg, and its own close row accounts
#: only the terminals minted in THAT leg.  Leg-outcome rows are compared by
#: their own W6 invariants (:func:`assert_leg_outcomes_match_oracle_topology`),
#: never excluded; EVERY other operational row kind keeps exact serial
#: equality -- the crash-durability tripwire for award/model/gate/order
#: corruption stays fully armed.
_LEG_OUTCOME_KIND = "leg-outcome"

#: D-20 (LC-7) + DX-26 (RATIFIED, RULING W7 clauses 6 and 15, Wave-3 batch):
#: every leg START emits one read-only ``close-preview`` disclosure, and every
#: leg EXIT re-emits the same payload as the immutable nonterminal
#: ``leg-partial.v1`` checkpoint.  Both are per-leg, report-only, and by
#: kernel law NEVER a licence input: the ruling guarantees the checkpoint
#: "never satisfies the closed status grammar, joins the predecessor chain,
#: settles a member, or licenses anything".  A killed-then-resumed campaign
#: runs more legs than its serial oracle, and each leg-exit checkpoint anchors
#: its own per-leg ``leg_ended_sha256``, so extra and divergent preview rows
#: are the LAWFUL signature of the extra legs (identical re-emissions dedupe
#: through the idempotent event identity).  Preview rows are compared by their
#: own D-20/DX-26 invariants
#: (:func:`assert_close_previews_match_oracle_topology`), never excluded --
#: and the byte-exact core comparison of models/gates/attempts/artifacts/
#: driver-state IS the kernel's "licenses nothing" tripwire: a checkpoint that
#: influenced replay would diverge the core and still fail loudly.
_CLOSE_PREVIEW_KIND = "close-preview"

#: The closed D-20 preview verdict vocabulary (driver ``_emit_close_preview``).
_CLOSE_PREVIEW_VERDICTS = ("would-refuse", "would-close", "preview-error")


def _partition_leg_outcomes(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split normalized operational events into (leg-outcome rows, the rest)."""

    outcomes = [event for event in events if event.get("event_kind") == _LEG_OUTCOME_KIND]
    rest = [event for event in events if event.get("event_kind") != _LEG_OUTCOME_KIND]
    return outcomes, rest


def assert_leg_outcomes_match_oracle_topology(
    resumed: list[dict[str, Any]], oracle: list[dict[str, Any]]
) -> None:
    """Assert the resumed campaign's leg-outcome rows obey the W6 contract.

    W6/OB-M12 (Wave-2 batch): leg-outcome rows are leg-topology-dependent, so
    a killed-then-resumed campaign can never reproduce the serial oracle's
    rows exactly -- it runs more legs.  Instead every row is held to the W6
    invariants, each of which would trip on a real crash-path regression:

    - the ENVELOPE of every resumed leg-outcome (status, run/machine identity,
      schema, queue counts -- everything but the per-leg ``event_id`` and
      ``details``) must reproduce the oracle's envelope exactly;
    - synthesized XOR orphaned: a ``synthesized`` row is the boot reconciler's
      orphan close and must read exactly ``orphaned_uncontained`` with a null
      ``terminal_delta``; ``orphaned_uncontained`` may NEVER appear on a real
      (non-synthesized) ``leg_ended`` mirror;
    - every ``exit_class`` stays inside the closed ``LegExitClass`` vocabulary
      and every real row's ``cause_fingerprint`` is one the oracle itself
      recorded for the same exit class (the PM-5 fingerprint binds the exit
      class plus per-leg evidence, so a multi-leg oracle may record several
      per class -- membership, not global determinism, is the invariant);
    - the resumed campaign's FINAL leg-outcome is a real (non-synthesized)
      close with the oracle's terminal exit class and cause fingerprint --
      the campaign genuinely re-derived the serial close;
    - the one-kill-one-resume matrices can add at most the killed leg's
      orphan close plus the resume leg's own close beyond the oracle's rows.

    ``terminal_delta`` deliberately has NO cross-leg conservation assertion:
    it is honest PER-LEG accounting.  A mint lost to a killed leg vanishes
    (the orphan row carries null), and a mint whose ledger line was torn away
    is lawfully re-derived and re-counted by the resume leg, so per-leg sums
    may under- or overshoot the serial total.  Double-award corruption stays
    tripwired by the byte-exact models/artifact-event comparison,
    :func:`assert_no_double_award`, and :func:`assert_artifact_model_bijection`.
    """

    assert oracle, "serial oracle recorded no leg-outcome rows"
    for event in oracle:
        assert event["details"]["synthesized"] is False, (
            f"serial no-kill oracle carries a synthesized orphan close: {event}"
        )
    template = {key: value for key, value in oracle[-1].items() if key not in ("event_id", "details")}
    detail_keys = set(oracle[-1]["details"])
    fingerprints_by_class: dict[str, set[str]] = {}
    for event in oracle:
        details = event["details"]
        fingerprints_by_class.setdefault(details["exit_class"], set()).add(
            details["cause_fingerprint"]
        )

    assert len(resumed) >= len(oracle), (
        "resumed campaign recorded fewer leg-outcome rows than the serial oracle"
    )
    assert len(resumed) <= len(oracle) + 2, (
        "resumed campaign recorded more extra leg-outcome rows than one kill "
        f"plus one resume can lawfully produce: {resumed}"
    )
    closed_classes = {member.value for member in LegExitClass}
    seen_event_ids: set[str] = set()
    synthesized_fingerprints: set[str] = set()
    for event in resumed:
        envelope = {key: value for key, value in event.items() if key not in ("event_id", "details")}
        assert envelope == template, (
            f"leg-outcome envelope diverged from the serial oracle: {envelope!r} != {template!r}"
        )
        assert event["event_id"] not in seen_event_ids, f"duplicated leg-outcome row: {event}"
        seen_event_ids.add(event["event_id"])
        details = event["details"]
        assert set(details) == detail_keys, (
            f"leg-outcome detail schema diverged from the serial oracle: {sorted(details)}"
        )
        assert details["exit_class"] in closed_classes, (
            f"leg-outcome exit class outside the closed vocabulary: {details['exit_class']!r}"
        )
        assert details["invocation_origin"] == oracle[-1]["details"]["invocation_origin"]
        for field in ("generation_id", "leg_ended_sha256", "cause_fingerprint"):
            assert isinstance(details[field], str) and details[field].startswith("sha256:"), (
                f"leg-outcome {field} is not a digest: {details[field]!r}"
            )
        if details["synthesized"]:
            assert details["exit_class"] == LegExitClass.ORPHANED_UNCONTAINED.value, (
                f"synthesized leg-outcome is not the boot-reconciler orphan close: {details}"
            )
            assert details["terminal_delta"] is None, (
                f"boot-reconciler orphan close minted a terminal delta: {details}"
            )
            synthesized_fingerprints.add(details["cause_fingerprint"])
        else:
            assert details["exit_class"] != LegExitClass.ORPHANED_UNCONTAINED.value, (
                f"orphaned_uncontained on a real leg_ended mirror: {details}"
            )
            assert isinstance(details["terminal_delta"], int) and details["terminal_delta"] >= 0, (
                f"real leg-outcome carries a non-count terminal delta: {details}"
            )
            oracle_fingerprints = fingerprints_by_class.get(details["exit_class"])
            if oracle_fingerprints is not None:
                assert details["cause_fingerprint"] in oracle_fingerprints, (
                    f"cause fingerprint diverged from the oracle for "
                    f"{details['exit_class']!r}: {details['cause_fingerprint']}"
                )
    assert len(synthesized_fingerprints) <= 1, (
        f"orphan closes disagree on their cause fingerprint: {synthesized_fingerprints}"
    )
    final = resumed[-1]["details"]
    oracle_final = oracle[-1]["details"]
    assert final["synthesized"] is False, (
        f"resumed campaign's final leg-outcome is a synthesized orphan close: {final}"
    )
    assert final["exit_class"] == oracle_final["exit_class"], (
        f"resumed campaign closed {final['exit_class']!r}, "
        f"oracle closed {oracle_final['exit_class']!r}"
    )
    assert final["cause_fingerprint"] == oracle_final["cause_fingerprint"]


def _partition_close_previews(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split normalized operational events into (close-preview rows, the rest)."""

    previews = [event for event in events if event.get("event_kind") == _CLOSE_PREVIEW_KIND]
    rest = [event for event in events if event.get("event_kind") != _CLOSE_PREVIEW_KIND]
    return previews, rest


def assert_close_previews_match_oracle_topology(
    resumed: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
    resumed_outcomes: list[dict[str, Any]],
) -> None:
    """Assert the resumed campaign's close-preview rows obey the D-20/DX-26 law.

    Preview rows are leg-topology-dependent: every leg start discloses one
    preview (D-20/LC-7) and every leg exit leaves the same payload as the
    ``leg-partial.v1`` checkpoint (DX-26, RULING W7 clauses 6 and 15), whose
    ``leg_ended_sha256`` is per-leg by construction.  A killed-then-resumed
    campaign therefore can never reproduce the serial oracle's preview rows
    exactly.  Instead every row is held to invariants, each of which trips on
    a real crash-path regression:

    - the ENVELOPE of every resumed preview (run/machine identity, schema,
      queue counts -- everything but ``event_id``, ``details``, and the
      verdict-derived ``status``) must reproduce the oracle's envelope, and
      ``status`` must be ``healthy`` exactly when the verdict is
      ``would-close`` (the report-only contract: a refusing or broken preview
      is a disclosed degradation, never silence);
    - every verdict stays inside the closed D-20 vocabulary, every ``at``
      inside {leg-start, leg-exit}, and every detail schema matches the
      oracle's schema for the same ``at`` phase (``preview-error`` rows are
      held to their own error shape);
    - every leg-exit row IS the DX-26 checkpoint: it names ``leg-partial.v1``,
      carries a closed ``exit_class``, and ANCHORS to exactly one real
      (non-synthesized) leg-outcome row of the same generation, exit class,
      and ``leg_ended_sha256`` -- a free-floating checkpoint, a checkpoint on
      a boot-reconciler orphan close, or two checkpoints for one leg exit all
      fail;
    - the oracle's leg-START preview is reproduced verbatim (modulo volatile
      fields) among the resumed rows: the killed leg and the resume leg probe
      the same durable state through the same DX-24 path, so a missing or
      mutated re-emission is a real divergence, not topology slack;
    - the resumed campaign's FINAL preview row is the leg-exit checkpoint of
      its final real close, agreeing with the serial oracle's final checkpoint
      on verdict and exit class -- the campaign genuinely re-derived the
      serial close AND its preview verdict from the converged state;
    - the one-kill-one-resume matrices can add at most two preview rows beyond
      the oracle's (the resume leg's own start disclosure and exit
      checkpoint).

    The checkpoint's "licenses nothing" kernel guarantee is deliberately NOT
    asserted here: it is armed by the byte-exact core comparison in
    :func:`assert_operational_equals_oracle_modulo_leg_disclosures` -- a
    preview that influenced replay diverges models/gates/artifacts and fails
    there.
    """

    assert oracle, "serial oracle recorded no close-preview rows"
    oracle_by_at: dict[str, list[dict[str, Any]]] = {}
    for event in oracle:
        oracle_by_at.setdefault(str(event["details"]["at"]), []).append(event)
    assert set(oracle_by_at) == {"leg-start", "leg-exit"}, (
        f"serial oracle preview phases are not the D-20/DX-26 pair: {sorted(oracle_by_at)}"
    )
    template = {
        key: value
        for key, value in oracle[-1].items()
        if key not in ("event_id", "details", "status")
    }
    schema_by_at = {
        at: set(rows[-1]["details"]) for at, rows in oracle_by_at.items()
    }

    assert len(resumed) >= len(oracle), (
        "resumed campaign recorded fewer close-preview rows than the serial oracle"
    )
    assert len(resumed) <= len(oracle) + 2, (
        "resumed campaign recorded more extra close-preview rows than one kill "
        f"plus one resume can lawfully produce: {resumed}"
    )

    real_outcomes_by_anchor: dict[str, dict[str, Any]] = {}
    for outcome in resumed_outcomes:
        if outcome["details"]["synthesized"] is False:
            real_outcomes_by_anchor[str(outcome["details"]["leg_ended_sha256"])] = outcome

    closed_classes = {member.value for member in LegExitClass}
    seen_event_ids: set[str] = set()
    seen_anchors: set[str] = set()
    for event in resumed:
        envelope = {
            key: value
            for key, value in event.items()
            if key not in ("event_id", "details", "status")
        }
        assert envelope == template, (
            f"close-preview envelope diverged from the serial oracle: "
            f"{envelope!r} != {template!r}"
        )
        assert event["event_id"] not in seen_event_ids, f"duplicated close-preview row: {event}"
        seen_event_ids.add(event["event_id"])
        details = event["details"]
        verdict = details["verdict"]
        assert verdict in _CLOSE_PREVIEW_VERDICTS, (
            f"close-preview verdict outside the closed vocabulary: {verdict!r}"
        )
        expected_status = "healthy" if verdict == "would-close" else "degraded"
        assert event["status"] == expected_status, (
            f"close-preview status {event['status']!r} inconsistent with "
            f"verdict {verdict!r}: {details}"
        )
        at = details["at"]
        assert at in ("leg-start", "leg-exit"), (
            f"close-preview phase outside the D-20/DX-26 pair: {at!r}"
        )
        assert isinstance(details["generation_id"], str) and details[
            "generation_id"
        ].startswith("sha256:"), (
            f"close-preview generation_id is not a digest: {details['generation_id']!r}"
        )
        if verdict == "preview-error":
            assert isinstance(details.get("error"), str) and details["error"], (
                f"preview-error row carries no typed error: {details}"
            )
        else:
            assert set(details) == schema_by_at[at], (
                f"close-preview detail schema diverged from the serial oracle "
                f"for {at!r}: {sorted(details)}"
            )
        if at == "leg-exit":
            assert details["checkpoint"] == "leg-partial.v1", (
                f"leg-exit preview is not the DX-26 checkpoint: {details}"
            )
            assert details["exit_class"] in closed_classes, (
                f"checkpoint exit class outside the closed vocabulary: {details['exit_class']!r}"
            )
            anchor = details["leg_ended_sha256"]
            assert isinstance(anchor, str) and anchor.startswith("sha256:"), (
                f"checkpoint leg_ended_sha256 is not a digest: {anchor!r}"
            )
            assert anchor not in seen_anchors, (
                f"two leg-partial checkpoints anchor one leg exit: {anchor}"
            )
            seen_anchors.add(anchor)
            outcome = real_outcomes_by_anchor.get(anchor)
            assert outcome is not None, (
                f"leg-partial checkpoint anchors no real leg-outcome row: {details}"
            )
            assert outcome["details"]["generation_id"] == details["generation_id"], (
                f"checkpoint generation diverged from its anchored leg exit: {details}"
            )
            assert outcome["details"]["exit_class"] == details["exit_class"], (
                f"checkpoint exit class diverged from its anchored leg exit: {details}"
            )

    oracle_start = {
        key: value
        for key, value in oracle_by_at["leg-start"][0].items()
        if key != "event_id"
    }
    assert any(
        {key: value for key, value in event.items() if key != "event_id"} == oracle_start
        for event in resumed
    ), "the serial oracle's leg-start preview was not reproduced by the resumed campaign"

    final = resumed[-1]["details"]
    oracle_final = oracle_by_at["leg-exit"][-1]["details"]
    assert final["at"] == "leg-exit", (
        f"resumed campaign's final preview row is not its close checkpoint: {final}"
    )
    assert final["verdict"] == oracle_final["verdict"], (
        f"resumed campaign's final preview verdict {final['verdict']!r} diverged "
        f"from the serial oracle's {oracle_final['verdict']!r}"
    )
    assert final["exit_class"] == oracle_final["exit_class"], (
        f"resumed campaign's final checkpoint closed {final['exit_class']!r}, "
        f"oracle closed {oracle_final['exit_class']!r}"
    )


def _partition_selection_disclosures(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split normalized operational events into (leg disclosures, the rest)."""

    disclosures = [
        event for event in events if event.get("event_kind") in _LEG_SELECTION_DISCLOSURE_KINDS
    ]
    rest = [
        event for event in events if event.get("event_kind") not in _LEG_SELECTION_DISCLOSURE_KINDS
    ]
    return disclosures, rest


def assert_operational_equals_oracle_modulo_leg_disclosures(
    resumed: list[dict[str, Any]], oracle: list[dict[str, Any]]
) -> None:
    """Assert operational equality under WS-1 clause 7 per-leg disclosures.

    Everything that is not a leg-start selection disclosure or a W6/OB-M12
    ``leg-outcome`` mirror must reproduce exactly.  The oracle's disclosures
    must be reproduced verbatim and in order, and every EXTRA disclosure the
    resumed campaign appended must report a clean selection -- zero
    dependency-stale members, nothing silently re-driven.  The resume leg's
    own census is thereby part of the durability proof, not comparison slack.
    Leg-outcome rows are leg-topology-dependent (Wave-2 batch: the killed
    leg's boot-reconciler orphan close and each extra leg's own close mirror
    lawfully differ from the serial run) and are held to their own W6
    invariants via :func:`assert_leg_outcomes_match_oracle_topology`.
    Close-preview rows (D-20 leg-start disclosure + DX-26 ``leg-partial.v1``
    exit checkpoint, Wave-3 batch) are equally leg-topology-dependent -- each
    checkpoint anchors its own leg's ``leg_ended_sha256`` -- and are held to
    their own kernel invariants via
    :func:`assert_close_previews_match_oracle_topology`.
    """

    resumed_disclosures, resumed_rest = _partition_selection_disclosures(resumed)
    oracle_disclosures, oracle_rest = _partition_selection_disclosures(oracle)
    resumed_outcomes, resumed_rest = _partition_leg_outcomes(resumed_rest)
    oracle_outcomes, oracle_rest = _partition_leg_outcomes(oracle_rest)
    resumed_previews, resumed_core = _partition_close_previews(resumed_rest)
    oracle_previews, oracle_core = _partition_close_previews(oracle_rest)
    assert resumed_core == oracle_core, "durable divergence from serial oracle in 'operational'"
    assert_leg_outcomes_match_oracle_topology(resumed_outcomes, oracle_outcomes)
    assert_close_previews_match_oracle_topology(
        resumed_previews, oracle_previews, resumed_outcomes
    )
    assert resumed_disclosures[: len(oracle_disclosures)] == oracle_disclosures, (
        "durable divergence from serial oracle in the leg selection disclosures"
    )
    for event in resumed_disclosures[len(oracle_disclosures) :]:
        details = event["details"]
        if event["event_kind"] == "work-selection-census":
            assert details["mode"] == "ordinary", (
                f"resume leg census reports non-ordinary mode: {details['mode']!r}"
            )
            assert details["stale_excluded"] == 0, (
                f"resume leg census reports stale-excluded members: {details}"
            )
        else:
            assert details["stale_count"] == 0, (
                f"resume leg staleness report is non-clean: {details}"
            )
            assert details["reasons_histogram"] == {}
            assert details["sample_ids"] == []


def files_below(path: Path) -> dict[str, str]:
    """Return relative-path -> content digest for every file below a root."""

    if not path.exists():
        return {}
    return {
        str(candidate.relative_to(path)): hash_bytes(candidate.read_bytes())
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file()
    }


def normalized_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip the wall-clock-tainted fields from artifact events for comparison."""

    normalized = []
    for event in events:
        row = {key: value for key, value in event.items() if key not in _VOLATILE_EVENT_FIELDS}
        normalized.append(row)
    return normalized


def normalized_operational(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip sequence and digest fields from operational events for comparison."""

    volatile = ("ledger_seq", "payload_sha256", "created_at")
    return [{key: value for key, value in event.items() if key not in volatile} for event in events]


def capture_durable_state(root: Path, paths: Any) -> dict[str, Any]:
    """Capture every durable artifact of a campaign, tolerating absent files.

    Parameters
    ----------
    root:
        Isolated campaign root.
    paths:
        Production :class:`DriverPaths` for the campaign.

    Returns
    -------
    dict[str, Any]
        Raw ledgers, materialization inventories, and driver-state content.
    """

    def _scan(path: Path) -> list[dict[str, Any]]:
        return scan_jsonl(path) if path.exists() else []

    driver_state: Optional[dict[str, Any]] = None
    if paths.driver_state.exists():
        driver_state = json.loads(paths.driver_state.read_text(encoding="utf-8"))
    return {
        "models": _scan(paths.ledgers.models),
        "attempts": _scan(paths.ledgers.attempts),
        "gates": _scan(paths.ledgers.gates),
        "artifact_events": _scan(paths.ledgers.artifacts),
        "operational": _scan(paths.operational_ledger),
        "repository_files": files_below(root / "menagerie"),
        "public_mirror_files": files_below(paths.runtime_root / "mirrors" / "public"),
        "driver_state": driver_state,
    }


def durable_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a captured state to the oracle-comparable durable projection."""

    return {
        "models": state["models"],
        "attempts": state["attempts"],
        "gates": state["gates"],
        "artifact_events": normalized_events(state["artifact_events"]),
        "operational": normalized_operational(state["operational"]),
        "repository_files": state["repository_files"],
        "public_mirror_files": state["public_mirror_files"],
        "driver_state": state["driver_state"],
    }


def assert_projection_equals_oracle(
    resumed: Mapping[str, Any],
    oracle: Mapping[str, Any],
    exclude: tuple[str, ...] = (),
) -> None:
    """Assert the resumed campaign converged to the serial no-kill projection.

    Parameters
    ----------
    resumed, oracle:
        Durable projections of the killed-then-resumed and serial campaigns.
    exclude:
        Keys whose divergence is a disclosed, loud artifact of the injected
        fault itself (for example operational failure events) rather than a
        silent divergence of campaign state.

    The ``operational`` key is compared leg-aware: WS-1 clause 7 makes every
    leg start disclose its selection census, and W6/OB-M12 (Wave-2 batch)
    mirrors every leg's end -- including the killed leg's boot-reconciler
    orphan close -- as a ``leg-outcome`` event, so the resumed campaign's
    extra legs may append extra disclosure pairs and extra leg-outcome rows,
    each held to its own invariants.  See
    :func:`assert_operational_equals_oracle_modulo_leg_disclosures`.
    """

    for key in oracle:
        if key in exclude:
            continue
        if key == "operational":
            assert_operational_equals_oracle_modulo_leg_disclosures(resumed[key], oracle[key])
            continue
        assert resumed[key] == oracle[key], f"durable divergence from serial oracle in {key!r}"


def assert_no_double_award(state: Mapping[str, Any]) -> None:
    """Assert at most one final artifact event exists per (stable_id, work_id)."""

    seen: set[tuple[str, str]] = set()
    for event in state["artifact_events"]:
        if event.get("event_kind") not in _FINAL_EVENT_KINDS:
            continue
        key = (str(event["stable_id"]), str(event["work_id"]))
        assert key not in seen, f"double award: two final artifact events for {key!r}"
        seen.add(key)


def assert_artifact_model_bijection(root: Path, state: Mapping[str, Any]) -> None:
    """Prove awarded models and final artifact events reference the same bytes.

    Every model row bearing artifact authority must resolve its committed
    event, agree with it on transaction, authorization, reconstruction, and
    claims, and every claimed public materialization must exist byte-exact.
    Conversely every final event must be claimed by exactly one model row.
    """

    models = state["models"]
    final_events = [
        event for event in state["artifact_events"] if event.get("event_kind") in _FINAL_EVENT_KINDS
    ]
    events_by_id = {event["artifact_event_id"]: event for event in final_events}
    awarded = [model for model in models if model.get("artifact_authority")]
    assert {model["stable_id"] for model in awarded} == {
        event["stable_id"] for event in final_events
    }
    claimed_event_ids: set[str] = set()
    for model in awarded:
        authority = model["artifact_authority"]
        event = events_by_id[authority["committed_event_id"]]
        claimed_event_ids.add(str(authority["committed_event_id"]))
        assert authority["state"] == event["event_kind"]
        assert authority["transaction_id"] == event["transaction_id"]
        assert authority["authorization_id"] == event["authorization_id"]
        assert authority["reconstruction_sha256"] == event["reconstruction"]["sha256"]
        assert authority["claim_ids"] == sorted(claim["claim_id"] for claim in event["claims"])
        objects = {obj["object_id"]: obj for obj in event["objects"]}
        for claim in event["claims"]:
            obj = objects[claim["object_id"]]
            if obj["mirror_class"] != "public":
                continue
            materialized = root / claim["logical_path"]
            assert materialized.is_file()
            assert hash_bytes(materialized.read_bytes()) == obj["content_sha256"]
            assert materialized.stat().st_size == obj["byte_count"]
    assert claimed_event_ids == set(events_by_id)


def sigkill_self() -> None:
    """Deliver an uncatchable SIGKILL to the calling process."""

    os.kill(os.getpid(), signal.SIGKILL)


def install_sigkill_on_append_model(nth: int = 1) -> None:
    """SIGKILL inside the check-free award window: publication done, no model row.

    ``CanonicalReducer.append_model`` is the sole models-ledger writer inside
    both atomic award-commit sections (run award and terminal award); dying on
    its ``nth`` invocation leaves the final artifact event durable with no
    matching model row -- the headline half-done award.
    """

    import menagerie.crawler.reducer as reducer_module

    state = {"calls": 0}
    original = reducer_module.CanonicalReducer.append_model

    def killed(self: Any, model: Mapping[str, Any]) -> Any:
        state["calls"] += 1
        if state["calls"] >= nth:
            sigkill_self()
        return original(self, model)

    reducer_module.CanonicalReducer.append_model = killed  # type: ignore[method-assign]


def install_sigkill_before_final_artifact_event() -> None:
    """SIGKILL mid-publication: reconstruction committed, final event never lands."""

    import menagerie.crawler.artifact_transactions as transactions_module

    original = transactions_module.ArtifactEventLedger.append_event

    def killed(self: Any, event: Mapping[str, Any]) -> Any:
        if event.get("event_kind") in _FINAL_EVENT_KINDS:
            sigkill_self()
        return original(self, event)

    transactions_module.ArtifactEventLedger.append_event = killed  # type: ignore[method-assign]


def install_sigkill_before_close_state_write() -> None:
    """SIGKILL during close: partition asserted, close status never persisted."""

    import menagerie.crawler.driver as driver_module

    original = driver_module._write_driver_state

    def killed(path: Path, state: Mapping[str, Any]) -> None:
        status = str(state.get("status", ""))
        if (
            status == "complete"
            or status.startswith("complete-with")
            or status.startswith("terminal-partition-complete")
        ):
            sigkill_self()
        original(path, state)

    driver_module._write_driver_state = killed  # type: ignore[assignment]


def install_sigkill_after_park_row_before_stash() -> None:
    """SIGKILL between the durable grant-park append and its in-run stash/event."""

    import menagerie.crawler.driver_admission as admission_module
    from menagerie.crawler.grant_parks import GRANT_PENDING_DISPOSITION

    def killed(driver: Any, stable_id: str, row: Mapping[str, Any]) -> None:
        del driver, stable_id
        if row.get("disposition") == GRANT_PENDING_DISPOSITION:
            sigkill_self()

    admission_module._stash_machine_deferral_row = killed  # type: ignore[assignment]


def install_enospc_short_write_on_ledger(ledger_path: Path) -> None:
    """Simulate ENOSPC mid-append: the kernel accepts a prefix, then refuses.

    Intercepts the production append-mode open of exactly ``ledger_path`` and
    hands back a writer that persists the first half of the record line before
    raising ``OSError(ENOSPC)`` -- the partial-line torn tail a full disk
    produces through ``JsonlLedger.append``'s unchecked buffered write.
    """

    import errno
    import pathlib

    target = str(ledger_path)
    original_open = pathlib.Path.open

    class _ShortWriter:
        """File proxy that tears the first write in half and reports ENOSPC."""

        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def write(self, data: bytes) -> int:
            self._handle.write(data[: max(1, len(data) // 2)])
            self._handle.flush()
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), target)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._handle, name)

        def __enter__(self) -> "_ShortWriter":
            self._handle.__enter__()
            return self

        def __exit__(self, *exc_info: Any) -> Any:
            return self._handle.__exit__(*exc_info)

    def opened(self: pathlib.Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = original_open(self, mode, *args, **kwargs)
        if str(self) == target and mode == "ab":
            return _ShortWriter(handle)
        return handle

    pathlib.Path.open = opened  # type: ignore[method-assign]


INJECTIONS: dict[str, Callable[..., None]] = {
    "sigkill-on-append-model": install_sigkill_on_append_model,
    "sigkill-before-final-artifact-event": install_sigkill_before_final_artifact_event,
    "sigkill-before-close-state-write": install_sigkill_before_close_state_write,
    "sigkill-after-park-row-before-stash": install_sigkill_after_park_row_before_stash,
}


def write_child_outcome(root: Path, outcome: Mapping[str, Any]) -> None:
    """Persist the forked child's terminal observation for the parent."""

    (root / CHILD_OUTCOME_MARKER).write_text(
        json.dumps(dict(outcome), sort_keys=True), encoding="utf-8"
    )


def read_child_outcome(root: Path) -> Optional[dict[str, Any]]:
    """Read the forked child's terminal observation, if it survived to write one."""

    marker = root / CHILD_OUTCOME_MARKER
    if not marker.exists():
        return None
    return json.loads(marker.read_text(encoding="utf-8"))


def run_forked(target: Callable[..., None], args: tuple[Any, ...], timeout: float = 240.0) -> int:
    """Run ``target(*args)`` in a forked child and return its raw exit code."""

    process = multiprocessing.get_context("fork").Process(target=target, args=args)
    process.start()
    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        raise AssertionError("forked durability child did not exit within its deadline")
    assert process.exitcode is not None
    return process.exitcode


def recreate_campaign_dir(root: Path) -> None:
    """Reset one campaign directory to a byte-identical empty starting point."""

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
