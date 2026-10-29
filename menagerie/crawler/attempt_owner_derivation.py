"""Replay-derived owner authority for driver-lane attempt failures (B3-XV-4).

:data:`~menagerie.crawler.driver_receipts.WORKER_FAILURE_REASON_OWNERS` is
deliberately worker-scoped: it adjudicates only the reasons the WORKER AREA can
mint. The 200a attempts ledger also carries 267 failure rows minted by the
driver/admission lanes (author sessions, environment lifecycle, evidence
identity checks, disk recovery) whose reasons that map never covered, so the
B3-XV-4 attempts-owner census read them as silent owner-less residue -- the
standing RED of the Wave-4 rehearsal receipt.

This module is the SECOND closed owner authority for exactly those rows. Its
design follows the four-part waiver doctrine's replay rules:

* **Replay-derived, zero producer steering.** The owner is recomputed on every
  census run from the row's own recorded ``(stage, reason_code,
  exception_type)`` triple -- plus, for the session-allowance triple, the paid
  ledger's settlement evidence -- never from any persisted owner field a
  producer could backfill. (The live per-attempt ``details.owner`` stamp is a
  separate, higher authority consulted by ``_failure_owner`` before this
  module is ever reached.)
* **Closed proof rules.** Eligibility is an exact enumerated triple table.
  Widening any table is a new deliberated change with its own review
  (``test_static_triple_table_is_exactly_the_adjudicated_set`` pins the exact
  contents); an unknown triple derives NOTHING and keeps the census red.
* **Disposition-honest.** A triple whose owner is genuinely unknowable from
  the row alone gets a typed unowned-disposition the census must DISCLOSE,
  segregated from owned rows -- never a guessed owner, never silence.

Doctrine anchors, per rule family:

* ``author``-owned rows are author claims or published results refused against
  machine authority (``malformed-result`` arms in ``driver_admission``; the
  AL-6 ``blocked-prerequisite`` claim; the ``coverage-incomplete`` claimed/
  computed identity mismatch) and genuine effort exhaustion (the T2-07/T1-13
  session-allowance rule and the per-session wall claim). Author-owned
  failures keep their terminal routes (BL-12(e)).
* ``machinery``-owned rows are the crawler's own machinery killing the run:
  the corpus-conflict ``LedgerConflictError`` class (see
  ``infra_release._CORPUS_CONFLICT_EXCEPTION_TYPE``), ``DriverIntegrationError``
  program defects ("a broken program is not a fact about any model",
  ``driver_progress``), and the environment intent-setup classes
  (``EnvironmentExactnessError``/``EnvironmentSolveError``) whose old
  ``(Exception, "environment", "build-failed")`` catch-all minted intent-wide
  MODEL terminals for pure host conditions (T5-2 mutation-verified, T3-T2
  3-lab) before the routing registry abolished it.
* ``host``-owned rows are host state: ``DiskRecoveryError`` teardown residue
  (the abolished ``failed:resource/disk-floor`` terminal, T3-T2c) and
  storm-burned session allowances (restorable infrastructure weather).
* The two unowned session-crash dispositions are the honest remainder: a
  no-publication session death is "otherwise indistinguishable from a broken
  host" (``AuthorSessionRetryExhausted`` docstring, BATON trap 3) and the
  contemporaneous lane-health witness does not travel on the attempt row; an
  untyped ``builtins.ValueError`` transport crash carries no adjudicable cause
  at all.

``host``/``machinery`` owners derived here fall under
:data:`~menagerie.crawler.driver_receipts.NEVER_MODEL_TERMINAL_OWNERS`
semantics for any DOWNSTREAM consumer, but this module changes no live
classification or terminal route: it is census/replay authority only, and the
records write (if any) happens under the operator packet at final-sweep step 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

__all__ = [
    "ATTEMPT_TRIPLE_OWNERS",
    "SESSION_ALLOWANCE_EXHAUSTION_TRIPLE",
    "UNOWNED_ATTEMPT_DISPOSITIONS",
    "UNOWNED_DISPOSITION_VOCABULARY",
    "AttemptOwnerDerivation",
    "derive_attempt_failure_owner",
]

#: Closed static owner table: exact ``(stage, reason_code, exception_type)``
#: triples, one row per adjudicated 200a evidence class. A triple absent here
#: (and from the two structures below) derives nothing -- silent residue that
#: keeps the census red until its own deliberated adjudication.
ATTEMPT_TRIPLE_OWNERS: Mapping[tuple[str, str, str], str] = {
    # Author claims / published results refused against machine authority.
    (
        "author",
        "effort-exhausted:wall-seconds",
        "menagerie.crawler.author_dispatch.AuthorEffortExhaustionClaim",
    ): "author",
    (
        "author",
        "malformed-result",
        "menagerie.crawler.artifact_transactions.ArtifactBindingError",
    ): "author",
    (
        "author",
        "malformed-result",
        "menagerie.crawler.author_dispatch.AuthorExecutorContractError",
    ): "author",
    (
        "author",
        "malformed-result",
        "menagerie.crawler.driver_contracts.AuthorBlockedPrerequisite",
    ): "author",
    (
        "environment",
        "blocked-prerequisite",
        "menagerie.crawler.driver_contracts.AuthorBlockedPrerequisite",
    ): "author",
    (
        "evidence",
        "coverage-incomplete",
        "menagerie.crawler.driver_contracts.DriverIntegrationError",
    ): "author",
    # The crawler's own machinery killed or refused the run.
    (
        "author",
        "session-crashed",
        "menagerie.crawler.recordio.LedgerConflictError",
    ): "machinery",
    # Item 9 note: since the fix ``_author_lane_failure`` routes the same
    # LedgerConflictError class to (runner, internal-error), whose owner the
    # WORKER map already adjudicates (``internal-error`` -> machinery); this
    # table stays disjoint from that map by doctrine, so no triple is added.
    (
        "author",
        "session-crashed",
        "menagerie.crawler.driver_contracts.DriverIntegrationError",
    ): "machinery",
    (
        "environment",
        "build-failed",
        "menagerie.crawler.driver_contracts.DriverIntegrationError",
    ): "machinery",
    (
        "environment",
        "build-failed",
        "menagerie.crawler.env_lifecycle.EnvironmentExactnessError",
    ): "machinery",
    (
        "environment",
        "solve-failed",
        "menagerie.crawler.env_lifecycle.EnvironmentSolveError",
    ): "machinery",
    # Host state.
    (
        "resource",
        "disk-floor",
        "menagerie.crawler.env_lifecycle.DiskRecoveryError",
    ): "host",
}

#: The one triple whose owner needs settlement evidence: the T2-07/T1-13 rule
#: (``author_spend.author_sessions_exhaustion_dimension``) decides per row
#: whether the generation's allowance was genuinely spent authoring (owner
#: ``author``) or burned by a machine-classified provider storm (owner
#: ``host``). Uncertainty fails toward ``author`` -- never toward the
#: restorable class.
SESSION_ALLOWANCE_EXHAUSTION_TRIPLE: tuple[str, str, str] = (
    "author",
    "effort-exhausted:author-sessions",
    "menagerie.crawler.driver_contracts.AuthorEffortCapExceeded",
)

#: Closed typed unowned dispositions: triples whose owner is genuinely
#: unknowable from the attempt row alone. The census DISCLOSES these rows,
#: segregated from every owned bucket; disclosure is a lawful terminal state
#: (GREEN-or-disclosed), silence is not.
UNOWNED_ATTEMPT_DISPOSITIONS: Mapping[tuple[str, str, str], str] = {
    (
        "author",
        "session-crashed",
        "menagerie.crawler.driver_contracts.AuthorSessionRetryExhausted",
    ): "unowned-historical:no-publication-session-death",
    (
        "author",
        "session-crashed",
        "builtins.ValueError",
    ): "unowned-historical:untyped-session-crash",
}

#: The closed disposition vocabulary (census consumers branch on these values).
UNOWNED_DISPOSITION_VOCABULARY: frozenset[str] = frozenset(
    UNOWNED_ATTEMPT_DISPOSITIONS.values()
)


@dataclass(frozen=True)
class AttemptOwnerDerivation:
    """One row's replay-derived ownership verdict.

    Exactly one of ``owner`` / ``disposition`` is set: an owned row names its
    actor, an unowned row names its typed disclosed disposition.

    Attributes
    ----------
    owner:
        Closed five-actor owner, or ``None`` for a disclosed unowned row.
    disposition:
        Closed ``unowned-historical:*`` disposition, or ``None`` for an owned
        row.
    rule:
        The closed rule that decided: ``"closed-triple"``,
        ``"session-allowance-dimension:<dimension>"``, or
        ``"unowned-disposition"``.
    """

    owner: Optional[str]
    disposition: Optional[str]
    rule: str


def _row_triple(error: Mapping[str, Any]) -> Optional[tuple[str, str, str]]:
    """Return the row's exact evidence triple, or ``None`` when any leg is
    absent or non-string (fail closed: a malformed row derives nothing)."""

    stage = error.get("stage")
    reason = error.get("reason_code")
    exception_type = error.get("exception_type")
    if (
        isinstance(stage, str)
        and isinstance(reason, str)
        and isinstance(exception_type, str)
    ):
        return (stage, reason, exception_type)
    return None


def derive_attempt_failure_owner(
    error: Mapping[str, Any],
    *,
    stable_id: str,
    work_id: str,
    spend_events: Iterable[Mapping[str, Any]],
) -> Optional[AttemptOwnerDerivation]:
    """Derive one failed attempt row's owner from its recorded evidence.

    Parameters
    ----------
    error:
        The attempt row's structured ``error`` mapping (v3 shape).
    stable_id, work_id:
        The row's exact model and work-generation identities (needed only by
        the session-allowance triple's settlement replay).
    spend_events:
        Merged paid-ledger reservation/settlement rows for the same records
        root. An empty iterable is valid evidence-absence: the settlement rule
        then fails toward ``author``.

    Returns
    -------
    AttemptOwnerDerivation | None
        The closed derivation, or ``None`` when the triple is outside every
        closed authority (the caller must keep such rows as silent owner-less
        residue -- never green, never guessed).
    """

    triple = _row_triple(error)
    if triple is None:
        return None
    owner = ATTEMPT_TRIPLE_OWNERS.get(triple)
    if owner is not None:
        return AttemptOwnerDerivation(owner=owner, disposition=None, rule="closed-triple")
    if triple == SESSION_ALLOWANCE_EXHAUSTION_TRIPLE:
        from menagerie.crawler.author_spend import author_sessions_exhaustion_dimension

        dimension = author_sessions_exhaustion_dimension(
            spend_events, str(stable_id), str(work_id)
        )
        return AttemptOwnerDerivation(
            owner="host" if dimension == "provider-storm" else "author",
            disposition=None,
            rule=f"session-allowance-dimension:{dimension}",
        )
    disposition = UNOWNED_ATTEMPT_DISPOSITIONS.get(triple)
    if disposition is not None:
        return AttemptOwnerDerivation(
            owner=None, disposition=disposition, rule="unowned-disposition"
        )
    return None
