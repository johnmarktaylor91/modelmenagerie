"""Typed author-refusal accounting on canonical attempt rows (C-REFUSAL-SCHEMA).

S1_T4 T4-T4: an author content refusal used to land as ``failed:author`` /
``session-crashed`` (or ``malformed-result``) with NO ``failure_class`` on the
canonical attempt row and the refusal prose surviving only in gitignored
scratch -- the queued 200/500-rung salvage audit had an UNDIFFERENTIATED cohort
to read. These tests pin the fix: a closed refusal vocabulary
(``content-refusal`` reason code, ``SESSION_FAILURE_CLASSES``), executor-side
classification with a machine-built structured notice, additive attempt-v3
fields (``error.failure_class``, ``error.failure_class_basis``,
``error.refusal_excerpt`` -- bounded, never the full content), and the T4/B5
rider: ``retries.stage_attempt`` reflects the sessions actually consumed
instead of a hardcoded ``1`` that undercounted session spend 3x.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pytest

from menagerie.crawler.author_executor import (
    SessionOutcome,
    session_failure_class,
    session_failure_class_basis,
    session_failure_evidence,
)
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION_V3,
    FAILURE_REASON_CODES,
    REFUSAL_EXCERPT_MAX_CHARS,
    SESSION_FAILURE_CLASSES,
)
from menagerie.crawler.driver_admission import (
    _run_operator_command,
    classify_author_exit,
)
from menagerie.crawler.driver_contracts import (
    AuthorArtifact,
    DriverConfig,
    RetryableOperatorError,
    WorkItem,
)
from menagerie.crawler.driver_models import _driver_failure_attempt
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.routing import EnvironmentPhase, IntentRoute
from menagerie.crawler.schema import load_schema, owned_schema_leaves, validate_payload
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    _driver,
    _paths,
    _snapshot,
)

#: A realistic author-session refusal: harness envelope success, refusal prose
#: in ``result``, rc=0, and no published result file. This is the A5-audit
#: "content refusal" shape the fake-provider corpus lacked.
_REFUSAL_TEXT = (
    "I can't help with authoring this model. Based on the paper's own"
    " description, this architecture is purpose-built for generating"
    " non-consensual imagery of identifiable people, and providing a working"
    " implementation would directly enable that use. I'm happy to help with"
    " other architectures in the intake instead."
)


def _outcome(
    harness: Optional[dict],
    *,
    returncode: int = 0,
    stdout_tail: str = "",
) -> SessionOutcome:
    """Build one machine-observed session outcome for classification tests."""

    return SessionOutcome(
        harness=harness,
        returncode=returncode,
        timed_out=False,
        wall_seconds=42.0,
        session_id="4232a8da-6579-4db4-96b2-a8602086b57f",
        argv=("claude", "-p", "brief"),
        stderr_tail="",
        stdout_tail=stdout_tail,
    )


def _refusal_outcome(text: str = _REFUSAL_TEXT) -> SessionOutcome:
    """Return the observed refusal shape: success envelope, refusal in result."""

    return _outcome(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "duration_ms": 61_842,
            "num_turns": 4,
            "total_cost_usd": 0.31,
            "result": text,
        }
    )


# -- closed vocabulary ------------------------------------------------------


@pytest.mark.smoke
def test_content_refusal_is_a_closed_author_reason() -> None:
    """The author stage vocabulary carries the typed refusal reason.

    Without a vocabulary member the reducer's stage/reason tripwire
    (``attempt error reason is not allowed for its stage``) makes a truthful
    refusal row unrecordable, which is exactly the laundering-onto-
    ``session-crashed``/``malformed-result`` mechanism T4-T4 indicts.
    """

    assert "content-refusal" in FAILURE_REASON_CODES["author"]
    reason_enum = load_schema(ATTEMPT_SCHEMA_VERSION_V3)
    common = reason_enum.get("$defs", {}).get("reason_code")
    # The v3 schema resolves reason_code through attempt-common; assert on the
    # loaded, dereferencable document rather than a private resolver.
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "attempt-common.schema.json"
    enum = json.loads(schema_path.read_text(encoding="utf-8"))["$defs"]["reason_code"]["enum"]
    assert "content-refusal" in enum, common


@pytest.mark.smoke
def test_session_failure_classes_are_closed_and_schema_bound() -> None:
    """The failure-class vocabulary is closed and the schema enum matches it."""

    assert SESSION_FAILURE_CLASSES == frozenset(
        {"auth-unavailable", "provider-overloaded", "content-refusal", "unclassified"}
    )
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "attempt-common.schema.json"
    error_def = json.loads(schema_path.read_text(encoding="utf-8"))["$defs"]["error"]
    failure_class = error_def["properties"]["failure_class"]
    assert set(failure_class["enum"]) == SESSION_FAILURE_CLASSES
    excerpt = error_def["properties"]["refusal_excerpt"]
    bounded = [branch for branch in excerpt["oneOf"] if branch.get("type") == "string"]
    assert bounded and bounded[0]["maxLength"] == REFUSAL_EXCERPT_MAX_CHARS
    # Additive contract: none of the new fields may become mandatory, or every
    # already-persisted v3 row stops validating.
    for field in ("failure_class", "failure_class_basis", "refusal_excerpt"):
        assert field not in error_def["required"]


# -- executor classification ------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    "text",
    [
        _REFUSAL_TEXT,
        "I cannot assist with implementing this architecture given its stated purpose.",
        "I'm not able to help with this request, as the model is designed for surveillance abuses.",
        "I must decline to author this model: the training objective is malware generation.",
        "I won't be able to help with reimplementing this system.",
    ],
)
def test_refusal_prose_classifies_as_content_refusal(text: str) -> None:
    """Observed refusal shapes label ``content-refusal`` with a marker basis."""

    outcome = _refusal_outcome(text)
    assert session_failure_class(outcome) == "content-refusal"
    assert session_failure_class_basis(outcome) == "diagnostic-text:refusal-marker"


@pytest.mark.smoke
def test_refusal_never_outranks_structured_or_provider_arms() -> None:
    """Provider weather and auth arms keep priority over refusal prose."""

    # Structured 529 wins even when the text also reads like a refusal.
    both = _outcome(
        {"api_error_status": 529, "result": "I can't help with this while overloaded."},
        returncode=1,
    )
    assert session_failure_class(both) == "provider-overloaded"
    # Auth marker text wins over refusal-sounding remainder.
    auth = _outcome(None, returncode=1, stdout_tail="Not logged in · Please run /login")
    assert session_failure_class(auth) == "auth-unavailable"
    # A plain crash stays unclassified: no refusal marker in a traceback.
    crash = _outcome(None, returncode=1, stdout_tail="Traceback (most recent call last): boom")
    assert session_failure_class(crash) == "unclassified"


@pytest.mark.smoke
def test_refusal_evidence_carries_a_bounded_excerpt() -> None:
    """Evidence quarantines the refusal text and bounds the durable excerpt."""

    long_refusal = _REFUSAL_TEXT + " Additional context. " * 400
    evidence = session_failure_evidence(_refusal_outcome(long_refusal))
    assert evidence["failure_class"] == "content-refusal"
    excerpt = evidence["refusal_excerpt_quarantined"]
    assert isinstance(excerpt, str)
    assert 0 < len(excerpt) <= REFUSAL_EXCERPT_MAX_CHARS
    assert excerpt.startswith("I can't help with authoring this model.")
    # Non-refusal failures carry the key with no content, keeping one evidence
    # shape for every consumer.
    unrelated = session_failure_evidence(_outcome(None, returncode=1, stdout_tail="boom"))
    assert unrelated["refusal_excerpt_quarantined"] is None


# -- attempt row builder ----------------------------------------------------


def _work_item(stable_id: str = "m_refusal") -> WorkItem:
    """Return a minimal routed work item; only its intake identity is read here."""

    intake = IntakeItem(
        stable_id=stable_id,
        name="RefusalNet",
        zoo="fixtures",
        variant="base",
        discovery_source="crawl_roster",
        legacy_row_sha256="0" * 64,
        preserved_legacy_flags=(),
        variant_scope="standalone",
        family_representative_id=stable_id,
    )
    return WorkItem(
        intake=intake,
        route=IntentRoute(stable_id=stable_id, intent="core", phase=EnvironmentPhase.PYTORCH),
    )


def _driver_config() -> DriverConfig:
    """Return a minimal driver config for row-builder tests."""

    return DriverConfig(campaign_id="c2-disco", machine_id="fixture-machine")


def _refusal_attempt_row(tmp_path: Path, *, stage_attempt: int = 3) -> dict:
    """Build one driver-observed refusal attempt through the real builder."""

    return _driver_failure_attempt(
        _work_item(),
        None,
        "author",
        "content-refusal",
        RuntimeError("author sessions refused on content grounds"),
        _driver_config(),
        diagnostics_root=tmp_path / ".crawl-local" / "diagnostics",
        environment=None,
        created_at="2026-08-11T00:00:00Z",
        stage_attempt=stage_attempt,
        session_failure={
            "failure_class": "content-refusal",
            "failure_class_basis": "diagnostic-text:refusal-marker",
            "refusal_excerpt": _REFUSAL_TEXT,
        },
    )


@pytest.mark.smoke
def test_driver_failure_attempt_carries_typed_refusal_fields(tmp_path: Path) -> None:
    """The canonical row itself says refusal: class, basis, bounded excerpt."""

    attempt = _refusal_attempt_row(tmp_path)
    error = attempt["error"]
    assert error["reason_code"] == "content-refusal"
    assert error["failure_class"] == "content-refusal"
    assert error["failure_class_basis"] == "diagnostic-text:refusal-marker"
    assert error["refusal_excerpt"] == _REFUSAL_TEXT
    assert len(error["refusal_excerpt"]) <= REFUSAL_EXCERPT_MAX_CHARS
    # B5: the row accounts every session the ladder consumed, not always 1.
    assert attempt["retries"]["stage_attempt"] == 3
    ledger_shape = {**attempt, "ledger_seq": 1, "payload_sha256": "sha256:" + "0" * 64}
    validate_payload(ledger_shape, ATTEMPT_SCHEMA_VERSION_V3)


@pytest.mark.smoke
def test_refusal_excerpt_is_bounded_and_survives_redaction(tmp_path: Path) -> None:
    """The excerpt stays a durable bounded string, never a sidecar pointer.

    ``_redact_attempt_diagnostics`` replaces externally controlled fields
    (``message``, ``traceback``, ...) with gitignored sidecar references --
    exactly the mechanism that made refusal prose unreadable after host
    teardown. The bounded excerpt is the deliberate, narrow exception: it must
    remain plain text on the canonical row while the message/traceback still
    redact.
    """

    oversized = _REFUSAL_TEXT + " padding sentence." * 200
    attempt = _driver_failure_attempt(
        _work_item(),
        None,
        "author",
        "content-refusal",
        RuntimeError("author sessions refused on content grounds"),
        _driver_config(),
        diagnostics_root=tmp_path / ".crawl-local" / "diagnostics",
        environment=None,
        created_at="2026-08-11T00:00:00Z",
        stage_attempt=2,
        session_failure={
            "failure_class": "content-refusal",
            "failure_class_basis": "diagnostic-text:refusal-marker",
            "refusal_excerpt": oversized,
        },
    )
    error = attempt["error"]
    assert isinstance(error["refusal_excerpt"], str)
    assert len(error["refusal_excerpt"]) == REFUSAL_EXCERPT_MAX_CHARS
    # The neighboring externally-controlled fields still redact to references.
    assert isinstance(error["message"], dict)
    assert error["message"]["redaction"] == "externally-controlled-text-v1"


@pytest.mark.smoke
def test_unrecognized_session_class_folds_to_unclassified(tmp_path: Path) -> None:
    """One odd class string can never make a terminal row unrecordable.

    The mapping is total, mirroring ``_blocked_terminal``'s doctrine: an
    unknown label folds to ``unclassified`` (the schema enum stays closed)
    while the basis preserves what was actually claimed.
    """

    attempt = _driver_failure_attempt(
        _work_item(),
        None,
        "author",
        "session-crashed",
        RuntimeError("session died"),
        _driver_config(),
        diagnostics_root=tmp_path / ".crawl-local" / "diagnostics",
        environment=None,
        created_at="2026-08-11T00:00:00Z",
        session_failure={
            "failure_class": "totally-new-class",
            "failure_class_basis": None,
            "refusal_excerpt": None,
        },
    )
    error = attempt["error"]
    assert error["failure_class"] == "unclassified"
    assert error["failure_class_basis"] == "driver:unrecognized-class:totally-new-class"
    ledger_shape = {**attempt, "ledger_seq": 1, "payload_sha256": "sha256:" + "0" * 64}
    validate_payload(ledger_shape, ATTEMPT_SCHEMA_VERSION_V3)


@pytest.mark.smoke
def test_rows_without_session_classification_keep_the_prior_shape(
    tmp_path: Path,
) -> None:
    """The schema change is additive: classification-free rows stay identical."""

    attempt = _driver_failure_attempt(
        _work_item(),
        None,
        "runner",
        "internal-error",
        RuntimeError("engine fault"),
        _driver_config(),
        diagnostics_root=tmp_path / ".crawl-local" / "diagnostics",
        environment=None,
        created_at="2026-08-11T00:00:00Z",
    )
    error = attempt["error"]
    for field in ("failure_class", "failure_class_basis", "refusal_excerpt"):
        assert field not in error
    assert attempt["retries"]["stage_attempt"] == 1
    ledger_shape = {**attempt, "ledger_seq": 1, "payload_sha256": "sha256:" + "0" * 64}
    validate_payload(ledger_shape, ATTEMPT_SCHEMA_VERSION_V3)


@pytest.mark.smoke
def test_new_schema_leaves_carry_authority_ownership() -> None:
    """Every additive leaf is owned; the exhaustive registry stays exhaustive."""

    ownership = {
        leaf.path: leaf.owner.value for leaf in owned_schema_leaves(ATTEMPT_SCHEMA_VERSION_V3)
    }
    assert ownership["$.error.failure_class"] == "untrusted-history"
    assert ownership["$.error.failure_class_basis"] == "untrusted-history"
    assert ownership["$.error.refusal_excerpt"] == "untrusted-history"


# -- classify_author_exit plumbing ------------------------------------------


@pytest.mark.smoke
def test_classify_author_exit_stamps_refusal_fields_from_the_notice() -> None:
    """The executor notice's refusal fields ride the typed retryable failure."""

    notice = {
        "type": "result",
        "subtype": "author-session-failure",
        "failure_class": "content-refusal",
        "failure_class_basis": "diagnostic-text:refusal-marker",
        "harness_is_error": False,
        "harness_subtype": "success",
        "harness_terminal_reason": None,
        "returncode": 0,
        "session_error_text_quarantined": _REFUSAL_TEXT,
        "refusal_excerpt_quarantined": _REFUSAL_TEXT[:REFUSAL_EXCERPT_MAX_CHARS],
    }
    stdout = "author executor stage2 failed: content-refusal (attempt deadbeef)\n" + json.dumps(
        notice, sort_keys=True
    )
    with pytest.raises(RetryableOperatorError) as raised:
        classify_author_exit("author", "m_refusal", 75, stdout, "")
    assert raised.value.failure_class == "content-refusal"
    assert raised.value.failure_class_basis == "diagnostic-text:refusal-marker"
    assert raised.value.refusal_excerpt == _REFUSAL_TEXT[:REFUSAL_EXCERPT_MAX_CHARS]


# -- driver end-to-end: exhaustion writes the typed refusal terminal ---------


class ContentRefusalAuthor(FakeAuthor):
    """Author lane whose victim replays the executor's exact refusal bytes.

    Every victim session runs a REAL subprocess that prints the executor's
    structured failure line plus the machine-built ``author-session-failure``
    notice with ``failure_class: content-refusal`` and exits with the
    retryable operator code, exactly as the production executor does for a
    session that completed, refused in prose, and published nothing.
    """

    def __init__(self, victims: dict[str, str]) -> None:
        """Bind victim models to the stage whose session refuses."""

        super().__init__()
        self.victims = dict(victims)
        self.crashes: dict[str, int] = {}

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Replay the refusal exit for victims; delegate for healthy models."""

        stage = self.victims.get(item.stable_id)
        if stage is None:
            return super().author(item, work_root, config, context)
        self.crashes[item.stable_id] = self.crashes.get(item.stable_id, 0) + 1
        notice = {
            "type": "result",
            "subtype": "author-session-failure",
            "failure_class": "content-refusal",
            "failure_class_basis": "diagnostic-text:refusal-marker",
            "harness_is_error": False,
            "harness_subtype": "success",
            "harness_terminal_reason": None,
            "returncode": 0,
            "session_error_text_quarantined": _REFUSAL_TEXT,
            "refusal_excerpt_quarantined": _REFUSAL_TEXT[:REFUSAL_EXCERPT_MAX_CHARS],
        }
        line = (
            f"author executor {stage} failed: content-refusal (attempt deadbeef)\n"
            f"{json.dumps(notice, sort_keys=True)}"
        )
        completed = _run_operator_command(
            [
                sys.executable,
                "-c",
                "import sys; "
                f"sys.stdout.write({line!r}); sys.exit(75)",
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
        raise AssertionError("refusal subprocess classified as a clean exit")


@pytest.mark.smoke
def test_refusal_exhaustion_writes_the_typed_terminal_and_session_count(
    tmp_path: Path,
) -> None:
    """S1_T4 merge gate: the refusal shape lands durably distinguishable.

    The victim is the last authored model, so the lane-health witness exists
    when its bounded retry budget spends. Under W5 the member is CONTAINED
    (author-wave deferred-family park, no verdict -- OC-Q6) but the EVIDENCE
    contract is unchanged: the persisted attempt row must say
    ``content-refusal`` (never ``session-crashed``) with the class, basis,
    and bounded excerpt, and -- T4/B5 -- ``retries.stage_attempt`` must equal
    the number of author sessions the ladder actually consumed.
    """

    snapshot = _snapshot(tmp_path, count=3)
    victim = snapshot.items[2].stable_id
    author = ContentRefusalAuthor({victim: "stage2"})

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=1,
        sleeper=lambda seconds: None,
    ).run()

    assert result.status == "complete-with-deferrals"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    # W5/OC-Q6: no machine verdict -- the member is accountably parked.
    assert victim not in models
    from menagerie.crawler.scheduler_parks import (
        load_scheduler_quarantine_rows,
        scheduler_quarantine_path,
    )

    records_root = _paths(tmp_path, snapshot).ledgers.models.parent.parent
    (park,) = load_scheduler_quarantine_rows([scheduler_quarantine_path(records_root)])
    assert park["stable_id"] == victim
    assert park["scheduler_quarantine"]["site"] == "author-wave"

    attempts = [
        record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.attempts)
        if record.get("stable_id") == victim
    ]
    assert attempts, "the exhaustion terminal must persist its attempt row"
    row = attempts[-1]
    error = row["error"]
    assert error["reason_code"] == "content-refusal"
    assert error["failure_class"] == "content-refusal"
    assert error["failure_class_basis"] == "diagnostic-text:refusal-marker"
    excerpt = error["refusal_excerpt"]
    assert isinstance(excerpt, str)
    assert 0 < len(excerpt) <= REFUSAL_EXCERPT_MAX_CHARS
    assert excerpt.startswith("I can't help with authoring this model.")
    assert row["retries"]["stage_attempt"] == author.crashes[victim]
    assert author.crashes[victim] > 1, "the ladder must actually have retried"


@pytest.mark.smoke
def test_session_crash_exhaustion_accounts_every_session(tmp_path: Path) -> None:
    """T4/B5 on the plain crash path: stage_attempt is the real session count."""

    from menagerie.crawler.tests.test_driver_abort_boundary import (
        SubprocessKilledAuthor,
    )

    snapshot = _snapshot(tmp_path, count=3)
    victim = snapshot.items[2].stable_id
    author = SubprocessKilledAuthor({victim: "stage2"})

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=1,
        sleeper=lambda seconds: None,
    ).run()

    assert result.status == "complete-with-deferrals"
    attempts = [
        record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.attempts)
        if record.get("stable_id") == victim
    ]
    assert attempts, "the contained exhaustion must persist its attempt row"
    row = attempts[-1]
    assert row["error"]["reason_code"] == "session-crashed"
    assert row["retries"]["stage_attempt"] == author.crashes[victim]
    assert author.crashes[victim] > 1, "the ladder must actually have retried"
