"""Reducer admission-state hygiene under interleaved appends (C1 rehearsal).

``append_model`` pins the pre-append dependency projection on the reducer as
``_validation_current`` for the duration of one admission pass. The shipped
code released the pin only on the successful non-replay path, so EVERY refusal
(bad parentage, grant-pending park, award tripwire) and even the idempotent
duplicate early-return left it dangling. ``_validation_current_records()`` --
consumed by ``prepare_model``'s family-representative lookups and the gate and
execution pipelines -- then served that stale snapshot instead of the live
projection until the NEXT ``append_model`` happened to reassign it. Under the
M3 per-model pipelined scheduler (refused append for model A, gate/attempt
appends for model B, then ``prepare_model`` for model B) that is a stale-read
in the award path.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import (
    CanonicalReducer as ProductionCanonicalReducer,
    ReductionError,
)
from menagerie.crawler.tests.conftest import (
    make_attempt,
    make_failed_attempt,
    make_gate,
    make_model,
)
from menagerie.crawler.tests.test_reducer import (
    CanonicalReducer as AdapterCanonicalReducer,
    _attempt_for_model,
    _variant_from_representative,
)

#: Sub-second load-bearing admission-state coverage for the mandatory smoke gate.
pytestmark = pytest.mark.smoke


def _paths(tmp_path: Path) -> LedgerPaths:
    """Return isolated canonical ledger paths."""

    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def _persist_first_revision(reducer: AdapterCanonicalReducer, stable_id: str) -> dict:
    """Append gate, attempt, and one accepted revision for one model."""

    attempt_id = f"attempt-{stable_id}"
    attempt = make_attempt(stable_id, attempt_id=attempt_id)
    attempt.pop("ledger_seq", None)
    attempt.pop("payload_sha256", None)
    reducer.append_attempt(attempt)
    candidate = make_model(stable_id, accepted=True, attempt_id=attempt_id)
    candidate.pop("record_seq", None)
    candidate.pop("record_revision", None)
    return reducer.append_model(candidate).record


def test_refused_append_leaves_no_pinned_validation_projection(tmp_path: Path) -> None:
    """A refusal anywhere in the admission pass must release the projection pin."""

    stable_ids = ["m_0", "m_1"]
    with AdapterCanonicalReducer(_paths(tmp_path), stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        bad = make_model("m_1", accepted=True, attempt_id="attempt-none")
        bad.pop("record_seq", None)
        bad.pop("record_revision", None)
        bad["parent_revision"] = "sha256:" + "9" * 64
        with pytest.raises(ReductionError, match="bad parentage"):
            ProductionCanonicalReducer.append_model(reducer, bad)
        assert not hasattr(reducer, "_validation_current"), (
            "a refused append left its pre-append projection pinned"
        )
        # Outside an admission pass the validation view is the ledger head.
        # Outside an admission pass a LIVE reducer's validation view is the
        # freshly derived dependency projection (main's PC1-era contract; the
        # ledger-head identity return survives only for semantic replays), so
        # equality of content -- never object identity -- is the invariant.
        assert dict(reducer._validation_current_records()) == dict(reducer._current)


def test_idempotent_append_leaves_no_pinned_validation_projection(tmp_path: Path) -> None:
    """The logical-duplicate early return must release the projection pin too."""

    stable_ids = ["m_0", "m_1"]
    with AdapterCanonicalReducer(_paths(tmp_path), stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        persisted = _persist_first_revision(reducer, "m_0")
        replay = {
            key: value
            for key, value in deepcopy(persisted).items()
            if key not in {"record_seq", "record_revision"}
        }
        result = ProductionCanonicalReducer.append_model(reducer, replay)
        assert not result.appended
        assert not hasattr(reducer, "_validation_current"), (
            "an idempotent append left its pre-append projection pinned"
        )


def test_prepare_model_after_refused_append_reads_live_currency(tmp_path: Path) -> None:
    """The M3 interleaving shape: refusal, then appends, then ``prepare_model``.

    End-to-end regression guard for the scheduler interleaving (the RED
    carriers for the pin leak itself are the two attribute tests above): a
    refused append for one model must not poison the family-representative
    lookup of a later ``prepare_model`` -- the prepared variant binds the LIVE
    representative revision and appends cleanly.
    """

    gate_ids = ["m_rep", "m_1"]
    stable_ids = [*gate_ids, "m_variant"]
    with AdapterCanonicalReducer(
        _paths(tmp_path),
        stable_ids,
        intake_variant_bindings={"m_variant": ("m_rep", "Large")},
    ) as reducer:
        reducer.append_gate(make_gate(gate_ids))
        representative = _persist_first_revision(reducer, "m_rep")

        bad = make_model("m_1", accepted=True, attempt_id="attempt-none")
        bad.pop("record_seq", None)
        bad.pop("record_revision", None)
        bad["parent_revision"] = "sha256:" + "9" * 64
        with pytest.raises(ReductionError, match="bad parentage"):
            ProductionCanonicalReducer.append_model(reducer, bad)

        superseding = deepcopy(representative)
        superseding.pop("record_seq", None)
        superseding.pop("record_revision", None)
        superseding["parent_revision"] = representative["record_revision"]
        superseding["status"]["supersedes_revision"] = representative["record_revision"]
        superseding["notes"] = "metadata-only re-vet"
        new_representative = reducer.append_model(superseding).record

        variant = _variant_from_representative(new_representative)
        # The helper copies the representative; a first variant revision has no
        # supersession lineage of its own.
        variant["status"]["supersedes_revision"] = None
        variant_attempt = _attempt_for_model(variant, "attempt-variant")
        variant_attempt.pop("ledger_seq", None)
        variant_attempt.pop("payload_sha256", None)
        reducer.append_attempt(variant_attempt)
        prepared = reducer.prepare_model(variant)
        assert (
            prepared["dependency_vector"]["representative_revision"]
            == new_representative["record_revision"]
        )
        assert reducer.append_model(variant).appended


def test_attested_generation_families_keep_every_generation(tmp_path):
    """The m8830 circuit-breaker shape: older generations of a rebuilt family
    must stay resolvable from ledger evidence, unlike the one-slot currency map."""

    from menagerie.crawler.driver_admission import (
        attested_generation_families,
        restored_environment_generations,
    )

    attempts = tmp_path / "attempts.jsonl"
    rows = [
        {
            "environment": {"family": "core"},
            "identities": {"environment": "sha256:gen-old"},
        },
        {
            "environment": {"family": "core"},
            "identities": {"environment": "sha256:gen-new"},
        },
    ]
    attempts.write_text(
        "\n".join(__import__("json").dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    bindings = attested_generation_families(attempts, ())
    assert bindings == {"sha256:gen-old": "core", "sha256:gen-new": "core"}
    currency = restored_environment_generations({}, attempts, ())
    assert currency == {"core": "sha256:gen-new"}


def test_mid_leg_attested_generation_resolves_at_terminal_assembly(tmp_path):
    """The leg-28 ``TerminalRecordingUnavailable`` shape (m1163/m1221/m2062):
    a generation attested ONLY by ledger rows appended after the reducer's
    construction-time derivation (its family rebuilt mid-leg) must resolve at
    terminal assembly from the reducer's own live ledgers, never refuse as
    absent-from-active-context."""

    paths = _paths(tmp_path)
    rebuilt_generation = "sha256:" + "c" * 64
    with AdapterCanonicalReducer(paths, ["m_example"]) as reducer:
        # Construction saw EMPTY ledgers, so both binding snapshots miss the
        # rebuilt generation; the active context maps a different generation.
        assert rebuilt_generation not in reducer.historical_generation_families
        assert rebuilt_generation not in set(
            reducer.context.environment_generations.values()
        )
        reducer.append_gate(make_gate(["m_example"]))
        attempt = make_attempt()
        attempt["identities"]["environment"] = rebuilt_generation
        attempt["environment"]["family"] = "core"
        reducer.append_attempt(attempt)
        model = make_model(accepted=True)
        model["execution"]["env_generation"] = rebuilt_generation
        record = reducer.append_model(model).record
        assert record["status"]["kind"] == "runs"
        # The live re-derive is additive, exactly like driver registration.
        assert (
            reducer.historical_generation_families[rebuilt_generation] == "core"
        )


@pytest.mark.parametrize(
    ("stage", "reason"),
    (
        ("runner", "attestation-mismatch"),
        ("runner", "unclassified-supervised-failure"),
        ("policy", "telemetry-integrity"),
    ),
)
def test_bl12_reason_codes_are_reducer_appendable(tmp_path, stage, reason):
    """The m12646/m13258/m13279 quarantine loop: BL-12 added these codes to
    attempt-common.schema.json but missed the python-side FAILURE_REASON_CODES,
    so the reducer refused every worker attempt carrying them and the env-turn
    quarantined the model as a machinery fault on every leg."""

    paths = _paths(tmp_path)
    attempt = make_failed_attempt(stage=stage, reason_code=reason)
    with AdapterCanonicalReducer(paths, ["m_example"]) as reducer:
        assert reducer.append_attempt(attempt).appended


def test_every_emitted_failure_reason_is_in_the_closed_vocabulary():
    """Class tripwire for the BL-12 miss: every literal (stage, reason) pair an
    attempt-emitting module can produce must be a member of
    FAILURE_REASON_CODES -- a schema-only registration reads as valid in CI
    while the reducer refuses it in production."""

    import re
    from pathlib import Path as _Path

    from menagerie.crawler.constants import FAILURE_REASON_CODES

    crawler_root = _Path(__file__).parents[1]
    pairs: set[tuple[str, str]] = set()
    for module in (
        "driver_receipts.py",
        "driver_admission.py",
        "driver.py",
        "worker_supervisor.py",
        "scheduler.py",
    ):
        source = (crawler_root / module).read_text(encoding="utf-8")
        pairs.update(
            re.findall(r'_attempt_error_fields\(\s*"([a-z-]+)",\s*"([a-z:_-]+)"', source)
        )
        # The policy_reasons dispatch table emits variable reasons for the
        # fixed "policy" stage.
        for match in re.findall(r'\("([a-z_]+)",\s*"([a-z-]+)"\),\n', source):
            field, reason = match
            if field.endswith(("_attempted", "_present", "_failure")):
                pairs.add(("policy", reason))
    missing = sorted(
        (stage, reason)
        for stage, reason in pairs
        if reason not in FAILURE_REASON_CODES.get(stage, frozenset())
    )
    assert not missing, f"emitted reasons missing from FAILURE_REASON_CODES: {missing}"


def test_register_attested_generation_families_is_additive_and_typed():
    from menagerie.crawler.reducer import CanonicalReducer

    reducer = CanonicalReducer.__new__(CanonicalReducer)
    reducer.historical_generation_families = {}
    CanonicalReducer.register_attested_generation_families(
        reducer, {"sha256:gen-old": "core", "": "bad", "sha256:x": ""}
    )
    assert reducer.historical_generation_families == {"sha256:gen-old": "core"}


def test_model_lane_failed_event_ids_are_per_occurrence():
    """The third step-100 crash shape (2026-08-13): the same model failing the
    same way in a later resume is a NEW occurrence -- its event id must differ,
    or the append is a ledger 'conflicting replay' of the first occurrence."""

    from menagerie.crawler.driver_admission import _model_lane_failure_event

    exc = RuntimeError("checker verdict unavailable")
    first = _model_lane_failure_event(
        stable_id="m8830",
        work_id="sha256:work",
        status_code="failed:runner",
        reason_code="lane-death",
        exc=exc,
        run_id="crawler-run",
        machine_id="worker-a",
        created_at="2026-08-13T04:00:00.000000Z",
    )
    second = _model_lane_failure_event(
        stable_id="m8830",
        work_id="sha256:work",
        status_code="failed:runner",
        reason_code="lane-death",
        exc=exc,
        run_id="crawler-run",
        machine_id="worker-a",
        created_at="2026-08-13T09:00:00.000000Z",
    )
    assert first["event_id"] != second["event_id"]
    assert first["event_id"].startswith("model-lane-failed-")
