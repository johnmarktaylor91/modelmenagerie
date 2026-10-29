"""Mixed-era canonical authority admits end to end; incoherent writes refuse.

Rung-12 ran four legs into four members of ONE crash family: a cached
artifact condition (schema-stale result, live-basis identity recompute,
vector-vs-record contradiction) aborting the whole campaign at rehydration.
This module pins the two closing contracts of the sweep:

- WRITE side: an authorization whose accepted gate contradicts the model's
  dependency vector refuses per model at ``authorize_publication`` -- the
  leg-4 poison (m1359: fidelity-failed terminal authorized under its accurate
  metadata gate) can never be minted again.
- READ side, end to end: a campaign whose canonical authority spans THREE
  eras -- a pre-rotation artifact (recorded prompt basis rotated), a
  mid-rotation artifact (identity claims its facts no longer derive), and a
  current-era artifact (same world) -- admits through a full driver run and
  COMPLETES: the first two invalidate loudly to fresh authorship, the third
  reuses with zero author sessions, and nothing crashes the driver.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import menagerie.crawler.driver_admission as driver_admission_module
import menagerie.crawler.reducer as reducer_module
from menagerie.crawler.driver_admission import (
    ARTIFACT_AUTHORITY_IDENTITY_STALE_DISPOSITION,
)
from menagerie.crawler.driver_contracts import (
    DriverIntegrationError,
    RecordedIdentityBasis,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer, ReductionError
from menagerie.crawler.tests.test_artifact_transactions import _context, _vector
from menagerie.crawler.tests.test_artifact_identity_recorded_basis import (
    _crafted_artifact,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    _driver,
    _paths,
    _snapshot,
    _test_authority_context,
)


def test_authorization_gate_contradicting_the_vector_refuses_at_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The leg-4 poison is refused per model before any event is appended.

    m1359: the driver authorized a fidelity-failed terminal's artifact under
    its accurate METADATA gate while the reducer-derived vector pinned the
    decisive FIDELITY gate. The read path requires those to agree, so the
    write must refuse the pair instead of minting a transaction every later
    projection quarantines.
    """

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    context = _test_authority_context(snapshot, _driver(tmp_path, snapshot).config)
    raw_vector = asdict(
        _vector(
            _context("m1359"),
            "m1359",
            "txn-" + "a" * 24,
            "result-m1359",
            "sha256:" + "b" * 64,
            checker_gate_identity="gate-fidelity-generation-3",
        )
    )
    monkeypatch.setattr(
        CanonicalReducer,
        "prepare_model",
        lambda self, model: {"dependency_vector": raw_vector},
    )
    monkeypatch.setattr(
        reducer_module,
        "resolve_exact_gate_item_membership",
        lambda records, *, accepted_gate_item, accepted_gate_item_sha256: (
            {"gate_id": "gate-metadata-generation-7"},
            accepted_gate_item,
        ),
    )
    # AW-2 (m10949): authorize_publication consults the reducer-selected
    # transaction pointer FIRST, reading only the staged identity fields, so
    # the stub must carry them; with nothing staged the pointer check passes
    # and the gate/vector tripwire still fires before any custody use.
    staged_stub = SimpleNamespace(
        event={"stable_id": "m1359", "work_id": "work-m1359"},
        transaction_id="txn-" + "a" * 24,
    )
    with CanonicalReducer(paths.ledgers, context) as reducer:
        with pytest.raises(
            ReductionError,
            match="contradicts the model's dependency vector at checker_gate_identity",
        ) as excinfo:
            reducer.authorize_publication(
                {},
                staged_stub,
                {"stable_id": "m1359", "work_id": "work-m1359"},
                {},
                object(),
            )
    message = str(excinfo.value)
    assert "gate-fidelity-generation-3" in message
    assert "gate-metadata-generation-7" in message

    # KNOWN CARVE-OUT pin: a family variant's vector inherits the
    # representative's gate while its publication is accepted under its own
    # item's gate -- pre-existing behavior the tripwire must not break (the
    # read side consumes such pairs as loud quarantined cache misses). The
    # variant call proceeds past the tripwire and fails only on the stub
    # custody object, never on the gate coherence check.
    with CanonicalReducer(paths.ledgers, context) as reducer:
        with pytest.raises(Exception) as variant_excinfo:
            reducer.authorize_publication(
                {
                    "stable_id": "m_variant",
                    "identity": {"family_representative_id": "m_representative"},
                },
                SimpleNamespace(
                    event={"stable_id": "m_variant", "work_id": "work-m_variant"},
                    transaction_id="txn-" + "a" * 24,
                ),
                {"stable_id": "m_variant", "work_id": "work-m_variant"},
                {},
                object(),
            )
    assert "checker_gate_identity" not in str(variant_excinfo.value)


def test_three_era_canonical_authority_admits_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full admission over pre-rotation, mid-rotation, and current-era caches.

    Era 1 (pre-rotation): internally consistent artifact whose recorded
    checker basis names a RETIRED prompt world -- the production rotation gate
    (``_identity_basis_rotation``: recorded basis vs live world, the exact
    predicate that fires after any prompt re-pin) invalidates it, and the
    model authors fresh.
    Era 2 (mid-rotation): artifact written by a code era that recorded NO
    prompt pins, whose claimed vet its facts no longer derive -- live-basis
    validation refuses it, and the model authors fresh.
    Era 3 (current): same-world artifact -- reused, zero author sessions.
    The driver COMPLETES with all three models at runs.
    """

    snapshot = _snapshot(tmp_path, count=3)
    paths = _paths(tmp_path, snapshot)
    fresh_author = FakeAuthor()
    driver = _driver(tmp_path, snapshot, author=fresh_author)
    items = driver._ordered_work(snapshot, {})
    assert len(items) == 3
    pre_rotation_item, mid_rotation_item, current_item = items

    live_checker = driver_admission_module._checker_prompt_hash()
    # Era 1: coherent bytes, but the recorded checker pin names a retired
    # world -- exactly what every pre-rotation artifact looks like after a
    # checker-prompt re-pin.
    pre_rotation = _crafted_artifact(tmp_path, driver, snapshot, pre_rotation_item)
    pre_rotation = replace(
        pre_rotation,
        recorded_identity_basis=RecordedIdentityBasis(
            author_prompt_sha256=str(pre_rotation.proposal["author"]["prompt_sha256"]),
            checker_prompt_sha256="sha256:" + "1" * 64,
        ),
    )
    # Era 2: no recorded pins, and a vet identity its facts no longer derive.
    mid_rotation = _crafted_artifact(tmp_path, driver, snapshot, mid_rotation_item)
    assert mid_rotation.recorded_identity_basis is None
    mid_rotation.proposal["vet_identity"] = "sha256:" + "d" * 64
    # Era 3: authored under the current world, basis recorded live.
    current = _crafted_artifact(tmp_path, driver, snapshot, current_item)
    current = replace(
        current,
        recorded_identity_basis=RecordedIdentityBasis(
            author_prompt_sha256=str(current.proposal["author"]["prompt_sha256"]),
            checker_prompt_sha256=live_checker,
        ),
    )

    # Era-2's tampered claim never stages (a defective cache entry is
    # invalidated at validation, before custody matters); the reusable
    # current-era artifact stages like the real acquisition path guarantees.
    crafted: dict[str, tuple[Any, bool]] = {
        pre_rotation_item.stable_id: (pre_rotation, False),
        mid_rotation_item.stable_id: (mid_rotation, False),
        current_item.stable_id: (current, True),
    }
    served: set[str] = set()
    real_rehydrate = driver_admission_module.AdmissionEnvironmentMixin._rehydrate_final_authority

    def rehydrate(self: Any, item: Any, reducer: Any, operational: Any) -> Any:
        """Serve each era's canonical artifact once, then defer to reality.

        The fixture bypasses transaction acquisition, so it replays the
        production rotation gate verbatim (the same ``_identity_basis_rotation``
        predicate and stale disclosure `_rehydrate_final_authority` applies
        before returning an artifact); everything downstream is the real
        admission machinery.
        """

        if item.stable_id in crafted and item.stable_id not in served:
            served.add(item.stable_id)
            artifact, stage = crafted[item.stable_id]
            basis = artifact.recorded_identity_basis
            if basis is not None:
                rotated = driver_admission_module._identity_basis_rotation(basis)
                if rotated:
                    self._report_identity_stale_artifact_authority(
                        item,
                        None,
                        DriverIntegrationError(
                            "recorded identity basis rotated: " + "; ".join(rotated)
                        ),
                        operational,
                        transaction_id=None,
                        recorded_basis_available=True,
                    )
                    return None
            if stage:
                return self._stage_author_result(item, artifact, reducer)
            return artifact
        return real_rehydrate(self, item, reducer, operational)

    monkeypatch.setattr(
        driver_admission_module.AdmissionEnvironmentMixin,
        "_rehydrate_final_authority",
        rehydrate,
    )
    real_certain = driver_admission_module.AdmissionEnvironmentMixin._author_session_is_certain
    monkeypatch.setattr(
        driver_admission_module.AdmissionEnvironmentMixin,
        "_author_session_is_certain",
        lambda self, item, reducer: (
            False if item.stable_id in crafted else real_certain(self, item, reducer)
        ),
    )

    result = driver.run()

    assert result.status == "complete"
    models = {row["stable_id"]: row for row in scan_jsonl(paths.ledgers.models)}
    assert {row["status"]["code"] for row in models.values()} == {"runs"}
    assert set(models) == set(crafted)
    # Eras 1 and 2 authored fresh exactly once; era 3 reused with no session.
    assert fresh_author.calls == {
        pre_rotation_item.stable_id: 1,
        mid_rotation_item.stable_id: 1,
    }
    stale_events = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition")
        == ARTIFACT_AUTHORITY_IDENTITY_STALE_DISPOSITION
    ]
    by_model = {event["details"]["stable_id"]: event["details"] for event in stale_events}
    assert set(by_model) == {pre_rotation_item.stable_id, mid_rotation_item.stable_id}
    assert (
        "recorded identity basis rotated"
        in (by_model[pre_rotation_item.stable_id]["violation_summary"])
    )
    assert "vet_identity" in by_model[mid_rotation_item.stable_id]["violation_summary"]
