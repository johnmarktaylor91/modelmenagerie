"""Production-path checker rig: script the DECISIONS, never the envelope.

``ScriptedChecker`` (test_slice_f_driver) hand-builds gate rows via ``make_gate``
and seals its own self-hash, so the production envelope construction and result
validation path (``build_*_envelope`` -> ``apply_machine_owned_gate_fields`` ->
``validate_checker_result``) never runs inside driver E2Es. That is a closed
loop: the rig, the builder, and the validator share no enforcement, which is
exactly where the 2026-08-04 pilot binding failure hid (fixed by changing the
mock, not the structure).

:class:`ContractCheckerLane` scripts only what a real checker owns -- the
verdicts, findings, and per-claim reasoning -- and delegates EVERYTHING the
machine owns to the production code: envelope construction (via the same
``_checker_item``/``_terminal_checker_item`` packs and ``build_*_envelope``
builders the ``CommandCheckerLane`` dispatch uses), machine-owned scaffold
stamping (``apply_machine_owned_gate_fields``), and full result validation
(``validate_checker_result_mapping``). A driver E2E through this lane therefore
inherits every production enforcement rule; content the production path refuses
raises here instead of silently publishing.

The module is also imported by ``scripted_checker_subprocess.py``, the argv
checker used by the true subprocess E2E through ``CommandCheckerLane``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.checker_dispatch import (
    TERMINAL_VERDICT_LOCKSTEP,
    apply_machine_owned_gate_fields,
    build_fidelity_envelope,
    build_metadata_vet_envelope,
    build_terminal_disposition_envelope,
    validate_checker_result_mapping,
)
from menagerie.crawler.constants import METADATA_BATCH_MIN, GateKind
from menagerie.crawler.driver_contracts import AuthorArtifact, CheckerOutcome, DriverConfig
from menagerie.crawler.driver_models import _checker_item, _terminal_checker_item
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.models import JsonObject


@dataclass(frozen=True)
class CheckerContractScript:
    """Scripted checker DECISIONS; everything else is machine-derived.

    ``metadata_verdicts``/``fidelity_verdicts`` map stable IDs to non-default
    verdicts (metadata: ``inaccurate``/``cannot-verify``; fidelity: any
    non-``match`` fidelity verdict). ``terminal_verdicts`` maps stable IDs to a
    non-``accepted`` disposition. Unlisted models are accepted.
    """

    metadata_verdicts: Mapping[str, str] = field(default_factory=dict)
    fidelity_verdicts: Mapping[str, str] = field(default_factory=dict)
    terminal_verdicts: Mapping[str, str] = field(default_factory=dict)
    required_repair: str = "correct the unsupported metadata claim"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _copied_bindings(envelope_item: Mapping[str, Any]) -> JsonObject:
    """Copy the identity bindings the checker contract says to echo verbatim."""

    return {
        "work_id": envelope_item["work_id"],
        "campaign_root_work_id": envelope_item["campaign_root_work_id"],
        "stable_id": envelope_item["stable_id"],
        "family_representative_id": envelope_item["family_representative_id"],
        "fidelity_identity": envelope_item.get("fidelity_identity"),
        "vet_identity": envelope_item["vet_identity"],
        "verified_hashes": dict(envelope_item["verified_hashes"]),
    }


def _accurate_integrity() -> JsonObject:
    return {
        "verdict": "accurate",
        "hash_mismatches": [],
        "excerpt_discrepancies": [],
        "locator_failures": [],
    }


def _claim_references(facts: Mapping[str, Any], claim: str) -> tuple[list[str], list[str]]:
    """Resolve one gated claim to its own proposal's evidence and sources.

    Mirrors what the frozen prompt asks of a real checker: cite only excerpts
    the proposal itself declares as supporting the exact claim, and name the
    sources those excerpts came from. Claims without a supporting excerpt (the
    keyword/absence classes) cite no evidence but still name checked sources.
    """

    evidence_ids: list[str] = []
    source_ids: list[str] = []
    for excerpt in facts["evidence"]["excerpts"]:
        supports = excerpt.get("supports") or []
        if claim in supports or f"proposed_facts.{claim}" in supports:
            evidence_ids.append(str(excerpt["evidence_id"]))
            source_ids.append(str(excerpt["source_id"]))
    if not source_ids:
        source_ids = [str(source["source_id"]) for source in facts["source_resolution"]["sources"]]
    return evidence_ids, sorted(set(source_ids))


def author_metadata_item(
    envelope_item: Mapping[str, Any],
    *,
    verdict: str = "accurate",
    required_repair: str = "correct the unsupported metadata claim",
) -> JsonObject:
    """Author one metadata gate item the way the checker contract demands.

    Every machine-inventoried claim gets exactly one field check (the
    complaint-disclosure rule); an adverse scripted verdict lands on the first
    claim with the rest explicitly accurate, which is the complete-disclosure
    shape a real adverse round must publish.
    """

    facts = envelope_item["proposal"]["proposed_facts"]
    checks: list[JsonObject] = []
    for claim in envelope_item["required_field_checks"]:
        evidence_ids, checked_source_ids = _claim_references(facts, claim)
        checks.append(
            {
                "field": claim,
                "verdict": "accurate",
                "evidence_ids": evidence_ids,
                "checked_source_ids": checked_source_ids,
                "reason": "claim re-derived from the frozen excerpts",
                "required_repair": None,
            }
        )
    item: JsonObject = {
        **_copied_bindings(envelope_item),
        "integrity": _accurate_integrity(),
        "verdict": "accurate",
        "field_checks": checks,
        "rung_check": {
            "selected_rung": facts["source_resolution"]["rung"],
            "highest_applicable": facts["source_resolution"]["rung"],
            "verdict": "accurate",
            "findings": [],
        },
        "unsupported_claims": [],
        "required_repairs": [],
        "confidence": "high",
        "terminal_disposition": None,
    }
    if verdict != "accurate":
        item["verdict"] = verdict
        item["required_repairs"] = [required_repair]
        checks[0]["verdict"] = verdict
        checks[0]["reason"] = required_repair
        checks[0]["required_repair"] = required_repair
    return item


def author_fidelity_item(
    envelope_item: Mapping[str, Any], *, fidelity_verdict: str = "match"
) -> JsonObject:
    """Author one fidelity gate item; non-match verdicts follow the lockstep."""

    item = author_metadata_item(envelope_item)
    item["fidelity"] = {
        "required": True,
        "verdict": fidelity_verdict,
        "material_checks": [],
        "unsupported_choices": [],
        "contradictions": [],
        "omissions": [],
        "permanent_scar": False,
    }
    if fidelity_verdict in {"major-drift", "slop"}:
        item["verdict"] = "inaccurate"
        item["required_repairs"] = ["reconcile the implementation with its frozen sources"]
    elif fidelity_verdict == "cannot-verify":
        item["verdict"] = "cannot-verify"
        item["required_repairs"] = ["supply the missing fidelity evidence"]
    return item


def author_terminal_item(
    envelope_item: Mapping[str, Any],
    *,
    disposition: str = "accepted",
) -> JsonObject:
    """Author one terminal-disposition item from its envelope's machine facts.

    The reference identities (author result binding, evidence, license, source
    manifest) are copied from the machine-derived envelope pack -- exactly what
    the frozen prompt instructs a real checker to do -- and only the verdict,
    findings, and repairs are authored.
    """

    author_result = envelope_item["author_result"]
    evidence_pack = envelope_item["evidence_pack"]
    preimage = envelope_item["recommendation_preimage"]
    accepted = disposition == "accepted"
    findings = (
        ["every declared excerpt re-derives from frozen bytes and supports the predicate"]
        if accepted
        else ["the resolved excerpts do not establish the recommended predicate"]
    )
    terminal: JsonObject = {
        "verdict": disposition,
        "findings": findings,
        "author_result_id": author_result["result_id"],
        "author_result_sha256": author_result["result_sha256"],
        "source_manifest_identity": author_result["source_manifest_identity"],
        "evidence_identity": evidence_pack["evidence_identity"],
        "license_identity": envelope_item["license_identity"],
        "kind": preimage["arm"],
        "predicate": evidence_pack["predicate"],
        "handoff_proposal_id": preimage.get("handoff_proposal_id"),
        "handoff_sha256": preimage.get("handoff_sha256"),
        "source_ids": list(evidence_pack["checked_source_ids"]),
        "evidence_ids": list(preimage["evidence_ids"]),
    }
    return {
        **_copied_bindings(envelope_item),
        "integrity": _accurate_integrity(),
        "verdict": TERMINAL_VERDICT_LOCKSTEP[disposition].value,
        "field_checks": [],
        "fidelity": {
            "required": False,
            "verdict": "not-applicable",
            "material_checks": [],
            "unsupported_choices": [],
            "contradictions": [],
            "omissions": [],
            "permanent_scar": False,
        },
        "rung_check": {
            "selected_rung": "R5_SKIP",
            "highest_applicable": "R5_SKIP",
            "verdict": "accurate",
            "findings": ["not-applicable: terminal advisory arm; no implementation rung selected"],
        },
        "unsupported_claims": []
        if accepted
        else ["the terminal predicate is not established by the cited excerpts"],
        "required_repairs": [] if accepted else ["withdraw or re-ground the recommendation"],
        "confidence": "high",
        "terminal_disposition": terminal,
    }


def author_result_for_envelope(
    envelope: Mapping[str, Any], script: CheckerContractScript
) -> JsonObject:
    """Author the complete checker-owned result body for one request envelope.

    Machine-owned scaffold fields are deliberately ABSENT: production stamping
    (``apply_machine_owned_gate_fields``) owns them, exactly as in the live
    operator wrapper.
    """

    kind = str(envelope["gate_kind"])
    items: list[JsonObject] = []
    for envelope_item in envelope["items"]:
        stable_id = str(envelope_item["stable_id"])
        if kind == GateKind.METADATA_BATCH.value:
            items.append(
                author_metadata_item(
                    envelope_item,
                    verdict=script.metadata_verdicts.get(stable_id, "accurate"),
                    required_repair=script.required_repair,
                )
            )
        elif kind == GateKind.FIDELITY.value:
            items.append(
                author_fidelity_item(
                    envelope_item,
                    fidelity_verdict=script.fidelity_verdicts.get(stable_id, "match"),
                )
            )
        else:
            items.append(
                author_terminal_item(
                    envelope_item,
                    disposition=script.terminal_verdicts.get(stable_id, "accepted"),
                )
            )
    return {"items": items}


def produce_validated_gate(
    envelope: Mapping[str, Any],
    script: CheckerContractScript,
    *,
    raw_override: Optional[Mapping[str, Any]] = None,
) -> JsonObject:
    """Stamp and validate one authored result through the PRODUCTION path.

    This is the same publication sequence the live operator wrapper and
    ``CommandCheckerLane`` enforce: machine-owned stamping first, then the full
    ``validate_checker_result`` core. Anything the production path refuses
    raises out of here -- the whole point of the rig.
    """

    raw = (
        dict(raw_override)
        if raw_override is not None
        else author_result_for_envelope(envelope, script)
    )
    started_at = _utc_now()
    stamped = apply_machine_owned_gate_fields(
        raw, envelope, started_at=started_at, finished_at=_utc_now()
    )
    return validate_checker_result_mapping(stamped, envelope)


class ContractCheckerLane:
    """Checker lane whose only scripted authority is the checker's decisions."""

    def __init__(self, script: Optional[CheckerContractScript] = None) -> None:
        self.script = script or CheckerContractScript()
        self.metadata_calls = 0
        self.fidelity_calls = 0
        self.terminal_calls = 0
        # The checker pool dispatches this fake from worker threads; the call
        # counters tests assert on must be guarded to stay exact.
        self._call_count_lock = threading.Lock()

    def adjudication_identity(self, gate_kind: str, config: DriverConfig) -> Any:
        """Declare the exact identity this lane stamps (enables reuse on W4+)."""

        del gate_kind
        from menagerie.crawler.driver import _checker_prompt_hash
        from menagerie.crawler.gate_reuse import CheckerAdjudicationIdentity

        return CheckerAdjudicationIdentity(
            provider="openai",
            model=config.checker_model,
            version=config.checker_version,
            prompt_sha256=_checker_prompt_hash(),
        )

    def _author(self, envelope: Mapping[str, Any]) -> JsonObject:
        """Author the checker-owned body; override to script a deviant checker."""

        return author_result_for_envelope(envelope, self.script)

    def _outcome(self, envelope: Mapping[str, Any]) -> CheckerOutcome:
        gate = produce_validated_gate(envelope, self.script, raw_override=self._author(envelope))
        return CheckerOutcome(gate=gate)

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Build the production metadata envelope and answer it in-process."""

        with self._call_count_lock:
            self.metadata_calls += 1
        items = [_checker_item(artifact) for artifact in artifacts]
        batch_id = stable_hash([item["work_id"] for item in items])[7:23]
        return self._outcome(
            build_metadata_vet_envelope(
                items,
                gate_round=1,
                output_path=work_root / "checker" / f"metadata-{batch_id}" / "result.json",
                checker_model=config.checker_model,
                checker_version=config.checker_version,
                request_nonce=batch_id,
                final_tail=len(items) < METADATA_BATCH_MIN,
            )
        )

    def check_fidelity(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Build the production fidelity envelope and answer it in-process."""

        with self._call_count_lock:
            self.fidelity_calls += 1
        stable_id = str(artifact.proposal["stable_id"])
        return self._outcome(
            build_fidelity_envelope(
                _checker_item(artifact),
                gate_round=1,
                output_path=work_root / stable_id / "checker-fidelity" / "result.json",
                checker_model=config.checker_model,
                checker_version=config.checker_version,
                request_nonce=f"fidelity-{stable_id}",
            )
        )

    def check_terminal(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Build the production terminal envelope and answer it in-process."""

        with self._call_count_lock:
            self.terminal_calls += 1
        item = _terminal_checker_item(artifact)
        stable_id = str(item["stable_id"])
        return self._outcome(
            build_terminal_disposition_envelope(
                item,
                gate_round=1,
                output_path=work_root / stable_id / "checker-terminal" / "result.json",
                checker_model=config.checker_model,
                checker_version=config.checker_version,
                request_nonce=f"terminal-{stable_id}",
            )
        )
