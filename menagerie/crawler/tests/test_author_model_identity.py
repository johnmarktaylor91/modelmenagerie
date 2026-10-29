"""AM1: one authority owns the author-model identity.

The C-AUTHOR-IDENTITY defect (2026-08-11): since 2026-08-05 the pilot
environment launched ``claude-fable-5-1`` (``MENAGERIE_AUTHOR_MODEL`` ->
``--model``) while every published record said ``claude-sonnet``, because
``provenance.author_model`` was the campaign configuration's CLAIM -- primed
into the stamped proposal through the envelope's identity disclosure -- and
nothing ever compared that claim to the model the executor actually launched.
Three authorities could disagree silently: the executor's spawn configuration,
the envelope/proposal claim, and the frozen tier-campaign map.

These tests pin the repaired contract:

1. the executor's publication receipt and durable attempt record attest the
   ACTUAL launched model, and an envelope claim that disagrees with it is a
   typed capture-defect event -- never a silent record;
2. record provenance takes the executor's attested model (actual wins) and
   flags the disagreement; artifacts without an attestation (history, cache
   rehydration) keep the stamped self-report unrewritten;
3. the frozen tier bindings agree everywhere they are spelled: the tier map,
   the partitioner specs, and the driver default all name ``claude-fable-5-1``
   for c1-mech (standing directive, 2026-08-11; ``claude-opus-5`` is the
   designated fallback tier);
4. the author prompt's identity line is model-neutral, so the prompt can never
   again prime a self-report that contradicts the launch configuration.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

# Importing the driver installs ``driver_models``' dependency table; without it
# the terminal assembler raises. Depend on it explicitly rather than on conftest
# import order, so this module is runnable on its own.
from menagerie.crawler import driver as _driver  # noqa: F401
from menagerie.crawler.author_executor import RECEIPT_VERSION, main
from menagerie.crawler.constants import TIER_CAMPAIGN_AUTHOR_MODELS
from menagerie.crawler.driver_contracts import DriverConfig, WorkItem
from menagerie.crawler.driver_models import (
    AUTHOR_MODEL_MISMATCH_FLAG,
    _assemble_terminal_model,
)
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.routing import EnvironmentPhase, IntentRoute
from menagerie.crawler.tests.conftest import (
    HASH,
    make_author_proposal,
    make_proposed_artifact,
)
from menagerie.crawler.tests.executor_test_support import (
    executor_environment,
    write_author_envelope,
    write_broker_fixtures,
    write_fake_claude,
)

_CREATED_AT = "2026-08-11T00:00:00Z"

#: The model the executor-test envelope fixture claims in ``identity_inputs``.
_ENVELOPE_CLAIM = "claude-sonnet"

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "claude_crawler_author_v2.txt"


# ---------------------------------------------------------------------------
# executor attestation: the receipt and attempt record carry the ACTUAL model
# ---------------------------------------------------------------------------


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One configured executor rig: fake harness, fixtures, log, author root."""

    fake = write_fake_claude(tmp_path / "bin")
    fixtures = write_broker_fixtures(tmp_path / "fixtures")
    log_dir = tmp_path / "log"
    executor_environment(monkeypatch, fake_claude=fake, fixtures=fixtures, log_dir=log_dir)
    return {
        "root": tmp_path / "work" / "m1" / "author",
        "monkeypatch": monkeypatch,
    }


def _run_author_round(rig) -> tuple[int, Path]:
    root = rig["root"]
    request = write_author_envelope(root, "m1")
    return main([str(request)]), root


def _published_receipt(capsys) -> dict[str, Any]:
    """Return the executor's publication receipt from captured stdout."""

    for line in reversed(capsys.readouterr().out.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict) and parsed.get("receipt_version") == RECEIPT_VERSION:
            return parsed
    raise AssertionError("executor printed no publication receipt")


def _attempt_dirs(root: Path) -> list[Path]:
    return sorted((root / "attempts").iterdir())


def _events(attempt_dir: Path) -> list[dict[str, Any]]:
    events_path = attempt_dir / "events.jsonl"
    if not events_path.is_file():
        return []
    return [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.smoke
def test_receipt_and_attempt_attest_the_actual_launched_model(rig, capsys) -> None:
    """The publication receipt and the durable attempt record are spawn-derived.

    This is the missing attestation that let 119 fable-authored records say
    sonnet: nothing machine-side ever recorded which model the executor
    actually passed to ``--model``.
    """

    rig["monkeypatch"].setenv("MENAGERIE_AUTHOR_MODEL", _ENVELOPE_CLAIM)
    code, root = _run_author_round(rig)
    assert code == 0
    receipt = _published_receipt(capsys)
    assert receipt["author_model"] == _ENVELOPE_CLAIM
    record = json.loads((_attempt_dirs(root)[-1] / "record.json").read_text())
    assert record["stage2"]["author_model"] == _ENVELOPE_CLAIM
    # Agreeing authorities leave no defect event behind.
    assert not [
        event
        for event in _events(_attempt_dirs(root)[-1])
        if event.get("event") == "author-model-binding-mismatch"
    ]


@pytest.mark.smoke
def test_envelope_claim_disagreeing_with_launch_is_a_typed_capture_defect(rig, capsys) -> None:
    """Launching one model under an envelope that claims another is recorded.

    Exactly the 2026-08-05 configuration: env says fable, campaign identity
    says sonnet. The session still completes -- the actual model wins at the
    record layer -- but the disagreement is a typed, durable event, never
    silence.
    """

    rig["monkeypatch"].setenv("MENAGERIE_AUTHOR_MODEL", "claude-fable-5-1")
    code, root = _run_author_round(rig)
    assert code == 0
    receipt = _published_receipt(capsys)
    assert receipt["author_model"] == "claude-fable-5-1"
    mismatches = [
        event
        for event in _events(_attempt_dirs(root)[-1])
        if event.get("event") == "author-model-binding-mismatch"
    ]
    assert len(mismatches) == 1
    assert mismatches[0]["launched_model"] == "claude-fable-5-1"
    assert mismatches[0]["claimed_model"] == _ENVELOPE_CLAIM


@pytest.mark.smoke
def test_an_unpinned_launch_cannot_silently_pose_as_an_attested_one(rig, capsys) -> None:
    """No ``--model`` means the harness default ran: unknown, said out loud.

    The receipt carries an explicit ``None`` -- never the envelope's claim --
    and the attempt records a typed ``author-model-unpinned`` event, so an
    unpinned production launch is visible instead of inheriting whatever the
    campaign configuration asserts.
    """

    rig["monkeypatch"].delenv("MENAGERIE_AUTHOR_MODEL", raising=False)
    code, root = _run_author_round(rig)
    assert code == 0
    receipt = _published_receipt(capsys)
    assert "author_model" in receipt and receipt["author_model"] is None
    assert [
        event
        for event in _events(_attempt_dirs(root)[-1])
        if event.get("event") == "author-model-unpinned"
    ]


# ---------------------------------------------------------------------------
# record provenance: the executor's attested model wins, disagreement is flagged
# ---------------------------------------------------------------------------


def _work_item(stable_id: str) -> WorkItem:
    intake = IntakeItem(
        stable_id=stable_id,
        name="IdentityNet",
        zoo="test-zoo",
        variant="base",
        discovery_source="crawl_roster",
        legacy_row_sha256="0" * 64,
        preserved_legacy_flags=(),
        variant_scope="standalone",
        family_representative_id=stable_id,
    )
    return WorkItem(
        intake=intake,
        route=IntentRoute(stable_id, "core", EnvironmentPhase.PYTORCH),
    )


def _proposed_artifact(tmp_path: Path, *, launched: str | None) -> Any:
    stable_id = "m_am1_provenance"
    proposal = make_author_proposal(stable_id)
    artifact = make_proposed_artifact(
        proposal, {"manifest_sha256": HASH, "sources": []}, tmp_path / stable_id
    )
    if launched is None:
        return artifact
    return dataclasses.replace(artifact, launched_author_model=launched)


def _assemble(tmp_path: Path, *, launched: str | None) -> dict[str, Any]:
    """Run the real terminal assembler; never a local re-derivation of it."""

    return _assemble_terminal_model(
        _work_item("m_am1_provenance"),
        _proposed_artifact(tmp_path, launched=launched),
        "failed:forward",
        "protocol-violation",
        None,
        [],
        [],
        DriverConfig(),
        _CREATED_AT,
        human_review=False,
        root_cause_fingerprint="fingerprint-am1",
    )


@pytest.mark.smoke
def test_provenance_records_the_executors_model_and_flags_the_self_report(
    tmp_path: Path,
) -> None:
    """Actual wins: the attested launch model is the provenance authority.

    The stamped proposal claims the fixture's ``claude``; the executor attested
    ``claude-fable-5-1``. The record says what RAN, and carries the typed
    capture-defect flag so the census can find every row whose claim disagreed.
    """

    model = _assemble(tmp_path, launched="claude-fable-5-1")
    assert model["provenance"]["author_model"] == "claude-fable-5-1"
    assert AUTHOR_MODEL_MISMATCH_FLAG in model["flags"]


@pytest.mark.smoke
def test_agreeing_authorities_produce_an_unflagged_record(tmp_path: Path) -> None:
    """When the attestation matches the claim there is nothing to disclose."""

    model = _assemble(tmp_path, launched="claude")
    assert model["provenance"]["author_model"] == "claude"
    assert AUTHOR_MODEL_MISMATCH_FLAG not in model["flags"]


@pytest.mark.smoke
def test_an_unattested_artifact_keeps_the_stamped_self_report(tmp_path: Path) -> None:
    """History stays readable and unrewritten: no attestation, no rewrite.

    Cache-rehydrated and pre-AM1 artifacts carry no launch attestation; their
    provenance remains the stamped proposal claim, unflagged -- the census
    interprets history, records never lie about holding evidence they lack.
    """

    model = _assemble(tmp_path, launched=None)
    assert model["provenance"]["author_model"] == "claude"
    assert AUTHOR_MODEL_MISMATCH_FLAG not in model["flags"]


def test_both_record_assemblers_share_the_one_provenance_resolver() -> None:
    """One authority has one spelling: both assemblers call the same resolver.

    A second literal ``proposal["author"]["model"]`` read inside an assembler
    would be a place the actual-wins rule could silently not apply.
    """

    import inspect

    from menagerie.crawler import driver_models

    for assembler in (
        driver_models._assemble_terminal_model,
        driver_models._assemble_run_model,
    ):
        source = inspect.getsource(assembler)
        assert "_provenance_author_model(" in source, assembler.__name__


# ---------------------------------------------------------------------------
# tier bindings: every spelling of the frozen c1-mech author model agrees
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_the_frozen_tier_bindings_name_fable_for_c1_mech() -> None:
    """Standing directive (JMT 2026-08-11): smartest Claudes for the crawl.

    c1-mech is bound to ``claude-fable-5-1``; ``claude-opus-5`` remains the
    c3-classics tier and the designated fallback if Fable throughput binds
    (MASTER_DOCKET Appendix 1, call 1). The closed-set semantics are untouched:
    exactly these four campaigns, each frozen to exactly one model.
    """

    from menagerie.crawler.partitioner import CAMPAIGN_SPECS

    assert TIER_CAMPAIGN_AUTHOR_MODELS == {
        "c1-mech": "claude-fable-5-1",
        "c2-disco": "claude-sonnet",
        "c3-classics": "claude-opus-5",
        "c4-native": "claude-sonnet",
    }
    assert TIER_CAMPAIGN_AUTHOR_MODELS == {
        spec.campaign_id: spec.author_model for spec in CAMPAIGN_SPECS
    }
    assert DriverConfig().author_model == "claude-fable-5-1"


@pytest.mark.smoke
def test_the_author_prompt_identity_line_is_model_neutral() -> None:
    """The prompt primes no model name, so it can never contradict the launch.

    Line 2 of the shipped prompt used to open "You are the Claude Sonnet/Opus
    author", which primed the self-report that all 119 fable-authored records
    inherited. The identity line names the frontier Claude author and no
    specific model; tier semantics elsewhere in the prompt (the C3/Opus
    promotion channel) name CAMPAIGNS, not this session's identity.
    """

    lines = _PROMPT_PATH.read_text(encoding="utf-8").splitlines()
    identity_line = lines[1]
    assert "frontier Claude author" in identity_line
    for model_name in ("Sonnet", "Opus", "Fable", "Haiku"):
        assert model_name not in identity_line


# ---------------------------------------------------------------------------
# three-authority agreement (the merge-gate test)
# ---------------------------------------------------------------------------


def test_three_authority_agreement_on_the_production_path(rig, capsys) -> None:
    """Spawn, claim, and tier map agree end to end for the mini-rung campaign.

    The executor launches the c1-mech tier model, the envelope claims the same
    model, and the stamped proposal's self-report equals both -- one authority,
    three consistent projections. This is the configuration every mini-rung leg
    must run under; any drift is caught by the mismatch machinery above.
    """

    import copy

    from menagerie.crawler.author_executor import _author_result_from_author_payload
    from menagerie.crawler.tests.test_author_executor import (
        _author_owned_proposal,
        _proposed_request,
    )

    tier_model = TIER_CAMPAIGN_AUTHOR_MODELS["c1-mech"]

    # Leg 1: the spawn attestation. Envelope claim and launch agree on the tier
    # model, so the round completes with a receipt naming it and no defect event.
    rig["monkeypatch"].setenv("MENAGERIE_AUTHOR_MODEL", tier_model)
    root = rig["root"]
    request_path = write_author_envelope(root, "m1")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["identity_inputs"]["author"]["model"] = tier_model
    request_path.write_text(json.dumps(request), encoding="utf-8")

    code = main([str(request_path)])
    assert code == 0
    receipt = _published_receipt(capsys)
    assert receipt["author_model"] == tier_model
    assert not [
        event
        for event in _events(_attempt_dirs(root)[-1])
        if event.get("event") == "author-model-binding-mismatch"
    ]

    # Leg 2: the proposal projection. The stamp writes the envelope's disclosed
    # binding into ``proposal.author``, so the self-report equals the tier model
    # by construction, whatever the session says about itself.
    stamped_request = copy.deepcopy(_proposed_request())
    stamped_request["identity_inputs"]["author"]["model"] = tier_model
    result = _author_result_from_author_payload(
        {"kind": "PROPOSED", "payload": {"proposal": _author_owned_proposal("m-fixture")}},
        stamped_request,
    )
    assert result["payload"]["proposal"]["author"]["model"] == tier_model
