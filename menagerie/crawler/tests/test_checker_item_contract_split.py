"""BL-10 (c): item-attributed contract refusals never destroy the whole batch.

Audit B-3 measured the class on the pilot: 156 of 233 verdict-contract refusal
events (67%) were the write-contract replay refusing ONE fully-accurate item
over its own frozen AUTHOR artifact -- raised as a whole-envelope error, so the
wrapper spent a provably futile re-ask, exited 64, and the driver destroyed all
five members' verdicts through the halving ladder and mis-attributed the park
to the checker lane. The split, pinned end to end:

* the per-item validators raise a TYPED error naming the item and whose defect
  it is (author-side write contract vs checker-side disclosure);
* the wrapper skips the futile re-ask for the author-side class (keeps the one
  bounded re-ask for the checker-side class) and exits the item-scoped code 65
  with the stable IDs on its status sidecar;
* the driver reconstructs the typed error from the sidecar and scopes the
  failure: innocent siblings re-batch without a downshift tax, the named model
  earns one bounded singleton round, and a repeat parks it TRUTHFULLY as
  ``author-evidence-defect`` -- the census looks in the right lane.
"""

from __future__ import annotations

import json
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

import pytest

from menagerie.crawler.constants import METADATA_BATCH_MAX
from menagerie.crawler.checker_dispatch import (
    CheckerItemContractError,
    validate_checker_result_mapping,
)
from menagerie.crawler.driver_admission import _raise_for_checker_exit
from menagerie.crawler.driver_contracts import CheckerOutcome, DriverConfig
from menagerie.crawler.operator_checker import CodexAttempt, execute_checker_request
from menagerie.crawler.operator_protocol import OperatorExitCode, status_sidecar_path
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_checker_lane_failure_scoping import _valve_rows
from menagerie.crawler.tests.test_operator_checker import (
    _events,
    _native_final_message,
    _request_and_result,
    _status,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    ScriptedChecker,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Validator seam: the per-item validators name the item and the party.
# ---------------------------------------------------------------------------


def test_write_contract_bookkeeping_refusal_is_typed_checker_side(tmp_path: Path) -> None:
    """The rung-9 shape (empty checked_source_ids on an accurate item).

    The defect lives in the gate item's own citation bookkeeping, not in the
    frozen author artifact -- rung 9's own checker emitted complete
    checked_source_ids on its repair rounds -- so the attribution names the
    checker and the wrapper keeps its bounded fed-back re-ask.
    """

    request_path, result = _request_and_result(tmp_path)
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    poisoned = deepcopy(result)
    for check in poisoned["items"][0]["field_checks"]:
        check["checked_source_ids"] = []
    from menagerie.crawler.tests.test_slice_d_checker_gates import _stamped

    with pytest.raises(CheckerItemContractError) as caught:
        validate_checker_result_mapping(_stamped(poisoned, envelope), envelope)
    assert caught.value.stable_ids == (str(result["items"][0]["stable_id"]),)
    assert caught.value.author_side is False


def _without_authored_support(envelope: dict[str, Any], field: str) -> dict[str, Any]:
    """Strip one claim from every authored excerpt's supports and re-seal.

    Parameters
    ----------
    envelope:
        Parsed checker request envelope.
    field:
        Gated claim path to orphan (no authored excerpt supports it anymore).

    Returns
    -------
    dict[str, Any]
        Deep-copied envelope with a recomputed ``envelope_sha256``.
    """

    from menagerie.crawler.identity import stable_hash

    mutated = deepcopy(envelope)
    for item in mutated["items"]:
        facts = item["proposal"]["proposed_facts"]
        for excerpt in facts["evidence"]["excerpts"]:
            excerpt["supports"] = [
                claim
                for claim in excerpt["supports"]
                if claim not in (field, f"proposed_facts.{field}")
            ]
    mutated["envelope_sha256"] = stable_hash(
        {key: value for key, value in mutated.items() if key != "envelope_sha256"}
    )
    return mutated


def test_write_contract_unsupportable_claim_is_typed_author_side(tmp_path: Path) -> None:
    """A claim NO authored excerpt supports is the author artifact's defect.

    No compliant re-answer over the frozen proposal can pass, so the re-ask is
    provably futile and the attribution stays author-side.
    """

    request_path, result = _request_and_result(tmp_path)
    envelope = _without_authored_support(
        json.loads(request_path.read_text(encoding="utf-8")), "website"
    )
    from menagerie.crawler.tests.test_slice_d_checker_gates import _stamped

    with pytest.raises(CheckerItemContractError) as caught:
        validate_checker_result_mapping(_stamped(deepcopy(result), envelope), envelope)
    assert caught.value.stable_ids == (str(result["items"][0]["stable_id"]),)
    assert caught.value.author_side is True
    assert "does not support exact field website" in str(caught.value)


def test_write_contract_wrong_excerpt_citation_is_typed_checker_side(tmp_path: Path) -> None:
    """Citing the wrong excerpt while a right one exists names the checker.

    The pilot's dominant refusal family (116 events, 8 identical recurrences):
    the frozen proposal carries an excerpt that supports the claim, the gate
    item cited a different one, and the fed-back re-ask repairs it.
    """

    from menagerie.crawler.identity import stable_hash

    request_path, result = _request_and_result(tmp_path)
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    mutated = deepcopy(envelope)
    facts = mutated["items"][0]["proposal"]["proposed_facts"]
    original = facts["evidence"]["excerpts"][0]
    original["supports"] = [
        claim for claim in original["supports"] if claim not in ("website", "proposed_facts.website")
    ]
    facts["evidence"]["excerpts"].append(
        dict(original, evidence_id="evidence-website", supports=["website"])
    )
    mutated["envelope_sha256"] = stable_hash(
        {key: value for key, value in mutated.items() if key != "envelope_sha256"}
    )
    from menagerie.crawler.tests.test_slice_d_checker_gates import _stamped

    # The result's website check still cites the original excerpt, which no
    # longer supports the claim -- but ``evidence-website`` does.
    with pytest.raises(CheckerItemContractError) as caught:
        validate_checker_result_mapping(_stamped(deepcopy(result), mutated), mutated)
    assert caught.value.author_side is False
    assert "does not support exact field website" in str(caught.value)


def test_disclosure_refusal_is_typed_checker_side(tmp_path: Path) -> None:
    """An adverse item with an incomplete complaint set names the checker."""

    request_path, result = _request_and_result(tmp_path)
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    partial = deepcopy(result)
    item = partial["items"][0]
    item["verdict"] = "inaccurate"
    item["field_checks"][0]["verdict"] = "inaccurate"
    item["field_checks"][0]["required_repair"] = "fix it"
    del item["field_checks"][1:]  # incomplete complaint disclosure
    from menagerie.crawler.tests.test_slice_d_checker_gates import _stamped

    with pytest.raises(CheckerItemContractError) as caught:
        validate_checker_result_mapping(_stamped(partial, envelope), envelope)
    assert caught.value.stable_ids == (str(result["items"][0]["stable_id"]),)
    assert caught.value.author_side is False


# ---------------------------------------------------------------------------
# Wrapper seam: exit 65 with item attribution; no futile re-ask author-side.
# ---------------------------------------------------------------------------


def test_author_side_defect_exits_item_scoped_without_a_reask(tmp_path: Path) -> None:
    from menagerie.crawler.identity import canonical_json_bytes

    request_path, result = _request_and_result(tmp_path)
    envelope = _without_authored_support(
        json.loads(request_path.read_text(encoding="utf-8")), "website"
    )
    request_path.write_bytes(canonical_json_bytes(envelope) + b"\n")
    attempts: list[str] = []

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        del timeout
        attempts.append(str(argv[-1]))
        last_message.write_bytes(_native_final_message(result))
        return CodexAttempt(
            0,
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.ITEM_CONTRACT_REJECTION
    assert len(attempts) == 1, "an author-side defect is deterministic; no re-ask is spent"
    status = _status(request_path)
    assert status["exit_code"] == int(OperatorExitCode.ITEM_CONTRACT_REJECTION)
    assert status["item_contract"] == {
        "stable_ids": [str(result["items"][0]["stable_id"])],
        "author_side": True,
    }
    refusals = [e for e in _events(request_path) if e["event"] == "item-contract-refused"]
    assert [e["retrying"] for e in refusals] == [False]


def test_distinct_checker_side_slips_each_earn_a_fed_back_reask(tmp_path: Path) -> None:
    """Two DIFFERENT bookkeeping slips both repair within the attempt cap.

    The pilot ledger's shape: the fed-back error is repaired, a fresh slip
    appears elsewhere, and the old shared single re-ask terminalized the
    convergent batch on the second, unrelated defect (102 of 107 consecutive
    refusal pairs named different defects). Each DISTINCT defect earns its own
    fed-back re-ask; the attempt cap still bounds the ladder.
    """

    request_path, result = _request_and_result(tmp_path)
    first_slip = deepcopy(result)
    first_slip["items"][0]["field_checks"][0]["checked_source_ids"] = []
    second_slip = deepcopy(result)
    second_slip["items"][0]["field_checks"][1]["checked_source_ids"] = []
    ladder = [first_slip, second_slip, result]
    prompts: list[str] = []

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        del timeout
        prompts.append(str(argv[-1]))
        last_message.write_bytes(_native_final_message(ladder[len(prompts) - 1]))
        return CodexAttempt(
            0,
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.SUCCESS
    assert len(prompts) == 3
    fields = [check["field"] for check in result["items"][0]["field_checks"][:2]]
    assert fields[0] in prompts[1] and fields[1] in prompts[2]
    refusals = [e for e in _events(request_path) if e["event"] == "item-contract-refused"]
    assert [e["retrying"] for e in refusals] == [True, True]
    assert all(e["author_side"] is False for e in refusals)


def test_repeated_identical_checker_side_slip_stays_terminal(tmp_path: Path) -> None:
    """The SAME named error twice is systematic divergence, exactly as before."""

    request_path, result = _request_and_result(tmp_path)
    slip = deepcopy(result)
    slip["items"][0]["field_checks"][0]["checked_source_ids"] = []
    calls = 0

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        del argv, timeout
        nonlocal calls
        calls += 1
        last_message.write_bytes(_native_final_message(slip))
        return CodexAttempt(
            0,
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.ITEM_CONTRACT_REJECTION
    assert calls == 2
    refusals = [e for e in _events(request_path) if e["event"] == "item-contract-refused"]
    assert [e["retrying"] for e in refusals] == [True, False]


def test_checker_side_item_defect_keeps_its_one_bounded_reask(tmp_path: Path) -> None:
    """A disclosure slip is checker-repairable: one re-ask, then exit 65."""

    request_path, result = _request_and_result(tmp_path)
    partial = deepcopy(result)
    item = partial["items"][0]
    item["verdict"] = "inaccurate"
    item["field_checks"][0]["verdict"] = "inaccurate"
    item["field_checks"][0]["required_repair"] = "fix it"
    del item["field_checks"][1:]
    attempts: list[str] = []

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        del timeout
        attempts.append(str(argv[-1]))
        last_message.write_bytes(_native_final_message(partial))
        return CodexAttempt(
            0,
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.ITEM_CONTRACT_REJECTION
    assert len(attempts) == 2, "the checker-side class keeps its one bounded re-ask"
    assert "complaint set is incomplete" in attempts[1]
    status = _status(request_path)
    assert status["item_contract"]["author_side"] is False


# ---------------------------------------------------------------------------
# Driver reconstruction seam: exit-65 sidecars become the typed error.
# ---------------------------------------------------------------------------


def test_exit_65_reconstructs_the_typed_item_error(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    status_sidecar_path(request_path).write_text(
        json.dumps(
            {
                "exit_code": 65,
                "classification": "permanent",
                "detail": "refused",
                "item_contract": {"stable_ids": ["m_named"], "author_side": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CheckerItemContractError) as caught:
        _raise_for_checker_exit(65, "", "", request_path=request_path)
    assert caught.value.stable_ids == ("m_named",)
    assert caught.value.author_side is True


def test_exit_65_without_attribution_falls_back_to_whole_batch(tmp_path: Path) -> None:
    """A torn sidecar fails closed into the generic contract route, never a pass."""

    from menagerie.crawler.driver_contracts import DriverIntegrationError

    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    with pytest.raises(DriverIntegrationError):
        _raise_for_checker_exit(65, "", "", request_path=request_path)


# ---------------------------------------------------------------------------
# Driver routing seam: siblings survive, the named model parks truthfully.
# ---------------------------------------------------------------------------


class NamedItemDefectChecker(ScriptedChecker):
    """Raise the typed author-side item error whenever the named model rides."""

    def __init__(self, defective_id: str) -> None:
        super().__init__()
        self.defective_id = defective_id
        self.batch_ids: list[tuple[str, ...]] = []

    def check_metadata(
        self, artifacts: Sequence[Any], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        ids = tuple(str(a.proposal["stable_id"]) for a in artifacts)
        self.batch_ids.append(ids)
        if self.defective_id in ids:
            raise CheckerItemContractError(
                f"fully accurate metadata item {self.defective_id} would refuse at "
                "canonical write: authored evidence does not support the claim",
                stable_ids=(self.defective_id,),
                author_side=True,
            )
        return super().check_metadata(artifacts, work_root, config)


def test_named_item_parks_truthfully_and_siblings_survive(tmp_path: Path) -> None:
    """One author-defective item never destroys its innocent batch siblings.

    Pre-fix, this scenario downshifted the whole batch through the halving
    ladder and parked the named model as a CHECKER failure. Now: the siblings
    re-batch together and gate normally; the named model rides one bounded
    singleton round and parks as ``author-evidence-defect``.
    """

    snapshot = _snapshot(tmp_path, count=3)
    defective = snapshot.items[0].stable_id
    checker = NamedItemDefectChecker(defective)
    result = _driver(tmp_path, snapshot, checker=checker).run()

    assert result.status == "complete-with-deferrals"
    # Multi-item width: batch of 3 (typed refusal) -> siblings re-batch
    # (success) -> named singleton (typed refusal again) -> park. No halving
    # tree. At width 1 every model is its own batch from the start, so the
    # split degenerates to the defective singleton parking directly.
    assert any(defective in ids for ids in checker.batch_ids)
    sibling_batches = [ids for ids in checker.batch_ids if defective not in ids]
    assert sibling_batches, "innocent siblings must re-batch together"
    assert all(len(ids) == min(2, METADATA_BATCH_MAX) for ids in sibling_batches)
    rows = _valve_rows(tmp_path, snapshot)
    assert [row["stable_id"] for row in rows] == [defective]
    assert rows[0]["checker_lane"]["failure_class"] == "author-evidence-defect"
    paths = _paths(tmp_path, snapshot)
    models = {r["stable_id"]: r for r in scan_jsonl(paths.ledgers.models)}
    siblings = {item.stable_id for item in snapshot.items} - {defective}
    for stable_id in siblings:
        assert models[stable_id]["status"]["code"] == "runs"
    assert defective not in models, "the named model parks; it never terminalizes"
    # The scoping is disclosed on the operational ledger.
    scoped = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition") == "checker-item-contract-scoped"
    ]
    assert scoped and scoped[0]["details"]["stable_ids"] == [defective]
    assert scoped[0]["details"]["author_side"] is True
