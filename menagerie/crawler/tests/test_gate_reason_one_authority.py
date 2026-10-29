"""T5-9: gate-reason derivation has ONE authority (RED gate on the M1 merge).

Defect 4's recurrence mechanism is a NEW inline verdict-to-reason derivation --
a fifth call site quietly restating what the shared kernels in ``constants.py``
own (``metadata_gate_failure_reason``, ``fidelity_gate_failure_reason``,
``terminal_disposition_failure_reason``). The kernels are total and pinned
elsewhere (``test_shared_reason_kernels_are_total``,
``test_metadata_reason_kernel_mirrors_the_fidelity_rung_check_arms``); what was
missing is the repo's own anti-substitution idiom applied to the CONSUMER side:

* an AST audit proving no production module outside ``constants.py`` PRODUCES a
  kernel-owned gate reason literal (reads/comparisons stay legal); and
* a closed consumer registry for the kernel functions, so a new verdict consumer
  (W4's reuse lane, M3's scheduler) must register here consciously; and
* the m1359 driver-loop arm: ``lineage_repairs`` and a rejected terminal
  disposition CO-OCCURRING through ``CrawlerDriver.run`` -- the exact production
  history that stopped the rung-12 campaign -- terminalizes with the honest
  single-adjudication reason.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

import pytest

from menagerie.crawler.constants import (
    TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
    TERMINAL_DISPOSITION_UNVERIFIABLE_REASON_CODE,
)
from menagerie.crawler.recordio import scan_jsonl

#: Reasons only the shared verdict kernels may produce. ``effort-cap-exhausted``
#: and ``checker-contract-invalid`` are deliberately absent: they are lane/effort
#: routings, not verdict-derived gate reasons.
_KERNEL_OWNED_REASONS = frozenset(
    {
        "inaccurate-cap-exhausted",
        "cannot-verify-cap-exhausted",
        "major-drift-cap-exhausted",
        "slop-cap-exhausted",
        "identity-mismatch",
        TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
        TERMINAL_DISPOSITION_UNVERIFIABLE_REASON_CODE,
    }
)

#: The closed production consumer registry for the three kernel functions. A new
#: consumer is allowed -- calling the kernel is the CORRECT pattern -- but it must
#: be registered here in the same reviewed change, which is what makes a silent
#: fifth call site impossible.
_KERNEL_CONSUMER_REGISTRY = {
    "metadata_gate_failure_reason": frozenset({"authority.py", "driver.py"}),
    "fidelity_gate_failure_reason": frozenset({"authority.py", "driver_admission.py", "gates.py"}),
    "terminal_disposition_failure_reason": frozenset({"authority.py", "driver.py"}),
}

_CRAWLER_ROOT = Path(__file__).parents[1]


def _production_modules() -> Iterator[Path]:
    """Yield every production crawler module (tests and constants excluded)."""

    for path in sorted(_CRAWLER_ROOT.rglob("*.py")):
        relative = path.relative_to(_CRAWLER_ROOT)
        if relative.parts[0] in {"tests", "__pycache__"}:
            continue
        if relative.name == "constants.py":
            continue
        yield path


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    """Return a child-to-parent map for one parsed module."""

    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


@pytest.mark.smoke
def test_gate_failure_reason_derivation_has_one_authority() -> None:
    """No production module outside constants.py may PRODUCE a kernel reason.

    A kernel-owned reason literal may be read (compared against) anywhere, but a
    literal flowing into a return, assignment, container, or call is a reason
    being DERIVED -- the fifth-call-site defect this test exists to refuse.
    """

    violations: list[str] = []
    for path in _production_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = _parent_map(tree)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in _KERNEL_OWNED_REASONS
            ):
                continue
            cursor: ast.AST = node
            allowed = False
            while cursor in parents:
                parent = parents[cursor]
                if isinstance(parent, ast.Compare):
                    allowed = True
                    break
                if isinstance(parent, (ast.Set, ast.Tuple, ast.List)):
                    cursor = parent
                    continue
                if isinstance(parent, ast.Expr):
                    # A bare string statement (docstring) derives nothing.
                    allowed = True
                    break
                if isinstance(parent, ast.stmt):
                    break
                cursor = parent
            if not allowed:
                violations.append(
                    f"{path.relative_to(_CRAWLER_ROOT)}:{node.lineno} produces"
                    f" kernel-owned reason {node.value!r} outside constants.py"
                )
    assert not violations, "\n".join(violations)


@pytest.mark.smoke
def test_kernel_consumers_match_the_closed_registry() -> None:
    """Every production call site of a reason kernel is consciously registered."""

    observed: dict[str, set[str]] = {name: set() for name in _KERNEL_CONSUMER_REGISTRY}
    for path in _production_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if name in observed:
                observed[name].add(path.name)
    for name, expected in _KERNEL_CONSUMER_REGISTRY.items():
        assert observed[name] == set(expected), (
            f"{name} consumers changed: observed {sorted(observed[name])},"
            f" registered {sorted(expected)}. Register the new consumer in"
            " _KERNEL_CONSUMER_REGISTRY in the same reviewed change."
        )


def test_rejected_disposition_after_repair_lineage_terminalizes_through_the_driver(
    tmp_path: Path,
) -> None:
    """The m1359 driver-loop arm: repair lineage + rejected disposition co-occur.

    Production history (rung-12, m1359): one rejected ``metadata_batch`` round
    below the bounded cap, the repair author answers a terminal recommendation,
    and the checker REJECTS the claim. Before the m1359 unification and K1's
    recency rule this history could not terminalize at all (the campaign died on
    a partition hole); the kernel-level shapes are pinned in
    ``test_gate_semantics_unification``, but no driver-loop test ever combined
    ``lineage_repairs=True`` with ``terminal_verdict="rejected"``. This one does,
    through ``CrawlerDriver.run``.
    """

    from dataclasses import replace

    from menagerie.crawler.author_dispatch import SkipRecommendation
    from menagerie.crawler.fetcher import cas_path as source_cas_object_path
    from menagerie.crawler.identity import stable_hash
    from menagerie.crawler.terminal_evidence import SOURCE_CAS_DIRNAME
    from menagerie.crawler.tests.test_slice_f_driver import (
        AuthorScript,
        CheckerScript,
        ScriptedAuthor,
        ScriptedChecker,
        _driver,
        _paths,
        _snapshot,
    )

    class GroundedSkipAuthor(ScriptedAuthor):
        """Attach a grounded declared evidence record to the scripted skip.

        A REJECTED terminal verdict must resolve its cited evidence against the
        frozen source bytes; the plain scripted skip carries no declared
        records, so its rejection could never reach the accuracy-gate rule at
        all (it died in the gate contract instead).
        """

        def author(self, item, work_root, config, context):  # noqa: ANN001, ANN201
            artifact = super().author(item, work_root, config, context)
            result = artifact.author_result
            if not isinstance(result, SkipRecommendation):
                return artifact
            source_row = artifact.source_manifest["sources"][0]
            source_bytes = Path(str(source_row["cas_path"])).read_bytes()
            text = source_bytes.decode("utf-8")
            # The dereference root for a terminal pack is the model work root's
            # own CAS, not the author round-trip staging dir: mirror the frozen
            # object where `resolve_terminal_evidence` reads it.
            cas_object = source_cas_object_path(
                artifact.model_dir.parent / SOURCE_CAS_DIRNAME,
                str(source_row["content_sha256"]),
            )
            cas_object.parent.mkdir(parents=True, exist_ok=True)
            if not cas_object.exists():
                cas_object.write_bytes(source_bytes)
            record = {
                "evidence_id": "evidence-1",
                "source_id": str(source_row["source_id"]),
                "locator": f"bytes:0-{len(text.encode('utf-8'))}",
                "text": text,
                "supports": [result.status_code.split(":", 1)[1]],
            }
            raw = dict(result.binding.raw_result)
            payload = dict(raw["payload"])
            payload["evidence_records"] = [record]
            payload["recommendation_sha256"] = stable_hash(
                {k: v for k, v in payload.items() if k != "recommendation_sha256"}
            )
            raw["payload"] = payload
            raw["result_sha256"] = stable_hash(
                {k: v for k, v in raw.items() if k != "result_sha256"}
            )
            binding = replace(
                result.binding,
                result_sha256=str(raw["result_sha256"]),
                raw_result=raw,
            )
            recommendation = replace(
                result,
                binding=binding,
                evidence_records=(record,),
                recommendation_sha256=str(payload["recommendation_sha256"]),
            )
            return replace(artifact, author_result=recommendation)

    class AdjudicatedSourceChecker(ScriptedChecker):
        """Bind the rejected verdict to the sources its cited evidence resolves."""

        def check_terminal(self, artifact, work_root, config):  # noqa: ANN001, ANN201
            outcome = super().check_terminal(artifact, work_root, config)
            result = artifact.author_result
            if not isinstance(result, SkipRecommendation) or outcome.gate is None:
                return outcome
            gate = outcome.gate
            disposition = gate["items"][0]["terminal_disposition"]
            disposition["source_ids"] = sorted(
                {str(record["source_id"]) for record in result.evidence_records}
            )
            gate["result_envelope_sha256"] = stable_hash(
                {
                    key: value
                    for key, value in gate.items()
                    if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
                }
            )
            return outcome

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    author = GroundedSkipAuthor(
        AuthorScript(
            terminal_outcomes=(("status", "skipped:no-description"),),
            terminal_id=stable_id,
            terminal_call=2,
        )
    )
    checker = AdjudicatedSourceChecker(
        script=CheckerScript(
            metadata_verdict="inaccurate",
            lineage_repairs=True,
            terminal_verdict="rejected",
        )
    )

    result = _driver(tmp_path, snapshot, author=author, checker=checker).run()

    assert result.status == "terminal-partition-complete"
    models = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    assert len(models) == 1
    status = models[0]["status"]
    # W13: a rejected disposition is a review HOLD, never an accuracy-gate
    # failure -- the accuracy gate rendered no verdict on this member.
    assert status["code"] == "blocked:disposition-review"
    assert status["kind"] == "blocked"
    assert status["stage"] is None
    assert status["reason_code"] == TERMINAL_DISPOSITION_REJECTED_REASON_CODE
    # One rejected metadata round is below the bounded cap: no cap may be claimed.
    assert "cap-exhausted" not in status["reason_code"]
    assert status["human_review"]["required"] is True
    gates = scan_jsonl(_paths(tmp_path, snapshot).ledgers.gates)
    kinds = {gate["gate_kind"] for gate in gates}
    # The co-occurrence no prior driver-loop test drove: BOTH lineages persist.
    assert "metadata_batch" in kinds
    assert "terminal_disposition" in kinds
    rejected = [
        item
        for gate in gates
        if gate["gate_kind"] == "terminal_disposition"
        for item in gate["items"]
        if item.get("terminal_disposition", {}).get("verdict") == "rejected"
    ]
    assert rejected and rejected[0]["terminal_disposition"]["findings"]
