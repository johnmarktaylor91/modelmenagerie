"""Deterministic recipe refusals route to bounded author repair, never a burn.

Rung-11 census defect class 3: eleven runnable timm models terminalized
``failed:import / import-exception`` on a deterministic ``RecipeError`` the
author could have repaired in one bounded generation. These tests drive the
new route end to end on the deterministic driver harness, prove the honest
import terminal survives cap exhaustion, and prove the loop is bounded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from menagerie.crawler.driver_contracts import (
    AuthorArtifact,
    EnvironmentBinding,
    WorkItem,
)
from menagerie.crawler.driver_receipts import (
    _attempt_error_fields,
    _deterministic_recipe_refusal,
    _exception_chain_message,
)
from menagerie.crawler.recipe import RecipeError
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.tests.test_slice_f_driver import (
    HASH,
    DriverConfig,
    FakeAuthor,
    FakeChecker,
    ScriptedForward,
    _driver,
    _paths,
    _rebind_fake_author_result,
    _refresh_proposal_identities,
    _snapshot,
)

#: The exact rung-11 class-3 refusal text (tb 9c4663a9..., 11 models).
_REFUSAL_TEXT = (
    "pretrained disable fields are not explicit constructor parameters: ['pretrained_cfg']"
)

_REFUSAL_MARKER = "refuse_me"


class RecipeRefusalRepairAuthor(FakeAuthor):
    """Propose a loader-refused recipe first and the repaired revision after.

    Generation 1 plants the refusal marker the scripted forward keys on;
    every later generation (the bounded repair) removes it, modelling an
    author that read the verbatim refusal and fixed the recipe.
    """

    def author(
        self, item: WorkItem, work_root: Path, config: DriverConfig, context: Any
    ) -> AuthorArtifact:
        """Return the marker proposal first and the repaired revision on repair."""

        artifact = super().author(item, work_root, config, context)
        generation = self.calls[item.stable_id]
        recipe = artifact.proposal["proposed_facts"]["implementation"]["library_recipe"]
        if generation == 1:
            recipe["kwargs"] = dict(recipe["kwargs"], **{_REFUSAL_MARKER: True})
        _refresh_proposal_identities(
            artifact.proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        return _rebind_fake_author_result(artifact)


class RecipeRefusalCapAuthor(FakeAuthor):
    """Answer every bounded repair generation with the still-refused recipe."""

    def author(
        self, item: WorkItem, work_root: Path, config: DriverConfig, context: Any
    ) -> AuthorArtifact:
        """Return a completed but still loader-refused revision every time."""

        artifact = super().author(item, work_root, config, context)
        recipe = artifact.proposal["proposed_facts"]["implementation"]["library_recipe"]
        recipe["kwargs"] = dict(recipe["kwargs"], **{_REFUSAL_MARKER: True})
        _refresh_proposal_identities(
            artifact.proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        return _rebind_fake_author_result(artifact)


class RecipeRefusingForward(ScriptedForward):
    """Fail marker proposals with the worker's exact pre-import refusal shape."""

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Return one import-refused attempt for marker proposals, clean otherwise."""

        attempts = [
            dict(attempt)
            for attempt in super().forward(artifact, environment, cold_runs, work_root)
        ]
        recipe = artifact.proposal["proposed_facts"]["implementation"].get("library_recipe") or {}
        if not (recipe.get("kwargs") or {}).get(_REFUSAL_MARKER):
            return attempts
        failed = attempts[0]
        failed["result"] = "failed"
        failed["stage"] = "import"
        failed["mode"] = None
        failed["worker_receipt"] = dict(failed["worker_receipt"])
        failed["worker_receipt"]["mode"] = None
        failed["worker_receipt"]["constructor_started"] = False
        failed["worker_receipt"]["constructor_completed"] = False
        failed["worker_receipt"]["input_completed"] = False
        failed["worker_receipt"]["forward_started"] = False
        failed["worker_receipt"]["forward_completed"] = False
        failed["error"] = {
            "stage": "import",
            "reason_code": "import-exception",
            "exception_type": "menagerie.crawler.recipe.RecipeError",
            "message": _REFUSAL_TEXT,
            "traceback": f"Traceback: RecipeError: {_REFUSAL_TEXT}",
            "no_traceback_reason": None,
            "native_crash": False,
            "root_cause_fingerprint": HASH,
            "details": {"receipt_error": None},
        }
        failed["raw_award_receipt"] = None
        failed["raw_award_receipt_sha256"] = None
        parent_attestation = dict(failed["parent_attestation"])
        parent_attestation["completion_line_sha256"] = None
        parent_attestation["named_raw_award_receipt_sha256"] = None
        parent_attestation["attestation_sha256"] = stable_hash(
            {key: value for key, value in parent_attestation.items() if key != "attestation_sha256"}
        )
        failed["parent_attestation"] = parent_attestation
        failed["unattested_partial"] = {
            "state": "unattested-partial",
            "stage": "import",
            "reason_code": "import-exception",
            "diagnostic_sha256": None,
        }
        return [failed]


class AdmissionRefusingForward(ScriptedForward):
    """Raise the typed refusal at request-build time, before any worker spawns.

    Models the admission-time pin revalidation
    (``gates.revalidate_recipe_revision_pin`` inside ``_worker_request``):
    the lane itself raises ``RecipeError`` for marker proposals.
    """

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Raise for marker proposals; return clean attempts otherwise."""

        recipe = artifact.proposal["proposed_facts"]["implementation"].get("library_recipe") or {}
        if (recipe.get("kwargs") or {}).get(_REFUSAL_MARKER):
            self.calls[str(artifact.proposal["stable_id"])] = (
                self.calls.get(str(artifact.proposal["stable_id"]), 0) + 1
            )
            raise RecipeError(_REFUSAL_TEXT)
        return super().forward(artifact, environment, cold_runs, work_root)


def test_admission_raised_refusal_routes_to_the_same_repair(tmp_path: Path) -> None:
    """A RecipeError raised at admission repairs exactly like a worker refusal.

    The admission-time pin revalidation raises BEFORE any worker spawns; the
    lane classifies it ``import / import-exception`` and the same bounded
    repair adjudicates it -- never ``failed:runner / internal-error``.
    """

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    author = RecipeRefusalRepairAuthor()
    forward = AdmissionRefusingForward()

    result = _driver(tmp_path, snapshot, author=author, forward=forward).run()
    paths = _paths(tmp_path, snapshot)
    models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}

    assert result.status == "complete"
    assert models[stable_id]["status"]["code"] == "runs"
    assert author.calls == {stable_id: 2}
    requests = sorted(
        (paths.work_root / stable_id / "repair").glob("recipe-refusal-generation-*.json")
    )
    assert len(requests) == 1
    assert json.loads(requests[0].read_text())["refusal"] == _REFUSAL_TEXT


def test_recipe_refusal_routes_to_bounded_repair_and_runs(tmp_path: Path) -> None:
    """A deterministic loader refusal is repaired, re-gated, and executed.

    Before this route every such refusal terminalized
    ``failed:import / import-exception`` with the author never told what the
    loader said (rung-11 census: m11014 m3488 m4334 m7362 m7558 m8245 m8830
    m8838 m8867 m8887 m9133, one shared traceback).
    """

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    author = RecipeRefusalRepairAuthor()
    checker = FakeChecker()
    forward = RecipeRefusingForward()

    result = _driver(tmp_path, snapshot, author=author, checker=checker, forward=forward).run()
    paths = _paths(tmp_path, snapshot)
    models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}

    assert result.status == "complete"
    assert models[stable_id]["status"]["code"] == "runs"
    assert author.calls == {stable_id: 2}, "one initial author, one bounded repair"
    assert checker.metadata_calls == 4, (
        "the repaired proposal must reach a fresh gate round (two primary "
        "adjudications, each accepted round plus its H-5 blind confirmation)"
    )
    repair_root = paths.work_root / stable_id / "repair"
    requests = sorted(repair_root.glob("recipe-refusal-generation-*.json"))
    assert len(requests) == 1
    request = json.loads(requests[0].read_text())
    assert request["refusal"] == _REFUSAL_TEXT, (
        "the repair request must carry the exact RecipeError text"
    )
    charges = sorted(repair_root.glob("recipe-refusal-charge-*.json"))
    assert len(charges) == 1
    assert json.loads(charges[0].read_text())["disposition"] == ("revision-accepted-for-execution")


def test_recipe_refusal_cap_exhaustion_keeps_the_honest_import_terminal(
    tmp_path: Path,
) -> None:
    """Exhausting the bounded budget terminalizes on the ORIGINAL typed refusal.

    The terminal stays ``failed:import / import-exception`` -- the loader's
    own vocabulary -- never a ``failed:runner`` protocol lie, and the refusal
    text survives into the terminal detail chain.
    """

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    author = RecipeRefusalCapAuthor()
    forward = RecipeRefusingForward()

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        forward=forward,
        run_repair_max=2,
    ).run()
    paths = _paths(tmp_path, snapshot)
    model = scan_jsonl(paths.ledgers.models)[-1]

    assert result.status == "terminal-partition-complete"
    assert model["status"]["code"] == "failed:import"
    assert model["status"]["reason_code"] == "import-exception"
    assert author.calls == {stable_id: 3}, "one initial author plus the full bounded cap"
    repair_root = paths.work_root / stable_id / "repair"
    charges = sorted(repair_root.glob("recipe-refusal-charge-*.json"))
    assert len(charges) == 2, "every completed revision consumes the cap"
    assert all(
        json.loads(charge.read_text())["disposition"] == "revision-accepted-for-execution"
        for charge in charges
    )


def test_recipe_refusal_repair_is_bounded_and_never_loops(tmp_path: Path) -> None:
    """A still-refused accepted revision charges the cap, so recursion halts.

    Unlike the detected-modes loop, the driver cannot statically verify a
    recipe repair; only the worker can. An uncharged accepted-but-still-refused
    revision would recurse through the repair forever, so acceptance charges.
    """

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    author = RecipeRefusalCapAuthor()
    forward = RecipeRefusingForward()

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        forward=forward,
        run_repair_max=1,
    ).run()
    paths = _paths(tmp_path, snapshot)
    model = scan_jsonl(paths.ledgers.models)[-1]

    assert result.status == "terminal-partition-complete"
    assert model["status"]["code"] == "failed:import"
    # The still-refused revision reproduces the SAME proposal and execution
    # identity, so the recursion replays the recorded failed attempt from the
    # deterministic attempt cache instead of spawning a second worker run.
    assert forward.calls == {stable_id: 1}
    assert author.calls == {stable_id: 2}, "one initial author plus the single budgeted repair"
    charges = sorted((paths.work_root / stable_id / "repair").glob("recipe-refusal-charge-*.json"))
    assert len(charges) == 1


def test_deterministic_recipe_refusal_detection_is_exact() -> None:
    """Only an import-stage typed RecipeError is a repairable refusal."""

    error = {
        "stage": "import",
        "exception_type": "menagerie.crawler.recipe.RecipeError",
        "message": _REFUSAL_TEXT,
    }

    def attempt(**overrides: Any) -> dict[str, Any]:
        return {"attempt_id": "sha256:" + "a" * 64, "error": {**error, **overrides}}

    assert _deterministic_recipe_refusal(attempt(), None) == _REFUSAL_TEXT
    assert _deterministic_recipe_refusal(attempt(stage="forward"), None) is None
    assert (
        _deterministic_recipe_refusal(attempt(exception_type="builtins.ModuleNotFoundError"), None)
        is None
    )
    assert _deterministic_recipe_refusal(attempt(message=""), None) is None
    assert _deterministic_recipe_refusal({"attempt_id": "x", "error": None}, None) is None


def test_redacted_refusal_message_is_resolved_and_hash_verified(tmp_path: Path) -> None:
    """A redacted error.message resolves through its sidecar; tampering fails closed."""

    from menagerie.crawler.identity import canonical_json_bytes, hash_bytes

    attempt_id = "sha256:" + "b" * 64
    reference = {
        "redaction": "externally-controlled-text-v1",
        "content_sha256": hash_bytes(canonical_json_bytes(_REFUSAL_TEXT)),
        "local_path": f".crawl-local/diagnostics/{attempt_id}.json",
        "diagnostic_key": "$.error.message",
    }
    attempt = {
        "attempt_id": attempt_id,
        "error": {
            "stage": "import",
            "exception_type": "menagerie.crawler.recipe.RecipeError",
            "message": reference,
        },
    }
    (tmp_path / f"{attempt_id}.json").write_text(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "externally_controlled_fields": {"$.error.message": _REFUSAL_TEXT},
            }
        )
    )
    assert _deterministic_recipe_refusal(attempt, tmp_path) == _REFUSAL_TEXT
    # Tampered sidecar bytes no longer reproduce the recorded hash: fail closed.
    (tmp_path / f"{attempt_id}.json").write_text(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "externally_controlled_fields": {"$.error.message": "tampered text"},
            }
        )
    )
    assert _deterministic_recipe_refusal(attempt, tmp_path) is None
    # A missing sidecar root also fails closed.
    assert _deterministic_recipe_refusal(attempt, None) is None


def test_attempt_error_message_surfaces_the_full_exception_chain() -> None:
    """The masked-primary-error class (rung-11 R2 lane) is closed at the seam.

    A worker message of ``No module named 'pit'`` whose traceback carries the
    primary ``RecipeError`` must surface BOTH in ``$.error.message``.
    """

    traceback_text = (
        "Traceback (most recent call last):\n"
        '  File "recipe.py", line 1649, in load_typed_adapter\n'
        "menagerie.crawler.recipe.RecipeError: typed adapter recipe revision "
        "mismatch: expected sha256:aa, observed sha256:bb\n"
        "\n"
        "During handling of the above exception, another exception occurred:\n"
        "\n"
        "Traceback (most recent call last):\n"
        '  File "adapter.py", line 1, in <module>\n'
        "ModuleNotFoundError: No module named 'pit'\n"
    )
    worker_error = {
        "reason_code": None,
        "exception_type": "builtins.ModuleNotFoundError",
        "message": "No module named 'pit'",
        "traceback": traceback_text,
    }
    fields = _attempt_error_fields(
        "import",
        "import-exception",
        worker_error,
        "worker failed while loading the recipe",
        native_crash=False,
        details={},
    )
    assert "typed adapter recipe revision mismatch" in fields["message"]
    assert "No module named 'pit'" in fields["message"]
    assert "full exception chain, oldest first" in fields["message"]


def test_attempt_error_message_is_unchanged_without_a_chain() -> None:
    """A plain single-exception failure keeps its exact historical message."""

    worker_error = {
        "exception_type": "builtins.RuntimeError",
        "message": "synthetic forward failure",
        "traceback": (
            "Traceback (most recent call last):\n"
            '  File "worker.py", line 1, in run\n'
            "RuntimeError: synthetic forward failure\n"
        ),
    }
    fields = _attempt_error_fields(
        "forward",
        "mode-run",
        worker_error,
        "meaningful mode forward failed",
        native_crash=False,
        details={},
    )
    assert fields["message"] == "synthetic forward failure"


def test_exception_chain_message_is_fail_open() -> None:
    """Anomalous traceback payloads degrade to the plain message, never crash."""

    assert _exception_chain_message({"message": "x", "traceback": None}) is None
    assert _exception_chain_message({"message": None, "traceback": "y"}) is None
    assert (
        _exception_chain_message(
            {
                "message": "x",
                "traceback": "During handling of the above exception, another exception occurred:",
            }
        )
        is None
    )


def _charge_row(
    generation: int,
    *,
    work_id: str = "work-a",
    campaign: str = "campaign-a",
    fingerprint: str = "sha256:" + "f" * 64,
) -> dict[str, Any]:
    """Build one scoped charge row for the counting-helper tests."""

    return {
        "stable_id": "m_example",
        "generation": generation,
        "gate_kind": "recipe_refusal",
        "work_id": work_id,
        "campaign_root_work_id": campaign,
        "root_cause_fingerprint": fingerprint,
        "disposition": "revision-rejected",
        "error_type": "DriverIntegrationError",
    }


def test_spent_repair_budget_is_scoped_and_contiguous(tmp_path: Path) -> None:
    """Only matching, contiguous, well-formed charge rows count as spent budget.

    Sol wave-8 finding 5: a bare per-model glob let a new work generation or a
    different refusal class inherit earlier spent budget, and malformed or
    gapped filenames counted without validation.
    """

    from menagerie.crawler.driver_receipts import (
        _recipe_repair_scope_tag,
        _spent_recipe_repair_generations,
    )

    fingerprint = "sha256:" + "f" * 64
    scope = {
        "work_id": "work-a",
        "campaign_root_work_id": "campaign-a",
        "root_cause_fingerprint": fingerprint,
    }
    tag = _recipe_repair_scope_tag(**scope)

    def spent() -> int:
        return _spent_recipe_repair_generations(
            tmp_path, scope_tag=tag, stable_id="m_example", **scope
        )

    assert spent() == 0
    (tmp_path / f"recipe-refusal-charge-{tag}-1.json").write_text(json.dumps(_charge_row(1)))
    assert spent() == 1
    # A gap never counts past it: {1, 3} is one spent generation, not two.
    (tmp_path / f"recipe-refusal-charge-{tag}-3.json").write_text(json.dumps(_charge_row(3)))
    assert spent() == 1
    (tmp_path / f"recipe-refusal-charge-{tag}-2.json").write_text(json.dumps(_charge_row(2)))
    assert spent() == 3
    # A row whose CONTENT names a foreign work generation never counts, even
    # under a colliding filename tag.
    (tmp_path / f"recipe-refusal-charge-{tag}-4.json").write_text(
        json.dumps(_charge_row(4, work_id="work-b"))
    )
    assert spent() == 3
    # Malformed residue is inert.
    (tmp_path / f"recipe-refusal-charge-{tag}-5.json").write_text("{not json")
    assert spent() == 3
    # A DIFFERENT refusal class or work generation sees none of these charges.
    other = dict(scope, root_cause_fingerprint="sha256:" + "e" * 64)
    other_tag = _recipe_repair_scope_tag(**other)
    assert (
        _spent_recipe_repair_generations(
            tmp_path, scope_tag=other_tag, stable_id="m_example", **other
        )
        == 0
    )
    fresh_work = dict(scope, work_id="work-fresh")
    fresh_tag = _recipe_repair_scope_tag(**fresh_work)
    assert (
        _spent_recipe_repair_generations(
            tmp_path, scope_tag=fresh_tag, stable_id="m_example", **fresh_work
        )
        == 0
    )


def test_charge_rows_are_validated_field_by_field(tmp_path: Path) -> None:
    """Sol's R2 probe row and every schema violation are inert residue.

    The probe: filename suffix ``-999``, ``generation: true`` (a boolean!),
    foreign stable ID, gate kind, and disposition -- but a matching scope
    triple. It counted as one spent generation; now every field of the closed
    schema must validate and the filename suffix must reproduce the content.
    """

    from menagerie.crawler.driver_receipts import (
        _recipe_repair_scope_tag,
        _spent_recipe_repair_generations,
    )

    fingerprint = "sha256:" + "f" * 64
    scope = {
        "work_id": "work-a",
        "campaign_root_work_id": "campaign-a",
        "root_cause_fingerprint": fingerprint,
    }
    tag = _recipe_repair_scope_tag(**scope)
    assert len(tag) == 64, "the filename tag must be the FULL scope digest"

    def spent() -> int:
        return _spent_recipe_repair_generations(
            tmp_path, scope_tag=tag, stable_id="m_example", **scope
        )

    # Sol's exact probe row.
    probe = {
        "stable_id": "someone-else",
        "generation": True,
        "gate_kind": "not-a-recipe-refusal",
        "disposition": "not-a-disposition",
        "error_type": "X",
        **scope,
    }
    (tmp_path / f"recipe-refusal-charge-{tag}-999.json").write_text(json.dumps(probe))
    assert spent() == 0

    # Each violation alone keeps an otherwise-valid generation-1 row inert.
    violations: list[dict[str, Any]] = [
        {"generation": True},  # bool is not an int generation
        {"generation": "1"},
        {"stable_id": "someone-else"},
        {"gate_kind": "run_modes"},
        {"disposition": "accepted"},
        {"disposition": "revision-rejected", "error_type": None},
        {"disposition": "revision-accepted-for-execution", "error_type": "X"},
        {"generation": 2},  # filename suffix must reproduce the content
    ]
    path = tmp_path / f"recipe-refusal-charge-{tag}-1.json"
    for override in violations:
        row = _charge_row(1)
        if override.get("disposition") == "revision-accepted-for-execution":
            row["error_type"] = None
        row.update(override)
        path.write_text(json.dumps(row))
        assert spent() == 0, f"violation must be inert: {override!r}"
    # A non-numeric filename suffix is inert even with valid content.
    path.unlink()
    (tmp_path / f"recipe-refusal-charge-{tag}-extra.json").write_text(json.dumps(_charge_row(1)))
    assert spent() == 0
    # The accepted disposition validates with error_type None.
    accepted = _charge_row(1)
    accepted["disposition"] = "revision-accepted-for-execution"
    accepted["error_type"] = None
    path.write_text(json.dumps(accepted))
    assert spent() == 1


def test_foreign_and_legacy_charge_residue_does_not_consume_the_budget(
    tmp_path: Path,
) -> None:
    """Residue from other scopes or the legacy unscoped naming grants no budget.

    The m10949 lesson generalized to finding 5: a model must never start past
    its cap because of markers another work generation, another refusal class,
    or a pre-scoping legacy run left behind. This resume repairs and completes
    with the full budget available.
    """

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    paths = _paths(tmp_path, snapshot)
    repair_root = paths.work_root / stable_id / "repair"
    repair_root.mkdir(parents=True, exist_ok=True)
    # Legacy unscoped markers from a pre-scoping run.
    for generation in (1, 2):
        (repair_root / f"recipe-refusal-charge-{generation}.json").write_text(
            json.dumps(
                {
                    "stable_id": stable_id,
                    "generation": generation,
                    "gate_kind": "recipe_refusal",
                    "disposition": "revision-rejected",
                    "error_type": "DriverIntegrationError",
                }
            )
        )
    # A foreign work generation's fully scoped markers.
    (repair_root / "recipe-refusal-charge-aaaaaaaaaaaa-1.json").write_text(
        json.dumps(_charge_row(1, work_id="work-foreign"))
    )

    author = RecipeRefusalRepairAuthor()
    forward = RecipeRefusingForward()
    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        forward=forward,
        run_repair_max=2,
    ).run()
    paths = _paths(tmp_path, snapshot)
    model = scan_jsonl(paths.ledgers.models)[-1]

    assert result.status == "complete"
    assert model["status"]["code"] == "runs"
    assert author.calls == {stable_id: 2}, (
        "the residue must not pre-consume generations: one initial author, one repair"
    )
