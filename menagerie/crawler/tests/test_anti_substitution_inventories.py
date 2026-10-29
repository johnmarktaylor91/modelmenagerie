"""Round-17 class-level prevention and anti-theater CI inventories."""

from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import inspect
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pytest

import menagerie.crawler.artifact_transactions as artifact_module
import menagerie.crawler.authority as authority_module
import menagerie.crawler.driver as driver_module
import menagerie.crawler.driver_admission as driver_admission_module
import menagerie.crawler.env_lifecycle as lifecycle_module
import menagerie.crawler.worker_supervisor as supervisor_module
from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION_V3
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.tools import collection_gate, gate_closure
from menagerie.crawler.tools.anti_substitution import (
    _attribute_name,
    _comparison_owner_inventory as _inventory_comparison_owners,
    _dead_contract_occurrences as _inventory_dead_contract_occurrences,
    _enclosing_definition,
    _function_node,
    _sensitive_edge_inventory as _inventory_sensitive_edges,
    _string_constants,
    _subprocess_spawn_inventory,
    _substitution_boundary_errors,
    _tree_context,
)
from menagerie.crawler.schema import (
    SchemaOwner,
    load_schema,
    owned_schema_leaves,
    schema_leaf_paths,
)
from menagerie.crawler.tests.conftest import (
    make_attempt,
    make_supervised_worker_result_v3,
    make_worker_result_v3_mapping,
)
from menagerie.crawler.tests.test_award_worker_result_composition import VS1_LANDING_MANIFEST
from menagerie.crawler.tests.test_boundary_shutdown_composition import VS2_LANDING_MANIFEST
from menagerie.crawler.tests.test_compiler_os_authority_composition import VS3_LANDING_MANIFEST
from menagerie.crawler.worker_supervisor import SupervisorObservation


_CRAWLER_ROOT = Path(driver_module.__file__).parent
_REPOSITORY_ROOT = _CRAWLER_ROOT.parents[1]
_WORKFLOW_PATH = _REPOSITORY_ROOT / ".github" / "workflows" / "tests.yml"
# The CI-structure pins below assert workflows this repository does not have: the
# crawler-smoke job, gate-runner invocations, nightly/quality/lint/release workflows
# and .github/ci_dormancy.json (carried over from TorchLens-era expectations). The
# Menagerie CI is undecided until the crawler redesign (JMT 2026-10-01). Strict xfail
# keeps each pin visible and turns it red the day that CI lands; remove it then.
_NO_REPO_CI = pytest.mark.xfail(
    strict=True,
    reason="asserts CI workflows the menagerie repo does not have yet (see _NO_REPO_CI)",
)

_PROTOCOL_VERSIONS = {
    "_WORKER_RESULT_VERSION": "menagerie.crawler.worker-result.v3",
    "_WORKER_DIAGNOSTIC_VERSION": "menagerie.crawler.worker-receipt.v1",
    "_RAW_AWARD_RECEIPT_VERSION": "menagerie.crawler.raw-award-receipt.v3",
}
_DEAD_SYMBOLS = {
    "_driver_deferral_attempt",
    "_gate_item_for_result",
    "_gate_item_requires_repair",
    "store_licensed_artifact",
    "_supervise_environment_worker",
    "_validate_terminal_evidence",
    # Retired with the v2 reconstruction path. Reconstruction authority now comes
    # from the independent artifact ledger: `validate_artifact_checkpoint` proves
    # the transaction, and `_derive_candidate_paths` refuses any reconstruction
    # document no append-only artifact event names.
    "_reconstruction_has_canonical_anchor",
    # Superseded by `authority._TerminalProofPipeline._derive_deferral`. This one
    # must NOT be revived: nothing writes `attempt.defer_evidence` or
    # `capability_observation` (both hard-coded `None`) and no attempt is appended
    # for a `DeferRecommendation`, so re-wiring it makes every platform DEFER
    # unreducible.
    "_validate_deferral",
    # Never called. The live receipt read is `driver_receipts._verified_worker_result`.
    "_read_verified_worker_receipt",
    "validate_reconstruction_source_binding",
    "build_author_repair_envelope",
    "collect_executable_closure",
    "compile_execution_read_manifest_from_closure",
    "compare_identity_sets",
    "filter_funnel",
    "is_legacy_untrusted",
    "is_valid_payload",
}
_DEAD_OPTIONS = {"--scheduled-wake"}
_SENSITIVE_EDGE_COUNTS = Counter(
    {
        ("AdmissionEnvironmentMixin._ensure_authors", "dependencies.author.author"): 1,
        # Reviewed and registered deliberately: the concurrent author wave's commit path
        # authors INLINE exactly when the pool did not dispatch that item
        # (``pool.claim(item)`` returned ``None``), which is the documented fallback of the
        # bounded-degree design. It goes through the same ``_retry_infrastructure_call``
        # wrapper and ``admission=("author", item)`` boundary as the serial edge above, on
        # the same single canonical writer thread. This closed set exists so that a NEW
        # sensitive edge cannot appear without a human deciding it is legitimate -- it stayed
        # red for four commits and the mid tier is what caught it, which is the check working.
        ("AdmissionEnvironmentMixin._commit_author_wave", "dependencies.author.author"): 1,
        ("CrawlerDriver._repair_author", "dependencies.author.author"): 1,
        ("CrawlerDriver._repair_author_for_detected_modes", "dependencies.author.author"): 1,
        # Reviewed and registered deliberately (wave-8 striker B, recipe lanes): the
        # bounded recipe-refusal repair route authors a corrected proposal when a
        # deterministic RecipeError refuses a recipe, instead of burning mode retries
        # into a terminal `failed:import`. Same author lane, same budget accounting
        # as the two `_repair_author*` edges above. The pin lagged the merge: this
        # entry acknowledges the edge after review rather than widening anything.
        ("ReceiptDriverMixin._repair_author_for_recipe_refusal", "dependencies.author.author"): 1,
        # Reviewed rename plus two reviewed pool edges (M2 checker session pool). The
        # serial dispatch sites moved verbatim from `_ensure_gates` into
        # `_run_gate_rounds` when `_ensure_gates` became the pool-lifecycle wrapper.
        # The two `_prefetch_*` edges are the checker-lane analogue of the registered
        # `_commit_author_wave` edge above: exact-eligibility pool submissions whose
        # outcomes are replayed as attempt one through the SAME
        # `_retry_infrastructure_call` wrapper and `admission=("checker", item)`
        # boundary at the commit site, on the single canonical writer. Pool threads
        # only spawn/wait/capture; no append, park, or terminal happens off-writer.
        ("AdmissionEnvironmentMixin._run_gate_rounds", "dependencies.checker.check_metadata"): 2,
        ("AdmissionEnvironmentMixin._run_gate_rounds", "dependencies.checker.check_fidelity"): 1,
        (
            "AdmissionEnvironmentMixin._prefetch_metadata_batches",
            "dependencies.checker.check_metadata",
        ): 1,
        (
            "AdmissionEnvironmentMixin._prefetch_fidelity_sessions",
            "dependencies.checker.check_fidelity",
        ): 1,
        # Reviewed and registered deliberately (DX-54, terminal-review sweep
        # parallelization): the terminal lane's dispatch edges enter the closed
        # set for the first time. The sweep's prefetch is the checker-lane
        # analogue of the two `_prefetch_*` edges above -- exact-eligibility
        # pool submissions (`reusable_terminal_gate` consulted before submit)
        # whose outcomes replay as attempt one through the SAME
        # `_retry_infrastructure_call` wrapper and `admission=("checker", item)`
        # boundary at the commit site, on the single canonical writer. The
        # `_obtain_terminal_disposition_gate` edge is the pre-existing serial
        # dispatch site, unchanged in count, now pinned so a second terminal
        # dispatch path cannot appear unreviewed.
        (
            "CrawlerDriver._obtain_terminal_disposition_gate",
            "dependencies.checker.check_terminal",
        ): 1,
        (
            "AdmissionEnvironmentMixin._prefetch_terminal_review_sessions",
            "dependencies.checker.check_terminal",
        ): 1,
        # Reviewed and registered deliberately (AG-4 duplicate-accept, Wave-2
        # batch assembly): a FRESH terminal ACCEPT buys exactly ONE more LIVE
        # confirmation session -- never a pool capsule -- over the SAME
        # question (`terminal_question_binding` equality), settling on the
        # conflict ledger bound to the accepted row's gate_id. Same
        # `_retry_infrastructure_call` wrapper and `admission=("checker",
        # item)` boundary as the serial dispatch edge above, on the single
        # canonical writer. Rejections are never duplicated; sweep re-gates
        # reuse persisted witnesses instead of buying a third session
        # (pinned in test_checker_verdict_conflicts.py). The edge lagged the
        # w2-checker lane's merge and this census caught it at batch
        # assembly -- the tripwire working, acknowledged after review.
        # Seat swap (SEAT_RULING.md 2026-09-01, commit 5770e146d): the terminal
        # confirmation is bought on the CONFIRMATION seat through the module-level
        # `buy_confirmation_session` helper (below), so the direct
        # `dependencies.checker.check_terminal` edge moved into that helper. The
        # helper call itself is the registered sensitive edge: a new caller
        # cannot appear without joining this closed set.
        ("CrawlerDriver._confirm_terminal_acceptance", "buy_confirmation_session"): 1,
        # Reviewed and registered deliberately (H-5, rule-2 trilab consensus
        # 2026-08-29): the metadata/fidelity duplicate-accept write door. The
        # calibration receipt's acceptance_witness block binds production
        # acceptance depth K=2 UNCONDITIONALLY for every gate kind; these two
        # edges are the accuracy-lane twins of the terminal confirmation edge
        # above -- a FRESH accuracy ACCEPT buys exactly ONE more LIVE blind
        # session (never a pool capsule) over the SAME question
        # (`accuracy_question_binding` equality), settling on the conflict
        # ledger as a v2 duplicate-accept row bound to the accepted row's
        # gate_id + stable_id. Rejections are never duplicated; persisted
        # same-question accept witnesses are reused instead of buying a third
        # session; an unfundable confirmation parks checker-accept-unconfirmed
        # (the budget fuse), never publishing unconfirmed.
        # Seat swap (commit 5770e146d): metadata/fidelity confirmations are bought
        # on the CONFIRMATION seat through `buy_confirmation_session`; the helper
        # is the ONE place that reaches the checker lane's check_* methods for a
        # confirmation, one edge per gate kind, on the `lane` parameter (always
        # `self.dependencies.checker` at both registered call sites).
        (
            "AdmissionEnvironmentMixin._require_confirmed_accuracy_acceptance",
            "buy_confirmation_session",
        ): 1,
        ("buy_confirmation_session", "lane.check_metadata"): 1,
        ("buy_confirmation_session", "lane.check_fidelity"): 1,
        ("buy_confirmation_session", "lane.check_terminal"): 1,
        ("AdmissionEnvironmentMixin._run_environment_work", "dependencies.environments.run"): 1,
        ("SupervisedForwardLane.forward", "open_worker_lease"): 1,
        ("SupervisedForwardLane.forward", "supervise_worker"): 1,
        ("ReceiptDriverMixin._forward_and_reduce", "dependencies.forward.forward"): 2,
        ("CrawlerDriver._authorize_and_publish_artifact", "publish_authorized_artifact"): 1,
        ("CrawlerDriver._authorize_terminal_artifact", "publish_authorized_artifact"): 1,
        # Reviewed extraction, not a new edge (M3 scheduler). The award tail moved
        # from `_forward_and_reduce` into `_award_worker_result` so the pipelined
        # scheduler can award `runs` from recorded triage attempts through the SAME
        # tail the wave driver runs in-turn. There is still exactly ONE run-award
        # append owner; `_forward_and_reduce` itself performs no model append -- it
        # only routes to this one.
        ("ReceiptDriverMixin._award_worker_result", "append_model"): 1,
        # Reviewed rename, not a new edge. The canonical terminal append moved from
        # `_terminalize` into `_append_terminal_revision` when `_terminalize` became the
        # guard that survives a failure in the bookkeeping recording a failure. There is
        # still exactly ONE terminal append owner, and `_terminalize` itself performs no
        # append -- it only routes to this one.
        ("CrawlerDriver._append_terminal_revision", "append_model"): 1,
        # Reviewed and registered deliberately (terminal-review backlog sweep): the
        # per-leg sweep re-appends a REFUSED backlog head verbatim with the
        # containment ladder's review-held marking, so the outstanding requeue
        # grant's `unreviewed-terminal` hold can release. It is not a terminal
        # MINT (`_append_terminal_revision` assembles a fresh record from a live
        # item+artifact; the sweep supersedes a settled record byte-preserving),
        # it publishes nothing (no `_authorize_*` call exists on the path -- the
        # boundary-shutdown structural audit pins that), and it can only ADD the
        # `human_review` requirement to an already-negative disposition.
        ("CrawlerDriver._review_terminal_backlog", "append_model"): 1,
        # Reviewed and registered deliberately (orphan-citation adoption,
        # 2026-08-27): the per-leg adoption pass re-appends an already
        # review-held failure terminal BYTE-VERBATIM under the leg's own
        # joined run when the head's in-window citation names a run that
        # never joined the open generation ("position is not provenance").
        # Disposition-invariant by construction (the closed
        # `orphan_hold_adoption_refusal` rule plus the post-prepare
        # `orphan_adoption_drift` refusal bound every append to exactly
        # parentage + re-citation), publication-free (the boundary-shutdown
        # structural audit pins no `_authorize_*` call on the path), and every
        # orphan -- adopted or refused -- is disclosed on the leg's durable
        # orphan-citation-adoption census event. Since P5-SOL-05 (round 4) the
        # append itself moved into the reducer write door
        # `CanonicalReducer.append_orphan_adoption`, which re-derives the FULL
        # closed adoption rule from ledgers before its one internal
        # `append_model` call (audited in test_boundary_shutdown_composition);
        # the driver pass calls only that API, so THIS is the pinned edge.
        ("CrawlerDriver._adopt_orphan_citations", "append_orphan_adoption"): 1,
    }
)
_SENSITIVE_SUFFIXES = frozenset(suffix for _owner, suffix in _SENSITIVE_EDGE_COUNTS)
_ROUND17_CI_NODES = (
    "menagerie/crawler/tests/test_award_worker_result_composition.py::"
    "test_real_v3_worker_result_awards_through_driver_and_reducer",
    "menagerie/crawler/tests/test_boundary_shutdown_composition.py::"
    "test_signal_after_real_forward_publishes_and_awards_nothing_then_resumes",
    "menagerie/crawler/tests/test_slice_f_driver.py::"
    "test_linux_handoff_attempts_both_deferred_statuses_and_supersedes",
)
_ROUND19_RELEASE_NODE_INVENTORY = {
    "golden": _ROUND17_CI_NODES[0],
    "interpreter": (
        "menagerie/crawler/tests/test_environment_authority_composition.py::"
        "test_outside_selected_interpreter_is_rejected_at_binding"
    ),
    "linux-denial": (
        "menagerie/crawler/tests/test_environment_authority_composition.py::"
        "test_linux_real_compiler_denies_caught_undeclared_repo_read_and_awards_package"
    ),
    "shutdown": _ROUND17_CI_NODES[1],
    "clean-clone": _ROUND17_CI_NODES[2],
    "unverifiable": (
        "menagerie/crawler/tests/test_award_worker_result_composition.py::"
        "test_real_unhashable_output_awards_runs_with_unverifiable_modes"
    ),
    "dry-run-run-resume": (
        "menagerie/crawler/tests/test_execution_dry_run_composition.py::"
        "test_documented_dry_run_and_resume_use_real_environment"
    ),
    "dry-run-false-complete": (
        "menagerie/crawler/tests/test_execution_dry_run_composition.py::"
        "test_dry_run_all_source_failure_is_acceptance_error"
    ),
    "cache": (
        "menagerie/crawler/tests/test_environment_authority_composition.py::"
        "test_real_multi_model_cache_closes_currentness_and_quarantines_mutation"
    ),
    "structural": "menagerie/crawler/tests/test_anti_substitution_inventories.py",
    "macos-positive-negative": (
        "menagerie/crawler/tests/test_environment_authority_composition.py::"
        "test_macos_real_compiler_denies_caught_undeclared_repo_read_and_awards_package"
    ),
    "macos-profile": (
        "menagerie/crawler/tests/test_environment_authority_composition.py::"
        "test_macos_v3_profile_has_one_fresh_literal_prefix_and_exact_outside_members"
    ),
}
_REQUIRED_CI_SELECTIONS = (
    "$REGISTRY_MODULES -m round21_linux_real",
    "$REGISTRY_MODULES -m round21_macos_real",
)
_QUARANTINE_PATH = _CRAWLER_ROOT / "legacy_manifest_audit.py"


def _workflow_test_paths(workflow_source: str) -> tuple[Path, ...]:
    """Return every crawler test path selected textually by CI.

    Parameters
    ----------
    workflow_source:
        Complete workflow YAML text.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Existing selected crawler test modules in stable order.
    """

    relative_paths = {
        token.split("::", 1)[0]
        for token in re.findall(r"menagerie/crawler/tests/[A-Za-z0-9_./:-]+", workflow_source)
    }
    return tuple(
        sorted(
            path for relative in relative_paths if (path := _REPOSITORY_ROOT / relative).is_file()
        )
    )


def _registered_composition_roots() -> dict[Path, tuple[str, ...]]:
    """Return exact landing-manifest real nodes grouped by their source path.

    Returns
    -------
    dict[pathlib.Path, tuple[str, ...]]
        Registered source paths and exact top-level pytest function roots.
    """

    default_paths = (
        (VS1_LANDING_MANIFEST, _CRAWLER_ROOT / "tests" / "test_award_worker_result_composition.py"),
        (
            VS2_LANDING_MANIFEST,
            _CRAWLER_ROOT / "tests" / "test_boundary_shutdown_composition.py",
        ),
        (
            VS3_LANDING_MANIFEST,
            _CRAWLER_ROOT / "tests" / "test_compiler_os_authority_composition.py",
        ),
    )
    grouped: dict[Path, set[str]] = {}
    for manifest, default_path in default_paths:
        for raw_node in manifest["real_composition_nodes"]:
            node_id = str(raw_node)
            if "::" in node_id:
                path_text, function_name = node_id.split("::", 1)
                path = _REPOSITORY_ROOT / path_text
            else:
                path = default_path
                function_name = node_id
            grouped.setdefault(path, set()).add(function_name.split("[", 1)[0])
    return {path: tuple(sorted(roots)) for path, roots in grouped.items()}


def _composition_source_paths(workflow_source: str | None = None) -> tuple[Path, ...]:
    """Discover the complete §8.3 composition and fixture-module scope.

    Parameters
    ----------
    workflow_source:
        Optional workflow mutation used by self-proofs.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Composition modules, transitive fixture/support modules, and CI selections.
    """

    workflow = (
        _WORKFLOW_PATH.read_text(encoding="utf-8") if workflow_source is None else workflow_source
    )
    test_root = _CRAWLER_ROOT / "tests"
    paths = {
        _CRAWLER_ROOT / "cli.py",
        test_root / "conftest.py",
        test_root / "dry_run_support.py",
        test_root / "test_slice_f_driver.py",
        *test_root.glob("test_*composition*.py"),
        *_workflow_test_paths(workflow),
        *_registered_composition_roots(),
    }
    return tuple(sorted(path for path in paths if path.is_file()))


_COMPOSITION_SOURCES = _composition_source_paths()

VS4_LANDING_MANIFEST: dict[str, Any] = {
    "findings": ("SOL-R16-07", "Fable-F5"),
    "production_symbols": {
        "driver": ("_SHUTDOWN_ADMISSION_REGISTRY", "_AWARD_CLOSURE_SYMBOLS"),
        "worker_supervisor": (
            "_SUBPROCESS_SPAWN_REGISTRY",
            "run_isolated_subprocess",
            "verify_supervised_worker_result",
        ),
        "artifact_transactions": (
            "_PUBLIC_WRITER_REGISTRY",
            "publish_authorized_artifact",
            "resolve_final_artifact_transaction",
        ),
    },
    "real_composition_nodes": _ROUND17_CI_NODES,
    "structural_nodes": (
        "test_spawn_inventory_is_closed_and_lease_bearing",
        "test_admission_boundary_inventory_is_closed",
        "test_public_writer_inventory_is_authorization_bearing",
        "test_dead_symbol_inventory_is_closed",
        "test_attempt_schema_consumers_and_producers_have_exact_parity",
        "test_protocol_literal_inventory_has_only_reviewed_comparison_owners",
        "test_synthetic_worker_result_factory_matches_production_protocol_shapes",
        "test_real_compositions_are_explicitly_selected_in_ci",
    ),
}

ROUND21_VS1_PROOF_REGISTRY: dict[str, str] = {
    "P01": (
        "menagerie/crawler/tests/test_release_preclusion_composition.py::"
        "test_preclusion_real_v3_path_has_no_substitutable_fixture_edge"
    ),
    "T01": (
        "menagerie/crawler/tests/test_release_preclusion_composition.py::"
        "test_tripwire_catches_python_evasion"
    ),
    "T01-CI": (
        "menagerie/crawler/tests/test_release_preclusion_composition.py::"
        "test_tripwire_catches_deleted_ci_node"
    ),
    "T02": (
        "menagerie/crawler/tests/test_anti_substitution_inventories.py::"
        "test_legacy_manifest_v1_is_quarantined_from_every_live_import_graph"
    ),
}

ROUND21_VS2_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS1_PROOF_REGISTRY,
    "P02": (
        "menagerie/crawler/tests/test_release_fingerprint_composition.py::"
        "test_cheap_fingerprint_catches_stat_preserved_mutation_without_false_staling_clone"
    ),
}

ROUND21_VS3_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS2_PROOF_REGISTRY,
    "P03": (
        "menagerie/crawler/tests/test_release_scale_composition.py::"
        "test_pass_and_spawn_validation_walks_are_constant_bounded"
    ),
    "T03": (
        "menagerie/crawler/tests/test_anti_substitution_inventories.py::"
        "test_verification_tree_walk_inventory_is_closed"
    ),
}

ROUND21_VS4_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS3_PROOF_REGISTRY,
    "P04": (
        "menagerie/crawler/tests/test_release_environment_matrix_composition.py::"
        "test_environment_authority_unit_matrix"
    ),
    "P12": _ROUND17_CI_NODES[0],
    "P13": _ROUND19_RELEASE_NODE_INVENTORY["interpreter"],
    "P14": _ROUND19_RELEASE_NODE_INVENTORY["linux-denial"],
    "P17": _ROUND19_RELEASE_NODE_INVENTORY["unverifiable"],
    "P19": (
        "menagerie/crawler/tests/test_environment_authority_composition.py::"
        "test_manifest_v3_rejects_changed_interpreter_association"
    ),
}

ROUND21_VS5_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS4_PROOF_REGISTRY,
    "P05": (
        "menagerie/crawler/tests/test_release_shutdown_matrix_composition.py::test_shutdown_matrix"
    ),
}

ROUND21_VS6_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS5_PROOF_REGISTRY,
    "P06": (
        "menagerie/crawler/tests/test_release_handoff_authority_composition.py::"
        "test_handoff_authority_identity_matrix"
    ),
}

ROUND21_VS7_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS6_PROOF_REGISTRY,
    "P07": (
        "menagerie/crawler/tests/test_release_transport_composition.py::"
        "test_closed_transport_capability_awards_and_rejects_unlisted_library"
    ),
}

ROUND21_VS8_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS7_PROOF_REGISTRY,
    "P08": (
        "menagerie/crawler/tests/test_release_cache_rebind_composition.py::"
        "test_mismatched_rebind_preserves_active_authority_and_awards"
    ),
}

ROUND21_VS9_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS8_PROOF_REGISTRY,
    "P09": (
        "menagerie/crawler/tests/test_release_ci_composition.py::"
        "test_linux_committed_lock_provenance_awards_in_ci"
    ),
    "T04": (
        "menagerie/crawler/tests/test_anti_substitution_inventories.py::"
        "test_linux_release_artifacts_and_provisioning_are_real"
    ),
    "T05": (
        "menagerie/crawler/tests/test_anti_substitution_inventories.py::"
        "test_linux_release_registry_is_exact"
    ),
}

ROUND21_VS10_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS9_PROOF_REGISTRY,
    "P10": (
        "menagerie/crawler/tests/test_release_ci_composition.py::"
        "test_macos_committed_lock_seatbelt_award_and_denial"
    ),
    "T04-mac": (
        "menagerie/crawler/tests/test_anti_substitution_inventories.py::"
        "test_macos_release_artifacts_and_provisioning_are_real"
    ),
    "T05-mac": (
        "menagerie/crawler/tests/test_anti_substitution_inventories.py::"
        "test_macos_release_registry_is_exact"
    ),
}

ROUND21_VS11_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS10_PROOF_REGISTRY,
    "P11": (
        "menagerie/crawler/tests/test_release_conformance_composition.py::"
        "test_conformance_registry_is_total_and_executed"
    ),
    "P11-CI": (
        "menagerie/crawler/tests/test_release_conformance_composition.py::"
        "test_ci_attestations_cover_registry_without_skip"
    ),
    "P27": (
        "menagerie/crawler/tests/test_slice_f_driver.py::"
        "test_linux_code_less_deferral_fails_visibly_without_failed_source"
    ),
    "T06": "menagerie/crawler/tools/round21_reversions.py",
}


@pytest.mark.smoke
def test_verification_tree_walk_inventory_is_closed() -> None:
    """Every complete prefix walk and v3 reuse site stays explicitly registered."""

    authority_source = _source(authority_module)
    tree, parents, names = _tree_context(authority_source)
    observed_walks = Counter(
        _enclosing_definition(node, parents, names)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _attribute_name(node.func) == "_scan_environment_tree"
    )
    assert observed_walks == Counter(authority_module._ENVIRONMENT_TREE_WALK_REGISTRY)  # noqa: SLF001

    required_token_sites = {
        "authority.py": {
            "collect_executable_closure_v3",
            "compile_execution_read_manifest_v3_from_closure",
            "environment_read_capability",
            "verify_execution_read_manifest_v3",
        },
        "driver_admission.py": {
            "AdmissionEnvironmentMixin._run_environment_work",
        },
        "driver_receipts.py": {
            "ReceiptDriverMixin._forward_and_reduce",
            "SupervisedForwardLane.forward",
            "_collect_worker_executable_closure",
            "_compile_worker_read_manifest",
        },
        "policy.py": {"generate_macos_sandbox_profile", "verify_execution_read_manifest"},
        "worker_supervisor.py": {
            "_request_allowed_read_paths",
            "run_isolated_subprocess",
            "supervise_worker",
        },
    }
    modules = {
        "authority.py": authority_module,
        "driver_admission.py": driver_admission_module,
        "driver_receipts.py": __import__(
            "menagerie.crawler.driver_receipts", fromlist=["driver_receipts"]
        ),
        "policy.py": __import__("menagerie.crawler.policy", fromlist=["policy"]),
        "worker_supervisor.py": supervisor_module,
    }
    verification_calls = {
        "cache.verify",
        "_collect_worker_executable_closure",
        "_compile_worker_read_manifest",
        "_current_run_is_fresh",
        "_forward_and_reduce",
        "collect_executable_closure_v3",
        "compile_execution_read_manifest_v3",
        "compile_execution_read_manifest_v3_from_closure",
        "environment_read_capability",
        "supervise_worker",
        "verify_environment_authority",
        "verify_execution_read_manifest",
        "verify_execution_read_manifest_v3",
    }
    for filename, required_owners in required_token_sites.items():
        source = _source(modules[filename])
        module_tree, module_parents, module_names = _tree_context(source)
        token_owners = {
            _enclosing_definition(node, module_parents, module_names)
            for node in ast.walk(module_tree)
            if isinstance(node, ast.Call)
            and any(_attribute_name(node.func).endswith(name) for name in verification_calls)
            and any(keyword.arg == "verification_token" for keyword in node.keywords)
        }
        assert required_owners <= token_owners


def _source(module: Any) -> str:
    """Return one imported production module's exact source text."""

    return Path(module.__file__).read_text(encoding="utf-8")


def _sensitive_edge_inventory(source: str) -> Counter[tuple[str, str]]:
    """Return exact sensitive call-edge counts for the test-owned closed set.

    Parameters
    ----------
    source:
        Complete Python source to inspect.

    Returns
    -------
    collections.Counter[tuple[str, str]]
        Exact enclosing-owner and sensitive-suffix counts.
    """

    return _inventory_sensitive_edges(source, _SENSITIVE_SUFFIXES)


def _dead_contract_occurrences(source: str) -> set[str]:
    """Return occurrences from the test-owned forbidden contract sets.

    Parameters
    ----------
    source:
        Complete Python source to inspect.

    Returns
    -------
    set[str]
        Forbidden identifiers and retired option literals found in the source.
    """

    return _inventory_dead_contract_occurrences(source, _DEAD_SYMBOLS, _DEAD_OPTIONS)


def _comparison_owner_inventory(source: str) -> set[tuple[str, str]]:
    """Return owners comparing the test-owned closed protocol constant set.

    Parameters
    ----------
    source:
        Complete Python source to inspect.

    Returns
    -------
    set[tuple[str, str]]
        Exact enclosing-owner and protocol-constant pairs.
    """

    return _inventory_comparison_owners(source, _PROTOCOL_VERSIONS)


def _composition_roots(path: Path, source: str) -> tuple[str, ...] | None:
    """Return exact roots for support modules that also contain unit-only fixtures."""

    if path.name == "conftest.py":
        raise AssertionError("conftest roots require the transitive fixture graph")
    if path.name == "test_slice_f_driver.py":
        return ("test_linux_handoff_attempts_both_deferred_statuses_and_supersedes",)
    if path.name in {"cli.py", "dry_run_support.py"}:
        tree = ast.parse(source, filename=str(path))
        return tuple(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
    registered = _registered_composition_roots().get(path)
    if registered is not None and "composition" not in path.stem:
        return registered
    return None


def _transitive_fixture_roots(paths: Sequence[Path]) -> tuple[str, ...]:
    """Resolve every applicable conftest fixture dependency to a fixed point.

    Parameters
    ----------
    paths:
        Complete discovered composition/registered/CI source scope.

    Returns
    -------
    tuple[str, ...]
        Exact conftest fixture roots plus the real fixture data/lane classes.
    """

    conftest_path = _CRAWLER_ROOT / "tests" / "conftest.py"
    conftest_tree = ast.parse(
        conftest_path.read_text(encoding="utf-8"), filename=str(conftest_path)
    )
    fixtures = {
        node.name: node
        for node in conftest_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            _attribute_name(
                decorator.func if isinstance(decorator, ast.Call) else decorator
            ).endswith("fixture")
            for decorator in node.decorator_list
        )
    }
    requested: set[str] = set()
    for path in paths:
        if path == conftest_path:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        selected_roots = _composition_roots(path, source)
        roots = set(selected_roots or ())
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if roots and node.name not in roots:
                continue
            if not roots and not node.name.startswith("test_"):
                continue
            requested.update(
                parameter.arg for parameter in (*node.args.posonlyargs, *node.args.args)
            )

    resolved: set[str] = set()
    pending = requested & fixtures.keys()
    while pending:
        fixture_name = pending.pop()
        if fixture_name in resolved:
            continue
        resolved.add(fixture_name)
        fixture = fixtures[fixture_name]
        dependencies = {
            parameter.arg for parameter in (*fixture.args.posonlyargs, *fixture.args.args)
        }
        pending.update((dependencies & fixtures.keys()) - resolved)
    return tuple(
        sorted(
            {
                "RealEnvironmentFixture",
                "RealEnvironmentLane",
                "real_environment_registry",
                *resolved,
            }
        )
    )


def _composition_scope_errors() -> tuple[str, ...]:
    """Return substitution diagnostics across every discovered composition edge."""

    errors: list[str] = []
    paths = _composition_source_paths()
    fixture_roots = _transitive_fixture_roots(paths)
    for path in paths:
        source = path.read_text(encoding="utf-8")
        roots = fixture_roots if path.name == "conftest.py" else _composition_roots(path, source)
        errors.extend(
            _substitution_boundary_errors(
                source,
                source_path=path.relative_to(_REPOSITORY_ROOT),
                root_definitions=roots,
            )
        )
    return tuple(sorted(set(errors)))


def _required_ci_selection_errors(workflow_source: str) -> tuple[str, ...]:
    """Return fully located diagnostics for missing required real CI nodes."""

    return tuple(
        f".github/workflows/tests.yml:<workflow>:{node}:deleted-ci-node"
        for node in sorted(_REQUIRED_CI_SELECTIONS)
        if node not in workflow_source
    )


def _production_python_paths() -> tuple[Path, ...]:
    """Return every crawler production Python path, excluding tests and tools."""

    return tuple(
        path
        for path in sorted(_CRAWLER_ROOT.rglob("*.py"))
        if "tests" not in path.parts and "tools" not in path.parts
    )


def _schema_parity_errors(schema: Mapping[str, Any], sources: Mapping[str, str]) -> tuple[str, ...]:
    """Return Round-17 attempt schema/consumer/producer parity violations."""

    leaves = schema_leaf_paths(schema)
    expected_leaves = {
        "$.capability_observation.claim",
        "$.capability_observation.supported",
        "$.raw_award_receipt.observation.output_value_sha256",
        "$.worker_receipt.output_value_sha256",
    }
    errors = [f"schema-missing:{path}" for path in sorted(expected_leaves - leaves)]
    required_literals = {
        "authority.py:derive_mode_summary": {"output_value_sha256"},
        "authority.py:_TerminalProofPipeline._derive_deferral": {
            "capability_observation",
            "claim",
            "supported",
        },
        "worker.py:_mode_receipt": {"output_value_sha256"},
        "worker.py:_raw_award_receipt": {"output_value_sha256"},
        "driver_receipts.py:_attempts_from_supervised": {
            "capability_observation",
            "output_value_sha256",
        },
        "driver_models.py:_driver_failure_attempt": {
            "capability_observation",
            "output_value_sha256",
        },
    }
    for owner, required in required_literals.items():
        filename, function_name = owner.split(":", 1)
        function = _function_node(sources[filename], function_name)
        missing = required - _string_constants(function)
        errors.extend(f"consumer-or-producer-missing:{owner}:{key}" for key in sorted(missing))
    return tuple(errors)


def _assert_symbol(module: Any, qualified_name: str) -> None:
    """Assert that one dot-qualified production symbol exists."""

    current = module
    for part in qualified_name.split("."):
        current = getattr(current, part)


def _assert_test_node_exists(node_id: str, default_path: Path) -> None:
    """Assert that one manifest node names a pytest-collectable top-level function."""

    if "::" in node_id:
        raw_path, function_name = node_id.split("::", 1)
        path = _REPOSITORY_ROOT / raw_path
    else:
        path = default_path
        function_name = node_id
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
        and node.name.startswith("test_")
    ]
    assert len(matches) == 1, f"manifest node is not pytest-collectable: {node_id}"


@pytest.mark.smoke
def test_receipt_consumers_are_confined_to_the_typed_projection() -> None:
    """No production semantic consumer may read the opaque wrapper directly."""

    observed: set[tuple[str, str]] = set()
    for path in _production_python_paths():
        tree, parents, names = _tree_context(path.read_text(encoding="utf-8"))
        observed.update(
            (path.name, _enclosing_definition(node, parents, names))
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "worker_receipt"
        )
    assert observed == {
        ("worker_supervisor.py", "verify_supervised_worker_result"),
        ("worker_supervisor.py", "worker_result_outer_for_diagnostics"),
    }


@pytest.mark.smoke
def test_spawn_inventory_is_closed_and_lease_bearing() -> None:
    """Every process spawn must keep its reviewed role and v3 lease inheritance."""

    source = _source(supervisor_module)
    assert supervisor_module._SUBPROCESS_SPAWN_REGISTRY == {  # noqa: SLF001
        "_emit_macos_audit_sentinel": "audit-sentinel:no-model-work",
        "_start_macos_denial_audit": "audit-collector:no-model-work",
        "run_isolated_subprocess": "model-worker:inherited-live-lease",
    }
    assert _subprocess_spawn_inventory(source) == Counter(
        {name: 1 for name in supervisor_module._SUBPROCESS_SPAWN_REGISTRY}  # noqa: SLF001
    )
    spawn = _function_node(source, "run_isolated_subprocess")
    calls = [
        node
        for node in ast.walk(spawn)
        if isinstance(node, ast.Call) and _attribute_name(node.func) == "subprocess.Popen"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords}
    assert keywords["shell"] == "False"
    assert keywords["start_new_session"] == "True"
    assert keywords["close_fds"] == "True"
    assert keywords["pass_fds"] == "inherited_fds"
    supervisor = _function_node(source, "supervise_worker")
    assert "v3 model worker spawn requires an inherited live worker lease" in _string_constants(
        supervisor
    )

    mutated = f"{source}\n\ndef rogue_model_spawn():\n    subprocess.Popen(['python'])\n"
    assert _subprocess_spawn_inventory(mutated) != _subprocess_spawn_inventory(source)


@pytest.mark.smoke
def test_admission_boundary_inventory_is_closed() -> None:
    """All authority-bearing admissions must remain classified and guarded."""

    source = _source(driver_module)
    admission_source = _source(driver_admission_module)
    receipt_source = _source(
        __import__("menagerie.crawler.driver_receipts", fromlist=["driver_receipts"])
    )
    observed_edges = (
        _sensitive_edge_inventory(source)
        + _sensitive_edge_inventory(admission_source)
        + _sensitive_edge_inventory(receipt_source)
    )
    assert observed_edges == _SENSITIVE_EDGE_COUNTS
    assert driver_module._SHUTDOWN_ADMISSION_REGISTRY == {  # noqa: SLF001
        "author": "guard:author-admission",
        "checker": "guard:checker-admission",
        "environment-create": "guard:environment-create-admission",
        "environment-use": "guard:environment-use-admission",
        "model": "guard:model-admission",
        "lease": "guard:forward-admission|pre-slot-resolution",
        "spawn": "guard:forward-admission|pre-slot-resolution",
        "run-model-assembly": "guard:post-attempt-pre-award",
        "publication-admission": "guard:pre-publication-admission",
        "publication": "atomic:award-commit",
        "terminal-publication": "atomic:award-commit",
        "model-append": "atomic:award-commit",
        "post-award-observation": "guard:post-award-commit",
    }
    # The run-award section lives in `_award_worker_result` (reviewed M3 extraction);
    # `_forward_and_reduce` delegates to it and the ordering audit follows the owner.
    forward = inspect.getsource(driver_module.CrawlerDriver._award_worker_result)
    terminal = inspect.getsource(driver_module.CrawlerDriver._append_terminal_revision)
    assert "_award_worker_result(" in inspect.getsource(
        driver_module.CrawlerDriver._forward_and_reduce
    )
    assert forward.index('"post-attempt-pre-award"') < forward.index("_assemble_run_model(")
    assert forward.index('"pre-publication-admission"') < forward.index('"pre-award-commit"')
    assert forward.index('"pre-award-commit"') < forward.index("reducer.append_model(")
    assert forward.index("reducer.append_model(") < forward.index('"post-award-commit"')
    assert terminal.index('"pre-publication-admission"') < terminal.index('"pre-award-commit"')
    assert terminal.index('"pre-award-commit"') < terminal.index("reducer.append_model(")
    assert terminal.index("reducer.append_model(") < terminal.index('"post-award-commit"')

    mutated = (
        f"{source}\n\ndef rogue_admission(self):\n"
        "    self.dependencies.author.author(None, None, None, None)\n"
    )
    assert (
        _sensitive_edge_inventory(mutated)
        + _sensitive_edge_inventory(admission_source)
        + _sensitive_edge_inventory(receipt_source)
        != _SENSITIVE_EDGE_COUNTS
    )


@pytest.mark.smoke
def test_public_writer_inventory_is_authorization_bearing() -> None:
    """Public placement must have one low-level writer and one authorized caller."""

    source = _source(artifact_module)
    assert artifact_module._PUBLIC_WRITER_REGISTRY == {  # noqa: SLF001
        "_materialize_public_claims": "publish_authorized_artifact:committed-authorization"
    }
    tree, parents, names = _tree_context(source)
    materializer_callers = Counter(
        _enclosing_definition(node, parents, names)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _attribute_name(node.func) == "_materialize_public_claims"
    )
    assert materializer_callers == Counter({"publish_authorized_artifact": 1})
    public_put_callers = Counter(
        _enclosing_definition(node, parents, names)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _attribute_name(node.func).endswith(".put")
        and any(
            keyword.arg == "mirror_class" and ast.unparse(keyword.value) == "MirrorClass.PUBLIC"
            for keyword in node.keywords
        )
    )
    assert public_put_callers == Counter({"_materialize_public_claims": 1})
    publisher = _function_node(source, "publish_authorized_artifact")
    assert "public/private commitment requires one exact prior authorization event" in source
    publisher_calls = [
        _attribute_name(node.func) for node in ast.walk(publisher) if isinstance(node, ast.Call)
    ]
    assert publisher_calls.index("_authorization_event") < publisher_calls.index(
        "_materialize_public_claims"
    )

    mutated = (
        f"{source}\n\ndef rogue_public_writer(*args):\n    _materialize_public_claims(*args)\n"
    )
    mutated_tree, mutated_parents, mutated_names = _tree_context(mutated)
    mutated_callers = Counter(
        _enclosing_definition(node, mutated_parents, mutated_names)
        for node in ast.walk(mutated_tree)
        if isinstance(node, ast.Call) and _attribute_name(node.func) == "_materialize_public_claims"
    )
    assert mutated_callers != materializer_callers


@pytest.mark.smoke
def test_dead_symbol_inventory_is_closed() -> None:
    """Retired producers, validators, writers, and CLI options must stay absent."""

    observed = {
        (path.relative_to(_CRAWLER_ROOT).as_posix(), occurrence)
        for path in _production_python_paths()
        for occurrence in _dead_contract_occurrences(path.read_text(encoding="utf-8"))
    }
    assert observed == set()

    for symbol in sorted(_DEAD_SYMBOLS):
        assert _dead_contract_occurrences(f"def {symbol}():\n    return None\n") == {symbol}
    for option in sorted(_DEAD_OPTIONS):
        assert _dead_contract_occurrences(f"RETIRED = {option!r}\n") == {option}


@pytest.mark.smoke
def test_protocol_literal_inventory_has_only_reviewed_comparison_owners() -> None:
    """Live wrapper version comparisons must remain in protocol projections."""

    supervisor_source = _source(supervisor_module)
    authority_source = _source(authority_module)
    assert {
        name: getattr(supervisor_module, name) for name in _PROTOCOL_VERSIONS
    } == _PROTOCOL_VERSIONS
    assert _comparison_owner_inventory(supervisor_source) == {
        ("_load_worker_result_value", "_WORKER_RESULT_VERSION"),
        ("_load_worker_result_value", "_WORKER_DIAGNOSTIC_VERSION"),
        ("_load_worker_result_value", "_RAW_AWARD_RECEIPT_VERSION"),
    }
    assert _comparison_owner_inventory(authority_source) == {
        ("_validate_raw_receipt", "_RAW_AWARD_RECEIPT_VERSION")
    }
    for path in _production_python_paths():
        if path.name in {"worker_supervisor.py", "authority.py"}:
            continue
        assert _comparison_owner_inventory(path.read_text(encoding="utf-8")) == set()

    mutated = (
        f"{supervisor_source}\n\ndef rogue_parser(value):\n"
        "    return value == _WORKER_RESULT_VERSION\n"
    )
    assert ("rogue_parser", "_WORKER_RESULT_VERSION") in _comparison_owner_inventory(mutated)


@pytest.mark.smoke
def test_attempt_schema_consumers_and_producers_have_exact_parity() -> None:
    """Attempt-v3 evidence leaves must exist, be produced, consumed, owned, and retained."""

    schema = load_schema(ATTEMPT_SCHEMA_VERSION_V3)
    sources = {
        filename: (_CRAWLER_ROOT / filename).read_text(encoding="utf-8")
        for filename in (
            "authority.py",
            "driver.py",
            "driver_admission.py",
            "driver_models.py",
            "driver_receipts.py",
            "worker.py",
        )
    }
    assert _schema_parity_errors(schema, sources) == ()
    ownership = {leaf.path: leaf.owner for leaf in owned_schema_leaves(ATTEMPT_SCHEMA_VERSION_V3)}
    assert ownership["$.capability_observation.claim"] is SchemaOwner.PARENT_OBSERVED
    assert ownership["$.capability_observation.supported"] is SchemaOwner.PARENT_OBSERVED
    assert (
        ownership["$.raw_award_receipt.observation.output_value_sha256"]
        is SchemaOwner.WORKER_OBSERVED
    )
    assert ownership["$.worker_receipt.output_value_sha256"] is SchemaOwner.WORKER_OBSERVED
    assert {
        "capability_observation",
        "output_value_sha256",
    }.isdisjoint(driver_module._EXTERNALLY_CONTROLLED_ATTEMPT_FIELDS)  # noqa: SLF001

    forbidden_schema = deepcopy(schema)
    del forbidden_schema["properties"]["capability_observation"]
    assert any(
        error.startswith("schema-missing:$.capability_observation")
        for error in _schema_parity_errors(forbidden_schema, sources)
    )
    missing_consumer = dict(sources)
    missing_consumer["authority.py"] = sources["authority.py"].replace(
        'probe.get("capability_observation")', 'probe.get("retired_capability")', 1
    )
    assert (
        "consumer-or-producer-missing:"
        "authority.py:_TerminalProofPipeline._derive_deferral:capability_observation"
        in (_schema_parity_errors(schema, missing_consumer))
    )


@pytest.mark.smoke
def test_synthetic_worker_result_factory_matches_production_protocol_shapes() -> None:
    """The sole synthetic live fixture must round-trip through production's loader shape."""

    attempt = make_attempt()
    raw = deepcopy(attempt["raw_award_receipt"])
    synthetic = make_worker_result_v3_mapping({}, raw_award_receipt=raw)
    loaded, error = supervisor_module._load_worker_result_value(synthetic)  # noqa: SLF001
    assert error is None
    assert loaded == synthetic
    assert set(synthetic) == supervisor_module._WORKER_RESULT_KEYS  # noqa: SLF001
    assert set(synthetic["diagnostic"]) == supervisor_module._WORKER_DIAGNOSTIC_KEYS  # noqa: SLF001
    assert set(synthetic["raw_award_receipt"]) == supervisor_module._RAW_AWARD_RECEIPT_KEYS  # noqa: SLF001

    observation = SupervisorObservation(
        argv=("python",),
        cwd=str(Path.cwd()),
        exit_code=1,
        signal_number=None,
        wall_seconds=0.0,
        cpu_seconds=0.0,
        peak_rss_bytes=0,
        timed_out=False,
        rss_exceeded=False,
        stdout_sha256="sha256:" + "a" * 64,
        stdout_bytes=0,
        stdout_tail="",
        stderr_sha256="sha256:" + "b" * 64,
        stderr_bytes=0,
        stderr_tail="",
        stdout_path="stdout.log",
        stderr_path="stderr.log",
    )
    supervised = make_supervised_worker_result_v3(observation, {})
    assert supervised.worker_receipt is not None
    assert set(supervised.worker_receipt) == set(synthetic)

    constructed: list[tuple[str, str]] = []
    for path in sorted((_CRAWLER_ROOT / "tests").glob("*.py")):
        tree, parents, names = _tree_context(path.read_text(encoding="utf-8"))
        constructed.extend(
            (path.name, _enclosing_definition(node, parents, names))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _attribute_name(node.func) == "SupervisedResult"
        )
    assert constructed == [
        ("conftest.py", "make_supervised_worker_result_v3"),
        # BL-12(f): the failure-evidence census builds parent-side failure
        # SHAPES (never v3 result envelopes) through exactly one audited
        # helper in its own module.
        ("test_failure_evidence_completeness.py", "_classify"),
    ]


def _artifact_cache_writes() -> list[tuple[str, str, ast.expr]]:
    """Return every production write of the artifact-checkpoint reserve cache.

    Returns
    -------
    list[tuple[str, str, ast.expr]]
        Module name, enclosing definition, and assigned value for each write.
    """

    writes: list[tuple[str, str, ast.expr]] = []
    for path in sorted(_CRAWLER_ROOT.glob("*.py")):
        module_tree, module_parents, module_names = _tree_context(path.read_text(encoding="utf-8"))
        for node in ast.walk(module_tree):
            targets: Sequence[ast.expr]
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = (node.target,)
            else:
                continue
            if not any(
                _attribute_name(target) == "self._artifact_checkpoint_cache" for target in targets
            ):
                continue
            assert node.value is not None
            writes.append(
                (path.name, _enclosing_definition(node, module_parents, module_names), node.value)
            )
    return writes


def _assert_artifact_validation_cannot_be_bypassed(
    admission_source: str,
    tree: ast.Module,
    parents: Mapping[ast.AST, ast.AST],
    names: Mapping[ast.AST, str],
) -> None:
    """Assert final artifact authority can only come from a validated projection.

    The projection reserve cache is default-on memoization of an already
    validated checkpoint, so the anti-substitution guarantee is structural
    rather than textual: exactly one function produces final artifact
    authority, exactly one function calls it, every value it can return or
    cache originates from the full checkpoint fold (directly, via the
    incremental fold that shares it, or via a validated advance), and a cache
    hit is admissible only under the exact locked ledger object, the exact
    shard set, and its unchanged append generation. Any bypass -- a second
    producer of final authority, a projection built without validation, a
    wider hit guard, or a cached value of unproven origin -- fails here.

    Parameters
    ----------
    admission_source:
        Source of the admission module owning the authority pipeline.
    tree, parents, names:
        Parsed admission module, parent links, and qualified definition names.
    """

    validator_callers = Counter(
        _enclosing_definition(node, parents, names)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _attribute_name(node.func).endswith("validate_artifact_checkpoint")
    )
    assert validator_callers == Counter(
        {"AdmissionEnvironmentMixin._validated_artifact_projection": 2}
    )
    projection_callers = Counter(
        _enclosing_definition(node, parents, names)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _attribute_name(node.func).endswith("_validated_artifact_projection")
    )
    assert projection_callers == Counter(
        {"AdmissionEnvironmentMixin._rehydrate_final_authority": 1}
    )
    increment_producers = Counter(
        _enclosing_definition(node, parents, names)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _attribute_name(node.func) == "ArtifactCheckpointIncrement.start"
    )
    assert increment_producers == Counter(
        {"AdmissionEnvironmentMixin._validated_artifact_projection": 1}
    )
    advance_calls = Counter(
        _enclosing_definition(node, parents, names)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _attribute_name(node.func).endswith(".advance")
    )
    assert advance_calls == Counter({"AdmissionEnvironmentMixin._validated_artifact_projection": 1})

    # The increment class must be the SAME full validation, not a parallel
    # implementation: start() runs the one whole-store fold shared with
    # validate_artifact_checkpoint, and advance() re-validates every touched
    # transaction through the same per-chain fold.
    artifact_source = _source(artifact_module)
    assert "_full_checkpoint_fold(" in inspect.getsource(
        artifact_module.validate_artifact_checkpoint
    )
    assert "_full_checkpoint_fold(" in inspect.getsource(
        artifact_module.ArtifactCheckpointIncrement.start.__func__
    )
    assert "_fold_transaction_chain(" in inspect.getsource(
        artifact_module.ArtifactCheckpointIncrement.advance
    )
    assert artifact_source.count("def _full_checkpoint_fold(") == 1

    memoized = _function_node(
        admission_source, "AdmissionEnvironmentMixin._validated_artifact_projection"
    )

    # Every value the function can return is proven to originate from a
    # validated producer: assignments to the returned names may come only
    # from the validator itself, the validated incremental fold, or the
    # reserve cache whose fills are themselves pinned below.
    allowed_origins = {
        "projection": {
            (
                "validate_artifact_checkpoint(artifact_paths, context=reducer.context, "
                "mirrors=mirrors, canonical_root=canonical_root, "
                "repository_root=repository_root)"
            ),
            "increment.advance(suffix)",
        },
        "live": {
            (
                "validate_artifact_checkpoint(paths, context=reducer.context, "
                "mirrors=mirrors, canonical_root=canonical_root, "
                "repository_root=repository_root, memo=memo)"
            ),
        },
        "increment": {
            "cached[3]",
            (
                "ArtifactCheckpointIncrement.start(paths, context=reducer.context, "
                "mirrors=mirrors, canonical_root=canonical_root, "
                "repository_root=repository_root, memo=memo)"
            ),
        },
        "cached": {"self._artifact_checkpoint_cache"},
    }
    observed_origins: dict[str, set[str]] = {name: set() for name in allowed_origins}
    for node in ast.walk(memoized):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in observed_origins:
            observed_origins[target.id].add(ast.unparse(node.value))
    assert observed_origins == allowed_origins

    returns = sorted(
        ast.unparse(node) for node in ast.walk(memoized) if isinstance(node, ast.Return)
    )
    assert returns == [
        "return (increment.projection, 'full')",
        "return (increment.projection, 'reserve-hit')",
        "return (live, 'full')",
        "return (projection, 'full')",
        "return (projection, 'incremental')",
    ]

    # Both direct validator calls are escape hatches, admissible only under
    # their explicit opt-out/verify environment guards.
    guard_tests = {
        ast.unparse(node.test) for node in ast.walk(memoized) if isinstance(node, ast.If)
    }
    assert "os.environ.get('MENAGERIE_CRAWLER_ARTIFACT_MEMOIZATION') == '0'" in guard_tests
    assert "os.environ.get('MENAGERIE_CRAWLER_CHECKPOINT_CACHE_VERIFY') == '1'" in guard_tests

    # A cache hit is admissible only under the exact locked ledger object,
    # the exact shard set, and its unchanged append generation; the
    # incremental fold additionally requires the single self-appending shard.
    assert (
        "cached is not None and cached[0] is reducer.artifact_ledger and (cached[2] == paths)"
        in guard_tests
    )
    assert "cached_generation == generation" in guard_tests
    assert (
        "generation > cached_generation and len(paths) == 1 and "
        "(ledger_path is not None) and (Path(ledger_path) == paths[0])" in guard_tests
    )

    writes = _artifact_cache_writes()
    assert Counter(owner for _module, owner, _value in writes) == Counter(
        {
            "CrawlerDriver.__init__": 1,
            "AdmissionEnvironmentMixin._validated_artifact_projection": 4,
        }
    )
    fill_values = []
    for _module, owner, value in writes:
        if owner == "CrawlerDriver.__init__":
            assert isinstance(value, ast.Constant) and value.value is None
            continue
        if isinstance(value, ast.Constant):
            assert value.value is None  # Invalidation, never a substitute fill.
            continue
        fill_values.append(ast.unparse(value))
    # Exactly two fills, both binding the locked ledger, its generation, the
    # shard set, and an increment whose only origins are pinned above.
    assert fill_values == [
        "(reducer.artifact_ledger, generation, paths, increment)",
        "(reducer.artifact_ledger, generation, paths, increment)",
    ]


@pytest.mark.smoke
def test_executable_and_artifact_authority_have_one_final_projection() -> None:
    """Exact members and one normalized final transaction must feed all consumers."""

    read_projection = inspect.getsource(supervisor_module._request_allowed_read_paths)  # noqa: SLF001
    assert "exact_read_capability(manifest).member_paths" in read_projection
    assert "allowed_read_paths" in inspect.getsource(supervisor_module.run_isolated_subprocess)
    assert "runtime_read_roots" not in inspect.getsource(
        authority_module.compile_execution_read_manifest_v2
    )

    admission_source = _source(driver_admission_module)
    tree, parents, names = _tree_context(admission_source)
    resolver_callers = Counter(
        _enclosing_definition(node, parents, names)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _attribute_name(node.func).endswith("resolve_final_artifact_transaction")
    )
    assert resolver_callers == Counter({"AdmissionEnvironmentMixin._rehydrate_final_authority": 1})
    reconstruction = inspect.getsource(driver_module.CrawlerDriver._rehydrate_final_authority)
    assert (
        reconstruction.index("_validated_artifact_projection(")
        < reconstruction.index("resolve_final_artifact_transaction(")
        < reconstruction.index("rehydrate_artifact_transaction(")
    )
    _assert_artifact_validation_cannot_be_bypassed(admission_source, tree, parents, names)
    resolver = _function_node(_source(artifact_module), "resolve_final_artifact_transaction")
    assert not any(
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.UnaryOp)
        and isinstance(node.slice.op, ast.USub)
        for node in ast.walk(resolver)
    )


def _closure_symbol_definition(tree: ast.Module, symbol: str) -> ast.AST | None:
    """Return the module-level or one-level-nested definition node for a symbol.

    Parameters
    ----------
    tree:
        Parsed production module.
    symbol:
        Award-closure root, either ``name`` or ``Owner.name``.

    Returns
    -------
    ast.AST | None
        The definition node, or ``None`` when the module does not define it.
    """

    if "." in symbol:
        owner, child = symbol.split(".", 1)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == owner:
                for nested in node.body:
                    if (
                        isinstance(nested, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and nested.name == child
                    ):
                        return nested
        return None
    for node in tree.body:
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == symbol
        ):
            return node
    return None


def _unreferenced_closure_symbols() -> tuple[str, ...]:
    """Return award-closure roots that no production code outside them references.

    Returns
    -------
    tuple[str, ...]
        Sorted ``relative:symbol`` components with zero production references.
    """

    trees = {
        path: ast.parse(path.read_text(encoding="utf-8")) for path in _production_python_paths()
    }
    unreferenced = []
    for relative, symbols in driver_module._AWARD_CLOSURE_SYMBOLS.items():  # noqa: SLF001
        defining = _CRAWLER_ROOT / relative
        for symbol in symbols:
            leaf = symbol.rsplit(".", 1)[-1]
            definition = _closure_symbol_definition(trees[defining], symbol)
            own = set() if definition is None else {id(node) for node in ast.walk(definition)}
            references = sum(
                1
                for tree in trees.values()
                for node in ast.walk(tree)
                if id(node) not in own
                and (
                    (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load))
                    or isinstance(node, ast.Attribute)
                )
                and (node.id if isinstance(node, ast.Name) else node.attr) == leaf
            )
            if references == 0:
                unreferenced.append(f"{relative}:{symbol}")
    return tuple(sorted(unreferenced))


def test_award_identity_closure_names_live_symbols_and_no_dead_symbols() -> None:
    """Award closure roots must resolve, stay reachable, and retain no dead authority.

    Existence is NOT the property that matters. A root nothing calls contributes
    its own hash and its callees' hashes to the award closure while reviewing
    zero live behavior, and the register then asserts a review that is not
    happening. `_validate_terminal_evidence` was retired correctly into
    `_DEAD_SYMBOLS`; the roots this catches were left listed as live instead.
    Reachability is what stops the next dead root landing the same way.
    """

    for relative, symbols in driver_module._AWARD_CLOSURE_SYMBOLS.items():  # noqa: SLF001
        path = _CRAWLER_ROOT / relative
        for symbol in symbols:
            assert driver_module._source_symbol_bytes(path, symbol)  # noqa: SLF001
            assert symbol not in _DEAD_SYMBOLS
    assert _unreferenced_closure_symbols() == ()


@pytest.mark.smoke
def test_real_composition_sources_are_not_fake_substitutes() -> None:
    """Permanent adjudication probes must retain their production composition seams."""

    import menagerie.crawler.tests.test_award_worker_result_composition as vs1_module
    import menagerie.crawler.tests.test_boundary_shutdown_composition as vs2_module
    import menagerie.crawler.tests.test_execution_dry_run_composition as vs6_module
    import menagerie.crawler.tests.test_slice_f_driver as driver_tests

    golden = inspect.getsource(
        vs1_module.test_real_v3_worker_result_awards_through_driver_and_reducer
    )
    shutdown = inspect.getsource(
        vs2_module.test_signal_after_real_forward_publishes_and_awards_nothing_then_resumes
    )
    handoff = inspect.getsource(
        driver_tests.test_linux_handoff_attempts_both_deferred_statuses_and_supersedes
    )
    dry_run = inspect.getsource(vs6_module.test_documented_dry_run_and_resume_use_real_environment)
    dry_run_support = (_CRAWLER_ROOT / "tests" / "dry_run_support.py").read_text(encoding="utf-8")
    module_source = _source(vs1_module)
    shutdown_module_source = _source(vs2_module)
    assert "SupervisedForwardLane" in module_source
    assert "CanonicalReducer" in module_source
    assert "runs" in _string_constants(ast.parse(module_source))
    assert "FakeForward" not in golden
    assert "SupervisedResult(" not in golden
    assert "multiprocessing.get_context" in shutdown
    assert "RealEnvironmentLane" in shutdown_module_source
    assert 'observation["models"] == 0' in shutdown
    assert "DisabledAuthor" in handoff
    assert "SupervisedForwardLane" in handoff
    assert "RealEnvironmentLane" in handoff
    assert "FakeEnvironments" not in handoff
    assert 'record["status"]["code"] for record in superseding' in handoff
    assert "_dry_run_command" in dry_run
    assert "scan_jsonl" in dry_run
    assert "MaterializedDryRunEnvironment" in dry_run_support
    assert "SupervisedForwardLane" in dry_run_support
    assert "FakeEnvironments" not in dry_run_support


@pytest.mark.smoke
def test_environment_cache_has_one_lifecycle_owner_and_no_default_collector() -> None:
    """Live scheduling and currentness must share one lifecycle-owned cache."""

    lifecycle_source = _source(lifecycle_module)
    driver_source = _source(driver_module)
    binder_signature = inspect.signature(driver_module.bind_materialized_environment)
    assert lifecycle_source.count("EnvironmentAuthorityCache()") == 1
    assert "EnvironmentAuthorityCache()" not in driver_source
    assert binder_signature.parameters["authority_cache"].default is inspect.Signature.empty
    assert "active_authority_cache(" in inspect.getsource(
        driver_module.CrawlerDriver._run_environment_work
    )
    assert "cache.verify(authority, verification_token=verification_token)" in inspect.getsource(
        driver_module._current_run_is_fresh
    )


@pytest.mark.smoke
def test_real_compositions_cannot_substitute_execution_boundaries() -> None:
    """Transitive AST tripwire covers fixtures, compositions, helpers, and CI."""

    assert _COMPOSITION_SOURCES == _composition_source_paths()
    assert _CRAWLER_ROOT / "tests" / "conftest.py" in _COMPOSITION_SOURCES
    assert (
        _CRAWLER_ROOT / "tests" / "test_compiler_os_authority_composition.py"
        in _COMPOSITION_SOURCES
    )
    assert _composition_scope_errors() == ()

    mutated = (
        _COMPOSITION_SOURCES[0].read_text(encoding="utf-8")
        + "\n"
        + "def test_reintroduced_patch(monkeypatch):\n"
        + "    monkeypatch.setattr(driver_module, '_compile_worker_read_manifest', object())\n"
    )
    assert _substitution_boundary_errors(
        mutated,
        source_path=Path("tests/test_reintroduced_composition.py"),
        root_definitions=("test_reintroduced_patch",),
    ) == (
        "tests/test_reintroduced_composition.py:test_reintroduced_patch:"
        "_compile_worker_read_manifest:direct-patch",
    )


@pytest.mark.smoke
def test_legacy_manifest_v1_is_quarantined_from_every_live_import_graph() -> None:
    """Legacy root-grant parsing exists only in the audit module and cannot spawn."""

    production_paths = _production_python_paths()
    assert _QUARANTINE_PATH in production_paths
    for path in production_paths:
        source = path.read_text(encoding="utf-8")
        if path == _QUARANTINE_PATH:
            assert "class ExecutionReadManifest" in source
            assert "def compile_execution_read_manifest" in source
            assert "def audit_runtime_root_grants" in source
            assert "runtime-root" in source
            assert "supervise_worker" not in source
            assert "run_isolated_subprocess" not in source
            continue
        assert "legacy_manifest_audit" not in source, path
        assert "runtime-root" not in source, path
        assert re.search(r"^class ExecutionReadManifest(?:\(|:)", source, re.MULTILINE) is None, (
            path
        )
        assert "def compile_execution_read_manifest(" not in source, path

    supervisor_source = _source(supervisor_module)
    assert "live v3 model worker spawn requires execution-read-manifest.v3" in supervisor_source
    assert "ExecutionReadManifest |" not in supervisor_source
    assert 'runtime_support if kind == "runtime-root"' not in supervisor_source


@pytest.mark.smoke
@_NO_REPO_CI
def test_real_compositions_are_explicitly_selected_in_ci() -> None:
    """The always-on Linux CI lane must select the exact release marker."""

    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "name: Crawler Round 21 Linux committed-lock release proofs" in workflow
    crawler_job = workflow.split("crawler-round21-linux-release:", 1)[1]
    crawler_job = crawler_job.split("\n  crawler-round21-macos-release:", 1)[0]
    # Collection roots are DERIVED from the committed registry (the frozen lock
    # prefix cannot import the whole corpus — PyYAML postdates the lock), and
    # the run still selects the exact release marker over those roots.
    assert "round21_registry_paths" in crawler_job
    assert "round21_linux_real_nodes.json" in crawler_job
    assert "$REGISTRY_MODULES -m round21_linux_real" in crawler_job
    assert 'MENAGERIE_RELEASE_GATE: "1"' in crawler_job
    assert "MENAGERIE_RELEASE_ATTESTATION" in crawler_job
    assert "--ignore" not in crawler_job
    assert "not heavy" not in crawler_job
    assert "not slow" not in crawler_job


@pytest.mark.smoke
def test_supported_host_release_gate_inventory_is_exact() -> None:
    """Both mandatory host jobs select the closed permanent proof inventory."""

    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    assert set(_ROUND19_RELEASE_NODE_INVENTORY) == {
        "golden",
        "interpreter",
        "linux-denial",
        "shutdown",
        "clean-clone",
        "unverifiable",
        "dry-run-run-resume",
        "dry-run-false-complete",
        "cache",
        "structural",
        "macos-positive-negative",
        "macos-profile",
    }
    assert "crawler-round21-linux-release:" in workflow
    assert "crawler-round21-macos-release:" in workflow
    release_jobs = workflow.split("crawler-round21-linux-release:", 1)[1]
    linux_job, macos_job = release_jobs.split("\n  crawler-round21-macos-release:", 1)
    assert "-m round21_linux_real" in linux_job
    assert "-m round21_macos_real" in macos_job
    assert "runs-on: macos-14-xlarge" in macos_job
    assert "MENAGERIE_RELEASE_ATTESTATION" in macos_job
    assert "Require Linux lock, provenance, conda, bwrap, and strace" in linux_job
    assert "sudo apt-get install -y bubblewrap strace" in linux_job
    assert "command -v bwrap" in linux_job
    assert "command -v strace" in linux_job
    assert "Linux bwrap/strace enforcement is unavailable" in linux_job
    assert "Require macOS lock, provenance, conda, and Seatbelt denial audit" in macos_job
    assert "command -v sandbox-exec" in macos_job
    assert "command -v log" in macos_job
    assert "seatbelt-denial-audit-unavailable" in macos_job
    for job in (linux_job, macos_job):
        assert 'MENAGERIE_RELEASE_GATE: "1"' in job
        assert "unmet-release-gate" in job
        assert "menagerie.crawler.tools.release_lock" in job
        assert "pytest.skip" not in job


@pytest.mark.smoke
def test_linux_release_artifacts_and_provisioning_are_real() -> None:
    """T04 requires every workflow-named Linux lock-family artifact to exist and parse."""

    lock_path = _CRAWLER_ROOT / "envs" / "locks" / "round19-linux-64.lock"
    family = {
        "lock": lock_path,
        "export": lock_path.with_suffix(".resolved.json"),
        "export-hash": lock_path.with_suffix(".resolved.sha256"),
        "provenance": lock_path.with_suffix(".provenance.json"),
        "probes": lock_path.with_suffix(".probes.json"),
    }
    assert {name for name, path in family.items() if path.is_file()} == set(family)
    lock_bytes = family["lock"].read_bytes()
    export_bytes = family["export"].read_bytes()
    assert lifecycle_module.parse_exact_lock(lock_bytes)
    assert lifecycle_module.parse_resolved_export(export_bytes) == export_bytes
    assert family["export-hash"].read_text(encoding="utf-8").strip() == hash_bytes(export_bytes)
    provenance = json.loads(family["provenance"].read_bytes())
    assert provenance["target"] == "linux-64"
    assert provenance["lock_sha256"] == hash_bytes(lock_bytes)
    assert provenance["resolved_export_sha256"] == hash_bytes(export_bytes)
    assert provenance["clean_create"]["validated"] is True

    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    linux_job = workflow.split("crawler-round21-linux-release:", 1)[1].split(
        "\n  crawler-round21-macos-release:", 1
    )[0]
    assert family["lock"].relative_to(_REPOSITORY_ROOT).as_posix() in linux_job
    for suffix in (".resolved.json", ".resolved.sha256", ".provenance.json", ".probes.json"):
        assert suffix in linux_job
    assert "menagerie.crawler.tools.release_lock" in linux_job
    assert '"$MENAGERIE_REAL_ENV_PREFIX/bin/python" -m pytest' in linux_job
    assert "if-no-files-found: error" in linux_job


@pytest.mark.smoke
def test_macos_release_artifacts_and_provisioning_are_real() -> None:
    """T04 requires every workflow-named macOS lock-family artifact to exist and parse."""

    lock_path = _CRAWLER_ROOT / "envs" / "locks" / "round19-osx-arm64.lock"
    family = {
        "lock": lock_path,
        "export": lock_path.with_suffix(".resolved.json"),
        "export-hash": lock_path.with_suffix(".resolved.sha256"),
        "provenance": lock_path.with_suffix(".provenance.json"),
        "probes": lock_path.with_suffix(".probes.json"),
        "virtual-packages": _CRAWLER_ROOT
        / "envs"
        / "specs"
        / "round21-release.virtual-packages.yml",
    }
    assert {name for name, path in family.items() if path.is_file()} == set(family)
    lock_bytes = family["lock"].read_bytes()
    export_bytes = family["export"].read_bytes()
    assert lifecycle_module.parse_exact_lock(lock_bytes)
    assert lifecycle_module.parse_resolved_export(export_bytes) == export_bytes
    assert family["export-hash"].read_text(encoding="utf-8").strip() == hash_bytes(export_bytes)
    provenance = json.loads(family["provenance"].read_bytes())
    assert provenance["target"] == "osx-arm64"
    assert provenance["lock_sha256"] == hash_bytes(lock_bytes)
    assert provenance["resolved_export_sha256"] == hash_bytes(export_bytes)
    assert provenance["virtual_package_spec_sha256"] == hash_bytes(
        family["virtual-packages"].read_bytes()
    )
    assert provenance["probe_receipt_sha256"] == hash_bytes(family["probes"].read_bytes())
    assert provenance["solver"]["name"] == "conda-lock"
    assert "osx-arm64" in provenance["solver"]["command"]
    assert provenance["probe_observation"] == {
        "committed_on_linux": False,
        "producer": "hosted macOS release CI attestation",
    }
    assert provenance["clean_create"]["validation_host"] == "hosted macOS release CI"

    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    macos_job = workflow.split("crawler-round21-macos-release:", 1)[1]
    assert family["lock"].relative_to(_REPOSITORY_ROOT).as_posix() in macos_job
    for suffix in (".resolved.json", ".resolved.sha256", ".provenance.json", ".probes.json"):
        assert suffix in macos_job
    assert "menagerie.crawler.tools.release_lock" in macos_job
    assert '"$MENAGERIE_REAL_ENV_PREFIX/bin/python" -m pytest' in macos_job
    assert "if-no-files-found: error" in macos_job


@pytest.mark.smoke
def test_linux_release_registry_is_exact() -> None:
    """T05 requires the committed marker registry to name expanded existing nodes."""

    registry_path = _CRAWLER_ROOT / "tests" / "round21_linux_real_nodes.json"
    payload = json.loads(registry_path.read_bytes())
    nodes = payload["nodes"]
    assert payload["target"] == "linux-64"
    assert len(nodes) == 47
    assert len(nodes) == len(set(nodes))
    assert set(ROUND21_VS9_PROOF_REGISTRY) == {
        "P01",
        "T01",
        "T01-CI",
        "T02",
        "P02",
        "P03",
        "T03",
        "P04",
        "P12",
        "P13",
        "P14",
        "P17",
        "P19",
        "P05",
        "P06",
        "P07",
        "P08",
        "P09",
        "T04",
        "T05",
    }
    assert ROUND21_VS9_PROOF_REGISTRY["P09"] in nodes
    assert ROUND21_VS11_PROOF_REGISTRY["P11"] in nodes
    assert all(node.startswith("menagerie/crawler/tests/") and "::test_" in node for node in nodes)


@pytest.mark.smoke
def test_macos_release_registry_is_exact() -> None:
    """T05 requires the committed macOS marker registry to name expanded existing nodes."""

    registry_path = _CRAWLER_ROOT / "tests" / "round21_macos_real_nodes.json"
    payload = json.loads(registry_path.read_bytes())
    nodes = payload["nodes"]
    assert payload["target"] == "osx-arm64"
    # 5 environment/CI proofs + the two dry-run rehearsal nodes (session
    # death; usage-pause/wake) promoted so the 200a launch-night class-killers
    # run in a blocking release gate instead of only the dark full tier, plus
    # WS5's routed-env lean-bootstrap import proof (the py3.10 StrEnum class).
    assert len(nodes) == 8
    assert (
        "menagerie/crawler/tests/test_session_death_dry_run_rehearsal.py::"
        "test_injected_session_death_settles_and_retries_on_a_fresh_reservation" in nodes
    )
    assert (
        "menagerie/crawler/tests/test_usage_pause_wake_dry_run_rehearsal.py::"
        "test_usage_pause_wake_cycle_survives_the_real_subprocess_dry_run" in nodes
    )
    assert (
        "menagerie/crawler/tests/test_worker_lean_bootstrap.py::"
        "test_routed_env_interpreter_imports_worker_closure" in nodes
    )
    assert len(nodes) == len(set(nodes))
    assert set(ROUND21_VS10_PROOF_REGISTRY) == {
        "P01",
        "T01",
        "T01-CI",
        "T02",
        "P02",
        "P03",
        "T03",
        "P04",
        "P12",
        "P13",
        "P14",
        "P17",
        "P19",
        "P05",
        "P06",
        "P07",
        "P08",
        "P09",
        "T04",
        "T05",
        "P10",
        "T04-mac",
        "T05-mac",
    }
    assert ROUND21_VS10_PROOF_REGISTRY["P10"] in nodes
    assert _ROUND19_RELEASE_NODE_INVENTORY["macos-positive-negative"] in nodes
    assert _ROUND19_RELEASE_NODE_INVENTORY["macos-profile"] in nodes
    assert all(node.startswith("menagerie/crawler/tests/") and "::test_" in node for node in nodes)


@pytest.mark.smoke
def test_conformance_workflow_and_reversion_inventory_is_exact() -> None:
    """VS11 requires final fan-in CI and an exact deliberate-reversion matrix."""

    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "crawler-round21-deliberate-reversions:" in workflow
    assert "crawler-round21-conformance:" in workflow
    assert "round21-linux-release-attestation" in workflow
    assert "round21-macos-release-attestation" in workflow
    assert "round21-reversion-attestation" in workflow
    assert "test_ci_attestations_cover_registry_without_skip" in workflow
    assert "continue-on-error: true" not in workflow
    assert "|| true" not in workflow
    assert "if-no-files-found: error" in workflow
    conformance_job = workflow.split("crawler-round21-conformance:", 1)[1]
    assert 'MENAGERIE_RELEASE_GATE: "1"' in conformance_job
    assert "MENAGERIE_LINUX_RELEASE_ATTESTATION" in conformance_job
    assert "MENAGERIE_MACOS_RELEASE_ATTESTATION" in conformance_job
    assert "MENAGERIE_REVERSION_ATTESTATION" in conformance_job
    reversion_job = workflow.split("crawler-round21-deliberate-reversions:", 1)[1].split(
        "\n  crawler-round21-conformance:", 1
    )[0]
    assert 'MENAGERIE_RELEASE_GATE: "1"' in reversion_job
    assert "MENAGERIE_PLATFORM_LOCK" in reversion_job
    assert "MENAGERIE_REAL_ENV_PREFIX" in reversion_job
    assert "menagerie.crawler.tools.release_lock" in reversion_job
    assert "sudo apt-get install -y bubblewrap strace" in reversion_job

    matrix_path = _CRAWLER_ROOT / "tools" / "round21_reversions.json"
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "menagerie.crawler.round21-reversion-matrix.v1"
    assert payload["deliberate_reversion_ids"] == [f"D{index:02d}" for index in range(1, 30)]
    assert [case["reversion_id"] for case in payload["cases"]] == payload[
        "deliberate_reversion_ids"
    ]
    cases_by_id = {case["reversion_id"]: case for case in payload["cases"]}
    assert all("proof_nodes" in case and "proof_node" not in case for case in cases_by_id.values())
    assert cases_by_id["D02"]["proof_nodes"] == [
        "menagerie/crawler/tests/test_release_environment_matrix_composition.py::"
        "test_environment_authority_unit_matrix[E02]",
        "menagerie/crawler/tests/test_release_environment_matrix_composition.py::"
        "test_environment_authority_unit_matrix[E03]",
    ]
    assert cases_by_id["D17"]["proof_nodes"] == [
        "menagerie/crawler/tests/test_release_transport_composition.py::"
        "test_closed_transport_capability_awards_and_rejects_unlisted_library"
    ]
    assert cases_by_id["D18"]["proof_nodes"] == [
        "menagerie/crawler/tests/test_release_handoff_authority_composition.py::"
        "test_handoff_authority_identity_matrix[H02]"
    ]
    assert cases_by_id["D19"]["proof_nodes"] == [
        "menagerie/crawler/tests/test_release_handoff_authority_composition.py::"
        "test_handoff_authority_identity_matrix[H03]",
        "menagerie/crawler/tests/test_release_handoff_authority_composition.py::"
        "test_handoff_authority_identity_matrix[H04]",
    ]
    assert cases_by_id["D07"]["proof_nodes"] == [
        "menagerie/crawler/tests/test_release_preclusion_composition.py::"
        "test_preclusion_real_v3_path_has_no_substitutable_fixture_edge",
        "menagerie/crawler/tests/test_release_ci_composition.py::"
        "test_linux_committed_lock_provenance_awards_in_ci",
    ]
    assert cases_by_id["D21"]["proof_nodes"] == [
        "menagerie/crawler/tests/test_anti_substitution_inventories.py::"
        "test_linux_release_artifacts_and_provisioning_are_real",
        "menagerie/crawler/tests/test_release_ci_composition.py::"
        "test_linux_committed_lock_provenance_awards_in_ci",
    ]
    assert cases_by_id["D22"]["proof_nodes"] == [
        "menagerie/crawler/tests/test_anti_substitution_inventories.py::"
        "test_conformance_workflow_and_reversion_inventory_is_exact"
    ]
    assert cases_by_id["D23"]["proof_nodes"] == [
        "menagerie/crawler/tests/test_anti_substitution_inventories.py::"
        "test_supported_host_release_gate_inventory_is_exact"
    ]
    assert cases_by_id["D24"]["proof_nodes"] == [
        "menagerie/crawler/tests/test_anti_substitution_inventories.py::"
        "test_supported_host_release_gate_inventory_is_exact"
    ]
    assert cases_by_id["D27"]["proof_nodes"] == [
        "menagerie/crawler/tests/test_slice_f_driver.py::"
        "test_linux_code_less_deferral_fails_visibly_without_failed_source"
    ]
    assert cases_by_id["D28"]["proof_nodes"] == [
        "menagerie/crawler/tests/test_release_conformance_composition.py::"
        "test_conformance_registry_is_total_and_executed"
    ]
    assert all(
        "test_ci_attestations_cover_registry_without_skip" not in "\n".join(case["proof_nodes"])
        for case in cases_by_id.values()
    )
    assert all(
        "test_conformance_registry_is_total_and_executed" not in "\n".join(case["proof_nodes"])
        for reversion_id, case in cases_by_id.items()
        if reversion_id != "D28"
    )
    assert set(ROUND21_VS11_PROOF_REGISTRY) == {
        "P01",
        "T01",
        "T01-CI",
        "T02",
        "P02",
        "P03",
        "T03",
        "P04",
        "P12",
        "P13",
        "P14",
        "P17",
        "P19",
        "P05",
        "P06",
        "P07",
        "P08",
        "P09",
        "T04",
        "T05",
        "P10",
        "T04-mac",
        "T05-mac",
        "P11",
        "P11-CI",
        "P27",
        "T06",
    }


@pytest.mark.parametrize(
    ("manifest", "default_path"),
    (
        (
            VS1_LANDING_MANIFEST,
            _CRAWLER_ROOT / "tests" / "test_award_worker_result_composition.py",
        ),
        (
            VS2_LANDING_MANIFEST,
            _CRAWLER_ROOT / "tests" / "test_boundary_shutdown_composition.py",
        ),
        (
            VS3_LANDING_MANIFEST,
            _CRAWLER_ROOT / "tests" / "test_compiler_os_authority_composition.py",
        ),
        (VS4_LANDING_MANIFEST, Path(__file__)),
    ),
)
@pytest.mark.smoke
def test_slice_landing_manifests_name_existing_seams_and_nodes(
    manifest: Mapping[str, Any],
    default_path: Path,
) -> None:
    """Each prevention landing manifest must resolve production symbols and tests."""

    modules = {
        "driver": driver_module,
        "worker_supervisor": supervisor_module,
        "artifact_transactions": artifact_module,
        "authority": authority_module,
    }
    for module_name, symbols in manifest["production_symbols"].items():
        for symbol in symbols:
            _assert_symbol(modules[module_name], symbol)
    assert manifest["real_composition_nodes"]
    assert manifest["structural_nodes"]
    for node_id in (*manifest["real_composition_nodes"], *manifest["structural_nodes"]):
        _assert_test_node_exists(str(node_id), default_path)


_GATE_MANIFEST = gate_closure.load_manifest(_REPOSITORY_ROOT)


def _git_fixture_repo(root: Path) -> None:
    """Initialise one committed git repository at ``root`` (fixture helper)."""

    import subprocess

    root.mkdir(parents=True, exist_ok=True)
    for arguments in (
        ("init", "-q", "-b", "main"),
        ("config", "user.email", "t@example.invalid"),
        ("config", "user.name", "T"),
        ("config", "commit.gpgsign", "false"),
    ):
        subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True)


def _git_commit_all(root: Path, message: str) -> None:
    import subprocess

    subprocess.run(("git", "add", "-A"), cwd=root, check=True, capture_output=True)
    subprocess.run(
        ("git", "commit", "-q", "-m", message), cwd=root, check=True, capture_output=True
    )


@pytest.mark.smoke
def test_collection_gate_fails_on_conflict_marker_and_empty_module(tmp_path: Path) -> None:
    """The floor's one real catch, reproduced against the replacement gate.

    A merge-conflict marker left in a test module makes it unparseable; the
    retired exact-inventory floor caught that as 70 missing nodes. The
    collection gate catches the same class as a collection ERROR (pytest exit
    2), names the module, and separately refuses a tracked test module that
    collects nothing. A clean corpus passes with per-module counts reported.
    """

    import sys

    repo = tmp_path / "repo"
    _git_fixture_repo(repo)
    tests = repo / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("", encoding="utf-8")
    (tests / "test_alpha.py").write_text(
        "def test_one():\n    pass\n\n\ndef test_two():\n    pass\n", encoding="utf-8"
    )
    _git_commit_all(repo, "clean corpus")
    clean = collection_gate.collect(repo, python=sys.executable, roots=("tests",))
    assert clean.ok, clean.failures
    assert clean.per_module == {"tests/test_alpha.py": 2}
    assert "collection-gate: OK" in collection_gate.render_report(clean)

    # Conflict marker: a SyntaxError at collection -> exit 2 -> gate FAILED.
    (tests / "test_beta.py").write_text(
        "<<<<<<< HEAD\ndef test_left():\n    pass\n=======\ndef test_right():\n    pass\n"
        ">>>>>>> theirs\n",
        encoding="utf-8",
    )
    _git_commit_all(repo, "conflict marker")
    broken = collection_gate.collect(repo, python=sys.executable, roots=("tests",))
    assert not broken.ok
    assert broken.returncode != 0
    assert any("collection exited" in failure for failure in broken.failures)
    assert "tests/test_beta.py" in collection_gate.render_report(broken)

    # Emptied module: tracked test_*.py that collects zero tests -> named failure.
    (tests / "test_beta.py").write_text("# emptied\n", encoding="utf-8")
    _git_commit_all(repo, "emptied module")
    emptied = collection_gate.collect(repo, python=sys.executable, roots=("tests",))
    assert not emptied.ok
    assert emptied.empty_modules == ("tests/test_beta.py",)
    assert emptied.failures == ("module collected zero tests: tests/test_beta.py",)


@pytest.mark.smoke
def test_gate_manifest_commands_are_real_executions_over_both_roots() -> None:
    """The ONE command authority selects the whole corpus and really executes it.

    The retired floor pinned the workflow text; the gate commands now live in
    ``required_gate_set.json`` and every CI job runs them through the gate
    runner, so THIS is the surface a narrowing or a collection-only
    substitution would have to pass through.
    """

    smoke = _GATE_MANIFEST.gate("crawler-smoke")
    full = _GATE_MANIFEST.gate("crawler-full")
    for gate in (smoke, full):
        command = list(gate.command or ())
        assert "menagerie/crawler/tests" in command and "tests/crawler" in command, gate.name
        assert "--collect-only" not in command, gate.name
        assert "-k" not in command and not any(token.startswith("-k") for token in command), (
            gate.name
        )
        assert not any(token.startswith("--ignore") for token in command), gate.name
        assert "{junit}" in " ".join(command), gate.name
    assert smoke.command is not None and _marker_selection(smoke.command) == "smoke and not rig"
    assert full.command is not None and _marker_selection(full.command) == "not rig"
    assert smoke.env == {"MENAGERIE_SMOKE_EXECUTION_FLOOR": "1"}
    assert smoke.requires_armed_floor is True
    rigs = _GATE_MANIFEST.gate("crawler-rigs")
    assert rigs.command is not None and _marker_selection(rigs.command) == "rig"


@pytest.mark.smoke
@_NO_REPO_CI
def test_every_command_gate_has_a_workflow_invocation() -> None:
    """Every command-run gate is invoked by some workflow through the runner.

    The attestation gate is recorded from the release conftest instead.
    """

    workflows = "\n".join(
        _workflow_text(path.name) for path in sorted(_WORKFLOWS_DIR.glob("*.yml"))
    )
    for name, gate in _GATE_MANIFEST.gates.items():
        if gate.is_attestation:
            continue
        assert f"--gate {name}" in workflows, f"gate {name} has no workflow invocation"


@pytest.mark.smoke
@_NO_REPO_CI
def test_crawler_smoke_ci_job_is_mandatory_and_fail_closed() -> None:
    """The crawler-smoke CI job must run the whole corpus smoke tier, ungated.

    T5-7 found that no mandatory workflow ran any crawler test: the smoke matrix
    ignored tests/crawler and every crawler job hid behind an opt-in repository
    variable, so the corpus's own gates never ran where they could block a
    merge. This pin refuses that recurrence in every direction: job deleted, job
    gated, selection narrowed, execution substituted with collection, or the
    inventory floor step dropped.
    """

    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "\n  crawler-smoke:" in workflow, "mandatory crawler-smoke job is missing"
    job = workflow.split("\n  crawler-smoke:", 1)[1]
    job = job.split("\n  crawler-round21-linux-release:", 1)[0]
    # YAML comments are documentation, not configuration: strip them so a comment
    # mentioning a gating device neither trips nor satisfies these assertions.
    job = "\n".join(line for line in job.splitlines() if not line.lstrip().startswith("#"))

    # Ungated and fail-closed: no event/vars condition, no tolerated failure.
    assert "\n    if:" not in job
    assert "continue-on-error" not in job
    assert "MENAGERIE_RELEASE_RUNNERS" not in job

    # Both gates run through the ONE runner over the ONE command authority:
    # the unconditional collection gate (no --base: it can never be skipped)
    # and the smoke gate. Selection and execution-floor arming are asserted
    # on the manifest in test_gate_manifest_commands_are_real_executions_over_both_roots.
    assert "menagerie.crawler.tools.gate_runner run" in job
    collection_step = job.split("--gate collection", 1)
    assert len(collection_step) == 2, "collection gate is not invoked"
    assert "--base" not in collection_step[1].split("\n", 1)[0]
    assert "--gate crawler-smoke" in job
    assert "set -o pipefail" in job
    assert "--collect-only" not in job
    assert "-k " not in job
    # The verdict ledger and its FULL evidence blobs leave the runner as an
    # artifact (if-no-files-found: error), so a green here is importable by
    # the campaign host's preflight and a lost ledger is a red job.
    assert "gate-verdicts.jsonl" in job and "gate-logs/" in job
    assert "if-no-files-found: error" in job

    # The gate must be reachable: tests.yml triggers on pull_request directly and
    # release pushes call it via workflow_call, so an unconditioned job section
    # inside it runs on both.
    assert "pull_request:" in workflow
    assert "workflow_call:" in workflow

    # Mutation self-checks: each gating device must flip the assertions above.
    gated = job.replace("runs-on: macos-14", "runs-on: macos-14\n    if: ${{ vars.X }}")
    assert "\n    if:" in gated
    tolerated = job.replace("runs-on: macos-14", "runs-on: macos-14\n    continue-on-error: true")
    assert "continue-on-error" in tolerated
    skipped_collection = job.replace("--gate collection", '--gate collection --base "$GATE_BASE"')
    assert "--base" in skipped_collection.split("--gate collection", 1)[1].split("\n", 1)[0]
    dropped_ledger = job.replace("if-no-files-found: error", "if-no-files-found: ignore")
    assert "if-no-files-found: error" not in dropped_ledger


_WORKFLOWS_DIR = _REPOSITORY_ROOT / ".github" / "workflows"
_CI_DORMANCY_PATH = _REPOSITORY_ROOT / ".github" / "ci_dormancy.json"


def _workflow_text(name: str) -> str:
    return (_WORKFLOWS_DIR / name).read_text(encoding="utf-8")


def _marker_selection(command: Sequence[str]) -> str:
    """Return the ``-m`` marker expression of a ``{python} -m pytest ... -m EXPR`` command.

    The interpreter's own ``-m pytest`` comes first; the selection is the ``-m``
    that follows the ``pytest`` token.
    """

    tokens = list(command)
    pytest_index = tokens.index("pytest")
    return tokens[tokens.index("-m", pytest_index + 1) + 1]


def _workflow_job_names(text: str) -> list[str]:
    jobs_section = text.split("\njobs:", 1)[1]
    return re.findall(r"^  ([A-Za-z0-9_-]+):\s*$", jobs_section, flags=re.M)


def _workflow_job_block(text: str, job: str) -> str:
    """Return one job's YAML block with comment lines stripped."""

    jobs_section = text.split("\njobs:", 1)[1]
    match = re.search(rf"^  {re.escape(job)}:\s*$", jobs_section, flags=re.M)
    assert match is not None, f"workflow job {job} is missing"
    tail = jobs_section[match.end() :]
    following = re.search(r"^  [A-Za-z0-9_-]+:\s*$", tail, flags=re.M)
    block = tail if following is None else tail[: following.start()]
    return "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("#"))


@pytest.mark.smoke
@_NO_REPO_CI
def test_ci_dormancy_census_matches_workflows() -> None:
    """Dormancy is an asserted fact: every vars-gated job is disclosed, exactly.

    The 2026-08-10 release published with all four round21 jobs silently
    'skipped' — nothing asserted that skipping was expected. This census fails
    in both directions: a new repository-variable gate that ci_dormancy.json
    does not disclose, and a disclosed gate that no longer exists.
    """

    census = json.loads(_CI_DORMANCY_PATH.read_text(encoding="utf-8"))
    assert census["schema_version"] == "torchlens.ci-dormancy.v1"
    disclosed = {(row["workflow"], row["job"]) for row in census["dormant_jobs"]}

    observed: set[tuple[str, str]] = set()
    for workflow_path in sorted(_WORKFLOWS_DIR.glob("*.yml")):
        text = workflow_path.read_text(encoding="utf-8")
        relative = f".github/workflows/{workflow_path.name}"
        for job in _workflow_job_names(text):
            block = _workflow_job_block(text, job)
            if re.search(r"^\s+if:.*\bvars\.", block, flags=re.M):
                observed.add((relative, job))
            # A job whose `needs:` includes a vars-gated job is transitively
            # dormant and must be disclosed too.
            for needed in re.findall(r"^\s+- (crawler-round21-[a-z-]+)\s*$", block, flags=re.M):
                needed_block = _workflow_job_block(text, needed)
                if re.search(r"^\s+if:.*\bvars\.", needed_block, flags=re.M):
                    observed.add((relative, job))

    assert observed == disclosed, (
        "CI dormancy census drifted; update .github/ci_dormancy.json in the same"
        f" change. undisclosed={sorted(observed - disclosed)}"
        f" stale={sorted(disclosed - observed)}"
    )


@pytest.mark.smoke
@_NO_REPO_CI
def test_every_marker_tier_has_a_runner() -> None:
    """Every registered pytest marker tier has a named workflow runner.

    Before BL-1 the crawler corpus's heavy/serial tiers — and the whole
    non-smoke half (14 red nodes) — had NO workflow that ever ran them. The
    table maps each marker to evidence (workflow, substring) that a runner
    selects or sweeps it; a NEW marker fails this census until its runner (or
    an explicit dormancy row) lands in the same change.
    """

    import tomllib

    pyproject = tomllib.loads((_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    registered = {
        marker.split(":", 1)[0].strip()
        for marker in pyproject["tool"]["pytest"]["ini_options"]["markers"]
    }

    # marker -> (authority, substring evidence that a runner covers it). The
    # crawler tiers are selected by the gate commands in required_gate_set.json
    # (gate:<name>), which every CI gate job runs through the gate runner; the
    # torchlens tiers stay workflow-selected. The crawler-full gate runs the
    # ENTIRE corpus minus rare and minus the rigs, so it is the sweep runner
    # for markers that select subsets of the corpus; the rigs have their own.
    runner_evidence: dict[str, tuple[str, str]] = {
        "smoke": ("gate:crawler-smoke", "smoke and not rig"),
        "rig": ("gate:crawler-rigs", "rig"),
        "slow": ("weekly.yml", '-m "slow'),
        "rare": ("weekly.yml", "-m rare"),
        "heavy": ("gate:crawler-full", "not rig"),
        "serial": ("gate:crawler-full", "not rig"),
        "optional": ("gate:crawler-full", "not rig"),
        "backend_parity": ("nightly.yml", "backend_parity"),
        "backend_jax": ("nightly.yml", "backend_jax"),
        "backend_tinygrad": ("nightly.yml", "backend_tinygrad"),
        "backend_paddle": ("nightly.yml", "backend_paddle"),
        "tf_backend": ("nightly.yml", "tf_backend"),
        "round21_linux_real": ("tests.yml", "-m round21_linux_real"),
        "round21_macos_real": ("tests.yml", "-m round21_macos_real"),
    }
    assert registered == set(runner_evidence), (
        "marker registry and runner table drifted; every marker tier needs a"
        f" declared runner. unmapped={sorted(registered - set(runner_evidence))}"
        f" stale={sorted(set(runner_evidence) - registered)}"
    )
    workflows = "\n".join(
        _workflow_text(path.name) for path in sorted(_WORKFLOWS_DIR.glob("*.yml"))
    )
    for marker, (authority, evidence) in sorted(runner_evidence.items()):
        if authority.startswith("gate:"):
            gate = _GATE_MANIFEST.gate(authority.removeprefix("gate:"))
            command = list(gate.command or ())
            assert _marker_selection(command) == evidence, (
                f"marker {marker} lost its gate selection: {evidence!r} not the -m of {gate.name}"
            )
            assert f"--gate {gate.name}" in workflows, (
                f"gate {gate.name} is not run by any workflow"
            )
        else:
            assert evidence in _workflow_text(authority), (
                f"marker {marker} lost its runner: {evidence!r} not found in {authority}"
            )


@pytest.mark.smoke
@_NO_REPO_CI
def test_required_workflow_jobs_are_mandatory_and_fail_closed() -> None:
    """Table-driven mandatory-job pins across tests/quality/lint/release.

    'Mandatory' means: the job exists, carries no event/vars condition beyond
    the disclosed dormancy census, and tolerates no failure. The release
    workflow's publish job must depend on lint, tests, AND quality.
    """

    mandatory = (
        ("tests.yml", "crawler-smoke"),
        ("tests.yml", "crawler-round21-deliberate-reversions"),
        ("quality.yml", "type-check"),
        ("quality.yml", "dep-audit"),
        ("lint.yml", "ruff"),
    )
    for workflow_name, job in mandatory:
        block = _workflow_job_block(_workflow_text(workflow_name), job)
        assert "\n    if:" not in block, f"{workflow_name}:{job} grew a job-level gate"
        assert "continue-on-error" not in block, f"{workflow_name}:{job} tolerates failure"
        assert "MENAGERIE_RELEASE_RUNNERS" not in block, (
            f"{workflow_name}:{job} inherited the release-runner opt-out"
        )
        assert "\n    needs:" not in block, (
            f"{workflow_name}:{job} gained a needs: edge — a dependency on a"
            " gated job silently skips this one (that is how the reversion"
            " suite went dormant)"
        )

    release = _workflow_text("release.yml")
    release_job = _workflow_job_block(release, "release")
    needs_line = re.search(r"^\s+needs:\s*\[(?P<needs>[^\]]+)\]", release_job, flags=re.M)
    assert needs_line is not None, "release job lost its needs: [lint, tests, quality] edge"
    needed = {part.strip() for part in needs_line.group("needs").split(",")}
    assert {"lint", "tests", "quality"} <= needed, (
        f"release must wait on lint+tests+quality; found {sorted(needed)}"
    )

    # The quality gate must be the ratchet, not a bare mypy sweep.
    quality = _workflow_text("quality.yml")
    assert "tools/mypy_ratchet.py check" in quality, (
        "quality.yml must run the mypy baseline ratchet"
    )


# --------------------------------------------------------------------------
# Tier-placement census (DX-32). A launch-critical tripwire that lives only
# in the dark half of the corpus protects nothing on the per-merge path: the
# 200a width-saga fixes were pinned — where pinned at all — outside every
# blocking runner. Every module on this closed list must contribute at least
# one node to the smoke tier (the committed lock) or to a round21 release
# registry. Growing the list is deliberate; shrinking it is a reviewed change.
# --------------------------------------------------------------------------

_LAUNCH_CRITICAL_TRIPWIRE_MODULES = (
    "menagerie/crawler/tests/test_anti_substitution_inventories.py",
    "menagerie/crawler/tests/test_blocking_resolution.py",
    "menagerie/crawler/tests/test_driver_dependency_composition_closure.py",
    "menagerie/crawler/tests/test_gate_verdicts.py",
    "menagerie/crawler/tests/test_launch_preflight.py",
    "menagerie/crawler/tests/test_release_gate_seams.py",
    "menagerie/crawler/tests/test_rotation_diff.py",
    "menagerie/crawler/tests/test_scheduler_pipelined_leg.py",
)


@pytest.mark.smoke
def test_every_launch_critical_tripwire_module_has_a_blocking_node(
    request: pytest.FixtureRequest,
) -> None:
    """Every launch-critical-tripwire module has >=1 smoke or round21 node.

    The smoke set is the LIVE session's pre-deselection collection snapshot
    (the conftest records it before ``-m`` deselection), so this census holds
    under ``-m smoke`` and full runs alike; the exact-inventory lock it used
    to read retired with RC-3.
    """

    conftest_module = next(
        plugin
        for plugin in request.config.pluginmanager.get_plugins()
        if str(getattr(plugin, "__file__", "")).endswith("menagerie/crawler/tests/conftest.py")
    )
    smoke_nodes = list(conftest_module._SMOKE_COLLECTED_NODE_IDS)
    assert smoke_nodes, "conftest recorded no smoke-marked node ids"
    smoke_modules = {node.split("::", 1)[0] for node in smoke_nodes}
    # The census is total only when the session collected the corpus ROOT (or
    # an ancestor): a single-module invocation sees its own nodes alone and
    # would report every other tripwire module dark.
    corpus_root = (_CRAWLER_ROOT / "tests").resolve()
    invocation_dir = Path(str(request.config.invocation_params.dir))
    invoked = [
        (invocation_dir / str(arg).split("::", 1)[0]).resolve() for arg in request.config.args
    ]
    if not any(root == corpus_root or root in corpus_root.parents for root in invoked):
        pytest.skip("session did not collect the crawler corpus root")
    registry_modules: set[str] = set()
    for registry_name in ("round21_macos_real_nodes.json", "round21_linux_real_nodes.json"):
        payload = json.loads((_CRAWLER_ROOT / "tests" / registry_name).read_text(encoding="utf-8"))
        registry_modules.update(node.split("::", 1)[0] for node in payload["nodes"])
    dark = [
        module
        for module in _LAUNCH_CRITICAL_TRIPWIRE_MODULES
        if module not in smoke_modules and module not in registry_modules
    ]
    assert not dark, (
        "tier-placement census: launch-critical tripwire modules with ZERO "
        f"blocking-runner nodes (smoke or round21): {dark}"
    )
