"""Round-17 graceful-shutdown award-boundary composition regression tests."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import pkgutil
import signal
import textwrap
from types import CodeType, FunctionType
from typing import Any

import pytest

import menagerie.crawler as crawler_package
import menagerie.crawler.driver as driver_module
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.driver import (
    AuthorArtifact,
    DriverConfig,
    SupervisedForwardLane,
    WorkItem,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.conftest import (
    RealEnvironmentFixture,
    RealEnvironmentLane,
    real_environment_registry,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakeChecker,
    TypedAdapterPatch,
    _driver,
    _paths,
    _snapshot,
    apply_typed_adapter_patch,
    finalize_typed_adapter_patch,
)


_TINY_ADAPTER = """from __future__ import annotations

import torch
import menagerie_round19_sentinel as round19_sentinel


class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


def build_model() -> object:
    assert round19_sentinel.INTERPRETER_SENTINEL == 'round19-selected-prefix'
    return Tiny()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 2, device=device),), {})
"""

VS2_LANDING_MANIFEST: dict[str, Any] = {
    "findings": ("SOL-R16-03",),
    "production_symbols": {
        "driver": (
            "_SHUTDOWN_ADMISSION_REGISTRY",
            "CrawlerDriver._forward_and_reduce",
            "CrawlerDriver._terminalize",
            "CrawlerDriver._append_terminal_revision",
        ),
    },
    "real_composition_nodes": (
        "test_signal_after_real_forward_publishes_and_awards_nothing_then_resumes",
        "test_signal_at_admission_boundary_publishes_and_awards_nothing_then_resumes",
    ),
    "structural_nodes": (
        "test_shutdown_admission_registry_and_atomic_award_sections_are_complete",
        "test_vs2_landing_manifest_is_complete",
    ),
}


class TinyAdapterAuthor(FakeAuthor):
    """Author one real eval-only typed adapter for the shutdown composition."""

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Return a proposal whose accepted code can run in the real worker."""

        artifact = super().author(item, work_root, config, context)
        artifact.source_manifest["sources"][0]["retrieval_status"] = "fetched"
        proposal = artifact.proposal
        apply_typed_adapter_patch(
            artifact,
            TypedAdapterPatch(
                source=_TINY_ADAPTER,
                evidence_supports=(
                    "implementation.code_manifest[].path",
                    "implementation.code_manifest[].sha256",
                    "implementation.source_to_code_map[].code_locator",
                    "implementation.source_to_code_map[].code_path",
                    "implementation.source_to_code_map[].disposition",
                    "implementation.source_to_code_map[].evidence_ids[]",
                    "implementation.source_to_code_map[].material_item",
                    "implementation.source_to_code_map[].source_id",
                    "implementation.source_to_code_map[].source_locator",
                    "implementation.upstream_files[].path",
                    "implementation.upstream_files[].sha256",
                    "implementation.upstream_files[].source_id",
                    "implementation.upstream_files[].use",
                ),
            ),
        )
        facts = proposal["proposed_facts"]
        source = facts["source_resolution"]["sources"][0]
        excerpt = facts["evidence"]["excerpts"][0]
        facts["source_resolution"].update(
            {
                "rung": "R2_VENDOR",
                "decision": "typed vendor adapter from exact mirrored upstream bytes",
                "attempted_rungs": [
                    {
                        "rung": "R1_LIBRARY",
                        "result": "unavailable",
                        "reason_code": "no-declarative-library-recipe",
                        "evidence_ids": ["evidence-1"],
                    },
                    {
                        "rung": "R2_VENDOR",
                        "result": "selected",
                        "reason_code": "exact-mirrored-source-adapted",
                        "evidence_ids": ["evidence-1"],
                    },
                ],
            }
        )
        facts["implementation"]["upstream_files"] = [
            {
                "source_id": "source-1",
                "path": "source.bin",
                "sha256": source["content_sha256"],
                "use": "exact source grounding for the typed adapter",
            }
        ]
        facts["implementation"]["source_to_code_map"] = [
            {
                "material_item": "Tiny forward",
                "source_id": "source-1",
                "source_locator": excerpt["locator"],
                "evidence_ids": ["evidence-1"],
                "code_path": "adapter.py",
                "code_locator": "Tiny.forward",
                "disposition": "vendor-adapted",
            }
        ]
        proposal["verified_hashes"]["source_to_code_map"] = stable_hash(
            facts["implementation"]["source_to_code_map"]
        )
        return finalize_typed_adapter_patch(artifact, config)


def _files_below(path: Path) -> list[str]:
    """Return stable relative names for every materialized file below ``path``."""

    if not path.exists():
        return []
    return sorted(
        str(candidate.relative_to(path)) for candidate in path.rglob("*") if candidate.is_file()
    )


def _called_symbols(function: Any) -> set[str]:
    """Return every direct name or attribute call in one production function."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            symbols.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            symbols.add(node.func.attr)
    return symbols


def _names_model_award(code: CodeType) -> bool:
    """Return whether one code object, or any nested one, names ``append_model``.

    Parameters
    ----------
    code:
        Compiled code object to scan.

    Returns
    -------
    bool
        True when the symbol is referenced anywhere in the compiled body.
    """

    if "append_model" in code.co_names:
        return True
    return any(
        isinstance(constant, CodeType) and _names_model_award(constant)
        for constant in code.co_consts
    )


def _bound_occurrences(tree: ast.AST, name: str) -> list[ast.AST]:
    """Return every construct in one function body that (re)binds ``name``.

    Store/Del name contexts, function parameters, except-handler captures,
    import aliases, nested def/class statements, and global/nonlocal
    declarations all count: any of them can point the name at a different
    object, so the replay classifier below treats a second occurrence as
    disqualifying rather than reasoning about which one wins.

    Parameters
    ----------
    tree:
        Parsed source of one production function.
    name:
        Local name whose binding sites are collected.

    Returns
    -------
    list[ast.AST]
        Every node that binds or rebinds the name.
    """

    bound: list[ast.AST] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            bound.append(node)
        elif isinstance(node, ast.arg) and node.arg == name:
            bound.append(node)
        elif isinstance(node, ast.ExceptHandler) and node.name == name:
            bound.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.extend(
                node for alias in node.names if (alias.asname or alias.name.split(".")[0]) == name
            )
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ):
            bound.append(node)
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            bound.append(node)
    return bound


def _appends_only_into_replay_reducers(member: FunctionType) -> bool:
    """Decide structurally whether one function's model appends are all replay.

    A ``.append_model(`` call is an in-memory REPLAY (read-side projection
    work, not a canonical award) exactly when its receiver provably cannot be
    a canonical reducer. The proof is dataflow, never the receiver's name:

    * the receiver is a plain local name;
    * that name is bound exactly ONCE in the whole function, by a single
      direct ``name = _replay_reducer(...)`` assignment (any second binding,
      parameter capture, loop target, or import alias disqualifies);
    * ``_replay_reducer`` itself is never rebound locally, and the name in
      the function's globals IS the reducer module's ``_replay_reducer`` --
      a module-local shadow of the same name classifies as canonical;
    * :func:`_assert_replay_reducer_is_write_free` separately proves that
      ``_replay_reducer`` returns a reducer whose every ledger is in-memory.

    Anything the rule cannot prove is classified CANONICAL, so an
    unclassifiable new append site fails the exact-set assertion below
    instead of hiding in the replay bucket.

    Parameters
    ----------
    member:
        Production function that calls ``.append_model(``.

    Returns
    -------
    bool
        True when EVERY ``.append_model(`` call in the body is a proven
        replay append.
    """

    from menagerie.crawler import reducer as reducer_module

    tree = ast.parse(textwrap.dedent(inspect.getsource(member)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append_model"
    ]
    if not calls:
        return False
    if member.__globals__.get("_replay_reducer") is not reducer_module._replay_reducer:
        return False
    if _bound_occurrences(tree, "_replay_reducer"):
        return False
    for call in calls:
        receiver = call.func.value
        if not isinstance(receiver, ast.Name):
            return False
        assigns = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == receiver.id
        ]
        if len(assigns) != 1 or len(_bound_occurrences(tree, receiver.id)) != 1:
            return False
        value = assigns[0].value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_replay_reducer"
        ):
            return False
    return True


#: Filesystem and write primitives no in-memory replay ledger may ever name.
_WRITE_PRIMITIVE_NAMES = frozenset(
    {
        "open",
        "write",
        "writelines",
        "write_text",
        "write_bytes",
        "unlink",
        "rename",
        "replace",
        "mkdir",
        "rmdir",
        "remove",
        "truncate",
        "flush",
        "fsync",
        "os",
        "io",
        "shutil",
        "pathlib",
        "Path",
        "TemporaryFile",
        "NamedTemporaryFile",
    }
)


def _names_write_primitive(code: CodeType) -> set[str]:
    """Return every write/filesystem primitive named by one code object."""

    named = set(code.co_names) & _WRITE_PRIMITIVE_NAMES
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            named |= _names_write_primitive(constant)
    return named


def _assert_replay_reducer_is_write_free() -> None:
    """Prove ``_replay_reducer`` builds a reducer with no canonical ledger path.

    The replay classifier above licenses an ``append_model`` call as read-side
    only because its receiver comes from ``_replay_reducer``; this is the other
    half of that proof. Structurally (source AST): every ledger attribute the
    factory installs is a direct ``_ReplayLedger(...)`` / ``_ReplayArtifactLedger(...)``
    construction, and the instance is marked ``_semantic_replay = True`` so
    ``append_model`` takes the replay arm. Behaviorally (compiled code): neither
    replay ledger class names any filesystem or write primitive, so nothing a
    replay append reaches can touch canonical bytes.
    """

    from menagerie.crawler import reducer as reducer_module

    tree = ast.parse(textwrap.dedent(inspect.getsource(reducer_module._replay_reducer)))
    ledger_constructors: dict[str, str] = {}
    semantic_flags: list[Any] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Attribute):
            continue
        if target.attr in {"_models", "_attempts", "_gates", "_artifacts"}:
            assert isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name), (
                f"replay ledger {target.attr} is not a direct constructor call"
            )
            ledger_constructors[target.attr] = node.value.func.id
        if target.attr == "_semantic_replay":
            semantic_flags.append(node.value)
    assert ledger_constructors == {
        "_models": "_ReplayLedger",
        "_attempts": "_ReplayLedger",
        "_gates": "_ReplayLedger",
        "_artifacts": "_ReplayArtifactLedger",
    }
    assert [flag.value for flag in semantic_flags if isinstance(flag, ast.Constant)] == [True]
    for ledger_class in (reducer_module._ReplayLedger, reducer_module._ReplayArtifactLedger):
        for member in vars(ledger_class).values():
            if isinstance(member, FunctionType):
                named = _names_write_primitive(member.__code__)
                assert not named, (
                    f"{ledger_class.__name__}.{member.__name__} names write "
                    f"primitives {sorted(named)}"
                )


def _model_award_functions() -> dict[str, Any]:
    """Return every crawler function that appends a model, classified.

    The atomic-award structural audit below is only worth anything if it inspects
    EVERY place a model can be awarded. Naming those places by hand is what let
    this test rot: the terminal award moved out of ``_terminalize`` into the
    revision appender it delegates to, and the test kept auditing the empty shell.
    So the sites are DISCOVERED from the code rather than declared, and the caller
    asserts the discovered set exactly. A new, moved, or duplicated award site
    fails this test instead of quietly escaping its gaze.

    Each discovered site is classified ``"canonical"`` or ``"replay"`` by
    :func:`_appends_only_into_replay_reducers` -- receiver dataflow, never a
    name allowlist -- and the classification is fail-closed: anything not
    provably an in-memory replay append counts as a canonical award.

    The sweep is deliberately package-wide rather than driver-wide. ``CrawlerDriver``
    is assembled from mixins across several modules, so any per-module list would
    reintroduce exactly the blind spot this helper exists to remove.

    Discovery reads COMPILED code, not source, so nothing can be excluded by being
    unreadable: a dataclass-generated method has no source but still has a code
    object, and is scanned like any other. A candidate that names the award but
    cannot then be read as source raises rather than being skipped, and a crawler
    module that cannot be imported fails the sweep instead of shrinking it.

    Returns
    -------
    dict[str, Any]
        ``"Owner.method"`` or ``"function"`` to ``(module name, kind)`` for
        every crawler function that calls ``.append_model(``, where kind is
        ``"canonical"`` or ``"replay"``.
    """

    found: dict[str, Any] = {}

    def record(label: str, member: FunctionType, module_name: str) -> None:
        """Record one member when its body really calls ``.append_model(``."""

        if not _names_model_award(member.__code__):
            return
        tree = ast.parse(textwrap.dedent(inspect.getsource(member)))
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append_model"
            for node in ast.walk(tree)
        ):
            kind = "replay" if _appends_only_into_replay_reducers(member) else "canonical"
            found[label] = (module_name, kind)

    for info in pkgutil.iter_modules(crawler_package.__path__):
        if info.ispkg:
            continue
        module = importlib.import_module(f"{crawler_package.__name__}.{info.name}")
        for owner_name, owner in vars(module).items():
            if not isinstance(owner, type) or owner.__module__ != module.__name__:
                continue
            for attribute, member in vars(owner).items():
                if isinstance(member, FunctionType):
                    record(f"{owner_name}.{attribute}", member, module.__name__)
        for name, member in vars(module).items():
            if isinstance(member, FunctionType) and member.__module__ == module.__name__:
                record(name, member, module.__name__)
    return found


def _assert_atomic_award_section(function: Any, publication_call: str) -> None:
    """Assert one award section guards, then commits, without an interior check.

    Parameters
    ----------
    function:
        Driver function that owns a canonical model award.
    publication_call:
        Source spelling of the artifact-publication call inside that award.

    Returns
    -------
    None
        Raises when the guarded ordering or the atomic section is broken.
    """

    source = inspect.getsource(function)
    pre_publication = source.index('"pre-publication"')
    publication_admission = source.index('"pre-publication-admission"')
    pre_commit = source.index('"pre-award-commit"')
    publication = source.index(publication_call)
    append = source.index("reducer.append_model(")
    post_commit = source.index('"post-award-commit"')
    assert pre_publication < publication_admission < pre_commit < publication < append < post_commit
    assert "_check_shutdown" not in source[publication:append]


def _run_after_forward_shutdown(root: str, fixture: RealEnvironmentFixture) -> None:
    """Run the real driver under its production signal handler in a child process."""

    root_path = Path(root)
    snapshot = _snapshot(root_path, count=1)
    paths = _paths(root_path, snapshot)

    def signal_after_forward(boundary: str, stable_id: str) -> None:
        """Request graceful shutdown exactly after the real worker forward."""

        del stable_id
        if boundary == "after-forward":
            os.kill(os.getpid(), signal.SIGTERM)

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    driver = _driver(
        root_path,
        snapshot,
        author=TinyAdapterAuthor(),
        forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
        environments=RealEnvironmentLane(fixture),
        boundary=signal_after_forward,
        registry=real_environment_registry(fixture),
    )
    interrupted = driver.run()

    artifact_events = scan_jsonl(paths.ledgers.artifacts)
    operational = scan_jsonl(paths.operational_ledger)
    attempts = scan_jsonl(paths.ledgers.attempts)
    observation: dict[str, Any] = {
        "status": interrupted.status,
        "admission_boundary": (
            interrupted.shutdown_interruption.admission_boundary
            if interrupted.shutdown_interruption is not None
            else None
        ),
        "models": len(scan_jsonl(paths.ledgers.models)),
        "attempt_results": [attempt["result"] for attempt in attempts],
        "artifact_event_kinds": [event["event_kind"] for event in artifact_events],
        "operational_event_kinds": [event["event_kind"] for event in operational],
        "public_mirror_files": _files_below(paths.runtime_root / "mirrors" / "public"),
        "repository_files": _files_below(root_path / "menagerie"),
        "worker_lease_exists": paths.worker_lease.exists(),
        "handlers_restored": all(
            signal.getsignal(signum) == previous_handlers[signum]
            for signum in (signal.SIGTERM, signal.SIGINT)
        ),
        "driver_state": json.loads(paths.driver_state.read_text(encoding="utf-8")),
    }

    def reject_second_forward(boundary: str, stable_id: str) -> None:
        """Prove resume consumes the durable real attempt without another worker."""

        del stable_id
        if boundary == "after-forward":
            raise AssertionError("resume unexpectedly admitted a second worker forward")

    resumed_driver = _driver(
        root_path,
        snapshot,
        author=TinyAdapterAuthor(),
        forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
        environments=RealEnvironmentLane(fixture),
        boundary=reject_second_forward,
        registry=real_environment_registry(fixture),
    )
    resumed = resumed_driver.run()
    observation.update(
        {
            "resume_status": resumed.status,
            "resume_models": len(scan_jsonl(paths.ledgers.models)),
            "resume_attempts": len(scan_jsonl(paths.ledgers.attempts)),
        }
    )
    (root_path / "shutdown-observation.json").write_text(
        json.dumps(observation, sort_keys=True), encoding="utf-8"
    )


def _run_admission_boundary_shutdown(
    root: str,
    fixture: RealEnvironmentFixture,
    target_boundary: str,
) -> None:
    """Signal at one admission event and record interruption plus real-prefix resume."""

    root_path = Path(root)
    snapshot = _snapshot(root_path, count=1)
    paths = _paths(root_path, snapshot)
    author = TinyAdapterAuthor()
    checker = FakeChecker()
    observed_boundaries: list[str] = []
    signal_sent = False

    def signal_at_boundary(boundary: str, stable_id: str) -> None:
        """Send one real SIGTERM at the selected supported lifecycle event."""

        nonlocal signal_sent
        del stable_id
        observed_boundaries.append(boundary)
        if boundary == target_boundary and not signal_sent:
            signal_sent = True
            os.kill(os.getpid(), signal.SIGTERM)

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    driver = _driver(
        root_path,
        snapshot,
        author=author,
        checker=checker,
        forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
        environments=RealEnvironmentLane(fixture),
        boundary=signal_at_boundary,
        registry=real_environment_registry(fixture),
    )
    interrupted = driver.run()

    attempts_before_resume = scan_jsonl(paths.ledgers.attempts)
    observation: dict[str, Any] = {
        "status": interrupted.status,
        "admission_boundary": (
            interrupted.shutdown_interruption.admission_boundary
            if interrupted.shutdown_interruption is not None
            else None
        ),
        "signal_sent": signal_sent,
        "observed_boundaries": observed_boundaries,
        "author_calls": sum(author.calls.values()),
        "checker_calls": checker.metadata_calls + checker.fidelity_calls,
        "models": len(scan_jsonl(paths.ledgers.models)),
        "attempt_results": [attempt["result"] for attempt in attempts_before_resume],
        "artifact_event_kinds": [
            event["event_kind"] for event in scan_jsonl(paths.ledgers.artifacts)
        ],
        "gate_count": len(scan_jsonl(paths.ledgers.gates)),
        "operational_event_kinds": [
            event["event_kind"] for event in scan_jsonl(paths.operational_ledger)
        ],
        "public_mirror_files": _files_below(paths.runtime_root / "mirrors" / "public"),
        "repository_files": _files_below(root_path / "menagerie"),
        "worker_lease_exists": paths.worker_lease.exists(),
        "handlers_restored": all(
            signal.getsignal(signum) == previous_handlers[signum]
            for signum in (signal.SIGTERM, signal.SIGINT)
        ),
        "driver_state": json.loads(paths.driver_state.read_text(encoding="utf-8")),
    }

    def reject_repeated_forward(boundary: str, stable_id: str) -> None:
        """Reject a second worker only when the pre-publication attempt is durable."""

        del stable_id
        if target_boundary == "pre-publication" and boundary == "after-forward":
            raise AssertionError("resume unexpectedly admitted a second worker forward")

    resumed_driver = _driver(
        root_path,
        snapshot,
        author=TinyAdapterAuthor(),
        checker=FakeChecker(),
        forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
        environments=RealEnvironmentLane(fixture),
        boundary=reject_repeated_forward,
        registry=real_environment_registry(fixture),
    )
    resumed = resumed_driver.run()
    resumed_attempts = scan_jsonl(paths.ledgers.attempts)
    observation.update(
        {
            "resume_status": resumed.status,
            "resume_models": len(scan_jsonl(paths.ledgers.models)),
            "resume_attempts": len(resumed_attempts),
            "resume_manifest_identities": sorted(
                {str(attempt["execution_read_manifest_identity"]) for attempt in resumed_attempts}
            ),
            "resume_environment_authority_ids": sorted(
                {
                    str(attempt["environment"]["environment_authority_id"])
                    for attempt in resumed_attempts
                }
            ),
            "resume_environment_ids": sorted(
                {str(attempt["environment"]["env_id"]) for attempt in resumed_attempts}
            ),
            "resume_environment_authority_epochs": sorted(
                {str(attempt["environment"]["authority_epoch"]) for attempt in resumed_attempts}
            ),
            "resume_selected_interpreters": sorted(
                {
                    str(attempt["environment"]["selected_interpreter_relative_path"])
                    for attempt in resumed_attempts
                }
            ),
        }
    )
    (root_path / f"shutdown-{target_boundary}-observation.json").write_text(
        json.dumps(observation, sort_keys=True), encoding="utf-8"
    )


def test_signal_after_real_forward_publishes_and_awards_nothing_then_resumes(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """SIGTERM after a real v3 forward must leave the slot unawarded and resumable."""

    process = multiprocessing.get_context("fork").Process(
        target=_run_after_forward_shutdown,
        args=(str(tmp_path), real_environment_fixture),
    )
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        pytest.fail("shutdown composition child did not exit")
    assert process.exitcode == 0

    observation = json.loads((tmp_path / "shutdown-observation.json").read_text(encoding="utf-8"))
    assert observation["status"] == "interrupted:shutdown"
    assert observation["models"] == 0
    assert observation["admission_boundary"] == "post-attempt-pre-award"
    assert observation["attempt_results"] == ["succeeded"]
    assert observation["artifact_event_kinds"] == ["staged-private"]
    assert observation["operational_event_kinds"].count("worker-shutdown-interrupted") == 1
    assert observation["public_mirror_files"] == []
    assert observation["repository_files"] == []
    assert observation["worker_lease_exists"] is False
    assert observation["handlers_restored"] is True
    assert observation["driver_state"] == {"status": "interrupted:shutdown"}
    assert observation["resume_status"] == "complete"
    assert observation["resume_models"] == 1
    assert observation["resume_attempts"] == 1


@pytest.mark.parametrize(
    (
        "target_boundary",
        "expected_admission_boundary",
        "expected_author_calls",
        "expected_checker_calls",
        "expected_attempt_results",
        "expected_artifact_events",
        "expected_gate_count",
    ),
    (
        ("pre-author", "author-admission", 0, 0, [], [], 0),
        ("pre-checker", "checker-admission", 1, 0, [], ["staged-private"], 0),
        (
            "pre-publication",
            "pre-publication-admission",
            1,
            1,
            ["succeeded"],
            ["staged-private"],
            1,
        ),
    ),
)
def test_signal_at_admission_boundary_publishes_and_awards_nothing_then_resumes(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
    target_boundary: str,
    expected_admission_boundary: str,
    expected_author_calls: int,
    expected_checker_calls: int,
    expected_attempt_results: list[str],
    expected_artifact_events: list[str],
    expected_gate_count: int,
) -> None:
    """Each pre-admission SIGTERM must append no model or public artifact."""

    process = multiprocessing.get_context("fork").Process(
        target=_run_admission_boundary_shutdown,
        args=(str(tmp_path), real_environment_fixture, target_boundary),
    )
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        pytest.fail(f"{target_boundary} shutdown composition child did not exit")
    assert process.exitcode == 0

    observation = json.loads(
        (tmp_path / f"shutdown-{target_boundary}-observation.json").read_text(encoding="utf-8")
    )
    assert observation["status"] == "interrupted:shutdown"
    assert observation["admission_boundary"] == expected_admission_boundary
    assert observation["signal_sent"] is True
    assert observation["observed_boundaries"][-1] == target_boundary
    assert observation["author_calls"] == expected_author_calls
    assert observation["checker_calls"] == expected_checker_calls
    assert observation["models"] == 0
    assert observation["attempt_results"] == expected_attempt_results
    assert "failed" not in observation["attempt_results"]
    assert observation["artifact_event_kinds"] == expected_artifact_events
    assert observation["gate_count"] == expected_gate_count
    assert observation["operational_event_kinds"].count("worker-shutdown-interrupted") == 1
    assert observation["public_mirror_files"] == []
    assert observation["repository_files"] == []
    assert observation["worker_lease_exists"] is False
    assert observation["handlers_restored"] is True
    assert observation["driver_state"] == {"status": "interrupted:shutdown"}
    assert observation["resume_status"] == "complete"
    assert observation["resume_models"] == 1
    assert observation["resume_attempts"] == 1
    assert len(observation["resume_manifest_identities"]) == 1
    assert len(observation["resume_environment_authority_ids"]) == 1
    assert observation["resume_environment_ids"] == [str(real_environment_fixture.prefix)]
    assert observation["resume_environment_authority_epochs"] == [
        "menagerie.crawler.environment-authority.v1"
    ]
    assert observation["resume_selected_interpreters"] == ["bin/python"]


@pytest.mark.smoke
def test_shutdown_admission_registry_and_atomic_award_sections_are_complete() -> None:
    """All award/publication edges remain guarded or inside the atomic section."""

    registry = driver_module._SHUTDOWN_ADMISSION_REGISTRY
    assert registry == {
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
    # Every model-append site in the crawler, discovered rather than declared, and
    # classified CANONICAL vs REPLAY by receiver dataflow (never by name): a site
    # only lands in the replay bucket when every one of its `.append_model(` calls
    # provably appends into the write-free in-memory reducer `_replay_reducer`
    # returns. The two canonical driver award boundaries are audited below; the run
    # award lives in `_award_worker_result` (reviewed M3 extraction:
    # `_forward_and_reduce` delegates to it so the pipelined scheduler can award
    # from recorded attempts through the SAME tail), while the terminal award moved
    # into the revision appender that `_terminalize` wraps with its
    # failure-containment ladder. The reducer's projection replays
    # (`project_dependency_current` and the PC1 incremental-patch mirror
    # `_replay_appended_currency`) append into throwaway replay reducers and cross
    # no shutdown boundary. This equality is the proof that the structural audit
    # below is total: any new, moved, or duplicated append site anywhere in the
    # package -- including one whose receiver is merely NAMED like a replay --
    # fails here first, as a canonical entry.
    _assert_replay_reducer_is_write_free()
    discovered = _model_award_functions()
    assert discovered == {
        "ReceiptDriverMixin._award_worker_result": (
            "menagerie.crawler.driver_receipts",
            "canonical",
        ),
        "CrawlerDriver._append_terminal_revision": ("menagerie.crawler.driver", "canonical"),
        # Reviewed (terminal-review backlog sweep): the third canonical append
        # owner supersedes a REFUSED backlog head verbatim with the review-held
        # marking so its requeue grant's `unreviewed-terminal` hold can
        # release. Publication-free by construction -- the audit below pins
        # that no `_authorize_*`/publish call exists on this path, so no
        # atomic award section applies.
        "CrawlerDriver._review_terminal_backlog": ("menagerie.crawler.driver", "canonical"),
        # Reviewed (orphan-citation adoption, 2026-08-27; write door moved to
        # the reducer per P5-SOL-05, round 4): the fourth canonical append
        # owner re-appends an already review-held failure terminal VERBATIM
        # under the leg's own joined run when the head's in-window citation is
        # orphaned ("position is not provenance"). The append site now lives
        # in `CanonicalReducer.append_orphan_adoption`, which re-derives the
        # ENTIRE closed adoption rule from ledgers before its one internal
        # `.append_model(` call; the driver's `_adopt_orphan_citations` pass
        # is the redundant first fence and calls only the API. Disposition-
        # invariant and publication-free by construction; the same audit below
        # pins that no `_authorize_*`/publish call exists on this path either.
        "CanonicalReducer.append_orphan_adoption": ("menagerie.crawler.reducer", "canonical"),
        "project_dependency_current": ("menagerie.crawler.reducer", "replay"),
        "CanonicalReducer._replay_appended_currency": ("menagerie.crawler.reducer", "replay"),
        # AW-3: the operator award-restoration arm VALIDATES its candidate
        # through a write-free replay reducer at the recorded basis (its one
        # `.append_model(` call), then persists under the operator tool's own
        # driver lock -- no live driver, no shutdown boundary crossed.
        "CanonicalReducer.append_award_restoration": (
            "menagerie.crawler.reducer",
            "replay",
        ),
    }
    assert {label for label, (_, kind) in discovered.items() if kind == "canonical"} == {
        "ReceiptDriverMixin._award_worker_result",
        "CrawlerDriver._append_terminal_revision",
        "CrawlerDriver._review_terminal_backlog",
        "CanonicalReducer.append_orphan_adoption",
    }
    # The sweep's append is a review-marking supersession, never an award or a
    # publication: it must guard the shutdown boundary and validate through
    # `prepare_model`, and it must NEVER reach an artifact authorization or
    # publish call -- the review-held revision can only ADD the human-review
    # requirement to an already-negative disposition.
    sweep_calls = _called_symbols(driver_module.CrawlerDriver._review_terminal_backlog)
    assert {
        "_check_shutdown",
        "prepare_model",
        "append_model",
        "terminal_review_hold_candidate",
    } <= sweep_calls
    assert not {
        "_authorize_and_publish_artifact",
        "_authorize_terminal_artifact",
        "publish_authorized_artifact",
    } & sweep_calls
    # The adoption's append is a re-citation supersession under the same
    # discipline: shutdown-guarded, validated through `prepare_model` with the
    # post-prepare drift refusal, gated by the closed adoption rule, and NEVER
    # reaching an authorization or publish call -- the adopted revision can
    # only re-cite an already review-held negative disposition. Since
    # P5-SOL-05 (round 4) the append itself lives in the reducer's
    # `append_orphan_adoption` write door, which re-derives the FULL closed
    # rule from ledgers before its one internal `append_model` call; the
    # driver pass stays the redundant first fence and calls only the API.
    adoption_calls = _called_symbols(driver_module.CrawlerDriver._adopt_orphan_citations)
    assert {
        "_check_shutdown",
        "prepare_model",
        "append_orphan_adoption",
        "terminal_review_hold_candidate",
        "orphan_hold_adoption_refusal",
        "orphan_adoption_drift",
    } <= adoption_calls
    assert not {
        "_authorize_and_publish_artifact",
        "_authorize_terminal_artifact",
        "publish_authorized_artifact",
    } & adoption_calls
    import menagerie.crawler.reducer as reducer_module  # noqa: PLC0415 -- audit-local

    door_calls = _called_symbols(reducer_module.CanonicalReducer.append_orphan_adoption)
    assert {
        "append_model",
        "orphan_hold_adoption_refusal",
        "orphan_adoption_drift",
        "_live_grant_parks",
        "generation_run_ids",
        "model_head_citations",
    } <= door_calls
    assert not {
        "_authorize_and_publish_artifact",
        "_authorize_terminal_artifact",
        "publish_authorized_artifact",
        "authorize_publication",
    } & door_calls
    # Round-2 conditions (Opus C3/C4, 2026-08-27): the discipline must hold
    # IN THE MEMBER LOOP, not merely somewhere in the function -- an
    # out-of-loop `_check_shutdown` satisfied the set-membership audit above
    # while the append burst ran unguarded and unbounded. Pin the source
    # order: the shutdown check and the per-leg bound both sit after the
    # member loop opens, and the hold-candidate builder sits inside the
    # typed-refusal try (no `except` between them), so a malformed head
    # refuses member-scope instead of crashing the leg untyped.
    adoption_source = inspect.getsource(driver_module.CrawlerDriver._adopt_orphan_citations)
    member_loop = adoption_source.index("for stable_id, entry, cited_run in orphans:")
    assert adoption_source.index('_check_shutdown("orphan-citation-adoption")', member_loop)
    assert adoption_source.index("ORPHAN_ADOPTION_PER_LEG", member_loop)
    builder_call = adoption_source.index("terminal_review_hold_candidate(", member_loop)
    guarding_try = adoption_source.rindex("try:", member_loop, builder_call)
    assert "except" not in adoption_source[guarding_try:builder_call]

    # `_forward_and_reduce` no longer holds the award itself, so the delegation edge
    # is what keeps the audited section on the run path at all.
    assert "_award_worker_result" in _called_symbols(
        driver_module.CrawlerDriver._forward_and_reduce
    )
    run_calls = _called_symbols(driver_module.CrawlerDriver._award_worker_result)
    terminal_calls = _called_symbols(driver_module.CrawlerDriver._append_terminal_revision)
    assert {
        "_check_shutdown",
        "_assemble_run_model",
        "_authorize_and_publish_artifact",
        "append_model",
    } <= run_calls
    assert {"_check_shutdown", "_authorize_terminal_artifact", "append_model"} <= terminal_calls
    # `_terminalize` no longer holds the award itself, so the delegation edge is what
    # keeps the audited section on the terminal path at all.
    assert "_append_terminal_revision" in _called_symbols(driver_module.CrawlerDriver._terminalize)

    run_source = inspect.getsource(driver_module.CrawlerDriver._award_worker_result)
    post_attempt = run_source.index('"post-attempt-pre-award"')
    assembly = run_source.index("_assemble_run_model(")
    pre_publication = run_source.index('"pre-publication"')
    assert post_attempt < assembly < pre_publication
    _assert_atomic_award_section(
        driver_module.CrawlerDriver._award_worker_result,
        "_authorize_and_publish_artifact(",
    )
    _assert_atomic_award_section(
        driver_module.CrawlerDriver._append_terminal_revision,
        "_authorize_terminal_artifact(",
    )


@pytest.mark.smoke
def test_vs2_landing_manifest_is_complete() -> None:
    """The VS2 landing unit names its driver seams and collected regressions."""

    for symbol in VS2_LANDING_MANIFEST["production_symbols"]["driver"]:
        current: Any = driver_module
        for part in symbol.split("."):
            current = getattr(current, part)
    expected_nodes = {
        *VS2_LANDING_MANIFEST["real_composition_nodes"],
        *VS2_LANDING_MANIFEST["structural_nodes"],
    }
    assert expected_nodes <= globals().keys()
