"""D-60 census: every append-only ledger writer is lock- or recovery-protected.

``environment-turn-effort.jsonl`` shipped as the ONE append-only ledger with
neither a writer lock nor an evidenced torn-tail recovery path, and silently
un-counted one env-turn attempt per crash. This census is the permanent
tripwire that stops the NEXT hand-rolled ledger from repeating that: every
append-mode open (``open(..., "a*")`` and ``os.open(..., O_APPEND)``) in
``menagerie/crawler`` production source is AST-enumerated at test runtime and
must appear in the committed closed set below, classified under a closed
discipline vocabulary. Adding an append site without registering it -- or
registering a JSONL ledger with neither protection arm -- fails set equality.

Discipline vocabulary (CLOSED -- widening it is a reviewed vocabulary change):

* ``writer-lock`` -- the append holds an exclusive kernel lock (data-inode
  flock, or DriverLock-serialized single writer with in-lock healing).
* ``evidenced-tail-recovery`` -- the ledger's read/open path routes torn
  tails through recordio's evidenced recovery (``recover_torn_tail`` /
  ``read_with_tail_recovery`` / ledger-chain ``tail_recovery``).
* ``writer-lock+evidenced-tail-recovery`` -- both arms (the valve pattern).
* ``recordio-internal`` -- the locking/recovery machinery itself.
* ``non-ledger`` -- append-mode opens that are not campaign JSONL ledgers
  (lock files, flock probes, subprocess stream sinks, replay-fixture writers).
* ``todo-disclosed`` -- an append-only JSONL ledger with NEITHER arm. Every
  such row is segregated in ``_TODO_DISCLOSED`` with a reason and asserted
  EXACTLY: the list can never grow silently, and fixing an entry must shrink
  it in the same reviewed change.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_CRAWLER_ROOT = Path(__file__).resolve().parents[1]

#: The closed discipline vocabulary. Order-free; membership is the contract.
_DISCIPLINES = frozenset(
    {
        "writer-lock",
        "evidenced-tail-recovery",
        "writer-lock+evidenced-tail-recovery",
        "recordio-internal",
        "non-ledger",
        "todo-disclosed",
    }
)

#: Source tokens that prove each protection arm at module level. The census
#: verifies every claimed arm against the module's own text, so a mapping row
#: cannot claim a protection its module does not reference.
_LOCK_TOKENS = ("fcntl.flock",)
_RECOVERY_TOKENS = (
    "recover_torn_tail",
    "read_with_tail_recovery",
    "tail_recovery(",
    "decode_ledger(",
)

#: COMMITTED CLOSED SET: every append-mode open site in production source,
#: keyed ``<module relpath>::<enclosing qualname>``, mapped to its declared
#: discipline. A new append site anywhere under menagerie/crawler (tests
#: excluded) fails set equality until it is classified here in a reviewed
#: change -- and "todo-disclosed" additionally requires a reasoned row in
#: ``_TODO_DISCLOSED`` below.
_APPEND_ONLY_LEDGER_CENSUS: dict[str, str] = {
    # -- valve ledgers: exclusive data-inode flock across append AND
    #    evidenced torn-tail recovery on read (the canonical hand-rolled shape).
    "checker_lane.py::append_checker_lane_deferral_row": "writer-lock+evidenced-tail-recovery",
    # AG-4 conflict disclosure ledger (Wave-2 batch assembly): exclusive
    # data-inode flock + durable fsync across the idempotent first-wins
    # append; reads route through read_with_tail_recovery with durable
    # evidence. The census caught the unregistered writer at assembly --
    # the D-60 tripwire working; classified here in the same reviewed change.
    "checker_conflicts.py::append_checker_verdict_conflict_row": (
        "writer-lock+evidenced-tail-recovery"
    ),
    "close_object.py::append_close_object": "writer-lock+evidenced-tail-recovery",
    "disk_floor.py::append_disk_floor_deferral_row": "writer-lock+evidenced-tail-recovery",
    "env_lifecycle.py::_append_environment_turn_row": "writer-lock+evidenced-tail-recovery",
    "environment_coverage.py::append_coverage_deferral_row": (
        "writer-lock+evidenced-tail-recovery"
    ),
    "environment_rebuild.py::append_rebuild_deferral_row": (
        "writer-lock+evidenced-tail-recovery"
    ),
    "grant_parks.py::append_grant_park_row": "writer-lock+evidenced-tail-recovery",
    "grant_parks.py::append_grant_park_void": "writer-lock+evidenced-tail-recovery",
    "host_capacity.py::append_capacity_deferral_row": "writer-lock+evidenced-tail-recovery",
    "leg_generation.py::_append_leg_event": "writer-lock+evidenced-tail-recovery",
    "leg_generation.py::open_leg_generation": "writer-lock+evidenced-tail-recovery",
    "paid_session_durability.py::AnchoredLedger.append": (
        "writer-lock+evidenced-tail-recovery"
    ),
    "quota_deferral.py::mint_quota_deferral": "writer-lock+evidenced-tail-recovery",
    "refusal_parks.py::append_refusal_park_row": "writer-lock+evidenced-tail-recovery",
    "refusal_parks.py::append_refusal_park_void": "writer-lock+evidenced-tail-recovery",
    "release_vocabulary.py::append_blocked_prerequisite_record": (
        "writer-lock+evidenced-tail-recovery"
    ),
    # R2-4/SOL-01: the append-once class refinement rides the registry's exact
    # writer shape (exclusive data-inode flock, chain-linked rows, idempotent
    # identical re-append, typed conflict refusal).
    "rung_staging.py::append_rung_class_refinement": "writer-lock+evidenced-tail-recovery",
    "rung_staging.py::append_rung_registration": "writer-lock+evidenced-tail-recovery",
    # RC-2: the measured-rotations ledger (records/control/rotations.jsonl).
    # Exclusive data-inode flock across the read-verify-link-append critical
    # section, chain-linked rows (links REQUIRED), idempotent identical
    # re-append, typed conflict refusal; reads route through decode_ledger's
    # evidenced torn-tail recovery.
    "rotations.py::append_rotation": "writer-lock+evidenced-tail-recovery",
    # D-59: the machine-scoped scheduler ownership registry (outside any
    # records root by design). Exclusive data-inode flock across the append;
    # reads fail closed on corruption (a torn host registry is adjudicated by
    # an operator through the census, never silently recovered).
    "scheduler_ownership.py::SchedulerOwnershipRegistry._append": "writer-lock",
    "scheduler_parks.py::append_scheduler_quarantine_row": (
        "writer-lock+evidenced-tail-recovery"
    ),
    "scheduler_parks.py::append_scheduler_quarantine_void": (
        "writer-lock+evidenced-tail-recovery"
    ),
    # -- writer-lock arm only: exclusive flock across append (torn tails are
    #    prevented or healed under the same lock; reads happen in-lock).
    # RC-1: the paced-start / configuration_canary ledger (records/control/
    # configuration-canary.jsonl). Exclusive data-inode flock across the
    # append; the driver lock is the outer single-writer authority.
    "configuration_canary.py::_append": "writer-lock",
    "invocation_identity.py::_append_invocation_row": "writer-lock",
    "promotion.py::append_promotion_row": "writer-lock",
    "rung_stratification.py::append_membership_acknowledgment": "writer-lock",
    # -- evidenced-tail-recovery arm only: lockless appends serialized by an
    #    outer authority (DriverLock) or single-tool use; reads recover torn
    #    tails through recordio with durable evidence.
    "checkpoint.py::append_canonical_requeue_grant": "evidenced-tail-recovery",
    "checkpoint.py::append_canonical_requeue_void": "evidenced-tail-recovery",
    # P5 fable-F5/F19: the append-only licensing-ack history. Appends are
    # serialized by the operator acknowledgment act itself (single tool);
    # reads route through read_with_tail_recovery with per-row digest stamps
    # refusing any tamper.
    "leg_licensing_guard.py::acknowledge_licensing_debt": "evidenced-tail-recovery",
    "rotation_registry.py::append_rotation_registration": "evidenced-tail-recovery",
    # -- the recovery/locking machinery itself.
    "recordio.py::JsonlLedger.__init__": "recordio-internal",
    "recordio.py::JsonlLedger._acquire_data_lock": "recordio-internal",
    "recordio.py::JsonlLedger.append": "recordio-internal",
    "recordio.py::recover_torn_tail": "recordio-internal",
    # -- append-mode opens that are not campaign JSONL ledgers.
    "doctor.py::SystemDoctorProbes.lock_available": "non-ledger",
    "doctor.py::_lock_available": "non-ledger",
    "driver_contracts.py::DriverLock.__enter__": "non-ledger",
    "operator_environment.py::_exclusive_file_lock": "non-ledger",
    "operator_environment.py::_try_exclusive_file_lock": "non-ledger",
    "records_root_guard.py::_probe_flock": "non-ledger",
    "supervisor.py::CrawlerSupervisor._run_driver_once": "non-ledger",
    "tools/detection_replay.py::ReplayHarness._advance": "non-ledger",
    "tools/launch_monitor.py::acquire_monitor_lock": "non-ledger",
    "worker_supervisor.py::_finish_macos_denial_audit": "non-ledger",
    # -- disclosed unprotected ledgers (see _TODO_DISCLOSED).
    "author_attempts.py::AttemptHandle.event": "todo-disclosed",
    "checkpoint_cache.py::ValidationMemo.flush": "todo-disclosed",
    "driver_admission.py::AdmissionEnvironmentMixin._append_instrumentation_row": (
        "todo-disclosed"
    ),
    "driver_admission.py::append_environment_clearance_event": "todo-disclosed",
    "notification_outbox.py::_append": "todo-disclosed",
    "operator_environment.py::_emit_telemetry": "todo-disclosed",
    "tools/eta_scorecard.py::append_row": "todo-disclosed",
    "tools/gate_verdicts.py::append_verdict": "todo-disclosed",
}

#: SEGREGATED disclosure of every append-only JSONL ledger with neither
#: protection arm, with the reason it is disclosed rather than fixed here
#: (each is outside the D-60 lane's write territory). Asserted EXACTLY:
#: fixing one must remove its row in the same change, and a new unprotected
#: ledger can never ride in silently. "designed" reasons are candidates for a
#: blessed third discipline (single-write O_APPEND) -- a reviewed vocabulary
#: change, not a unilateral one.
_TODO_DISCLOSED: dict[str, str] = {
    "author_attempts.py::AttemptHandle.event": (
        "per-attempt events.jsonl: single-attempt-owner append, no flock, no"
        " evidenced recovery; forensic log read by sweep tooling"
    ),
    "checkpoint_cache.py::ValidationMemo.flush": (
        "derived validation-memo cache: single-write O_APPEND; a torn tail"
        " triggers a diagnosed rebuild (diagnostics row, not the durable"
        " recovery-evidence ledger); write failures are deliberately non-fatal"
    ),
    "driver_admission.py::AdmissionEnvironmentMixin._append_instrumentation_row": (
        "local instrumentation sidecar: designed single-write O_APPEND,"
        " diagnostics-only consumer, no lock, no evidenced recovery"
    ),
    "driver_admission.py::append_environment_clearance_event": (
        "operator clearance sidecar: designed single-write O_APPEND with a"
        " documented tolerant release-only reader; an undecodable receipt is"
        " skipped WITHOUT evidence -- a silently un-counted clearance keeps an"
        " intent parked"
    ),
    "notification_outbox.py::_append": (
        "critical-send outbox: no writer lock; reader skips a torn final line"
        " tolerantly WITHOUT evidence (documented never-fatal contract) -- a"
        " skipped send-intent re-sends, a skipped send-settled re-pages"
    ),
    "operator_environment.py::_emit_telemetry": (
        "operator telemetry sink: lockless buffered append, tolerant"
        " downstream reader, no evidenced recovery; diagnostics-only"
    ),
    "tools/eta_scorecard.py::append_row": (
        "advisory ETA sidecar: lockless append and a reader that silently"
        " skips EVERY malformed row, torn or interior"
    ),
    "tools/gate_verdicts.py::append_verdict": (
        "gate-verdict ledger: lockless append; strict reader REFUSES a torn"
        " tail typed (no silent un-count, but a crash mid-append bricks"
        " verdict consumption until manual repair)"
    ),
}


def _enclosing_append_sites(source: str) -> list[tuple[str, str]]:
    """Enumerate ``(qualname, kind)`` for every append-mode open in one module.

    ``kind`` is ``open:<mode>`` for a literal append-mode ``open`` and
    ``os.O_APPEND`` for an ``os.open`` whose flag expression names O_APPEND.
    Detection is AST-shaped so formatting and line drift cannot hide a site.
    """

    tree = ast.parse(source)
    sites: list[tuple[str, str]] = []
    stack: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def _qualname(self) -> str:
            return ".".join(stack) if stack else "<module>"

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            kind = self._append_kind(node)
            if kind is not None:
                sites.append((self._qualname(), kind))
            self.generic_visit(node)

        @staticmethod
        def _append_kind(node: ast.Call) -> str | None:
            func = node.func
            mode: object = None
            if isinstance(func, ast.Attribute) and func.attr == "open":
                if isinstance(func.value, ast.Name) and func.value.id == "os":
                    if "O_APPEND" in ast.unparse(node):
                        return "os.O_APPEND"
                    return None
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    mode = node.args[0].value
            elif isinstance(func, ast.Name) and func.id == "open":
                if (
                    len(node.args) > 1
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    mode = node.args[1].value
            else:
                return None
            for keyword in node.keywords:
                if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                    mode = keyword.value.value
            if isinstance(mode, str) and mode.startswith("a"):
                return f"open:{mode}"
            return None

    _Visitor().visit(tree)
    return sites


def _production_sources() -> dict[str, str]:
    """Map every production module relpath to its source text."""

    sources: dict[str, str] = {}
    for path in sorted(_CRAWLER_ROOT.rglob("*.py")):
        relative = path.relative_to(_CRAWLER_ROOT)
        if relative.parts[0] == "tests":
            continue
        sources[relative.as_posix()] = path.read_text(encoding="utf-8")
    return sources


def _discovered_inventory() -> dict[str, list[str]]:
    """AST-derive the live append-site inventory over production source."""

    inventory: dict[str, list[str]] = {}
    for relative, source in _production_sources().items():
        for qualname, kind in _enclosing_append_sites(source):
            inventory.setdefault(f"{relative}::{qualname}", []).append(kind)
    return inventory


def test_append_site_inventory_matches_committed_closed_set() -> None:
    """Set equality both ways: no unregistered writer, no stale census row."""

    discovered = set(_discovered_inventory())
    committed = set(_APPEND_ONLY_LEDGER_CENSUS)
    unregistered = sorted(discovered - committed)
    stale = sorted(committed - discovered)
    assert not unregistered, (
        "append-mode open sites missing from the D-60 ledger census -- classify"
        f" each in _APPEND_ONLY_LEDGER_CENSUS in a reviewed change: {unregistered}"
    )
    assert not stale, (
        "census rows with no surviving append site -- remove each in the same"
        f" change that removed the writer: {stale}"
    )


def test_every_census_row_uses_the_closed_discipline_vocabulary() -> None:
    """No row may invent a protection class the vocabulary does not define."""

    unknown = {
        key: value
        for key, value in _APPEND_ONLY_LEDGER_CENSUS.items()
        if value not in _DISCIPLINES
    }
    assert not unknown, f"unknown ledger discipline(s): {unknown}"


def test_every_claimed_protection_arm_is_verified_in_module_source() -> None:
    """A census row cannot claim a lock or recovery its module never references."""

    sources = _production_sources()
    failures: list[str] = []
    for key, discipline in _APPEND_ONLY_LEDGER_CENSUS.items():
        module = key.split("::", 1)[0]
        source = sources.get(module)
        if source is None:
            continue  # set-equality test already reports the stale row
        if discipline in {"writer-lock", "writer-lock+evidenced-tail-recovery"}:
            if not any(token in source for token in _LOCK_TOKENS):
                failures.append(f"{key}: claims writer-lock, no lock token in {module}")
        if discipline in {
            "evidenced-tail-recovery",
            "writer-lock+evidenced-tail-recovery",
        }:
            if not any(token in source for token in _RECOVERY_TOKENS):
                failures.append(
                    f"{key}: claims evidenced-tail-recovery, no recovery token in {module}"
                )
    assert not failures, "\n".join(failures)


def test_todo_disclosures_are_segregated_and_exact() -> None:
    """Unprotected ledgers are disclosed exactly; the list can never grow silently."""

    todo_rows = {
        key
        for key, value in _APPEND_ONLY_LEDGER_CENSUS.items()
        if value == "todo-disclosed"
    }
    assert todo_rows == set(_TODO_DISCLOSED), (
        "todo-disclosed census rows and the reasoned _TODO_DISCLOSED table must"
        " match exactly -- fixing a ledger removes BOTH rows; disclosing a new"
        " one adds BOTH in a reviewed change"
    )
    empty = sorted(key for key, reason in _TODO_DISCLOSED.items() if not reason.strip())
    assert not empty, f"todo-disclosed rows require a non-empty reason: {empty}"


def test_environment_turn_ledger_is_protected() -> None:
    """The D-60 subject ledger itself can never regress to unprotected."""

    discipline = _APPEND_ONLY_LEDGER_CENSUS["env_lifecycle.py::_append_environment_turn_row"]
    assert discipline == "writer-lock+evidenced-tail-recovery"


def test_scanner_detects_a_new_hand_rolled_append() -> None:
    """Anti-substitution self-check: the detector sees what it must see."""

    synthetic = (
        "import os\n"
        "def sneak(path):\n"
        "    with path.open('a+b') as handle:\n"
        "        handle.write(b'{}\\n')\n"
        "def sneak_os(path):\n"
        "    return os.open(path, os.O_WRONLY | os.O_APPEND)\n"
        "def sneak_kw(path):\n"
        "    return open(path, mode='ab')\n"
        "def innocent(path):\n"
        "    return path.open('rb')\n"
    )
    found = dict(_enclosing_append_sites(synthetic))
    assert found == {
        "sneak": "open:a+b",
        "sneak_os": "os.O_APPEND",
        "sneak_kw": "open:ab",
    }
