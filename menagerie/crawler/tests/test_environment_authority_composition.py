"""Round-19 sealed environment-authority composition regressions."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
from typing import Any, Mapping, Optional

import pytest

import menagerie.crawler.driver as driver_module
from menagerie.crawler.authority import (
    AuthorityDerivationError,
    EnvironmentAuthorityCache,
    RuntimeMember,
    compile_execution_read_manifest_v3,
    verify_execution_read_manifest_v3,
)
from menagerie.crawler.driver import (
    AuthorArtifact,
    SupervisedForwardLane,
)
from menagerie.crawler.driver_contracts import DriverPaths
from menagerie.crawler.environment_rebuild import (
    load_rebuild_deferral_rows,
    rebuild_deferral_path,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.intake import create_intake_snapshot
from menagerie.crawler.policy import detect_os_sandbox, generate_macos_sandbox_profile
from menagerie.crawler.proposal import model_code_manifest
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer
from menagerie.crawler.tests.conftest import (
    RealEnvironmentFixture,
    RealEnvironmentLane,
    _copy_up_real_environment_member,
    hardlink_bytes,
    real_environment_registry,
)
from menagerie.crawler.tests.dry_run_support import DRY_RUN_CASES, TinyModelAuthor
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakeChecker,
    _driver,
    _paths,
    _rebind_fake_author_result,
    _refresh_proposal_identities,
    _snapshot,
    _test_authority_context,
    _write_jsonl,
)


HASH = "sha256:" + "a" * 64

_DECLARED_ADAPTER = """from __future__ import annotations

import torch
import menagerie_round19_sentinel as round19_sentinel
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from declared_model.layers import Tiny


def build_model() -> object:
    assert round19_sentinel.INTERPRETER_SENTINEL == 'round19-selected-prefix'
    return Tiny()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 2, device=device),), {})
"""

_DECLARED_MODEL = """from __future__ import annotations

import torch


class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1
"""


def _denial_adapter(undeclared: Path) -> str:
    """Return typed adapter source that catches one undeclared repository read.

    Parameters
    ----------
    undeclared:
        Exact adjacent repository file excluded from the code manifest.

    Returns
    -------
    str
        Complete trusted adapter source.
    """

    return f"""from __future__ import annotations

import ctypes
import os
import torch
import menagerie_round19_sentinel as round19_sentinel


class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


def _open_undeclared() -> None:
    descriptor = ctypes.CDLL(None, use_errno=True).open(
        os.fsencode({str(undeclared)!r}), os.O_RDONLY
    )
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), {str(undeclared)!r})
    os.close(descriptor)


def build_model() -> object:
    assert round19_sentinel.INTERPRETER_SENTINEL == 'round19-selected-prefix'
    try:
        _open_undeclared()
    except OSError:
        pass
    return Tiny()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 2, device=device),), {{}})
"""


def test_real_hardlink_clone_binds_complete_environment_authority(
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """The real Torch prefix binds hardlinks and startup/runtime content as one unit."""

    binding = real_environment_fixture.binding
    authority = binding.environment_authority
    assert authority is not None
    assert binding.python_executable == real_environment_fixture.prefix / "bin" / "python"
    assert binding.env_generation == authority.environment_generation
    assert binding.base_environment_generation == authority.base_environment_generation
    assert binding.environment_content_sha256 == authority.content_manifest_sha256
    assert binding.environment_authority_id == authority.authority_id
    assert real_environment_fixture.startup_pth in authority.startup_pth_paths
    assert any(
        entry.relative_path.endswith("__future__.py")
        for entry in authority.content_manifest.entries
    )
    assert any(entry.relative_path.endswith(".so") for entry in authority.content_manifest.entries)


def test_hardlinked_prefix_is_one_sealed_authority_and_mutation_stales(
    tmp_path: Path,
) -> None:
    """Hardlinks, startup ``.pth``, native bytes, and metadata form one unit."""

    prefix = tmp_path / "prefix"
    staging = tmp_path / "staging"
    hardlink_bytes(staging / "python", prefix / "bin" / "python", b"python")
    (prefix / "bin" / "python").chmod(0o755)
    hardlink_bytes(
        staging / "future.py",
        prefix / "lib" / "python3.11" / "__future__.py",
        b"future = True\n",
    )
    hardlink_bytes(
        staging / "startup.pth",
        prefix / "lib" / "python3.11" / "site-packages" / "sentinel.pth",
        b"import sentinel\n",
    )
    hardlink_bytes(staging / "native.so", prefix / "lib" / "libsentinel.so", b"native")
    hardlink_bytes(
        staging / "metadata.json",
        prefix / "conda-meta" / "sentinel-1-0.json",
        b"{}\n",
    )

    cache = EnvironmentAuthorityCache()
    authority = cache.bind(
        prefix=prefix,
        selected_interpreter=prefix / "bin" / "python",
        base_environment_generation=HASH,
    )

    assert authority.prefix == prefix.resolve()
    assert authority.selected_interpreter == prefix / "bin" / "python"
    assert authority.environment_generation != HASH
    assert authority.content_manifest_sha256.startswith("sha256:")
    assert all(path.stat().st_nlink > 1 for path in prefix.rglob("*") if path.is_file())
    assert cache.full_seals == 1
    cache.verify(authority)
    assert cache.cheap_validations == 1
    assert cache.full_seals == 1

    changed = prefix / "lib" / "python3.11" / "__future__.py"
    changed.unlink()
    changed.write_text("future = False\n", encoding="utf-8")
    with pytest.raises(AuthorityDerivationError, match="content seal"):
        cache.verify(authority)
    assert cache.rehashes == 1
    assert cache.full_seals == 2


_FRESH_MEMBER_NAMES = ("DryRunFreshLeg2", "DryRunFreshLeg3", "DryRunFreshLeg4")


class _WidenedTinyAuthor(TinyModelAuthor):
    """Alias widened-leg intake names onto the frozen tiny corpus.

    Each alias keeps its OWN canonical identity so no duplicate or family
    machinery engages across the widened rungs.
    """

    def __init__(self) -> None:
        super().__init__()
        for alias in _FRESH_MEMBER_NAMES:
            self._cases[alias] = self._cases["DryRunStableConvNet"]

    def author(self, item: Any, work_root: Path, config: Any, context: Any) -> AuthorArtifact:
        artifact = super().author(item, work_root, config, context)
        facts = artifact.proposal["proposed_facts"]
        if item.intake.name in _FRESH_MEMBER_NAMES:
            facts["identity"]["canonical_name"] = item.intake.name
            _refresh_proposal_identities(
                artifact.proposal,
                checker_model=config.checker_model,
                checker_version=config.checker_version,
            )
            return _rebind_fake_author_result(artifact)
        return artifact


def _registered_widened_snapshot(
    records_root: Path,
    prior_snapshot: Any,
    member_rows: list[dict[str, Any]],
    out_root: Path,
    label: str,
) -> Any:
    """Register one widened successor rung and return its records-side snapshot.

    WS-1 made the ordinary leg's scope cut absolute: settled current members
    are never re-driven, so the only lawful way to give a later leg work is a
    widened successor rung registered through the production new-work door
    (the exact quarantine-test ceremony). Registering under the records root
    also anchors the lineage the driver's prior-intake carve reads, so the
    prior legs' runs awards stay dependency-CURRENT under the new snapshot.
    """

    from menagerie.crawler.intake import load_intake_snapshot
    from menagerie.crawler.rung_staging import RUNG_CLASS_SUCCESSOR, register_existing_snapshot

    out_root.mkdir(parents=True, exist_ok=True)
    master = out_root / "master.jsonl"
    deferred = out_root / "deferred.jsonl"
    _write_jsonl(master, member_rows)
    _write_jsonl(deferred, [])
    raw = create_intake_snapshot(master, deferred, out_root / "intake")
    awards = {
        str(row["stable_id"]): str(row["record_revision"])
        for row in scan_jsonl(records_root / "models" / "current-shard.jsonl")
        if row.get("status", {}).get("kind") == "runs"
    }
    from menagerie.crawler.tests.rt3_support import fixture_composition_attestation

    staged = register_existing_snapshot(
        records_root,
        raw.root,
        campaign_id="c1-mech",
        label=label,
        registered_by="pytest",
        created_at="2026-08-11T01:00:00Z",
        historical_only=False,
        rung_class=RUNG_CLASS_SUCCESSOR,
        derived_from_snapshot_id=prior_snapshot.snapshot_id,
        award_authorities=awards or None,
        # SOL-02: a successor stage presents the PD-7-bearing attestation.
        composition_attestation=fixture_composition_attestation(
            records_root,
            "c1-mech",
            stable_ids=tuple(item.stable_id for item in raw.items),
            label=label,
            snapshot_id=raw.snapshot_id,
            snapshot_sha256=raw.snapshot_sha256,
        ),
    )
    return load_intake_snapshot(records_root / "intake" / staged.registration.snapshot_id)


def test_real_multi_model_cache_closes_currentness_and_quarantines_mutation(
    tmp_path: Path,
    isolated_real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """One lifecycle cache serves fresh legs cheaply and rejects changed prefix bytes.

    WS-1 rewrite: an ordinary leg never re-drives settled current members, so
    a rerun of a settled campaign is a no-op that touches no environment. The
    cache lifecycle is therefore driven the way a real campaign drives it --
    successive WIDENED successor rungs, each adding one never-recorded member
    that authors and executes under the SAME sealed prefix. Unchanged bytes
    revalidate cheaply (no reseal, no rehash); the first leg over MUTATED
    bytes invalidates, reseals, quarantines intent-scoped, and parks exactly
    its own pending work identity while every recorded runs award survives.
    """

    from menagerie.crawler.tests.campaign_binding_support import bind_snapshot_at_records_root

    real_environment_fixture = isolated_real_environment_fixture
    zoo = "cache-fixtures"
    base_rows = [
        {"name": case.name, "zoo": zoo, "variant": "base"}
        for case in (DRY_RUN_CASES[0], DRY_RUN_CASES[1], DRY_RUN_CASES[3])
    ]
    fresh_rows = [{"name": name, "zoo": zoo, "variant": "base"} for name in _FRESH_MEMBER_NAMES]

    master = tmp_path / "cache-master.jsonl"
    deferred = tmp_path / "cache-deferred.jsonl"
    _write_jsonl(master, base_rows)
    _write_jsonl(deferred, [])
    snapshot = create_intake_snapshot(master, deferred, tmp_path / "intake")
    paths = _paths(tmp_path, snapshot)
    records_root = paths.ledgers.models.parent.parent
    bind_snapshot_at_records_root(
        records_root, snapshot, extra_roster_rows=tuple(fresh_rows)
    )
    environments = RealEnvironmentLane(real_environment_fixture)
    registry = real_environment_registry(real_environment_fixture)

    def leg(leg_snapshot: Any) -> Any:
        return _driver(
            tmp_path,
            leg_snapshot,
            author=_WidenedTinyAuthor(),
            checker=FakeChecker(),
            forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
            environments=environments,
            registry=registry,
            paths_override=DriverPaths(
                tmp_path / "runtime", leg_snapshot.root, paths.ledgers
            ),
        )

    first = leg(snapshot).run()
    assert first.status.startswith("complete")
    initial_models = scan_jsonl(paths.ledgers.models)
    initial_attempts = scan_jsonl(paths.ledgers.attempts)
    assert len(initial_models) == 3
    assert {row["status"]["code"] for row in initial_models} == {"runs"}
    assert all(row["result"] == "succeeded" for row in initial_attempts)
    cache = environments.environment_authority_cache
    full_seals_after_first = cache.full_seals
    cheap_after_first = cache.cheap_validations

    second_snapshot = _registered_widened_snapshot(
        records_root, snapshot, base_rows + fresh_rows[:1], tmp_path / "leg2", "cache-leg2"
    )
    second = leg(second_snapshot).run()
    cheap_after_second = cache.cheap_validations
    third_snapshot = _registered_widened_snapshot(
        records_root, second_snapshot, base_rows + fresh_rows[:2], tmp_path / "leg3", "cache-leg3"
    )
    third = leg(third_snapshot).run()
    cheap_after_third = cache.cheap_validations
    full_seals_while_unchanged = cache.full_seals
    rehashes_while_unchanged = cache.rehashes
    assert second.status.startswith("complete")
    assert third.status.startswith("complete")
    settled_models = scan_jsonl(paths.ledgers.models)
    settled_attempts = scan_jsonl(paths.ledgers.attempts)
    assert settled_models[: len(initial_models)] == initial_models
    assert len(settled_models) == 5
    assert {row["status"]["code"] for row in settled_models} == {"runs"}

    authority = real_environment_fixture.binding.environment_authority
    assert authority is not None
    relative_member = next(
        entry.relative_path
        for entry in authority.content_manifest.entries
        if entry.entry_type == "regular-file"
        and entry.relative_path.startswith("include/")
        and entry.relative_path.endswith(".h")
        and not entry.relative_path.startswith("conda-meta/")
    )
    changed = real_environment_fixture.prefix / relative_member
    source = real_environment_fixture.source_prefix / relative_member
    original, private_status = _copy_up_real_environment_member(changed, source)
    changed.write_bytes(original + b"\n# round19 cache mutation\n")
    changed.chmod(stat.S_IMODE(private_status.st_mode))
    fourth_snapshot = _registered_widened_snapshot(
        records_root, third_snapshot, base_rows + fresh_rows, tmp_path / "leg4", "cache-leg4"
    )
    fresh_leg4_id = next(
        item.stable_id for item in fourth_snapshot.items if item.name == _FRESH_MEMBER_NAMES[2]
    )
    try:
        mutated = leg(fourth_snapshot).run()
    finally:
        changed.unlink()
        os.link(source, changed)

    operational_events = scan_jsonl(paths.operational_ledger)
    integrity_quarantines = [
        event
        for event in operational_events
        if event.get("details", {}).get("disposition") == "environment-integrity-quarantined"
    ]
    intent_park_events = [
        event
        for event in operational_events
        if event.get("details", {}).get("disposition") == "environment-setup-failure-parked"
        and "stable_id" not in event.get("details", {})
    ]
    park_rows = load_rebuild_deferral_rows(
        [rebuild_deferral_path(paths.ledgers.models.parent.parent)]
    )

    # Quarantine fan-out follows the closed env-failure routing registry (E2,
    # 9a88695e "host conditions park, never terminalize", DD-1): a mutated
    # prefix is a fact about the intent's SETUP, never about any model, so the
    # pending work identity PARKS intent-scoped in the environment-rebuild
    # deferral ledger and the canonical runs records -- awarded under the sealed
    # pre-mutation authority -- are never rewritten into ``failed:environment``
    # terminals (the abolished fan-out that wiped 11 released awards on
    # 2026-08-08, T2-08). The cache-closure measurements above are unchanged
    # round-19 VS7 semantics, now driven by the widened rungs' fresh members.
    assert {
        "first_status_complete": first.status.startswith("complete"),
        "one_full_seal_after_first": full_seals_after_first == 1,
        "cheap_second_leg_increased": cheap_after_second > cheap_after_first,
        "cheap_third_leg_increased": cheap_after_third > cheap_after_second,
        "one_full_seal_while_unchanged": full_seals_while_unchanged == 1,
        "zero_rehashes_while_unchanged": rehashes_while_unchanged == 0,
        "mutation_added_cheap_validation": cache.cheap_validations > cheap_after_third,
        "full_seals": cache.full_seals,
        "rehashes": cache.rehashes,
        "invalidations": cache.invalidations,
        "mutation_status": mutated.status,
        "integrity_quarantines": len(integrity_quarantines),
        "runs_records_survive_quarantine": scan_jsonl(paths.ledgers.models) == settled_models,
        "quarantine_appends_no_attempts": (scan_jsonl(paths.ledgers.attempts) == settled_attempts),
        "intent_park_events": len(intent_park_events),
        "parked_work_identities": sorted(str(row["stable_id"]) for row in park_rows),
        "park_adjudications": {
            (row["quarantine"]["failure_type"], row["quarantine"]["condition_scope"])
            for row in park_rows
        },
    } == {
        "first_status_complete": True,
        "one_full_seal_after_first": True,
        "cheap_second_leg_increased": True,
        "cheap_third_leg_increased": True,
        "one_full_seal_while_unchanged": True,
        "zero_rehashes_while_unchanged": True,
        "mutation_added_cheap_validation": True,
        "full_seals": 2,
        "rehashes": 1,
        "invalidations": 1,
        "mutation_status": "complete-with-deferrals",
        "integrity_quarantines": 1,
        "runs_records_survive_quarantine": True,
        "quarantine_appends_no_attempts": True,
        "intent_park_events": 1,
        "parked_work_identities": [fresh_leg4_id],
        # The recorded failure_type is the defining module of the exception
        # type at record time; b86a1de5f moved AuthorityDerivationError to
        # completion_line (authority re-exports it for importers).
        "park_adjudications": {
            ("menagerie.crawler.completion_line.AuthorityDerivationError", "intent-setup")
        },
    }


def test_repeated_environment_integrity_quarantine_is_two_events_not_a_conflict(
    tmp_path: Path,
    isolated_real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """Quarantining the same standing environment failure twice records two occurrences.

    Regression for the 2026-08-03 resume crash: two environment cycles hitting
    the same integrity failure derived the same ``event_id`` while their
    logical payloads differed (``queued_work_counts``), so the second append
    died on the operational ledger's conflicting-replay tripwire and the
    campaign became permanently un-resumable. The identity now covers the
    complete payload: each occurrence is its own durable event, and each leg
    ends with honest intent-setup parks (E2 park-not-terminalize) instead of a
    crashed ledger append. WS-1's absolute ordinary scope cut means one leg
    quarantines an intent at most once, so the second occurrence is driven by
    a widened successor rung's fresh member re-entering the still-poisoned
    environment on the next leg -- payloads [2, 1], exactly the shape the
    2026-08-03 crash produced within one resume.
    """

    from menagerie.crawler.tests.campaign_binding_support import bind_snapshot_at_records_root

    real_environment_fixture = isolated_real_environment_fixture
    zoo = "quarantine-fixtures"
    base_rows = [
        {"name": case.name, "zoo": zoo, "variant": "base"}
        for case in (DRY_RUN_CASES[0], DRY_RUN_CASES[1], DRY_RUN_CASES[3])
    ]
    fresh_row = {"name": _FRESH_MEMBER_NAMES[0], "zoo": zoo, "variant": "base"}
    master = tmp_path / "quarantine-master.jsonl"
    deferred = tmp_path / "quarantine-deferred.jsonl"
    _write_jsonl(master, base_rows)
    _write_jsonl(deferred, [])
    snapshot = create_intake_snapshot(master, deferred, tmp_path / "intake")
    paths = _paths(tmp_path, snapshot)
    records_root = paths.ledgers.models.parent.parent
    bind_snapshot_at_records_root(records_root, snapshot, extra_roster_rows=(fresh_row,))
    environments = RealEnvironmentLane(real_environment_fixture)
    registry = real_environment_registry(real_environment_fixture)
    driver = _driver(
        tmp_path,
        snapshot,
        author=_WidenedTinyAuthor(),
        checker=FakeChecker(),
        forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
        environments=environments,
        registry=registry,
        review_at=1,
    )

    paused = driver.run()
    assert paused.status == "paused:review-checkpoint"
    runs_before = [
        row for row in scan_jsonl(paths.ledgers.models) if row["status"]["code"] == "runs"
    ]
    assert len(runs_before) == 1
    runs_ids = {str(row["stable_id"]) for row in runs_before}

    # Any recurring in-use AuthorityDerivationError stands in for the load-induced
    # seal instability observed in the campaign failures; a FIFO in the prefix is
    # deterministic where hardlink ctime churn is not, and the authority walk
    # refuses it on EVERY environment cycle it stands through. WS-1 made the
    # ordinary scope cut absolute (a leg quarantines its own pending work at
    # most once per intent), so the second occurrence of the SAME standing
    # failure is driven the modern way: a widened successor rung whose fresh
    # member re-enters the same poisoned environment on the next leg. The
    # regression target is unchanged -- two occurrences with different
    # payloads (queued counts) must land as two DISTINCT durable events, never
    # the 2026-08-03 same-event-id conflicting-replay crash.
    fifo = real_environment_fixture.prefix / "round19-integrity-fifo"
    os.mkfifo(fifo)
    try:
        resumed = driver.run(after_review=True)
        widened = _registered_widened_snapshot(
            records_root,
            snapshot,
            base_rows + [fresh_row],
            tmp_path / "leg3",
            "quarantine-leg3",
        )
        fresh_id = next(
            item.stable_id for item in widened.items if item.name == _FRESH_MEMBER_NAMES[0]
        )
        third = _driver(
            tmp_path,
            widened,
            author=_WidenedTinyAuthor(),
            checker=FakeChecker(),
            forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
            environments=environments,
            registry=registry,
            paths_override=DriverPaths(tmp_path / "runtime", widened.root, paths.ledgers),
        ).run()
    finally:
        fifo.unlink()

    integrity_events = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition") == "environment-integrity-quarantined"
    ]
    latest_models: dict[str, Any] = {}
    for row in scan_jsonl(paths.ledgers.models):
        latest_models[str(row["stable_id"])] = row
    park_rows = load_rebuild_deferral_rows(
        [rebuild_deferral_path(paths.ledgers.models.parent.parent)]
    )
    # Both quarantines then route through the closed env-failure registry (E2,
    # 9a88695e "host conditions park, never terminalize"): each leg's pending
    # work parks intent-scoped, the one pre-pause runs record survives, and
    # each leg closes honestly as complete-with-deferrals instead of minting
    # ``failed:environment`` model terminals for a setup fact no model observed.
    assert {
        "resumed_status": resumed.status,
        "third_status": third.status,
        "occurrences": len(integrity_events),
        "distinct_event_ids": len({event["event_id"] for event in integrity_events}),
        "queued_counts": [event["queued_work_counts"]["models"] for event in integrity_events],
        "failure_types": {event["details"]["failure_type"] for event in integrity_events},
        "terminal_codes": {row["status"]["code"] for row in latest_models.values()},
        "terminal_models": len(latest_models),
        "parked_work_identities": sorted({str(row["stable_id"]) for row in park_rows}),
        "park_adjudications": {
            (row["quarantine"]["failure_type"], row["quarantine"]["condition_scope"])
            for row in park_rows
        },
    } == {
        "resumed_status": "complete-with-deferrals",
        "third_status": "complete-with-deferrals",
        "occurrences": 2,
        "distinct_event_ids": 2,
        "queued_counts": [2, 1],
        # The recorded failure_type is the defining module of the exception
        # type at record time; b86a1de5f moved AuthorityDerivationError to
        # completion_line (authority re-exports it for importers).
        "failure_types": {"menagerie.crawler.completion_line.AuthorityDerivationError"},
        "terminal_codes": {"runs"},
        "terminal_models": 1,
        "parked_work_identities": sorted(
            [*(item.stable_id for item in snapshot.items if item.stable_id not in runs_ids)]
            + [fresh_id]
        ),
        "park_adjudications": {
            ("menagerie.crawler.completion_line.AuthorityDerivationError", "intent-setup")
        },
    }


def test_outside_selected_interpreter_is_rejected_at_binding(tmp_path: Path) -> None:
    """An interpreter escape such as ``/bin/false`` fails before worker spawn."""

    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    # The escape target must be an executable that really exists outside the prefix, or
    # resolve(strict=True) fails first and the binding is rejected as "unavailable" without
    # ever reaching the containment check this test exists to prove. /bin/false is absent on
    # macOS, so name the running interpreter, which is present on every supported platform.
    (prefix / "bin" / "python").symlink_to(sys.executable)

    with pytest.raises(AuthorityDerivationError, match="selected interpreter.*prefix"):
        EnvironmentAuthorityCache().bind(
            prefix=prefix,
            selected_interpreter=prefix / "bin" / "python",
            base_environment_generation=HASH,
        )


def test_manifest_v3_rejects_changed_interpreter_association(tmp_path: Path) -> None:
    """The v3 verifier binds argv authority to exact sealed interpreter bytes."""

    prefix = tmp_path / "prefix"
    staging = tmp_path / "staging"
    interpreter = prefix / "bin" / "python"
    hardlink_bytes(staging / "python", interpreter, b"python")
    interpreter.chmod(0o755)
    code = tmp_path / "adapter.py"
    code.write_text("VALUE = 1\n", encoding="utf-8")
    authority = EnvironmentAuthorityCache().bind(
        prefix=prefix,
        selected_interpreter=interpreter,
        base_environment_generation=HASH,
    )
    manifest = compile_execution_read_manifest_v3(
        stable_id="m_round19",
        work_id="work-round19",
        execution_identity=HASH,
        code_manifest_identity=stable_hash(["adapter.py"]),
        environment_authority=authority,
        code_members=((code, hash_bytes(code.read_bytes()), "python-source"),),
    )
    verify_execution_read_manifest_v3(manifest)

    interpreter.unlink()
    interpreter.write_bytes(b"changed")
    interpreter.chmod(0o755)
    with pytest.raises(AuthorityDerivationError):
        verify_execution_read_manifest_v3(manifest)


def _typed_artifact(
    item: Any,
    paths: Any,
    driver: Any,
    context: Any,
    adapter_source: str,
    *,
    declared_members: Optional[Mapping[str, str]] = None,
) -> AuthorArtifact:
    """Stage one typed adapter and bind its complete declared code closure.

    Parameters
    ----------
    item, paths, driver, context:
        Existing real-composition driver authority and one intake item.
    adapter_source:
        Entry-point source written as ``adapter.py``.
    declared_members:
        Additional model-package members intentionally admitted by static imports.

    Returns
    -------
    AuthorArtifact
        Rebound proposed artifact consumed by the shipped compiler.
    """

    artifact = FakeAuthor().author(item, paths.work_root, driver.config, context)
    adapter_path = artifact.model_dir / "adapter.py"
    adapter_path.write_text(adapter_source, encoding="utf-8")
    for relative, source in (declared_members or {}).items():
        member = artifact.model_dir / relative
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text(source, encoding="utf-8")
    adapter_digest = hash_bytes(adapter_path.read_bytes())
    code_manifest = [dict(row) for row in model_code_manifest(adapter_path, artifact.model_dir)]
    facts = artifact.proposal["proposed_facts"]
    facts["implementation"].update(
        {
            "recipe_type": "typed-adapter",
            "code_path": "adapter.py",
            "code_sha256": adapter_digest,
            "builder_symbol": "build_model",
            "dummy_call_symbol": "make_dummy_call",
            "library_recipe": None,
            "code_manifest": code_manifest,
        }
    )
    facts["input_contract"]["args"][0]["shape"] = [1, 2]
    facts["modes"]["meaningful_modes"] = ["eval"]
    facts["external_metadata"]["modes"]["meaningful_modes"] = ["eval"]
    facts["evidence"]["excerpts"][0]["supports"] = sorted(
        set(facts["evidence"]["excerpts"][0]["supports"])
        | {
            "implementation.code_manifest[].path",
            "implementation.code_manifest[].sha256",
        }
    )
    artifact.proposal["verified_hashes"]["code"] = adapter_digest
    artifact.proposal["verified_hashes"]["code_manifest"] = stable_hash(code_manifest)
    _refresh_proposal_identities(
        artifact.proposal,
        checker_model=driver.config.checker_model,
        checker_version=driver.config.checker_version,
    )
    return _rebind_fake_author_result(artifact)


def _run_host_denial_composition(
    tmp_path: Path,
    fixture: RealEnvironmentFixture,
    *,
    expected_sandbox: str,
) -> None:
    """Run one positive package and one caught undeclared-read denial end to end.

    Parameters
    ----------
    tmp_path:
        Isolated campaign root.
    fixture:
        Strict real hardlink-clone environment authority.
    expected_sandbox:
        Host OS sandbox kind required by this release proof.
    """

    sandbox = detect_os_sandbox()
    if sandbox is None or sandbox.kind != expected_sandbox:
        message = f"{expected_sandbox} host enforcement is unavailable"
        if os.environ.get("MENAGERIE_RELEASE_GATE") == "1":
            pytest.fail(f"unmet-release-gate: {message}")
        pytest.skip(message)

    snapshot = _snapshot(tmp_path, count=2)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    items = driver._ordered_work(snapshot, {})
    context = _test_authority_context(snapshot, driver.config)
    environment = fixture.binding
    context = replace(
        context,
        environment_generations={
            **context.environment_generations,
            environment.family: environment.env_generation,
        },
    )

    positive = _typed_artifact(
        items[0],
        paths,
        driver,
        context,
        _DECLARED_ADAPTER,
        declared_members={
            "declared_model/__init__.py": "",
            "declared_model/layers.py": _DECLARED_MODEL,
        },
    )
    undeclared = (
        paths.work_root / items[1].stable_id / "author" / "fake-model" / "undeclared_repo.py"
    )
    denial = _typed_artifact(
        items[1],
        paths,
        driver,
        context,
        _denial_adapter(undeclared),
    )
    undeclared.write_text("UNDECLARED_REPOSITORY_MEMBER = True\n", encoding="utf-8")
    assert all(
        row["path"] != undeclared.name
        for row in denial.proposal["proposed_facts"]["implementation"]["code_manifest"]
    )

    with CanonicalReducer(paths.ledgers, context) as reducer:

        def append_attempt(attempt: Mapping[str, Any]) -> None:
            """Persist one worker-derived attempt through the canonical reducer."""

            reducer.append_attempt(attempt)

        positive = driver._stage_author_result(items[0], positive, reducer)
        positive_gate = (
            FakeChecker().check_metadata([positive], paths.work_root, driver.config).gate
        )
        assert positive_gate is not None
        reducer.append_gate(positive_gate)
        positive_attempts = SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()).forward(
            positive,
            environment,
            1,
            paths.work_root,
            worker_lock_path=paths.worker_lock,
            worker_lease_path=paths.worker_lease,
            run_id=driver.config.run_id,
            attempt_sink=append_attempt,
        )
        assert len(positive_attempts) == 1
        assert positive_attempts[0]["result"] == "succeeded", positive_attempts[0]["error"]
        assert positive_attempts[0]["raw_award_receipt"] is not None
        assert (
            positive_attempts[0]["parent_attestation"]["named_raw_award_receipt_sha256"]
            == positive_attempts[0]["raw_award_receipt_sha256"]
        )
        positive_argv = list(positive_attempts[0]["invocation"]["argv"])
        assert str(environment.python_executable) in positive_argv, (
            "selected interpreter is missing from worker argv"
        )
        interpreter_index = positive_argv.index(str(environment.python_executable))
        assert positive_argv[interpreter_index : interpreter_index + 4] == [
            str(environment.python_executable),
            "-B",
            "-m",
            "menagerie.crawler.worker",
        ]
        positive_model = driver_module._assemble_run_model(
            items[0], positive, positive_attempts, [positive_gate], driver.config
        )
        decisions = driver._license_decisions(positive)
        assert positive.staged is not None
        assert set(decisions) == {claim.claim_id for claim in positive.staged.custody_claims}
        object_digests = {obj.object_id: obj.content_sha256 for obj in positive.staged.objects}
        assert all(
            decisions[claim.claim_id].content_sha256 == object_digests[claim.object_id]
            for claim in positive.staged.custody_claims
        )
        driver._authorize_and_publish_artifact(
            positive,
            positive_model,
            [positive_gate],
            reducer,
        )
        reducer.append_model(reducer.prepare_model(positive_model))
        public_mirror = paths.runtime_root / "mirrors" / "public"
        positive_public_bytes = {
            path.relative_to(public_mirror).as_posix(): path.read_bytes()
            for path in public_mirror.rglob("*")
            if path.is_file()
        }
        assert positive_public_bytes

        denial = driver._stage_author_result(items[1], denial, reducer)
        denial_gate = FakeChecker().check_metadata([denial], paths.work_root, driver.config).gate
        assert denial_gate is not None
        reducer.append_gate(denial_gate)
        denial_attempts = SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()).forward(
            denial,
            environment,
            1,
            paths.work_root,
            worker_lock_path=paths.worker_lock,
            worker_lease_path=paths.worker_lease,
            run_id=driver.config.run_id,
            attempt_sink=append_attempt,
        )
        assert len(denial_attempts) == 1
        assert denial_attempts[0]["result"] == "failed"
        assert denial_attempts[0]["stage"] == "policy"
        assert denial_attempts[0]["error"]["reason_code"] == "checkpoint-read"
        assert str(undeclared) in denial_attempts[0]["policy_observation"]["checkpoint_paths"]
        if expected_sandbox == "sandbox-exec":
            assert denial_attempts[0]["supervisor_observation"]["signal"] == signal.SIGKILL

        for artifact, attempts in ((positive, positive_attempts), (denial, denial_attempts)):
            closure = driver_module._collect_worker_executable_closure(artifact, environment)
            execution_identity = driver_module._execution_identity(
                artifact,
                environment,
                closure_identity=closure.identity,
            )
            recomputed = driver_module._compile_worker_read_manifest(
                artifact,
                environment,
                execution_identity,
                closure=closure,
            )
            assert {attempt["execution_read_manifest_identity"] for attempt in attempts} == {
                recomputed.manifest_id
            }

    persisted_attempts = scan_jsonl(paths.ledgers.attempts)
    denied_attempt = next(
        row for row in persisted_attempts if row["stable_id"] == items[1].stable_id
    )
    assert str(undeclared) in denied_attempt["policy_observation"]["checkpoint_paths"]
    models = scan_jsonl(paths.ledgers.models)
    assert [(row["stable_id"], row["status"]["code"]) for row in models] == [
        (items[0].stable_id, "runs")
    ]
    assert {
        path.relative_to(public_mirror).as_posix(): path.read_bytes()
        for path in public_mirror.rglob("*")
        if path.is_file()
    } == positive_public_bytes


def test_linux_real_compiler_denies_caught_undeclared_repo_read_and_awards_package(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Real bwrap/strace denial persists while a declared package earns ``runs``."""

    if sys.platform != "linux":
        message = "Linux release host is unavailable"
        if os.environ.get("MENAGERIE_RELEASE_GATE") == "1":
            pytest.fail(f"unmet-release-gate: {message}")
        pytest.skip(message)
    fixture = request.getfixturevalue("real_environment_fixture")
    _run_host_denial_composition(tmp_path, fixture, expected_sandbox="bubblewrap")


def test_macos_real_compiler_denies_caught_undeclared_repo_read_and_awards_package(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Real Seatbelt denial persists while the same declared package earns ``runs``."""

    if sys.platform != "darwin":
        message = "macOS release host is unavailable"
        if os.environ.get("MENAGERIE_RELEASE_GATE") == "1":
            pytest.fail(f"unmet-release-gate: {message}")
        pytest.skip(message)
    fixture = request.getfixturevalue("real_environment_fixture")
    _run_host_denial_composition(tmp_path, fixture, expected_sandbox="sandbox-exec")


def _macos_profile_manifest(tmp_path: Path) -> tuple[Any, dict[str, Path]]:
    """Build one freshly verified four-part v3 profile fixture.

    Parameters
    ----------
    tmp_path:
        Isolated authority tree.

    Returns
    -------
    tuple[Any, dict[str, pathlib.Path]]
        Shipped v3 manifest and its named semantic members.
    """

    prefix = tmp_path / "sealed-prefix"
    interpreter = prefix / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"python")
    interpreter.chmod(0o755)
    runtime = prefix / "lib" / "runtime.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("RUNTIME = True\n", encoding="utf-8")
    external = tmp_path / "external-escape.py"
    external.write_text("EXTERNAL = True\n", encoding="utf-8")
    (prefix / "lib" / "external.py").symlink_to(external)
    authority = EnvironmentAuthorityCache().bind(
        prefix=prefix,
        selected_interpreter=interpreter,
        base_environment_generation=HASH,
    )
    model = tmp_path / "model.py"
    crawler = tmp_path / "crawler_bootstrap.py"
    asset = tmp_path / "standard-asset.txt"
    request_path = tmp_path / "request.json"
    for path, content in (
        (model, "MODEL = True\n"),
        (crawler, "CRAWLER = True\n"),
        (asset, "asset\n"),
        (request_path, "{}\n"),
    ):
        path.write_text(content, encoding="utf-8")
    manifest = compile_execution_read_manifest_v3(
        stable_id="m_macos_profile",
        work_id="work-macos-profile",
        execution_identity=HASH,
        code_manifest_identity=stable_hash([model.name]),
        environment_authority=authority,
        code_members=(
            RuntimeMember(
                model,
                hash_bytes(model.read_bytes()),
                "python-source",
                "accepted-model-code-manifest",
            ),
        ),
        worker_members=(
            RuntimeMember(
                crawler,
                hash_bytes(crawler.read_bytes()),
                "python-source",
                "crawler-worker-import-closure",
            ),
        ),
        standard_input_asset=(asset, hash_bytes(asset.read_bytes()), "standard-test-asset"),
    )
    return manifest, {
        "prefix": prefix.resolve(),
        "model": model.resolve(),
        "crawler": crawler.resolve(),
        "external": external.resolve(),
        "asset": asset.resolve(),
        "request": request_path.resolve(),
        "runtime": runtime.resolve(),
    }


def test_macos_v3_profile_has_one_fresh_literal_prefix_and_exact_outside_members(
    tmp_path: Path,
) -> None:
    """Seatbelt grants one fresh v3 prefix and exact outside file/directory capabilities."""

    manifest, members = _macos_profile_manifest(tmp_path)
    scratch = tmp_path / "scratch"
    result = tmp_path / "result"
    profile = generate_macos_sandbox_profile(
        (scratch, result),
        allowed_read_paths=(members["request"],),
        execution_read_manifest=manifest,
    )

    # Read grants name the exact file-read-data operation. The file-read* wildcard loses to
    # the exact (deny file-read-data) that opens the profile, so a wildcard grant would be
    # inert and the sandbox would deny every read on the host.
    prefix_grant = f"(allow file-read-data (subpath {json.dumps(str(members['prefix']))}))"
    assert profile.count(prefix_grant) == 1
    subpath_rules = {
        line for line in profile.splitlines() if line.startswith("(allow file-read-data (subpath ")
    }
    assert subpath_rules == {
        '(allow file-read-data (subpath "/System"))',
        '(allow file-read-data (subpath "/usr/lib"))',
        # B1 (2026-08-12): /usr/share by category, matching both read classifiers.
        '(allow file-read-data (subpath "/usr/share"))',
        '(allow file-read-data (subpath "/Library/Apple"))',
        '(allow file-read-data (subpath "/private/etc"))',
        '(allow file-read-data (subpath "/private/var/db/timezone"))',
        '(allow file-read-data (subpath "/dev"))',
        f"(allow file-read-data (subpath {json.dumps(str(scratch.resolve()))}))",
        f"(allow file-read-data (subpath {json.dumps(str(result.resolve()))}))",
        prefix_grant,
    }
    for kind in ("model", "crawler", "external", "asset", "request"):
        literal = f"(allow file-read-data (literal {json.dumps(str(members[kind]))}))"
        assert profile.count(literal) == 1
    outside_members = tuple(
        members[kind] for kind in ("model", "crawler", "external", "asset", "request")
    )
    traversal_directories = {
        parent for member in outside_members for parent in member.parents if parent != Path("/")
    }
    for directory in traversal_directories:
        encoded_directory = json.dumps(str(directory))
        directory_grant = (
            "(allow file-read-data "
            f"(require-all (vnode-type DIRECTORY) (literal {encoded_directory})))"
        )
        assert profile.count(directory_grant) == 1
        assert f"(allow file-read-data (literal {encoded_directory}))" not in profile
        assert f"(allow file-read-data (subpath {encoded_directory}))" not in profile
    undeclared_sibling = members["model"].with_name("undeclared-sibling.py")
    assert f"(literal {json.dumps(str(undeclared_sibling))})" not in profile
    bytecode_rules = {
        (
            "(allow file-read-data "
            f'(regex #"^{re.escape(str(source.parent / "__pycache__" / source.stem))}'
            '\\.[^./]+\\.pyc$"))'
        )
        for source in (members["model"], members["crawler"], members["external"])
    }
    assert {line for line in profile.splitlines() if "(regex " in line} == {
        "(allow file-read-data (require-all (vnode-type DIRECTORY) "
        '(regex #"^/private/tmp/__KMP_REGISTERED_LIB_[0-9]+$")))',
        *bytecode_rules,
    }
    for forbidden_root in (
        Path.cwd().resolve(),
        Path.home().resolve(),
        Path.home().resolve() / ".local" / "lib",
    ):
        assert f"(subpath {json.dumps(str(forbidden_root))})" not in profile
    assert f"(allow file-write* (subpath {json.dumps(str(members['prefix']))}))" not in profile
    assert str(members["runtime"]) not in profile


def test_macos_profile_refuses_stale_environment_prefix_grant(
    tmp_path: Path,
) -> None:
    """A stale v3 authority cannot create a Seatbelt prefix grant."""

    manifest, members = _macos_profile_manifest(tmp_path)
    members["runtime"].write_text("RUNTIME = False\n", encoding="utf-8")
    with pytest.raises(AuthorityDerivationError, match="content seal"):
        generate_macos_sandbox_profile((), execution_read_manifest=manifest)
