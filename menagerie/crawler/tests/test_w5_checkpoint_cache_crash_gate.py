"""W5 merge gate: the checkpoint cache under crash windows (C-W5-MERGE).

Composes the uncatchable-kill matrix with W5's authority-cache layers: a
driver resumed across a SIGKILL must consult the projection cache, degrade to
full live derivation on every poisoned, torn, stale, or absent memo state
(with the typed ``checkpoint-cache-event`` disclosure), and still converge to
the serial no-kill oracle byte-for-byte.  The shadow-verify cell injects a
divergence into a REAL incremental fold -- not a fake projection -- and
proves ``MENAGERIE_CRAWLER_CHECKPOINT_CACHE_VERIFY=1`` detects it, emits the
typed ``incremental-divergence`` event, and answers with the live result.

The whole module skips on trees without the W5 checkpoint cache.
"""

from __future__ import annotations

import dataclasses
import json
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip(
    "menagerie.crawler.checkpoint_cache",
    reason="W5 authority-cache branch only: menagerie.crawler.checkpoint_cache is absent",
)

from menagerie.crawler.artifact_transactions import ArtifactCheckpointIncrement
from menagerie.crawler.checkpoint_cache import ValidationMemo
from menagerie.crawler.recordio import record_validation_memo_key, scan_jsonl
from menagerie.crawler.tests import test_checkpoint_cache as cache_world
from menagerie.crawler.tests.kill_matrix_support import (
    assert_artifact_model_bijection,
    assert_no_double_award,
    assert_projection_equals_oracle,
    capture_durable_state,
    durable_projection,
    recreate_campaign_dir,
    run_forked,
)
from menagerie.crawler.tests.test_grant_pending_parks import (
    _rewrite_intake_with_changed_sources,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    InaccurateChecker,
    RepairingIdentityAuthor,
    _driver,
    _paths,
    _snapshot,
)
from menagerie.crawler.tests.test_uncatchable_kill_durability_matrix import (
    _child_run_plain_with_injection,
    _park_leg2_driver,
    _plain_oracle,
)

pytestmark = pytest.mark.heavy


def _memo_path(paths: Any) -> Path:
    """Return the campaign's persistent validation-memo store path."""

    return paths.runtime_root / "derived" / "checkpoint-cache" / "validation-memo-v1.jsonl"


def _instrumentation_rows(paths: Any) -> list[dict[str, Any]]:
    """Return every local instrumentation row the campaign has emitted."""

    ledger = paths.runtime_root / "instrumentation" / "ledger-hot-path.jsonl"
    if not ledger.exists():
        return []
    return scan_jsonl(ledger, validate=False)


def _cache_event_reasons(paths: Any) -> list[str]:
    """Return the reasons of every typed checkpoint-cache disclosure."""

    return [
        str(row.get("reason"))
        for row in _instrumentation_rows(paths)
        if row.get("metric") == "checkpoint-cache-event"
    ]


def _consultation_modes(paths: Any) -> list[tuple[str, bool]]:
    """Return (mode, enabled) for every hot-path consultation metric."""

    return [
        (str(row.get("checkpoint_cache_mode")), bool(row.get("checkpoint_cache_enabled")))
        for row in _instrumentation_rows(paths)
        if row.get("metric") != "checkpoint-cache-event"
    ]


def _seed_memo_with_real_keys(paths: Any) -> int:
    """Persist genuine positive memo rows for every durable artifact record."""

    memo = ValidationMemo.open(_memo_path(paths).parent)
    seeded = 0
    for record in scan_jsonl(paths.ledgers.artifacts, validate=False):
        key = record_validation_memo_key(record)
        if key is not None:
            memo.record("record", key)
            seeded += 1
    memo.flush()
    assert seeded > 0
    assert _memo_path(paths).exists()
    return seeded


def _kill_award_then_capture(root: Path, count: int) -> Any:
    """Fork-kill the campaign inside the award window and return its paths."""

    exitcode = run_forked(
        _child_run_plain_with_injection, (root, count, "sigkill-on-append-model", False, None)
    )
    assert exitcode == -signal.SIGKILL
    return _paths(root, _snapshot(root, count=count))


def test_resume_after_award_kill_runs_with_the_cache_on_and_matches_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent memo state: the crash-resume consults the default-on cache cleanly.

    The resume's sole projection consultation must run with the cache enabled
    (cold ``full`` mode), flush a memo store bound to the current environment,
    emit zero degradation disclosures, and reproduce the serial oracle.
    """

    monkeypatch.delenv("MENAGERIE_CRAWLER_ARTIFACT_MEMOIZATION", raising=False)
    monkeypatch.delenv("MENAGERIE_CRAWLER_CHECKPOINT_CACHE_VERIFY", raising=False)
    root = tmp_path / "campaign"
    root.mkdir()
    oracle = _plain_oracle(root, count=2)

    paths = _kill_award_then_capture(root, count=2)
    assert not _memo_path(paths).exists()

    snapshot = _snapshot(root, count=2)
    resumed = _driver(root, snapshot).run()
    assert resumed.status == "complete"
    modes = _consultation_modes(paths)
    assert ("full", True) in modes
    assert all(enabled for _mode, enabled in modes)
    assert _cache_event_reasons(paths) == []
    assert _memo_path(paths).exists()
    rows = [json.loads(line) for line in _memo_path(paths).read_text(encoding="utf-8").splitlines()]
    assert rows and all(row["kind"] in {"record", "transaction-document"} for row in rows)
    after = capture_durable_state(root, paths)
    assert_no_double_award(after)
    assert_artifact_model_bijection(root, after)
    assert_projection_equals_oracle(durable_projection(after), oracle)


_POISONS = ("corrupt-row", "torn-tail", "foreign-environment")
_EXPECTED_REASON = {
    "corrupt-row": "checkpoint-cache-corrupt",
    "torn-tail": "checkpoint-cache-torn-tail",
    "foreign-environment": "checkpoint-cache-stale-environment",
}


def _poison_memo(paths: Any, poison: str) -> None:
    """Damage the persisted memo store in one exactly-shaped way."""

    memo_path = _memo_path(paths)
    if poison == "corrupt-row":
        with memo_path.open("ab") as handle:
            handle.write(b'{"kind": "record", not json\n')
    elif poison == "torn-tail":
        with memo_path.open("ab") as handle:
            handle.write(b'{"kind": "record", "key": "sha256:')
    elif poison == "foreign-environment":
        first = json.loads(memo_path.read_text(encoding="utf-8").splitlines()[0])
        foreign = dict(first, environment="sha256:" + "f" * 64, key="sha256:" + "0" * 64)
        with memo_path.open("ab") as handle:
            handle.write(json.dumps(foreign, sort_keys=True).encode("utf-8") + b"\n")
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(poison)


@pytest.mark.parametrize("poison", _POISONS)
def test_resume_after_award_kill_degrades_to_live_on_poisoned_memo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poison: str
) -> None:
    """Every poisoned/torn/stale memo state degrades loudly to live derivation.

    The memo is seeded with genuine positive rows for the killed campaign's
    real artifact records, then damaged.  The resume must disclose the exact
    typed ``checkpoint-cache-event`` reason, never trust the damaged store,
    and converge to the serial no-kill oracle regardless.
    """

    monkeypatch.delenv("MENAGERIE_CRAWLER_ARTIFACT_MEMOIZATION", raising=False)
    monkeypatch.delenv("MENAGERIE_CRAWLER_CHECKPOINT_CACHE_VERIFY", raising=False)
    root = tmp_path / "campaign"
    root.mkdir()
    oracle = _plain_oracle(root, count=2)

    paths = _kill_award_then_capture(root, count=2)
    _seed_memo_with_real_keys(paths)
    _poison_memo(paths, poison)

    snapshot = _snapshot(root, count=2)
    resumed = _driver(root, snapshot).run()
    assert resumed.status == "complete"
    assert _EXPECTED_REASON[poison] in _cache_event_reasons(paths)
    assert ("full", True) in _consultation_modes(paths)
    after = capture_durable_state(root, paths)
    assert_no_double_award(after)
    assert_artifact_model_bijection(root, after)
    assert_projection_equals_oracle(durable_projection(after), oracle)


def test_shadow_verify_detects_an_injected_divergence_in_a_real_incremental_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VERIFY=1 catches a REAL corrupted advance and answers with live truth.

    Unlike the W5 unit test, nothing here is a sentinel object: a real
    ``ArtifactCheckpointIncrement`` folds a real committed transaction, and
    the injected fault silently drops one transaction from the advance's
    returned projection.  Shadow verify must flag the divergence with the
    typed event, report mode ``full``, and hand back the live projection
    containing every committed transaction.
    """

    monkeypatch.delenv("MENAGERIE_CRAWLER_ARTIFACT_MEMOIZATION", raising=False)
    monkeypatch.setenv("MENAGERIE_CRAWLER_CHECKPOINT_CACHE_VERIFY", "1")
    campaign = cache_world._campaign(tmp_path)
    cache_world._commit(campaign, "m_zoo0", b"gate-payload-0")
    driver_root = tmp_path / "driver-campaign"
    driver_root.mkdir()
    driver = _driver(driver_root, _snapshot(driver_root, 1))
    reducer = SimpleNamespace(artifact_ledger=campaign["ledger"], context=campaign["context"])
    kwargs = {
        "artifact_paths": (campaign["ledger"].path,),
        "mirrors": campaign["mirrors"],
        "canonical_root": campaign["canonical_root"],
        "repository_root": campaign["repository_root"],
    }

    first, first_mode = driver._validated_artifact_projection(reducer, **kwargs)
    assert first_mode == "full"
    assert len(first.transactions) == 1

    cache_world._commit(campaign, "m_zoo1", b"gate-payload-1")
    real_advance = ArtifactCheckpointIncrement.advance

    def corrupted_advance(self: Any, new_events: Any) -> Any:
        projection = real_advance(self, new_events)
        truncated = dict(projection.transactions)
        truncated.pop(sorted(truncated)[-1])
        return dataclasses.replace(projection, transactions=truncated)

    monkeypatch.setattr(ArtifactCheckpointIncrement, "advance", corrupted_advance)

    result, mode = driver._validated_artifact_projection(reducer, **kwargs)

    assert mode == "full"
    assert len(result.transactions) == 2
    assert {key[0] for key in result.transactions} == {"m_zoo0", "m_zoo1"}
    assert "incremental-divergence" in _cache_event_reasons(driver.paths)


def test_park_minting_close_is_identical_with_cache_on_and_kill_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache never changes a close that mints parks (T5-6 matrix arm).

    The grant-pending park scenario runs once with the default-on cache and
    once under the ``MENAGERIE_CRAWLER_ARTIFACT_MEMOIZATION=0`` kill switch at
    the same path; parks, statuses, and every durable projection surface must
    be indistinguishable.
    """

    monkeypatch.delenv("MENAGERIE_CRAWLER_CHECKPOINT_CACHE_VERIFY", raising=False)
    root = tmp_path / "campaign"
    root.mkdir()

    def park_legs() -> dict[str, Any]:
        snapshot = _snapshot(root, count=2)
        first = _driver(
            root,
            snapshot,
            author=RepairingIdentityAuthor(),
            checker=InaccurateChecker(),
        ).run()
        assert first.status == "terminal-partition-complete"
        changed = _rewrite_intake_with_changed_sources(root, 2)
        # COMPOSE (AW x WS-1, clause 4 / V3): the changed intake stales every
        # settled head, so leg 2 is the operator's NAMED re-drive.
        second = _park_leg2_driver(root, changed).run()
        assert second.status == "terminal-partition-complete-with-review-holds"
        return durable_projection(capture_durable_state(root, _paths(root, changed)))

    monkeypatch.setenv("MENAGERIE_CRAWLER_ARTIFACT_MEMOIZATION", "0")
    kill_switch = park_legs()
    recreate_campaign_dir(root)

    monkeypatch.delenv("MENAGERIE_CRAWLER_ARTIFACT_MEMOIZATION", raising=False)
    default_on = park_legs()
    assert_projection_equals_oracle(default_on, kill_switch)
