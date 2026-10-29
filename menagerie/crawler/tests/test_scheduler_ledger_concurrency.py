"""T5-5 RED set: `JsonlLedger` under the scheduler's concurrency (M3 merge gate).

`JsonlLedger.append` derives the next sequence from `self._records` with no
thread lock; it is safe only under the all-appends-on-one-thread convention.
The scheduler makes that convention load-bearing, so the ledger must enforce
it: a foreign-thread append gets an immediate typed refusal -- a silent race
is the only wrong answer. Flock contention between writers gets a bounded
wait before the typed `SingleWriterError` instead of an instant hard error.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION_V3
from menagerie.crawler.recordio import (
    JsonlLedger,
    LedgerThreadAffinityError,
    SingleWriterError,
    read_with_tail_recovery,
    scan_jsonl,
)
from menagerie.crawler.tests.conftest import make_attempt

#: Sub-second load-bearing coverage promoted into the mandatory smoke tier (T5-7).
pytestmark = pytest.mark.smoke


def _attempt(stable_id: str, attempt_id: str) -> dict[str, object]:
    payload = make_attempt(stable_id, attempt_id=attempt_id)
    payload.pop("ledger_seq")
    payload.pop("payload_sha256")
    return payload


def test_jsonl_ledger_append_is_thread_safe_or_refuses(tmp_path: Path) -> None:
    """A foreign-thread append is refused with a typed error, never raced.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    outcomes: list[BaseException | None] = []
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(_attempt("m_owner", "attempt-1"))

        def foreign_append() -> None:
            try:
                ledger.append(_attempt("m_foreign", "attempt-2"))
            except BaseException as exc:  # noqa: BLE001 - the exception IS the assertion
                outcomes.append(exc)
            else:
                outcomes.append(None)

        thread = threading.Thread(target=foreign_append)
        thread.start()
        thread.join()
        assert len(outcomes) == 1
        refusal = outcomes[0]
        assert isinstance(refusal, LedgerThreadAffinityError)
        # The refusal happened before any persistence side effect.
        assert ledger.record_count == 1
        # The owning thread keeps working after the refusal.
        result = ledger.append(_attempt("m_owner", "attempt-3"))
        assert result.appended
    records = scan_jsonl(path)
    assert [record["ledger_seq"] for record in records] == [1, 2]


def test_foreign_thread_refusal_is_a_ledger_error(tmp_path: Path) -> None:
    """The affinity refusal participates in the typed ledger error hierarchy.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    from menagerie.crawler.recordio import LedgerError

    assert issubclass(LedgerThreadAffinityError, LedgerError)


def test_flock_contention_waits_bounded_then_acquires(tmp_path: Path) -> None:
    """A second writer waits out transient contention instead of hard-failing.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    holder = JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3)
    release_at = time.monotonic() + 0.2

    def release_soon() -> None:
        while time.monotonic() < release_at:
            time.sleep(0.01)
        holder.close()

    thread = threading.Thread(target=release_soon)
    thread.start()
    try:
        # The bounded wait outlives the holder's release, so this acquires
        # instead of raising.
        with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3, lock_timeout=5.0) as waiter:
            waiter.append(_attempt("m_waiter", "attempt-1"))
    finally:
        thread.join()
    assert len(scan_jsonl(path)) == 1


def test_flock_contention_raises_typed_after_bound(tmp_path: Path) -> None:
    """Contention past the bound stays a typed `SingleWriterError`.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    started = time.monotonic()
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3):
        with pytest.raises(SingleWriterError):
            JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3, lock_timeout=0.2)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0


def test_second_process_writer_is_refused(tmp_path: Path) -> None:
    """A separately started process cannot acquire the live writer lock.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    probe = "\n".join(
        (
            "import sys",
            "from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION_V3",
            "from menagerie.crawler.recordio import JsonlLedger, SingleWriterError",
            "try:",
            "    JsonlLedger(sys.argv[1], ATTEMPT_SCHEMA_VERSION_V3)",
            "except SingleWriterError:",
            "    raise SystemExit(0)",
            "raise SystemExit(19)",
        )
    )
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3):
        result = subprocess.run(
            [sys.executable, "-c", probe, str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0, result.stderr


def test_symlink_alias_cannot_bypass_writer_lock(tmp_path: Path) -> None:
    """Two path spellings of one ledger resolve to the same writer exclusion.

    The side lock is name-derived, so an alias would mint a DIFFERENT side
    lock -- but the data-inode flock rides the canonical bytes both spellings
    share, so the alias opener is refused at the data inode.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(path)
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(_attempt("m_owner", "attempt-1"))
        with pytest.raises(SingleWriterError):
            JsonlLedger(alias, ATTEMPT_SCHEMA_VERSION_V3)


@pytest.mark.parametrize("attack", ["delete", "replace"])
def test_side_lock_inode_attack_cannot_admit_second_process(tmp_path: Path, attack: str) -> None:
    """The data-inode lock survives side-lock deletion or replacement.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    attack:
        Side-lock inode attack to perform while the owner remains live.
    """

    path = tmp_path / "attempts.jsonl"
    probe = "\n".join(
        (
            "import sys",
            "from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION_V3",
            "from menagerie.crawler.recordio import JsonlLedger, SingleWriterError",
            "try:",
            "    JsonlLedger(sys.argv[1], ATTEMPT_SCHEMA_VERSION_V3)",
            "except SingleWriterError:",
            "    raise SystemExit(0)",
            "raise SystemExit(23)",
        )
    )
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(_attempt("m_owner", "attempt-1"))
        lock_path = path.with_suffix(f"{path.suffix}.lock")
        lock_path.unlink()
        if attack == "replace":
            lock_path.write_bytes(b"replacement inode")
        result = subprocess.run(
            [sys.executable, "-c", probe, str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        ledger.append(_attempt("m_owner", "attempt-2"))
    assert [row["ledger_seq"] for row in scan_jsonl(path)] == [1, 2]


def test_recovering_reader_refuses_while_jsonl_writer_is_live(
    tmp_path: Path,
) -> None:
    """The generic recovery reader cannot race a live JsonlLedger writer.

    While the writer is live (it holds the data-inode flock for its whole
    lifetime -- hours on a campaign leg) the reader gets a TYPED refusal, not
    an indefinite block: a "torn" tail under a live writer is an in-flight
    append, not a death. After the writer closes, recovery proceeds and the
    in-flight fragment is evidenced and truncated.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    ledger = JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3)
    ledger.append(_attempt("m_owner", "attempt-1"))
    complete_bytes = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"in_flight":')

    def check_prefix(retained: Path) -> object:
        """Strictly scan a candidate retained prefix."""

        return scan_jsonl(retained)

    with pytest.raises(SingleWriterError):
        read_with_tail_recovery(path, prefix_check=check_prefix)
    assert path.read_bytes() == complete_bytes + b'{"in_flight":', (
        "a refused recovery must not have truncated anything"
    )
    ledger.close()
    assert read_with_tail_recovery(path, prefix_check=check_prefix) == complete_bytes
    assert path.read_bytes() == complete_bytes
