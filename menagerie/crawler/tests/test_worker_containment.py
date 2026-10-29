"""Worker containment tripwires: bounded waits, group teardown, group memory.

BL-12(a)-(d),(h): every wait is bounded and owned, no survivors outlive a
lease, memory containment sees the whole process group, streams never
materialize in driver memory, and v3 completion is a parent-secret-bound file
that model stdout can neither destroy nor forge.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import pytest

import menagerie.crawler.worker as worker_module
from menagerie.crawler.driver_receipts import (
    WORKER_RSS_ABSOLUTE_CEILING_BYTES,
    derive_worker_rss_limit_bytes,
    SupervisedForwardLane,
)
from menagerie.crawler.driver_contracts import DriverIntegrationError
from menagerie.crawler.execution_lock import (
    ExecutionLockTimeout,
    acquire_global_execution_flock,
    read_global_execution_flock_holder,
    release_global_execution_flock,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.policy import detect_os_sandbox
from menagerie.crawler.worker_supervisor import (
    WORKER_COMPLETION_FILE_VERSION,
    WORKER_COMPLETION_PROOF_VERSION,
    WORKER_COMPLETION_FILE_NAME,
    WorkerUnreapableError,
    _reap_worker_root,
    _teardown_group_after_reap,
    _verified_worker_completion,
    _verified_worker_completion_file,
    capture_process_group,
    effective_rss_kill_threshold_bytes,
    run_isolated_subprocess,
    worker_completion_challenge_proof,
)

_SANDBOX = detect_os_sandbox()
needs_sandbox = pytest.mark.skipif(_SANDBOX is None, reason="no OS sandbox on this host")


# ---------------------------------------------------------------------------
# (b) bounded, observable, shutdown-aware global execution flock
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_global_flock_acquire_times_out_typed_and_names_the_holder(tmp_path: Path) -> None:
    """A held flock yields a typed deadline error carrying the holder record."""

    lock_path = tmp_path / "execution.flock"
    holding = acquire_global_execution_flock(
        lock_path,
        timeout_seconds=5,
        holder_metadata={"work_id": "w-holder", "stable_id": "m-holder"},
    )
    try:
        started = time.monotonic()
        with pytest.raises(ExecutionLockTimeout) as caught:
            acquire_global_execution_flock(lock_path, timeout_seconds=0.4, poll_seconds=0.05)
        elapsed = time.monotonic() - started
        assert elapsed < 5.0
        assert caught.value.reason == "deadline"
        assert caught.value.holder is not None
        assert caught.value.holder["work_id"] == "w-holder"
        assert caught.value.holder["pid"] == os.getpid()
        assert "execution-lock-held-by" in str(caught.value)
    finally:
        release_global_execution_flock(holding)


@pytest.mark.smoke
def test_global_flock_acquire_respects_shutdown_event(tmp_path: Path) -> None:
    """An operator shutdown interrupts the wait long before the deadline."""

    lock_path = tmp_path / "execution.flock"
    holding = acquire_global_execution_flock(lock_path, timeout_seconds=5)
    shutdown = threading.Event()
    shutdown.set()
    try:
        started = time.monotonic()
        with pytest.raises(ExecutionLockTimeout) as caught:
            acquire_global_execution_flock(
                lock_path,
                timeout_seconds=30,
                poll_seconds=0.05,
                shutdown_event=shutdown,
            )
        assert time.monotonic() - started < 5.0
        assert caught.value.reason == "shutdown"
    finally:
        release_global_execution_flock(holding)


@pytest.mark.smoke
def test_holder_record_readable_while_lock_free(tmp_path: Path) -> None:
    """The holder record is durable provenance, parseable after release."""

    lock_path = tmp_path / "execution.flock"
    descriptor = acquire_global_execution_flock(
        lock_path, timeout_seconds=5, holder_metadata={"run_id": "r1"}
    )
    release_global_execution_flock(descriptor)
    holder = read_global_execution_flock_holder(lock_path)
    assert holder is not None
    assert holder["run_id"] == "r1"


# ---------------------------------------------------------------------------
# (b) bounded reap with typed host fault
# ---------------------------------------------------------------------------


class _FakeUnreapable:
    """Duck-typed Popen whose wait always times out and never reaps."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: Optional[int] = None

    def wait(self, timeout: Optional[float] = None) -> int:
        raise subprocess.TimeoutExpired(cmd="fake-worker", timeout=timeout or 0.0)

    def poll(self) -> Optional[int]:
        return None


@pytest.mark.smoke
def test_reap_worker_root_is_bounded_and_raises_typed_host_fault() -> None:
    """A never-reapable root raises WorkerUnreapableError instead of blocking."""

    process = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(60)"),
        start_new_session=True,
    )
    group = capture_process_group(process)
    fake = _FakeUnreapable(process.pid)
    fake_group = type(group)(
        process=fake,  # type: ignore[arg-type]
        pid=group.pid,
        pgid=group.pgid,
        isolated=group.isolated,
        start_token=group.start_token,
    )
    try:
        started = time.monotonic()
        with pytest.raises(WorkerUnreapableError) as caught:
            _reap_worker_root(fake, fake_group, grace_seconds=0.2)  # type: ignore[arg-type]
        assert time.monotonic() - started < 10.0
        assert caught.value.pid == process.pid
        assert caught.value.pgid == group.pgid
    finally:
        process.kill()
        process.wait(timeout=10)


@pytest.mark.smoke
def test_reap_worker_root_returns_promptly_for_an_exited_child() -> None:
    """The normal path reaps instantly with no escalation."""

    process = subprocess.Popen((sys.executable, "-c", "pass"), start_new_session=True)
    group = capture_process_group(process)
    assert _reap_worker_root(process, group, grace_seconds=10.0) is None
    assert process.returncode is not None


# ---------------------------------------------------------------------------
# (c) no survivors outlive supervision, even on the normal-exit path
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_post_reap_group_teardown_kills_forked_survivors() -> None:
    """A grandchild that outlives its exiting parent is drained post-reap."""

    from menagerie.crawler.worker_supervisor import _live_process_group_members

    if _live_process_group_members(os.getpgrp()) is None:
        pytest.skip("process-group enumeration unavailable in this confined environment")
    process = subprocess.Popen(
        (
            sys.executable,
            "-c",
            "import os,sys,time\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    time.sleep(60)\n"
            "else:\n"
            "    print(pid, flush=True)\n"
            "    os._exit(0)\n",
        ),
        start_new_session=True,
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    grandchild = int(process.stdout.readline())
    process.wait(timeout=10)
    outcome, survivors = _teardown_group_after_reap(process.pid)
    assert outcome in {"clean", "reaped"}
    assert survivors == ()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(grandchild, signal.SIGKILL)
        pytest.fail("forked grandchild survived the post-reap group teardown")


@needs_sandbox
def test_normal_exit_supervision_reports_group_teardown(tmp_path: Path) -> None:
    """run_isolated_subprocess records a benign teardown outcome on normal exit."""

    observation = run_isolated_subprocess(
        (sys.executable, "-c", "pass"),
        tmp_path / "supervisor",
        timeout_seconds=60,
        rss_limit_bytes=1024**3,
    )
    assert observation.exit_code == 0
    assert observation.group_teardown_outcome in {"clean", "reaped"}
    assert observation.group_survivor_pids == ()


# ---------------------------------------------------------------------------
# (d) memory containment over the whole process group + host-informed cap
# ---------------------------------------------------------------------------


@needs_sandbox
def test_process_group_rss_enforcement_catches_a_forking_model(tmp_path: Path) -> None:
    """A grandchild's allocation trips the cap the root alone would evade.

    The pre-fix sampler watched only the root PID, so a model splitting its
    footprint across fork children bypassed the cap by architecture
    (fable W-F6 / TB-06).
    """

    program = (
        "import os, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    block = bytearray(600 * 1024 * 1024)\n"
        "    block[::4096] = b'x' * len(block[::4096])\n"
        "    time.sleep(30)\n"
        "else:\n"
        "    time.sleep(30)\n"
    )
    observation = run_isolated_subprocess(
        (sys.executable, "-c", program),
        tmp_path / "supervisor",
        timeout_seconds=25,
        rss_limit_bytes=256 * 1024 * 1024,
    )
    assert observation.rss_exceeded is True, (
        f"timed_out={observation.timed_out} peak={observation.peak_rss_bytes} "
        f"basis={observation.peak_rss_basis}"
    )
    assert observation.peak_rss_basis == "process-group-sum"
    # DX-8/RE-8: the kill fires at the soft watermark below the hard cap, and
    # the watermark facts are recorded per attempt.
    threshold = effective_rss_kill_threshold_bytes(256 * 1024 * 1024)
    assert observation.peak_rss_bytes > threshold
    assert observation.rss_kill_threshold_bytes == threshold
    assert observation.rss_overshoot_bytes == observation.peak_rss_bytes - threshold


@needs_sandbox
def test_soft_watermark_kills_before_the_hard_cap(tmp_path: Path) -> None:
    """DX-8/RE-8: a gradual allocator dies at the watermark, never near the cap.

    Near-cap completions become deferrals (the JMT-adopted B3-XQ-4 behavior
    change), and the parent sampler keeps real headroom below the hard cap so
    a fast allocator cannot blow through it into host OOM between samples (the
    suspected 8/23 host-crash mechanism).
    """

    limit = 512 * 1024 * 1024
    threshold = effective_rss_kill_threshold_bytes(limit)
    assert threshold < limit
    program = (
        "import time\n"
        "blocks = []\n"
        "for _ in range(40):\n"
        "    block = bytearray(10 * 1024 * 1024)\n"
        "    block[::4096] = b'x' * len(block[::4096])\n"
        "    blocks.append(block)\n"
        "    time.sleep(0.05)\n"
        "time.sleep(30)\n"
    )
    observation = run_isolated_subprocess(
        (sys.executable, "-c", program),
        tmp_path / "supervisor",
        timeout_seconds=25,
        rss_limit_bytes=limit,
    )
    assert observation.rss_exceeded is True, (
        f"timed_out={observation.timed_out} peak={observation.peak_rss_bytes}"
    )
    assert observation.rss_kill_threshold_bytes == threshold
    assert observation.peak_rss_bytes > threshold
    # The whole point: the kill landed with headroom to spare below the cap.
    assert observation.peak_rss_bytes < limit
    assert observation.rss_overshoot_bytes == observation.peak_rss_bytes - threshold


@pytest.mark.smoke
def test_worker_rss_cap_is_host_informed() -> None:
    """The cap derives from the host figure with headroom, clamped and typed."""

    cap, basis = derive_worker_rss_limit_bytes(64 * 1024**3)
    assert (cap, basis) == (WORKER_RSS_ABSOLUTE_CEILING_BYTES, "absolute-ceiling")
    cap, basis = derive_worker_rss_limit_bytes(16 * 1024**3)
    assert basis in {"host-derived", "absolute-ceiling"}
    assert cap <= 12 * 1024**3
    cap, basis = derive_worker_rss_limit_bytes(8 * 1024**3)
    assert basis == "host-derived"
    assert cap == 4 * 1024**3
    assert cap < 8 * 1024**3
    with pytest.raises(DriverIntegrationError, match="refusing at admission"):
        derive_worker_rss_limit_bytes(2 * 1024**3)


@pytest.mark.smoke
def test_production_lane_derives_and_names_its_cap() -> None:
    """The default lane construction records a basis and a bound below RAM."""

    lane = SupervisedForwardLane()
    assert lane.rss_limit_basis in {"host-derived", "absolute-ceiling"}
    assert lane.rss_limit_bytes <= WORKER_RSS_ABSOLUTE_CEILING_BYTES
    explicit = SupervisedForwardLane(rss_limit_bytes=1024**3)
    assert explicit.rss_limit_basis == "configured"
    assert explicit.rss_limit_bytes == 1024**3


# ---------------------------------------------------------------------------
# (h) streams never materialize whole in driver memory
# ---------------------------------------------------------------------------


@needs_sandbox
def test_stream_facts_are_computed_without_whole_stream_materialization(
    tmp_path: Path,
) -> None:
    """A multi-megabyte stdout yields exact facts and a bounded tail."""

    observation = run_isolated_subprocess(
        (
            sys.executable,
            "-c",
            "import sys\n"
            "sys.stdout.write('x' * (8 * 1024 * 1024))\n"
            "sys.stdout.write('FINAL-TAIL-MARKER')\n",
        ),
        tmp_path / "supervisor",
        timeout_seconds=60,
        rss_limit_bytes=1024**3,
    )
    assert observation.exit_code == 0
    assert observation.stdout_bytes == 8 * 1024 * 1024 + len("FINAL-TAIL-MARKER")
    assert observation.stdout_tail.endswith("FINAL-TAIL-MARKER")
    assert len(observation.stdout_tail) <= 1500


# ---------------------------------------------------------------------------
# (a) completion: parent-secret file channel; stdout cannot destroy or forge
# ---------------------------------------------------------------------------

_V3_PAYLOAD = {
    "raw_award_receipt_sha256": "sha256:" + "1" * 64,
    "request_nonce": "nonce-1",
    "request_sha256": "sha256:" + "2" * 64,
}
_V3_LINE = "MENAGERIE_WORKER_COMPLETION_V3 " + json.dumps(
    _V3_PAYLOAD, sort_keys=True, separators=(",", ":")
)


def _completion_file(tmp_path: Path, challenge: str, line: str = _V3_LINE) -> Path:
    """Write one worker-shaped completion file bound to ``challenge``."""

    path = tmp_path / WORKER_COMPLETION_FILE_NAME
    record = {
        "version": WORKER_COMPLETION_FILE_VERSION,
        "completion_line": line,
        "challenge_proof": worker_completion_challenge_proof(challenge, line),
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


@pytest.mark.smoke
def test_completion_file_verifies_only_with_the_real_challenge(tmp_path: Path) -> None:
    """The parent secret is actually CHECKED: a wrong challenge never verifies."""

    challenge = "a" * 64
    path = _completion_file(tmp_path, challenge)
    verified = _verified_worker_completion_file(
        path,
        challenge,
        request_nonce="nonce-1",
        request_sha256="sha256:" + "2" * 64,
    )
    assert verified is not None
    assert verified["raw_award_receipt_sha256"] == "sha256:" + "1" * 64
    forged = _verified_worker_completion_file(
        path,
        "b" * 64,
        request_nonce="nonce-1",
        request_sha256="sha256:" + "2" * 64,
    )
    assert forged is None


@pytest.mark.smoke
def test_completion_file_without_proof_or_with_foreign_association_refuses(
    tmp_path: Path,
) -> None:
    """A proofless or re-associated record is not a completion."""

    challenge = "a" * 64
    path = tmp_path / WORKER_COMPLETION_FILE_NAME
    path.write_text(
        json.dumps({"version": WORKER_COMPLETION_FILE_VERSION, "completion_line": _V3_LINE}),
        encoding="utf-8",
    )
    assert (
        _verified_worker_completion_file(
            path,
            challenge,
            request_nonce="nonce-1",
            request_sha256="sha256:" + "2" * 64,
        )
        is None
    )
    path = _completion_file(tmp_path, challenge)
    assert (
        _verified_worker_completion_file(
            path,
            challenge,
            request_nonce="another-nonce",
            request_sha256="sha256:" + "2" * 64,
        )
        is None
    )


@pytest.mark.smoke
def test_legacy_stdout_completion_survives_buffered_model_output() -> None:
    """W-F1's reproduced displacement no longer destroys a v1 completion.

    Model print output flushed at interpreter exit lands AFTER the completion
    line; the pre-fix verifier required the completion to be the LAST line of
    a strictly-decodable stream.
    """

    challenge = "c" * 64
    receipt_sha = "sha256:" + "3" * 64
    proof = stable_hash(
        {
            "version": "menagerie.crawler.worker-completion.v1",
            "challenge": challenge,
            "receipt_sha256": receipt_sha,
        }
    )
    line = "MENAGERIE_WORKER_COMPLETION_V1 " + json.dumps(
        {"receipt_sha256": receipt_sha, "proof": proof},
        sort_keys=True,
        separators=(",", ":"),
    )
    displaced = (line + "\nmodel output line\n\xff-partial").encode(
        "utf-8", errors="replace"
    ) + b"\xfe"
    verified = _verified_worker_completion(displaced, challenge)
    assert verified is not None
    assert verified["receipt_sha256"] == receipt_sha


@pytest.mark.smoke
def test_completion_constants_do_not_drift_between_worker_and_supervisor() -> None:
    """The worker's mirrored channel constants match the supervisor's."""

    assert worker_module._WORKER_COMPLETION_FILE_NAME == WORKER_COMPLETION_FILE_NAME
    assert worker_module._WORKER_COMPLETION_FILE_VERSION == WORKER_COMPLETION_FILE_VERSION
    assert worker_module._WORKER_COMPLETION_PROOF_VERSION == WORKER_COMPLETION_PROOF_VERSION


@pytest.mark.smoke
def test_worker_main_writes_the_completion_file_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """worker.main() persists a proof-bound completion file for a v3 success."""

    challenge = "d" * 64
    line = _V3_LINE

    def fake_run_worker(request: Any) -> dict[str, Any]:
        return {"raw_award_receipt": dict(_V3_PAYLOAD)}

    class FakeRequest:
        protocol_version = "menagerie.crawler.worker-request.v3"
        receipt_path = tmp_path / "result" / "receipt.json"

    monkeypatch.setenv("MENAGERIE_PARENT_COMPLETION_CHALLENGE", challenge)
    monkeypatch.setattr(worker_module, "run_worker", fake_run_worker)
    monkeypatch.setattr(worker_module, "_start_parent_watchdog", lambda: None)
    monkeypatch.setattr(
        worker_module.WorkerRequest, "from_mapping", classmethod(lambda *a, **k: FakeRequest())
    )
    monkeypatch.setattr(
        worker_module,
        "completion_line_for_raw_award_receipt",
        lambda raw: line,
    )
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    FakeRequest.receipt_path.parent.mkdir(parents=True, exist_ok=True)
    exit_code = worker_module.main(
        ["--request", str(request_path), "--receipt", str(FakeRequest.receipt_path)]
    )
    assert exit_code == 0
    completion_path = FakeRequest.receipt_path.parent / WORKER_COMPLETION_FILE_NAME
    verified = _verified_worker_completion_file(
        completion_path,
        challenge,
        request_nonce="nonce-1",
        request_sha256="sha256:" + "2" * 64,
    )
    assert verified is not None
    assert verified["completion_line"] == line
