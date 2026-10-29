"""Queue-lane completion binds to the claim, never the current descriptor (T1-02).

The production author lane retries a stalled job under the same ``job_id`` with a fresh
``attempt_nonce``. A stale manager that completes after such a rotation must be refused:
rebuilding the job from the *current* pending descriptor would stamp the new attempt's
nonce onto the old attempt's result, which the lane then commits as the new attempt's
answer. The lease written at claim time is the only record of which attempt this
servicer actually holds, so ``complete``/``fail`` must verify it before stamping.

The second mechanism is the wall clock: a batch round that outlives the grant wall (for
example under a provider usage pause) must not wall-fail jobs whose author sessions
finished well under grant. Wall consumption is claim-to-publication, never
claim-to-serial-complete.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pytest

from menagerie.crawler.author_pool import (
    AuthorPool,
    AuthorPoolError,
    main as author_pool_main,
)
from menagerie.crawler.author_queue import (
    AUTHOR_JOB_VERSION,
    DEFAULT_EFFORT_GRANT,
    EffortGrant,
    QueueJob,
    job_paths,
    read_json,
    write_json_atomic,
)
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.tools import pool_batch

START = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)


def _write_request(root: Path, payload: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "request.json"
    path.write_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return path


def _enqueue(
    queue_root: Path,
    *,
    nonce: str = "nonce-A",
    grant: Optional[EffortGrant] = None,
    job_id: str = "author-fixture",
) -> QueueJob:
    """Publish one pending author job exactly as the lane would."""

    work_root = queue_root.parent / "work" / job_id
    output_path = work_root / "result.json"
    body = {
        "envelope_version": "menagerie.crawler.author-envelope.v3",
        "work_id": "work-m1",
        "stable_id": "m1",
        "campaign_id": "campaign-m1",
        "required_output_path": str(output_path),
    }
    request_path = _write_request(work_root, body)
    paths = job_paths(queue_root, job_id)
    descriptor = {
        "envelope_version": AUTHOR_JOB_VERSION,
        "job_id": job_id,
        "attempt_nonce": nonce,
        "kind": "author",
        "stable_id": "m1",
        "work_id": "work-m1",
        "author_model": None,
        "campaign_id": "campaign-m1",
        "tier_campaign_id": "c1-mech",
        "request_path": str(request_path.resolve()),
        "request_sha256": hash_bytes(request_path.read_bytes()),
        "required_output_path": str(output_path.resolve()),
        "receipt_path": str(paths["receipt"]),
        "backoff_path": str(paths["backoff"]),
        "failure_path": str(paths["failure"]),
        "claim_path": str(paths["claim"]),
        "effort_grant": (grant or DEFAULT_EFFORT_GRANT).to_dict(),
        "stall_timeout_seconds": 2700.0,
        "enqueued_at": "2026-07-27T12:00:00Z",
    }
    from menagerie.crawler.author_queue import ensure_author_queue

    ensure_author_queue(queue_root)
    write_json_atomic(paths["pending"], descriptor)
    return QueueJob.from_mapping(descriptor)


def _rotate(queue_root: Path, job: QueueJob, *, nonce: str) -> QueueJob:
    """Re-publish the pending descriptor under a fresh attempt nonce, as a lane retry does."""

    payload = read_json(job_paths(queue_root, job.job_id)["pending"])
    assert payload is not None
    payload["attempt_nonce"] = nonce
    write_json_atomic(job_paths(queue_root, job.job_id)["pending"], payload)
    return QueueJob.from_mapping(payload)


def _publish_result(job: QueueJob, *, payload: Optional[dict[str, Any]] = None) -> None:
    job.required_output_path.parent.mkdir(parents=True, exist_ok=True)
    job.required_output_path.write_text(json.dumps(payload or {"ok": True}), encoding="utf-8")


def _mutable_clock(start: datetime) -> tuple[dict[str, datetime], Any]:
    state = {"now": start}
    return state, (lambda: state["now"])


# -- S2 repro (a): completion after a descriptor rotation is refused --------


def test_cli_complete_after_rotation_refuses_and_stamps_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stale manager's complete must not stamp the rotated attempt's nonce.

    Claim under nonce A, rotate the descriptor to nonce B (lane retry), then run the
    operator ``complete``. Today the CLI rebuilds the job from the current descriptor,
    so the receipt stamps B and passes the lane's nonce match by construction -- the old
    result is committed as the new attempt's answer.
    """

    queue = tmp_path / "queue"
    job = _enqueue(queue, nonce="nonce-A")
    pool = AuthorPool(queue, owner="session-1")
    claimed = pool.claim(job.job_id)
    _publish_result(job)
    rotated = _rotate(queue, job, nonce="nonce-B")

    rc = author_pool_main(
        [
            "--queue",
            str(queue),
            "--owner",
            "session-1",
            "complete",
            "--job",
            job.job_id,
            "--claimed-at",
            claimed.claimed_at.isoformat(),
            "--tool-calls",
            "3",
        ]
    )
    capsys.readouterr()
    assert rc != 0, "completion after rotation must be refused"
    receipt = read_json(rotated.receipt_path)
    assert receipt is None, f"no receipt may be stamped after rotation, got {receipt}"
    assert read_json(rotated.failure_path) is None, (
        "a rotation refusal must not stamp a failure sidecar with the new nonce either"
    )


def test_pool_complete_verifies_the_lease_nonce(tmp_path: Path) -> None:
    """The in-process path refuses too: the lease, not the ClaimedJob, is authority."""

    queue = tmp_path / "queue"
    job = _enqueue(queue, nonce="nonce-A")
    pool = AuthorPool(queue, owner="session-1")
    claimed = pool.claim(job.job_id)
    _publish_result(job)
    _rotate(queue, job, nonce="nonce-B")
    with pytest.raises(AuthorPoolError, match="supersed"):
        pool.complete(claimed, tool_calls=1)
    assert read_json(job.receipt_path) is None


def test_fail_verifies_the_lease_before_stamping(tmp_path: Path) -> None:
    """A typed failure is an answer; answering a rotated attempt is refused the same way."""

    queue = tmp_path / "queue"
    job = _enqueue(queue, nonce="nonce-A")
    pool = AuthorPool(queue, owner="session-1")
    pool.claim(job.job_id)
    rotated = _rotate(queue, job, nonce="nonce-B")
    with pytest.raises(AuthorPoolError, match="supersed"):
        pool.fail(rotated, reason="subagent-transient", retryable=True)
    assert read_json(rotated.failure_path) is None


def test_fail_without_any_lease_is_refused(tmp_path: Path) -> None:
    """Stamping requires holding the claim; an unclaimed job cannot be answered."""

    queue = tmp_path / "queue"
    job = _enqueue(queue)
    pool = AuthorPool(queue, owner="session-1")
    with pytest.raises(AuthorPoolError, match="lease"):
        pool.fail(job, reason="subagent-transient", retryable=True)
    assert read_json(job.failure_path) is None


def test_complete_under_a_foreign_lease_is_refused(tmp_path: Path) -> None:
    """A lease force-taken by another servicer revokes this session's right to answer."""

    queue = tmp_path / "queue"
    job = _enqueue(queue)
    pool = AuthorPool(queue, owner="session-1")
    claimed = pool.claim(job.job_id)
    AuthorPool(queue, owner="session-2").claim(job.job_id, force=True)
    _publish_result(job)
    with pytest.raises(AuthorPoolError, match="session-2"):
        pool.complete(claimed, tool_calls=1)
    assert read_json(job.receipt_path) is None


def test_cli_complete_claimed_at_must_match_the_lease(tmp_path: Path) -> None:
    """An echoed --claimed-at that contradicts the lease is an operator error, not data."""

    queue = tmp_path / "queue"
    job = _enqueue(queue)
    pool = AuthorPool(queue, owner="session-1")
    pool.claim(job.job_id)
    _publish_result(job)
    rc = author_pool_main(
        [
            "--queue",
            str(queue),
            "--owner",
            "session-1",
            "complete",
            "--job",
            job.job_id,
            "--claimed-at",
            "2026-07-27T09:00:00Z",
            "--tool-calls",
            "1",
        ]
    )
    assert rc != 0
    assert read_json(job.receipt_path) is None


def test_cli_complete_derives_the_claim_from_the_lease(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--claimed-at is a cross-check now; the lease alone is enough to complete."""

    queue = tmp_path / "queue"
    job = _enqueue(queue)
    pool = AuthorPool(queue, owner="session-1")
    pool.claim(job.job_id)
    _publish_result(job)
    rc = author_pool_main(
        [
            "--queue",
            str(queue),
            "--owner",
            "session-1",
            "complete",
            "--job",
            job.job_id,
            "--tool-calls",
            "1",
        ]
    )
    capsys.readouterr()
    assert rc == 0
    receipt = read_json(job.receipt_path)
    assert receipt is not None
    assert receipt["attempt_nonce"] == "nonce-A"


# -- S2 repro (b): per-job wall clock, not batch-serial claimed_at ----------


def test_wall_is_claim_to_publication_not_claim_to_serial_complete(tmp_path: Path) -> None:
    """A finished-under-grant job survives a round that outlives the grant wall.

    The author session published its result 120s after claim, well under the 1800s
    grant. The batch loop only gets to `complete` 35 minutes later (provider pause).
    Today wall = now - claimed_at = 2100s > 1800s and the healthy attempt is failed
    permanently with effort-cap-exhausted.
    """

    queue = tmp_path / "queue"
    job = _enqueue(queue, grant=EffortGrant(200, 20, 1800.0))
    state, clock = _mutable_clock(START)
    pool = AuthorPool(queue, owner="session-1", clock=clock)
    claimed = pool.claim(job.job_id)
    _publish_result(job)
    published = START + timedelta(seconds=120)
    os.utime(job.required_output_path, times=(published.timestamp(), published.timestamp()))

    state["now"] = START + timedelta(minutes=35)
    receipt = pool.complete(claimed, tool_calls=5)
    assert receipt["consumption"]["wall_seconds"] == pytest.approx(120.0)
    assert read_json(job.failure_path) is None


def test_wall_never_exceeds_elapsed_time_since_claim(tmp_path: Path) -> None:
    """A publication instant in the clock's future is clamped to elapsed, never trusted."""

    queue = tmp_path / "queue"
    job = _enqueue(queue)
    state, clock = _mutable_clock(START)
    pool = AuthorPool(queue, owner="session-1", clock=clock)
    claimed = pool.claim(job.job_id)
    state["now"] = START + timedelta(seconds=93)
    _publish_result(job)  # real mtime is ~2026-08, far past the fake clock
    receipt = pool.complete(claimed, tool_calls=1)
    assert receipt["consumption"]["wall_seconds"] == pytest.approx(93.0)


def test_a_genuinely_over_wall_session_still_fails_typed(tmp_path: Path) -> None:
    """The honest wall still enforces the grant: a late publication is a real overrun."""

    queue = tmp_path / "queue"
    job = _enqueue(queue, grant=EffortGrant(200, 20, 60.0))
    state, clock = _mutable_clock(START)
    pool = AuthorPool(queue, owner="session-1", clock=clock)
    claimed = pool.claim(job.job_id)
    _publish_result(job)
    published = START + timedelta(seconds=400)
    os.utime(job.required_output_path, times=(published.timestamp(), published.timestamp()))
    state["now"] = START + timedelta(seconds=500)
    with pytest.raises(AuthorPoolError, match="exceeded its grant"):
        pool.complete(claimed, tool_calls=1)
    failure = read_json(job.failure_path)
    assert failure is not None
    assert failure["reason"] == "effort-cap-exhausted"


def test_renew_preserves_the_original_claim_instant(tmp_path: Path) -> None:
    """Renewal extends liveness; it must not restart the wall clock."""

    queue = tmp_path / "queue"
    job = _enqueue(queue)
    state, clock = _mutable_clock(START)
    pool = AuthorPool(queue, owner="session-1", clock=clock)
    pool.claim(job.job_id)
    lease_before = read_json(job.claim_path)
    assert lease_before is not None
    state["now"] = START + timedelta(seconds=300)
    pool.renew(job.job_id)
    lease_after = read_json(job.claim_path)
    assert lease_after is not None
    assert lease_after["claimed_at"] == lease_before["claimed_at"]


def test_renew_refuses_a_foreign_lease(tmp_path: Path) -> None:
    """Renewing over another servicer's lease would silently flip its owner."""

    queue = tmp_path / "queue"
    job = _enqueue(queue)
    AuthorPool(queue, owner="session-a").claim(job.job_id)
    with pytest.raises(AuthorPoolError, match="session-a"):
        AuthorPool(queue, owner="session-b").renew(job.job_id)


def test_cli_claim_prints_the_owner_to_echo_at_complete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Complete runs under the claiming identity, so claim must say what that is."""

    queue = tmp_path / "queue"
    job = _enqueue(queue)
    rc = author_pool_main(
        ["--queue", str(queue), "--owner", "session-1", "claim", "--job", job.job_id]
    )
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["owner"] == "session-1"


def test_the_lease_records_the_claim_instant_the_pool_measured(tmp_path: Path) -> None:
    """The lease claimed_at and the printed claimed_at are one instant, exactly."""

    queue = tmp_path / "queue"
    job = _enqueue(queue)
    pool = AuthorPool(queue, owner="session-1", clock=lambda: START)
    claimed = pool.claim(job.job_id)
    lease = read_json(job.claim_path)
    assert lease is not None
    assert lease["claimed_at"] == "2026-07-27T12:00:00Z"
    assert claimed.claimed_at == START


# -- release requires an owner check ----------------------------------------


def test_release_refuses_a_foreign_lease(tmp_path: Path) -> None:
    """Dropping someone else's live lease would hand their job to a second dispatcher."""

    queue = tmp_path / "queue"
    job = _enqueue(queue)
    AuthorPool(queue, owner="session-a").claim(job.job_id)
    other = AuthorPool(queue, owner="session-b")
    with pytest.raises(AuthorPoolError, match="session-a"):
        other.release(job.job_id)
    assert job.claim_path.is_file(), "the foreign lease must survive the refused release"


def test_release_by_the_owner_removes_the_lease(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    job = _enqueue(queue)
    pool = AuthorPool(queue, owner="session-a")
    pool.claim(job.job_id)
    pool.release(job.job_id)
    assert not job.claim_path.is_file()


def test_release_without_a_lease_is_a_no_op(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    job = _enqueue(queue)
    AuthorPool(queue, owner="session-a").release(job.job_id)
    assert not job.claim_path.is_file()


# -- pool_batch: identity-derived owner, manifest binding, digest check -----


def _batch(
    queue: Path, repo_root: Path, *args: str, owner: Optional[str] = None
) -> tuple[int, str, str]:
    """Run pool_batch main() in-process with stdout/stderr captured."""

    argv = ["--queue", str(queue), "--repo-root", str(repo_root), "--python", sys.executable]
    if owner is not None:
        argv += ["--owner", owner]
    argv += ["--campaign", "c1-mech", *args]
    import contextlib
    import io

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = pool_batch.main(argv)
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture(name="repo_root")
def _repo_root_fixture() -> Path:
    root = Path(__file__).resolve().parents[3]
    assert (root / "menagerie" / "crawler" / "author_pool.py").is_file()
    return root


def _subprocess_env_ok(repo_root: Path) -> bool:
    """pool_batch shells out to `python -m menagerie.crawler.author_pool`."""

    probe = subprocess.run(
        [sys.executable, "-c", "import menagerie.crawler.author_pool"],
        cwd=repo_root,
        capture_output=True,
    )
    return probe.returncode == 0


@pytest.mark.heavy
def test_batch_round_survives_outliving_the_grant_wall(tmp_path: Path, repo_root: Path) -> None:
    """Pure-tool S2 (b): claim, publish under grant, complete 35 minutes later.

    The claim and the completion are separate pool_batch invocations (separate
    processes), exactly as the runbook drives them. The manifest, not a shared
    magic owner string, must carry the claiming identity to the completion.
    """

    if not _subprocess_env_ok(repo_root):
        pytest.skip("menagerie is not importable from the repo root in a subprocess")
    queue = tmp_path / "queue"
    job = _enqueue(queue, grant=EffortGrant(200, 20, 1800.0))
    out_dir = tmp_path / "round"

    rc, stdout, stderr = _batch(queue, repo_root, "claim", "--count", "1", "--out", str(out_dir))
    assert rc == 0, stderr
    manifest_path = out_dir / "manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(rows) == 1

    # The author session finishes 120s after claim; the round then outlives the
    # 1800s grant wall before the serial complete runs. Simulate the passage of
    # time by backdating the claim 35 minutes and pinning the publication mtime
    # 120s after it.
    lease = read_json(job.claim_path)
    assert lease is not None
    backdated = datetime.now(timezone.utc) - timedelta(minutes=35)
    backdated_iso = backdated.isoformat().replace("+00:00", "Z")
    lease["claimed_at"] = backdated_iso
    write_json_atomic(job.claim_path, lease)
    rows[0]["claimed_at"] = backdated_iso
    manifest_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _publish_result(job)
    published = (backdated + timedelta(seconds=120)).timestamp()
    os.utime(job.required_output_path, times=(published, published))

    counts = tmp_path / "counts.json"
    counts.write_text(json.dumps({job.job_id: {"tool_calls": 4}}), encoding="utf-8")
    rc, stdout, stderr = _batch(
        queue,
        repo_root,
        "complete",
        "--manifest",
        str(manifest_path),
        "--counts",
        str(counts),
    )
    assert rc == 0, f"healthy finished job wall-failed: {stderr}"
    receipt = read_json(job.receipt_path)
    assert receipt is not None
    assert receipt["consumption"]["wall_seconds"] == pytest.approx(120.0, abs=5.0)
    assert read_json(job.failure_path) is None


@pytest.mark.heavy
def test_batch_owner_is_identity_derived_and_recorded_in_the_manifest(
    tmp_path: Path, repo_root: Path
) -> None:
    """No shared magic owner: two sessions must not look like one servicer."""

    if not _subprocess_env_ok(repo_root):
        pytest.skip("menagerie is not importable from the repo root in a subprocess")
    queue = tmp_path / "queue"
    job = _enqueue(queue)
    out_dir = tmp_path / "round"
    rc, stdout, stderr = _batch(queue, repo_root, "claim", "--count", "1", "--out", str(out_dir))
    assert rc == 0, stderr
    rows = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    owner = rows[0].get("owner")
    assert owner, "the manifest must record the claiming identity"
    assert owner != "pool-manager", "the owner must not be a shared constant"
    lease = read_json(job.claim_path)
    assert lease is not None
    assert lease["owner"] == owner


@pytest.mark.heavy
def test_batch_complete_verifies_the_result_digest_against_the_claimed_row(
    tmp_path: Path, repo_root: Path
) -> None:
    """A manifest row answers exactly the result it dispatched, byte-verified."""

    if not _subprocess_env_ok(repo_root):
        pytest.skip("menagerie is not importable from the repo root in a subprocess")
    queue = tmp_path / "queue"
    job = _enqueue(queue)
    out_dir = tmp_path / "round"
    rc, stdout, stderr = _batch(queue, repo_root, "claim", "--count", "1", "--out", str(out_dir))
    assert rc == 0, stderr
    manifest_path = out_dir / "manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    _publish_result(job)

    # A stale manifest row pointing at a different result file must be refused,
    # even though the pool's own complete succeeds against the descriptor.
    decoy = tmp_path / "decoy-result.json"
    decoy.write_text(json.dumps({"ok": "decoy"}), encoding="utf-8")
    rows[0]["required_output_path"] = str(decoy)
    manifest_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    counts = tmp_path / "counts.json"
    counts.write_text(json.dumps({job.job_id: {"tool_calls": 1}}), encoding="utf-8")
    rc, stdout, stderr = _batch(
        queue,
        repo_root,
        "complete",
        "--manifest",
        str(manifest_path),
        "--counts",
        str(counts),
    )
    assert rc != 0, "a digest/path mismatch against the claimed row must fail the round"
    assert "digest" in (stdout + stderr).lower() or "mismatch" in (stdout + stderr).lower()
