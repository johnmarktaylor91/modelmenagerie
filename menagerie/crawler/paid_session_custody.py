"""CONCERN B -- paid-session CUSTODY: kernel primitives for the paid subtree.

This module answers exactly two questions and imports no money symbol to answer
either: *is the paid subtree still alive?* and *does a credential-bearing
pathname exist?*  It holds no budget, no policy, and no durable row schema, so a
custody change cannot alter a charge and an accounting change cannot weaken
custody.

One authoritative kernel oracle plus one additive one
-----------------------------------------------------
The paid subtree is the whole tree below the driver's guard -- the wrapper AND
the grandchild that actually holds the provider credential and does the buying.
Neither available Darwin oracle covers it alone:

``flock`` (AUTHORITATIVE)
    An inherited exclusive lock on a lease-named file.  It survives the parent's
    death, needs no custodian, trusts no PID, and is answerable by any process
    at any time.  A descendant that closes the descriptor (a wrapper spawning
    with ``close_fds``) drops out of it.  A credential-bearing descendant cannot:
    the credential and this descriptor are handed over in ONE ``pass_fds``.
process-group membership (ADDITIVE)
    The driver spawns the guard with ``start_new_session``, so every descendant
    is in the guard's group unless it calls ``setsid``/``setpgid``.  Enumerating
    the group catches exactly the descendant the ``flock`` misses.

The asymmetry is deliberate.  The flock must be FREE before custody is released;
the group probe can only ever ADD a refusal.  The membership enumeration is not
available on every host (it shells out to ``ps``), and treating its silence as
blocking would refuse every clean paid session -- the brick this design exists to
remove.  Where it is unavailable, the flock stands alone and the residual widens
accordingly.

Escaping the flock requires closing the inherited descriptor, which also drops
the credential; escaping both oracles additionally requires leaving the process
group.  Darwin -- with no ``PR_SET_PDEATHSIG``, no pidfd, and no cgroups -- offers
no way to prevent that.  The ONE residual is disclosed by
:data:`CUSTODY_RESIDUAL`, and the honest invariant is the ruling's: no
*undisclosed*, unbounded, credential-bearing orphan.

Nothing a producer writes is ever evidence
------------------------------------------
There is deliberately no self-certification primitive here.  A token a wrapper
writes about itself is evidence about the caller, never about the caller's
descendants, and a positive outcome derived from it is exactly the
producer-supplied field the waiver doctrine forbids.  ``verified-exit`` is
therefore a MACHINE observation, or it is not granted.

This module also owns NO raw process-group targeting.  Killing a group requires
verifying it first, and after the guard is reaped its start token is gone, so
the only routines that may signal are the supervisor's hardened ones, called
while the leader is still alive and verifiable.
"""

from __future__ import annotations

import errno
import fcntl
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from menagerie.crawler.worker_supervisor import (
    # The supervisor's already-reviewed group-membership enumeration and start
    # token reader, reused rather than re-derived: this module deliberately owns
    # NO raw process-group targeting of its own.
    _live_process_group_members as live_process_group_members,
    process_start_token,
)

CUSTODY_FD_ENV = "MENAGERIE_PAID_SUBTREE_LOCK_FD"

#: Kernel liveness answers.  ``UNANSWERABLE`` is never read as permission: the
#: lock pathname is gone or names a different inode, so the oracle has no
#: opinion and the caller must fail closed.
ALIVE = "alive"
FREE = "free"
UNANSWERABLE = "unanswerable"
LIVENESS_ANSWERS = frozenset({ALIVE, FREE, UNANSWERABLE})

#: The CLOSED set of restart actions :func:`custody_recovery_decision` may return.
#: A consumer branches on these values and nothing else, so a new disposition
#: cannot appear without a reviewed change here and in every caller's ladder.
CUSTODY_ACTIONS = frozenset(
    {
        "already-dead",
        "stale-boot",
        "reap-verified-group",
        "active-refuse",
        "unreapable-refuse",
    }
)

#: The custody topology matrix every RED tripwire must cover.  Widening this set
#: is a new claim about what the kernel gives us and needs its own reviewed
#: change; ``paid_grandchild_credential`` is the case rounds 1-3 kept missing.
CUSTODY_TOPOLOGIES = frozenset(
    {
        "driver_sigkill",
        "operator_different_group",
        "mid_session_reboot",
        "wrapper_hang",
        "paid_grandchild_credential",
    }
)

#: The single irreducible Darwin residual, stated in full so no consumer can
#: quietly claim more than the kernel offers.
CUSTODY_RESIDUAL = (
    "Darwin offers no PR_SET_PDEATHSIG, no pidfd, and no cgroup, so a descendant "
    "that both closes the inherited custody descriptor and leaves the paid process "
    "group cannot be observed. Such an orphan stays financially bounded by its one "
    "reservation (it cannot reach the writer to buy again), holds no durable "
    "credential pathname, and is reaped on restart; it is disclosed, never claimed "
    "to be impossible."
)


@dataclass
class SubtreeLockHandle:
    """Parent-side handle for one inherited paid-subtree liveness lock."""

    path: Path
    fd: int
    device: int
    inode: int

    def close_parent_copy(self) -> None:
        """Close the parent's copy while leaving inheriting descendants authoritative."""

        if self.fd < 0:
            return
        os.close(self.fd)
        self.fd = -1


@dataclass(frozen=True)
class AnonymousCredential:
    """An unlinked credential descriptor exposed through ``/dev/fd``."""

    fd: int
    path: str


@dataclass(frozen=True)
class PaidCustodyLease:
    """Durable nonsecret identity for one financially bounded paid subtree."""

    lease_id: str
    spend_id: str
    boot_id: str
    host_id: str
    lock_path: Path
    leader_pid: Optional[int]
    leader_start_token: Optional[str]
    leader_pgid: Optional[int]
    deadline_at: str
    reserved_wall_seconds: float
    lock_device: Optional[int] = None
    lock_inode: Optional[int] = None

    @property
    def is_bound(self) -> bool:
        """Return whether a post-spawn custody binding named a leader."""

        return (
            self.leader_pid is not None
            or self.leader_start_token is not None
            or self.leader_pgid is not None
        )

    @property
    def lock_identity(self) -> Optional[tuple[int, int]]:
        """Return the bound lock inode identity, when the binding recorded one."""

        if self.lock_device is None or self.lock_inode is None:
            return None
        return (self.lock_device, self.lock_inode)


@dataclass(frozen=True)
class CustodyRecoveryDecision:
    """Pure restart decision derived from a lease and kernel observations."""

    action: str
    detail: str
    financially_bounded: bool
    disclosed: bool
    credential_path: None = None

    def __post_init__(self) -> None:
        """Refuse any restart action outside the closed set."""

        if self.action not in CUSTODY_ACTIONS:
            raise ValueError(f"custody recovery action {self.action!r} is outside the closed set")


def open_subtree_lock(path: Path) -> SubtreeLockHandle:
    """Acquire and return one inheritable exclusive paid-subtree lock.

    Parameters
    ----------
    path:
        Nonsecret durable lock pathname named by the paid-session lease.

    Returns
    -------
    SubtreeLockHandle
        Locked descriptor ready for ``pass_fds`` inheritance, carrying the inode
        identity a later liveness probe must match.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    os.set_inheritable(descriptor, True)
    return SubtreeLockHandle(
        path=path, fd=descriptor, device=status.st_dev, inode=status.st_ino
    )


def subtree_liveness(
    path: Path, *, expected_identity: Optional[tuple[int, int]] = None
) -> str:
    """Return the kernel's answer about one paid subtree's liveness.

    The pathname is opened WITHOUT ``O_CREAT``: a live subtree holds its
    ``flock`` on the *inode*, so minting a fresh empty inode for a missing
    pathname would let one ``rm`` make the oracle report a live subtree dead.
    A missing pathname, or one that now names a different inode than the
    binding recorded, is :data:`UNANSWERABLE` -- never :data:`FREE`.

    Parameters
    ----------
    path:
        Lease-named liveness lock.
    expected_identity:
        ``(st_dev, st_ino)`` the durable custody binding recorded, when bound.

    Returns
    -------
    str
        One of :data:`ALIVE`, :data:`FREE`, or :data:`UNANSWERABLE`.
    """

    try:
        descriptor = os.open(path, os.O_RDWR)
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ENOTDIR}:
            return UNANSWERABLE
        raise
    try:
        status = os.fstat(descriptor)
        if expected_identity is not None and (status.st_dev, status.st_ino) != expected_identity:
            return UNANSWERABLE
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return ALIVE
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return FREE
    finally:
        os.close(descriptor)


def subtree_is_alive(
    path: Path, *, expected_identity: Optional[tuple[int, int]] = None
) -> bool:
    """Return whether the subtree must be treated as alive, failing closed.

    Parameters
    ----------
    path:
        Lease-named liveness lock.
    expected_identity:
        ``(st_dev, st_ino)`` recorded at bind time, when available.

    Returns
    -------
    bool
        ``True`` when the kernel says alive OR cannot answer.
    """

    return subtree_liveness(path, expected_identity=expected_identity) != FREE


def process_group_liveness(pgid: Optional[int]) -> str:
    """Return whether any member of one paid process group still exists.

    This reuses the supervisor's already-hardened MEMBERSHIP enumeration rather
    than a raw ``killpg`` probe.  Membership is the stronger evidence: POSIX
    forbids reusing a process-group ID while any process keeps it as its group,
    so a non-empty enumeration proves the ID still names OUR paid subtree and
    cannot have been recycled onto someone else's.  The enumerator's own
    "cannot be proven either way" answer becomes :data:`UNANSWERABLE`, which is
    never read as death.

    Parameters
    ----------
    pgid:
        Process-group id the driver minted with ``start_new_session``, or
        ``None`` when no group was ever bound.

    Returns
    -------
    str
        :data:`ALIVE`, :data:`FREE`, or :data:`UNANSWERABLE`.
    """

    if pgid is None or pgid <= 1:
        return FREE
    members = live_process_group_members(pgid)
    if members is None:
        return UNANSWERABLE
    return ALIVE if members else FREE


def custodian_provably_dead(liveness: str, group_liveness: str) -> bool:
    """Return whether one bound paid custodian is machine-provably dead.

    This is the custody deadline reaper's evidence bar, STRICTER than the
    restart ladder's ``already-dead`` arm: BOTH independent kernel signals must
    positively read dead.  The ``flock`` must be ``FREE`` (the authoritative
    oracle: every credential-bearing descendant inherited the descriptor) AND
    the process group must be enumerably EMPTY.  ``UNANSWERABLE`` on either
    side -- a deleted or replaced lock inode, a host that cannot enumerate
    process groups -- is silence, and silence is never proof of death: the
    lease stays disclosed instead of settled.  Time is deliberately not an
    input here; a live custodian is never reaped on time alone.

    Parameters
    ----------
    liveness:
        :func:`subtree_liveness` answer for the lease's ``flock``.
    group_liveness:
        :func:`process_group_liveness` answer for the lease's process group.

    Returns
    -------
    bool
        ``True`` only when both oracles positively answer ``FREE``.

    Raises
    ------
    ValueError
        If either answer is outside the closed liveness vocabulary.
    """

    if liveness not in LIVENESS_ANSWERS or group_liveness not in LIVENESS_ANSWERS:
        raise ValueError("custody liveness answers are outside the closed set")
    return liveness == FREE and group_liveness == FREE


def leader_is_provably_gone(lease: PaidCustodyLease) -> bool:
    """Return whether the leased subtree leader is machine-provably dead.

    This is the ONLY evidence an operator drain may rely on when the ``flock``
    oracle is unanswerable: the recorded leader pid either no longer exists or
    no longer carries the start token the binding pinned, and its process group
    holds no member.  A live orphan therefore cannot be drained by deleting its
    lock file.

    Parameters
    ----------
    lease:
        Durable nonsecret lease.

    Returns
    -------
    bool
        ``True`` only when nothing bound to the lease can still be running.
    """

    if not lease.is_bound:
        return True
    if process_group_liveness(lease.leader_pgid) == ALIVE:
        return False
    if lease.leader_pid is None:
        return False
    try:
        os.kill(lease.leader_pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    observed_token = process_start_token(lease.leader_pid)
    if observed_token is None:
        # Absence of evidence is never proof of death.  An UNREADABLE start token
        # (a ``ps`` failure, a permission refusal, a host without either probe)
        # leaves a live orphan indistinguishable from a recycled pid, and this is
        # the only evidence an operator drain may rely on -- so it must read as
        # "cannot prove gone", exactly as the supervisor's own group-teardown
        # ownership proof does (PSLRE4 C-O1).
        return False
    return observed_token != lease.leader_start_token


@contextmanager
def anonymous_credential(payload: bytes, *, directory: Path) -> Iterator[AnonymousCredential]:
    """Yield credential bytes through an unlinked descriptor.

    The temporary pathname is unlinked while the file is still empty.  Secret
    bytes are written only after unlink, so there is no instant at which a
    pathname names credential material, and a SIGKILL of any descendant leaves
    nothing on disk to remove.

    Parameters
    ----------
    payload:
        Exact provider credential configuration bytes.
    directory:
        Directory in which to allocate the anonymous inode.

    Yields
    ------
    AnonymousCredential
        Readable descriptor and its ``/dev/fd`` spelling.
    """

    directory.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=".paid-credential-empty-", dir=directory)
    path = Path(raw_path)
    try:
        path.unlink()
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(descriptor, view[written:])
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.set_inheritable(descriptor, True)
        yield AnonymousCredential(fd=descriptor, path=f"/dev/fd/{descriptor}")
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def inherited_custody_fds() -> tuple[int, ...]:
    """Return the verified inherited paid-subtree lock descriptor, if present.

    Returns
    -------
    tuple[int, ...]
        Empty outside paid custody, otherwise the one live descriptor.
    """

    raw = os.environ.get(CUSTODY_FD_ENV)
    if raw is None:
        return ()
    try:
        descriptor = int(raw)
        os.fstat(descriptor)
    except (OSError, TypeError, ValueError):
        return ()
    return (descriptor,)


def custody_recovery_decision(
    lease: PaidCustodyLease,
    *,
    current_boot_id: str,
    liveness: str,
    group_liveness: str,
    identity_matches: bool,
    past_deadline: bool,
) -> CustodyRecoveryDecision:
    """Derive the fail-closed restart action for one paid lease.

    Parameters
    ----------
    lease:
        Durable nonsecret lease bound before provider exposure.
    current_boot_id:
        Machine-observed boot identity at reconciliation.
    liveness:
        :func:`subtree_liveness` answer for the lease's ``flock``.
    group_liveness:
        :func:`process_group_liveness` answer for the lease's process group.
    identity_matches:
        Whether PID, start token, and PGID all still identify the leased leader.
    past_deadline:
        Whether the lease's machine-owned wall deadline has expired.

    Returns
    -------
    CustodyRecoveryDecision
        Closed action.  Every outcome is disclosed and financially bounded; an
        unverifiable live identity is the named Darwin residual, refusing new
        paid admission until an operator drain resolves it.
    """

    if liveness not in LIVENESS_ANSWERS or group_liveness not in LIVENESS_ANSWERS:
        raise ValueError("custody liveness answers are outside the closed set")
    bounded = lease.reserved_wall_seconds >= 0.0
    if lease.boot_id != current_boot_id:
        return CustodyRecoveryDecision(
            "stale-boot",
            "old boot cannot retain a local paid process or descriptor",
            bounded,
            True,
        )
    observed_alive = ALIVE in {liveness, group_liveness}
    # Only the FLOCK's silence blocks. It is the authoritative oracle: exact,
    # answerable by any process, and unanswerable only when its pathname was
    # deleted or replaced -- which is itself the attack. The group probe is
    # strictly ADDITIVE, covering the descendant that dropped the descriptor;
    # a host where group membership cannot be enumerated at all must fall back
    # to the flock alone, not refuse every clean session.
    unanswerable = liveness == UNANSWERABLE
    if not lease.is_bound:
        # Nothing was ever spawned under this lease, so nothing can be holding
        # its lock or its group: an absent lock pathname is "never created", not
        # a deleted one.  A lock that IS held on an unbound lease is the
        # reserve-to-bind crash window and still fails closed.
        if observed_alive:
            return CustodyRecoveryDecision(
                "unreapable-refuse",
                f"Darwin residual: {CUSTODY_RESIDUAL}",
                bounded,
                True,
            )
        return CustodyRecoveryDecision(
            "already-dead", "paid custody was reserved but never bound to a subtree", bounded, True
        )
    if not observed_alive and not unanswerable:
        return CustodyRecoveryDecision(
            "already-dead",
            "bound paid subtree released its lock and its process group is empty",
            bounded,
            True,
        )
    if observed_alive and identity_matches and past_deadline:
        return CustodyRecoveryDecision(
            "reap-verified-group",
            "leader start token and process group match past the lease deadline",
            bounded,
            True,
        )
    if observed_alive and identity_matches:
        return CustodyRecoveryDecision(
            "active-refuse", "verified paid subtree remains inside its deadline", bounded, True
        )
    return CustodyRecoveryDecision(
        "unreapable-refuse",
        f"Darwin residual: {CUSTODY_RESIDUAL}",
        bounded,
        True,
    )
