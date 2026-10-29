"""Private persisted campaign configuration for unattended crawler resumes."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    DEFAULT_AUTHOR_WAVE_CONCURRENCY,
    DEFAULT_CHECKER_WAVE_CONCURRENCY,
    HOST_CAPABILITY_DEFERRAL_STATUS_CODE,
    MAX_AUTHOR_WAVE_CONCURRENCY,
    MAX_CHECKER_WAVE_CONCURRENCY,
)
from menagerie.crawler.executable_paths import normalize_executable
from menagerie.crawler.identity import canonical_json_bytes, durable_fsync, stable_hash

CAMPAIGN_CONFIG_FORMAT = "menagerie.crawler.campaign-config.v6"
_LEGACY_CAMPAIGN_CONFIG_FORMAT_V5 = "menagerie.crawler.campaign-config.v5"
_LEGACY_CAMPAIGN_CONFIG_FORMAT_V4 = "menagerie.crawler.campaign-config.v4"
_LEGACY_CAMPAIGN_CONFIG_FORMAT_V3 = "menagerie.crawler.campaign-config.v3"

ONLY_STATUS_VOCABULARY: frozenset[str] = frozenset(
    {
        "deferred:*",
        "deferred:needs-cuda",
        "deferred:needs-x86",
        HOST_CAPABILITY_DEFERRAL_STATUS_CODE,
    }
)
"""Closed ``--only-status`` re-validation selectors.

The host-capability code joins the two platform handoffs because it is the same KIND of
operator action: a member this host cannot run, re-run where it can. Leaving it out was
the mirror-image of the bug WS-2 fixes -- minting a machine-scoped head whose only
operator lane forbade the very hosts that could discharge it.
"""


@dataclass(frozen=True)
class LaunchEnvironment:
    """The launch's environment contract, frozen so a bare wake can rebuild it.

    A usage-pause resume fires through launchd, which hands the process a bare
    environment. Seeding it with ``launchctl setenv`` puts the campaign's
    correctness in per-user machine state that no ledger records and a reboot
    erases -- the same class of bug as the retired 4/4 invocation defaults, one
    layer down. These four values are therefore frozen beside the lane commands
    the config already carries.

    Deliberately NOT here: the provider reconcile command and both lane widths.
    They are already first-class ``CampaignConfig`` fields, and the environment
    the resume exports is DERIVED from those fields. Two persisted copies of one
    value is exactly how a resume silently reverts half of a tuned launch.

    Parameters
    ----------
    environment_state_root:
        The sealed CAS root (``MENAGERIE_ENVIRONMENT_STATE_ROOT``). The
        environment-readiness receipt is checked against it, so a resume that
        loses it refuses admission outright.
    tmp_root:
        The environment lane's staging root. Its override channel is the
        ``--tmp-root`` flag, not an environment name.
    author_model:
        The author executor's ``--model`` value (``MENAGERIE_AUTHOR_MODEL``),
        read by the executor CHILD -- so it has to be in the environment the
        resumed driver passes down, not only in the driver's own config.
    require_author_receipt:
        Whether the author lane demands the executor's attempt-bound publication
        receipt (``MENAGERIE_AUTHOR_REQUIRE_RECEIPT``). Losing it on resume
        silently downgrades the lane to accepting unattested publications.
    """

    environment_state_root: Optional[Path]
    tmp_root: Optional[Path]
    author_model: Optional[str]
    require_author_receipt: bool

    #: Exact persisted member names, so a load validates in both directions.
    FIELDS = frozenset(
        {
            "environment_state_root",
            "tmp_root",
            "author_model",
            "require_author_receipt",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Return the closed JSON representation.

        Returns
        -------
        dict[str, Any]
            Canonical persisted launch environment contract.
        """

        return {
            "environment_state_root": (
                str(self.environment_state_root)
                if self.environment_state_root is not None
                else None
            ),
            "tmp_root": str(self.tmp_root) if self.tmp_root is not None else None,
            "author_model": self.author_model,
            "require_author_receipt": self.require_author_receipt,
        }

    @classmethod
    def empty(cls) -> "LaunchEnvironment":
        """Return the contract a pre-v6 campaign config carries: nothing.

        Returns
        -------
        LaunchEnvironment
            An all-absent contract, which restores no value and diverges from
            none.
        """

        return cls(None, None, None, False)

    @classmethod
    def from_dict(cls, value: Any) -> "LaunchEnvironment":
        """Parse and validate one persisted launch environment contract.

        Parameters
        ----------
        value:
            Decoded JSON object, or ``None`` for a legacy config.

        Returns
        -------
        LaunchEnvironment
            Validated contract.
        """

        if value is None:
            return cls.empty()
        if not isinstance(value, Mapping) or set(value) != cls.FIELDS:
            raise ValueError("campaign config launch environment field contract is invalid")
        author_model = value.get("author_model")
        if author_model is not None and (
            not isinstance(author_model, str) or not author_model
        ):
            raise ValueError("campaign config launch environment author model is invalid")
        require_receipt = value.get("require_author_receipt")
        if not isinstance(require_receipt, bool):
            raise ValueError(
                "campaign config launch environment receipt requirement must be boolean"
            )
        raw_state_root = value.get("environment_state_root")
        raw_tmp_root = value.get("tmp_root")
        return cls(
            environment_state_root=(
                None
                if raw_state_root is None
                else _absolute_path(raw_state_root, "launch environment state root")
            ),
            tmp_root=(
                None
                if raw_tmp_root is None
                else _absolute_path(raw_tmp_root, "launch environment tmp root")
            ),
            author_model=author_model,
            require_author_receipt=require_receipt,
        )


@dataclass(frozen=True)
class CampaignConfig:
    """Resolved operator inputs that must survive a clean wake environment."""

    #: Module-level closed vocabulary, exposed on the class so every validator and
    #: test reads ONE list (the two copies in campaign config and DriverConfig were
    #: how the code stayed unroutable in four places at once).
    ONLY_STATUS_VOCABULARY = ONLY_STATUS_VOCABULARY

    repo_root: Path
    intake_root: Path
    target: str
    run_id: str
    author_queue_root: Optional[Path]
    author_command: tuple[str, ...]
    checker_command: tuple[str, ...]
    environment_command: tuple[str, ...]
    notify_command: Optional[tuple[str, ...]]
    public_mirror: Path
    private_mirror: Path
    review_checkpoint_at: Optional[int]
    progress_milestones: tuple[int, ...]
    phase: Optional[str]
    only_status: Optional[str]
    author_wall_seconds: Optional[float]
    author_concurrency: int = DEFAULT_AUTHOR_WAVE_CONCURRENCY
    checker_concurrency: int = DEFAULT_CHECKER_WAVE_CONCURRENCY
    provider_reconcile_command: tuple[str, ...] = ()
    launch_environment: LaunchEnvironment = LaunchEnvironment(None, None, None, False)

    def to_dict(self) -> dict[str, Any]:
        """Return the closed JSON representation.

        Returns
        -------
        dict[str, Any]
            Canonical persisted campaign configuration.
        """

        return {
            "format": CAMPAIGN_CONFIG_FORMAT,
            "repo_root": str(self.repo_root),
            "intake_root": str(self.intake_root),
            "target": self.target,
            "run_id": self.run_id,
            "author_queue_root": (
                str(self.author_queue_root) if self.author_queue_root is not None else None
            ),
            "author_command": list(self.author_command),
            "checker_command": list(self.checker_command),
            "environment_command": list(self.environment_command),
            "notify_command": (
                list(self.notify_command) if self.notify_command is not None else None
            ),
            "mirrors": {
                "public": str(self.public_mirror),
                "private": str(self.private_mirror),
            },
            "review_checkpoint_at": self.review_checkpoint_at,
            "progress_milestones": list(self.progress_milestones),
            "phase": self.phase,
            "only_status": self.only_status,
            "author_wall_seconds": self.author_wall_seconds,
            "author_concurrency": self.author_concurrency,
            "checker_concurrency": self.checker_concurrency,
            "provider_reconcile_command": list(self.provider_reconcile_command),
            "launch_environment": self.launch_environment.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CampaignConfig":
        """Parse and validate one persisted campaign configuration.

        Parameters
        ----------
        value:
            Decoded JSON object.

        Returns
        -------
        CampaignConfig
            Validated resolved campaign configuration.
        """

        expected = {
            "format",
            "repo_root",
            "intake_root",
            "target",
            "run_id",
            "author_queue_root",
            "author_command",
            "checker_command",
            "environment_command",
            "notify_command",
            "mirrors",
            "review_checkpoint_at",
            "progress_milestones",
            "phase",
            "only_status",
            "author_wall_seconds",
            "author_concurrency",
            "checker_concurrency",
            "provider_reconcile_command",
            "launch_environment",
        }
        legacy_v5_expected = expected - {"launch_environment"}
        legacy_v4_expected = legacy_v5_expected - {"provider_reconcile_command"}
        legacy_v3_expected = legacy_v4_expected - {
            "author_concurrency",
            "checker_concurrency",
        }
        format_value = value.get("format")
        if not (
            (set(value) == expected and format_value == CAMPAIGN_CONFIG_FORMAT)
            or (
                set(value) == legacy_v5_expected
                and format_value == _LEGACY_CAMPAIGN_CONFIG_FORMAT_V5
            )
            or (
                set(value) == legacy_v4_expected
                and format_value == _LEGACY_CAMPAIGN_CONFIG_FORMAT_V4
            )
            or (
                set(value) == legacy_v3_expected
                and format_value == _LEGACY_CAMPAIGN_CONFIG_FORMAT_V3
            )
        ):
            raise ValueError("campaign config has an invalid field contract")
        mirrors = value.get("mirrors")
        if not isinstance(mirrors, Mapping) or set(mirrors) != {"public", "private"}:
            raise ValueError("campaign config mirrors are invalid")
        author = _command(value.get("author_command"), "author")
        checker = _command(value.get("checker_command"), "checker")
        environment = _command(value.get("environment_command"), "environment")
        raw_provider_reconcile = value.get("provider_reconcile_command")
        provider_reconcile = (
            ()
            if raw_provider_reconcile in (None, [])
            else _command(raw_provider_reconcile, "provider reconciliation")
        )
        raw_notify = value.get("notify_command")
        notify = None if raw_notify is None else _command(raw_notify, "notify")
        milestones = value.get("progress_milestones")
        if not isinstance(milestones, list) or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in milestones
        ):
            raise ValueError("campaign config progress milestones are invalid")
        review = value.get("review_checkpoint_at")
        if review is not None and (
            not isinstance(review, int) or isinstance(review, bool) or review < 0
        ):
            raise ValueError("campaign config review checkpoint is invalid")
        # The tuned wall grant is persisted for exactly the reason the tuned
        # author concurrency is: a supervised restart rebuilds the driver's
        # invocation from this file, so anything left out of it is silently
        # reverted to a default mid-campaign. `None` means "take the campaign's
        # own grant", which the frozen intake still determines on its own.
        raw_wall = value.get("author_wall_seconds")
        author_wall_seconds: Optional[float] = None
        if raw_wall is not None:
            if (
                not isinstance(raw_wall, (int, float))
                or isinstance(raw_wall, bool)
                or not float(raw_wall) > 0.0
                or float(raw_wall) in (float("inf"), float("-inf"))
                or float(raw_wall) != float(raw_wall)
            ):
                raise ValueError("campaign config author wall grant is invalid")
            author_wall_seconds = float(raw_wall)
        # V3 could not persist widths at all. Migrate it to the BL-11 campaign
        # widths instead of silently recreating the retired 4/4 invocation defaults.
        raw_author_concurrency = value.get(
            "author_concurrency", DEFAULT_AUTHOR_WAVE_CONCURRENCY
        )
        raw_checker_concurrency = value.get(
            "checker_concurrency", DEFAULT_CHECKER_WAVE_CONCURRENCY
        )
        if (
            not isinstance(raw_author_concurrency, int)
            or isinstance(raw_author_concurrency, bool)
            or not 1 <= raw_author_concurrency <= MAX_AUTHOR_WAVE_CONCURRENCY
        ):
            raise ValueError("campaign config author concurrency is invalid")
        if (
            not isinstance(raw_checker_concurrency, int)
            or isinstance(raw_checker_concurrency, bool)
            or not 1 <= raw_checker_concurrency <= MAX_CHECKER_WAVE_CONCURRENCY
        ):
            raise ValueError("campaign config checker concurrency is invalid")
        target = value.get("target")
        phase = value.get("phase")
        only_status = value.get("only_status")
        run_id = value.get("run_id")
        raw_author_queue = value.get("author_queue_root")
        if target not in {"osx-arm64", "linux-x86_64-cuda"}:
            raise ValueError("campaign config target is invalid")
        if phase not in {None, "pytorch", "native-tail"}:
            raise ValueError("campaign config phase is invalid")
        if only_status not in {None, *CampaignConfig.ONLY_STATUS_VOCABULARY}:
            raise ValueError("campaign config only-status is invalid")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("campaign config run ID is invalid")
        repo_root = _absolute_path(value.get("repo_root"), "repo root")
        intake_root = _absolute_path(value.get("intake_root"), "intake root")
        public_mirror = _absolute_path(mirrors.get("public"), "public mirror")
        private_mirror = _absolute_path(mirrors.get("private"), "private mirror")
        author_queue_root = (
            None
            if raw_author_queue is None
            else _absolute_path(raw_author_queue, "author queue root")
        )
        return cls(
            repo_root=repo_root,
            intake_root=intake_root,
            target=target,
            run_id=run_id,
            author_queue_root=author_queue_root,
            author_command=author,
            checker_command=checker,
            environment_command=environment,
            notify_command=notify,
            public_mirror=public_mirror,
            private_mirror=private_mirror,
            review_checkpoint_at=review,
            progress_milestones=tuple(milestones),
            phase=phase,
            only_status=only_status,
            author_wall_seconds=author_wall_seconds,
            author_concurrency=raw_author_concurrency,
            checker_concurrency=raw_checker_concurrency,
            provider_reconcile_command=provider_reconcile,
            # Absent for every pre-v6 config, which restores nothing and
            # diverges from nothing -- a legacy campaign resumes exactly as it
            # did, it just keeps needing its environment seeded externally.
            launch_environment=LaunchEnvironment.from_dict(value.get("launch_environment")),
        )


def default_campaign_config_path(repo_root: Path, intake_root: Path) -> Path:
    """Return the private config path for one exact intake.

    Parameters
    ----------
    repo_root, intake_root:
        Resolved repository and immutable intake roots.

    Returns
    -------
    pathlib.Path
        Campaign-specific gitignored configuration path.
    """

    suffix = stable_hash(str(intake_root.resolve())).removeprefix("sha256:")[:12]
    return (
        repo_root.resolve()
        / ".crawl-local"
        / "campaign-configs"
        / (f"{intake_root.name}-{suffix}.json")
    )


def write_campaign_config(path: Path, config: CampaignConfig) -> None:
    """Atomically persist one mode-0600 campaign configuration.

    Parameters
    ----------
    path, config:
        Destination and fully resolved configuration.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(config.to_dict()) + b"\n")
            handle.flush()
            durable_fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_campaign_config(path: Path) -> CampaignConfig:
    """Load a private mode-safe campaign configuration.

    Parameters
    ----------
    path:
        Persisted configuration path.

    Returns
    -------
    CampaignConfig
        Validated configuration.
    """

    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise ValueError(f"campaign config must be mode 0600: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"campaign config is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("campaign config must contain one JSON object")
    return CampaignConfig.from_dict(value)


def resolve_command(command: Sequence[str], *, cwd: Path) -> tuple[str, ...]:
    """Resolve an argv command to stable absolute executable and path arguments.

    Parameters
    ----------
    command:
        Nonempty parsed command.
    cwd:
        Base for relative existing path arguments.

    Returns
    -------
    tuple[str, ...]
        Clean-environment-safe command.
    """

    if not command:
        raise ValueError("cannot resolve an empty command")
    executable = command[0]
    # Normalize WITHOUT following the final symlink: a virtualenv interpreter must keep
    # pointing at its own pyvenv.cfg, or the configured venv silently collapses into the
    # base environment and its site-packages. See menagerie/crawler/executable_paths.py.
    resolved_executable = normalize_executable(executable, cwd=cwd)
    if resolved_executable is None:
        raise ValueError(f"command executable is not resolvable: {executable}")
    resolved: list[str] = [str(resolved_executable)]
    for argument in command[1:]:
        candidate = Path(argument).expanduser()
        rooted = candidate if candidate.is_absolute() else cwd / candidate
        resolved.append(str(rooted.resolve()) if rooted.exists() else argument)
    return tuple(resolved)


def _command(value: Any, name: str) -> tuple[str, ...]:
    """Validate one persisted argv array.

    Parameters
    ----------
    value, name:
        Candidate value and diagnostic field name.

    Returns
    -------
    tuple[str, ...]
        Validated immutable argv.
    """

    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"campaign config {name} command is invalid")
    command = tuple(value)
    if not Path(command[0]).is_absolute():
        raise ValueError(f"campaign config {name} executable must be absolute")
    return command


def _absolute_path(value: Any, name: str) -> Path:
    """Validate one persisted absolute path.

    Parameters
    ----------
    value, name:
        Candidate path value and diagnostic name.

    Returns
    -------
    pathlib.Path
        Absolute path.
    """

    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"campaign config {name} must be absolute")
    return Path(value)


def _fsync_directory(path: Path) -> None:
    """Fsync a directory after an atomic configuration replacement.

    Parameters
    ----------
    path:
        Directory containing the replaced file.
    """

    descriptor = os.open(path, os.O_RDONLY)
    try:
        durable_fsync(descriptor)
    finally:
        os.close(descriptor)
