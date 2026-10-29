"""Test-only end-to-end dry-run support for the crawler CLI acceptance gate."""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from menagerie.crawler.author_dispatch import AUTHOR_EXIT_RETRYABLE, AuthorBackoffSignal
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.authority import EnvironmentAuthorityCache
from menagerie.crawler.constants import AuthorPauseReason, EnvironmentPhase
from menagerie.crawler.driver_admission import (
    _raise_for_checker_exit,
    _run_operator_command,
    classify_author_exit,
)
from menagerie.crawler.driver import (
    AuthorArtifact,
    CrawlerDriver,
    DriverConfig,
    DriverDependencies,
    DriverPaths,
    SupervisedForwardLane,
    WorkItem,
)
from menagerie.crawler.driver_contracts import AuthorBackoffError
from menagerie.crawler.env_lifecycle import (
    LifecycleResult,
    ProbeResult,
    SequentialEnvironmentLifecycle,
    parse_probe_receipt_bytes,
)
from menagerie.crawler.envs import (
    EnvironmentIntent,
    EnvironmentRegistry,
    IntentProbes,
    LockArtifacts,
)
from menagerie.crawler.fetcher import UnpinnedTargetError
from menagerie.crawler.identity import (
    canonical_json_bytes,
    hash_bytes,
    stable_hash,
)
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.operator_protocol import OperatorExitCode
from menagerie.crawler.paid_session_lifecycle import ProviderReconciliation
from menagerie.crawler.proposal import model_code_manifest
from menagerie.crawler.tests.conftest import NOW, _committed_fixture_intent
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakeChecker,
    FakeNotifier,
    _rebind_fake_author_result,
    _refresh_proposal_identities,
)


@dataclass(frozen=True)
class DryRunCase:
    """One tiny real model and its expected source-gated observations."""

    name: str
    modality: tuple[str, ...]
    shape: tuple[int, ...]
    divergence: str
    adapter_source: str
    fidelity_required: bool = False


_MLP_SOURCE = """from __future__ import annotations

import menagerie_round19_sentinel
import torch


class TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


def build_model() -> object:
    assert menagerie_round19_sentinel.INTERPRETER_SENTINEL == 'round19-selected-prefix'
    return TinyMLP()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 4, device=device),), {})
"""

_CONV_STATISTICAL_SOURCE = """from __future__ import annotations

import menagerie_round19_sentinel
import torch


class TinyStatisticalConvNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.norm = torch.nn.BatchNorm2d(4)
        self.dropout = torch.nn.Dropout2d(p=0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.dropout(torch.relu(self.norm(self.conv(value))))
        return value.mean(dim=(2, 3))


def build_model() -> object:
    assert menagerie_round19_sentinel.INTERPRETER_SENTINEL == 'round19-selected-prefix'
    return TinyStatisticalConvNet()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(2, 3, 8, 8, device=device),), {})
"""

_STRUCTURAL_SOURCE = """from __future__ import annotations

import menagerie_round19_sentinel
import torch


class TinyStructuralBranch(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            return value.mean(dim=(2, 3))
        return value.flatten(1)


def build_model() -> object:
    assert menagerie_round19_sentinel.INTERPRETER_SENTINEL == 'round19-selected-prefix'
    return TinyStructuralBranch()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 3, 8, 8, device=device),), {})
"""

_CONV_STABLE_SOURCE = """from __future__ import annotations

import menagerie_round19_sentinel
import torch


class TinyStableConvNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value).mean(dim=(2, 3))


def build_model() -> object:
    assert menagerie_round19_sentinel.INTERPRETER_SENTINEL == 'round19-selected-prefix'
    return TinyStableConvNet()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 3, 8, 8, device=device),), {})
"""

DRY_RUN_CASES = (
    DryRunCase("DryRunMLP", ("unknown",), (1, 4), "none", _MLP_SOURCE),
    DryRunCase(
        "DryRunStatisticalConvNet",
        ("vision",),
        (2, 3, 8, 8),
        "statistical",
        _CONV_STATISTICAL_SOURCE,
    ),
    DryRunCase(
        "DryRunStructuralBranch",
        ("vision",),
        (1, 3, 8, 8),
        "structural",
        _STRUCTURAL_SOURCE,
        fidelity_required=True,
    ),
    DryRunCase("DryRunStableConvNet", ("vision",), (1, 3, 8, 8), "none", _CONV_STABLE_SOURCE),
)

DRY_RUN_ITEMS = (
    *((case.name, "base") for case in DRY_RUN_CASES),
    ("DryRunMLP", "calibration-1"),
    ("DryRunMLP", "calibration-2"),
    ("DryRunStatisticalConvNet", "calibration-1"),
    ("DryRunStatisticalConvNet", "calibration-2"),
    ("DryRunStableConvNet", "calibration-1"),
    ("DryRunStableConvNet", "calibration-2"),
)
DRY_RUN_PHASE = EnvironmentPhase.PYTORCH

#: Deterministic source-partition injection (existing arm).
INJECT_ALL_SOURCE_FAILURE_ENV = "MENAGERIE_DRY_RUN_INJECT_ALL_SOURCE_FAILURE"
#: Deterministic paid-session death injection (existing arm; budget grammar).
INJECT_SESSION_DEATH_ENV = "MENAGERIE_DRY_RUN_INJECT_SESSION_DEATH"
#: Which author guard session the injected death lands in: ``author`` (default,
#: the attempt's second guard) or ``source-request`` (the attempt's FIRST paid
#: guard, PM-22's previously unfailable stage).
INJECT_SESSION_DEATH_STAGE_ENV = "MENAGERIE_DRY_RUN_INJECT_SESSION_DEATH_STAGE"
#: Deterministic paid CHECKER-session death injection (PM-22/B2-X-V41: every
#: paid lane failable post-reserve). Same budget grammar as the author knob;
#: keys are the closed ``GateKind`` values (``metadata_batch:1``) or a bare
#: count consumed by the first checker sessions of any kind.
INJECT_CHECKER_SESSION_DEATH_ENV = "MENAGERIE_DRY_RUN_INJECT_CHECKER_SESSION_DEATH"
#: Inject ONE real typed provider usage backoff into the named lane (``author``).
#: Mirrors the all-source-failure arm: env-gated, deterministic, and it swaps a
#: lane implementation rather than reaching inside the driver.
INJECT_USAGE_PAUSE_ENV = "MENAGERIE_DRY_RUN_INJECT_USAGE_PAUSE"
#: Seconds after the driver clock at which the injected provider reset lands.
INJECT_USAGE_PAUSE_RESET_ENV = "MENAGERIE_DRY_RUN_INJECT_USAGE_PAUSE_RESET_SECONDS"
#: Deterministic advance of the dry-run driver clock, in seconds. The dry run's
#: clock is frozen for determinism, so a wake callback that must observe "after
#: the provider reset" advances it explicitly instead of sleeping in real time.
CLOCK_OFFSET_ENV = "MENAGERIE_DRY_RUN_CLOCK_OFFSET_SECONDS"

#: Default injected reset distance: a few seconds past the pause, the shape a real
#: short usage window takes.
DEFAULT_INJECTED_RESET_SECONDS = 5
#: Injected usage-limit response text, ASCII and classifier-shaped.
INJECTED_USAGE_RESPONSE = "dry-run injected usage limit reached"
#: One-shot marker, relative to the campaign root, proving the injected pause
#: fired exactly once across CLI processes.
USAGE_PAUSE_MARKER = Path("injections") / "usage-pause-fired.json"
#: Append-only record of every OS-scheduler registration the dry run WOULD make.
WAKE_OS_REGISTRATIONS = Path("wake-os-registrations.jsonl")


def dry_run_clock_now() -> str:
    """Return the frozen dry-run instant, advanced by the explicit offset.

    Returns
    -------
    str
        RFC 3339 ``Z`` timestamp equal to ``NOW`` plus ``CLOCK_OFFSET_ENV`` seconds.
    """

    offset = int(os.environ.get(CLOCK_OFFSET_ENV, "0"))
    if offset == 0:
        return NOW
    moment = datetime.fromisoformat(NOW.replace("Z", "+00:00")) + timedelta(seconds=offset)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def injected_reset_at() -> str:
    """Return the injected provider reset instant for the current clock.

    Returns
    -------
    str
        RFC 3339 ``Z`` reset timestamp a few seconds past the pause.
    """

    seconds = int(
        os.environ.get(INJECT_USAGE_PAUSE_RESET_ENV, str(DEFAULT_INJECTED_RESET_SECONDS))
    )
    moment = datetime.fromisoformat(dry_run_clock_now().replace("Z", "+00:00")) + timedelta(
        seconds=seconds
    )
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TinyModelAuthor(FakeAuthor):
    """Extend the canned author with runnable typed adapters for the tiny corpus."""

    def __init__(self) -> None:
        """Index the frozen tiny corpus by intake name."""

        super().__init__()
        self._cases = {case.name: case for case in DRY_RUN_CASES}

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Return a canned accepted proposal bound to one real tiny adapter.

        Parameters
        ----------
        item, work_root, config, context:
            Mandatory driver work item, output root, configuration, and frozen authority context.

        Returns
        -------
        AuthorArtifact
            Authenticated author result containing the selected tiny adapter.
        """

        artifact = super().author(item, work_root, config, context)
        case = self._cases[item.intake.name]
        adapter_path = artifact.model_dir / "adapter.py"
        adapter_bytes = case.adapter_source.encode("utf-8")
        adapter_path.write_bytes(adapter_bytes)

        proposal = artifact.proposal
        proposal["proposal_id"] = f"proposal-{item.stable_id}"
        proposal["proposal_sha256"] = stable_hash(
            {"stable_id": item.stable_id, "adapter_sha256": hash_bytes(adapter_bytes)}
        )
        facts = proposal["proposed_facts"]
        facts["identity"]["canonical_name"] = case.name
        if case.name == "DryRunMLP":
            facts["identity"]["canonical_name"] = "MLP"
            facts["taxonomy"]["family"] = "MLP"
            facts["external_metadata"]["family"] = "MLP"
            facts["external_metadata"]["architecture_class"] = ["MLP"]
        facts["external_metadata"]["modality"] = list(case.modality)
        facts["external_metadata"]["modes"] = {
            "meaningful_modes": ["train", "eval"],
            "train_eval_divergence": case.divergence,
        }
        facts["modes"] = {
            "meaningful_modes": ["train", "eval"],
            "per_mode_run": {},
            "train_eval_divergence": case.divergence,
            "divergence_evidence": "expected tiny-model train/eval behavior",
        }
        implementation = facts["implementation"]
        implementation.update(
            {
                "recipe_type": "typed-adapter",
                "code_path": "adapter.py",
                "code_sha256": hash_bytes(adapter_bytes),
                "builder_symbol": "build_model",
                "dummy_call_symbol": "make_dummy_call",
                "library_recipe": None,
            }
        )
        code_manifest = [dict(row) for row in model_code_manifest(adapter_path, artifact.model_dir)]
        implementation["code_manifest"] = code_manifest
        facts["evidence"]["excerpts"][0]["supports"] = sorted(
            set(facts["evidence"]["excerpts"][0]["supports"])
            | {
                "implementation.code_manifest[].path",
                "implementation.code_manifest[].sha256",
            }
        )
        contract = facts["input_contract"]
        contract["args"][0].update(
            {
                "semantic_role": "features" if case.name == "DryRunMLP" else "image",
                "shape": list(case.shape),
            }
        )
        proposal["verified_hashes"]["code"] = hash_bytes(adapter_bytes)
        proposal["verified_hashes"]["code_manifest"] = stable_hash(code_manifest)
        fidelity_identity = stable_hash({"stable_id": item.stable_id, "kind": "dry-run-fidelity"})
        proposal["fidelity_identity"] = fidelity_identity
        resolution = facts["source_resolution"]
        resolution["rung"] = "R3_PORT"
        resolution["attempted_rungs"] = [
            {
                "rung": rung,
                "result": "selected" if rung == "R3_PORT" else "unavailable",
                "reason_code": "documented-search",
                "evidence_ids": ["evidence-1"],
            }
            for rung in ("R1_LIBRARY", "R2_VENDOR", "R3_PORT")
        ]
        excerpt = facts["evidence"]["excerpts"][0]
        port_text = (
            f"{excerpt['text']} The staged source code is a faithful port of the documented "
            f"forward architecture and input contract for {case.name}, using "
            f"{' '.join(case.modality)} modality with {case.divergence} train eval divergence. "
            f"The grounded family is {facts['taxonomy']['family']}."
        )
        port_bytes = port_text.encode("utf-8")
        excerpt.update(
            {
                "locator": f"bytes:0-{len(port_bytes)}",
                "text": port_text,
                "text_sha256": hash_bytes(port_bytes),
            }
        )
        implementation["source_to_code_map"] = [
            {
                "material_item": "documented forward architecture",
                "source_id": "source-1",
                "source_locator": excerpt["locator"],
                "evidence_ids": ["evidence-1"],
                "code_path": "adapter.py",
                "code_locator": "complete typed adapter",
                "disposition": "faithful-port",
            }
        ]
        excerpt["supports"] = sorted(
            set(excerpt["supports"])
            | {
                "implementation.source_to_code_map[].code_locator",
                "implementation.source_to_code_map[].code_path",
                "implementation.source_to_code_map[].disposition",
                "implementation.source_to_code_map[].evidence_ids[]",
                "implementation.source_to_code_map[].material_item",
                "implementation.source_to_code_map[].source_id",
                "implementation.source_to_code_map[].source_locator",
            }
        )
        source_row = resolution["sources"][0]
        source_row.update(
            {
                "content_sha256": hash_bytes(port_bytes),
                "byte_count": len(port_bytes),
            }
        )
        manifest_row = artifact.source_manifest["sources"][0]
        Path(str(manifest_row["cas_path"])).write_bytes(port_bytes)
        manifest_row.update(
            {
                "content_sha256": hash_bytes(port_bytes),
                "byte_count": len(port_bytes),
            }
        )
        source_manifest_identity = stable_hash(artifact.source_manifest["sources"])
        artifact.source_manifest["manifest_sha256"] = source_manifest_identity
        proposal["source_manifest_identity"] = source_manifest_identity
        proposal["verified_hashes"]["source_manifest"] = source_manifest_identity
        facts["fidelity"].update(
            {
                "required": True,
                "reason": (
                    "dry-run typed port exercises structural fidelity"
                    if case.fidelity_required
                    else "dry-run typed port requires canonical fidelity"
                ),
                "verdict": None,
                "fidelity_identity": fidelity_identity,
                "gate_id": None,
                "current": False,
            }
        )
        _refresh_proposal_identities(
            proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        return _rebind_fake_author_result(artifact)


class AllSourceFailureAuthor(TinyModelAuthor):
    """Inject a deterministic all-source-failure terminal partition for acceptance testing."""

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Fail source resolution before any proposal or model attempt can be accepted.

        Parameters
        ----------
        item, work_root, config, context:
            Mandatory author-lane inputs retained to implement the live protocol exactly.

        Raises
        ------
        UnpinnedTargetError
            Always. The typed fetch-contract rejection is the only honest spelling
            for an injected SOURCE partition: the failure classifier records a
            generic exception as ``failed:author`` / ``session-crashed``, because a
            crashed session says nothing about whether the model's source exists.
        """

        del item, work_root, config, context
        raise UnpinnedTargetError("injected dry-run all-source failure")


class UsagePauseInjection:
    """Raise ONE real typed author usage backoff, then defer to the wrapped lane.

    The pause is the production signal, not a simulation of it: the lane raises
    :class:`AuthorBackoffError` exactly as ``CommandAuthorLane`` does when Claude
    answers with an exhausted usage window, so the driver takes its ordinary
    ``paused:usage-limit`` path -- durable pause row, wake episode, scheduled
    recheck -- with a reset the resumed leg can actually pass.

    One-shot and DURABLE: the marker file survives the pausing process, so a resumed
    CLI process authors for real even with the injection still armed. That ordering
    is what makes the rehearsal a cycle instead of a livelock.

    Kept as a mixin so the SAME injected signal composes over either corpus author:
    the tiny real corpus for the real-subprocess rehearsal, and the synthetic unit
    corpus when the forward lane is faked and no adapter is executed.
    """

    def __init__(self, marker_path: Path, *args: Any, **kwargs: Any) -> None:
        """Bind the cross-process one-shot marker.

        Parameters
        ----------
        marker_path:
            Durable marker naming the already-injected pause.
        args, kwargs:
            Forwarded to the wrapped author lane's initializer.
        """

        super().__init__(*args, **kwargs)
        self.marker_path = marker_path

    def _inject_usage_pause_once(self, stable_id: str) -> None:
        """Raise the typed provider backoff on the first author call only.

        Parameters
        ----------
        stable_id:
            Model whose author call absorbs the injected pause.

        Raises
        ------
        AuthorBackoffError
            On the first call only, carrying the typed quota-exhaustion signal.
        """

        if self.marker_path.is_file():
            return
        reset_at = injected_reset_at()
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json_bytes(
            {
                "stable_id": stable_id,
                "reset_at": reset_at,
                "provider": "anthropic",
                "reason": AuthorPauseReason.QUOTA_EXHAUSTED.value,
            }
        )
        handle = os.open(self.marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(handle, payload + b"\n")
            os.fsync(handle)
        finally:
            os.close(handle)
        raise AuthorBackoffError(
            AuthorBackoffSignal(
                reason=AuthorPauseReason.QUOTA_EXHAUSTED,
                retry_after_seconds=None,
                reset_at=reset_at,
                response_excerpt=INJECTED_USAGE_RESPONSE,
                provider="anthropic",
            )
        )

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Pause the campaign once, then behave exactly like the wrapped lane.

        Parameters
        ----------
        item, work_root, config, context:
            Mandatory author-lane inputs, forwarded unchanged after the pause.

        Returns
        -------
        AuthorArtifact
            The wrapped lane's ordinary artifact on every call after the pause.
        """

        self._inject_usage_pause_once(item.stable_id)
        return super().author(item, work_root, config, context)  # type: ignore[misc]


class UsagePauseAuthor(UsagePauseInjection, TinyModelAuthor):
    """The injected one-shot usage pause over the real tiny dry-run corpus."""


class RecordingWakeScheduler:
    """Record the OS-scheduler registrations a dry run would make, never exec them.

    Exactly the unit builder's doctrine (``test_slice_f_driver._driver``): the wake
    RECORDS -- operational rows, episode projection, definition files -- stay real,
    and only the ``launchctl``/``crontab`` effects are stubbed. An acceptance dry-run
    must never mutate the host's real scheduler, and on a CI host or under a sandbox
    the real installer cannot work at all. The real OS path is covered by the gated
    test in ``menagerie/crawler/tests/test_wakeup_os_scheduler.py``.
    """

    def __init__(self, path: Path) -> None:
        """Bind the append-only registration log.

        Parameters
        ----------
        path:
            Durable JSONL log shared by every CLI process of one campaign.
        """

        self.path = path

    def _record(self, action: str, definition: Any) -> None:
        """Append one durable registration fact.

        Parameters
        ----------
        action:
            ``install``, ``verify``, or ``deactivate``.
        definition:
            Rendered wake definition owning the projection.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(
                canonical_json_bytes(
                    {
                        "action": action,
                        "episode_id": definition.episode.episode_id,
                        "backend": definition.backend.value,
                        "definition_path": str(definition.definition_path),
                        "not_before": definition.episode.not_before,
                        "reset_at": definition.episode.reset_at,
                    }
                )
                + b"\n"
            )

    def install(self, definition: Any) -> None:
        """Record an install instead of registering with the OS scheduler."""

        self._record("install", definition)

    def verify(self, definition: Any) -> bool:
        """Record and affirm the recorded registration.

        Returns
        -------
        bool
            Always ``True``: the recorded registration is the projection under test.
        """

        self._record("verify", definition)
        return True

    def deactivate(self, definition: Any) -> None:
        """Record a deactivation instead of unregistering with the OS scheduler."""

        self._record("deactivate", definition)


def read_wake_os_registrations(campaign_root: Path) -> tuple[Mapping[str, Any], ...]:
    """Read the recorded OS-scheduler registrations in append order.

    Parameters
    ----------
    campaign_root:
        Disposable dry-run campaign root.

    Returns
    -------
    tuple[Mapping[str, Any], ...]
        Recorded install/verify/deactivate facts.
    """

    path = campaign_root / WAKE_OS_REGISTRATIONS
    if not path.is_file():
        return ()
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


#: Source-request guard stand-in: a real paid guard subprocess that publishes nothing
#: and exits clean, exactly like a successful source-request round trip.
_SOURCE_GUARD_EXIT_ZERO = (sys.executable, "-c", "import sys; sys.exit(0)")

#: Author guard stand-in that dies mid-session: custody is already bound and provider
#: exposure released when this exit lands, and nothing was published. The stderr line
#: is the executor's own session-crashed spelling, which ``classify_author_exit``
#: deliberately does NOT match as a model-scoped contract failure -- a session that
#: died without publishing classifies as retryable transport, the same path a real
#: author-session transport death takes.
_AUTHOR_GUARD_SESSION_DEATH = (
    sys.executable,
    "-c",
    (
        "import sys; "
        "print('author executor stage2 failed: session-crashed (attempt 1)', file=sys.stderr); "
        f"sys.exit({AUTHOR_EXIT_RETRYABLE})"
    ),
)

#: Source-request guard that dies mid-session: the attempt's FIRST paid guard,
#: killed after its reservation exists and before any discovery is published.
#: Same retryable-transport classification contract as the author-guard death.
_SOURCE_GUARD_SESSION_DEATH = (
    sys.executable,
    "-c",
    (
        "import sys; "
        "print('author executor stage1 failed: session-crashed (attempt 1)', file=sys.stderr); "
        f"sys.exit({AUTHOR_EXIT_RETRYABLE})"
    ),
)

#: Checker guard that dies mid-session with the wrapper's declared retryable
#: exit: no verdict exists, the transport is to blame, and the production
#: classifier must route it to the bounded retry ladder -- never to a
#: model-scoped or campaign-scoped outcome.
_CHECKER_GUARD_SESSION_DEATH = (
    sys.executable,
    "-c",
    (
        "import sys; "
        "print('checker wrapper failed: session-crashed (attempt 1)', file=sys.stderr); "
        f"sys.exit({int(OperatorExitCode.RETRYABLE_INFRASTRUCTURE)})"
    ),
)


@dataclass(frozen=True)
class SessionDeathPlan:
    """Parsed ``MENAGERIE_DRY_RUN_INJECT_SESSION_DEATH`` death budgets."""

    global_budget: Optional[int]
    keyed_budgets: Mapping[str, int] = field(default_factory=dict)


def parse_session_death_plan(value: str) -> SessionDeathPlan:
    """Parse the session-death injection env value, refusing malformed budgets.

    Parameters
    ----------
    value:
        Either a bare positive count (``"2"``: the first two author invocations
        die once each) or comma-separated ``key:count`` entries (``"DryRunMLP:1"``)
        where each key matches an item's stable id or intake name.

    Returns
    -------
    SessionDeathPlan
        Immutable death budgets for one dry-run campaign.

    Raises
    ------
    ValueError
        On an empty, non-positive, duplicate-key, or otherwise malformed value.
        An injection the composition silently ignored would rehearse nothing.
    """

    text = value.strip()
    if not text:
        raise ValueError("MENAGERIE_DRY_RUN_INJECT_SESSION_DEATH is empty")
    if text.isdigit():
        count = int(text)
        if count <= 0:
            raise ValueError("session-death count must be positive")
        return SessionDeathPlan(global_budget=count)
    budgets: dict[str, int] = {}
    for entry in text.split(","):
        key, sep, raw = entry.strip().rpartition(":")
        if not sep or not key or not raw.isdigit() or int(raw) <= 0:
            raise ValueError(f"invalid session-death entry: {entry!r}")
        if key in budgets:
            raise ValueError(f"duplicate session-death key: {key!r}")
        budgets[key] = int(raw)
    return SessionDeathPlan(global_budget=None, keyed_budgets=budgets)


class SessionDeathAuthor(TinyModelAuthor):
    """Reserve normally, then die inside the attempt's real paid author guard.

    Launch-night lesson (2026-08-19): the dry run's one failure injection raised
    BEFORE any paid session was reserved, so the reserve -> custody bind -> die ->
    retry composition first executed in production, where the retry ladder's
    reservation-reuse bug halted the step-200 launch. This author makes every paid
    lane failable mid-flight in rehearsal: an injected invocation runs the real
    two-guard transport (``_run_operator_command`` -- source-request guard, then
    author guard), so the reservation is admitted by the writer, custody binds and
    releases through the production custody machinery, and the death is classified
    by the production ``classify_author_exit`` into the retry ladder. The retry
    then authors normally.
    """

    def __init__(self, plan: SessionDeathPlan, *, stage: str = "author") -> None:
        """Bind one immutable death plan behind a wave-concurrency-safe lock.

        Parameters
        ----------
        plan:
            Immutable death budgets for one dry-run campaign.
        stage:
            Which guard session dies: ``author`` (the attempt's second guard,
            the historical arm) or ``source-request`` (the attempt's FIRST
            paid guard). Any other value refuses loudly -- an injection aimed
            at a stage that does not exist would rehearse nothing.
        """

        if stage not in ("author", "source-request"):
            raise ValueError(f"unknown session-death stage: {stage!r}")
        super().__init__()
        self._stage = stage
        self._plan_lock = threading.Lock()
        self._global_remaining = plan.global_budget or 0
        self._keyed_remaining = dict(plan.keyed_budgets)

    def _consume_death(self, item: WorkItem) -> bool:
        """Consume one configured death for this invocation, if any remains."""

        with self._plan_lock:
            for key in (item.stable_id, item.intake.name):
                if self._keyed_remaining.get(key, 0) > 0:
                    self._keyed_remaining[key] -= 1
                    return True
            if self._global_remaining > 0:
                self._global_remaining -= 1
                return True
        return False

    def _die_in_author_guard_session(self, item: WorkItem) -> None:
        """Run the attempt's real guard transport and die in the author session.

        Parameters
        ----------
        item:
            Work item whose author attempt dies after custody bind, before
            publication.

        Raises
        ------
        Exception
            Whatever typed transport failure ``classify_author_exit`` mints for
            the dead author session -- never a hand-rolled spelling, so the
            injected death takes exactly the classified path a real transport
            death takes through the retry ladder.
        """

        source_command = (
            _SOURCE_GUARD_SESSION_DEATH
            if self._stage == "source-request"
            else _SOURCE_GUARD_EXIT_ZERO
        )
        source = _run_operator_command(source_command, timeout_seconds=30.0, env=None)
        classify_author_exit(
            "source-request", item.stable_id, source.returncode, source.stdout, source.stderr
        )
        session = _run_operator_command(_AUTHOR_GUARD_SESSION_DEATH, timeout_seconds=30.0, env=None)
        classify_author_exit(
            "author", item.stable_id, session.returncode, session.stdout, session.stderr
        )
        raise AssertionError(
            "an injected author session death must classify as a typed transport failure"
        )

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Die mid-session while a death budget remains, else author normally.

        Parameters
        ----------
        item, work_root, config, context:
            Mandatory author-lane inputs, forwarded verbatim to the tiny author
            on every non-injected (and every retry) invocation.

        Returns
        -------
        AuthorArtifact
            The tiny corpus artifact, once this invocation is not injected.
        """

        if self._consume_death(item):
            self._die_in_author_guard_session(item)
        return super().author(item, work_root, config, context)


class SessionDeathChecker(FakeChecker):
    """Reserve normally, then die inside the attempt's real paid checker guard.

    PM-22/B2-X-V41: the author lane's post-reserve failability
    (:class:`SessionDeathAuthor`) covered one paid lane; the checker lane's
    three paid stages (``metadata_batch``, ``fidelity``,
    ``terminal_disposition``) had no rehearsal that could kill a session after
    its reservation existed. An injected invocation runs a real guard
    subprocess that exits with the wrapper's declared retryable code, and the
    death is minted by the production ``_raise_for_checker_exit`` classifier
    into the bounded retry ladder -- never a hand-rolled exception. The retry
    then checks normally.

    Budget keys are the closed ``GateKind`` values; a bare count dies on the
    first checker sessions of any kind.
    """

    def __init__(self, plan: SessionDeathPlan) -> None:
        """Bind one immutable death plan behind a concurrency-safe lock."""

        super().__init__()
        self._plan_lock = threading.Lock()
        self._global_remaining = plan.global_budget or 0
        self._keyed_remaining = dict(plan.keyed_budgets)

    def _consume_death(self, gate_kind: str) -> bool:
        """Consume one configured death for this gate kind, if any remains."""

        with self._plan_lock:
            if self._keyed_remaining.get(gate_kind, 0) > 0:
                self._keyed_remaining[gate_kind] -= 1
                return True
            if self._global_remaining > 0:
                self._global_remaining -= 1
                return True
        return False

    def _die_in_checker_guard_session(self, gate_kind: str, work_root: Path) -> None:
        """Run a real checker guard subprocess and die on the classified path.

        Parameters
        ----------
        gate_kind:
            Paid checker stage whose session dies post-reserve.
        work_root:
            Attempt work root; names the (absent) request sidecar exactly as a
            crashed wrapper leaves it, so evidence falls back to the stream
            tail like a real transport death.

        Raises
        ------
        Exception
            Whatever typed transport failure ``_raise_for_checker_exit`` mints
            for the dead checker session.
        """

        session = _run_operator_command(
            _CHECKER_GUARD_SESSION_DEATH, timeout_seconds=30.0, env=None
        )
        _raise_for_checker_exit(
            session.returncode,
            session.stdout,
            session.stderr,
            request_path=work_root / "checker" / f"injected-death-{gate_kind}" / "request.json",
        )
        raise AssertionError(
            "an injected checker session death must classify as a typed transport failure"
        )

    def check_metadata(
        self, artifacts: Any, work_root: Path, config: DriverConfig
    ) -> Any:
        """Die mid-session while a metadata budget remains, else check normally."""

        if self._consume_death("metadata_batch"):
            self._die_in_checker_guard_session("metadata_batch", work_root)
        return super().check_metadata(artifacts, work_root, config)

    def check_fidelity(
        self, artifact: Any, work_root: Path, config: DriverConfig
    ) -> Any:
        """Die mid-session while a fidelity budget remains, else check normally."""

        if self._consume_death("fidelity"):
            self._die_in_checker_guard_session("fidelity", work_root)
        return super().check_fidelity(artifact, work_root, config)

    def check_terminal(
        self, artifact: Any, work_root: Path, config: DriverConfig
    ) -> Any:
        """Die mid-session while a terminal budget remains, else check normally."""

        if self._consume_death("terminal_disposition"):
            self._die_in_checker_guard_session("terminal_disposition", work_root)
        return super().check_terminal(artifact, work_root, config)


class MaterializedDryRunEnvironment(SequentialEnvironmentLifecycle):
    """Expose one already-provisioned prefix through the production lifecycle protocol."""

    def __init__(
        self,
        prefix: Path,
        intent: EnvironmentIntent,
        probe_results: tuple[ProbeResult, ...],
    ) -> None:
        """Store exact prefix and committed lifecycle artifacts.

        Parameters
        ----------
        prefix, intent, probe_results:
            Real materialized prefix plus its exact lock/export/probe contract.
        """

        self.prefix = prefix
        self.intent = intent
        self.probe_results = probe_results
        self._active = prefix
        self._authority_cache = EnvironmentAuthorityCache()

    def run(
        self,
        intent: EnvironmentIntent,
        *,
        use: Any,
    ) -> LifecycleResult:
        """Give the driver the real prefix so it performs strict production binding.

        Parameters
        ----------
        intent:
            Routed intent, which must equal the artifact-backed fixture intent.
        use:
            Driver callback that strictly calls ``bind_materialized_environment`` before work.

        Returns
        -------
        LifecycleResult
            Durable lifecycle observation for the already-materialized fixture.
        """

        if intent != self.intent:
            raise AssertionError("dry-run environment intent differs from fixture artifacts")
        use(self.prefix, self.probe_results)
        return LifecycleResult(
            intent=intent.name,
            target=intent.lock.target,
            export_sha256=str(intent.lock.declared_export_hash),
            probe_results=self.probe_results,
            disk_before=0,
            disk_after_create=0,
            disk_after_teardown=0,
            disk_recovery_checked=False,
        )


def _materialized_environment(
    prefix: Path,
) -> tuple[MaterializedDryRunEnvironment, EnvironmentRegistry]:
    """Load the real fixture prefix's exact lock/export/probe artifacts.

    Parameters
    ----------
    prefix:
        Session hardlink-clone prefix selected explicitly by the CLI.

    Returns
    -------
    tuple[MaterializedDryRunEnvironment, EnvironmentRegistry]
        Production lifecycle lane and single-intent routing registry.
    """

    resolved_prefix = prefix.resolve(strict=True)
    if not (resolved_prefix / "conda-meta").is_dir():
        raise ValueError(
            f"dry-run environment is not a materialized conda prefix: {resolved_prefix}"
        )
    if os.environ.get("MENAGERIE_PLATFORM_LOCK"):
        intent, probe_results = _committed_fixture_intent(resolved_prefix)
        registry = EnvironmentRegistry(
            intents={intent.name: intent},
            phase_order=(DRY_RUN_PHASE,),
            small_set_target=True,
            hard_cap=None,
            global_split_guidance="single committed-lock real fixture environment",
        )
        return MaterializedDryRunEnvironment(resolved_prefix, intent, probe_results), registry
    artifact_root = resolved_prefix.parent / "artifacts"
    target = "round19-real-host"
    lock_path = artifact_root / f"{target}.lock"
    export_path = artifact_root / f"{target}.resolved.json"
    export_hash_path = artifact_root / f"{target}.resolved.sha256"
    probe_path = artifact_root / f"{target}.probes.json"
    required = (lock_path, export_path, export_hash_path, probe_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"dry-run environment fixture artifacts are missing: {missing}")
    probes = IntentProbes(("torch", "menagerie_round19_sentinel"), (), ())
    probe_results = parse_probe_receipt_bytes(probes, probe_path.read_bytes())
    lock = LockArtifacts(
        target=target,
        lock_path=lock_path,
        export_path=export_path,
        export_hash_path=export_hash_path,
        lock_bytes=lock_path.read_bytes(),
        export_bytes=export_path.read_bytes(),
        declared_export_hash=export_hash_path.read_text(encoding="utf-8").strip(),
    )
    intent = EnvironmentIntent(
        name="core",
        phase=DRY_RUN_PHASE,
        framework="pytorch",
        description="Round-19 real-prefix acceptance dry-run fixture",
        split_guidance="fixture-only",
        channels=("conda-forge",),
        dependencies=("python", "pytorch"),
        probes=probes,
        lock=lock,
        generation=None,
    )
    registry = EnvironmentRegistry(
        intents={intent.name: intent},
        phase_order=(DRY_RUN_PHASE,),
        small_set_target=True,
        hard_cap=None,
        global_split_guidance="single real fixture environment",
    )
    return MaterializedDryRunEnvironment(resolved_prefix, intent, probe_results), registry


class RecordingNotifier(FakeNotifier):
    """Persist fake notification calls so separate CLI processes remain observable."""

    def __init__(self, path: Path) -> None:
        """Bind the append-only fake notification log."""

        super().__init__()
        self.path = path

    def notify(
        self,
        summary: str,
        *,
        idempotency_key: str,
        notification_class: Optional[str] = None,
    ) -> bool:
        """Capture and append one ASCII notification summary."""

        del notification_class
        delivered = super().notify(summary, idempotency_key=idempotency_key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(canonical_json_bytes({"summary": summary}) + b"\n")
        return delivered


def _dry_run_provider_reconciliation(_idempotency_key: str) -> ProviderReconciliation:
    """Answer one paid-session status query as provider-accepted with zero wall.

    Parameters
    ----------
    _idempotency_key:
        Durable paid-session idempotency key, unused because the deterministic
        dry-run provider consummates every fake author/checker session.

    Returns
    -------
    ProviderReconciliation
        Constant accepted receipt, the same spelling as the unit-test fake so
        both deterministic compositions settle identically.
    """

    return ProviderReconciliation(True, 0.0, "sha256:fake-provider-accepted")


def create_dry_run_snapshot(campaign_root: Path) -> IntakeSnapshot:
    """Create or reload the immutable four-architecture dry-run intake snapshot."""

    sources = campaign_root / "intake-sources"
    sources.mkdir(parents=True, exist_ok=True)
    master = sources / "master.jsonl"
    deferred = sources / "deferred.jsonl"
    rows = [
        {"name": name, "zoo": "dry-run-fixtures", "variant": variant}
        for name, variant in DRY_RUN_ITEMS
    ]
    master.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    deferred.write_bytes(b"")
    return create_intake_snapshot(master, deferred, campaign_root / "intake")


def dry_run_paths(campaign_root: Path, snapshot: IntakeSnapshot) -> DriverPaths:
    """Return isolated runtime and canonical ledger paths for one dry-run campaign."""

    records = campaign_root / "records"
    return DriverPaths(
        campaign_root / "runtime",
        snapshot.root,
        LedgerPaths(
            records / "models" / "current-shard.jsonl",
            records / "attempts" / "local.jsonl",
            records / "gates" / "current-shard.jsonl",
        ),
    )


def build_dry_run_driver(
    repo_root: Path,
    campaign_root: Path,
    environment_prefix: Path,
    *,
    review_checkpoint_at: Optional[int],
    progress_milestones: tuple[int, ...],
    run_id: str,
    scheduler: str = "waves",
) -> CrawlerDriver:
    """Build the real driver with only author/checker/notifier lanes deterministic.

    Parameters
    ----------
    repo_root, campaign_root, environment_prefix:
        Checked-out source, disposable campaign, and explicit real environment prefix.
    review_checkpoint_at, progress_milestones, run_id:
        Ordinary driver checkpoint, notification, and invocation configuration.
    scheduler:
        Leg scheduler the acceptance dry-run must actually use. This is the only
        shipped end-to-end path with a real worker, a real environment binder, and
        a real reducer, so a scheduler that cannot reach it has no rehearsal
        vehicle at any effort level. Defaults to the historical ``waves``.

    Returns
    -------
    CrawlerDriver
        Driver using the shipped supervisor/compiler and strict materialized-environment binder.
    """

    snapshot = create_dry_run_snapshot(campaign_root)
    paths = dry_run_paths(campaign_root, snapshot)
    environments, registry = _materialized_environment(environment_prefix)
    author: TinyModelAuthor
    session_death_value = os.environ.get(INJECT_SESSION_DEATH_ENV)
    if os.environ.get(INJECT_ALL_SOURCE_FAILURE_ENV) == "1":
        author = AllSourceFailureAuthor()
    elif os.environ.get(INJECT_USAGE_PAUSE_ENV) == "author":
        author = UsagePauseAuthor(campaign_root / USAGE_PAUSE_MARKER)
    elif session_death_value:
        author = SessionDeathAuthor(
            parse_session_death_plan(session_death_value),
            stage=os.environ.get(INJECT_SESSION_DEATH_STAGE_ENV, "author"),
        )
    else:
        author = TinyModelAuthor()
    checker: FakeChecker
    checker_death_value = os.environ.get(INJECT_CHECKER_SESSION_DEATH_ENV)
    if checker_death_value:
        checker = SessionDeathChecker(parse_session_death_plan(checker_death_value))
    else:
        checker = FakeChecker()
    wake_scheduler = RecordingWakeScheduler(campaign_root / WAKE_OS_REGISTRATIONS)
    dependencies = DriverDependencies(
        author=author,
        checker=checker,
        forward=SupervisedForwardLane(timeout_seconds=30, cwd=repo_root),
        environments=environments,
        notifier=RecordingNotifier(campaign_root / "notifications.jsonl"),
        clock=dry_run_clock_now,
        # Always fake the OS-scheduler boundary, exactly as the unit builder does: a
        # usage pause inside a dry run must not register a real launchctl/crontab job
        # for a disposable campaign, and the real installer cannot work under a
        # sandbox or on a host without a user session. Every wake RECORD stays real.
        wakeup_installer=wake_scheduler.install,
        wakeup_verifier=wake_scheduler.verify,
        wakeup_deactivator=wake_scheduler.deactivate,
        # Settlement is fail-closed without a provider status capability, so the
        # deterministic composition must carry its fake provider's reconciler.
        provider_reconciler=_dry_run_provider_reconciliation,
    )
    config = DriverConfig(
        target="osx-arm64",
        run_id=run_id,
        machine_id=f"dry-run-{environment_prefix.name}",
        review_checkpoint_at=review_checkpoint_at,
        progress_milestones=progress_milestones,
        notify_command=None,
        author_model="fake-claude",
        author_version="dry-run",
        checker_model="fake-codex",
        checker_version="dry-run",
        # Valve/park emission requires a bound tier campaign (T2-12): the
        # unbound dry-run config was exactly how the pilot's campaign_id="None"
        # rows were minted.
        campaign_id="c1-mech",
        scheduler=scheduler,
    )
    return CrawlerDriver(paths, config, dependencies, registry=registry)


def read_notification_summaries(path: Path) -> tuple[str, ...]:
    """Read persisted fake notification summaries in append order."""

    if not path.is_file():
        return ()
    return tuple(
        str(json.loads(line)["summary"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
