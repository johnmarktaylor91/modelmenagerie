"""OS-scheduler integration tests for recurring wake projections (A3).

The driver suite stubs the OS-scheduler boundary everywhere (installing a real
launchd job from a unit test fails under Seatbelt and on any CI host without a
per-user GUI session, and would mutate the developer's real scheduler where it
succeeds). That left ``wakeup._install_wakeup``'s real ``launchctl`` path with
zero coverage on any platform.

This module closes that with the portable two-leg pattern:

1. **Always-on artifact shape** -- every platform parses the exact bytes each
   backend renderer would hand to its OS scheduler (launchd plist through
   ``plistlib``, systemd timer/service units, the marked cron block) and pins
   their shape, including argv escaping.
2. **Gated real roundtrip** -- macOS only, executes the real ``launchctl``
   bootstrap/print/bootout cycle against the rendered plist and runtime-skips
   with a typed reason when the host denies bootstrap (Seatbelt confinement or
   a session without a per-user launchd domain) rather than failing red.
"""

from __future__ import annotations

import os
from pathlib import Path
import plistlib
import shlex
import shutil
import subprocess
import sys

import pytest

from menagerie.crawler.wakeup import (
    LAUNCHD_AGENT_ROOT_ENV,
    WakeupBackend,
    WakeupConfigurationError,
    _deactivate_wakeup,
    _install_wakeup,
    _launchd_label,
    _verify_wakeup,
    _write_definition,
    build_wake_episode,
    render_wakeup_definition,
)

#: Typed skip reasons: a reader of a skip line must be able to tell "this host
#: cannot run the real scheduler leg" from "the test rotted".
SKIP_NOT_MACOS = "wakeup-os-integration: real launchd roundtrip requires macOS"
SKIP_NO_LAUNCHCTL = "wakeup-os-integration: launchctl is not on PATH"
SKIP_BOOTSTRAP_DENIED = (
    "wakeup-os-integration: launchd bootstrap denied (OS sandbox or no per-user GUI session)"
)

#: An argv argument exercising every XML-sensitive byte the plist escaper must
#: survive, plus a shell metacharacter the cron/systemd renderers must quote.
_HOSTILE_ARGUMENT = "--note=<&>\"' $(reboot)"


def _episode(callback_argv: tuple[str, ...] = ("/usr/bin/true",)):
    """Build one valid frozen wake episode for rendering tests.

    Parameters
    ----------
    callback_argv:
        Base callback argv before the episode identity argument is appended.

    Returns
    -------
    menagerie.crawler.authority.WakeEpisode
        Frozen episode value.
    """

    return build_wake_episode(
        provider="anthropic",
        reset_at="2026-08-12T18:00:00Z",
        reset_observation="observed",
        callback_argv=callback_argv,
        retry_interval_seconds=900,
    )


@pytest.mark.smoke
def test_launchd_plist_projection_parses_and_pins_scheduler_contract(
    tmp_path: Path,
) -> None:
    """The rendered plist is exactly what launchd would load, argv intact.

    Parameters
    ----------
    tmp_path:
        Isolated wakeup projection root.
    """

    episode = _episode(("/usr/bin/true", _HOSTILE_ARGUMENT))
    definition = render_wakeup_definition(tmp_path, episode, WakeupBackend.LAUNCHD)

    assert definition.definition_path.suffix == ".plist"
    assert definition.service_path is None
    payload = plistlib.loads(definition.scheduler_definition.encode("utf-8"))
    # REQUIRED SUBSET, never an exact set. The exact-set form pinned the four
    # keys of the AU-1 defect and therefore FORBADE its fix: adding the
    # WorkingDirectory and PYTHONPATH the callback needs to import its own
    # package turned this smoke test red (TRACKB_opus F11). What the plist must
    # not lose is asserted positively below; what it may gain is the renderer's
    # business. Executability itself is proven by running the rendered callback
    # under launchd's conditions in test_wake_callback_liveness.
    assert {"Label", "ProgramArguments", "StartInterval", "RunAtLoad"} <= set(payload)
    assert payload["Label"] == _launchd_label(episode)
    assert payload["Label"] == f"org.menagerie.crawler-wake.{episode.episode_id}"
    # The escaper must round-trip the hostile argument byte-for-byte and the
    # rendered argv must carry the episode identity for the fired callback.
    assert payload["ProgramArguments"] == list(episode.callback_argv)
    assert _HOSTILE_ARGUMENT in payload["ProgramArguments"]
    assert payload["ProgramArguments"][-2:] == ["--wake-episode-id", episode.episode_id]
    assert payload["StartInterval"] == episode.retry_interval_seconds
    assert payload["RunAtLoad"] is True
    # The two keys whose absence made every production fire die pre-import.
    assert Path(payload["WorkingDirectory"]).is_absolute()
    assert payload["EnvironmentVariables"]["PYTHONPATH"] == payload["WorkingDirectory"]


@pytest.mark.smoke
def test_systemd_timer_projection_pins_unit_pair_shape(tmp_path: Path) -> None:
    """The systemd projection renders a linked timer/service pair with quoted argv.

    Parameters
    ----------
    tmp_path:
        Isolated wakeup projection root.
    """

    episode = _episode(("/usr/bin/true", _HOSTILE_ARGUMENT))
    definition = render_wakeup_definition(tmp_path, episode, WakeupBackend.SYSTEMD_TIMER)

    assert definition.definition_path.suffix == ".timer"
    assert definition.service_path is not None
    assert definition.service_path.suffix == ".service"
    assert definition.service_definition is not None
    assert f"OnUnitInactiveSec={episode.retry_interval_seconds}s" in (
        definition.scheduler_definition
    )
    assert f"Unit={definition.service_path.name}" in definition.scheduler_definition
    exec_start = next(
        line for line in definition.service_definition.splitlines() if line.startswith("ExecStart=")
    )
    # shlex round-trip: the hostile argument survives quoting exactly.
    assert shlex.split(exec_start.removeprefix("ExecStart=")) == list(episode.callback_argv)


@pytest.mark.smoke
def test_cron_projection_pins_marked_block_shape(tmp_path: Path) -> None:
    """The cron projection is one marked, removable, exactly-quoted block.

    Parameters
    ----------
    tmp_path:
        Isolated wakeup projection root.
    """

    episode = _episode(("/usr/bin/true", _HOSTILE_ARGUMENT))
    definition = render_wakeup_definition(tmp_path, episode, WakeupBackend.CRON)

    lines = definition.scheduler_definition.splitlines()
    marker = f"# menagerie-wake-episode {episode.episode_id}"
    assert lines[0] == f"{marker} begin"
    assert lines[-1] == f"{marker} end"
    assert "CRON_TZ=UTC" in lines
    cadence_minutes = episode.retry_interval_seconds // 60
    schedule_line = next(line for line in lines if line.startswith("*/"))
    prefix = f"*/{cadence_minutes} * * * * "
    assert schedule_line.startswith(prefix)
    # cron runs from $HOME with no PYTHONPATH, so the command is a `cd` + env
    # prefix around the callback. The argv still has to survive quoting exactly:
    # split off the prefix and round-trip the tail.
    command = schedule_line.removeprefix(prefix)
    words = shlex.split(command)
    assert words[0] == "cd"
    assert Path(words[1]).is_absolute()
    assert words[2] == "&&"
    assert words[3] == f"PYTHONPATH={words[1]}"
    assert words[4:] == list(episode.callback_argv)


@pytest.mark.skipif(sys.platform != "darwin", reason=SKIP_NOT_MACOS)
def test_real_launchd_bootstrap_verify_and_bootout_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real launchctl path installs, verifies, and deactivates one episode.

    Covers ``_install_wakeup``'s launchd branch on the only platform that has
    it: bootstrap into the per-user GUI domain, ``launchctl print``-backed
    verification, and idempotent bootout. The callback is ``/usr/bin/true``
    (with a per-process disambiguating argument so concurrent suite runs on one
    host cannot collide on the launchd label), so the single ``RunAtLoad``
    execution a successful bootstrap triggers is harmless.

    Hosts that deny bootstrap -- Seatbelt-confined runs and CI sessions without
    a per-user launchd domain -- skip with the typed reason instead of failing.

    The agent root is redirected into ``tmp_path`` because the launchd install
    is boot-persistent now (BL-6 (b)): a test may never write into the
    operator's real ``~/Library/LaunchAgents``.

    Parameters
    ----------
    tmp_path:
        Isolated wakeup projection root.
    monkeypatch:
        Redirects the login-persistent agent root away from the real one.
    """

    if shutil.which("launchctl") is None:
        pytest.skip(SKIP_NO_LAUNCHCTL)
    monkeypatch.setenv(LAUNCHD_AGENT_ROOT_ENV, str(tmp_path / "LaunchAgents"))
    episode = _episode(("/usr/bin/true", f"--suite-process={os.getpid()}"))
    definition = render_wakeup_definition(tmp_path, episode, WakeupBackend.LAUNCHD)
    _write_definition(definition)
    domain_target = f"gui/{os.getuid()}/{_launchd_label(episode)}"
    try:
        try:
            _install_wakeup(definition)
        except WakeupConfigurationError as exc:
            assert "launchctl" in str(exc)
            pytest.skip(f"{SKIP_BOOTSTRAP_DENIED}: {exc}")
        assert _verify_wakeup(definition) is True
        _deactivate_wakeup(definition)
        assert _verify_wakeup(definition) is False
    finally:
        # Never leave a scheduler registration behind, whatever the outcome.
        subprocess.run(
            ["launchctl", "bootout", domain_target],
            check=False,
            capture_output=True,
            text=True,
        )
