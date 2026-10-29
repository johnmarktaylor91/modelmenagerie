"""The wake continuation contract: one vocabulary, both branches, round-trip (W15).

The adjudicated D1 defect: ``_wakeup_callback_argv`` appended the
environment-readiness receipt ONLY to the direct callback; the supervisor
branch rebuilt argv from a hand-enumerated flag list that never contained it,
the supervisor's run parser had no ``--environment-readiness-receipt`` option,
and ``_driver_command`` forwarded nothing -- so the FIRST real supervised
pause of a production campaign resumed into the readiness refusal (three flat
restarts, circuit open, rung dead: the leg-10/11 class). These tests pin the
repaired contract at every seam:

* the SUPERVISOR branch of the callback carries the receipt (the exact D1 RED);
* the supervisor parses and forwards the complete contract verbatim, derived
  from the one continuation vocabulary rather than a fourth hand copy;
* ``parse(render(config)) == config`` over the whole vocabulary, through the
  real crawler CLI.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from menagerie.crawler import cli as cli_module
from menagerie.crawler import supervisor as supervisor_module
from menagerie.crawler.continuation_manifest import (
    CONTINUATION_FLAG_SPECS,
    forwarded_continuation_argv,
    render_continuation_flags,
    supervisor_run_flag_names,
)
from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot


def _supervised_driver(tmp_path: Path):
    """Return a driver whose callback resolves to the SUPERVISOR branch."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    supervisor_shim = tmp_path / "menagerie" / "crawler" / "tools" / "crawler_supervisor.sh"
    supervisor_shim.parent.mkdir(parents=True, exist_ok=True)
    supervisor_shim.write_text("#!/bin/sh\n", encoding="utf-8")
    campaign_config = tmp_path / "campaign-config.json"
    campaign_config.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "environment-readiness-receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    driver.config = replace(
        driver.config,
        campaign_config_path=campaign_config,
        campaign_id="c1-mech",
        environment_readiness_receipt_path=receipt,
    )
    return driver, receipt


@pytest.mark.smoke
def test_supervisor_branch_callback_carries_the_readiness_receipt(
    tmp_path: Path,
) -> None:
    """The D1 pin: the supervisor form of the wake callback replays the receipt."""

    driver, receipt = _supervised_driver(tmp_path)
    argv = driver._wakeup_callback_argv()  # noqa: SLF001
    # This IS the supervisor branch, not the direct one.
    assert argv[0].endswith("menagerie/crawler/tools/crawler_supervisor.sh"), argv
    assert "--environment-readiness-receipt" in argv, (
        "the supervisor branch dropped the readiness receipt: the first real "
        f"supervised pause strands at the CLI readiness gate (argv: {argv})"
    )
    assert argv[argv.index("--environment-readiness-receipt") + 1] == str(receipt)
    # And the direct branch still carries it too (caaeb362's original fix).
    driver.config = replace(driver.config, campaign_config_path=None, campaign_id=None)
    direct = driver._wakeup_callback_argv()  # noqa: SLF001
    assert "--environment-readiness-receipt" in direct


@pytest.mark.smoke
def test_supervisor_parses_and_forwards_the_complete_contract(tmp_path: Path) -> None:
    """Callback -> supervisor parse -> child argv keeps every contract flag.

    The composed end-to-end D1 test: the exact supervisor-form argv the wake
    would execute is parsed by the REAL supervisor parser, and the driver
    command it would spawn still carries the receipt and every other
    continuation flag -- through to the real crawler CLI.
    """

    driver, receipt = _supervised_driver(tmp_path)
    argv = driver._wakeup_callback_argv()  # noqa: SLF001
    # The shim execs `python -m menagerie.crawler.supervisor <argv after
    # --python <exe>>` -- reproduce that parse with the real parser.
    supervisor_args = supervisor_module.build_parser().parse_args(
        [str(part) for part in argv[3:]]
    )
    forwarded = forwarded_continuation_argv(supervisor_args)
    assert "--environment-readiness-receipt" in forwarded
    assert forwarded[forwarded.index("--environment-readiness-receipt") + 1] == str(receipt)
    assert "--continuation-identity" in forwarded
    # The forwarded contract parses through the real crawler CLI resume form.
    parsed = cli_module.build_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "resume",
            "--campaign-config",
            str(driver.config.campaign_config_path),
            *forwarded,
        ]
    )
    assert str(parsed.environment_readiness_receipt) == str(receipt)


@pytest.mark.smoke
def test_continuation_vocabulary_round_trips_through_the_real_cli(
    tmp_path: Path,
) -> None:
    """parse(render(config)) == config over the WHOLE continuation vocabulary."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    config = replace(
        driver.config,
        phase="pytorch",
        redrive_targets=("m_0001", "m_0002"),
        redrive_reason="operator: round-trip probe",
        ack_stale_excluded=True,
        acknowledge_award_holds=True,
        environment_readiness_receipt_path=receipt,
    )
    flags = render_continuation_flags(config)
    parsed = cli_module.build_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "run",
            "--resume",
            "--intake",
            str(snapshot.root),
            "--target",
            config.target,
            "--run-id",
            config.run_id,
            *flags,
        ]
    )
    assert parsed.scheduler == config.scheduler
    assert parsed.phase == config.phase
    assert tuple(parsed.redrive_targets) == config.redrive_targets
    assert parsed.redrive_reason == config.redrive_reason
    assert parsed.ack_stale_excluded is True
    assert parsed.acknowledge_award_holds is True
    assert parsed.author_concurrency == config.author_concurrency
    assert parsed.checker_concurrency == config.checker_concurrency
    assert str(parsed.environment_readiness_receipt) == str(receipt)

    # Every vocabulary flag is accepted by the supervisor's run parser -- a
    # spelling added to the vocabulary without a parser entry fails HERE, not
    # at the first production wake.
    probe_values = []
    for spec in CONTINUATION_FLAG_SPECS:
        probe_values.append(spec.flag)
        if spec.kind != "bool":
            probe_values.append("1")
    supervisor_module.build_parser().parse_args(
        [
            "run",
            "--campaign-id",
            "c1-mech",
            "--repo-root",
            str(tmp_path),
            "--campaign-config",
            str(tmp_path / "campaign-config.json"),
            *probe_values,
        ]
    )
    assert supervisor_run_flag_names() == tuple(
        spec.flag for spec in CONTINUATION_FLAG_SPECS
    )
