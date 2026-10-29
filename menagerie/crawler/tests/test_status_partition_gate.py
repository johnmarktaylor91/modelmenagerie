"""The rung-boundary partition check must be able to FAIL, not only report.

``crawler status --verify-partition`` is the boundary check the ladder protocol
runs at every rung close (procedures/RUN.md, LADDER_PROTOCOL). Before this pin it
computed the partition, printed ``"partition_valid": false``, and returned
``EXIT_OK`` -- so any gate scripted on the exit code (the only thing a supervisor,
a wake callback, or a CI job can key on) read a broken partition as a clean one.

Progress inspection mid-leg legitimately shows an incomplete partition, so the
default stays exit-0 and the fail-closed behaviour is the explicit
``--require-valid-partition`` boundary flag. The flag without ``--verify-partition``
is a usage error rather than a silent no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler import authority as authority_module
from menagerie.crawler.cli import EXIT_ERROR, EXIT_OK, EXIT_USAGE
from menagerie.crawler.cli import main as cli_main
from menagerie.crawler.driver import DriverPaths
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.tests.campaign_binding_support import bind_snapshot_to_campaign
from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot


def _status(tmp_path: Path, snapshot_root: Path, *extra: str) -> int:
    """Run ``crawler status`` against an isolated records root."""

    return cli_main(
        [
            "--repo-root",
            str(tmp_path),
            "status",
            "--intake",
            str(snapshot_root),
            "--records-root",
            str(tmp_path / "records"),
            *extra,
        ]
    )


@pytest.mark.smoke
def test_verify_partition_still_reports_without_refusing_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The historical contract: report the invalid partition, exit OK.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    capsys:
        Captured CLI output.
    """

    snapshot = _snapshot(tmp_path, count=2)
    bind_snapshot_to_campaign(tmp_path, snapshot)

    assert _status(tmp_path, snapshot.root, "--verify-partition") == EXIT_OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["partition_valid"] is False
    assert payload["missing"] == sorted(item.stable_id for item in snapshot.items)


@pytest.mark.smoke
def test_require_valid_partition_refuses_and_names_the_hole(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The boundary form fails closed and prints the exact coverage hole.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    capsys:
        Captured CLI output.
    """

    snapshot = _snapshot(tmp_path, count=2)
    bind_snapshot_to_campaign(tmp_path, snapshot)

    exit_code = _status(tmp_path, snapshot.root, "--verify-partition", "--require-valid-partition")
    captured = capsys.readouterr()
    assert exit_code == EXIT_ERROR
    # The payload is still printed: a refusing gate must remain diagnosable.
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert payload["partition_valid"] is False
    assert "terminal partition invalid" in captured.err
    for item in snapshot.items:
        assert item.stable_id in captured.err


@pytest.mark.smoke
def test_require_valid_partition_without_verify_partition_is_a_usage_error(
    tmp_path: Path,
) -> None:
    """The boundary flag never silently degrades to no verification at all.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    snapshot = _snapshot(tmp_path, count=1)
    bind_snapshot_to_campaign(tmp_path, snapshot)

    assert _status(tmp_path, snapshot.root, "--require-valid-partition") == EXIT_USAGE


@pytest.mark.smoke
def test_verify_partition_agrees_with_a_driver_closed_leg(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The boundary check must agree with the close that produced the ledgers.

    F12, both halves, end to end: a leg the driver closed ``complete`` (every
    model AWARDED, i.e. carrying executed revisions) must read
    ``partition_valid: true`` through ``status --verify-partition``. The CLI
    projection resolves awards as current only when it restores BOTH the
    registered prior-intake lineage and the per-family environment generations
    from ledger evidence -- omit either and a healthy, correctly-closed rung
    red-lights the boundary gate (the pilot mini-rung read 1 of 127 current).

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    capsys:
        Captured CLI output.
    """

    snapshot = _snapshot(tmp_path, count=3)
    bind_snapshot_to_campaign(tmp_path, snapshot)
    paths = DriverPaths(
        tmp_path / "runtime",
        snapshot.root,
        LedgerPaths(
            tmp_path / "records" / "models" / "current-shard.jsonl",
            tmp_path / "records" / "attempts" / "local.jsonl",
            tmp_path / "records" / "gates" / "current-shard.jsonl",
        ),
    )
    result = _driver(tmp_path, snapshot, paths_override=paths).run()
    assert result.status == "complete"

    code = _status(tmp_path, snapshot.root, "--verify-partition", "--require-valid-partition")
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == EXIT_OK
    assert payload["partition_valid"] is True
    assert payload["missing"] == []
    assert payload["terminal"] == 3
    assert payload["ledger_heads"] == 3
    assert payload["stale"]["count"] == 0


@pytest.mark.smoke
def test_verify_partition_after_a_mid_leg_authority_rotation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ROTATION arm this pin never had (BD-1; RED before the fix).

    The pin above closes a leg and checks the gate immediately: nothing rotates,
    no second leg, no code landing, no env rebuild -- so it could not express
    the only situation the gate exists for, a boundary crossed BETWEEN legs
    after a fix wave. That is why the suite stayed green while three
    independent labs ran the same command against the real step-100 books and
    got exit 1 with 24 recorded terminals (7 of them awards) reported
    ``missing`` and ``status:runs: 0`` in the funnel.

    With one shipped-policy identity rotated after the close, the gate must
    report ZERO missing members (every one holds a recorded head) and name
    the AXIS that moved. Before BD-1 the same fixture reported 3 phantom
    missing members, ``terminal: 0``, and an empty funnel; before RC-2 it
    failed closed as 3 stale heads. Since RC-2 the recorded basis pins
    unconditionally, so the partition is valid and the move is disclosed in
    the recheck census instead.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    capsys:
        Captured CLI output.
    monkeypatch:
        Rotates the shipped author-dispatcher component identity.
    """

    snapshot = _snapshot(tmp_path, count=3)
    bind_snapshot_to_campaign(tmp_path, snapshot)
    paths = DriverPaths(
        tmp_path / "runtime",
        snapshot.root,
        LedgerPaths(
            tmp_path / "records" / "models" / "current-shard.jsonl",
            tmp_path / "records" / "attempts" / "local.jsonl",
            tmp_path / "records" / "gates" / "current-shard.jsonl",
        ),
    )
    assert _driver(tmp_path, snapshot, paths_override=paths).run().status == "complete"

    # One mid-leg hotfix, expressed exactly as the campaign felt it: the
    # shipped dispatcher component's content identity is no longer the one
    # every recorded vector carries.
    monkeypatch.setattr(authority_module, "AUTHOR_DISPATCHER_COMPONENT", "author_pool.py")

    code = _status(tmp_path, snapshot.root, "--verify-partition", "--require-valid-partition")
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip().splitlines()[-1])
    # RC-2: the boundary stays GREEN -- every recorded head replays at its
    # recorded basis -- and the move is disclosed where it belongs, the
    # recheck census (an unmeasured producer-axis move reads soundness).
    assert code == 0, captured.err
    assert payload["partition_valid"] is True
    assert payload["missing"] == []
    assert payload["ledger_heads"] == 3
    assert payload["awards_stale_on_live_basis"] == 0
    assert payload["stale"]["by_axis"] == {}
    assert list(payload["recheck_census"]["by_axis"]) == ["author_dispatcher_identity"]
    assert payload["recheck_census"]["counts"]["needs_recheck"] == 3
