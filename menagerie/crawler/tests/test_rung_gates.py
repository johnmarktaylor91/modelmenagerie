"""RED suite for the LADDER_PROTOCOL rung-binding gate checker.

The protocol binds specific merged capabilities to rung sizes ("before 500:
disk-GC report-only; before 500-1000: ledger-open perf, provider governance;
..."). Until now that map lived only in prose -- nothing refused staging a
rung whose bound gates were missing. The checker encodes the map with
import-capability probes and fails CLOSED: a gate whose probe target does not
exist is unmet, never skipped.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.smoke


def _gates():
    from menagerie.crawler.tools.rung_gates import evaluate_rung_gates

    return evaluate_rung_gates


def test_step_200_gates_are_all_satisfied() -> None:
    """Everything the protocol binds at or below 200 is merged on this base."""

    rows = _gates()(next_rung_size=200)
    binding = [row for row in rows if row["binds"] and row["blocking"]]
    assert all(row["satisfied"] for row in binding), [
        row["gate"] for row in binding if not row["satisfied"]
    ]


def test_rung_500_binds_the_500_gates() -> None:
    """disk-GC and the stratified tail bind at 500 and are satisfied."""

    rows = {row["gate"]: row for row in _gates()(next_rung_size=500)}
    assert rows["disk-gc-report-only"]["binds"] is True
    assert rows["disk-gc-report-only"]["satisfied"] is True
    assert rows["stratified-tail"]["satisfied"] is True
    assert rows["ledger-open-perf"]["satisfied"] is True


def test_rung_1000_reports_unbuilt_gates_unmet() -> None:
    """Provider governance / observability / catalog-epoch fail closed."""

    rows = {row["gate"]: row for row in _gates()(next_rung_size=1000)}
    assert rows["provider-governance"]["binds"] is True
    assert rows["provider-governance"]["satisfied"] is False
    assert rows["observability"]["satisfied"] is False
    assert rows["catalog-epoch"]["satisfied"] is False


def test_event_gates_report_without_blocking_size_checks() -> None:
    """Two-host and rotation gates are event-bound: visible, never size-bound."""

    rows = {row["gate"]: row for row in _gates()(next_rung_size=200)}
    assert rows["shard-merge"]["binds"] is False
    assert rows["shard-merge"]["satisfied"] is True
    assert rows["rotation-transaction"]["binds"] is False
    assert rows["rotation-transaction"]["satisfied"] is True


def test_cli_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    """Exit 0 when every binding gate is met; exit 2 when one is missing."""

    from menagerie.crawler.tools.rung_gates import main

    assert main(["--next-rung-size", "200"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["next_rung_size"] == 200
    assert main(["--next-rung-size", "1000"]) == 2
    err = capsys.readouterr()
    assert "provider-governance" in err.out + err.err
