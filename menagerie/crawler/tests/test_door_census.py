"""Door census: never-executed production stages are disclosed, never silent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.tools.door_census import (
    PRODUCTION_STAGES,
    door_census,
    main,
)


def _rows_for(stages: list[str]) -> list[dict[str, str]]:
    """Build one attempt row per stage name."""

    return [{"stage": stage, "stable_id": f"m{index}"} for index, stage in enumerate(stages)]


@pytest.mark.smoke
def test_all_doors_exercised_reads_clean() -> None:
    """Every stage at or above the floor discloses nothing."""

    census = door_census(_rows_for(sorted(PRODUCTION_STAGES)))
    assert census["status"] == "all-doors-exercised"
    assert census["dark_stages"] == []


@pytest.mark.smoke
def test_missing_stage_is_disclosed() -> None:
    """THE NAMED TRIPWIRE: a synthetic ledger missing a stage discloses it."""

    stages = sorted(PRODUCTION_STAGES)
    missing = stages.pop()
    census = door_census(_rows_for(stages))
    assert census["status"] == "dark-doors-disclosed"
    assert census["dark_stages"] == [missing]
    assert census["executions_by_stage"][missing] == 0


@pytest.mark.smoke
def test_foreign_stage_rows_are_accounted_not_counted() -> None:
    """A row with a stage outside the closed vocabulary never inflates a door."""

    census = door_census(_rows_for([*sorted(PRODUCTION_STAGES), "made-up-stage"]))
    assert census["foreign_stage_rows"] == {"made-up-stage": 1}
    assert "made-up-stage" not in census["executions_by_stage"]


@pytest.mark.smoke
def test_cli_refuse_dark_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """--refuse-dark makes the census composable as a gate."""

    ledger = tmp_path / "attempts.jsonl"
    ledger.write_text(
        "\n".join(json.dumps(row) for row in _rows_for(sorted(PRODUCTION_STAGES)[:-1])) + "\n",
        encoding="utf-8",
    )
    assert main(["--attempts", str(ledger), "--refuse-dark"]) == 2
    assert main(["--attempts", str(ledger)]) == 0
    capsys.readouterr()
