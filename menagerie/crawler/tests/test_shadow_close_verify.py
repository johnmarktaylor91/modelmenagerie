"""Shadow-close verifier: new close-verdict classes alarm; disclosed debt doesn't."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.tools.shadow_close_verify import (
    ShadowCloseError,
    extract_classes,
    main,
    verify,
)


def _status(tmp_path: Path, name: str, **overrides) -> Path:
    """Write one verify-partition status payload fixture."""

    payload = {
        "partition_valid": True,
        "missing": [],
        "stale": {"unaccounted": [], "deferred_for_redrive": {}},
    }
    payload.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.smoke
def test_first_run_seeds_disclosed_baseline_without_alarm(tmp_path: Path) -> None:
    """Pre-existing debt is disclosed into the baseline, never alarmed."""

    status = _status(
        tmp_path, "s0.json", stale={"unaccounted": ["m-old"], "deferred_for_redrive": {}}
    )
    baseline = tmp_path / "baseline.json"
    alarms, had_baseline = verify(status, baseline)
    assert not had_baseline and alarms == {}
    stored = json.loads(baseline.read_text(encoding="utf-8"))
    assert stored["classes"]["unaccounted_stale"] == ["m-old"]


@pytest.mark.smoke
def test_injected_stale_axis_row_alarms(tmp_path: Path) -> None:
    """THE NAMED TRIPWIRE: a NEW unaccounted stale id after a leg alarms."""

    baseline = tmp_path / "baseline.json"
    verify(_status(tmp_path, "s0.json"), baseline)
    injected = _status(
        tmp_path,
        "s1.json",
        stale={"unaccounted": ["m-stale-axis"], "deferred_for_redrive": {}},
    )
    assert main(["--status-json", str(injected), "--baseline", str(baseline)]) == 2
    # The alarmed member joins the baseline: the SAME debt does not re-alarm,
    # a FURTHER new one does.
    assert main(["--status-json", str(injected), "--baseline", str(baseline)]) == 0
    further = _status(
        tmp_path,
        "s2.json",
        stale={"unaccounted": ["m-stale-axis", "m-two"], "deferred_for_redrive": {}},
    )
    assert main(["--status-json", str(further), "--baseline", str(baseline)]) == 2


@pytest.mark.smoke
def test_partition_flip_and_new_disposition_alarm(tmp_path: Path) -> None:
    """An invalid partition or a new deferral disposition is a new class."""

    baseline = tmp_path / "baseline.json"
    verify(_status(tmp_path, "s0.json"), baseline)
    flipped = _status(tmp_path, "s1.json", partition_valid=False)
    alarms, _ = verify(flipped, baseline)
    assert alarms == {"partition_invalid": ["partition-invalid"]}
    disposed = _status(
        tmp_path,
        "s2.json",
        partition_valid=False,
        stale={"unaccounted": [], "deferred_for_redrive": {"deferred:new-kind": ["m1"]}},
    )
    alarms, _ = verify(disposed, baseline)
    assert alarms == {"deferral_dispositions": ["deferred:new-kind"]}


@pytest.mark.smoke
def test_non_verifying_status_refuses(tmp_path: Path) -> None:
    """A status payload without a partition verdict refuses typed."""

    path = tmp_path / "plain.json"
    path.write_text(json.dumps({"members": 210}), encoding="utf-8")
    with pytest.raises(ShadowCloseError, match="verify-partition"):
        verify(path, tmp_path / "baseline.json")
    with pytest.raises(ShadowCloseError, match="partition verdict"):
        extract_classes({"members": 210})


@pytest.mark.smoke
def test_foreign_baseline_bytes_refuse(tmp_path: Path) -> None:
    """The verifier never diffs against a non-baseline artifact."""

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"classes": {}}), encoding="utf-8")
    with pytest.raises(ShadowCloseError, match="not a shadow-close baseline"):
        verify(_status(tmp_path, "s0.json"), baseline)
