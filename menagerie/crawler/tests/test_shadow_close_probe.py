"""Shadow-close probe (DX-24): in-flight detection of would-refuse members.

The probe's whole value is fidelity: it must run the EXACT production close
predicate (``assert_partition`` via the status boundary form), read-only, and
convert "the close will die in four hours" into an exit code a monitor can key
on now. These tests pin both verdicts against real driver-produced roots: a
driver-closed leg probes clean, and a root with unaccounted members names them
in the refusal and exits 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.tests.campaign_binding_support import bind_snapshot_to_campaign
from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot
from menagerie.crawler.tools.shadow_close_probe import (
    PROBE_OK,
    PROBE_WOULD_REFUSE,
    main as probe_main,
)


def _probe(tmp_path: Path, snapshot_root: Path, *extra: str) -> int:
    """Run one probe invocation against an isolated records root."""

    return probe_main(
        [
            "--intake",
            str(snapshot_root),
            "--records-root",
            str(tmp_path / "records"),
            "--repo-root",
            str(tmp_path),
            *extra,
        ]
    )


@pytest.mark.smoke
def test_probe_is_silent_green_on_a_driver_closed_leg(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A leg the driver lawfully closed probes clean with exit 0."""

    snapshot = _snapshot(tmp_path, count=2)
    bind_snapshot_to_campaign(tmp_path, snapshot)
    assert _driver(tmp_path, snapshot).run().status == "complete"

    assert _probe(tmp_path, snapshot.root) == PROBE_OK
    out = capsys.readouterr()
    assert "partition would close" in out.out
    assert out.err == ""


@pytest.mark.smoke
def test_probe_names_would_refuse_members_and_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unaccounted member is named NOW, not at leg end.

    An intake with no records at all is the maximal coverage hole: every
    member is one the close would refuse. The refusal text is the production
    ``assert_partition`` text verbatim -- one implementation, zero drift.
    """

    snapshot = _snapshot(tmp_path, count=2)
    bind_snapshot_to_campaign(tmp_path, snapshot)

    assert _probe(tmp_path, snapshot.root) == PROBE_WOULD_REFUSE
    captured = capsys.readouterr()
    assert "terminal partition invalid" in captured.out
    for item in snapshot.items:
        assert item.stable_id in captured.out
    assert "WOULD REFUSE" in captured.err


@pytest.mark.smoke
def test_probe_json_carries_the_verdict_and_the_status_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The machine form carries verdict, refusal, and the full status payload."""

    snapshot = _snapshot(tmp_path, count=1)
    bind_snapshot_to_campaign(tmp_path, snapshot)

    assert _probe(tmp_path, snapshot.root, "--json") == PROBE_WOULD_REFUSE
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[0])
    assert payload["probe"] == "shadow-close"
    assert payload["would_refuse"] is True
    assert "terminal partition invalid" in payload["refusal"]
    assert payload["status"]["partition_valid"] is False
