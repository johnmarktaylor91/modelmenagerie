"""PD-8: the ``state-placement`` arm and its launchd-context probe instrument.

The TCC seam this arm guards (rehearsal evidence ``r2-tcc-launchd-probe.json``,
2026-08-25): a launchd-spawned process gets ``EPERM`` on removable volumes in
BOTH directions while the SAME command works from a Terminal shell, so the W4
roots-contract sentinel run from a Terminal-context launch proves nothing about
what the wake/resume path can touch. The placement arm refuses AT LAUNCH any
wake-touched root off the launchd-proven anchor volume; the probe tool is the
ground-truth instrument that proves the anchor from a REAL launchd context.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from menagerie.crawler.tools.launch_preflight import (
    EXPECTED_PREFLIGHT_ARMS,
    STATE_PLACEMENT_UNSAFE,
    evaluate_state_placement,
)
from menagerie.crawler.tools.tcc_launchd_probe import (
    PROBE_TIMEOUT_ERROR,
    TCC_PLACEMENT_UNPROVEN,
    TCC_PROBE_RECEIPT_SCHEMA_VERSION,
    judge_receipt,
    parse_root_specs,
    render_probe_plist,
    run_payload,
)

pytestmark = pytest.mark.smoke


def _fake_devices(mapping: dict[Path, int]):
    """Return a ``device_of`` that resolves through the given path table."""

    def device_of(path: Path) -> int:
        for candidate, device in mapping.items():
            if path == candidate or candidate in path.parents:
                return device
        raise AssertionError(f"unmapped device probe: {path}")

    return device_of


# --- the placement evaluator ---


def test_state_placement_arm_is_in_the_closed_census() -> None:
    """The arm exists in the closed vocabulary, per the W4 registry rule."""

    assert "state-placement" in EXPECTED_PREFLIGHT_ARMS


def test_anchor_volume_roots_pass_with_device_evidence(tmp_path: Path) -> None:
    """Roots sharing the anchor device pass and carry placement evidence."""

    home = tmp_path / "home"
    records = tmp_path / "records"
    home.mkdir()
    records.mkdir()
    result = evaluate_state_placement(
        {"records-root": records},
        account_home=home,
        platform_name="darwin",
    )
    assert result["ok"] is True
    assert result["enforced"] is True
    rows = {row["root"]: row for row in result["rows"]}
    assert rows["records-root"]["placement"] == "anchor-volume"
    assert rows["records-root"]["st_dev"] == result["anchor"]["st_dev"]


def test_foreign_volume_root_refuses_typed(tmp_path: Path) -> None:
    """A wake-touched root on another device refuses with the TCC rationale."""

    home = tmp_path / "home"
    ssd = tmp_path / "ssd-state"
    home.mkdir()
    ssd.mkdir()
    result = evaluate_state_placement(
        {"state-root": ssd},
        account_home=home,
        platform_name="darwin",
        device_of=_fake_devices({home: 1, ssd: 2}),
    )
    assert result["ok"] is False
    (refusal,) = [
        item for item in result["refusals"] if "state-root" in item
    ]
    assert refusal.startswith(STATE_PLACEMENT_UNSAFE)
    assert "EPERM" in refusal
    assert "launchd" in refusal
    rows = {row["root"]: row for row in result["rows"]}
    assert rows["state-root"]["placement"] == "foreign-volume"


def test_missing_declared_root_is_unprovable_and_refuses(tmp_path: Path) -> None:
    """Placement of a missing declared root never walks up to a healthy parent."""

    home = tmp_path / "home"
    home.mkdir()
    result = evaluate_state_placement(
        {"env-root": tmp_path / "unmounted" / "volume"},
        account_home=home,
        platform_name="darwin",
    )
    assert result["ok"] is False
    (refusal,) = [item for item in result["refusals"] if "env-root" in item]
    assert refusal.startswith(STATE_PLACEMENT_UNSAFE)
    assert "unmounted" in refusal


def test_undeclared_roots_are_recorded_never_asserted(tmp_path: Path) -> None:
    """A ``None`` root is a disclosure row, not a refusal."""

    home = tmp_path / "home"
    home.mkdir()
    result = evaluate_state_placement(
        {"tmp-root": None},
        account_home=home,
        platform_name="darwin",
    )
    assert result["ok"] is True
    rows = {row["root"]: row for row in result["rows"]}
    assert rows["tmp-root"]["declared"] is False


def test_non_darwin_records_placement_without_refusing(tmp_path: Path) -> None:
    """The guarded seam is macOS TCC; elsewhere the arm discloses, unenforced."""

    home = tmp_path / "home"
    foreign = tmp_path / "foreign"
    home.mkdir()
    foreign.mkdir()
    result = evaluate_state_placement(
        {"state-root": foreign},
        account_home=home,
        platform_name="linux",
        device_of=_fake_devices({home: 1, foreign: 2}),
    )
    assert result["ok"] is True
    assert result["enforced"] is False
    rows = {row["root"]: row for row in result["rows"]}
    assert rows["state-root"]["placement"] == "foreign-volume"


def test_derived_infrastructure_roots_are_censused(tmp_path: Path) -> None:
    """The wake plist root and the execution-flock home appear derived."""

    home = tmp_path / "home"
    home.mkdir()
    result = evaluate_state_placement(
        {},
        account_home=home,
        platform_name="darwin",
    )
    assert result["ok"] is True
    rows = {row["root"]: row for row in result["rows"]}
    assert rows["launchd-agent-root"]["derived"] is True
    assert rows["execution-flock-home"]["derived"] is True
    # Neither exists under the synthetic home yet: placement is proven on the
    # nearest existing ancestor, which mkdir semantics bind to the anchor.
    assert rows["launchd-agent-root"]["placement"] == "anchor-volume"


def test_infrastructure_root_escaping_the_anchor_refuses(tmp_path: Path) -> None:
    """A symlinked ancestor resolving outside the home refuses, fail-closed."""

    home = tmp_path / "home"
    elsewhere = tmp_path / "elsewhere"
    home.mkdir()
    elsewhere.mkdir()
    (home / "Library").symlink_to(elsewhere)
    result = evaluate_state_placement(
        {},
        account_home=home,
        platform_name="darwin",
    )
    assert result["ok"] is False
    (refusal,) = [
        item for item in result["refusals"] if "launchd-agent-root" in item
    ]
    assert refusal.startswith(STATE_PLACEMENT_UNSAFE)
    rows = {row["root"]: row for row in result["rows"]}
    assert rows["launchd-agent-root"]["placement"] == "escapes-anchor"


def test_unstatable_anchor_makes_every_placement_unprovable(tmp_path: Path) -> None:
    """No anchor, no proof: everything refuses rather than passing silently."""

    home = tmp_path / "home"
    records = tmp_path / "records"
    home.mkdir()
    records.mkdir()

    def refusing_device(path: Path) -> int:
        raise OSError("synthetic stat failure")

    result = evaluate_state_placement(
        {"records-root": records},
        account_home=home,
        platform_name="darwin",
        device_of=refusing_device,
    )
    assert result["ok"] is False
    assert any("account home" in item for item in result["refusals"])
    rows = {row["root"]: row for row in result["rows"]}
    assert rows["records-root"]["placement"] == "unprovable"


# --- the launchd probe instrument ---


def test_parse_root_specs_refuses_malformed_specs() -> None:
    """NAME=PATH parsing refuses missing separators, dupes, relative paths."""

    parsed = parse_root_specs(["state=/tmp/state"])
    assert parsed == {"state": Path("/tmp/state")}
    for bad in (["nopath"], ["=path"], ["name="], ["a=/x", "a=/y"], ["rel=x/y"]):
        with pytest.raises(ValueError):
            parse_root_specs(bad)


def test_payload_writes_a_complete_receipt(tmp_path: Path) -> None:
    """The payload proves writability with the real sentinel and records context."""

    root = tmp_path / "root"
    root.mkdir()
    emit = tmp_path / "receipt.json"
    receipt = run_payload(emit=emit, roots={"state": root}, expect_denied={})
    persisted = json.loads(emit.read_bytes())
    assert persisted == json.loads(json.dumps(receipt))
    assert persisted["schema_version"] == TCC_PROBE_RECEIPT_SCHEMA_VERSION
    (row,) = persisted["roots"]
    assert row["name"] == "state"
    assert row["read"]["readable"] is True
    assert row["write"]["writable"] is True
    assert list(root.iterdir()) == []  # the sentinel never litters


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
def test_payload_records_a_denied_root(tmp_path: Path) -> None:
    """A permission-refused root records its typed error, never raises."""

    denied = tmp_path / "denied"
    denied.mkdir()
    denied.chmod(0o000)
    try:
        receipt = run_payload(
            emit=tmp_path / "receipt.json",
            roots={},
            expect_denied={"ssd": denied},
        )
    finally:
        denied.chmod(0o700)
    (row,) = receipt["expect_denied"]
    assert row["write"]["writable"] is False
    assert row["write"]["error"] == "PermissionError"


def test_probe_timeout_converts_a_hang_into_a_typed_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blocking syscall records ProbeTimeout instead of hanging the payload.

    Observed live (2026-08-25): from a launchd-spawned venv python, the TCC
    consent evaluation on a removable volume BLOCKS the syscall indefinitely
    instead of returning EPERM — a wake act would go DARK, not red.
    """

    import time as time_module

    root = tmp_path / "root"
    root.mkdir()

    def hanging_listdir(path):
        time_module.sleep(3.0)
        return []

    monkeypatch.setattr(os, "listdir", hanging_listdir)
    receipt = run_payload(
        emit=tmp_path / "receipt.json",
        roots={},
        expect_denied={"ssd": root},
        probe_timeout_seconds=0.2,
    )
    (row,) = receipt["expect_denied"]
    assert row["read"]["readable"] is False
    assert row["read"]["error"] == PROBE_TIMEOUT_ERROR


def test_judge_accepts_a_probe_timeout_as_denial_evidence(tmp_path: Path) -> None:
    """The consent-evaluation hang is launchd-context denial, judged green."""

    receipt = _launchd_receipt(tmp_path)
    receipt["expect_denied"] = [
        {
            "name": "ssd",
            "path": str(tmp_path / "ssd"),
            "read": {"readable": False, "error": PROBE_TIMEOUT_ERROR},
            "write": {"writable": False, "error": PROBE_TIMEOUT_ERROR},
        }
    ]
    verdict = judge_receipt(
        receipt,
        roots={"state": tmp_path / "root"},
        expect_denied={"ssd": tmp_path / "ssd"},
    )
    assert verdict == {"ok": True, "refusals": []}


def _launchd_receipt(tmp_path: Path) -> dict:
    """Return a green-shaped receipt as a REAL launchd payload would mint it."""

    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    receipt = run_payload(
        emit=tmp_path / "receipt.json", roots={"state": root}, expect_denied={}
    )
    receipt = dict(receipt)
    receipt["ppid"] = 1
    receipt["launchd_spawned"] = True
    return receipt


def test_judge_green_on_a_complete_launchd_receipt(tmp_path: Path) -> None:
    """A launchd-spawned receipt covering every root judges green."""

    receipt = _launchd_receipt(tmp_path)
    verdict = judge_receipt(
        receipt, roots={"state": tmp_path / "root"}, expect_denied={}
    )
    assert verdict == {"ok": True, "refusals": []}


def test_judge_refuses_a_terminal_context_receipt(tmp_path: Path) -> None:
    """A receipt minted outside launchd proves nothing and refuses typed."""

    root = tmp_path / "root"
    root.mkdir()
    receipt = run_payload(
        emit=tmp_path / "receipt.json", roots={"state": root}, expect_denied={}
    )
    assert receipt["launchd_spawned"] is False  # pytest is not launchd
    verdict = judge_receipt(receipt, roots={"state": root}, expect_denied={})
    assert verdict["ok"] is False
    assert any("not launchd-spawned" in item for item in verdict["refusals"])
    assert all(item.startswith(TCC_PLACEMENT_UNPROVEN) for item in verdict["refusals"])


def test_judge_refuses_a_receipt_that_dropped_a_root(tmp_path: Path) -> None:
    """Passing by omission is refused: every asked root must carry a probe."""

    receipt = _launchd_receipt(tmp_path)
    verdict = judge_receipt(
        receipt,
        roots={"state": tmp_path / "root", "records": tmp_path / "records"},
        expect_denied={},
    )
    assert verdict["ok"] is False
    assert any("records" in item for item in verdict["refusals"])


def test_judge_refuses_when_the_negative_control_succeeds(tmp_path: Path) -> None:
    """A writable negative control means the probe is not measuring TCC."""

    receipt = _launchd_receipt(tmp_path)
    receipt["expect_denied"] = [
        {
            "name": "ssd",
            "path": str(tmp_path / "ssd"),
            "read": {"readable": True, "n_entries": 0},
            "write": {"writable": True},
        }
    ]
    verdict = judge_receipt(
        receipt,
        roots={"state": tmp_path / "root"},
        expect_denied={"ssd": tmp_path / "ssd"},
    )
    assert verdict["ok"] is False
    assert any("negative-control" in item for item in verdict["refusals"])


def test_render_probe_plist_carries_argv_label_and_runatload(tmp_path: Path) -> None:
    """The one-shot agent definition binds label, payload argv, and RunAtLoad."""

    content = render_probe_plist(
        label="org.menagerie.tcc-probe.test",
        python_executable="/usr/bin/python3",
        repo_root=tmp_path,
        receipt_path=tmp_path / "receipt.json",
        log_root=tmp_path,
        roots={"state": tmp_path / "state"},
        expect_denied={"ssd": Path("/Volumes/X & Y/state")},
    )
    assert "<key>RunAtLoad</key><true/>" in content
    assert "org.menagerie.tcc-probe.test" in content
    assert "--payload" in content
    assert f"<string>state={tmp_path / 'state'}</string>" in content
    assert "<string>ssd=/Volumes/X &amp; Y/state</string>" in content
    assert f"<key>PYTHONPATH</key><string>{tmp_path}</string>" in content
