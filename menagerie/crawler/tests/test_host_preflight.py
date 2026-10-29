"""DX-62: per-probe refuse shapes + census exhaustiveness for the host pack."""

from __future__ import annotations

import resource
from pathlib import Path

import pytest

from menagerie.crawler.tools import host_preflight
from menagerie.crawler.tools.host_preflight import (
    BOOT_IDENTITY_UNAVAILABLE,
    CLI_PIN_UNAVAILABLE,
    HOST_PREFLIGHT_ARMS,
    POWER_ASSERTION_UNHELD,
    RECORDS_FS_TYPE_FORBIDDEN,
    RLIMIT_NOFILE_TOO_LOW,
    evaluate_host_preflight,
    filesystem_type,
)

pytestmark = pytest.mark.smoke


def _pack(tmp_path: Path, **overrides: object) -> dict:
    """Run the pack over a healthy fixture with test-friendly floors."""

    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    arguments: dict = {
        "records_root": records,
        "roots": {"records-root": records},
        # The suite's own context may be descriptor-limited; per-probe floors
        # are exercised explicitly below.
        "minimum_nofile": 1,
    }
    arguments.update(overrides)
    return evaluate_host_preflight(**arguments)


def test_census_is_exactly_the_closed_vocabulary(tmp_path: Path) -> None:
    """Every probe registers, armed or unarmed — none can exist outside."""

    pack = _pack(tmp_path)
    names = [row["name"] for row in pack["armed_guard_census"]]
    assert tuple(names) == tuple(sorted(HOST_PREFLIGHT_ARMS))
    unarmed = {
        row["name"] for row in pack["armed_guard_census"] if not row["armed"]
    }
    # Foreign-seam probes disclose as unarmed until their owners wire them.
    assert {"ntp-offset", "token-validity", "model-receipts", "adapter-identity"} <= unarmed


def test_rlimit_floor_refuses_when_the_hard_cap_is_below(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 256-descriptor daemon context refuses; the probe cannot raise past hard."""

    monkeypatch.setattr(
        host_preflight.resource, "getrlimit", lambda _which: (256, 256)
    )
    monkeypatch.setattr(
        host_preflight.resource,
        "setrlimit",
        lambda _which, _limits: (_ for _ in ()).throw(ValueError("above hard cap")),
    )
    pack = _pack(tmp_path, minimum_nofile=4096)
    assert pack["ok"] is False
    assert any(RLIMIT_NOFILE_TOO_LOW in refusal for refusal in pack["refusals"])


def test_rlimit_probe_raises_soft_toward_hard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raiseable soft limit is raised, recorded, and passes the floor."""

    limits = {"current": (256, 10240)}

    def fake_getrlimit(_which: int) -> tuple[int, int]:
        return limits["current"]

    def fake_setrlimit(_which: int, new: tuple[int, int]) -> None:
        limits["current"] = new

    monkeypatch.setattr(host_preflight.resource, "getrlimit", fake_getrlimit)
    monkeypatch.setattr(host_preflight.resource, "setrlimit", fake_setrlimit)
    pack = _pack(tmp_path, minimum_nofile=4096)
    census = {row["name"]: row for row in pack["armed_guard_census"]}
    assert census["rlimit-nofile"]["evidence"]["initial"] == [256, 10240]
    assert census["rlimit-nofile"]["evidence"]["effective"][0] >= 4096
    assert not any(RLIMIT_NOFILE_TOO_LOW in refusal for refusal in pack["refusals"])


def test_boot_identity_failure_refuses_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No derivable boot generation refuses instead of minting a leg."""

    def raising() -> str:
        raise RuntimeError("cannot establish the current boot identity")

    monkeypatch.setattr(host_preflight, "current_boot_id", raising)
    pack = _pack(tmp_path)
    assert pack["ok"] is False
    assert any(BOOT_IDENTITY_UNAVAILABLE in refusal for refusal in pack["refusals"])


def test_records_fs_type_outside_the_allowlist_refuses(tmp_path: Path) -> None:
    """A records root on a non-durable filesystem is a typed refusal."""

    pack = _pack(tmp_path, fs_allowlist=("zfs",))
    assert pack["ok"] is False
    assert any(RECORDS_FS_TYPE_FORBIDDEN in refusal for refusal in pack["refusals"])


def test_records_fs_type_passes_on_this_host(tmp_path: Path) -> None:
    """The real host filesystem resolves and sits inside the allowlist."""

    fs_type = filesystem_type(tmp_path)
    assert fs_type is not None
    pack = _pack(tmp_path, fs_allowlist=(fs_type,))
    census = {row["name"]: row for row in pack["armed_guard_census"]}
    assert census["records-fs-type"]["evidence"]["fs_type"] == fs_type
    assert not any(
        RECORDS_FS_TYPE_FORBIDDEN in refusal for refusal in pack["refusals"]
    )


def test_missing_declared_root_refuses_through_volume_anchors(
    tmp_path: Path,
) -> None:
    """The pack re-proves roots in the acting process; a dead root refuses."""

    records = tmp_path / "records"
    records.mkdir()
    pack = _pack(
        tmp_path,
        roots={"records-root": records, "tmp-root": tmp_path / "unmounted" / "tmp"},
    )
    assert pack["ok"] is False
    assert any("tmp-root" in refusal for refusal in pack["refusals"])


def test_absent_cli_pin_refuses_and_present_pin_records_identity(
    tmp_path: Path,
) -> None:
    """Pinned CLIs are rotation axes: absence refuses, presence is recorded."""

    red = _pack(tmp_path, cli_pins={"claude": tmp_path / "no-such-cli"})
    assert red["ok"] is False
    assert any(CLI_PIN_UNAVAILABLE in refusal for refusal in red["refusals"])

    green = _pack(tmp_path, cli_pins={"echo": Path("/bin/echo")})
    census = {row["name"]: row for row in green["armed_guard_census"]}
    assert census["cli-versions"]["armed"] is True
    assert census["cli-versions"]["evidence"]["pins"]["echo"]["present"] is True


def test_unheld_darwin_power_assertion_refuses(tmp_path: Path) -> None:
    """A driver without its own sleep immunity cannot open a Darwin leg."""

    pack = _pack(
        tmp_path,
        power_assertion_state={
            "mechanism": "caffeinate",
            "held": False,
            "pid": None,
            "detail": "spawn failed",
        },
    )
    assert pack["ok"] is False
    assert any(POWER_ASSERTION_UNHELD in refusal for refusal in pack["refusals"])

    held = _pack(
        tmp_path,
        power_assertion_state={
            "mechanism": "caffeinate",
            "held": True,
            "pid": 7,
            "detail": None,
        },
    )
    assert not any(
        POWER_ASSERTION_UNHELD in refusal for refusal in held["refusals"]
    )


def test_healthy_pack_is_green_and_census_complete(tmp_path: Path) -> None:
    """The composed step passes on a healthy host and keeps real rlimits."""

    soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    pack = _pack(tmp_path, minimum_nofile=min(soft, 256))
    assert pack["ok"] is True, pack["refusals"]
    census = {row["name"]: row for row in pack["armed_guard_census"]}
    assert census["boot-id"]["evidence"]["boot_id"]
    assert census["libomp-census"]["armed"] is True
