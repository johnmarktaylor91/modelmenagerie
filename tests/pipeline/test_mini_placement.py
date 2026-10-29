"""Remote recipe placement keeps hub admission and canonical data separate."""

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from menagerie.pipeline import DataLayout, admission, drive, runner, tick, tick_full


def worker(**changes: Any) -> dict[str, Any]:
    return {"cap": 1, "budget_gib": 4, "peak_gib": 1, "ssh_alias": "worker-a-ssh",
            "account": "test-account", "runtime": ["claude"],
            "identity": ["runtime-identity", "--json"], **changes}


@pytest.fixture(autouse=True)
def host_roles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fleet host roles naming one phone-build mini, so no test reads the real registry."""
    import json

    path = tmp_path / "fleet" / "host-roles.json"  # outside tests' own tmp_path files
    path.parent.mkdir()
    path.write_text(json.dumps({"roles": {"phone-build": {"hosts": ["phone-mini"]}}}))
    monkeypatch.setenv("MENAGERIE_HOST_ROLES", str(path))
    return path


def test_off_does_not_read_leases(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(admission, "lanes", lambda **kw: pytest.fail("off reads leases"))
    assert mini_placement.slots({}) == []
    assert mini_placement.slots({"mini_workers": {"worker-a": {"cap": 0}}}) == []
    assert tick.DEFAULTS["mini_workers"] == {}


@pytest.mark.parametrize("changes", [
    {"peak_gib": 0}, {"cap": 3, "budget_gib": 12}, {"cap": True}, {"budget_gib": float("nan")},
    {"peak_gib": -1}, {"runtime": "claude"}, {"ssh_alias": "-bad"}, {"deliver": []},
    {"phone_first": "yes"},
])
def test_invalid_workers_get_no_slot_and_a_status_line(changes: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(admission, "lanes", lambda **kw: [])
    monkeypatch.setattr(mini_placement, "serving", lambda: True)
    cfg = {"mini_workers": {"worker-a": worker(**changes)}}
    assert mini_placement.slots(cfg) == []
    assert mini_placement.status(cfg)["worker-a"].startswith("off: ")


def test_missing_runtime_config_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    config = {k: v for k, v in worker().items() if k not in ("runtime", "identity")}
    line = mini_placement.status({"mini_workers": {"worker-a": config}})["worker-a"]
    assert line == "off: mini runtime not configured (runtime, identity missing)"
    monkeypatch.setattr(mini_placement, "serving", lambda: True)
    with pytest.raises(admission.Refused, match="runtime not configured"):
        mini_placement.launch(["f-one"], role="model-implementation", data=DataLayout(Path("/nonexistent")),
                              host="worker-a", config=config)


def test_runtime_profile_comes_from_the_host_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from menagerie.pipeline import mini_placement

    registry = tmp_path / "compute-hosts.json"
    registry.write_text(json.dumps({"hosts": {"worker-a": {"capabilities": {"runtime_profiles": [
        {"name": "pool-claude", "runtime": ["fleet-runtime"], "identity": ["fleet-identity"],
         "deliver": ["fleet-deliver", "{job}"]}]}}}}))
    monkeypatch.setenv("MENAGERIE_COMPUTE_HOSTS", str(registry))
    config = {k: v for k, v in worker().items() if k not in ("runtime", "identity")}
    spec = mini_placement._spec("worker-a", {**config, "runtime_profile": "pool-claude"})
    assert spec["runtime"] == ["fleet-runtime"] and spec["deliver"] == ["fleet-deliver", "{job}"]
    line = mini_placement.status({"mini_workers": {"worker-a": {**config, "runtime_profile": "other"}}})
    assert line["worker-a"] == "off: runtime profile 'other' not in the host registry for worker-a"


def test_malformed_entries_are_isolated_and_named(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(mini_placement, "serving", lambda: True)
    monkeypatch.setattr(admission, "lanes", lambda **kw: [])
    assert mini_placement.slots({"mini_workers": ["worker-a"]}) == []
    assert mini_placement.status({"mini_workers": ["worker-a"]})["mini_workers"].startswith("off: ")
    cfg = {"mini_workers": {"worker-a": "not a mapping", "bad name!": worker(), "worker-b": worker()}}
    assert mini_placement.slots(cfg) == ["worker-b"]
    lines = mini_placement.status(cfg)
    assert lines["worker-a"].startswith("off: ") and lines["bad name!"].startswith("off: ")


def test_phone_build_minis_need_an_idle_phone_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    assert mini_placement._phone_build("phone-mini") is True
    assert mini_placement._phone_build("worker-a") is False
    monkeypatch.setenv("MENAGERIE_HOST_ROLES", "/nonexistent/host-roles.json")
    assert mini_placement._phone_build("worker-a") is True  # unknown: phone builds first
    monkeypatch.setattr(mini_placement, "serving", lambda: True)
    monkeypatch.setattr(mini_placement, "_phone_idle", lambda spec: False)
    with pytest.raises(admission.Refused, match="phone builds first"):
        mini_placement.launch(["f-one"], role="model-implementation", data=DataLayout(Path("/nonexistent")),
                              host="worker-a", config=worker())


def test_leases_consume_worker_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(mini_placement, "serving", lambda: True)
    cfg = {"mini_workers": {"worker-a": worker(), "worker-b": worker()}}
    monkeypatch.setattr(admission, "lanes", lambda **kw: [
        {"kind": "2b", "placement": "worker-a"}, {"kind": "2b"}])
    assert mini_placement.slots(cfg) == ["worker-b"]
    assert mini_placement.status(cfg) == {"worker-a": "on: cap 1", "worker-b": "on: cap 1"}


def test_no_slots_while_the_relay_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement, mini_relay

    monkeypatch.setattr(runner, "serving", lambda path: True)
    monkeypatch.setattr(mini_relay, "serving", lambda: False)
    monkeypatch.setattr(admission, "lanes", lambda **kw: [])
    cfg = {"mini_workers": {"worker-a": worker()}}
    assert mini_placement.slots(cfg) == []
    assert mini_placement.status(cfg)["worker-a"] == "held: bridge or mini relay not serving"


def test_phone_first_mini_refuses_while_its_phone_lock_is_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(mini_placement, "serving", lambda: True)
    answers = {"busy": '{"held": true, "devices": []}', "idle": '{"held": false, "devices": []}',
               "unknown": "not json"}
    for state, text in answers.items():
        monkeypatch.setattr(admission, "_sh", lambda argv, text=text, **kw: subprocess.CompletedProcess(
            argv, 0, "log line\n" + text + "\n", ""))
        assert mini_placement._phone_idle(worker(phone_first=True)) is (state == "idle")
    monkeypatch.setattr(mini_placement, "_phone_idle", lambda spec: False)
    with pytest.raises(admission.Refused, match="phone builds first"):
        mini_placement.launch(["f-one"], role="model-implementation", data=DataLayout(Path("/nonexistent")),
                              host="worker-a", config=worker(phone_first=True))


@pytest.fixture
def scheduler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[DataLayout, dict[str, Any]]:
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    path = tmp_path / "list.txt"
    path.write_text("m1 r3\nm2 r3\nm3 r3\nm4 r3\n")
    monkeypatch.setattr(drive, "next_steps", lambda ids, **kw: [
        {"id": i, "step": "2b"} for i in ids])
    monkeypatch.setattr(drive, "families_of", lambda ids, **kw: {i: "f-" + i for i in ids})
    monkeypatch.setattr(tick_full, "_busy", lambda: (Counter(), set()))
    monkeypatch.setattr(tick_full, "_caps", lambda cfg: {"2a": 4, "2b": 3})
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 1)
    monkeypatch.setattr(runner, "serving", lambda path: True)
    from menagerie.pipeline import mini_relay

    monkeypatch.setattr(mini_relay, "serving", lambda: True)
    monkeypatch.setattr(admission, "lanes", lambda **kw: [])
    return DataLayout(tmp_path / "data"), {**tick.DEFAULTS, "worklist": str(path), "2b_on_hub": True,
                                          "mini_workers": {"worker-a": worker(), "worker-b": worker()}}


def test_minis_first_hub_fallback_and_total_cap(scheduler: Any) -> None:
    data, cfg = scheduler
    result = tick_full.top_up(data, cfg, dry_run=True)
    assert result["2b"] == ["f-m1", "f-m2", "f-m3"]
    assert result["2b_placements"] == ["worker-a", "worker-b", None]
    assert result["2b_minis"] == {"worker-a": "on: cap 1", "worker-b": "on: cap 1"}


def test_2b_takes_only_worker_slots_by_default(scheduler: Any) -> None:
    data, cfg = scheduler
    result = tick_full.top_up(data, {**cfg, "2b_on_hub": False}, dry_run=True)
    assert result["2b"] == ["f-m1", "f-m2"] and result["hub_slots"] == 1
    assert result["2b_placements"] == ["worker-a", "worker-b"]
    assert result["2b_where"] == ("workers only (2b_on_hub false: 2b runs model code; hub slots are kept "
                                  "for 2a); this pass: 1 on worker-a, 1 on worker-b")


def test_refused_worker_placement_waits_instead_of_using_the_hub(scheduler: Any,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import implement, mini_placement

    data, cfg = scheduler
    seen = []

    def start(families, **kw):
        seen.append((families[0], kw.get("host")))
        raise admission.Refused("no_admission", "runtime unavailable")

    monkeypatch.setattr(mini_placement, "start", start)
    monkeypatch.setattr(implement, "launch", lambda *a, **kw: pytest.fail("used the hub"))
    monkeypatch.setattr(mini_placement, "slots", lambda cfg: ["worker-a"])
    monkeypatch.setattr(tick_full, "_launch_2a", lambda chunks, data: [])
    result = tick_full.top_up(data, {**cfg, "2b_on_hub": False}, dry_run=False)
    assert seen == [("f-m1", "worker-a")] and result["runs"] == ["held: runtime unavailable"]


def test_mini_slots_work_when_hub_is_full(scheduler: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    data, cfg = scheduler
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 0)
    assert tick_full.top_up(data, cfg, dry_run=True)["2b"] == ["f-m1", "f-m2"]


def test_guard_applies_to_every_placement(scheduler: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    data, cfg = scheduler
    monkeypatch.setattr(runner, "serving", lambda path: False)
    result = tick_full.top_up(data, cfg, dry_run=True)
    assert result["2b"] == []
    assert result["2b_placements"] == []
    assert result["waiting_2b"] == 4
    assert set(result["2b_minis"].values()) == {"held: bridge or mini relay not serving"}


def test_minis_do_not_create_hub_research_slots(scheduler: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    data, cfg = scheduler
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 0)
    monkeypatch.setattr(drive, "next_steps", lambda ids, **kw: [
        {"id": i, "step": "2a" if i == "m1" else "2b"} for i in ids])
    assert tick_full.top_up(data, cfg, dry_run=True)["2a"] == []


def test_launch_receives_host_and_refusal_falls_back(scheduler: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import implement, mini_placement

    data, cfg = scheduler
    seen = []

    def start(families, **kw):
        seen.append((families[0], kw.get("host")))
        raise admission.Refused("no_admission", "runtime unavailable")

    monkeypatch.setattr(mini_placement, "start", start)
    monkeypatch.setattr(implement, "launch", lambda family, **kw: seen.append((family, None)) or "hub-run")
    monkeypatch.setattr(mini_placement, "slots", lambda cfg: ["worker-a"])
    monkeypatch.setattr(tick_full, "_launch_2a", lambda chunks, data: [])
    result = tick_full.top_up(data, cfg, dry_run=False)
    assert seen == [("f-m1", "worker-a"), ("f-m1", None)]
    assert result["runs"] == ["hub-run", "held: hub fallback slots used"]


def test_mini_launch_rechecks_guard_before_brief(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(runner, "serving", lambda path: False)
    with pytest.raises(admission.Refused, match="relay not serving"):
        mini_placement.launch(["f-one"], role="model-implementation",
                              data=DataLayout(tmp_path), host="worker-a", config=worker())


def test_submit_pins_source_and_registers_durable_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import subprocess

    from menagerie.pipeline import mini_placement

    monkeypatch.setenv("FLEET_RUN_ID", "parent-1")
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path))
    seen = []

    def sh(argv, **kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, "abc123\n" if argv[0] == "git"
                                           else "run_id=child-1\n", "")

    monkeypatch.setattr(admission, "_sh", sh)
    monkeypatch.setattr(mini_placement, "_admitted", lambda label: None)
    assert mini_placement._submit("worker-a", worker(), ["f-one"], "2b-mini-1",
                                  tmp_path / "2b/f-one/BRIEF.md", {"model": "test-model"}) == "child-1"
    argv = seen[-1]
    assert argv[argv.index("--parent") + 1] == "parent-1"
    assert argv[argv.index("--ref") + 1] == "abc123"
    assert argv[argv.index("--memory-gib") + 1] == "1"
    assert argv[argv.index("--pull") + 1] == "outputs/lane/**"
    assert "--no-fallback" in argv
    # The lane's checkout leaves out the bulk data it never reads (ledger, store index, roster).
    excluded = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--sparse-exclude"]
    assert excluded == list(mini_placement.LANE_SPARSE_EXCLUDE)
    assert "menagerie/data/ledger/" in excluded and "--full" not in argv
    # The worker runs the stdlib-only lane file with its system Python, by absolute path so a
    # Homebrew python3 first on the job's PATH never becomes the Mac's privacy subject, and never
    # names the hub.
    assert argv[-4:-1] == ["/usr/bin/python3", "menagerie/pipeline/mini_lane.py", "run"]
    manifest = json.loads(argv[-1])
    assert manifest["brief"] == "work/f-one/BRIEF.md"
    assert manifest["result"] == "work/f-one/BRIEF.result.md"
    assert str(tmp_path) not in argv[-1]
    mini_placement._submit("worker-a", worker(), ["f-one", "f-two"], "2b-mini-2",
                           tmp_path / "briefs/2b-mini-2/BRIEF.md", {"model": "test-model"})
    assert json.loads(seen[-1][-1])["result"] == "work/_batch/BRIEF.result.md"


def test_runtime_identity_mismatch_refuses_before_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(admission, "_sh", lambda argv, **kw: subprocess.CompletedProcess(
        argv, 0, '{"harness":"claude","account":"another-pool"}', ""))
    monkeypatch.setattr(admission, "_admit", lambda *a: pytest.fail("quota checked before identity"))
    with pytest.raises(admission.Refused, match="identity"):
        mini_placement._runtime("model-implementation", worker(), False, "worker-a")


def test_pool_worker_probes_the_admitted_account_after_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from menagerie.pipeline import mini_placement

    order = []
    monkeypatch.setattr(admission, "_resolve", lambda role, account, **kw: {"harness": "claude", "model": "m"})
    monkeypatch.setattr(admission, "_admit", lambda *a: order.append("admit") or ([], "models"))
    monkeypatch.setattr(admission, "_sh", lambda argv, **kw: order.append(argv[-1]) or subprocess.CompletedProcess(
        argv, 0, '{"harness":"claude","account":"models"}', ""))
    worker_spec = worker(account="pool", identity=["runtime-identity", "{account}"])
    plan, role, account = mini_placement._runtime("model-implementation", worker_spec, False, "worker-a")
    assert plan["model"] == "m"
    assert (role, account) == ("model-implementation", "models")
    assert order[0] == "admit"
    assert order[1].endswith("runtime-identity models")


def test_pool_worker_refuses_when_runtime_cannot_serve_the_admitted_account(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(admission, "_resolve", lambda role, account, **kw: {"harness": "claude", "model": "m"})
    monkeypatch.setattr(admission, "_admit", lambda *a: ([], "personal"))
    monkeypatch.setattr(admission, "_sh", lambda argv, **kw: subprocess.CompletedProcess(
        argv, 0, '{"harness":"claude","account":"models"}', ""))
    with pytest.raises(admission.Refused, match="identity"):
        mini_placement._runtime("model-implementation", worker(account="pool"), False, "worker-a")


def test_named_worker_still_refuses_another_admitted_account(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(admission, "_resolve", lambda role, account, **kw: {"harness": "claude", "model": "m"})
    monkeypatch.setattr(admission, "_admit", lambda *a: ([], "models"))
    monkeypatch.setattr(admission, "_sh", lambda argv, **kw: subprocess.CompletedProcess(
        argv, 0, '{"harness":"claude","account":"test-account"}', ""))
    with pytest.raises(admission.Refused, match="differs"):
        mini_placement._runtime("model-implementation", worker(), False, "worker-a")


def test_pool_launch_threads_the_admitted_account_to_lease_and_submit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import contextlib
    import json

    from menagerie.pipeline import implement, mini_placement

    submitted = {}
    monkeypatch.setattr(mini_placement, "serving", lambda: True)
    monkeypatch.setattr(mini_placement, "_families", lambda *a: None)
    monkeypatch.setattr(mini_placement, "read_json", lambda *a: object())
    monkeypatch.setattr(implement, "fabrications", lambda *a: [])
    monkeypatch.setattr(mini_placement, "_runtime", lambda *a: ({"model": "m"}, "model-implementation", "models"))
    monkeypatch.setattr(admission, "_lease_lock", lambda: contextlib.nullcontext(tmp_path))
    monkeypatch.setattr(admission, "_held", lambda *a: None)
    monkeypatch.setattr(admission, "gauges", lambda account: {"claude": account})
    monkeypatch.setattr(drive, "event", lambda *a, **kw: None)
    monkeypatch.setattr(implement, "batch_brief", lambda *a, **kw: tmp_path / "BRIEF.md")
    monkeypatch.setattr(mini_placement, "_submit",
                        lambda host, spec, *a: submitted.update(spec) or "run-1")
    monkeypatch.setattr(mini_placement, "_admitted", lambda label: None)
    mini_placement.launch(["f-one"], role="model-implementation", data=DataLayout(tmp_path / "data"),
                          host="worker-a", config=worker(account="pool"))
    lease, = tmp_path.glob("*.json")
    assert json.loads(lease.read_text())["account"] == "models"  # the relay's deliver hook reads this
    assert submitted["account"] == "models"  # the manifest's runtime and in-job identity use this


def test_pool_worker_refuses_when_routing_moves_the_role_off_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(admission, "_resolve", lambda role, account, **kw: {"harness": "codex", "model": "x"})
    monkeypatch.setattr(admission, "_admit", lambda *a: pytest.fail("admitted a non-Claude seat"))
    with pytest.raises(admission.Refused, match="no compatible"):
        mini_placement._runtime("model-implementation", worker(account="pool"), False, "worker-a")


def test_pool_worker_asks_for_a_claude_seat_for_every_role(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from menagerie.pipeline import mini_placement

    asked = []
    monkeypatch.setattr(admission, "_resolve", lambda role, account, **kw: asked.append(
        (role, kw.get("harness"))) or {"harness": "claude", "model": "m"})
    monkeypatch.setattr(admission, "_admit", lambda *a: ([], "models"))
    monkeypatch.setattr(admission, "_sh", lambda argv, **kw: subprocess.CompletedProcess(
        argv, 0, '{"harness":"claude","account":"models"}', ""))
    mini_placement._runtime("model-default", worker(account="pool"), False, "worker-a")
    assert asked == [("model-default", "claude")]


def test_runtime_probe_uses_configured_existing_ssh_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from menagerie.pipeline import mini_placement

    seen = []
    monkeypatch.setattr(admission, "_sh", lambda argv, **kw: (
        seen.append(argv) or subprocess.CompletedProcess(argv, 0, '{}', '')))
    with pytest.raises(admission.Refused):
        mini_placement._runtime("model-implementation", worker(), False, "worker-a")
    assert "worker-a-ssh" in seen[0]
    assert "worker-a" not in seen[0]


def test_uncertain_remote_start_does_not_abort_other_tick_placements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(mini_placement, "start", lambda *a, **kw: (_ for _ in ()).throw(
        subprocess.TimeoutExpired("start", 120)))
    run, used = tick_full._placed_2b("f-one", DataLayout(tmp_path), "worker-a", {}, 1)
    assert run.startswith("held:")
    assert used is False


def test_proven_remote_refusal_allows_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(admission, "_sh", lambda argv, **kw: subprocess.CompletedProcess(
        argv, 0, '{"verdict":"refused","state":"TERMINAL"}', ""))
    with pytest.raises(admission.Refused, match="before payload"):
        mini_placement._admitted("job-1")


def test_unknown_start_retains_provisional_family_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import contextlib
    import json
    import subprocess

    from menagerie.pipeline import implement, mini_placement

    monkeypatch.setattr(mini_placement, "serving", lambda: True)
    monkeypatch.setattr(mini_placement, "_families", lambda *a: None)
    monkeypatch.setattr(mini_placement, "read_json", lambda *a: object())
    monkeypatch.setattr(implement, "fabrications", lambda *a: [])
    monkeypatch.setattr(mini_placement, "_runtime",
                        lambda *a: ({"model": "test-model"}, "model-implementation", "admitted-account"))
    monkeypatch.setattr(admission, "_lease_lock", lambda: contextlib.nullcontext(tmp_path))
    monkeypatch.setattr(admission, "_held", lambda *a: None)
    monkeypatch.setattr(implement, "batch_brief", lambda *a, **kw: tmp_path / "BRIEF.md")

    def unknown(*args: Any) -> str:
        raise subprocess.TimeoutExpired("start", 120)

    monkeypatch.setattr(mini_placement, "_submit", unknown)
    with pytest.raises(subprocess.TimeoutExpired):
        mini_placement.launch(["f-one"], role="model-implementation", data=DataLayout(tmp_path / "data"),
                              host="worker-a", config=worker())
    lease, = tmp_path.glob("*.json")
    kept = json.loads(lease.read_text())
    assert kept["keys"] == ["f-one"]
    assert kept["run"].startswith("pending:")
    assert kept["account"] == "admitted-account"  # a pool worker's lease names the admitted account


# --- Linux workers and the trace host --------------------------------------------------------


def linux(**changes: Any) -> dict[str, Any]:
    return worker(**{"os": "linux", "cap": 2, "cpus": 1, "budget_cpus": 3, "budget_gib": 20,
                     "peak_gib": 2, "trace_host": True, **changes})


@pytest.mark.parametrize("changes", [
    {"os": "windows"}, {"cap": 13}, {"cap": 3, "budget_cpus": 2}, {"budget_cpus": None},
    {"budget_cpus": 1.5}, {"cpus": 0}, {"cpus": True}, {"cpus": 0.3}, {"cpus": 1.5},
    {"cap": 3, "budget_gib": 6},
    {"trace_host": "yes"},
])
def test_invalid_linux_workers_get_no_slot(changes: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(admission, "lanes", lambda **kw: [])
    monkeypatch.setattr(mini_placement, "serving", lambda: True)
    cfg = {"mini_workers": {"linux-a": linux(**{"trace_host": False, **changes})}}
    assert mini_placement.slots(cfg) == []
    assert mini_placement.status(cfg)["linux-a"].startswith("off: ")


def test_mini_caps_stay_at_two_lanes() -> None:
    from menagerie.pipeline import mini_placement

    assert mini_placement.validated(worker(cap=2, budget_gib=8))["cap"] == 2
    with pytest.raises(ValueError, match="cap must be 0 to 2"):
        mini_placement.validated(worker(cap=3, budget_gib=12))
    with pytest.raises(ValueError, match="cap must be 0 to 2"):
        mini_placement.validated(worker(os="macos", cap=3, budget_gib=12))
    with pytest.raises(ValueError, match="declares one CPU"):
        mini_placement.validated(worker(cap=1, cpus=0.25))
    with pytest.raises(ValueError, match="one CPU"):
        mini_placement.validated(worker(cpus=2))


def test_enabled_linux_workers_must_say_whether_they_are_the_trace_host() -> None:
    from menagerie.pipeline import mini_placement

    config = {k: v for k, v in linux().items() if k != "trace_host"}
    with pytest.raises(ValueError, match="trace_host"):
        mini_placement.validated(config)
    assert mini_placement.validated({**config, "cap": 0})["cap"] == 0


def test_linux_worker_off_the_trace_host_takes_its_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(admission, "lanes", lambda **kw: [{"kind": "2b", "placement": "linux-a"}])
    monkeypatch.setattr(mini_placement, "serving", lambda: True)
    cfg = {"mini_workers": {"linux-a": linux(trace_host=False)}}
    assert mini_placement.slots(cfg) == ["linux-a"]
    assert mini_placement.status(cfg) == {"linux-a": "on: cap 2"}


def test_trace_host_is_declared_or_matched_by_name_or_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.delenv("MENAGERIE_RUNNER_HOST", raising=False)
    assert mini_placement._trace_host("linux-a", linux(), {}) is True
    assert mini_placement._trace_host("linux-a", linux(trace_host=False), {"disk": {"host": "linux-a"}}) is False
    assert mini_placement._trace_host("mini-a", worker(), {}) is False
    # The disk guard names the ssh alias, the placement the registry name: either matches.
    assert mini_placement._trace_host("mini-a", worker(), {"disk": {"host": "worker-a-ssh"}}) is True
    monkeypatch.setenv("MENAGERIE_RUNNER_HOST", "mini-a")
    assert mini_placement._trace_host("mini-a", worker(), {}) is True
    assert mini_placement._trace_host("mini-a", worker(trace_host=False), {}) is False


@pytest.mark.parametrize("state", ["paused", "critical", "unknown"])
def test_trace_host_holds_while_the_disk_guard_holds(state: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(admission, "lanes", lambda **kw: [])
    monkeypatch.setattr(mini_placement, "serving", lambda: True)
    cfg = {"mini_workers": {"linux-a": linux(trace_host=True)}, "disk_state": state}
    assert mini_placement.slots(cfg) == []
    assert mini_placement.status(cfg)["linux-a"] == f"on: cap 2; held: disk guard {state} on the trace host"
    with pytest.raises(admission.Refused, match=f"disk guard {state}"):
        mini_placement.launch(["f-one"], role="model-implementation", data=DataLayout(Path("/nonexistent")),
                              host="linux-a", config=linux(trace_host=True), cfg=cfg)


def test_trace_host_without_a_saved_disk_reading_holds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path))
    from datetime import datetime, timedelta, timezone

    assert mini_placement.host_cap("linux-a", linux(trace_host=True), {})[0] == 0
    tick.save_state({"disk_guard": {"state": "ok"}})  # no reading time: unknown
    assert mini_placement.host_cap("linux-a", linux(trace_host=True), {})[0] == 0
    fresh = datetime.now(timezone.utc) - timedelta(minutes=5)
    tick.save_state({"disk_guard": {"state": "ok", "read_at": fresh.isoformat(timespec="seconds")}})
    assert mini_placement.host_cap("linux-a", linux(trace_host=True), {})[0] == 2
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    tick.save_state({"disk_guard": {"state": "ok", "read_at": stale.isoformat(timespec="seconds")}})
    assert mini_placement.host_cap("linux-a", linux(trace_host=True), {}) == (
        0, "held: disk guard unknown on the trace host")


def test_trace_host_lanes_fit_the_share_the_runners_leave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement, runner_job

    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path))
    monkeypatch.delenv("MENAGERIE_LANES_2B", raising=False)
    # At the 2b cap of 14, A wants three workers (9 CPUs, 56 GiB) beside B (6, 36): 3 CPUs, 20 GiB left.
    assert runner_job.lane_room(14) == (3, 20)
    assert runner_job.lane_room(16) == (0, 0)
    assert runner_job.lane_room(1) == (9, 56)
    assert runner_job.lane_room(1, big_runner=True) == (6, 36)  # night: A 3/20, B 3/16, G 6/40
    ok = {"disk_state": "ok", "2b_on_hub": True}  # hub 2b: A is reserved for the whole 2b cap
    spec = linux(trace_host=True, cap=3, budget_cpus=3)
    assert mini_placement.host_cap("linux-a", spec, ok) == (3, "")
    allowed, why = mini_placement.host_cap("linux-a", linux(trace_host=True, cap=3, peak_gib=6, budget_cpus=3), ok)
    assert allowed == 2 and why == "runner share leaves room for 2 (3 CPUs, 20 GiB)"
    allowed, why = mini_placement.host_cap("linux-a", linux(trace_host=True, cpus=2, budget_cpus=4), ok)
    assert allowed == 1
    monkeypatch.setenv("MENAGERIE_LANES_2B", "16")
    assert mini_placement.host_cap("linux-a", spec, ok)[0] == 0
    # The tick's lower limit does not bound manual hub launches, so it never frees runner room.
    assert mini_placement.host_cap("linux-a", spec, {**ok, "lane_limits": {"2b": 10}})[0] == 0
    monkeypatch.setenv("MENAGERIE_LANES_2B", "10")
    assert mini_placement.host_cap("linux-a", spec, ok)[0] == 3


def test_trace_host_lanes_use_the_room_a_cannot_reach_when_2b_stays_off_the_hub(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path))
    monkeypatch.delenv("MENAGERIE_LANES_2B", raising=False)
    monkeypatch.setattr(mini_placement, "_spec", lambda host, config: dict(config))
    minis = {"mini-a": worker(), "mini-b": worker()}
    trace = linux(trace_host=True, cap=8, budget_cpus=8, peak_gib=1)
    cfg = {"disk_state": "ok", "mini_workers": {**minis, "linux-a": trace}}
    # Two mini lanes plus six here: A at two workers (6 CPUs, 38 GiB) beside B (6, 36) leaves 6
    # CPUs, so seven do not fit; eleven would need A at three workers. B's seat is never room.
    assert mini_placement.host_cap("linux-a", trace, cfg) == (6, "runner share leaves room for 6 (6 CPUs, 38 GiB)")
    assert mini_placement.host_cap("linux-a", {**trace, "cap": 6, "budget_cpus": 6}, cfg) == (6, "")
    # Lanes declaring a quarter CPU (they wait on the model API; their tries run on A): twelve here
    # make fourteen reachable, A at three workers leaves 3 CPUs and 20 GiB, and all twelve fit.
    quarter = linux(trace_host=True, cap=12, cpus=0.25, budget_cpus=6, peak_gib=1)
    assert mini_placement.validated(quarter)["cpus"] == 0.25
    assert mini_placement.host_cap("linux-a", quarter, {**cfg, "mini_workers": {**minis, "linux-a": quarter}}) == (12, "")
    # With 2b on the hub, A keeps its reservation for the whole 2b cap, as before.
    assert mini_placement.host_cap("linux-a", trace, {**cfg, "2b_on_hub": True})[0] == 3


def test_trace_host_takes_no_lanes_while_the_big_runner_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.setattr(mini_placement, "_big_night", lambda host, spec: True)
    allowed, why = mini_placement.host_cap("linux-a", linux(), {"disk_state": "ok", "big_runner": True})
    assert allowed == 0 and "big runner" in why
    assert mini_placement.host_cap("linux-a", linux(trace_host=False), {"big_runner": True}) == (2, "")
    monkeypatch.setattr(mini_placement, "_big_night", lambda host, spec: False)  # by day: lanes as usual
    allowed, why = mini_placement.host_cap("linux-a", linux(), {"disk_state": "ok", "big_runner": True})
    assert "big runner" not in why


LIVE_G = {"job": "g1", "submitted_at": "2026-10-27T06:30:00+00:00"}


SPEC = {"cpus": 0.25, "peak_gib": 1}
LANES = [{"kind": "2b", "placement": "zm"}, {"kind": "2b", "placement": "zm"},
         {"kind": "2b", "placement": "mini"}, {"kind": "2a"}]


@pytest.mark.parametrize("g,window,work,fits,held", [
    (LIVE_G, False, 0, False, True),   # G runs and run-on would not admit a lane beside it: held
    (LIVE_G, False, 0, True, False),   # G runs and room remains: lanes resume beside it
    ({**LIVE_G, "cancelled": "g1"}, True, 5, False, False),  # cancelled to give way to A
    ({**LIVE_G, "refused_at": "2026-10-27T06:30:00+00:00"}, True, 5, True, True),  # refused: lanes drain for it
    ({}, True, 5, True, True),         # the night with rows for G, and room once the lanes finish
    ({}, True, 5, False, False),       # no room for G even then (another project): lanes run
    ({}, True, 0, True, False),        # a night with no big rows: lanes as usual
    ({}, False, 5, True, False),       # by day
])
def test_g_comes_before_the_trace_host_s_lanes_at_night(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, g: dict[str, Any], window: bool, work: int,
        fits: bool, held: bool) -> None:
    from menagerie.pipeline import mini_placement, runner_job, tick

    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path))

    state = {"big_work": work, "runners": {"G": g, "B": {"job": "b1"}}}
    monkeypatch.setattr(tick, "load_state", lambda: state)
    monkeypatch.setattr(runner_job, "big_hold_open", lambda now=None: window)
    monkeypatch.setattr(mini_placement, "_host_lanes", lambda host: sum(
        1 for lane in LANES if lane.get("kind") == "2b" and lane.get("placement") == host))
    monkeypatch.setattr(admission, "lanes", lambda **kw: pytest.fail("took the lease lock"))
    asked = []
    monkeypatch.setattr(mini_placement, "_g_fits", lambda b, freed, **kw: asked.append((b["job"], freed)) or fits)
    monkeypatch.setattr(mini_placement, "_lane_fits", lambda spec: asked.append(spec) or fits)
    assert mini_placement._big_night("zm", SPEC) is held
    assert asked in ([], [("b1", (0.5, 2))], [SPEC])


def test_the_hold_lasts_while_lanes_drain_and_ends_with_none_left(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement, runner_job, tick

    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path))
    monkeypatch.setattr(tick, "load_state", lambda: {"big_work": 5, "runners": {"B": {"job": "b1"}}})
    monkeypatch.setattr(runner_job, "big_hold_open", lambda now=None: True)
    answers, lanes = [True, False, False, False], [3, 2, 0, 0]
    monkeypatch.setattr(mini_placement, "_g_fits", lambda b, freed, **kw: answers.pop(0))
    monkeypatch.setattr(mini_placement, "_host_lanes", lambda host: lanes.pop(0))
    assert mini_placement._big_night("zm", SPEC) is True    # G fits once the lanes go: hold
    assert mini_placement._big_night("zm", SPEC) is True    # refused with lanes left: still draining
    assert mini_placement._big_night("zm", SPEC) is False   # none left and still refused: lanes back
    assert mini_placement._big_night("zm", SPEC) is False   # and no hold without a fit
    monkeypatch.setattr(runner_job, "big_hold_open", lambda now=None: False)
    (tmp_path / "tick").mkdir(exist_ok=True)
    (tmp_path / "tick" / mini_placement.HOLD_FILE).write_text("x")
    assert mini_placement._big_night("zm", SPEC) is False   # by day the hold file goes
    assert not (tmp_path / "tick" / mini_placement.HOLD_FILE).exists()


def test_g_fits_probes_g_with_b_at_night_size_in_place_of_b_less_the_lanes_once_per_pass(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement, runner_job

    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path))  # the default runners
    seen = []
    def probe(self: Any, runner: Any, replacing: str | None) -> tuple[int, str]:
        seen.append((runner.cpus, runner.memory_gib, replacing))
        return runner_job.PROBE_REFUSED, "NO_CAPACITY: memory declarations exceed usable RAM"
    monkeypatch.setattr(runner_job.RunOn, "probe", probe)
    monkeypatch.setattr(mini_placement, "_g_fits_memo", {})
    day_b, night_b = {"job": "b1", "size": 1}, {"job": "b1", "size": 0}
    assert mini_placement._g_fits(day_b) is False and mini_placement._g_fits(day_b) is False
    assert mini_placement._g_fits(day_b, (2.25, 9)) is False
    assert mini_placement._g_fits(night_b, (2.25, 9)) is False  # B already small: G alone
    assert mini_placement._g_fits(night_b, (2.25, 9), replaces_b=True) is False  # G in B's place
    assert mini_placement._lane_fits(SPEC) is False
    assert seen == [(9, 56, "b1"), (6.75, 47, "b1"), (3.75, 31, None), (3.75, 31, "b1"),
                    (0.25, 1, None)]


def test_a_failing_trace_host_check_holds_the_host_not_the_tick(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path))
    (tmp_path / "tick").mkdir()
    (tmp_path / "tick" / "runners.json").write_text('[{"name": "A", "bogus": 1}]')
    monkeypatch.setattr(admission, "lanes", lambda **kw: [])
    monkeypatch.setattr(mini_placement, "serving", lambda: True)
    cfg = {"mini_workers": {"linux-a": linux()}, "disk_state": "ok"}
    assert mini_placement.slots(cfg) == []
    assert "held: trace-host check failed (TypeError" in mini_placement.status(cfg)["linux-a"]
    bad_guard = {"mini_workers": {"mini-a": worker()}, "disk": {"pause_gib": 1.0}}
    assert mini_placement.slots(bad_guard) == []
    assert "GuardConfigError" in mini_placement.status(bad_guard)["mini-a"]


def test_reservation_rechecks_the_trace_host_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import contextlib
    import json

    from menagerie.pipeline import mini_placement

    (tmp_path / "other.json").write_text(json.dumps({"placement": "linux-a", "keys": ["f-x"]}))
    monkeypatch.setattr(admission, "_lease_lock", lambda: contextlib.nullcontext(tmp_path))
    monkeypatch.setattr(admission, "_held", lambda *a: None)
    monkeypatch.setattr(admission, "_live", lambda path, lease: "RUNNING")
    monkeypatch.setattr(mini_placement, "serving", lambda: True)
    spec = linux(trace_host=True, cap=2, peak_gib=6)  # the share fits two at 6 GiB; one is live
    path, _ = mini_placement._reserve(["f-one"], "linux-a", spec, "2b-mini-a", {"model": "m"},
                                      "model-implementation", {"disk_state": "ok"})
    assert path.is_file()
    with pytest.raises(admission.Refused, match="slot taken"):
        mini_placement._reserve(["f-two"], "linux-a", spec, "2b-mini-b", {"model": "m"},
                                "model-implementation", {"disk_state": "ok"})


def test_linux_submit_declares_os_and_cpus_and_is_not_niced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import subprocess

    from menagerie.pipeline import mini_placement

    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path))
    monkeypatch.setenv("MENAGERIE_HOST_ROLES", "/nonexistent/host-roles.json")  # unknown roles
    seen = []
    monkeypatch.setattr(admission, "_sh", lambda argv, **kw: seen.append(argv) or subprocess.CompletedProcess(
        argv, 0, "abc123\n" if argv[0] == "git" else "run_id=child-1\n", ""))
    monkeypatch.setattr(mini_placement, "_admitted", lambda label: None)
    mini_placement._submit("linux-a", linux(cpus=2, budget_cpus=4), ["f-one"], "2b-mini-1",
                           tmp_path / "2b/f-one/BRIEF.md", {"model": "test-model"})
    argv = seen[-1]
    assert argv[argv.index("--os") + 1] == "linux"
    assert argv[argv.index("--cpus") + 1] == "2"
    assert argv[argv.index("--memory-gib") + 1] == "2"
    assert json.loads(argv[-1])["nice"] is None
    mini_placement._submit("linux-a", linux(cpus=0.25, budget_cpus=4), ["f-one"], "2b-mini-3",
                           tmp_path / "2b/f-one/BRIEF.md", {"model": "test-model"})
    assert seen[-1][seen[-1].index("--cpus") + 1] == "0.25"
    mini_placement._submit("mini-a", worker(), ["f-one"], "2b-mini-2",
                           tmp_path / "2b/f-one/BRIEF.md", {"model": "test-model"})
    assert seen[-1][seen[-1].index("--os") + 1] == "macos"
    assert seen[-1][seen[-1].index("--cpus") + 1] == "1"
    assert json.loads(seen[-1][-1])["nice"] == mini_placement.PHONE_NICE  # unknown roles: phones first


def test_linux_worker_never_probes_a_phone_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    monkeypatch.setenv("MENAGERIE_HOST_ROLES", "/nonexistent/host-roles.json")
    monkeypatch.setattr(mini_placement, "serving", lambda: True)
    monkeypatch.setattr(mini_placement, "_phone_idle", lambda spec: pytest.fail("probed a phone lock"))
    spec = mini_placement._admissible("linux-a", linux(trace_host=False), {})
    assert spec["os"] == "linux"


def test_linux_worker_lanes_run_when_the_hub_is_full(scheduler: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    data, cfg = scheduler
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 0)
    monkeypatch.setattr(tick_full, "_caps", lambda cfg: {"2a": 4, "2b": 14})
    cfg = {**cfg, "mini_workers": {"linux-a": linux(trace_host=True)}, "disk_state": "ok"}
    result = tick_full.top_up(data, cfg, dry_run=True)
    assert result["2b"] == ["f-m1", "f-m2"]
    assert result["2b_placements"] == ["linux-a", "linux-a"]
    assert result["hub_slots"] == 0
    held = tick_full.top_up(data, {**cfg, "disk_state": "paused"}, dry_run=True)
    assert held["2b"] == [] and held["2b_minis"]["linux-a"].endswith("held: disk guard paused on the trace host")


def test_worker_launch_receives_the_tick_config(scheduler: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    data, cfg = scheduler
    seen = []
    monkeypatch.setattr(mini_placement, "start", lambda families, **kw: seen.append(kw) or "started")
    run, used = tick_full._placed_2b("f-one", data, "worker-a", {**cfg, "disk_state": "ok"}, 0)
    assert (run, used) == ("started", False)  # a worker lane uses no hub slot
    assert seen[0]["cfg"]["disk_state"] == "ok" and seen[0]["host"] == "worker-a"


class _Started:
    def __init__(self, family: str) -> None:
        self.host, self.family = "worker-a", family


def test_worker_lane_admissions_are_read_after_every_lane_started(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    order = []
    monkeypatch.setattr(mini_placement, "start", lambda families, **kw: order.append(
        "start " + families[0]) or _Started(families[0]))
    monkeypatch.setattr(mini_placement, "finish", lambda started: order.append(
        "finish " + started.family) or "run-" + started.family)
    runs = tick_full._launch_2b(["f-a", "f-b", "f-c"], DataLayout(tmp_path),
                                placements=["worker-a"] * 3, cfg={}, hub_room=0)
    assert runs == ["run-f-a", "run-f-b", "run-f-c"]
    # The run-on admission waits overlap: no lane's wait holds up the next lane's start.
    assert order == ["start f-a", "start f-b", "start f-c", "finish f-a", "finish f-b", "finish f-c"]


def test_a_refusal_read_after_the_starts_still_falls_back_to_the_hub(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import implement, mini_placement

    def finish(started: Any) -> str:
        if started.family == "f-a":
            raise admission.Refused("no_admission", "mini worker refused before payload start")
        return "run-" + started.family

    monkeypatch.setattr(mini_placement, "start", lambda families, **kw: _Started(families[0]))
    monkeypatch.setattr(mini_placement, "finish", finish)
    monkeypatch.setattr(implement, "launch", lambda family, **kw: "hub-" + family)
    runs = tick_full._launch_2b(["f-a", "f-b"], DataLayout(tmp_path),
                                placements=["worker-a", "worker-a"], cfg={}, hub_room=1)
    assert runs == ["hub-f-a", "run-f-b"]
    held = tick_full._launch_2b(["f-a"], DataLayout(tmp_path), placements=["worker-a"], cfg={},
                                hub_room=0)
    assert held == ["held: mini worker refused before payload start"]


def test_no_lane_starts_after_the_pass_launch_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from menagerie.pipeline import mini_placement, research

    monkeypatch.setattr(mini_placement, "start", lambda *a, **kw: pytest.fail("started a 2b lane"))
    monkeypatch.setattr(research, "launch", lambda *a, **kw: pytest.fail("started a 2a lane"))
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path))
    with tick_full._launch_window(time.monotonic() - 1):
        assert tick_full._launch_2b(["f-a"], DataLayout(tmp_path), placements=["worker-a"],
                                    cfg={}, hub_room=0) == [tick_full.LAUNCH_SPENT]
        assert tick_full._launch_2a([["m1"]], DataLayout(tmp_path)) == [tick_full.LAUNCH_SPENT]
    assert tick_full._LAUNCH_BY.get() is None  # the window closes with the block


def _started_lane(tmp_path: Path) -> Any:
    import json

    from menagerie.pipeline import mini_placement

    path = tmp_path / "2b-mini-1.json"
    lease = {"label": "2b-mini-1", "run": "run-1", "keys": ["f-one", "f-two"], "account": "models"}
    path.write_text(json.dumps({**lease, "run": "pending:2b-mini-1"}))
    return mini_placement.Started("worker-a", path, lease, ("f-one", "f-two"), "github")


def test_finish_writes_the_lease_and_launch_events_once_admitted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from menagerie.pipeline import mini_placement

    events = []
    monkeypatch.setattr(mini_placement, "_admitted", lambda label: None)
    monkeypatch.setattr(admission, "gauges", lambda account: {"claude": account})
    monkeypatch.setattr(drive, "event", lambda run, family, verb, **kw: events.append((run, family, verb)))
    assert mini_placement.finish(_started_lane(tmp_path)) == "run-1"
    lease = json.loads((tmp_path / "2b-mini-1.json").read_text())
    assert lease["run"] == "run-1" and "expires" in lease
    assert events == [("run-1", "", "launch"), ("run-1", "f-one", "batch:assigned"),
                      ("run-1", "f-one", "tier"), ("run-1", "f-two", "batch:assigned"),
                      ("run-1", "f-two", "tier")]


def test_finish_releases_the_lease_on_a_proven_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_placement

    def refused(label: str) -> None:
        raise admission.Refused("no_admission", "mini worker refused before payload start")

    monkeypatch.setattr(mini_placement, "_admitted", refused)
    monkeypatch.setattr(drive, "event", lambda *a, **kw: pytest.fail("wrote a launch event"))
    with pytest.raises(admission.Refused):
        mini_placement.finish(_started_lane(tmp_path))
    assert not (tmp_path / "2b-mini-1.json").exists()


@pytest.mark.parametrize("verdict", ["refused", "waiting", "admission_timeout"])
def test_admitted_raises_on_a_terminal_refusal(monkeypatch: pytest.MonkeyPatch, verdict: str) -> None:
    import json as _json
    import subprocess

    from menagerie.pipeline import mini_placement

    out = _json.dumps({"verdict": verdict, "state": "TERMINAL"})
    monkeypatch.setattr(mini_placement.admission, "_sh",
                        lambda cmd: subprocess.CompletedProcess(cmd, 0, out, ""))
    with pytest.raises(mini_placement.admission.Refused):
        mini_placement._admitted("2b-mini-test")


@pytest.mark.parametrize(("ran", "launch_failed", "refused"),
                         [(False, "rc=44", True), (True, None, False), (None, "rc=44", False)])
def test_a_launch_that_failed_before_any_run_is_not_admitted(
        monkeypatch: pytest.MonkeyPatch, ran: bool | None, launch_failed: str | None,
        refused: bool) -> None:
    import json as _json
    import subprocess

    from menagerie.pipeline import mini_placement

    # run-on state after a claim: failed, and ran false only when the worker shows no run.
    out = _json.dumps({"verdict": "failed", "state": "TERMINAL", "ran": ran,
                       "launch_failed": launch_failed, "lane_verdict": "refused"})
    monkeypatch.setattr(mini_placement.admission, "_sh",
                        lambda cmd: subprocess.CompletedProcess(cmd, 1, out, ""))
    if refused:
        with pytest.raises(mini_placement.admission.Refused, match="launch failed.*rc=44"):
            mini_placement._admitted("2b-mini-test")
    else:
        assert mini_placement._admitted("2b-mini-test") is None


def test_admitted_returns_once_started(monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json
    import subprocess

    from menagerie.pipeline import mini_placement

    out = _json.dumps({"verdict": "admitted", "state": "RUNNING"})
    monkeypatch.setattr(mini_placement.admission, "_sh",
                        lambda cmd: subprocess.CompletedProcess(cmd, 0, out, ""))
    assert mini_placement._admitted("2b-mini-test") is None


@pytest.mark.parametrize(("verdict", "rc"), [("admission_timeout", 75), ("refused", 6)])
def test_admitted_reads_a_refusal_from_run_on_state_s_nonzero_exit(
        monkeypatch: pytest.MonkeyPatch, verdict: str, rc: int) -> None:
    import json as _json
    import subprocess

    from menagerie.pipeline import mini_placement

    # run-on state exits 75 (admission_timeout) or 6 (refused) with its JSON record on stdout.
    out = _json.dumps({"verdict": verdict, "state": "TERMINAL", "outcome": "WAITING"})
    monkeypatch.setattr(mini_placement.admission, "_sh",
                        lambda cmd: subprocess.CompletedProcess(cmd, rc, out, ""))
    monkeypatch.setattr(mini_placement.time, "sleep", lambda _: pytest.fail("polled past a refusal"))
    with pytest.raises(mini_placement.admission.Refused, match="before payload"):
        mini_placement._admitted("2b-mini-test")


def test_admitted_keeps_polling_when_run_on_state_prints_no_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json
    import subprocess

    from menagerie.pipeline import mini_placement

    answers = iter([subprocess.CompletedProcess([], 2, "", "lookup failed"),
                    subprocess.CompletedProcess([], 0, _json.dumps({"verdict": "admitted",
                                                                     "state": "RUNNING"}), "")])
    monkeypatch.setattr(mini_placement.admission, "_sh", lambda cmd: next(answers))
    monkeypatch.setattr(mini_placement.time, "sleep", lambda _: None)
    assert mini_placement._admitted("2b-mini-test") is None


@pytest.mark.parametrize("b,night,held", [
    ({"job": "b1", "refused_at": "2026-10-29T03:00:00+00:00"}, False, True),  # down, refused: hold
    ({"job": "b2", "submitted_at": "2026-10-29T03:20:00+00:00",
      "refused_at": "2026-10-29T03:00:00+00:00"}, False, False),  # started again since: lanes back
    ({"job": "b1", "submitted_at": "2026-10-29T00:10:00+00:00"}, False, False),  # B runs
    ({"job": "b1", "refused_at": "2026-10-29T06:00:00+00:00"}, True, False),  # night: G's rule
    ({}, False, False),  # no record yet
])
def test_the_trace_host_holds_new_lanes_while_b_is_refused_by_day(
        monkeypatch: pytest.MonkeyPatch, b: dict[str, Any], night: bool, held: bool) -> None:
    from menagerie.pipeline import mini_placement, runner_job, tick

    monkeypatch.setattr(tick, "load_state", lambda: {"runners": {"B": b}})
    monkeypatch.setattr(runner_job, "night_open", lambda now=None: night)
    assert mini_placement._b_held_out() is held
    monkeypatch.setattr(mini_placement, "_big_night", lambda host, spec: False)
    allowed, why = mini_placement.host_cap("linux-a", linux(), {"disk_state": "ok"})
    assert (allowed == 0 and "runner B" in why) is held
    off = mini_placement.host_cap("linux-a", linux(), {"disk_state": "ok", "b_before_lanes": False})
    assert "runner B" not in off[1]
    assert mini_placement.host_cap("linux-a", linux(trace_host=False), {}) == (2, "")  # Mac lanes untouched
