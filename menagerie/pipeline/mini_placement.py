"""Optional remote workers for recipe agents; canonical pipeline verbs stay on the hub.

Worker names, the runtime command, its identity probe and an optional hub login hook are
deployment configuration: ``mini_workers.<host>`` in the tick config, whose ``runtime_profile``
may name a profile in the fleet host registry (``capabilities.runtime_profiles``) that supplies
``runtime``, ``identity`` and ``deliver``; its ``account`` is one pool account, or ``pool`` to run
on whichever account admission picks. Empty configuration is off, and a worker missing any
of them gets no slot and a status line (``status``). The existing
remote job runner owns resource admission, supervision and memory receipts; the hub relay
(``mini_relay``) stages inputs, answers verbs and ingests the pulled results.

A worker is a Mac mini (``os`` ``macos``, the default: up to ``MAC_CAP_MAX`` lanes, phone builds
first) or a Linux worker (``os`` ``linux``: up to ``LINUX_CAP_MAX`` lanes inside its declared
``budget_cpus`` and ``budget_gib``, each lane declaring ``cpus`` (whole, or quarters below one: a
lane waits on the model API while its tries run on runner A) and ``peak_gib`` to run-on, and saying whether it is
the trace host). On the trace runners' host (``trace_host``; a mini whose name or alias is the
disk guard's host) lanes start only while the disk guard reads ``ok``, never while the night big
runner comes first (it runs with no room left for a lane, or would fit once this host's lanes
finish), and only in the share the runners leave with runner A sized for the 2b lanes the
workers can reach (``runner_job.lane_room``, ``_reachable_2b``), so in declared terms they never
take room a runner needs. A lane already running when the guard
pauses finishes, like a running row.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, admission, drive, mini_relay, now, read_json, runner
from menagerie.pipeline.mini_lane import LANE_TIMEOUT_S, expand

SSH = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
MEMORY_MARGIN = 1.25  # Match the remote runner's physical-footprint allowance.
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
RUNTIME_KEYS = ("runtime", "identity", "prepare", "deliver")  # what a runtime profile may supply
PHONE_NICE = 10  # a lane on a phone-build mini runs below phone builds' priority
LEASE_MINUTES = 210  # the 180-minute hard timeout plus time to pull and ingest
POOL = "pool"  # an ``account`` value: serve whichever pool account admission picks
# The lane runs on the system Python by absolute path. run-on puts Homebrew's bin first on a Mac's
# PATH, and a Homebrew Python becomes macOS's responsible process for everything under it, so the
# agent's startup opens of ~/Desktop, ~/Documents and ~/Downloads raised a privacy dialog for each
# new Homebrew Python and stalled every lane on that Mac (26 Oct). The system Python keeps the job's
# bash responsible, which macOS has already answered; the lane file is stdlib-only for Python 3.9.
LANE_PYTHON = "/usr/bin/python3"
# A lane's checkout leaves out the bulk data it never reads or writes: the lane runs the stdlib-only
# lane file, its agent works in the hub-staged .mini-lane/ folder, and every record write goes
# through the hub relay. Adapters, model records and research stay as examples the agent may read.
LANE_SPARSE_EXCLUDE = ("menagerie/data/ledger/", "menagerie/data/store-index/", "menagerie/data/cite/",
                       "menagerie/data/catalog/", "menagerie/data/sweeps/",
                       "menagerie/data/master_catalog.jsonl", "menagerie/data/crawl_history.json",
                       "menagerie/data/crawl_roster.jsonl", "menagerie/data/routing_manifest.tsv",
                       "menagerie/crawler/records/", "site/site-data/")
PHONE_JOB = ["sh", "-c", 'exec "$HOME/.local/bin/phone-job" "$@"', "phone-job", "--status", "--json"]
OS_NAMES = ("macos", "linux")  # run-on's --os; a mini is macos
LINUX_CAP_MAX = 12  # a bound on configuration; the budgets and the runner share decide below it
MAC_CAP_MAX = 2  # a Mac lane measured 0.2 GiB at peak (Oct 2026); run-on's memory admission keeps the reserve


def resolved(host: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """The placement config, completed from the host registry runtime profile it names."""
    name = config.get("runtime_profile")
    if name is None:
        return dict(config)
    path = Path(os.environ.get("MENAGERIE_COMPUTE_HOSTS")
                or admission._desk() / "provision/compute-hosts.json")
    try:
        hosts = json.loads(path.read_text(encoding="utf-8"))["hosts"]
        profiles = hosts[host]["capabilities"]["runtime_profiles"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"host registry has no runtime profiles for {host}") from exc
    found = [p for p in profiles if isinstance(p, Mapping) and p.get("name") == name]
    if len(found) != 1:
        raise ValueError(f"runtime profile {name!r} not in the host registry for {host}")
    supplied = {k: found[0][k] for k in RUNTIME_KEYS if k in found[0]}
    return {**supplied, **config}


def _spec(host: Any, config: Any) -> dict[str, Any]:
    if not isinstance(host, str) or not NAME.fullmatch(host):
        raise ValueError("mini worker names must be safe registry host names")
    if not isinstance(config, Mapping):
        raise ValueError("a mini worker's configuration must be a mapping")  # noqa: TRY004
    return validated(resolved(host, config))


def validated(config: Mapping[str, Any]) -> dict[str, Any]:
    """An enabled worker must have a measured peak and an identified existing runtime."""
    system = config.get("os", "macos")
    if system not in OS_NAMES:
        raise ValueError(f"worker os must be one of {', '.join(OS_NAMES)}")
    cap = config.get("cap", 0)
    if system == "macos" and (type(cap) is not int or not 0 <= cap <= MAC_CAP_MAX):
        raise ValueError(f"mini worker cap must be 0 to {MAC_CAP_MAX}")
    if type(cap) is not int or not 0 <= cap <= LINUX_CAP_MAX:
        raise ValueError(f"linux worker cap must be 0 to {LINUX_CAP_MAX}")
    if not isinstance(config.get("trace_host", False), bool):
        raise ValueError("worker trace_host must be true or false")  # noqa: TRY004
    if cap == 0:
        return dict(config)
    _memory(config, cap)
    _cpus(config, cap, system)
    _commands(config)
    return {**config, "os": system}


def _cpus(config: Mapping[str, Any], cap: int, system: str) -> None:
    """Each lane's declared CPUs; a Linux worker's lanes fit its declared CPU budget and it says
    whether it is the trace host (a registry name need not match the disk guard's ssh alias).
    A Linux lane may declare a quarter, half or three quarters of a CPU (run-on's declaration for
    work that waits on a model API), so its share counts what the lanes use."""
    cpus = config.get("cpus", 1)
    if system == "macos" and cpus != 1:
        raise ValueError("a mini lane declares one CPU")
    if not _lane_cpus(cpus):
        raise ValueError("worker cpus must be a whole number per lane, or 0.25, 0.5 or 0.75")
    if system == "linux":
        if "trace_host" not in config:
            raise ValueError("enabled linux workers must set trace_host (true on the trace runners' host)")
        budget = config.get("budget_cpus")
        if type(budget) is not int or budget < 1:
            raise ValueError("enabled linux workers need a whole positive budget_cpus")
        if cap * cpus > budget:
            raise ValueError("linux worker cap exceeds its CPU budget")


def _lane_cpus(cpus: Any) -> bool:
    if type(cpus) is int:
        return cpus >= 1
    return type(cpus) is float and cpus in (0.25, 0.5, 0.75)


def _memory(config: Mapping[str, Any], cap: int) -> None:
    """The measured peak and physical-footprint margin bound concurrent lane count."""
    peak, budget = config.get("peak_gib"), config.get("budget_gib")
    for value in (peak, budget):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("enabled mini workers need positive measured peak_gib and budget_gib")
    if cap > math.floor(budget / (MEMORY_MARGIN * peak)):
        raise ValueError("mini worker cap exceeds its measured memory budget")


def _argv(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, str) and v for v in value)


def _commands(config: Mapping[str, Any]) -> None:
    """Require the configured runtime and its identity probe, plus a safe SSH alias."""
    missing = [key for key in ("runtime", "identity") if key not in config]
    if missing:
        raise ValueError(f"mini runtime not configured ({', '.join(missing)} missing)")
    if not all(_argv(config[key]) for key in ("runtime", "identity")):
        raise ValueError("mini worker runtime and identity must be command argv")
    if any(not isinstance(config.get(key), str) or not NAME.fullmatch(config[key])
           for key in ("ssh_alias", "account")):
        raise ValueError("mini workers need a safe hub-to-worker SSH alias and identified account")
    _options(config)


def _options(config: Mapping[str, Any]) -> None:
    """The optional hub login hook, worker receiver and the phone-builds-first flag."""
    for key in ("deliver", "prepare"):
        if key in config and not _argv(config[key]):
            raise ValueError(f"mini worker {key} must be command argv")
    if not isinstance(config.get("phone_first", False), bool):
        raise ValueError("mini worker phone_first must be true or false")  # noqa: TRY004


def status(cfg: Mapping[str, Any]) -> dict[str, str]:
    """One plain line per configured worker: why it has no slot, or how many it has."""
    lines = {}
    try:
        configured = _configured(cfg)
    except ValueError as exc:
        return {"mini_workers": f"off: {exc}"}
    for host, config in configured.items():
        try:
            spec = _spec(host, config)
        except ValueError as exc:
            lines[host] = f"off: {exc}"
            continue
        if not spec.get("cap"):
            lines[host] = "off: cap 0"
            continue
        allowed, why = host_cap(host, spec, cfg)
        lines[host] = f"on: cap {spec['cap']}" + (f"; {why}" if allowed < spec["cap"] else "")
    if any(line.startswith("on") for line in lines.values()) and not serving():
        lines = {h: ("held: bridge or mini relay not serving" if line.startswith("on") else line)
                 for h, line in lines.items()}
    return lines


def slots(cfg: Mapping[str, Any]) -> list[str]:
    """Free configured recipe slots, counting remote leases in the same global lane table."""
    try:
        enabled = _workers(cfg)
    except ValueError:
        return []  # ``status`` names the malformed configuration
    if not enabled or not serving():
        return []
    busy = Counter(x.get("placement") for x in admission.lanes(data=DataLayout.from_env())
                   if x["kind"] == "2b")
    return [host for host, spec in enabled.items()
            for _ in range(max(0, host_cap(host, spec, cfg)[0] - busy[host]))]


def _trace_host(host: str, spec: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    """Whether the trace runners and their store share this worker: its ``trace_host`` (required
    on Linux), else whether its name or ssh alias is the disk guard's host."""
    from menagerie.pipeline import disk_guard

    if "trace_host" in spec:
        return spec["trace_host"]
    runner_host = disk_guard.Config.from_mapping(cfg.get("disk")).host
    return bool(runner_host) and runner_host in (host, spec.get("ssh_alias"))


def _disk_state(cfg: Mapping[str, Any]) -> str:
    """This pass's disk guard state (the tick's ``disk_state``), else the saved one while it is
    fresher than the guard's own unknown limit; anything else is ``unknown``."""
    from menagerie.pipeline import disk_guard, tick

    if cfg.get("disk_state"):
        return str(cfg["disk_state"])
    saved = tick.load_state().get("disk_guard") or {}
    try:
        read = datetime.fromisoformat(str(saved["read_at"]))
        read = read if read.tzinfo else read.replace(tzinfo=timezone.utc)
    except (KeyError, ValueError):
        return "unknown"
    if datetime.now(timezone.utc) - read > disk_guard.UNKNOWN_AFTER:
        return "unknown"
    return str(saved.get("state") or "unknown")


def host_cap(host: str, spec: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[int, str]:
    """The lanes ``host`` may start on now and, when fewer than its cap, why.

    On the trace host no lane starts unless the disk guard reads ``ok`` or while the night big
    runner comes first (``_big_night``: G runs with no room left for a lane, or would fit once this
    host's lanes finish), and
    lanes fit the share the runners leave with A sized for the 2b lanes the workers can reach
    (``_reachable_2b``; with ``2b_on_hub`` the whole 2b admission cap, since the tick's lower
    ``lane_limits`` do not bound manual launches), each pledged 1.25 times its peak. A check that fails holds the host, with the reason, rather than the tick.
    """
    cap = spec["cap"]
    try:
        return _trace_cap(host, spec, cfg, cap)
    except Exception as exc:  # noqa: BLE001 - fail closed; the status line names why.
        return 0, f"held: trace-host check failed ({type(exc).__name__}: {exc})"[:300]


def _trace_cap(host: str, spec: Mapping[str, Any], cfg: Mapping[str, Any], cap: int) -> tuple[int, str]:
    from menagerie.pipeline import runner_job

    if not _trace_host(host, spec, cfg):
        return cap, ""
    disk = _disk_state(cfg)
    if disk != "ok":
        return 0, f"held: disk guard {disk} on the trace host"
    if cfg.get("b_before_lanes", True) and _b_held_out():
        return 0, "held: runner B is down and refused for capacity; stage 3 comes first"
    if cfg.get("big_runner") and _big_night(host, spec):
        return 0, ("held: the night big runner comes first (running with no room for a lane, or "
                   "fitting once this host's lanes finish)")
    # The most lanes that fit the room left once A is sized for them (more lanes, larger A).
    for want in range(cap, -1, -1):
        cpus, gib = runner_job.lane_room(_reachable_2b(host, cfg, want))
        fit = min(math.floor(cpus / spec.get("cpus", 1)), math.floor(gib / (MEMORY_MARGIN * spec["peak_gib"])))
        if fit >= want:
            break
    if want < cap:
        return want, f"runner share leaves room for {want} ({cpus} CPUs, {gib} GiB)"
    return cap, ""


def _reachable_2b(host: str, cfg: Mapping[str, Any], own: int) -> int:
    """The live 2b lanes runner A must be sized for when ``host`` runs ``own`` of them.

    With ``2b_on_hub`` off the tick places 2b lanes only on the workers, so A (one worker per five
    live lanes) is reserved for the workers' caps, not the nominal 2b cap, and the room A could
    never use goes to the trace host's lanes. Runner B's seat is never counted as room. A manual
    hub launch beyond this makes A's start fall back to a smaller size (``runner_job``'s ladder).
    """
    cap = admission.lane_cap("2b")
    if cfg.get("2b_on_hub", False):
        return cap
    others = sum(spec["cap"] for name, spec in _workers(cfg).items() if name != host)
    return min(cap, others + own)


def _b_held_out() -> bool:
    """Whether runner B is down by day and its last start was refused for capacity, so the trace
    host starts no new 2b lanes (Crawler, 28 Oct: the stage 3 runners come before the trace host's
    2b lanes at any hour). Running lanes finish, nothing is killed, and lanes resume once the tick
    starts B again at any size of its ladder. From 10 PM to 6 AM Pacific the big-runner rule
    (``_big_night``) decides instead. Reads the tick state of the last pass."""
    from menagerie.pipeline import runner_job, tick

    if runner_job.night_open():
        return False
    return runner_job.refused(runner_job.Record.from_mapping(
        (tick.load_state().get("runners") or {}).get("B")))


def _big_night(host: str, spec: Mapping[str, Any]) -> bool:
    """Whether the night big runner holds new lanes off the trace host now (Crawler, 28 Oct: at
    night G comes before the trace host's 2b lanes; runner A is never displaced).

    While G's job runs, lanes start only while run-on would admit one beside it (``_lane_fits``).
    Before it starts, in its night (with the lead-in) with rows for it, the hold begins when run-on
    would admit G with this host's live 2b lanes gone (``_g_fits``) and lasts while lanes drain
    (``HOLD_FILE``), since the answer moves with the runners' attempts: running lanes finish,
    nothing is killed, and G starts once its own probe admits it. With no lanes left and G still
    refused (another project's job holds the room), the hold ends and the lanes get their slots.
    Reads the tick state of the last pass."""
    # Complexity C-12: the hold is one small state machine (G running, window, begin, drain, end)
    # and reads clearer whole than split across helpers.
    from menagerie.pipeline import runner_job, tick

    state = tick.load_state()
    runners = state.get("runners") or {}
    g = runner_job.Record.from_mapping(runners.get("G"))
    if _g_running(g):
        return not _lane_fits(spec)
    marker = work_dir("tick") / HOLD_FILE
    if not (runner_job.big_hold_open() and runner_job.big_work(state)):
        marker.unlink(missing_ok=True)
        return False
    lanes = _host_lanes(host)
    fits = _g_fits(runners.get("B") or {}, (lanes * spec.get("cpus", 1), lanes * spec["peak_gib"]),
                   replaces_b=bool(tick.config().get("g_replaces_b", False)))
    held = marker.is_file() and time.time() - marker.stat().st_mtime < HOLD_MAX_S
    if fits and not held:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(now() + "\n", encoding="utf-8")
    elif not fits and (not held or lanes == 0):
        marker.unlink(missing_ok=True)
        return False
    return True


def _host_lanes(host: str) -> int:
    """The 2b lane leases placed on ``host``, read without the lease lock: the slot checks run
    inside it (``launch`` under ``_reserve``'s callers), so ``admission.lanes`` would deadlock. An
    ended lane's lease not yet released counts until the next pass releases it."""
    count = 0
    for path in work_dir("leases").glob("*.json"):
        try:
            lease = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        count += lease.get("kind") == "2b" and lease.get("placement") == host
    return count


def _g_running(g: Any) -> bool:
    """G's recorded job is its live one: not the job the tick cancelled, not a refused submit."""
    if not g.job or g.job == g.cancelled:
        return False
    return not (g.refused_at and g.submitted_at and g.refused_at >= g.submitted_at)


G_FITS_TTL_S = 300  # one probe per tick pass, however many slot checks ask
HOLD_FILE = "g-night-hold"  # in the tick folder: this night's hold of the trace host's lanes for G
HOLD_MAX_S = 9 * 3600  # one night with its lead-in; an older file is last night's
_g_fits_memo: dict[str, tuple[float, bool]] = {}


def _g_fits(b: Mapping[str, Any], freed: tuple[float, float] = (0, 0), *,
            replaces_b: bool = False) -> bool:
    """Whether run-on admits G now, less ``freed`` (the CPUs and GiB the trace host's live lanes
    declare, released as they finish). With ``replaces_b`` G takes B's place at night: G alone in
    place of B's live job. Else G alone once B runs at its night size (``b``'s record says size 0),
    or G plus B's night size in place of B's job. A runs at its night size all day, so this is G's
    night layout from the 9:30 PM lead-in on. An unknown answer holds, as before; only a refusal
    frees the lanes."""
    from menagerie.pipeline import runner_job

    g = next((r for r in runner_job.runners() if r.name == "G"), None)
    if g is None:
        return False
    b_cpus, b_gib, replacing = _b_beside_g(b, replaces_b)
    layout = replace(g, cpus=max(1, g.cpus + b_cpus - freed[0]),
                     memory_gib=max(1, g.memory_gib + b_gib - freed[1]))
    return _admits(layout, replacing)


def _b_beside_g(b: Mapping[str, Any], replaces_b: bool) -> tuple[int, int, str | None]:
    """B's night CPUs and GiB added to G's probe, and the B job it replaces."""
    from menagerie.pipeline import runner_job

    live = b.get("job") if b.get("job") and b.get("job") != b.get("cancelled") else None
    if replaces_b:
        return 0, 0, live
    if b.get("size") == 0 and live:
        return 0, 0, None  # B already small: G alone beside it
    _, cpus, gib = runner_job.B_SIZES[0]
    return cpus, gib, live


def _lane_fits(spec: Mapping[str, Any]) -> bool:
    """Whether run-on admits one more lane of ``spec`` beside the running runners now."""
    from menagerie.pipeline import runner_job

    g = next((r for r in runner_job.runners() if r.name == "G"), None)
    if g is None:
        return False
    return _admits(replace(g, cpus=spec.get("cpus", 1), memory_gib=spec["peak_gib"]), None)


def _admits(layout: Any, replacing: str | None) -> bool:
    """``run-on probe`` for ``layout``, remembered for a pass (``G_FITS_TTL_S``)."""
    from menagerie.pipeline import runner_job

    key = f"{layout.cpus}/{layout.memory_gib}/{replacing or ''}"
    hit = _g_fits_memo.get(key)
    if hit and time.monotonic() - hit[0] < G_FITS_TTL_S:
        return hit[1]
    code, _ = runner_job.RunOn().probe(layout, replacing)
    fits = code != runner_job.PROBE_REFUSED
    _g_fits_memo[key] = (time.monotonic(), fits)
    return fits


def _configured(cfg: Mapping[str, Any]) -> dict[Any, Any]:
    """The configured workers; each entry is checked on its own (``_spec``)."""
    workers = cfg.get("mini_workers") or {}
    if not isinstance(workers, Mapping):
        raise ValueError("mini_workers must map registry host names to worker configuration")  # noqa: TRY004
    return dict(workers)


def _workers(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Enabled, valid workers; an invalid one is off and named in ``status``, never fatal."""
    enabled = {}
    for host, config in _configured(cfg).items():
        try:
            spec = _spec(host, config)
        except ValueError:
            continue
        if spec.get("cap", 0):
            enabled[host] = spec
    return enabled


def serving() -> bool:
    """The bridge forwards tries to the runner; the relay answers the minis' verbs."""
    bridge = Path(os.environ.get("MENAGERIE_BRIDGE_SPOOL") or work_dir("hubspool"))
    return runner.serving(bridge) and mini_relay.serving()


def _phone_build(host: str) -> bool:
    """Whether the fleet host roles list the host for phone builds; unknown counts as yes."""
    path = Path(os.environ.get("MENAGERIE_HOST_ROLES")
                or admission._desk() / "provision/host-roles.json")
    try:
        return host in json.loads(path.read_text(encoding="utf-8"))["roles"]["phone-build"]["hosts"]
    except (OSError, ValueError, KeyError, TypeError):
        return True


def _phone_idle(spec: Mapping[str, Any]) -> bool:
    """Phone builds come first on a phone-build mini: unknown lock state counts as busy."""
    probe = admission._sh([*SSH, spec["ssh_alias"], shlex.join(PHONE_JOB)])
    try:
        state = json.loads(probe.stdout.strip().splitlines()[-1]) if probe.returncode == 0 else None
    except (ValueError, IndexError):
        return False
    devices = state.get("devices", []) if isinstance(state, dict) else None
    return _free(state) and isinstance(devices, list) and all(_free(d) for d in devices)


def _free(lock: Any) -> bool:
    """The phone tool's own rule (tools/phone/core/hosts.py ``idle``): only a known free lock."""
    return isinstance(lock, dict) and lock.get("held") is False and not lock.get("owner") \
        and not lock.get("queue")


def _runtime(role: str, spec: Mapping[str, Any], repair: bool, host: str) -> tuple[dict[str, Any], str, str]:
    """Use only a role and account the installed mini runtime can actually serve.

    A worker whose ``account`` is ``pool`` serves whichever pool account admission picks; its
    identity is then probed for that account after admission. A named account is probed first.
    The runtime serves Claude alone, so each role is resolved for that harness: background
    routing's other-lab seat cannot run here, and the role's own Claude seat is used instead
    (``admission._resolve``'s ``harness``), still behind the pool's admission gate.
    Returns the seat plan, the selected role and the account the lane runs on.
    """
    cwd = runner.CODE_ROOT
    pooled = spec["account"] == POOL
    if not pooled:
        _identity(spec, spec["account"])
    roles = [role] if repair else [role, admission.other_role(role)]
    for candidate in filter(None, roles):
        plan = admission._resolve(candidate, spec["account"], harness="claude")
        if plan["harness"] != "claude":
            continue
        _, account = admission._admit("claude", candidate, cwd, "2b")
        if pooled:
            _identity(spec, account)
        elif account != spec["account"]:
            raise admission.Refused("no_admission", "mini runtime account differs from admitted account")
        return plan, candidate, account
    raise admission.Refused("no_admission", "no compatible mini runtime for the requested role")


def _identity(spec: Mapping[str, Any], account: str) -> None:
    """The worker's runtime must answer as Claude on ``account``, else the lane is refused."""
    # The same probe runs inside the job; here, before any job exists, its job id is "preflight".
    probe = expand(spec["identity"], {"job": "preflight", "account": account})
    checked = admission._sh([*SSH, spec["ssh_alias"], shlex.join(probe)])
    try:
        identity = json.loads(checked.stdout) if checked.returncode == 0 else {}
    except ValueError as exc:
        raise admission.Refused("no_admission", "mini runtime identity unavailable") from exc
    if not isinstance(identity, dict) or identity.get("harness") != "claude" or identity.get("account") != account:
        raise admission.Refused("no_admission", "mini runtime identity does not match the admitted pool")


def launch(families: Sequence[str], *, role: str, data: DataLayout, host: str,
           config: Mapping[str, Any], tier: str | None = None,
           cfg: Mapping[str, Any] | None = None) -> str:
    """Lease a worker slot and its families, then start a pinned durable remote recipe job and
    read its admission (``start`` then ``finish``).

    ``cfg`` is the tick config the trace-host checks read (``host_cap``); outside the tick the
    saved tick config and state are read."""

    return finish(start(families, role=role, data=data, host=host, config=config, tier=tier,
                        cfg=cfg))


@dataclass(frozen=True)
class Started:
    """A remote lane whose job start is proven and whose run-on admission is not read yet; its
    lease still says ``pending:<label>``, so the families and the slot stay held."""

    host: str
    path: Path
    lease: dict[str, Any]
    families: tuple[str, ...]
    tier: str | None


def start(families: Sequence[str], *, role: str, data: DataLayout, host: str,
          config: Mapping[str, Any], tier: str | None = None,
          cfg: Mapping[str, Any] | None = None) -> Started:
    """``launch`` up to the proven job start, so a caller starting several lanes can read their
    admissions together (``finish``) instead of waiting out each in turn."""
    from menagerie.pipeline import implement, tick

    cfg = tick.config() if cfg is None else cfg
    spec = _admissible(host, config, cfg)
    _families(families, data)
    label = "2b-mini-" + os.urandom(6).hex()
    from menagerie.pipeline.records import Packet

    # Keep the guarded lifecycle and accepted launch events together at this entry point.
    repair = any(implement.fabrications(read_json(Packet, data.packet(f)), data) for f in families)
    plan, selected_role, account = _runtime(role, spec, repair, host)
    spec = {**spec, "account": account}  # the lane runs on the admitted account
    path, lease = _reserve(families, host, spec, label, plan, selected_role, cfg)
    try:
        brief = implement.batch_brief(families, label=label, data=data)
    except Exception:
        path.unlink()  # Preparation failed before any remote job could start.
        raise
    run = _submit(host, spec, families, label, brief, plan)  # unproven: the lease stays held
    lease = {**lease, "run": run, "brief": str(brief),
             "result": str(brief.with_name("BRIEF.result.md"))}
    return Started(host, path, lease, tuple(families), tier)


def finish(started: Started) -> str:
    """Read a started lane's run-on admission: a proven refusal releases its lease and raises
    ``Refused`` (the caller may fall back); otherwise the lease takes the run and the launch
    event is written. An unreadable admission raises and keeps the provisional lease."""

    try:
        _admitted(started.lease["label"])
    except admission.Refused:
        started.path.unlink()
        raise
    lease = {**started.lease, "expires": admission._stamp(datetime.now(timezone.utc)
                                                          + timedelta(minutes=LEASE_MINUTES))}
    admission._write_atomic(started.path, json.dumps(lease) + "\n")
    run = lease["run"]
    drive.event(run, "", "launch", **{k: v for k, v in lease.items() if k != "run"},
                gauges=admission.gauges(lease["account"]))
    for family in started.families:
        if len(started.families) > 1:
            drive.event(run, family, "batch:assigned")
        if started.tier:
            drive.event(run, family, "tier", tier=started.tier)
    return run


def _admissible(host: str, config: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The worker's spec when a lane may start there now, else a ``Refused`` naming why."""
    if not serving():
        raise admission.Refused("no_admission", "bridge or mini relay not serving")
    try:
        spec = _spec(host, config)
    except ValueError as exc:
        raise admission.Refused("no_admission", f"mini worker off: {exc}") from exc
    if not spec.get("cap"):
        raise admission.Refused("no_admission", "mini worker disabled or invalid")
    allowed, why = host_cap(host, spec, cfg)
    if not allowed:
        raise admission.Refused("no_admission", f"worker {host}: {why}")
    if _phone_first(host, spec) and not _phone_idle(spec):
        raise admission.Refused("no_admission", "phone builds first: the mini's phone lock is busy")
    return spec


def _phone_first(host: str, spec: Mapping[str, Any]) -> bool:
    """Phone builds come first on a mini so flagged or listed; a Linux worker builds no phones."""
    return spec.get("os", "macos") == "macos" and bool(spec.get("phone_first") or _phone_build(host))


def _families(families: Sequence[str], data: DataLayout) -> None:
    from menagerie.pipeline import implement

    if not families or len(families) > 6 or len(set(families)) != len(families):
        raise ValueError("a mini lane needs one to six distinct families")
    if any(not implement.LABEL_FAMILY.fullmatch(f) for f in families):
        raise ValueError("mini lane families must be safe family keys")
    if len(families) > 1 and len({implement.source_case(f, data=data) for f in families}) != 1:
        raise ValueError("a mini batch must use one source case")

def _reserve(families: Sequence[str], host: str, spec: Mapping[str, Any], label: str,
             plan: Mapping[str, Any], role: str,
             cfg: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Reserve before any network preparation; ambiguous starts retain the family and slot."""
    with admission._lease_lock() as leases:
        admission._held(leases, families, label)
        live = [json.loads(p.read_text()) for p in leases.glob("*.json")
                if admission._live(p, json.loads(p.read_text())) is not None]
        occupied = sum(x.get("placement") == host for x in live)
        if occupied >= host_cap(host, spec, cfg)[0]:
            raise admission.Refused("no_admission", "mini worker slot taken")
        if not serving():
            raise admission.Refused("no_admission", "bridge or mini relay not serving")
        lease = {"label": label, "run": "pending:" + label, "kind": "2b", "keys": list(families),
                 "role": role, "harness": "claude", "model": plan["model"],
                 "effort": plan.get("effort"), "routing": plan.get("routing"), "ids": [],
                 "account": spec["account"], "placement": host, "web": False,
                 "mini": {"ssh_alias": spec["ssh_alias"], "deliver": spec.get("deliver")},
                 "parent": os.environ.get("FLEET_RUN_ID"), "started": now(),
                 "expires": admission._stamp(datetime.now(timezone.utc)
                                             + timedelta(minutes=LEASE_MINUTES))}
        path = leases / f"{label}.json"
        admission._write_atomic(path, json.dumps(lease) + "\n")
    return path, lease

def _submit(host: str, spec: Mapping[str, Any], families: Sequence[str], label: str,
            brief: Path, plan: Mapping[str, Any]) -> str:
    """The remote runner pins source, checks host clearance and records the job's peak.

    The job gets only what it runs with: the runtime, identity probe, model and the paths of
    its brief and result inside the hub-staged ``.mini-lane/`` folder. It never reaches the hub.
    """
    cwd = Path(runner.CODE_ROOT)
    ref = admission._sh(["git", "rev-parse", "HEAD"], cwd=cwd)
    if ref.returncode:
        raise OSError("cannot pin mini lane source")
    brief_rel, result_rel = mini_relay.layout(families, brief)
    manifest = {"runtime": spec["runtime"], "identity": spec["identity"], "model": plan["model"],
                "effort": plan.get("effort") or "high", "account": spec["account"],
                "label": label, "families": list(families), "brief": brief_rel,
                "result": result_rel, "lane_timeout_s": LANE_TIMEOUT_S,
                "prepare": spec.get("prepare"),
                "nice": PHONE_NICE if _phone_first(host, spec) else None}
    parent = os.environ.get("FLEET_RUN_ID")
    argv = [admission._fleet_run(), "start", "--kind", "remote", "--label", label,
            "--cwd", str(cwd), "--project", "menagerie", "--require-result",
            "--adapter", "run-on", "--deadline", "165m", "--hard-timeout", "180m",
            "--client-key", "run-on:" + label,
            *(["--parent", parent] if parent else []), "--", admission._adapter(), "run-on",
            admission._tool("run-on"), label,
            "--mode", "batch", "--project", "menagerie",
            "--host", host, "--os", spec.get("os", "macos"), "--no-fallback",
            "--ref", ref.stdout.strip(), "--cpus", str(spec.get("cpus", 1)),
            "--memory-gib", str(spec["peak_gib"]),
            *(arg for path in LANE_SPARSE_EXCLUDE for arg in ("--sparse-exclude", path)),
            "--job", label, "--pull", "outputs/lane/**", "--", LANE_PYTHON,
            "menagerie/pipeline/mini_lane.py", "run", json.dumps(manifest)]
    started = admission._sh(argv, cwd=cwd)
    run = re.search(r"run_id=(\S+)", started.stdout)
    if started.returncode or not run:
        raise OSError("mini start was not proven; provisional family lease retained")
    return run[1]


REFUSED_VERDICTS = ("refused", "waiting", "admission_timeout")


def _run_on_state(label: str) -> dict[str, Any]:
    """``run-on state --json``: its record whatever the exit code, since it exits non-zero for
    exactly the verdicts ``_admitted`` acts on (75 admission_timeout, 6 refused); none when no
    JSON came back."""
    result = admission._sh([admission._tool("run-on"), "state", label, "--json"])
    try:
        state = json.loads(result.stdout)
    except ValueError:
        return {}
    return state if isinstance(state, dict) else {}


def _admitted(label: str) -> None:
    """Only a proven pre-payload refusal authorizes the immediate hub fallback."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = _run_on_state(label)
        # admission_timeout: run-on's admission wait ran out with no room (NO_CAPACITY); nothing ran
        if state.get("verdict") in REFUSED_VERDICTS and state.get("state") == "TERMINAL":
            raise admission.Refused("no_admission", "mini worker refused before payload start")
        # A launch that failed after the claim, with the worker showing no run: nothing ran, so
        # the slot and the families go back at once instead of waiting out the relay's pulls.
        if state.get("ran") is False and state.get("launch_failed"):
            raise admission.Refused("no_admission", "mini worker launch failed before payload "
                                    f"start ({state['launch_failed']})")
        if state.get("verdict") not in ("pending", "unknown", None):
            return
        time.sleep(1)
