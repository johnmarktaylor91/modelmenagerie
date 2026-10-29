"""Stage 3 for one model record: forward, trace, store, render and ledger rows.

Owned by the stage-3 lane; ``stage3 serve`` runs it on the worker for each queued member, and
``python -m menagerie.pipeline.flow`` runs it for a list of ids. For each model ``run_model`` runs,
through the sandboxed runner and one attempt at a time:
- ``forward`` (eval, then train on a fresh build, with the divergence class); a forward that
  fails with ``memory_cap``, ``oversized``, ``needs_gpu`` or ``timeout`` goes straight to a
  structure-only capture, and any other failed forward ends the model there;
- ``trace`` under a cap sized from the forward's measured peak (fix 10: 5.25 times, or the
  family's measured ratio, or the member's own measured peak, as ``Sizing`` says; floor 8 GiB,
  or ``ROW_FLOOR_GIB`` on runner B with ``per_row_floor``; clamped to the declaration); a projection beyond the worker gets a typed
  ``oversized:projected`` trace row naming its placement and a structure-only capture at once;
  a ``memory_cap`` is retried once with the whole declaration (on a runner with several workers, one
  18 GiB worker slot, ``_retry_caps``); a trace that still ends in
  ``memory_cap``, ``timeout``, ``oversized`` (``oversized:output`` is the runner's attempt output
  cap) or ``needs_gpu`` gets a structure-only capture and no train trace;
- ``trace-structure-only`` alone for a recipe placed off the worker (``records.OFF_WORKER``);
- a train-mode ``trace`` when the forward found a structural divergence;
- ``render`` from the stored portable save (the runnable save when no portable one exists).
On runner B with ``per_row_floor`` the forward and render pledge their own peaks too (``_row_caps``:
the measured trace peak bounds the forward, the trace attempt's peak its render; at least
``ROW_FLOOR_GIB``, never above the flat pledge), and a pledge that breaks retries once in the slot.
Every saved trace, summary and render artifact goes into the store, and every row into the
ledger. A record with no recipe carries its typed give-up and is not run. Torch is imported only
in the sandbox. Explicit ``caps`` (the in-process path) skip the sizing and the retry.

With ``spool`` set, every request goes through that runner spool (``runner serve``) and its attempt
directory is the spool's; adapter recipes carry their committed adapter's text (``adapter_src``),
because the runner's own data root is not the one the records came from.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from menagerie.pipeline import DataLayout, StoreLayout, read_json
from menagerie.pipeline.records import (
    OFF_WORKER, Caps, LedgerRow, ModelRecord, Placement, RenderRow, RunResult,
)

MIME = {"tlspec": "application/x-tlspec", "tlspec-portable": "application/x-tlspec",
        "summary": "application/json", "svg": "image/svg+xml", "pdf": "application/pdf",
        "thumb": "image/webp", "tiles": "application/x-dzi", "html": "text/html",
        "netron": "application/json", "model_explorer": "application/json",
        "log": "text/plain"}
# Fix 10, BUILD-PLAN r2 section 5: a trace's projected peak is this many times the forward's
# measured peak, never under the floor. Both peaks are the process tree's, the quantity the
# runner caps (refit 2026-10-05 on 144 catalog rows: trace over forward tree peak median 2.40,
# p95 3.78, max 5.14; it was 3.5 times the worker's own forward peak, which ran 7 to 9 for the
# largest models). Up to WORKER_GIB it fits a worker slot, up to CLUSTER_GIB the single-worker
# slot (when the declaration holds it), up to REVIEWED_GIB the cluster's cpu batch, and beyond
# that the reviewed allocation.
TRACE_PEAK_RATIO = 5.25
TRACE_FLOOR_GIB = 8.0
# A full trace that hit `memory_cap` at a peak needs more than that peak: the next allocation
# tried is at least this many times it (a failure at 40 GiB skips the 64 GiB batch).
MEMORY_CAP_STEP = 1.5
WORKER_GIB, CLUSTER_GIB, REVIEWED_GIB = 18.0, 50.0, 60.0
BIG_GIB = 40.0  # runner G's declaration (runner_job.DEFAULT_RUNNERS); spool-G takes (WORKER_GIB, BIG_GIB]
# Runner B's floor with the tick's ``per_row_floor`` (Crawler, 28 Oct): worker full traces peaked at a
# median 4.4 GiB that day (90th percentile 15.2), so an 8 GiB floor packed B's 36 GiB three or four deep.
ROW_FLOOR_GIB = 4.0
# Failure heads after which a structure-only capture is taken at once (section 5).
STRUCTURE_ONLY_AFTER = ("memory_cap", "oversized", "needs_gpu", "timeout")
GPU_WINDOW = "infra:gpu_window"  # the worker's refusal of a `gpu` request outside the window
GpuPart = Literal["all", "cpu", "gpu"]


@dataclass
class Outcome:
    """Everything one model produced, for the ledger and the rung table."""

    id: str
    results: list[RunResult] = field(default_factory=list)
    rows: list[LedgerRow | RenderRow] = field(default_factory=list)
    stored: dict[str, str] = field(default_factory=dict)  # kind/path -> sha256
    placement: Placement | None = None  # where the full trace belongs, when not this worker
    held: bool = False  # the GPU attempt waits for the GPU window (``infra:gpu_window``)

    def last(self, mode: str) -> RunResult | None:
        found = [r for r in self.results if r.mode == mode]
        return found[-1] if found else None


def model_mode_of(result: RunResult) -> str:
    """The model mode a request ran in (request names end ``-<mode>-<stamp>-<hex>``)."""

    return result.request.split("-")[-3]


def _store_files(result: RunResult, attempt: Path, *, store: StoreLayout,
                 data: DataLayout) -> dict[str, str]:
    from menagerie.pipeline import store as store_mod

    stored = {}
    for item in result.files:
        path = attempt / item.path
        if item.kind in MIME and path.exists() and item.kind != "log":
            line = store_mod.put(path, store=store, data=data, mime=MIME[item.kind],
                                 license_class="derived")
            stored[f"{item.kind}:{item.path}"] = line.sha256
    return stored


def adapter_src(record: ModelRecord, data: DataLayout) -> str | None:
    """The committed adapter file's text for an adapter recipe, else ``None``."""

    recipe = record.recipe
    if recipe is None or recipe.kind != "adapter" or not recipe.adapter:
        return None
    return (data.root / recipe.adapter).read_text(encoding="utf-8")


def _attempt_root(attempts: Path, spool: Path | None) -> Path:
    """Where attempt directories are: the spool's own through a runner, else ``attempts``."""

    return attempts if spool is None else Path(spool) / "attempts"


def _run(request: Any, outcome: Outcome, *, store: StoreLayout, data: DataLayout,
         attempts: Path, spool: Path | None = None,
         settled_forward: RunResult | None = None) -> RunResult:
    from menagerie.pipeline import runner

    if request.mode == "forward" and settled_forward is not None:
        return settled_forward  # existing evidence is not appended or stored again
    if spool is None:
        result = runner.attempt(request, attempt_dir=attempts / request.request)
    else:
        result = runner.run(request, spool=spool)
    outcome.results.append(result)
    outcome.rows.extend(result.rows)
    outcome.stored.update(_store_files(result, attempts / request.request, store=store,
                                       data=data))
    return result


def render_request(record: ModelRecord, trace: RunResult, trace_attempt: Path, *,
                   caps: Caps | None, attempts: Path, adapter: str | None = None) -> Any | None:
    """The render request for a saved trace, keyed on the trace row's own ``trace`` sha.

    The ledger matches a render row to the trace row through that sha (the runnable save, or the
    one save a structure-only capture or a refused runnable save has). The portable save draws
    like the live trace, so when the capture wrote one it is staged into the render attempt as
    ``render.tlspec``, which render mode reads before the store blob the request names. Without
    one (``trace.tlspec`` is then itself the portable save, or the only save) the attempt's
    ``trace.tlspec`` is staged, the same tree as that blob; with neither on disk, render mode
    reads the blob.
    """

    from menagerie.pipeline import runner
    from menagerie.pipeline.worker import RENDER_FILE, TRACE_FILE

    files = {f.kind: f for f in trace.files}
    key = files.get("tlspec")
    if key is None:
        return None
    request = runner.request_for(record, "render", caps=caps, trace=key.sha256,
                                 model_mode=model_mode_of(trace),  # type: ignore[arg-type]
                                 adapter_src=adapter)
    for save, name in ((files.get("tlspec-portable"), RENDER_FILE), (key, TRACE_FILE)):
        if save is not None and (trace_attempt / save.path).is_dir():
            stage_render(trace_attempt / save.path, attempts / request.request / name)
            break
    return request


def stage_render(save: Path, staged: Path) -> None:
    """Stage a trace save for render as hardlinks (render only loads it, so the save costs
    its size once on disk); copy it when the two are on different filesystems."""

    import shutil

    try:
        shutil.copytree(save, staged, dirs_exist_ok=True, copy_function=os.link)
    except (OSError, shutil.Error):
        shutil.rmtree(staged, ignore_errors=True)
        shutil.copytree(save, staged, dirs_exist_ok=True)


@dataclass(frozen=True)
class Sizing:
    """What the hub's ledger measured for a member (``stage3.sizing``), for its trace projection.

    ``measured_gib``: the member's own passing full trace peak with the runner's headroom; it is
    the projection when present. Otherwise the forward's peak times ``ratio`` (the family's
    largest measured trace to forward ratio, else ``TRACE_PEAK_RATIO``), and at least
    ``at_least_gib`` (``MEMORY_CAP_STEP`` times the member's own largest ``memory_cap`` peak).
    """

    ratio: float = TRACE_PEAK_RATIO
    at_least_gib: float | None = None
    measured_gib: float | None = None
    basis: str = f"{TRACE_PEAK_RATIO} x the forward's measured peak"
    floor_gib: float = TRACE_FLOOR_GIB
    big_runner: bool = False

    def project(self, forward_gib: float | None) -> float | None:
        if self.measured_gib:
            return self.measured_gib
        found = [g for g in (forward_gib * self.ratio if forward_gib else None,
                             self.at_least_gib) if g]
        return max(found) if found else None


@dataclass(frozen=True)
class TracePlan:
    """The eval trace's caps, its projected peak and, when it is beyond the worker, where it goes."""

    caps: Caps | None
    projected_gib: float | None
    placement: Placement | None = None


def _head(cls: str | None) -> str:
    return (cls or "").partition(":")[0]


def placement_for(projected_gib: float, *, big_runner: bool = False) -> Placement:
    """The placement section 5 gives a projected trace peak."""

    if projected_gib <= WORKER_GIB:
        return "worker"
    # With the big runner on, G is the single worker: past its 40 GiB is the cluster's (2026-10-28).
    if projected_gib <= (BIG_GIB if big_runner else CLUSTER_GIB):
        return "single-worker"
    return "axon-cpu" if projected_gib <= REVIEWED_GIB else "axon-reviewed"


def forward_peak_mb(row: LedgerRow) -> int | None:
    """A forward row's peak on the cap's basis: the attempt's process-tree peak the runner writes
    into ``peak_rss_mb``, else the worker's own ``forward_peak_rss_mb``."""

    return row.peak_rss_mb or row.forward_peak_rss_mb


def _forward_peak_gib(record: ModelRecord, forward: RunResult | None) -> float | None:
    """The forward's measured peak (the process tree's, as the runner caps it, else the worker's
    own); else the recipe's measured ``memory_gib``; else unknown."""

    rows = [r for r in (forward.rows if forward else ()) if isinstance(r, LedgerRow)]
    peaks = [forward_peak_mb(r) for r in rows]
    found = [p for p in peaks if p]
    if found:
        return max(found) / 1024
    return record.recipe.memory_gib if record.recipe else None


def trace_plan(record: ModelRecord, forward: RunResult | None, *, timeout_s: int | None = None,
               memory_gib: float | None = None, sizing: Sizing | None = None) -> TracePlan:
    """Size the eval trace from the forward's measured peak (fix 10) and ``sizing``.

    ``memory_gib`` (a queued request's own cap) replaces the projection; ``timeout_s`` replaces
    the trace timeout. A projection over what the declaration holds (or over ``CLUSTER_GIB``) is
    beyond this worker: no caps, and the placement it belongs to.
    """

    from menagerie import host_budget
    from menagerie.pipeline import runner

    base = runner.default_caps("trace")
    if timeout_s:
        base = replace(base, timeout_s=timeout_s)
    sizing = sizing or Sizing()
    projected = memory_gib or sizing.project(_forward_peak_gib(record, forward))
    if projected is None:
        return TracePlan(base if timeout_s else None, None)
    share = host_budget.declared_memory_gb()
    limit = BIG_GIB if sizing.big_runner else CLUSTER_GIB
    if projected > min(share, limit):
        return TracePlan(None, round(projected, 2), _past_share(projected, sizing.big_runner))
    rss_mb = int(min(share, max(sizing.floor_gib, projected)) * 1024)
    return TracePlan(replace(base, rss_mb=rss_mb), round(projected, 2))


def run_model(record: ModelRecord, *, caps: Caps | None, store: StoreLayout, data: DataLayout,
              attempts: Path, spool: Path | None = None, timeout_s: int | None = None,
              memory_gib: float | None = None, part: GpuPart = "all",
              sizing: Sizing | None = None, settled_forward: RunResult | None = None,
              trace_only: bool = False) -> Outcome:
    """Run one model end to end; return what it produced (rows not yet in the ledger).

    ``caps`` ``None`` takes the runner's per-mode defaults (``runner.default_caps``) for the
    forward and sizes the trace from the forward's peak; ``timeout_s`` and ``memory_gib`` are a
    queued request's own trace timeout and cap. A ``gpu-window`` recipe gets its structure-only
    capture at once and one ``gpu`` attempt; ``part`` runs only the CPU capture (``cpu``) or
    only the GPU attempt (``gpu``). A GPU attempt refused as ``infra:gpu_window`` is dropped and
    the outcome is ``held``.
    """

    from menagerie.pipeline import runner

    outcome = Outcome(record.id)
    if record.recipe is None:  # a typed give-up on the record itself; nothing to run
        return outcome
    adapter = adapter_src(record, data)
    attempts = _attempt_root(attempts, spool)

    def go(mode: str, **kw: Any) -> RunResult:
        kw.setdefault("caps", caps)
        request = runner.request_for(record, mode, adapter_src=adapter, **kw)  # type: ignore[arg-type]
        return _run(request, outcome, store=store, data=data, attempts=attempts, spool=spool,
                    settled_forward=settled_forward)

    traces = _captures(record, go, outcome, part=part, caps=caps, timeout_s=timeout_s,
                       memory_gib=memory_gib, sizing=sizing or Sizing(),
                       trace_only=trace_only, has_forward=settled_forward is not None)
    def draw(trace: RunResult, cap: Caps | None) -> RunResult | None:
        request = render_request(record, trace, attempts / trace.request, caps=cap,
                                 attempts=attempts, adapter=adapter)
        return None if request is None else _run(request, outcome, store=store, data=data,
                                                  attempts=attempts, spool=spool)

    for trace in traces:
        if any(isinstance(r, RenderRow) for r in trace.rows):
            continue  # rendered in process: no save level accepted the trace
        _render(record, trace, draw, caps=caps, sizing=sizing or Sizing())
    return outcome


def _render(record: ModelRecord, trace: RunResult, draw: Any, *, caps: Caps | None,
            sizing: Sizing) -> None:
    """Render a trace under its per-row pledge where one applies (the trace's own peak), retried
    once in the slot when that pledge breaks; else under ``caps`` or the flat pledge."""

    first = caps if caps is not None else _row_caps(record, "render", _trace_peak_gib(trace),
                                                     sizing)
    drawn = draw(trace, first)
    if caps is None and first is not None and drawn is not None and _memory_cap(drawn):
        draw(trace, _retry_caps(first))


def _row_caps(record: ModelRecord, mode: str, basis_gib: float | None,
              sizing: Sizing) -> Caps | None:
    """A forward's or render's per-row pledge where the trace floor is ``ROW_FLOOR_GIB`` (runner B
    with ``per_row_floor``): the measured peak ``basis_gib`` with the runner's headroom, at least the floor and never
    above the flat pledge ``runner.request_for`` would give; ``None`` (that flat pledge) elsewhere
    or without a basis. A pledge that breaks is retried once under ``_retry_caps``."""

    from menagerie import host_budget
    from menagerie.pipeline import runner

    if sizing.floor_gib != ROW_FLOOR_GIB or not basis_gib:
        return None
    base = runner.default_caps(mode)  # type: ignore[arg-type]
    flat = base.rss_mb
    if record.recipe is not None and record.recipe.memory_gib:
        wanted = int(record.recipe.memory_gib * 1024 * runner.PEAK_HEADROOM)
        flat = min(int(host_budget.declared_memory_gb() * 1024), max(flat, wanted))
    pledge = int(max(ROW_FLOOR_GIB, basis_gib * runner.PEAK_HEADROOM) * 1024)
    return replace(base, rss_mb=min(flat, pledge))


def _trace_peak_gib(trace: RunResult) -> float | None:
    """The trace attempt's own measured peak: the basis of its render's pledge (render loads
    that trace's save, without activations)."""

    peaks = [r.peak_rss_mb for r in trace.rows if isinstance(r, LedgerRow) and r.peak_rss_mb]
    return max(peaks) / 1024 if peaks else None


def _captures(record: ModelRecord, go: Any, outcome: Outcome, *, part: GpuPart,
              caps: Caps | None, timeout_s: int | None, memory_gib: float | None,
              sizing: Sizing, trace_only: bool, has_forward: bool) -> list[RunResult]:
    """Choose captures by placement and forward evidence before the render phase."""

    if record.recipe.placement == "gpu-window":
        return _gpu_window(go, outcome, part)
    if record.recipe.placement in OFF_WORKER:
        return [go("trace-structure-only")]
    if trace_only and not has_forward:
        return _without_forward(record, go, outcome, caps=caps, timeout_s=timeout_s,
                                memory_gib=memory_gib, sizing=sizing)
    return _traces(record, go, outcome, caps=caps, timeout_s=timeout_s,
                   memory_gib=memory_gib, sizing=sizing)


def _without_forward(record: ModelRecord, go: Any, outcome: Outcome, *, caps: Caps | None,
                     timeout_s: int | None, memory_gib: float | None,
                     sizing: Sizing) -> list[RunResult]:
    """A historical trace may lack a forward row; retry it without inventing that row."""

    plan = TracePlan(caps, None) if caps is not None else trace_plan(
        record, None, timeout_s=timeout_s, memory_gib=memory_gib, sizing=sizing)
    if plan.placement is not None:
        outcome.placement = plan.placement
        outcome.results.append(_beyond(record, plan, sizing.basis))
        outcome.rows.extend(outcome.results[-1].rows)
        return [go("trace-structure-only")]
    traces = _full_trace(go, first=plan.caps, retry=caps is None)
    if _head(traces[-1].cls) in STRUCTURE_ONLY_AFTER:
        if _memory_cap(traces[-1]):
            outcome.placement = _past_share(plan.projected_gib, sizing.big_runner)
        traces.append(go("trace-structure-only"))
    if record.recipe.modes and record.recipe.modes.divergence == "structural":
        traces.extend(_train_trace(go, plan.caps, retry=caps is None))
    return traces


def _gpu_window(go: Any, outcome: Outcome, part: GpuPart) -> list[RunResult]:
    """The structure-only capture on the CPU, then one GPU attempt (forward, then trace)."""

    traces = [go("trace-structure-only")] if part in ("all", "cpu") else []
    if part in ("all", "gpu"):
        gpu = go("gpu")
        if gpu.cls == GPU_WINDOW:  # outside the window: hold it for the window, never retry
            outcome.results.remove(gpu)
            outcome.rows[:] = [r for r in outcome.rows if r not in gpu.rows]
            outcome.held = True
        else:
            traces.append(gpu)
    return traces


def _traces(record: ModelRecord, go: Any, outcome: Outcome, *, caps: Caps | None,
            timeout_s: int | None, memory_gib: float | None,
            sizing: Sizing) -> list[RunResult]:
    """Forward, then the traces section 5 calls for; returns the traces to render."""

    if sizing.big_runner and caps is None:
        from menagerie import host_budget
        from menagerie.pipeline import runner

        forward_caps = replace(runner.default_caps("forward"),
                               rss_mb=int(host_budget.declared_memory_gb() * 1024),
                               threads=host_budget.declared_cpus())
        forward = go("forward", caps=forward_caps)
    else:
        forward = _forward(record, go, sizing)
    if forward.outcome != "pass":
        if _head(forward.cls) not in STRUCTURE_ONLY_AFTER:
            return []  # the forward's own failure is the model's verdict
        if _memory_cap(forward):  # its trace projects past the cap the forward broke
            outcome.placement = _past_share(sizing.project(_forward_peak_gib(record, forward)),
                                            sizing.big_runner)
        return [go("trace-structure-only")]
    plan = TracePlan(caps, None) if caps is not None else trace_plan(
        record, forward, timeout_s=timeout_s, memory_gib=memory_gib, sizing=sizing)
    if plan.placement is not None:  # beyond this worker: typed, then structure only
        outcome.placement = plan.placement
        outcome.results.append(_beyond(record, plan, sizing.basis))
        outcome.rows.extend(outcome.results[-1].rows)
        return [go("trace-structure-only")]
    traces = _full_trace(go, first=plan.caps, retry=caps is None)
    if _head(traces[-1].cls) in STRUCTURE_ONLY_AFTER:
        if _memory_cap(traces[-1]):
            outcome.placement = _past_share(plan.projected_gib, sizing.big_runner)
        return [*traces, go("trace-structure-only")]
    if forward.modes and forward.modes.divergence == "structural":
        traces.extend(_train_trace(go, plan.caps, retry=caps is None))
    return traces


def _forward(record: ModelRecord, go: Any, sizing: Sizing) -> RunResult:
    """The forward under its per-row pledge where one applies (its measured trace peak bounds
    it), retried once in the slot when that pledge breaks; else under the flat pledge."""

    from menagerie.pipeline import runner

    measured = sizing.measured_gib / runner.PEAK_HEADROOM if sizing.measured_gib else None
    first = _row_caps(record, "forward", measured, sizing)
    if first is None:
        return go("forward")
    forward = go("forward", caps=first)
    return go("forward", caps=_retry_caps(first)) if _memory_cap(forward) else forward


def _beyond(record: ModelRecord, plan: TracePlan, basis: str) -> RunResult:
    """The typed trace result for a projection beyond the worker; nothing is run."""

    from menagerie import host_budget
    from menagerie.pipeline import runner

    request = runner.request_for(record, "trace", caps=runner.default_caps("trace"))
    msg = (f"projected trace peak {plan.projected_gib} GiB ({basis}) is beyond this worker's "
           f"{host_budget.declared_memory_gb():g} GiB; placement {plan.placement}; "
           "structure-only capture taken")
    return runner.not_run(request, "oversized:projected", msg)


def _past_share(projected_gib: float | None, big_runner: bool = False) -> Placement:
    """Where a trace goes that hit ``memory_cap`` at the whole declaration: past this worker."""

    from menagerie import host_budget

    found = placement_for(max(projected_gib or 0.0, host_budget.declared_memory_gb()),
                          big_runner=big_runner)
    return "single-worker" if found == "worker" else found


def _memory_cap(result: RunResult) -> bool:
    return _head(result.cls) == "memory_cap"


def _full_trace(go: Any, *, first: Caps | None = None, retry: bool) -> list[RunResult]:
    """The eval trace under ``first``; with sized or default caps, a ``memory_cap`` is retried
    once under ``_retry_caps`` (straight to the slot, however low the first cap was: Crawler,
    28 Oct) before structure-only is called for. A first cap already at the retry's cap is not
    retried."""

    trace = go("trace", caps=first) if first is not None else go("trace")
    traces = [trace]
    again = _retry_caps(first)
    if not (retry and _memory_cap(trace)) or (first is not None and first.rss_mb >= again.rss_mb):
        return traces
    return [*traces, go("trace", caps=again)]


def _retry_caps(first: Caps | None) -> Caps:
    """A ``memory_cap`` retry's caps: the whole declaration on a one-worker runner (it runs alone),
    else one worker slot (``WORKER_GIB``), so the retry runs beside the other workers instead of
    stopping them. A cap that broke at or above a slot is not retried there: the row goes
    structure-only, and its full trace is the big runner's (``placement_for``)."""

    from menagerie import host_budget
    from menagerie.pipeline import runner

    whole_mb = int(host_budget.declared_memory_gb() * 1024)
    timeout = first.timeout_s if first is not None else runner.TIMEOUT_S["trace"]
    if int(os.environ.get("RUNNER_WORKERS") or 1) <= 1:
        return Caps(rss_mb=whole_mb, timeout_s=timeout, threads=host_budget.declared_cpus())
    slot = min(whole_mb, int(WORKER_GIB * 1024))
    return Caps(rss_mb=slot, timeout_s=timeout, threads=runner.default_caps("trace").threads)


def _train_trace(go: Any, first: Caps | None, *, retry: bool) -> list[RunResult]:
    """The train-mode trace under the eval trace's planned caps; like the eval trace, a
    ``memory_cap`` under sized caps is retried once under ``_retry_caps``."""

    train = go("trace", model_mode="train", caps=first)
    again = _retry_caps(first)
    if not (retry and _memory_cap(train)) or (first is not None and first.rss_mb >= again.rss_mb):
        return [train]
    return [train, go("trace", model_mode="train", caps=again)]


def _saved_traces(sid: str, roots: Sequence[Path]) -> dict[str, tuple[Path, RunResult]]:
    """The newest saved trace per model mode among a model's earlier trace attempts."""

    found = sorted((p for root in roots for p in Path(root).glob(f"{sid}-trace-*")
                    if (p / "result.json").is_file()), key=lambda p: p.name.split("-")[-2])
    latest: dict[str, tuple[Path, RunResult]] = {}
    for attempt in found:
        result = read_json(RunResult, attempt / "result.json")
        if any(f.kind == "tlspec" for f in result.files):
            latest[model_mode_of(result)] = (attempt, result)
    return latest


def rerender(ids: Sequence[str], roots: Sequence[Path], *, caps: Caps, data: DataLayout,
             store: StoreLayout, attempts: Path) -> list[RenderRow]:
    """Render again every model's newest passing trace found under ``roots`` (earlier attempt
    directories), keyed as ``render_request`` keys it; append the render rows to the ledger."""

    from menagerie.pipeline import ledger

    rows: list[RenderRow] = []
    for sid in ids:
        record = read_json(ModelRecord, data.model(sid))
        outcome = Outcome(sid)
        for attempt, result in _saved_traces(sid, roots).values():
            request = render_request(record, result, attempt, caps=caps, attempts=attempts,
                                     adapter=adapter_src(record, data))
            if request is not None:
                _run(request, outcome, store=store, data=data, attempts=attempts)
        made = [r for r in outcome.rows if isinstance(r, RenderRow)]
        if made:
            ledger.append(made, data=data)
        rows += made
        print(json.dumps({"id": sid, "renders": [(r.trace[:12], r.outcome) for r in made]}),
              flush=True)
    return rows


def _forward_cells(record: ModelRecord, forward: RunResult | None) -> dict[str, Any]:
    if forward is None:
        why = f"giveup:{record.give_up.cls}" if record.give_up else "skipped: cluster"
        return {"forward": why, "forward_class": None, "divergence": None,
                "input_boundary": None}
    return {"forward": forward.outcome, "forward_class": forward.cls,
            "divergence": forward.modes.divergence if forward.modes else None,
            "input_boundary": forward.input_boundary}


def _trace_cells(traces: list[RunResult], rows: list[LedgerRow]) -> dict[str, Any]:
    passed = [r for r in rows if r.outcome == "pass"]
    return {"trace": ";".join(f"{r.mode}:{r.outcome}" for r in traces),
            "trace_class": ";".join(r.cls or "" for r in traces),
            "n_ops": rows[0].n_ops if rows else None,
            "runnable": ";".join(r.runnable or "" for r in rows),
            "hashes": bool(passed) and all(r.hashes is not None for r in passed),
            "msg": next((r.msg for r in traces if r.msg), None)}


def _render_cells(renders: list[RenderRow]) -> dict[str, Any]:
    svgs = {a.variant for r in renders for a in r.renders if a.variant and a.kind == "svg"}
    return {"render": ";".join(r.outcome for r in renders), "variants": ";".join(sorted(svgs)),
            "artifacts": sum(len(r.renders) for r in renders),
            "render_s": round(sum(r.secs or 0 for r in renders), 1)}


def table_row(record: ModelRecord, outcome: Outcome) -> dict[str, Any]:
    """One line of the rung table for a model."""

    traces = [r for r in outcome.results if r.mode.startswith("trace") or r.mode == "gpu"]
    trace_rows = [r for r in outcome.rows if isinstance(r, LedgerRow) and r.stage == "trace"]
    renders = [r for r in outcome.rows if isinstance(r, RenderRow)]
    return {"stable_id": record.id, "name": record.name,
            **_forward_cells(record, outcome.last("forward")),
            **_trace_cells(traces, trace_rows), **_render_cells(renders)}


def run(ids: Sequence[str], *, caps: Caps | None, data: DataLayout, store: StoreLayout,
        attempts: Path, out: Path, spool: Path | None = None) -> list[dict[str, Any]]:
    """Run every id end to end, append the rows to the ledger, write the table as JSON lines."""

    from menagerie.pipeline import ledger

    table = []
    out.parent.mkdir(parents=True, exist_ok=True)
    for sid in dict.fromkeys(ids):
        record = read_json(ModelRecord, data.model(sid))
        outcome = run_model(record, caps=caps, store=store, data=data, attempts=attempts,
                            spool=spool)
        if outcome.rows:
            ledger.append(outcome.rows, data=data)
        row = table_row(record, outcome)
        table.append(row)
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        print(json.dumps(row, default=str), flush=True)
    return table


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m menagerie.pipeline.flow OUT.jsonl ID... [--spool S]``; caps from the declaration,
    or the runner's per-mode defaults through a spool."""

    import argparse

    from menagerie import host_budget
    from menagerie.paths import work_dir

    parser = argparse.ArgumentParser(prog="python -m menagerie.pipeline.flow")
    parser.add_argument("out", type=Path)
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--rerender-from", type=Path, action="append", default=[],
                        help="render the traces in these attempt directories again; no runs")
    parser.add_argument("--spool", type=Path,
                        help="submit every request through this runner spool and wait")
    args = parser.parse_args(argv)
    cpus, memory = host_budget.declared_cpus(), host_budget.declared_memory_gb()
    caps = Caps(rss_mb=int(memory * 1024), timeout_s=1800, threads=cpus)
    places = {"data": DataLayout.from_env(), "store": StoreLayout.from_env(),
              "attempts": work_dir("attempts")}
    if args.rerender_from:
        rerender(args.ids, args.rerender_from, caps=caps, **places)
    elif args.spool:
        run(args.ids, caps=None, out=args.out, spool=args.spool, **places)
    else:
        run(args.ids, caps=caps, out=args.out, **places)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
