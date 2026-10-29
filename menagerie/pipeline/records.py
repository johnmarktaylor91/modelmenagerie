"""Every file the pipeline writes, as frozen dataclasses with read, write and schema checks.

One record type per file type: sweep lines and query rows, roster rows, family packets,
model records with their recipe, ledger rows (forward, trace and review), render rows, the
June legacy ledger shard, describe records, citation lines, store index lines, census rows,
dossiers and resolver hints, plus the runner's ``RunRequest`` and ``RunResult``. Field lists
are complete: reading a file with an unknown or missing key fails, and every value is checked
against its annotation when a record is built. Top-level records carry a ``schema`` key;
``None`` optionals are omitted on write, so files stay small. File I/O and paths are in
``menagerie.pipeline``.
"""

from __future__ import annotations

import functools
import json
import re
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from dataclasses import MISSING, dataclass, fields, is_dataclass
from dataclasses import field as dc_field
from typing import Any, ClassVar, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

SHA256 = r"[0-9a-f]{64}"
STABLE_ID = r"m\d+"
FAMILY_ID = r"f-[a-z0-9][a-z0-9._-]*"

# Section 4.5 of the design, closed. A class is `<head>` or `<head>:<detail>`.
# `construct_error` and `forward_error` are the runner's classes for a recipe that raises.
FAILURE_CLASSES = frozenset(
    {
        "not_found", "ambiguous", "insufficient_description", "inaccessible",
        "not_architecture", "duplicate_of", "research_gap", "missing_dependency",
        "compiled_ext", "needs_gpu", "oversized", "needs_weights", "budget",
        "not_traceable_framework", "fabrication", "torchlens_gap", "memory_cap", "timeout",
        "capture_error", "construct_error", "forward_error", "runnable", "render", "infra",
        "publish_error",
        # Rung 2 (BUILD-PLAN r2 sections 4 and 5): catalog stand-ins and rows needing an adapter
        # are typed, not traced; a native environment that will not resolve; a failed fidelity
        # review after its repair round.
        "standin", "needs_adapter", "env_unresolved", "unpublishable",
        # A torch trace whose replay TorchLens could not confirm (its validation state is
        # `unverified`, e.g. the retrace check stopped it by design): not a replay failure.
        "torchlens_unverified",
        # Rung 3 (BUILD-PLAN r3 section 1, item 2): a describe record that failed its grounding
        # check after one return, or a family with no saved text to describe; a family with no
        # paper (`no_paper:<why>`); June code whose constructor is neither library defaults nor a
        # named published config; a row waiting on an environment (`needs_env:<name>`).
        "describe", "no_paper", "june_code", "needs_env",
        # A forward whose output holds NaN or inf (`degenerate_forward:nonfinite`; JMT via Lead,
        # 2026-10-28): never validated, so the worker types it before any capture.
        "degenerate_forward",
    }
)
CLASS_PATTERN = r"(" + "|".join(sorted(FAILURE_CLASSES)) + r")(:\S+)?"
# Heads whose detail is a closed set: `standin:exec` (an exec-string recipe), `standin:compact`
# (the base-pytorch-compact zoo), `standin:kwargs` (kwargs that differ from the library's
# defaults or a named published config), `standin:dense` (a 2b recipe that replaced a library on
# the model's path with a hand-written dense equivalent: its trace never publishes under the real
# name); `oversized:beyond_allocation` (refused by the largest allocation); `unpublishable:fidelity`
# (fabrication after the repair round); `standin:no_published_config` (a `standin:kwargs` row with
# neither library defaults nor a named published config to rebuild at); `describe:ungrounded`,
# `describe:no_text`; `june_code:flagged`; `degenerate_forward:nonfinite`.
CLASS_DETAILS = {
    "standin": frozenset({"exec", "compact", "kwargs", "dense", "no_published_config"}),
    "unpublishable": frozenset({"fidelity"}),
    "describe": frozenset({"ungrounded", "no_text"}),
    "june_code": frozenset({"flagged"}),
    "degenerate_forward": frozenset({"nonfinite"}),
}
# Heads whose detail is open but required: `no_paper:<why>`, `needs_env:<environment>`.
NEEDS_DETAIL = frozenset({"no_paper", "needs_env"})

Backend = Literal["torch", "tf", "jax", "paddle", "mlx", "tinygrad"]
Divergence = Literal["none", "statistical", "structural"]
InputBoundary = Literal["real", "wrapper", "none"]
ModelMode = Literal["eval", "train"]
# `gpu`: one attempt on a CUDA device (forward, then trace), only inside the GPU window.
RunMode = Literal["forward", "trace", "trace-structure-only", "render", "gpu"]
# Where a recipe runs (fix 10's placement rule): a worker slot, the single-worker slot (the whole
# share), an AXON batch kind, the worker's GPU window, or a structure-only capture alone.
Placement = Literal["worker", "single-worker", "axon-cpu", "axon-gpu", "axon-reviewed",
                    "gpu-window", "structure-only"]
CLUSTER_PLACEMENTS = frozenset({"axon-cpu", "axon-gpu", "axon-reviewed"})
# Placements whose rows skip the worker's forward and full trace: here they get a structure-only
# capture, and their full attempt runs elsewhere.
OFF_WORKER = CLUSTER_PLACEMENTS | {"structure-only"}
RenderKind = Literal["svg", "pdf", "thumb", "tiles", "html", "netron", "model_explorer"]
# A diagram variant key: `<vis_mode>-<collapse>`, with `-train` for the train-mode set.
VARIANT = r"(unrolled|rolled)-(none|auto|max)(-train)?"
TRACE_LEVEL_KINDS = frozenset({"html", "netron", "model_explorer"})


class RecordError(ValueError):
    """A record or file does not match its schema."""


def check_class(cls: str | None, where: str) -> None:
    """Refuse a failure class whose head has a closed detail set and a detail outside it."""

    if cls is None:
        return
    head, _, detail = cls.partition(":")
    allowed = CLASS_DETAILS.get(head)
    if allowed is not None and detail not in allowed:
        raise RecordError(f"{where}: {cls!r} needs one of {sorted(f'{head}:{d}' for d in allowed)}")
    if head in NEEDS_DETAIL and not detail:
        raise RecordError(f"{where}: {cls!r} needs a detail ({head}:<detail>)")


# Environment specs: one folder per environment under the repository's `envs/`.
ENV_SPECS = Path(__file__).resolve().parents[2] / "envs"


# Closed describe vocabularies: one JSON file per field (`{"field", "source", "values"}`).
VOCAB = Path(__file__).resolve().parent / "assets" / "vocab"


@functools.cache
def vocab(name: str) -> frozenset[str]:
    """The closed vocabulary for a describe field (``assets/vocab/<name>.json``)."""

    path = VOCAB / f"{name}.json"
    if not path.is_file():
        raise RecordError(f"no vocabulary {name!r} at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = raw.get("values")
    if raw.get("field") != name or not isinstance(values, list) or not values \
            or not all(isinstance(v, str) and v for v in values) or len(set(values)) != len(values):
        raise RecordError(f"{path}: needs field {name!r} and a non-empty list of distinct strings")
    return frozenset(values)


@functools.cache
def known_envs() -> frozenset[str]:
    """The environment names a recipe may use: the folders under ``envs/`` with a project file
    (``envs/axon`` holds the cluster's additions, not an environment)."""

    if not ENV_SPECS.is_dir():
        raise RecordError(f"no environment specs at {ENV_SPECS}")
    return frozenset(p.name for p in ENV_SPECS.iterdir()
                     if ((p / "pyproject.toml").is_file() or
                         (p / "pixi.toml").is_file() and (p / "pixi.lock").is_file())
                     and not p.name.startswith((".", "_")))


def spec(default: Any = MISSING, *, key: str | None = None, pattern: str | None = None,
         factory: Callable[[], Any] | None = None) -> Any:
    """Declare a field with an optional JSON key (for Python keywords) and string pattern."""

    meta = {k: v for k, v in (("key", key), ("pattern", pattern)) if v is not None}
    if factory is not None:
        return dc_field(default_factory=factory, metadata=meta)
    return dc_field(default=default, metadata=meta)


@dataclass(frozen=True)
class Record:
    """Base class: every field is type-checked on construction, then ``check()`` runs."""

    SCHEMA: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        hints = _hints(type(self))
        for f in fields(self):
            where = f"{type(self).__name__}.{f.name}"
            _check(hints[f.name], getattr(self, f.name), where, f.metadata.get("pattern"))
        self.check()

    def check(self) -> None:
        """Cross-field rules; overridden where a record has them."""


R = TypeVar("R", bound=Record)


@functools.cache
def _hints(cls: type) -> dict[str, Any]:
    return get_type_hints(cls)


def _shape(tp: Any) -> tuple[Any, tuple[Any, ...]]:
    """``get_origin`` and ``get_args`` of a field type, cached: every row of every record parse
    asks them for the same few types, and uncached they were most of a ledger read's time."""
    try:
        return _shape_cached(tp)
    except TypeError:  # an unhashable annotation: ask typing directly
        return get_origin(tp), get_args(tp)


@functools.cache
def _shape_cached(tp: Any) -> tuple[Any, tuple[Any, ...]]:
    return get_origin(tp), get_args(tp)


def _is_union(tp: Any) -> bool:
    return _shape(tp)[0] in (Union, types.UnionType)


def _fail(where: str, value: Any, expected: Any) -> RecordError:
    return RecordError(f"{where}: {value!r} is not {expected}")


def _check_scalar(tp: Any, value: Any, where: str, pattern: str | None) -> None:
    if tp is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif tp is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    else:
        ok = isinstance(value, tp)
    if not ok:
        raise _fail(where, value, getattr(tp, "__name__", tp))
    if pattern and isinstance(value, str) and not re.fullmatch(pattern, value):
        raise _fail(where, value, f"a match for {pattern}")


def _check(tp: Any, value: Any, where: str, pattern: str | None = None) -> None:
    origin, args = _shape(tp)
    if tp is Any:
        return
    if origin is Literal:
        if value not in args:
            raise _fail(where, value, f"one of {args}")
    elif _is_union(tp):
        if value is None and type(None) in args:
            return
        for arm in (a for a in args if a is not type(None)):
            try:
                _check(arm, value, where, pattern)
                return
            except RecordError:
                continue
        raise _fail(where, value, tp)
    elif origin is tuple:
        if not isinstance(value, tuple):
            raise _fail(where, value, "a tuple")
        for i, item in enumerate(value):
            _check(args[0], item, f"{where}[{i}]", pattern)
    elif origin is dict:
        if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
            raise _fail(where, value, "an object with string keys")
        for k, item in value.items():
            _check(args[1], item, f"{where}.{k}", pattern)
    else:
        _check_scalar(tp, value, where, pattern)


def _decode(tp: Any, raw: Any) -> Any:
    origin, args = _shape(tp)
    if isinstance(tp, type) and is_dataclass(tp) and isinstance(raw, Mapping):
        return from_dict(tp, raw)
    if _is_union(tp):
        if raw is None:
            return None
        arms = [a for a in args if a is not type(None)]
        records = [a for a in arms if isinstance(a, type) and is_dataclass(a)]
        if records and isinstance(raw, Mapping):
            chosen = [a for a in records if a.SCHEMA == raw.get("schema")] or records[:1]
            return from_dict(chosen[0], raw)
        return _decode(arms[0], raw) if len(arms) == 1 else raw
    if origin is tuple and isinstance(raw, list):
        return tuple(_decode(args[0], item) for item in raw)
    if origin is dict and isinstance(raw, Mapping):
        return {k: _decode(args[1], v) for k, v in raw.items()}
    return raw


def from_dict(cls: type[R], raw: Mapping[str, Any]) -> R:
    """Build a record from parsed JSON, refusing unknown, missing or mistyped fields."""

    if not isinstance(raw, Mapping):
        raise RecordError(f"{cls.__name__}: expected an object, got {type(raw).__name__}")
    data = dict(raw)
    if cls.SCHEMA is not None and data.pop("schema", None) != cls.SCHEMA:
        raise RecordError(f"{cls.__name__}: schema is {raw.get('schema')!r}, want {cls.SCHEMA!r}")
    hints, kwargs = _hints(cls), {}
    for f in fields(cls):
        key = f.metadata.get("key", f.name)
        if key in data:
            kwargs[f.name] = _decode(hints[f.name], data.pop(key))
        elif f.default is MISSING and f.default_factory is MISSING:
            raise RecordError(f"{cls.__name__}: missing field {key!r}")
    if data:
        raise RecordError(f"{cls.__name__}: unknown fields {sorted(data)}")
    return cls(**kwargs)


def _encode(value: Any) -> Any:
    if isinstance(value, Record):
        return to_dict(value)
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    return value


def to_dict(record: Record) -> dict[str, Any]:
    """Return the JSON object for a record: ``schema`` first, ``None`` optionals omitted."""

    out: dict[str, Any] = {"schema": record.SCHEMA} if record.SCHEMA else {}
    for f in fields(record):
        value = getattr(record, f.name)
        if value is None and f.default is None:
            continue
        out[f.metadata.get("key", f.name)] = _encode(value)
    return out


def dumps(record: Record) -> str:
    """Serialize one record as a single compact JSON line (no trailing newline)."""

    return json.dumps(to_dict(record), ensure_ascii=False, separators=(",", ":"))


# --- Stage 1: sweeps and roster ------------------------------------------------------


@dataclass(frozen=True)
class AxisState(Record):
    state: Literal["drained", "frozen", "open"]
    frontier: str | None = None


@dataclass(frozen=True)
class SweepLine(Record):
    """One line of ``sweeps/index.jsonl``."""

    SCHEMA = "sweep/1"
    sweep: str
    kind: Literal["agent", "registry-diff", "arxiv-delta", "citation-expansion", "prior-art-diff"]
    started: str
    method: str
    axes: dict[str, AxisState]
    ended: str | None = None
    roster_sha_before: str | None = spec(None, pattern=SHA256)
    roster_sha_after: str | None = spec(None, pattern=SHA256)
    admitted: int = 0
    rejected: int = 0
    note: str = ""


@dataclass(frozen=True)
class Rejection(Record):
    name: str
    reason: str = spec(
        pattern=r"duplicate|not_architecture|component|method|insufficient_description"
        r"|variant_of:\S+"
    )


@dataclass(frozen=True)
class SweepQuery(Record):
    """One per-query row of ``sweeps/<sweep>.jsonl``."""

    SCHEMA = "sweep-query/1"
    sweep: str
    axis: str
    cell: str
    tool_role: str
    query: str
    lang: str
    ran_at: str
    seen: int
    admitted: tuple[str, ...] = spec((), pattern=STABLE_ID)
    rejected: tuple[Rejection, ...] = ()


@dataclass(frozen=True)
class RosterRow(Record):
    """One line of ``crawl_roster.jsonl``: the existing fields plus provenance and hints."""

    SCHEMA = "roster/1"
    name: str
    zoo: str
    variant: str = ""
    family: str | None = None
    source_url: str | None = None
    domain: str | None = None
    era: str | None = None
    triage_class: str | None = None
    aliases: tuple[str, ...] = ()
    stable_id: str | None = spec(None, pattern=STABLE_ID)
    found_by: tuple[str, ...] = ()
    hint_urls: tuple[str, ...] = ()
    hint_framework: str | None = None
    hint_paper: str | None = None
    state: str = spec("active", pattern=rf"active|not_architecture|duplicate_of:{STABLE_ID}")


# --- Stage 2a: family packet ---------------------------------------------------------

AvailabilitySlot = Literal["found", "not_found", "inaccessible", "ambiguous", "insufficient"]


@dataclass(frozen=True)
class Member(Record):
    id: str = spec(pattern=STABLE_ID)
    delta: str = ""


@dataclass(frozen=True)
class Identity(Record):
    canonical: str
    aliases: tuple[str, ...] = ()
    architecture: bool = True


@dataclass(frozen=True)
class Source(Record):
    """One saved source; which optional fields apply depends on ``role``."""

    sid: str
    role: Literal["paper", "code", "library", "license", "bib", "page"]
    url: str | None = None
    blob: str | None = spec(None, pattern=SHA256)
    text: str | None = spec(None, pattern=SHA256)
    official: bool | None = None
    framework: str | None = None
    rev: str | None = None
    weights: Literal["found", "not_found", "unknown"] | None = None
    dist: str | None = None
    version: str | None = None
    module: str | None = None
    symbol: str | None = None
    of: str | None = None
    spdx: str | None = None
    api: Literal["openalex", "s2", "crossref", "bibtex"] | None = None
    fetched: str | None = None

    def check(self) -> None:
        required = {
            "paper": ("url",), "code": ("url",), "library": ("dist", "version", "symbol"),
            "license": ("of",), "bib": ("api",), "page": ("url",),
        }[self.role]
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise RecordError(f"Source {self.sid} ({self.role}) needs {missing}")


@dataclass(frozen=True)
class Citation(Record):
    title: str
    authors: tuple[str, ...] = ()
    first_public: str | None = None
    basis: str | None = None
    published: str | None = None
    venue: str | None = None
    arxiv: str | None = None
    doi: str | None = None
    openreview: str | None = None
    openalex: str | None = None
    s2: str | None = None
    from_: str | None = spec(None, key="from")
    bib: tuple[str, ...] = ()


@dataclass(frozen=True)
class Availability(Record):
    paper: AvailabilitySlot
    code: AvailabilitySlot
    license: AvailabilitySlot
    description: Literal["sufficient", "insufficient", "not_found", "inaccessible", "ambiguous"]


@dataclass(frozen=True)
class Claim(Record):
    field: str
    sid: str
    quote: str
    loc: str = ""


@dataclass(frozen=True)
class Search(Record):
    queries: tuple[str, ...] = ()
    places: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Amendment(Record):
    at: str
    why: str
    evidence: tuple[str, ...]
    changed: tuple[str, ...]
    by: str


@dataclass(frozen=True)
class Packet(Record):
    """``research/<xx>/<family>.json``: what 2a found for one family, every claim quoted."""

    SCHEMA = "packet/1"
    family: str = spec(pattern=FAMILY_ID)
    written: str = spec()
    by: str = spec()
    members: tuple[Member, ...] = spec()
    identity: Identity = spec()
    sources: tuple[Source, ...] = spec()
    availability: Availability = spec()
    citation: Citation | None = None
    code_order: tuple[str, ...] = ()
    notes_for_implementer: str = ""
    claims: tuple[Claim, ...] = ()
    search: Search = Search()
    amendments: tuple[Amendment, ...] = ()

    def check(self) -> None:
        roles = {s.sid: s.role for s in self.sources}
        if len(roles) != len(self.sources):
            raise RecordError(f"Packet {self.family}: duplicate source ids")
        refs = [c.sid for c in self.claims] + [s.of for s in self.sources if s.of]
        if self.citation:
            refs += list(self.citation.bib) + ([self.citation.from_] if self.citation.from_ else [])
        for amendment in self.amendments:
            refs += list(amendment.evidence)
        dangling = sorted(set(refs) - set(roles))
        if dangling:
            raise RecordError(f"Packet {self.family}: unknown source ids {dangling}")
        if any(roles[sid] not in ("code", "library") for sid in self.code_order if sid in roles):
            raise RecordError(f"Packet {self.family}: code_order names a non-code source")
        if set(self.code_order) - set(roles):
            raise RecordError(f"Packet {self.family}: code_order names unknown sources")
        slots = (self.availability.paper, self.availability.code, self.availability.license)
        unfound = any(s != "found" for s in slots) or self.availability.description != "sufficient"
        if unfound and not self.search.places:
            raise RecordError(f"Packet {self.family}: a slot is not found, yet no places searched")


# --- Stage 2b: model record and recipe -----------------------------------------------


@dataclass(frozen=True)
class HFRef(Record):
    id: str
    card: str | None = spec(None, pattern=SHA256)
    config: str | None = spec(None, pattern=SHA256)


@dataclass(frozen=True)
class RecipeInput(Record):
    """The input contract: preset, shape, REQ 14 kind, canonical asset id and a one-line note."""

    preset: str
    kind: str = spec(pattern=r"standard-[a-z0-9-]+|random-fallback")
    shape: tuple[int, ...] | None = None
    asset: str | None = None
    note: str = ""


@dataclass(frozen=True)
class Modes(Record):
    eval: str = spec(pattern=r"pass|fail:\S+")
    train: str | None = spec(None, pattern=r"pass|fail:\S+")
    divergence: Divergence | None = None


@dataclass(frozen=True)
class Recipe(Record):
    rev: str = spec(pattern=r"[0-9a-f]{12}")
    kind: Literal["library", "adapter", "classic"] = spec()
    env: str = spec()
    backend: Backend = spec()
    input: RecipeInput = spec()
    call: str | None = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = spec(factory=dict)
    adapter: str | None = None
    config: dict[str, Any] = spec(factory=dict)
    # Packages installed over the environment for this recipe (pip requirement strings).
    extras: tuple[str, ...] = ()
    # Store sha of the code snapshot (a saved repository tree) the adapter imports.
    vendor: str | None = spec(None, pattern=SHA256)
    # Store shas of further saved trees the adapter materializes (a compiled library's source, such
    # as mamba_ssm or flash-attention), staged with ``vendor`` wherever the store is copied. ``None``
    # rather than empty, so records and requests without it read the same to older readers.
    code: tuple[str, ...] | None = spec(None, pattern=SHA256)
    modes: Modes | None = None
    memory_gib: float | None = None
    placement: Placement | None = None
    sources: tuple[str, ...] = ()

    def check(self) -> None:
        if self.env not in known_envs():
            raise RecordError(f"Recipe: env {self.env!r} is not one of {sorted(known_envs())}")
        if self.kind == "library" and not self.call:
            raise RecordError("Recipe: a library recipe needs `call`")
        if self.kind in ("adapter", "classic") and not self.adapter:
            raise RecordError(f"Recipe: a {self.kind} recipe needs `adapter`")


@dataclass(frozen=True)
class GiveUp(Record):
    cls: str = spec(key="class", pattern=CLASS_PATTERN)
    reason: str = spec()
    at: str = spec()
    command: str | None = None
    missing: str | None = None
    searched: tuple[str, ...] = ()
    # Who typed it: a 2a or 2b agent lane (`implement.giveup`) or a scripted route. Absent on
    # give-ups from before 2026-10-09 (`drive.agent_tried` reads the ledger for those).
    by: Literal["agent", "script"] | None = None

    def check(self) -> None:
        check_class(self.cls, "GiveUp")


@dataclass(frozen=True)
class ModelRecord(Record):
    """``models/<xx>/<stable_id>.json``: identity from 2a, recipe or give-up from 2b."""

    SCHEMA = "model/1"
    id: str = spec(pattern=STABLE_ID)
    name: str = spec()
    zoo: str = spec()
    family: str = spec(pattern=FAMILY_ID)
    source_case: Literal["library", "vendor", "port", "native", "reimplement", "classic"] = spec()
    variant: str = ""
    delta: dict[str, Any] = spec(factory=dict)
    hf: HFRef | None = None
    recipe: Recipe | None = None
    give_up: GiveUp | None = None


# --- Stage 3 and review: ledger rows -------------------------------------------------


@dataclass(frozen=True)
class Hashes(Record):
    skeleton: str = spec(pattern=SHA256)
    schema: int = spec()
    architecture: str | None = spec(None, pattern=SHA256)
    model: str | None = spec(None, pattern=SHA256)


@dataclass(frozen=True)
class Offense(Record):
    """One value escape TorchLens refused a structure-only capture for (an entry of its
    ``ValueDependentBranchError.fields["offenses"]``); the message alone loses the line."""

    file: str
    line: int | None = None
    consumer_kind: str | None = None
    tensor_label: str | None = None
    escape_method: str | None = None
    substrate: str | None = None


@dataclass(frozen=True)
class LedgerRow(Record):
    """One forward, trace or review attempt in ``ledger/<yyyy-mm>.jsonl``."""

    SCHEMA = "ledger/1"
    id: str = spec(pattern=STABLE_ID)
    stage: Literal["forward", "trace", "review"] = spec()
    at: str = spec()
    host: str = spec()
    env: str = spec()
    recipe_rev: str = spec(pattern=r"[0-9a-f]{12}")
    outcome: Literal["pass", "fail", "infra", "giveup"] = spec()
    mode: ModelMode | None = None
    capture: Literal["full", "structure_only"] | None = None
    backend: Backend | None = None
    tl: str | None = None
    fw: str | None = None
    cls: str | None = spec(None, key="class", pattern=CLASS_PATTERN)
    n_ops: int | None = None
    hashes: Hashes | None = None
    graph_hash: str | None = spec(None, pattern=SHA256)
    shape_digest: str | None = None
    params: int | None = None
    params_source: Literal["counted", "geometry"] | None = None
    peak_rss_mb: int | None = None
    forward_secs: float | None = None
    capture_secs: float | None = None
    forward_peak_rss_mb: int | None = None
    input_boundary: InputBoundary | None = None
    validation: str | None = None
    validation_state: (
        Literal["passed", "failed", "unverified", "unavailable", "structure_only"] | None
    ) = None
    # How TorchLens attributed modules: torch traces `torch_module`; preview backends
    # `object_module`, `pytree_module` or `function_root` (published labelled).
    module_mode: (
        Literal["torch_module", "object_module", "pytree_module", "function_root"] | None
    ) = None
    runnable: str | None = spec(None, pattern=r"ready|refused:\S+|n/a")
    # Whether a full trace's validation and save captures followed one plain warm-up forward
    # (none on rows from before 2026-10-05 and on structure-only rows, whose capture, like the hash
    # family's, is of a never-run build); `false` means the warm-up raised, and `msg` says why.
    warmed: bool | None = None
    trace: str | None = spec(None, pattern=SHA256)
    summary: str | None = spec(None, pattern=SHA256)
    evidence: str | None = spec(None, pattern=SHA256)
    msg: str | None = None
    offenses: tuple[Offense, ...] | None = None
    verdict: Literal["faithful", "minor", "fabrication", "cannot_judge"] | None = None
    findings: str | None = spec(None, pattern=SHA256)
    # The reviewer role of a review row (calibration is per role and prompt hash).
    reviewer: str | None = None

    def check(self) -> None:
        check_class(self.cls, f"LedgerRow {self.id}")
        if self.reviewer is not None and self.stage != "review":
            raise RecordError(f"LedgerRow {self.id}: only a review row names a reviewer")
        if self.outcome in ("fail", "giveup") and not self.cls:
            raise RecordError(f"LedgerRow {self.id}: a {self.outcome} row needs a class")
        if self.outcome == "pass" and self.cls:
            raise RecordError(f"LedgerRow {self.id}: a pass row has no class")
        if self.stage == "review" and self.verdict is None:
            raise RecordError(f"LedgerRow {self.id}: a review row needs a verdict")
        if self.capture == "structure_only" and (
            self.validation_state != "structure_only"
            or self.params_source not in (None, "geometry")
        ):
            raise RecordError(f"LedgerRow {self.id}: structure-only rows are never validated")


@dataclass(frozen=True)
class RenderArtifact(Record):
    kind: RenderKind
    sha256: str = spec(pattern=SHA256)
    bytes: int = spec()
    variant: str | None = spec(None, pattern=VARIANT)


@dataclass(frozen=True)
class RenderSkip(Record):
    kind: RenderKind
    why: str = spec(pattern=r"render:\S+")
    variant: str | None = spec(None, pattern=VARIANT)


@dataclass(frozen=True)
class RenderRow(Record):
    """One render attempt, keyed on the trace sha and ``renderer_rev``; never touches a verdict."""

    SCHEMA = "render/1"
    id: str = spec(pattern=STABLE_ID)
    at: str = spec()
    trace: str = spec(pattern=SHA256)
    renderer_rev: str = spec()
    tl: str = spec()
    outcome: Literal["pass", "partial", "fail", "infra"] = spec()
    stage: Literal["render"] = "render"
    renders: tuple[RenderArtifact, ...] = ()
    skipped: tuple[RenderSkip, ...] = ()
    secs: float | None = None
    evidence: str | None = spec(None, pattern=SHA256)

    def check(self) -> None:
        for item in (*self.renders, *self.skipped):
            if (item.variant is None) != (item.kind in TRACE_LEVEL_KINDS):
                raise RecordError(f"RenderRow {self.id}: {item.kind} variant must be "
                                  f"{'absent' if item.kind in TRACE_LEVEL_KINDS else 'set'}")


@dataclass(frozen=True)
class LegacyRow(Record):
    """One June ``routing_manifest.tsv`` row, loaded as the ``ledger/2026-06.jsonl`` shard."""

    SCHEMA = "legacy/1"
    id: str = spec(pattern=STABLE_ID)
    name: str = spec()
    model_id: int = spec()
    recipe_rev: str = spec(pattern=SHA256)
    status: str = spec()
    metadata_ok: bool = spec()
    scope: str = spec()
    elapsed: float = spec()
    cluster: str = spec()
    error: str = ""
    n_ops: int | None = None
    graph_hash: str | None = spec(None, pattern=SHA256)

    @classmethod
    def from_manifest(cls, row: Mapping[str, str]) -> LegacyRow:
        """Convert one ``csv.DictReader`` row of ``routing_manifest.tsv``."""

        if row["validate_metadata_ok"] not in ("true", "false"):
            raise RecordError(f"LegacyRow: validate_metadata_ok is {row['validate_metadata_ok']!r}")
        return cls(
            id=row["stable_id"], name=row["name"], model_id=int(row["model_id"]),
            recipe_rev=row["recipe_revision_sha256"], status=row["status"],
            n_ops=int(row["n_ops"]) if row["n_ops"] else None,
            metadata_ok=row["validate_metadata_ok"] == "true",
            scope=row["scope"], elapsed=float(row["elapsed"]), cluster=row["dependency_cluster"],
            error=row["error"] or "", graph_hash=row["graph_shape_hash"] or None,
        )


LedgerLine = Union[LedgerRow, RenderRow, LegacyRow]
_LEDGER_TYPES: dict[str, type[Record]] = {t.SCHEMA: t for t in (LedgerRow, RenderRow, LegacyRow)}


def parse_ledger_line(raw: Mapping[str, Any]) -> LedgerLine:
    """Parse one ledger line into the record type its ``schema`` names."""

    cls = _LEDGER_TYPES.get(raw.get("schema"))  # type: ignore[arg-type]
    if cls is None:
        raise RecordError(f"ledger line has unknown schema {raw.get('schema')!r}")
    return from_dict(cls, raw)  # type: ignore[return-value]


# --- Describe, citations, store ------------------------------------------------------


@dataclass(frozen=True)
class Lineage(Record):
    relation: Literal["parent", "improves_on", "inspired_by", "successor", "variant_of"]
    target: str
    sid: str
    quote: str
    target_id: str | None = None
    variant_dimension: str | None = None


@dataclass(frozen=True)
class MemberDescribe(Record):
    output_type: str | None = None
    modality_output: str | None = None
    delta_markdown: str = ""


@dataclass(frozen=True)
class Grounding(Record):
    sentence: str
    sid: str
    quote: str


@dataclass(frozen=True)
class DescribeRecord(Record):
    """``describe/<xx>/<family>.json``: taxonomy, prose and lineage derived from saved text."""

    SCHEMA = "describe/1"
    family: str = spec(pattern=FAMILY_ID)
    by: str = spec()
    domain: str | None = None
    field: str | None = None
    subfield: str | None = None
    task: str | None = None
    tasks: tuple[str, ...] = ()
    modality: str | None = None
    modalities: tuple[str, ...] = ()
    modality_output: str | None = None
    output_type: str | None = None
    paradigm: str | None = None
    arch_class: str | None = None
    arch_subclass: str | None = None
    is_generative: bool | None = None
    era: str | None = None
    tags: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    institutions: tuple[str, ...] = ()
    labs: tuple[str, ...] = ()
    lineage: tuple[Lineage, ...] = ()
    members: dict[str, MemberDescribe] = spec(factory=dict)
    tagline: str | None = None
    summary_markdown: str | None = None
    key_contribution: str | None = None
    grounding: tuple[Grounding, ...] = ()
    # Rung 3: the packet source ids the writer read, the writer prompt's hash prefix, and where the
    # grounding check stands (`returned` once, then `ungrounded` is typed `describe:ungrounded`).
    inputs: tuple[str, ...] = ()
    prompt: str | None = spec(None, pattern=r"[0-9a-f]{8}([0-9a-f]{56})?")
    grounding_state: Literal["unchecked", "grounded", "returned", "ungrounded"] = "unchecked"
    # A typed outcome (`describe:no_text`, `describe:ungrounded`) when the writer could not ground
    # a record; None on a grounded one (lane r3-describe).
    cls: str | None = spec(None, key="class", pattern=CLASS_PATTERN)

    def check(self) -> None:
        read = set(self.inputs)
        cited = {g.sid for g in self.grounding} | {item.sid for item in self.lineage}
        if cited - read:
            raise RecordError(f"DescribeRecord {self.family}: cites sources it did not read: "
                              f"{sorted(cited - read)}")


@dataclass(frozen=True)
class CitationLine(Record):
    """One line of ``cite/<yyyy-mm>.jsonl``."""

    SCHEMA = "cite/1"
    family: str = spec(pattern=FAMILY_ID)
    at: str = spec()
    s2: int | None = None
    openalex: int | None = None
    from_: Literal["s2", "openalex"] | None = spec(None, key="from")


@dataclass(frozen=True)
class StoreIndexLine(Record):
    """One line of ``store-index/<yyyy-mm>.jsonl``: a blob the store holds."""

    SCHEMA = "store-index/1"
    sha256: str = spec(pattern=SHA256)
    bytes: int = spec()
    mime: str = spec()
    fetched: str = spec()
    license_class: str = spec()
    url: str | None = None
    derived_from: str | None = spec(None, pattern=SHA256)
    # Where the blob's bytes are now (rung 3 pack and offload): a later line for the same sha
    # restates the first with these set. ``encoding`` ``tar.gz`` is a packed tree at
    # ``<sha>.tar.gz`` beside the raw path, ``gzip`` a gzipped file at ``<sha>.gz``; ``location``
    # ``box:<snapshot>`` means only that full-restore-verified box snapshot holds it. Absent means
    # raw and local, as put.
    encoding: Literal["raw", "tar.gz", "gzip"] | None = None
    location: str | None = spec(None, pattern=rf"local|box:{SHA256}")


@dataclass(frozen=True)
class CensusRow(Record):
    """One GitHub repository of the rung 3 census (BUILD-PLAN r3 section 3.2), keyed by c2 row.

    ``eligible`` holds when the repository is alive, not archived, uses PyTorch (requirements or
    imports) and is under 200 MB; otherwise ``why`` names the first test it failed. ``draw`` is
    set by ``census draw``: ``top`` for the 500 by score, ``control`` for the 50 in hash order.
    """

    SCHEMA = "census/1"
    id: str = spec(pattern=STABLE_ID)
    repo: str = spec(pattern=r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
    checked: str = spec()
    alive: bool = spec()
    family: str | None = spec(None, pattern=FAMILY_ID)
    domain: str | None = None
    archived: bool | None = None
    pytorch: Literal["requirements", "imports", "none"] | None = None
    size_kb: int | None = None
    stars: int | None = None
    pushed: str | None = None
    license: str | None = None
    paper_hint: str | None = None
    eligible: bool = False
    why: str | None = None
    score: float | None = None
    draw: Literal["top", "control"] | None = None
    # Set by ``census harvest`` (speed-up BUILD-PLAN section 1, item 2) from the repository's
    # README, citation file and license at the census checkpoint: ``paper_refs`` are the arXiv
    # ids and DOIs found, ``paper_title`` a BibTeX or CITATION.cff title, ``harvested`` when.
    # All absent on a row the harvest has not read; ``paper_refs`` ``[]`` means read, none found.
    readme_sha: str | None = spec(None, pattern=SHA256)
    citation_file: str | None = None
    paper_refs: tuple[str, ...] | None = None
    paper_title: str | None = None
    harvested: str | None = None

    def _check_harvest(self) -> None:
        harvest = (self.readme_sha, self.citation_file, self.paper_refs, self.paper_title)
        if self.harvested is None and any(v is not None for v in harvest):
            raise RecordError(f"CensusRow {self.id}: harvest fields need `harvested`")
        if self.harvested is not None and self.paper_refs is None:
            raise RecordError(f"CensusRow {self.id}: a harvested row lists its `paper_refs`")

    def check(self) -> None:
        self._check_harvest()
        passes = (self.alive and self.archived is False and self.pytorch in ("requirements", "imports")
                  and self.size_kb is not None and self.size_kb < 200 * 1024)
        if self.eligible != passes:
            raise RecordError(f"CensusRow {self.id}: `eligible` must equal the four tests")
        if self.eligible == (self.why is not None):
            raise RecordError(f"CensusRow {self.id}: `why` is set exactly when not eligible")
        if self.draw is not None and not self.eligible:
            raise RecordError(f"CensusRow {self.id}: only an eligible repository is drawn")


@dataclass(frozen=True)
class Dossier(Record):
    """A scripted research dossier for one census repository (``research dossier``, idea 4).

    A hypothesis the 2a lane confirms or replaces line by line, never a settled packet: the family
    proposal, the code source at the default branch's resolved ``commit``, the paper candidate with
    its agreeing-title verdict (``agrees`` only when its title passes the rule against the roster
    name and the repository name, else ``held`` for the agent), and the license.
    """

    SCHEMA = "dossier/1"
    id: str = spec(pattern=STABLE_ID)
    family: str = spec(pattern=FAMILY_ID)
    repo: str = spec(pattern=r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
    commit: str = spec(pattern=r"[0-9a-f]{40}")
    built: str = spec()
    branch: str | None = None
    paper: str | None = None
    paper_title: str | None = None
    paper_verdict: Literal["agrees", "held"] | None = None
    license: str | None = None
    readme_sha: str | None = spec(None, pattern=SHA256)
    by: Literal["census-harvest"] = "census-harvest"

    def check(self) -> None:
        if (self.paper is None) != (self.paper_verdict is None):
            raise RecordError(f"Dossier {self.id}: a paper candidate carries its title verdict")
        if self.paper_title is not None and self.paper is None:
            raise RecordError(f"Dossier {self.id}: a paper title needs its paper candidate")


@dataclass(frozen=True)
class ResolverHint(Record):
    """One candidate source the 2a router found for a name-only row, with its confirm verdict.

    A hint routes, it never settles: only a ``yes`` reaches the row's 2a brief (under "Hints
    (untrusted)"), and no verdict types a row. ``role`` is the confirm lane's role, ``evidence``
    the sids of the saved responses the verdict read, ``via`` the search that found it.
    """

    SCHEMA = "hint/1"
    id: str = spec(pattern=STABLE_ID)
    candidate: str = spec()
    kind: Literal["code", "paper", "weights", "page"] = spec()
    verdict: Literal["yes", "no", "unsure"] = spec()
    role: str = spec()
    at: str = spec()
    via: str | None = None
    evidence: tuple[str, ...] = ()


# Lane event verbs of the speed-up build (``drive.event``; section 1, item 2). Events, never
# give-up classes or ledger outcomes: ``2b:deferred`` hands a batched family back to 2b with its
# draft, ``cohort:stopped`` is a tripwire on a worklist cohort, ``hint:confirmed`` and
# ``hint:rejected`` record a router verdict. None of them settles or counts a row.
EVENT_VERBS = frozenset({"2b:deferred", "cohort:stopped", "hint:confirmed", "hint:rejected"})


# --- Runner request and result -------------------------------------------------------


@dataclass(frozen=True)
class Caps(Record):
    rss_mb: int
    timeout_s: int
    threads: int = 1


@dataclass(frozen=True)
class RunRequest(Record):
    """One spool request: a recipe, a mode and the caps for one sandboxed attempt."""

    SCHEMA = "run-request/1"
    request: str = spec()
    id: str = spec(pattern=STABLE_ID)
    mode: RunMode = spec()
    recipe: Recipe = spec()
    caps: Caps = spec()
    intervention_ready: bool = True
    model_mode: ModelMode = "eval"
    trace: str | None = spec(None, pattern=SHA256)
    extras: tuple[str, ...] = ()
    # The adapter file's text, so a draft runs without a commit and the runner needs no checkout.
    adapter_src: str | None = None

    @property
    def timeout_s(self) -> int:
        """The attempt's wall-clock cap (``caps.timeout_s``)."""

        return self.caps.timeout_s

    def check(self) -> None:
        if (self.mode == "render") != (self.trace is not None):
            raise RecordError(f"RunRequest {self.request}: `trace` is set exactly for render mode")


@dataclass(frozen=True)
class OutputFile(Record):
    kind: str
    path: str
    sha256: str = spec(pattern=SHA256)
    bytes: int = spec()


@dataclass(frozen=True)
class RunResult(Record):
    """``result.json``, written last by the attempt; its absence makes the attempt ``infra``."""

    SCHEMA = "run-result/1"
    request: str = spec()
    id: str = spec(pattern=STABLE_ID)
    mode: RunMode = spec()
    outcome: Literal["pass", "fail", "infra"] = spec()
    started: str = spec()
    ended: str = spec()
    host: str = spec()
    cls: str | None = spec(None, key="class", pattern=CLASS_PATTERN)
    msg: str | None = None
    tail: tuple[str, ...] = ()
    output_shapes: tuple[tuple[int, ...], ...] = ()
    params: int | None = None
    modes: Modes | None = None
    input_boundary: InputBoundary | None = None
    rows: tuple[LedgerRow | RenderRow, ...] = ()
    files: tuple[OutputFile, ...] = ()

    def check(self) -> None:
        check_class(self.cls, f"RunResult {self.request}")
        if (self.outcome == "pass") == (self.cls is not None):
            raise RecordError(f"RunResult {self.request}: a class is set exactly when not passing")
