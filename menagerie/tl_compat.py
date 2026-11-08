"""Menagerie's adapters for TorchLens names that differ between the pin and the 2.38 and 3.0 lines.

Menagerie runs on the pinned TorchLens (3.0.2) and must keep running at the 2.36.0 reader pin,
where TorchLens 2.38 and 3.0 renamed or removed several names Menagerie reads
(``docs/torchlens-boundary.md``). Every such name goes through one function here, which reads the
installed TorchLens and uses whichever name it offers; nothing outside this module and
``tl_validation`` names a version-specific TorchLens attribute.

Value families. Some values change meaning across TorchLens versions: the op count
(``Trace.num_ops`` through 2.37, ``len(Trace.ops)`` from 2.38), the graph hash
(``Trace.graph_shape_hash``, then ``Trace.provenance.fingerprint``, address-free with different
values), the skeleton hash (``tl.hash.trace``, then ``tl.fingerprint.trace``) and Menagerie's own
``architecture`` and ``model`` digests (computed over TorchLens's per-op records, whose fields
changed with them). Every such value is stored beside the TorchLens version that made it (the
ledger row's ``tl``), and ``same_value`` refuses to compare two values from different families
(``MixedPinError``) instead of reporting them as different, or as equal. Two pins inside one
family still compare: a repin check from 2.35.3 to 2.36.0 is a deliberate comparison of the same
measure.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from typing import Any

# The first TorchLens release of each family, per value kind; a version takes the family of the
# last break at or below it. Add a break here when TorchLens changes what a value means.
FAMILY_BREAKS: dict[str, tuple[tuple[tuple[int, int], str], ...]] = {
    "n_ops": (((0, 0), "Trace.num_ops"), ((2, 38), "len(Trace.ops)")),
    "graph_hash": (((0, 0), "Trace.graph_shape_hash"),
                   ((2, 38), "Trace.provenance.fingerprint")),
    "skeleton": (((0, 0), "tl.hash.trace"), ((2, 38), "tl.fingerprint.trace")),
    "architecture": (((0, 0), "op records to 2.37"), ((2, 38), "op records from 2.38")),
    "model": (((0, 0), "op records to 2.37"), ((2, 38), "op records from 2.38")),
}

# CaptureOptions keywords Menagerie writes (2.36 names) and their 2.38 names.
CAPTURE_RENAMES = {"random_seed": "seed", "intervention_ready": "replayable",
                   "save_rng_states": "save_rng"}

# CaptureOptions keywords 2.38 no longer takes because ``tl.trace(save=...)`` carries them; the
# caller passes the same selection as ``save=`` (``save=None`` sets "save no activations" on both
# lines), so the keyword is dropped when the installed CaptureOptions lacks it.
CAPTURE_CARRIED_BY_TRACE = frozenset({"layers_to_save"})

# CaptureOptions keywords from 3.0 that keep a pinned default: 2.36 measured the CPU forward peak
# (a process-wide resident high-water reset) on every capture, and 3.0 does it only under
# ``measure_host_peak_memory=True``. Dropped where the installed CaptureOptions lacks them.
CAPTURE_WHERE_ACCEPTED = frozenset({"measure_host_peak_memory"})

# ``Trace.draw`` keywords Menagerie writes (2.36 names) and their 3.0 names, for a ``Trace.draw``
# that no longer takes the pinned ones. ``vis_mode`` becomes the bool ``rolled`` and
# ``vis_save_only`` becomes ``open_viewer`` negated (3.0 opens no viewer by default); ``draw``
# handles those two.
DRAW_RENAMES = {"vis_outpath": "path", "vis_fileformat": "file_format",
                "vis_node_placement": "layout", "vis_call_depth": "depth", "vis_theme": "theme",
                "vis_renderer": "renderer", "node_mode": "node_style",
                "show_buffer_layers": "show_buffers", "vis_graph_overrides": "graph_overrides",
                "vis_edge_overrides": "edge_overrides",
                "vis_grad_edge_overrides": "grad_edge_overrides",
                "vis_module_overrides": "module_overrides",
                "vis_intervention_mode": "intervention_mode", "vis_show_cone": "show_cone",
                "scale": "size_scale"}
VIS_MODES = ("unrolled", "rolled")

# ValidationStatus counts: 2.36 name and 2.38 name.
STATUS_COUNTS = {"replayed": ("replayed_node_count", "num_replayed_ops"),
                 "failed": ("failed_node_count", "num_failed_ops"),
                 "effect_region": ("effect_region_node_count", "num_effect_region_ops"),
                 "pure_unverified": ("pure_unverified_node_count", "num_pure_unverified_ops")}

# Trace-level fields 2.38 moved into a group (``trace.cost``, ``trace.graph``, ``trace.options``,
# ``trace.source``, ``trace.layers``): pinned name to its attribute path from the trace.
TRACE_FIELD_HOMES: dict[str, tuple[str, ...]] = {
    **{name: ("cost", name) for name in (
        "total_param_memory", "total_activation_memory", "forward_peak_memory",
        "total_flops_forward", "total_flops_backward", "total_macs_forward",
        "total_macs_backward", "flops_by_op_type")},
    **{name: ("graph", name) for name in (
        "branching_factor", "max_in_degree", "max_out_degree", "is_branching", "is_recurrent",
        "has_conditional_branching")},
    "is_dynamic_graph": ("graph", "is_dynamic"),
    "max_layer_op_count": ("graph", "max_ops_per_layer"),
    "module_identity_mode": ("options", "module_identity_mode"),
    "model_class_name": ("source", "class_name"),
    "model_class_qualname": ("source", "class_qualname"),
    "recurrent_layers": ("layers", "recurrent"),
    "macs_by_op_type": ("cost", "macs_by_kind"),
}
# Op and Layer fields 3.0 moved into a group (TorchLens ``docs/reference/deprecations.md``, the
# removed-spellings ledger): pinned name to its attribute path from the op.
OP_FIELD_HOMES: dict[str, tuple[str, ...]] = {
    "activation_memory": ("cost", "activation_memory"),
    "flops_forward": ("cost", "flops_forward"),
    "atomic_module": ("module_context", "atomic_module"),
    "module_call_depth": ("module_context", "call_depth"),
    "func_qualname": ("source", "func_qualname"),
    "is_buffer": ("graph", "is_buffer"),
    "max_distance_from_input": ("graph", "max_distance_from_input"),
    "non_tensor_kwargs": ("call", "non_tensor_kwargs"),
    "non_tensor_pos_args": ("call", "non_tensor_pos_args"),
}
# Counts the pinned trace carried as a field and 3.0 gives as the length of a collection: pinned
# name to the collection's attribute path (``num_compute_ops`` counts ops, as at 2.36, not the
# compute Layers). Each was checked against 2.36.0's value on the same models (report
# ``2026-11-06-tl301-readiness``).
TRACE_FIELD_COUNTS: dict[str, tuple[str, ...]] = {
    "num_compute_ops": ("ops", "compute"),
    "num_conditionals": ("graph", "conditionals"),
    "num_modules": ("modules",),
    "num_module_calls": ("module_calls",),
    "num_param_tensors": ("params",),
    "num_buffer_write_ops": ("buffer_write_ops",),
}

_MISSING = object()


class TorchLensNameError(RuntimeError):
    """The installed TorchLens offers neither the pinned nor the 2.38 name Menagerie needs."""


class MixedPinError(ValueError):
    """Two values from different TorchLens value families, or of unknown version, were compared."""


# --- versions and families -------------------------------------------------------------------


def installed_version() -> str:
    """The installed TorchLens version string."""

    import torchlens as tl

    return str(tl.__version__)


def _major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"\s*(\d+)\.(\d+)", version or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def family(kind: str, tl_version: str | None) -> str | None:
    """The value family ``kind`` has at ``tl_version``; ``None`` for a version that does not parse.

    An unrecorded version (``None`` or empty) is the oldest family: Menagerie's rows and records
    began carrying ``tl`` and ``torchlens_version`` before 2.38 existed, and every producer that
    can run 2.38 records it, so a value without one was made at 2.37 or earlier.
    """

    breaks = FAMILY_BREAKS[kind]
    if not tl_version:
        return breaks[0][1]
    parsed = _major_minor(tl_version)
    if parsed is None:
        return None
    return [name for first, name in breaks if parsed >= first][-1]


def comparable(kind: str, tl_a: str | None, tl_b: str | None) -> bool:
    """Whether a ``kind`` value made at ``tl_a`` may be compared with one made at ``tl_b``."""

    fam_a, fam_b = family(kind, tl_a), family(kind, tl_b)
    return fam_a is not None and fam_a == fam_b


def same_value(kind: str, a: Any, tl_a: str | None, b: Any, tl_b: str | None) -> bool:
    """``a == b`` for two ``kind`` values, refusing a pair from different value families.

    Values are compared as strings, as the TSV tables store them (``None`` reads as empty).
    """

    if not comparable(kind, tl_a, tl_b):
        raise MixedPinError(
            f"{kind} at TorchLens {tl_a or 'unknown'} ({family(kind, tl_a) or 'unknown family'}) "
            f"and at {tl_b or 'unknown'} ({family(kind, tl_b) or 'unknown family'}) measure "
            "different things; they are not compared")
    return str("" if a is None else a) == str("" if b is None else b)


def compare_note(kind: str, a: Any, tl_a: str | None, b: Any, tl_b: str | None) -> str | None:
    """For a report row: ``None`` when equal, else why not (differs, or not comparable)."""

    try:
        return None if same_value(kind, a, tl_a, b, tl_b) else f"{kind} differs"
    except MixedPinError:
        return (f"{kind} not comparable: TorchLens {tl_a or 'unknown'} vs {tl_b or 'unknown'} "
                "are different value families")


# --- trace values ----------------------------------------------------------------------------


def num_ops(trace: Any) -> int:
    """The trace's op count: ``Trace.num_ops`` through 2.37, ``len(Trace.ops)`` from 2.38."""

    value = getattr(trace, "num_ops", _MISSING)
    if value is _MISSING:
        ops = getattr(trace, "ops", _MISSING)
        if ops is _MISSING:
            raise TorchLensNameError("the trace has neither num_ops nor ops")
        return len(ops)
    return int(value)


def layer_list(trace: Any) -> Any:
    """Every captured entry in execution order: ``Trace.layer_list`` through 2.37; from 2.38 the
    trace itself iterates the same list (``trace.layers`` is the per-layer aggregate, not this)."""

    value = getattr(trace, "layer_list", _MISSING)
    return list(trace) if value is _MISSING else value


def _path(obj: Any, path: tuple[str, ...]) -> Any:
    for name in path:
        obj = getattr(obj, name, _MISSING)
        if obj is _MISSING:
            break
    return obj


def trace_field(trace: Any, name: str, default: Any = None) -> Any:
    """A trace-level field by its pinned name: on the trace through 2.37, at its 2.38 home
    (``TRACE_FIELD_HOMES``) or as its collection's length (``TRACE_FIELD_COUNTS``) after;
    ``default`` when none has it (as for a preview trace)."""

    value = getattr(trace, name, _MISSING)
    if value is _MISSING and name in TRACE_FIELD_HOMES:
        value = _path(trace, TRACE_FIELD_HOMES[name])
    elif value is _MISSING and name in TRACE_FIELD_COUNTS:
        collection = _path(trace, TRACE_FIELD_COUNTS[name])
        value = _MISSING if collection is _MISSING else len(collection)
    return default if value is _MISSING else value


def layers(trace: Any) -> list[Any]:
    """The per-layer aggregates (every pass of a layer in one): ``Trace.layer_logs`` values
    through 2.37, ``Trace.layers`` from 2.38."""

    logs = getattr(trace, "layer_logs", _MISSING)
    return list(trace.layers) if logs is _MISSING else list(logs.values())


def layer_calls(trace: Any) -> dict[str, int]:
    """Each layer's pass count by layer label: ``Trace.layer_num_calls`` through 2.37, else read
    from the per-layer aggregates."""

    calls = getattr(trace, "layer_num_calls", _MISSING)
    if calls is not _MISSING:
        return dict(calls)
    return {str(layer.layer_label): int(layer.num_passes) for layer in layers(trace)}


def op_field(op: Any, name: str, default: Any = None) -> Any:
    """An Op or Layer field by its pinned name: on the record through 2.38, at its 3.0 home
    (``OP_FIELD_HOMES``) after; ``default`` when neither has it."""

    value = getattr(op, name, _MISSING)
    if value is _MISSING and name in OP_FIELD_HOMES:
        value = _path(op, OP_FIELD_HOMES[name])
    return default if value is _MISSING else value


def layer_has_params(layer: Any) -> bool:
    """Whether a layer has parameters: ``uses_params`` through 2.37, ``has_params`` from 2.38."""

    value = getattr(layer, "uses_params", _MISSING)
    return bool(getattr(layer, "has_params", False) if value is _MISSING else value)


def op_type(op: Any) -> Any:
    """An op's layer type (``conv2d``, ``gelu``): on the op through 2.37, on ``op.layer`` from 2.38;
    ``None`` when neither has it."""

    value = getattr(op, "layer_type", _MISSING)
    if value is _MISSING:
        return getattr(getattr(op, "layer", None), "layer_type", None)
    return value


def graph_hash(trace: Any, *, missing_ok: bool = False) -> str | None:
    """The trace's graph hash, or ``None``: ``graph_shape_hash`` through 2.37, else
    ``provenance.fingerprint`` (a different value family; see ``FAMILY_BREAKS``).

    A trace with neither name raises, unless ``missing_ok`` (then ``None``).
    """

    value = getattr(trace, "graph_shape_hash", _MISSING)
    if value is _MISSING:
        provenance = getattr(trace, "provenance", None)
        value = getattr(provenance, "fingerprint", _MISSING)
        if value is _MISSING:
            if missing_ok:
                return None
            raise TorchLensNameError(
                "the trace has neither graph_shape_hash nor provenance.fingerprint")
    return str(value) if value else None


def _fingerprint_namespace() -> Any:
    import torchlens as tl

    namespace = getattr(tl, "hash", None) or getattr(tl, "fingerprint", None)
    if namespace is None or not hasattr(namespace, "trace"):
        raise TorchLensNameError("TorchLens has neither tl.hash.trace nor tl.fingerprint.trace")
    return namespace


def skeleton(trace: Any) -> str:
    """TorchLens's address-free skeleton hash of a trace (``tl.hash.trace``, then
    ``tl.fingerprint.trace``)."""

    return str(_fingerprint_namespace().trace(trace))


def status_count(status: Any, name: str) -> Any:
    """A ``ValidationStatus`` count by short name (``replayed``, ``failed``, ``effect_region``,
    ``pure_unverified``) under either version's attribute name; ``None`` when absent."""

    old, new = STATUS_COUNTS[name]
    value = getattr(status, old, _MISSING)
    return getattr(status, new, None) if value is _MISSING else value


# --- capture and save ------------------------------------------------------------------------


def _accepted(callable_: Callable[..., Any]) -> Mapping[str, Any]:
    return inspect.signature(callable_).parameters


def capture_options(**options: Any) -> Any:
    """``CaptureOptions`` from keywords in the pinned names, renamed where the installed
    TorchLens took the 2.38 name (``CAPTURE_RENAMES``) and dropped where ``tl.trace(save=...)``
    now carries the setting (``CAPTURE_CARRIED_BY_TRACE``) or the pin does it by default
    (``CAPTURE_WHERE_ACCEPTED``)."""

    from torchlens.options import CaptureOptions

    params = _accepted(CaptureOptions)
    kwargs = {}
    for name, value in options.items():
        new = CAPTURE_RENAMES.get(name)
        if name in params:
            kwargs[name] = value
        elif new and new in params:
            kwargs[new] = value
        elif name in CAPTURE_CARRIED_BY_TRACE or name in CAPTURE_WHERE_ACCEPTED:
            continue
        else:
            raise TorchLensNameError(f"CaptureOptions takes neither {name}= nor {new}=")
    return CaptureOptions(**kwargs)


def save(trace: Any, path: Any, **options: Any) -> None:
    """Save a trace: ``tl.save(trace, path)`` through 2.37, ``trace.save(path)`` from 2.38.

    The installed version decides, not ``hasattr``: 2.38 keeps ``tl.save`` as a callable stub
    that raises and points at ``Trace.save``.
    """

    import torchlens as tl

    legacy = getattr(tl, "save", None)
    if (_major_minor(installed_version()) or (0, 0)) < (2, 38) and callable(legacy):
        legacy(trace, path, **options)
    else:
        trace.save(path, **options)


def draw(trace: Any, **options: Any) -> Any:
    """``trace.draw`` from keywords in the pinned names (``vis_mode=``, ``vis_outpath=``,
    ``vis_fileformat=``, ``vis_node_placement=``, ``vis_save_only=``), passed as written to a
    ``Trace.draw`` that takes them (through 2.37) and renamed for one that does not (3.0:
    ``DRAW_RENAMES``). Either way the drawing is written to ``<vis_outpath>.<vis_fileformat>``."""

    params = _accepted(trace.draw)
    if "vis_outpath" in params:
        return trace.draw(**options)
    kwargs: dict[str, Any] = {}
    for name, value in options.items():
        if name == "vis_mode":
            if value not in VIS_MODES:
                raise ValueError(f"vis_mode must be one of {VIS_MODES}, not {value!r}")
            kwargs["rolled"] = value == "rolled"
        elif name == "vis_save_only":
            kwargs["open_viewer"] = not value
        elif name in params:
            kwargs[name] = value
        elif DRAW_RENAMES.get(name) in params:
            kwargs[DRAW_RENAMES[name]] = value
        else:
            raise TorchLensNameError(
                f"Trace.draw takes neither {name}= nor {DRAW_RENAMES.get(name) or 'a 3.0 name'}=")
    return trace.draw(**kwargs)


def report_options(**options: Any) -> Any:
    """The HTML report's options: ``tl.export.ReportOptions`` through 2.38,
    ``tl.options.HtmlReportOptions`` from 3.0 (same keywords: ``deterministic``, ``share_safe``)."""

    import torchlens as tl

    cls = getattr(tl.export, "ReportOptions", None)
    if cls is None:
        cls = getattr(getattr(tl, "options", None), "HtmlReportOptions", None)
    if cls is None:
        raise TorchLensNameError(
            "TorchLens has neither tl.export.ReportOptions nor tl.options.HtmlReportOptions")
    return cls(**options)
