"""Tests for ``menagerie.tl_compat``: one Menagerie name for each TorchLens name that differs
between the pin (2.36) and the 2.38 and 3.0 lines, and refusal to compare values across value
families.

The trace-shaped tests use stand-in objects carrying either line's names, so both branches run
whichever TorchLens is installed; the CaptureOptions test uses the installed TorchLens.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from menagerie import tl_compat


def test_family_breaks_at_238_and_unknown_is_none() -> None:
    assert tl_compat.family("n_ops", "2.36.0") == "Trace.num_ops"
    assert tl_compat.family("n_ops", "2.37.9") == "Trace.num_ops"
    assert tl_compat.family("n_ops", "2.38.1") == "len(Trace.ops)"
    assert tl_compat.family("graph_hash", "2.38.0+gad6555f") == "Trace.provenance.fingerprint"
    assert tl_compat.family("graph_hash", None) == "Trace.graph_shape_hash"
    assert tl_compat.family("n_ops", "") == "Trace.num_ops"
    assert tl_compat.family("graph_hash", "unknown") is None


def test_same_family_compares_across_pins() -> None:
    assert tl_compat.same_value("n_ops", 12, "2.35.3", "12", "2.36.0")
    assert not tl_compat.same_value("graph_hash", "a", "2.36.0", "b", "2.36.0")
    assert tl_compat.same_value("graph_hash", None, "2.36.0", "", "2.36.0")


def test_an_unrecorded_version_compares_only_with_older_families() -> None:
    assert tl_compat.same_value("n_ops", 7, None, "7", "2.36.0")
    assert tl_compat.same_value("graph_hash", "h", "2.35.2", "h", "")


@pytest.mark.parametrize(("tl_a", "tl_b"), [("2.36.0", "2.38.1"), ("2.38.1", "2.35.3"),
                                             (None, "2.38.1"), ("2.38.0", ""),
                                             ("unknown", "2.36.0")])
def test_mixed_or_unknown_pair_is_refused(tl_a: str | None, tl_b: str | None) -> None:
    with pytest.raises(tl_compat.MixedPinError):
        tl_compat.same_value("graph_hash", "x", tl_a, "x", tl_b)
    assert not tl_compat.comparable("graph_hash", tl_a, tl_b)


def test_compare_note() -> None:
    assert tl_compat.compare_note("n_ops", 3, "2.36.0", 3, "2.36.0") is None
    assert tl_compat.compare_note("n_ops", 3, "2.36.0", 4, "2.36.0") == "n_ops differs"
    note = tl_compat.compare_note("n_ops", 3, "2.36.0", 3, "2.38.1")
    assert note is not None and "not comparable" in note


def test_num_ops_and_graph_hash_read_either_name() -> None:
    old = SimpleNamespace(num_ops=7, graph_shape_hash="abc")
    new = SimpleNamespace(ops=[1, 2, 3], provenance=SimpleNamespace(fingerprint="def"))
    assert tl_compat.num_ops(old) == 7
    assert tl_compat.num_ops(new) == 3
    assert tl_compat.graph_hash(old) == "abc"
    assert tl_compat.graph_hash(new) == "def"
    assert tl_compat.graph_hash(SimpleNamespace(graph_shape_hash="")) is None


def test_missing_names_raise_unless_allowed() -> None:
    with pytest.raises(tl_compat.TorchLensNameError):
        tl_compat.num_ops(SimpleNamespace())
    with pytest.raises(tl_compat.TorchLensNameError):
        tl_compat.graph_hash(SimpleNamespace())
    assert tl_compat.graph_hash(SimpleNamespace(), missing_ok=True) is None


def test_status_count_reads_either_name() -> None:
    old = SimpleNamespace(failed_node_count=2, replayed_node_count=9)
    new = SimpleNamespace(num_failed_ops=1, num_replayed_ops=8)
    assert tl_compat.status_count(old, "failed") == 2
    assert tl_compat.status_count(new, "failed") == 1
    assert tl_compat.status_count(new, "replayed") == 8
    assert tl_compat.status_count(SimpleNamespace(), "effect_region") is None


def test_capture_options_takes_pinned_names_on_installed_torchlens() -> None:
    options = tl_compat.capture_options(inference_only=True, random_seed=5,
                                        save_rng_states=False, layers_to_save=None)
    seed = getattr(options, "random_seed", getattr(options, "seed", None))
    assert seed == 5
    assert options.inference_only is True


def test_capture_options_keeps_the_host_peak_where_accepted() -> None:
    from torchlens.options import CaptureOptions

    options = tl_compat.capture_options(inference_only=True, measure_host_peak_memory=True)
    if "measure_host_peak_memory" in tl_compat._accepted(CaptureOptions):  # 3.0
        assert options.measure_host_peak_memory is True
    assert options.inference_only is True


def test_capture_options_refuses_unknown_name() -> None:
    with pytest.raises(tl_compat.TorchLensNameError):
        tl_compat.capture_options(no_such_option=True)


def test_save_uses_whichever_entry_point_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchlens as tl

    calls: list[tuple[str, object]] = []

    class Stub:
        def save(self, path: object, **options: object) -> None:
            calls.append(("method", path))

    monkeypatch.setattr(tl, "save", lambda trace, path, **kw: calls.append(("tl.save", path)),
                        raising=False)
    monkeypatch.setattr(tl_compat, "installed_version", lambda: "2.36.0")
    tl_compat.save(Stub(), "a")
    # 2.38 keeps a callable ``tl.save`` stub that raises; the version picks the method.
    monkeypatch.setattr(tl_compat, "installed_version", lambda: "2.38.1")
    tl_compat.save(Stub(), "b")
    assert calls == [("tl.save", "a"), ("method", "b")]


def test_publish_refuses_a_distinct_count_over_mixed_families() -> None:
    from menagerie.pipeline import publish
    from menagerie.pipeline.records import RecordError

    basis = {"basis": "architecture"}
    same = [{"torchlens_version": "2.35.3", "stable_id": "a"},
            {"torchlens_version": "2.36.0", "stable_id": "b"}]
    publish._one_value_family(same, basis)
    mixed = [*same, {"torchlens_version": "2.38.1", "stable_id": "c"}]
    with pytest.raises(RecordError, match="value families"):
        publish._one_value_family(mixed, basis)


def test_layer_list_reads_the_attribute_or_iterates_the_trace() -> None:
    class Iterable238:
        def __iter__(self):
            return iter(["a", "b"])

    assert tl_compat.layer_list(SimpleNamespace(layer_list=["x"])) == ["x"]
    assert tl_compat.layer_list(Iterable238()) == ["a", "b"]


def test_trace_field_reads_the_trace_or_its_238_group() -> None:
    old = SimpleNamespace(total_flops_forward=10, is_dynamic_graph=True)
    new = SimpleNamespace(cost=SimpleNamespace(total_flops_forward=12),
                          graph=SimpleNamespace(is_dynamic=False, is_recurrent=True))
    assert tl_compat.trace_field(old, "total_flops_forward", 0) == 10
    assert tl_compat.trace_field(new, "total_flops_forward", 0) == 12
    assert tl_compat.trace_field(old, "is_dynamic_graph", None) is True
    assert tl_compat.trace_field(new, "is_dynamic_graph", None) is False
    assert tl_compat.trace_field(new, "is_recurrent") is True
    assert tl_compat.trace_field(new, "num_modules", 0) == 0
    mode_238 = SimpleNamespace(options=SimpleNamespace(module_identity_mode="pytree_module"))
    assert tl_compat.trace_field(mode_238, "module_identity_mode") == "pytree_module"
    assert tl_compat.trace_field(SimpleNamespace(module_identity_mode="function_root"),
                                 "module_identity_mode") == "function_root"


def test_trace_field_reads_the_30_homes_and_counts() -> None:
    new = SimpleNamespace(
        source=SimpleNamespace(class_name="ResNet", class_qualname="timm.ResNet"),
        layers=SimpleNamespace(recurrent=["lstm_1"]), modules=[1, 2, 3], module_calls=[1, 2],
        params=[1] * 4, graph=SimpleNamespace(conditionals=[1]),
        ops=SimpleNamespace(compute=[1] * 7), buffer_write_ops=[1, 2],
        cost=SimpleNamespace(macs_by_kind={"conv2d": {"forward": 9}}))
    assert tl_compat.trace_field(new, "model_class_name") == "ResNet"
    assert tl_compat.trace_field(new, "model_class_qualname") == "timm.ResNet"
    assert tl_compat.trace_field(new, "recurrent_layers") == ["lstm_1"]
    counts = {name: tl_compat.trace_field(new, name, 0) for name in tl_compat.TRACE_FIELD_COUNTS}
    assert counts == {"num_compute_ops": 7, "num_conditionals": 1, "num_modules": 3,
                      "num_module_calls": 2, "num_param_tensors": 4, "num_buffer_write_ops": 2}
    assert tl_compat.trace_field(new, "macs_by_op_type") == {"conv2d": {"forward": 9}}
    old = SimpleNamespace(num_modules=5, model_class_name="Old")
    assert tl_compat.trace_field(old, "num_modules", 0) == 5
    assert tl_compat.trace_field(old, "model_class_name") == "Old"
    assert tl_compat.trace_field(SimpleNamespace(), "num_conditionals", 0) == 0
    assert tl_compat.trace_field(new, "no_such_field", "default") == "default"


def test_op_field_reads_the_op_or_its_30_group() -> None:
    old = SimpleNamespace(max_distance_from_input=4, func_qualname="torch.add")
    new = SimpleNamespace(graph=SimpleNamespace(max_distance_from_input=5, is_buffer=True),
                          module_context=SimpleNamespace(call_depth=2),
                          cost=SimpleNamespace(flops_forward=None))
    assert tl_compat.op_field(old, "max_distance_from_input") == 4
    assert tl_compat.op_field(new, "max_distance_from_input") == 5
    assert tl_compat.op_field(new, "module_call_depth", 0) == 2
    assert tl_compat.op_field(new, "is_buffer", False) is True
    assert tl_compat.op_field(new, "flops_forward", "unset") is None  # a recorded None stays
    assert tl_compat.op_field(old, "func_qualname") == "torch.add"
    assert tl_compat.op_field(new, "func_qualname", "") == ""


class _Draw236:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def draw(self, vis_mode: str = "unrolled", vis_outpath: str = "modelgraph",
             vis_save_only: bool = False, vis_fileformat: str = "pdf",
             vis_node_placement: str = "auto", collapse: str = "none",
             show_legend: bool | None = None) -> None:
        self.calls.append(dict(vis_mode=vis_mode, vis_outpath=vis_outpath,
                               vis_save_only=vis_save_only, vis_fileformat=vis_fileformat,
                               vis_node_placement=vis_node_placement, collapse=collapse,
                               show_legend=show_legend))


class _Draw30:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def draw(self, path: str | None = None, *, rolled: bool = False, collapse: str = "none",
             layout: str = "auto", show_legend: bool | None = None,
             file_format: str | None = None, open_viewer: bool = False) -> None:
        self.calls.append(dict(path=path, rolled=rolled, collapse=collapse, layout=layout,
                               show_legend=show_legend, file_format=file_format,
                               open_viewer=open_viewer))


def test_draw_passes_pinned_keywords_through_and_renames_them_for_30() -> None:
    options = dict(vis_mode="rolled", collapse="auto", vis_outpath="out/rolled-auto",
                   vis_fileformat="svg", vis_node_placement="rank", vis_save_only=True,
                   show_legend=False)
    old, new = _Draw236(), _Draw30()
    tl_compat.draw(old, **options)
    tl_compat.draw(new, **options)
    assert old.calls == [options]
    assert new.calls == [dict(path="out/rolled-auto", rolled=True, collapse="auto",
                              layout="rank", show_legend=False, file_format="svg",
                              open_viewer=False)]
    tl_compat.draw(new, vis_mode="unrolled", vis_outpath="x")
    assert new.calls[-1]["rolled"] is False and new.calls[-1]["path"] == "x"


def test_draw_refuses_an_unknown_keyword_or_view_on_30() -> None:
    with pytest.raises(tl_compat.TorchLensNameError):
        tl_compat.draw(_Draw30(), vis_no_such_option=1)
    with pytest.raises(ValueError, match="vis_mode"):
        tl_compat.draw(_Draw30(), vis_mode="sideways")


def test_layer_views_read_either_version() -> None:
    layer_a = SimpleNamespace(layer_label="lin", num_passes=2, uses_params=True)
    old = SimpleNamespace(layer_logs={"lin": layer_a}, layer_num_calls={"lin": 2})
    layer_b = SimpleNamespace(layer_label="lin", num_passes=3, has_params=False)
    new = SimpleNamespace(layers=[layer_b])
    assert tl_compat.layers(old) == [layer_a]
    assert tl_compat.layer_calls(old) == {"lin": 2}
    assert tl_compat.layer_calls(new) == {"lin": 3}
    assert tl_compat.layer_has_params(layer_a) is True
    assert tl_compat.layer_has_params(layer_b) is False


def test_op_type_reads_the_op_or_its_layer() -> None:
    assert tl_compat.op_type(SimpleNamespace(layer_type="conv2d")) == "conv2d"
    assert tl_compat.op_type(SimpleNamespace(layer=SimpleNamespace(layer_type="gelu"))) == "gelu"
    assert tl_compat.op_type(SimpleNamespace()) is None


def test_report_options_from_either_home(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchlens as tl

    made = tl_compat.report_options(deterministic=True, share_safe=True)
    assert made.deterministic is True and made.share_safe is True
    monkeypatch.delattr(tl.export, "ReportOptions", raising=False)
    monkeypatch.setattr(tl, "options", SimpleNamespace(HtmlReportOptions=SimpleNamespace),
                        raising=False)
    assert tl_compat.report_options(share_safe=True).share_safe is True
