"""Parity tests for the menagerie recipe/runtime module split."""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.catalog import CatalogRow, load_rows
from menagerie.recipe import build_model_and_input

CATALOG_ARTIFACT = Path(__file__).resolve().parents[1] / "menagerie" / "data" / "catalog.db"
pytestmark = pytest.mark.skipif(
    not CATALOG_ARTIFACT.exists(),
    reason="requires local menagerie data artifacts; not part of the portable suite",
)


SAMPLE_MODEL_IDS = (
    3,
    23,
    24,
    26,
    27,
    79,
    80,
    82,
    92,
    # 182 = timm aimv2_1b_patch14_336 (1.2B params at 336px): >180 s per CPU forward under
    # capture, so it is classified slow (the recipe itself is fine), not dropped.
    pytest.param(182, marks=pytest.mark.slow),
    185,
    186,
    # 187 = timm aimv2_huge_patch14_224 (~600M params): >180 s per CPU forward under capture.
    pytest.param(187, marks=pytest.mark.slow),
    222,
    244,
    249,
    252,
    274,
    275,
    276,
    298,
    305,
    311,
    312,
    315,
)
# Optional heavy zoo dependencies required by specific sample rows; skipped when
# the dependency is not installed, following the repo's importorskip pattern.
OPTIONAL_DEPS_BY_MODEL_ID = {
    298: ("mmcv", "mmdet"),  # mmdet_autoassign (mmdetection-configs zoo)
}
LEGACY_GENERATE_IMPORTS = (
    "CACHE_ROOTS",
    "DependencyPlan",
    "assert_min_free",
    "build_input_for_row",
    "cleanup_runtime",
    "combine_notes",
    "cuda_is_available",
    "default_jobs",
    "dependency_plan",
    "device_note",
    "disk_free_gb",
    "group_by_dependency",
    "install_dependency_plan",
    "instantiate_model",
    "is_classics_row",
    "is_device_related_error",
    "log_event",
    "move_model_and_input_to_device",
    "purge_new_cache_entries",
    "safe_path_part",
    "select_rows",
    "snapshot_cache",
    "unrenderable_reason",
)


def _rows_by_id() -> dict[int, CatalogRow]:
    """Load catalog rows keyed by model ID.

    Returns
    -------
    dict[int, CatalogRow]
        Catalog rows keyed by their integer model ID.
    """

    return {row.model_id: row for row in load_rows()}


def _trace_signature(row: CatalogRow) -> tuple[str, int]:
    """Build, trace, and summarize one catalog row.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    tuple[str, int]
        Graph-shape hash and node/op count.
    """

    import torch
    import torchlens as tl

    torch.manual_seed(0)
    model, input_value = build_model_and_input(row)
    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        trace = tl.trace(model, input_value, capture=tl.options.CaptureOptions(inference_only=True))
    graph_shape_hash = str(getattr(trace, "graph_shape_hash", "") or "")
    n_nodes = int(getattr(trace, "num_ops", 0) or len(getattr(trace, "layer_logs", {}) or {}))
    return graph_shape_hash, n_nodes


def test_catalog_build_counts_are_stable() -> None:
    """The rebuilt catalog keeps the module-split parity counts.

    Snapshot bumped 2026-06-25 (3226 -> 3283 classics; 11655 -> 11712 rows) and again
    2026-09-03: the classics waves through wave 20 plus the CPU-safe example-input fixes
    let a fresh CPU host rebuild the full catalog (4962 classics; 13503 rows). Verified
    deterministic and duplicate-free (13503 rows == 13503 unique stable_ids).
    """

    rows = load_rows()
    assert len(rows) == 13503
    assert len({row.stable_id for row in rows}) == len(rows)
    assert sum(row.source == "classics" for row in rows) == 4962


def test_generate_menagerie_legacy_import_surface_is_preserved() -> None:
    """Names consumed through the old generator import path remain available."""

    import menagerie.generate_menagerie as generate_menagerie

    missing = [name for name in LEGACY_GENERATE_IMPORTS if not hasattr(generate_menagerie, name)]
    assert missing == []


@pytest.mark.parametrize("model_id", SAMPLE_MODEL_IDS)
def test_recipe_build_model_and_input_trace_signature_is_stable(model_id: int) -> None:
    """Recipe-built model/input pairs have stable trace signatures across repeated calls."""

    for optional_dep in OPTIONAL_DEPS_BY_MODEL_ID.get(model_id, ()):
        pytest.importorskip(optional_dep)
    rows = _rows_by_id()
    row = rows[model_id]
    first = _trace_signature(row)
    second = _trace_signature(row)
    assert first == second
    assert first[0]
    assert first[1] > 0
