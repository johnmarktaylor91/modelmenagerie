"""Readers of a trace save after the twin fix (2026-10-28): none needs ``tlspec-portable``.

Since the worker stopped writing ``render.tlspec`` beside a refused runnable save, a trace's
files may list ``tlspec`` alone. Only the modules in ``PORTABLE_AWARE`` name the portable save,
and each has its own fallback test (worker render mode, flow staging and rerender, regression
sizing, the store collectors). Every other reader keys on the ledger's ``trace`` sha, which is
always ``trace.tlspec``, so a missing portable save cannot change what it reads. Torch-free.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2] / "menagerie"
NAMES = re.compile(r"tlspec-portable|RENDER_FILE|render\.tlspec")
# The modules that write, stage, store or size the portable save, each with a fallback test.
PORTABLE_AWARE = {
    "pipeline/worker.py",  # writes it; render mode falls back to trace.tlspec (test_worker)
    "pipeline/worker_preview.py",  # writes it beside a runnable save (test_worker_preview)
    "pipeline/flow.py",  # stages it, else trace.tlspec (test_flow render staging, rerender)
    "pipeline/cluster.py",  # puts whatever saves exist in the store (MIME)
    "pipeline/regression.py",  # stores and sizes whatever saves exist (test_regression)
}
# Readers that pick a trace save by the ledger's ``trace`` sha (or its summary), never the
# portable save: site export, harvest, describe, packet, review, check-in, implement, the
# replay table, the store ceiling, the axon feeder and the dataset export.
LEDGER_KEYED = ("pipeline/publish.py", "pipeline/harvest.py", "pipeline/describe.py",
                "pipeline/describe_text.py", "pipeline/packet.py", "pipeline/review.py",
                "pipeline/checkin.py", "pipeline/implement.py", "pipeline/replay_table.py",
                "pipeline/store_ceiling.py", "pipeline/axon_feed.py", "pipeline/dataset.py")


def _naming() -> set[str]:
    skip = ("data/", "classics/", "crawler/")
    return {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.py")
            if not p.relative_to(ROOT).as_posix().startswith(skip)
            and NAMES.search(p.read_text(encoding="utf-8", errors="replace"))}


def test_only_the_audited_modules_name_the_portable_save() -> None:
    """A new reader of ``tlspec-portable`` must join the audit, with its own fallback test."""

    assert _naming() == PORTABLE_AWARE


@pytest.mark.parametrize("module", LEDGER_KEYED)
def test_a_ledger_keyed_reader_never_needs_the_portable_save(module: str) -> None:
    assert (ROOT / module).is_file(), module
    assert not NAMES.search((ROOT / module).read_text(encoding="utf-8"))


def test_the_site_export_offers_a_refused_trace_save_by_its_ledger_sha() -> None:
    from menagerie.pipeline import publish

    sha = "ab" * 32
    row = SimpleNamespace(trace=sha, runnable="refused:unsupported_tensor_constant")
    got = publish._tlspec(row, SimpleNamespace(blob_bytes={sha: 1_829_589_099}))
    assert got == {"url": f"/export/assets/{sha}.tlspec", "sha256": sha,
                   "bytes": 1_829_589_099, "runnable": False}
