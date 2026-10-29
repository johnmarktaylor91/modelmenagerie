"""P5 fable-F9 — the width re-arm checklist is committed, never comment-only.

RED against bd5c7837b: the three SMOKE width-gated skips are disclosed through
the four-part ``smoke_execution_waivers.json`` registry, but every NON-smoke
width-gated skip (eleven ``skipif(METADATA_BATCH_MAX < N)`` nodes across six
files) and the two width-conditioned golden-corpus retirements
(``metadata_mixed_batch``, ``metadata_mixed_disputed``) existed only as
comments -- invisible to the execution-floor census and to any future width
re-arm sweep. This module pins the committed ``width_rearm_inventory.json``
against a live scan of the test sources and against the calibrated production
width, so that:

* bumping ``METADATA_BATCH_MAX`` refuses here until the inventory is walked
  (every gated skip re-runs, every retired corpus class re-records) in the
  same reviewed change;
* adding a width-gated skip without committing it to the inventory refuses;
* silently deleting a width-gated test no longer leaves the inventory stale.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from menagerie.crawler.constants import METADATA_BATCH_MAX

pytestmark = pytest.mark.smoke

_TESTS_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_ROOT.parents[2]
_INVENTORY_PATH = _TESTS_ROOT / "width_rearm_inventory.json"

_WIDTH_GATE_PATTERN = re.compile(r"METADATA_BATCH_MAX\s*<\s*(\d+)")


def _scan_width_gated_skips() -> dict[tuple[str, int], int]:
    """Recompute (file, threshold) -> count from live test source bytes.

    A gate is a ``METADATA_BATCH_MAX < N`` comparison within three lines of a
    ``skipif`` marker -- exactly the decorator shape every width-gated node
    uses. Asserts and comments never match (no ``skipif`` context).
    """

    found: dict[tuple[str, int], int] = {}
    for root in (_TESTS_ROOT, _REPO_ROOT / "tests" / "crawler"):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("test_*.py")):
            lines = path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                match = _WIDTH_GATE_PATTERN.search(line)
                if match is None:
                    continue
                context = "\n".join(lines[max(0, index - 3) : index + 1])
                if "skipif" not in context:
                    continue
                key = (path.relative_to(_REPO_ROOT).as_posix(), int(match.group(1)))
                found[key] = found.get(key, 0) + 1
    return found


def test_width_gated_skips_match_the_committed_inventory() -> None:
    """Every width-gated skip is committed; every committed row still exists."""

    inventory = json.loads(_INVENTORY_PATH.read_text(encoding="utf-8"))
    committed = {
        (row["file"], int(row["threshold"])): int(row["count"])
        for row in inventory["width_gated_skips"]
    }
    live = _scan_width_gated_skips()
    assert live == committed, (
        "width-gated skips drifted from width_rearm_inventory.json -- a new, "
        "moved, or removed skipif(METADATA_BATCH_MAX < N) must update the "
        f"committed inventory in the same reviewed change (live={live!r})"
    )


def test_production_width_pin_forces_the_rearm_walk() -> None:
    """A width bump refuses until the whole inventory is walked and updated."""

    inventory = json.loads(_INVENTORY_PATH.read_text(encoding="utf-8"))
    assert METADATA_BATCH_MAX == inventory["production_width"], (
        "METADATA_BATCH_MAX moved: walk width_rearm_inventory.json in the same "
        "reviewed change -- every width-gated skip re-arms (smoke waiver rows "
        "must be pruned per their registry) and every retired corpus class "
        "re-records"
    )
    # The retirements the comment-only disclosure hid: pinned by name, and the
    # golden corpus tuple must actually exclude them at the retired width.
    retired = sorted(row["entry"] for row in inventory["corpus_retirements"])
    assert retired == ["metadata_mixed_batch", "metadata_mixed_disputed"]
    from menagerie.crawler.tests.test_checker_envelope_golden import CORPUS_ENTRIES

    if METADATA_BATCH_MAX == 1:
        assert not set(retired) & set(CORPUS_ENTRIES)
    else:  # pragma: no cover -- the re-armed width
        assert set(retired) <= set(CORPUS_ENTRIES)


def test_smoke_waived_width_nodes_are_a_subset_of_the_inventory_files() -> None:
    """The waiver registry and the re-arm inventory can never disagree."""

    waivers = json.loads(
        (_TESTS_ROOT / "smoke_execution_waivers.json").read_text(encoding="utf-8")
    )
    inventory = json.loads(_INVENTORY_PATH.read_text(encoding="utf-8"))
    inventory_files = {row["file"] for row in inventory["width_gated_skips"]}
    for row in waivers["waivers"]:
        if row["reason_code"] != "no-consumer-at-production-width":
            continue
        file_part = row["node_id"].split("::", 1)[0]
        assert file_part in inventory_files, (
            f"waived width node {row['node_id']} is not covered by "
            "width_rearm_inventory.json"
        )
