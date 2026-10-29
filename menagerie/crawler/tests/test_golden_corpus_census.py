"""Golden-corpus membership census (rule-2 round 3, W-7 discipline, minimal form).

The smoke inventory floor pins TESTS against silent substitution; nothing
pinned the golden transcript CORPUS the calibration receipt adjudicates over.
This census is the minimal membership floor: the exact committed entry set,
each entry's gate kind, recorded checker model, and the stable_ids it asks.
Any removal, rename, item swap, or checker-model drift fails here until the
change is made in the same reviewed commit that updates this table -- exactly
the anti-substitution shape of ``smoke_inventory_floor.json``, applied to the
corpus. (The full W-7 golden manifest with a retirement census remains an
open consensus item; this floor is deliberately its smallest honest core.)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_CORPUS_DIR = Path(__file__).resolve().parent / "golden_checker_transcripts"

#: The committed corpus: name -> (gate kind, recorded checker model, asked ids).
#: Additions are welcome (additive re-harvests are the discipline); removals
#: and substitutions must edit this table in the same reviewed change.
_EXPECTED_MEMBERSHIP: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "metadata_accept": ("metadata_batch", "gpt-5.6-sol", ("m10470",)),
    "metadata_accept_lineage_grounded": ("metadata_batch", "gpt-5.6-sol", ("m10949",)),
    "metadata_refusal": ("metadata_batch", "gpt-5.6-sol", ("m10470",)),
    "metadata_rung_adverse": ("metadata_batch", "gpt-5.6-sol", ("m8189",)),
    "metadata_component_drift": ("metadata_batch", "gpt-5.6-sol", ("m2247",)),
    "fidelity_accept": ("fidelity", "gpt-5.6-sol", ("m10759",)),
    "fidelity_refusal": ("fidelity", "gpt-5.6-sol", ("m3353",)),
    "terminal_accept": ("terminal_disposition", "gpt-5.6-sol", ("m4413",)),
    "terminal_refusal": ("terminal_disposition", "gpt-5.6-sol", ("m2991",)),
    "terminal_disputed": ("terminal_disposition", "gpt-5.6-sol", ("m10759",)),
}


@pytest.mark.smoke
def test_committed_corpus_membership_is_exactly_the_censused_set() -> None:
    """Every committed entry is censused; every censused entry is committed."""

    committed = {path.stem for path in _CORPUS_DIR.glob("*.json")}
    assert committed == set(_EXPECTED_MEMBERSHIP), (
        "golden corpus membership drifted from the census; update the census "
        "table in the same reviewed change that adds/retires an entry"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("name", sorted(_EXPECTED_MEMBERSHIP))
def test_censused_entry_identity(name: str) -> None:
    """Each entry's kind, recorded model, and asked stable_ids match the census."""

    kind, model, stable_ids = _EXPECTED_MEMBERSHIP[name]
    entry = json.loads((_CORPUS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    assert entry["kind"] == kind
    assert entry["checker_model"] == model
    asked = tuple(
        str(item.get("stable_id")) for item in entry["request"]["items"]
    )
    assert asked == stable_ids
