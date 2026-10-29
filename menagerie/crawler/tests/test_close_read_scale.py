"""The close/licensing read path is O(tail), not O(history) x O(#closes) (r2 T4).

BL-7 closed the O(campaign-history) read class on the four core fact ledgers and
left permanent tripwires. BL-5 / Track A then re-opened the same class one layer
up, in the SAME wave, on the close/licensing spine -- and nothing measured it,
because no suite had a campaign with more than one close in it.

RED, measured on this file's own ladder rig before the fix -- models history H
rows, N closed rungs, ONE staging-gate evaluation, counting ledger rows DECODED
(chain-link verifications not yet counted):

    ledger rows decoded  =  (N + 6) x H  +  O(N)

    H=1000  N=2  ->   8088 rows decoded
    H=1000  N=4  ->  10176
    H=1000  N=8  ->  14392
    H=1000  N=12 -> 18708

The product term came from four places, all of them re-derivations of things
that had not changed:

* ``ledger_chain_refusals`` resolved ONE committed chain head per close row per
  census source -- 15 sources x N closes whole-ledger decodes and whole-history
  chain verifications -- and then re-verified the whole close ledger once per
  registration to check its licensing anchor;
* the close ledger itself was decoded, per-row re-validated (a canonical
  re-serialization plus two digests each), and chain-verified on every one of the
  dozen consultations ONE licensing decision makes of it;
* ``model_head_citations`` folded the whole models history TWICE per call (a
  ``scan_jsonl`` for the head selection and a ``source_rows`` re-parse for row
  indices), and ``award_transition_census`` a third time, for every close; and
* ``_accounting_close_for`` re-read and re-hashed a whole partition proof per
  CARRIED member per candidate ancestor close, copying the proof's O(#members)
  member table to look up one key.

Survivable at step-200. A hard wall at the 500 rung and an OOM at 20k. GREEN, on
the same rig and the same metric:

    ledger rows decoded  =  2 x H  +  17 x N  +  9

No product term at all: the closes contribute a constant each, and the campaign's
history is folded once per observed ledger state.

The census the tests below actually run is stricter: it ALSO counts chain-link
verifications, because a read path that stops decoding history but still
re-verifies its whole chain per close has not been fixed. On that metric the
fixed tree measures ``3 x H + 29 x N``, and de-optimizing the single held
per-source chain derivation back to a per-lookup one immediately reads 8 x H at
four rungs -- which is how these ceilings were shown to have teeth rather than
being chosen to pass.

This is a PERFORMANCE-ONLY change, and section A is the proof: every licensing
digest a four-rung ladder commits -- close content addresses, sealed
partition-proof addresses, the predecessor spine, the persisted Track A links,
the verified close-chain heads, all 15 census source chain heads, the three
meta-ledger heads, and every registration's licensing anchor -- is asserted
against literals CAPTURED FROM THE STRENGTHENED TREE. Section D pins the fail-closed
half: held state is served only over bytes that re-hash, so a rewritten prefix
re-derives from zero and a corrupt ledger refuses on every call.

Round 3 measured what round 2 had not, on the same rig, and found the fix half
done. Rows decoded were bounded; BYTES and MEMORY were not:

    staging gate, ledger BYTES read, H=800:   N=2 -> 4,983,800   N=12 -> 10,399,695
                                              => 541,590 bytes PER PRIOR CLOSE
    staging gate, ledger BYTES read, N=4:     H=400 -> 3,384,557 H=1200 -> 8,712,750
                                              => ~15 whole-history passes per gate
    staging gate, PEAK resident:              ~12.5x the models ledger, per gate

because every held derivation re-hashed its WHOLE prefix on every call -- the
decode was incremental, the PROOF was not -- and the Track A chain derivation
materialized the campaign's entire history as decoded rows (~2.65x its bytes) to
derive a projection consumed only as heads. GREEN, same rig, same metric:

    BYTES per prior close:    541,590  ->  53,860   (and FLAT in history)
    whole-history passes:          15  ->       5   (and FLAT in #closes)
    models ledger bytes read:  1.79 MB at N=2, 1.81 MB at N=12 -- no product term

A prefix proof can never cost less than the bytes it covers, so what sections F
and G bound is the number of TIMES one decision performs one: inside a
:func:`menagerie.crawler.close_reads.read_epoch` an unchanged extent is proven
once and reused, and section F pins that repeating a chain derivation 24 times
inside one epoch reads ZERO further bytes while the same 24 calls outside one
read the whole ledger 24 times.

Section map:
  A -- digest equality: the strengthened licensing digests, byte-for-byte.
  B -- the cost bound: independent of #closes, bounded in history.
  C -- the 20k ladder projection.
  D -- fail-closed: held state never launders changed or corrupt bytes.
  E -- the structural census: no whole-history materializer on the read path.
  F -- the BYTE cost bound, and the epoch that makes a repeat read free.
  G -- peak resident memory, and the held set's real byte budget.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import pytest

import menagerie.crawler.close_reads as close_reads_module
from menagerie.crawler.close_object import CloseObjectError, close_object_path
from menagerie.crawler.ledger_chain import (
    LedgerChainError,
    chain_state,
    decode_ledger,
    ledger_chain_refusals,
)
from menagerie.crawler.leg_generation import source_row_count
from menagerie.crawler.tests.close_scale_support import (
    big_models_ledger,
    build_ladder,
    deep_size,
    derive_tip_proof,
    install_read_census,
    ladder_digests,
    measure_read_cost,
    pad_models_history,
    stage_next_rung,
)

_CRAWLER_ROOT = Path(close_reads_module.__file__).parent

# NOTHING here is hidden from collection. Round-2 set ``__test__ = False`` on this
# module and had one already-locked node call a wrapper, reasoning that widening
# the node-ID lock was out of scope -- which left the entire T4 scale contract
# INVISIBLE to the anti-substitution lock: deleting every clause below kept the
# lock green, because the lock only ever knew about the one wrapper's caller
# (round-3 R3F-5). These are ordinary collected nodes now, so removing one is a
# lock failure that names it, and the committed lock is regenerated in the same
# reviewed change that lands them.


# --------------------------------------------------------------------------- #
# A -- digest equality (strengthened FB-8 census contract)                     #
# --------------------------------------------------------------------------- #

#: Every licensing digest a four-rung ladder commits after FB-8 made census
#: instrumentation explicit and covered between-window appends. A read-path
#: rewrite that changes any of them is not merely a performance change.
# Regenerated at the Wave-2 batch assembly: the DX-23 close-citation key
# (leg_citations) joins the hashed partition-proof body, so every freshly
# built close's digest moved by design (D-19 precedent: close-object
# validation tolerates new keys; historical closes are untouched).
# Regenerated again for RULING W7 (PD-1): freshly minted proofs carry the v4
# era schema (partition-proof.v4), so every fresh ladder digest moved by
# design; sealed v3 history re-derives byte-identically under its own era.
# meta.registry regenerated for W16/S-2 (certificate-kind consumer split):
# registration rows durably carry the staging-declared ``rung_class`` and the
# shared fixture door (rt3_support.stage) declares ``successor_rung``, so the
# registry chain head moved by design. Every close, proof, chain-link, and
# other chain-head literal is byte-identical to the PD-1 regeneration.
_STRENGTHENED_LADDER_DIGESTS: dict[str, Any] = {   'chain_heads': {   'artifacts.flat': 'sha256:820a5dd1f55ae78be18cc586d8414fba66c506852f278c23c4c7294687dcd1fd',
                       'artifacts.shard': 'sha256:61682f64dc1c9e8f84c13aeb56059cb2ec0c1f23bd5bebebcb95d7c9d8805047',
                       'attempts.flat': 'sha256:83698d8f931872a843de8844a1472ceadc6189674a56d09ac6d364f74d90ee15',
                       'attempts.local': 'sha256:d3f6ccd5ad0c711982820d033e13cb5179de5298d3ee96b0f203d19f20fa8427',
                       'gates.flat': 'sha256:9e8e58d5db7e1f03e91ee4d1ccd41bf2c49f65e92a9fce0f875b712f216522e4',
                       'gates.shard': 'sha256:701da37f96b75abeb35e8a1222b407b9fd200891011079008863fa05e91a90a5',
                       'meta.closes': 'sha256:37d78936143e9dd28b8fd015be123caab538d61e80b359f6101bc42b6bb6c310',
                       'meta.leg-events': 'sha256:05a8f6a91ba6fd0d368a4446ea392ec190c138afce2d0441e993c0a28290978a',
                       'meta.registry': 'sha256:61fea0ceaeb9b73a4ee6a512a682e2959be86afb01cecd4620a1f825c8080911',
                       'models.flat': 'sha256:4e33b96bdbb9e669a40acfc42efe51392bd9c3d834ef9d2fc4bf04cb2665ffa4',
                       'models.shard': 'sha256:19fa18b0995fefb600e9cb9e90e990117a5bc77450c068b56e3345ddbe51e737',
                       'park.grant': 'sha256:e9be815dca4c9c6a6feddf2fbffc6014c8d633ae47ae7c88a3eae3e9f79b40e1',
                       'park.refusal': 'sha256:71ea2057291acabfb70f6e68ec5a5f7b9adcc6db11d01ea131b918a302ba1fe2',
                       'valve.capacity': 'sha256:be7a7aa08c37ea35ca6d77cdd194a2e335956a17195d18a23353d31d4bf14a2e',
                       'valve.checker-lane': 'sha256:214647a9501a4d63dd5bce8367d926274df33f430947bfcdea5f1f3a8fd742d8',
                       'valve.coverage': 'sha256:3cd51ce2b5defa94d32022d5354ca5b1df0e84bb25a7a8ce2f7ce8931b7eb4b7',
                       'valve.disk-floor': 'sha256:a42d3180d91d491d17c71691b1864d93059ce0b7946c13332b6f815430041c32',
                       'valve.rebuild': 'sha256:bad9b92ab20f5359740251a344791ebadd0dd679b0246e3a1b5290fc273025aa',
                       'valve.scheduler-quarantine': 'sha256:6c3ed66f2e858f7a135d6ddf9fa0e7c890c605836ebecea10b58947ea139d249'},
    'chain_refusals': [],
    'closes': [   {   'chain': {   'prev': 'sha256:271af3d2b928d614ec478cbe7adf5ba006c1579065dccee147a0cadb7a9a4580',
                                   'seq': 0,
                                   'sha256': 'sha256:76f08370300e1e820aab24e74ed115e44e7a19d9af5b353f627a1df2218d957c'},
                      'close_chain_head': 'sha256:76f08370300e1e820aab24e74ed115e44e7a19d9af5b353f627a1df2218d957c',
                      'close_sha256': 'sha256:bbc649993fc8dc4945fdc9d0574d30a3d28c97675d66886a821f9701f9d8ddc6',
                      'generation_id': 'sha256:d056604aaddbb4a66a0f5536cb946d8e500c4d3ce241e500a8fc098c3ae521a8',
                      'partition_proof_sha256': 'sha256:873b1746f2b20e52685063d8b3e72c3b73728c037993d647d5a4c0927fdf45c4',
                      'predecessor_close_sha256': None},
                  {   'chain': {   'prev': 'sha256:76f08370300e1e820aab24e74ed115e44e7a19d9af5b353f627a1df2218d957c',
                                   'seq': 1,
                                   'sha256': 'sha256:df51171339c768316eb9f2f293373d8222189822ab8ecf4bdd841595b4adb63c'},
                      'close_chain_head': 'sha256:df51171339c768316eb9f2f293373d8222189822ab8ecf4bdd841595b4adb63c',
                      'close_sha256': 'sha256:dd85383625664b1037c85e120e59184da994b302b8afac6c073f4b3955ccd205',
                      'generation_id': 'sha256:0a5c85550674c1c0987da14441435f093cdd79fb81d6ec278ddcd48beb842806',
                      'partition_proof_sha256': 'sha256:2e57625b2e4b0986ab966482d9cb40f7b7d3bffa26dc19a84962111a060fa90a',
                      'predecessor_close_sha256': 'sha256:bbc649993fc8dc4945fdc9d0574d30a3d28c97675d66886a821f9701f9d8ddc6'},
                  {   'chain': {   'prev': 'sha256:df51171339c768316eb9f2f293373d8222189822ab8ecf4bdd841595b4adb63c',
                                   'seq': 2,
                                   'sha256': 'sha256:37e64637df87c2fac1f4bd7b6593e14d4ef7b688e68fd339dfd92264171f79c3'},
                      'close_chain_head': 'sha256:37e64637df87c2fac1f4bd7b6593e14d4ef7b688e68fd339dfd92264171f79c3',
                      'close_sha256': 'sha256:dd8bccc737a779ac3cc20987f38202723ccfc74d4209b89f4eca6497a956660f',
                      'generation_id': 'sha256:21041e303bf0194dccb0e42f3da378ad48f558d281fdc3c343b1d94842b2ff78',
                      'partition_proof_sha256': 'sha256:9cc357300490a9ebf2007d471ee85b0c2fd22c14df9d3ad5af2f2acbe52cfaa1',
                      'predecessor_close_sha256': 'sha256:dd85383625664b1037c85e120e59184da994b302b8afac6c073f4b3955ccd205'},
                  {   'chain': {   'prev': 'sha256:37e64637df87c2fac1f4bd7b6593e14d4ef7b688e68fd339dfd92264171f79c3',
                                   'seq': 3,
                                   'sha256': 'sha256:37d78936143e9dd28b8fd015be123caab538d61e80b359f6101bc42b6bb6c310'},
                      'close_chain_head': 'sha256:37d78936143e9dd28b8fd015be123caab538d61e80b359f6101bc42b6bb6c310',
                      'close_sha256': 'sha256:2fde42793bf9f48b14be54ae6602455f210c51ab8d0f54dfcfdb39f88e70516d',
                      'generation_id': 'sha256:c1361aa1e3b1737a63726b9e34cd96ada8dac542152c2173bb2134f0ceb8cac3',
                      'partition_proof_sha256': 'sha256:4bcfe892c92055c4f37536773bb039a9417c1700eddc46046c6efb6bfcf83604',
                      'predecessor_close_sha256': 'sha256:dd8bccc737a779ac3cc20987f38202723ccfc74d4209b89f4eca6497a956660f'}],
    'registrations': [   {   'closes_chain_head': None,
                             'licensed_by_close_sha256': None,
                             'snapshot_id': 'intake-dfe43c7efd44d00ae5a5',
                             'snapshot_sha256': 'sha256:dfe43c7efd44d00ae5a568e0457ce9a74eef0f759cd24201fa1722982b51e0f4'},
                         {   'closes_chain_head': 'sha256:76f08370300e1e820aab24e74ed115e44e7a19d9af5b353f627a1df2218d957c',
                             'licensed_by_close_sha256': 'sha256:bbc649993fc8dc4945fdc9d0574d30a3d28c97675d66886a821f9701f9d8ddc6',
                             'snapshot_id': 'intake-61846371fb8d5064e69e',
                             'snapshot_sha256': 'sha256:61846371fb8d5064e69e418d4a968a3e3255fe04029e980c67a24a8f28654dee'},
                         {   'closes_chain_head': 'sha256:df51171339c768316eb9f2f293373d8222189822ab8ecf4bdd841595b4adb63c',
                             'licensed_by_close_sha256': 'sha256:dd85383625664b1037c85e120e59184da994b302b8afac6c073f4b3955ccd205',
                             'snapshot_id': 'intake-e247e14dcfe19325c6e1',
                             'snapshot_sha256': 'sha256:e247e14dcfe19325c6e191102523461b77ebde9c9221047fced7e44748f04eb3'},
                         {   'closes_chain_head': 'sha256:37e64637df87c2fac1f4bd7b6593e14d4ef7b688e68fd339dfd92264171f79c3',
                             'licensed_by_close_sha256': 'sha256:dd8bccc737a779ac3cc20987f38202723ccfc74d4209b89f4eca6497a956660f',
                             'snapshot_id': 'intake-8862ddd93e0968e4bf2c',
                             'snapshot_sha256': 'sha256:8862ddd93e0968e4bf2c2ba43728650b0dd94fcaca8a7fa933ea5119d7db9f2f'}]}

#: The tip rung's sealed proof address, recomputed through the production
#: derivation. Equal to the fourth close's bound proof above BY DEFINITION --
#: that equality IS the licensing gate -- and asserted separately so a
#: derivation that silently stopped reading some evidence fails here too.
_STRENGTHENED_TIP_PROOF_SHA256 = (
    'sha256:4bcfe892c92055c4f37536773bb039a9417c1700eddc46046c6efb6bfcf83604'
)


@pytest.mark.heavy
def test_ladder_licensing_digests_equal_the_strengthened_tree(tmp_path: Path) -> None:
    """Every strengthened licensing digest stays fixed, bit for bit.

    The semantics half of round-2 T4. A four-rung ladder is built and closed
    through the production path, and every content address, chain link, chain
    head, and registry anchor it commits is compared against literals captured
    from the strengthened FB-8 tree.
    """

    ladder = build_ladder(tmp_path, rungs=4)
    assert ladder_digests(ladder.records) == _STRENGTHENED_LADDER_DIGESTS


@pytest.mark.heavy
def test_tip_proof_recomputes_to_the_strengthened_content_address(tmp_path: Path) -> None:
    """The gate's recomputation seals the strengthened proof digest.

    ``proof_refusals`` licenses a boundary by re-deriving the sealed proof over
    the pinned bytes and demanding hash equality with the producer's. Running
    that derivation directly proves the incremental reads feed it exactly the
    evidence the whole-history reads did.
    """

    ladder = build_ladder(tmp_path, rungs=4)
    derived = derive_tip_proof(ladder)
    assert derived["partition_proof_sha256"] == _STRENGTHENED_TIP_PROOF_SHA256


@pytest.mark.heavy
def test_ladder_tip_still_licenses_its_successor(tmp_path: Path) -> None:
    """The whole staging door still accepts an honest ladder, refusals empty.

    A positive control beside the cost pins: a bound that were achieved by
    reading LESS evidence would show up here as a refusal.
    """

    ladder = build_ladder(tmp_path, rungs=4)
    assert ledger_chain_refusals(ladder.records) == ()
    staged = stage_next_rung(ladder)
    assert staged.registration.licensed_by_close_sha256 == (
        _STRENGTHENED_LADDER_DIGESTS["closes"][-1]["close_sha256"]
    )


# --------------------------------------------------------------------------- #
# B -- the cost bound                                                         #
# --------------------------------------------------------------------------- #

#: Rows a single close may contribute to one evaluation. A close row itself, its
#: leg-open/leg-run/leg-plan events, and its registry row are all O(1) per close
#: and legitimately re-read; the ceiling exists so that number cannot quietly
#: become "and its share of history".
_ROWS_PER_CLOSE_CEILING = 40

#: How many times one evaluation may fold the campaign's models history. Three
#: is the honest cold floor on this census -- the head/award projection, the
#: Track A chain decode over the same ledger, and that chain's per-row link
#: verification -- and the fixed tree measures exactly 3.0. The pre-fix path
#: measures ``#closes + 6``, and reverting the held per-source chain alone
#: measures 8 at four rungs, so the one-pass headroom below is a margin against
#: a fourth legitimate fold, never enough to hide a per-close term.
_HISTORY_PASS_CEILING = 4


def _staging_gate_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> int:
    """Return ledger rows decoded by ONE cold staging-gate evaluation."""

    ladder = build_ladder(tmp_path, **kwargs)
    census = install_read_census(monkeypatch)
    stage_next_rung(ladder)
    return census.rows


def _close_derivation_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> int:
    """Return ledger rows decoded by ONE cold sealed-proof derivation."""

    ladder = build_ladder(tmp_path, **kwargs)
    census = install_read_census(monkeypatch)
    derive_tip_proof(ladder)
    return census.rows


@pytest.mark.heavy
def test_staging_gate_cost_does_not_grow_with_the_number_of_prior_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ten more closes may not add ten more walks of the campaign's history.

    THE round-2 T4 tripwire. Pre-fix, at H=1000, going from 2 closes to 12 cost
    an extra 10620 decoded rows -- 10 extra whole-history walks. The ceiling
    below is ``rows per close x closes added``, which no product term can fit
    under.
    """

    history = 800
    small = _staging_gate_rows(tmp_path / "small", monkeypatch, rungs=2, history_rows=history)
    large = _staging_gate_rows(tmp_path / "large", monkeypatch, rungs=12, history_rows=history)
    added_closes = 12 - 2
    assert large - small <= _ROWS_PER_CLOSE_CEILING * added_closes, (
        f"one staging gate decoded {small} rows over 2 closes and {large} over 12: "
        f"{large - small} extra rows for {added_closes} extra closes exceeds the "
        f"{_ROWS_PER_CLOSE_CEILING}-rows-per-close ceiling, so the read path is "
        "O(history) x O(#closes) again"
    )
    assert large - small < history, (
        "the extra closes cost at least a whole extra pass over campaign history"
    )


@pytest.mark.heavy
def test_close_derivation_cost_does_not_grow_with_the_number_of_prior_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The producer side is bounded too: sealing a proof is not O(#closes)."""

    history = 400
    small = _close_derivation_rows(tmp_path / "small", monkeypatch, rungs=2, history_rows=history)
    large = _close_derivation_rows(tmp_path / "large", monkeypatch, rungs=12, history_rows=history)
    assert large - small <= _ROWS_PER_CLOSE_CEILING * (12 - 2), (
        f"one close derivation decoded {small} rows over 2 closes and {large} over 12"
    )


@pytest.mark.heavy
def test_one_derivation_reads_each_ancestor_proof_at_most_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The carry resolver reads a given ancestor proof once, not once per member.

    Pre-fix, ``_accounting_close_for`` re-read and re-derived the content address
    of a whole partition proof for EVERY carried member against EVERY candidate
    ancestor close -- an O(#members) canonical re-serialization inside an
    O(#members) loop, i.e. O(#members**2) work for one rung. The proof read epoch
    proves each address once per derivation; this pins that the count is bounded
    by the number of DISTINCT proofs, never by the membership.
    """

    ladder = build_ladder(tmp_path, rungs=6)
    reads: list[str] = []
    real_read_bytes = Path.read_bytes

    def counting(self: Path) -> bytes:
        if "leg-proofs" in self.parts:
            reads.append(self.name)
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    close_reads_module.reset_incremental_reads()
    derive_tip_proof(ladder)
    assert len(reads) == len(set(reads)), (
        f"one derivation read {len(reads)} proof files but only "
        f"{len(set(reads))} distinct ones: {sorted(reads)}"
    )
    assert len(set(reads)) <= ladder.rungs, (
        f"one derivation read {len(set(reads))} distinct proofs for a "
        f"{ladder.rungs}-rung lineage"
    )


@pytest.mark.heavy
def test_staging_gate_folds_campaign_history_a_bounded_number_of_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Doubling history may at most triple-count the rows it added.

    The slope of rows-decoded against history rows IS the number of whole-history
    passes one evaluation makes. Pre-fix it was ``#closes + 6``; the ceiling
    below pins it to a constant.
    """

    rungs = 4
    lean = _staging_gate_rows(tmp_path / "lean", monkeypatch, rungs=rungs, history_rows=400)
    fat = _staging_gate_rows(tmp_path / "fat", monkeypatch, rungs=rungs, history_rows=1200)
    passes = (fat - lean) / (1200 - 400)
    assert passes <= _HISTORY_PASS_CEILING, (
        f"one staging gate folds campaign history {passes:.2f} times "
        f"({lean} rows at H=400, {fat} at H=1200); the ceiling is "
        f"{_HISTORY_PASS_CEILING}"
    )


@pytest.mark.heavy
def test_warm_process_rereads_only_the_appended_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second evaluation in one process re-decodes nothing that has not moved.

    The driver closes many legs in ONE process, so the generation-bound half of
    the fix has to show up as "the tail, and only the tail": an unchanged tree
    costs nothing, and one appended row costs one row.
    """

    ladder = build_ladder(tmp_path, rungs=4, history_rows=400)
    cold = install_read_census(monkeypatch)
    derive_tip_proof(ladder)
    assert cold.rows > 400, "the cold derivation should have folded history once"

    warm = install_read_census(monkeypatch, cold=False)
    derive_tip_proof(ladder)
    # The only rows a warm derivation may still read are the generation
    # WINDOW's own -- the award-transition census reads ``[floor, close)`` off
    # disk by byte offset every time, which is the whole point of binding it to
    # the window. Nothing pre-window may be re-read.
    assert warm.rows <= _ROWS_PER_CLOSE_CEILING, (
        f"a warm re-derivation over unchanged bytes decoded {warm.rows} rows: "
        f"{warm.by_decoder}"
    )
    assert set(warm.by_decoder) <= {"leg_generation.source_row_window"}, (
        f"a warm re-derivation re-entered a history-wide decoder: {warm.by_decoder}"
    )

    pad_models_history(ladder.records, 1, prefix="tail")
    tail = install_read_census(monkeypatch, cold=False)
    derive_tip_proof(ladder)
    assert 0 < tail.rows <= _ROWS_PER_CLOSE_CEILING, (
        f"one appended models row cost {tail.rows} decoded rows: {tail.by_decoder}"
    )


# --------------------------------------------------------------------------- #
# C -- the 20k ladder projection                                              #
# --------------------------------------------------------------------------- #

#: The ladder's terminal rung and a generous close count for it. The 20k rung is
#: the target the whole crawler exists to reach; 40 closes is more than the
#: 200 -> 20k ladder plans (one per rung, plus re-closes).
_LADDER_TARGET_HISTORY = 20_000
_LADDER_TARGET_CLOSES = 40

#: Rows one staging gate may decode at the terminal rung. Derived, not guessed:
#: ``_HISTORY_PASS_CEILING x 20000 + _ROWS_PER_CLOSE_CEILING x 40``.
_LADDER_TARGET_CEILING = (
    _HISTORY_PASS_CEILING * _LADDER_TARGET_HISTORY
    + _ROWS_PER_CLOSE_CEILING * _LADDER_TARGET_CLOSES
)


@pytest.mark.heavy
def test_twenty_thousand_rung_projection_stays_under_its_derived_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fit the measured cost model and project it to the 20k rung.

    A real 20k-model, 40-close ladder is far too slow for any test tier, so the
    bound is established the way a complexity claim has to be: measure the two
    slopes (per history row, per close) on tractable ladders, then evaluate the
    fitted model at the target. The pre-fix model -- whose per-close slope was
    itself proportional to history -- projects to 46 x 20000 = 920,000 rows per
    gate; this asserts the fitted model has no such term.
    """

    lean = _staging_gate_rows(tmp_path / "lean", monkeypatch, rungs=4, history_rows=400)
    fat = _staging_gate_rows(tmp_path / "fat", monkeypatch, rungs=4, history_rows=1200)
    short = _staging_gate_rows(tmp_path / "short", monkeypatch, rungs=2, history_rows=400)
    tall = _staging_gate_rows(tmp_path / "tall", monkeypatch, rungs=12, history_rows=400)

    per_history_row = (fat - lean) / (1200 - 400)
    per_close = (tall - short) / (12 - 2)
    projected = per_history_row * _LADDER_TARGET_HISTORY + per_close * _LADDER_TARGET_CLOSES
    assert projected <= _LADDER_TARGET_CEILING, (
        f"the fitted model ({per_history_row:.3f} rows per history row, "
        f"{per_close:.3f} rows per close) projects {projected:.0f} decoded rows for "
        f"one staging gate at the {_LADDER_TARGET_HISTORY}-model rung with "
        f"{_LADDER_TARGET_CLOSES} closes; the derived ceiling is "
        f"{_LADDER_TARGET_CEILING}"
    )
    # The per-close slope must be a CONSTANT, not history's shadow: measure it
    # again at three times the history and require it not to follow.
    short_fat = _staging_gate_rows(tmp_path / "sf", monkeypatch, rungs=2, history_rows=1200)
    tall_fat = _staging_gate_rows(tmp_path / "tf", monkeypatch, rungs=12, history_rows=1200)
    per_close_fat = (tall_fat - short_fat) / (12 - 2)
    assert per_close_fat <= _ROWS_PER_CLOSE_CEILING, (
        f"the per-close cost rose to {per_close_fat:.3f} rows when history tripled: "
        "the product term is back"
    )


# --------------------------------------------------------------------------- #
# D -- fail-closed: held state never launders changed or corrupt bytes         #
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_held_derivation_refuses_a_rewritten_prefix(tmp_path: Path) -> None:
    """A GROWN ledger whose held prefix no longer hashes re-derives from zero.

    This is the tripwire half of the substrate. The fast path is served only for
    an identical physical extent; any growth re-hashes the whole held prefix, so
    a history rewritten under a live reader can never be served from held state
    -- it falls back to the cold derivation, which sees the rewrite.
    """

    path = tmp_path / "ledger.jsonl"
    rows = [{"a": 1}, {"a": 2}, {"a": 3}]
    path.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    close_reads_module.reset_incremental_reads()
    assert [row["a"] for row in decode_ledger(path)] == [1, 2, 3]

    rewritten = [{"a": 1}, {"a": 99}, {"a": 3}, {"a": 4}]
    path.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rewritten))
    assert [row["a"] for row in decode_ledger(path)] == [1, 99, 3, 4], (
        "a rewritten prefix was served from held state"
    )


@pytest.mark.smoke
def test_held_derivation_refuses_a_shrunk_ledger(tmp_path: Path) -> None:
    """A truncated ledger re-derives; held rows can never outlive their bytes."""

    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    close_reads_module.reset_incremental_reads()
    assert len(decode_ledger(path)) == 3
    path.write_bytes(b'{"a": 1}\n')
    assert len(decode_ledger(path)) == 1


@pytest.mark.smoke
def test_a_corrupt_row_refuses_on_every_call_not_just_the_first(tmp_path: Path) -> None:
    """Nothing is memoized across a failure: corruption refuses every time.

    A derivation that raises stores nothing, so the second reader of a corrupt
    ledger cannot be handed a half-built projection or a stale clean one.
    """

    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"a": 1}\n')
    close_reads_module.reset_incremental_reads()
    assert len(decode_ledger(path)) == 1
    with path.open("ab") as handle:
        handle.write(b"{not json}\n")
    for _attempt in range(3):
        with pytest.raises(LedgerChainError):
            decode_ledger(path)


@pytest.mark.smoke
def test_close_ledger_corruption_refuses_on_every_call(tmp_path: Path) -> None:
    """The close ledger's held validation is equally unforgiving on replay."""

    from menagerie.crawler.close_object import close_ledger_rows

    path = tmp_path / "leg-closes.jsonl"
    path.write_bytes(b'{"schema_version": "nope"}\n')
    close_reads_module.reset_incremental_reads()
    for _attempt in range(3):
        with pytest.raises(CloseObjectError):
            close_ledger_rows(path)


@pytest.mark.smoke
def test_held_derivations_are_capped_so_a_long_run_cannot_accumulate(
    tmp_path: Path,
) -> None:
    """Held state is bounded: a driver that touches many ledgers evicts, not grows.

    The substrate lives for the lifetime of a driver process that closes many
    legs across many campaigns. A cache with no ceiling would be an unbounded
    leak, which is the opposite of the problem this fix exists to solve.
    """

    close_reads_module.reset_incremental_reads()
    for index in range(close_reads_module._MAX_HELD + 8):
        path = tmp_path / f"ledger-{index}.jsonl"
        path.write_bytes(b'{"a": 1}\n')
        decode_ledger(path)
    assert close_reads_module.held_derivation_count() == close_reads_module._MAX_HELD


@pytest.mark.smoke
def test_chain_state_survives_an_appended_row_incrementally(tmp_path: Path) -> None:
    """An extended chain equals the chain a cold reader derives over the same bytes."""

    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"row": "a"}\n{"row": "b"}\n')
    close_reads_module.reset_incremental_reads()
    warm_first = chain_state(path, scope="meta.test").head_sha256
    with path.open("ab") as handle:
        handle.write(b'{"row": "c"}\n')
    warm_extended = chain_state(path, scope="meta.test")
    close_reads_module.reset_incremental_reads()
    cold = chain_state(path, scope="meta.test")
    assert warm_extended == cold
    assert warm_first != cold.head_sha256


@pytest.mark.heavy
def test_held_reads_survive_a_torn_tail_and_still_recover_it(tmp_path: Path) -> None:
    """Torn-tail recovery is unchanged, and the held read sees the recovered bytes."""

    ladder = build_ladder(tmp_path, rungs=2)
    models = ladder.records / "models" / "current-shard.jsonl"
    clean_rows = source_row_count(models)
    close_reads_module.reset_incremental_reads()
    with (close_object_path(ladder.records)).open("ab") as handle:
        handle.write(b'{"torn": ')
    assert ledger_chain_refusals(ladder.records) == ()
    assert source_row_count(models) == clean_rows


# --------------------------------------------------------------------------- #
# E -- the structural census                                                   #
# --------------------------------------------------------------------------- #

#: The close/licensing spine's ledger modules. Every whole-file read of a
#: GROWING ledger inside them must be on the allowlist below.
_CLOSE_PATH_MODULES = (
    "close_object.py",
    "ledger_chain.py",
    "leg_generation.py",
    "partition_proof.py",
)

#: Whole-file readers the census forbids on a growing ledger.
_WHOLE_FILE_CALLS = frozenset({"read_bytes", "read_text", "scan_jsonl", "splitlines"})

#: The ONLY sanctioned whole-file reads on the close/licensing spine, each with
#: the reason it is bounded. Adding an entry is a reviewed decision about a
#: read that is either O(1) in campaign history or reached only by a crash.
_SANCTIONED_WHOLE_FILE_READS: dict[tuple[str, str], str] = {
    (
        "close_object.py",
        "_ledger_bytes",
    ): "torn-tail crash recovery: reached only when the final byte is not a newline",
    (
        "close_object.py",
        "_decode_close_lines",
    ): "the whole-bytes decoder the recovery prefix check is handed; not a read path",
    (
        "ledger_chain.py",
        "decode_payload",
    ): "the whole-bytes decoder the recovery prefix check is handed; not a read path",
    (
        "ledger_chain.py",
        "tail_recovery",
    ): "torn-tail crash recovery: reached only when the final byte is not a newline",
    (
        "leg_generation.py",
        "census_source_bytes",
    ): "torn-tail crash recovery for a census source; the streaming readers seek first",
    (
        "partition_proof.py",
        "write_partition_proof",
    ): "the content-addressed proof store: O(rung members), written once, never a ledger",
    (
        "partition_proof.py",
        "load_partition_proof",
    ): "the content-addressed proof store: O(rung members), held per observed extent",
}


def _enclosing_functions(tree: ast.AST) -> dict[ast.AST, str]:
    """Map every node to the name of the function that lexically encloses it."""

    owner: dict[ast.AST, str] = {}

    def walk(node: ast.AST, name: str) -> None:
        for child in ast.iter_child_nodes(node):
            child_name = (
                child.name
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                else name
            )
            owner[child] = child_name
            walk(child, child_name)

    walk(tree, "<module>")
    return owner


@pytest.mark.smoke
def test_no_unsanctioned_whole_file_ledger_read_on_the_close_path() -> None:
    """Whole-file ledger reads on the close spine are a CLOSED, reviewed set.

    The class-level prevention, modelled on the ``os.fsync`` barrier census: a
    read added next quarter that pulls a growing ledger into memory fails here
    the moment it is written, and the only way to pass is to route through the
    incremental substrate (or to justify a new allowlist entry in review).
    """

    observed: dict[tuple[str, str], list[str]] = {}
    for module in _CLOSE_PATH_MODULES:
        path = _CRAWLER_ROOT / module
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            name = (
                called.attr
                if isinstance(called, ast.Attribute)
                else called.id
                if isinstance(called, ast.Name)
                else None
            )
            if name not in _WHOLE_FILE_CALLS:
                continue
            observed.setdefault((module, owner.get(node, "<module>")), []).append(name)

    unsanctioned = {
        key: sorted(set(calls))
        for key, calls in observed.items()
        if key not in _SANCTIONED_WHOLE_FILE_READS
    }
    assert unsanctioned == {}, (
        "these close/licensing read sites pull a whole ledger into memory and are "
        "not on the reviewed allowlist -- route them through "
        f"menagerie.crawler.close_reads instead: {unsanctioned}"
    )
    stale = sorted(set(_SANCTIONED_WHOLE_FILE_READS) - set(observed))
    assert stale == [], (
        f"the whole-file allowlist names sites that no longer exist: {stale}"
    )


@pytest.mark.smoke
def test_no_production_caller_folds_all_of_history_through_source_rows() -> None:
    """``source_rows`` has no production caller: the close path is windowed.

    ``source_row_window`` reads a leg's own generation window by byte offset;
    ``source_rows`` is the O(campaign-history) reference spelling. A production
    caller of the latter is the O(history)-per-close class returning.
    """

    callers: list[str] = []
    for path in sorted(_CRAWLER_ROOT.rglob("*.py")):
        if "tests" in path.parts or path.name == "leg_generation.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called = node.func
                name = (
                    called.attr
                    if isinstance(called, ast.Attribute)
                    else called.id
                    if isinstance(called, ast.Name)
                    else None
                )
                if name == "source_rows":
                    callers.append(f"{path.name}:{owner.get(node, '<module>')}")
    assert callers == [], (
        f"these production sites fold all of campaign history: {sorted(set(callers))}"
    )


def test_a_committed_source_without_chain_head_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N3: omitting the committed Track A authority is a refusal, never a skip."""

    import menagerie.crawler.ledger_chain as ledger_chain_module
    import menagerie.crawler.leg_generation as leg_generation_module
    from menagerie.crawler.close_object import close_object_path
    from menagerie.crawler.rung_staging import rung_registry_path

    records_root = tmp_path / "records"
    source_path = records_root / "models" / "current-shard.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"")
    committed = {
        "close_kind": "leg",
        "campaign_id": "c1-mech",
        "close_sha256": "sha256:" + "a" * 64,
        "predecessor_close_sha256": None,
        "sources": {"models.shard": {"rows": 0, "ledger_sha256": "sha256:empty"}},
    }

    def decode(path: Path) -> tuple[dict[str, Any], ...]:
        """Return the one attacker-shaped committed close only for its ledger."""

        if path == close_object_path(records_root):
            return (committed,)
        if path == rung_registry_path(records_root):
            return ()
        return ()

    monkeypatch.setattr(ledger_chain_module, "decode_ledger", decode)
    monkeypatch.setattr(
        ledger_chain_module,
        "verified_chain",
        lambda _path, *, scope: ledger_chain_module.ChainState(
            rows=0, head_sha256=f"sha256:{scope}", heads=()
        ),
    )
    monkeypatch.setattr(ledger_chain_module, "_close_head_index", lambda _path: {})
    monkeypatch.setattr(
        leg_generation_module,
        "close_ledger_sources",
        lambda _records_root: {"models.shard": source_path},
    )

    findings = ledger_chain_module.ledger_chain_refusals(records_root)
    assert any("models.shard" in finding and "chain_head" in finding for finding in findings)


# --------------------------------------------------------------------------- #
# F -- the BYTE cost bound                                                     #
# --------------------------------------------------------------------------- #

#: BYTES one prior close may add to one evaluation. The close ledger, the
#: leg-event ledger, and the registry all grow with #closes and are legitimately
#: read a CONSTANT number of times each, which is what this per-close constant
#: pays for (the fixed tree measures ~54 KiB). What it cannot pay for is a share
#: of campaign HISTORY, which is why the same slope is re-measured at three times
#: the history below and required not to follow.
_BYTES_PER_CLOSE_CEILING = 128 * 1024

#: Whole models-ledger passes one cold evaluation may read. Five is the honest
#: floor on this metric -- the head/award projection, the row-span index, the
#: Track A chain, the close-identity digest, and the window-floor prefix proof,
#: each of which reads the ledger once and each of which is a DIFFERENT
#: derivation over the same bytes. The pre-fix path measured ~15 and grew with
#: #closes.
_HISTORY_BYTE_PASS_CEILING = 8

#: Repeats of one derivation inside a single epoch. Large enough that a per-call
#: whole-prefix re-hash is unmistakable: 24 passes over the ladder's ledger.
_EPOCH_REPEATS = 24


def _staging_gate_cost(tmp_path: Path, **kwargs: Any) -> tuple[Any, Any]:
    """Return the BYTE cost of ONE cold staging-gate evaluation, and its ladder."""

    ladder = build_ladder(tmp_path, **kwargs)
    cost = measure_read_cost(
        lambda: stage_next_rung(ladder), records=ladder.records, peak=False
    )
    return cost, ladder


@pytest.mark.heavy
def test_staging_gate_bytes_do_not_grow_with_the_number_of_prior_closes(
    tmp_path: Path,
) -> None:
    """Ten more closes may not add ten more whole-history HASHES.

    THE round-3 B1 tripwire. Round-2's fix stopped re-DECODING history per close
    but left every held derivation re-hashing its whole prefix per call, so at
    H=800 going from 2 closes to 12 still read an extra 5.4 MB -- ten more walks
    of the campaign's bytes, differing from the pre-round-2 defect only in which
    half of the read it paid for.
    """

    history = 800
    small, _ = _staging_gate_cost(tmp_path / "small", rungs=2, history_rows=history)
    large, _ = _staging_gate_cost(tmp_path / "large", rungs=12, history_rows=history)
    added = 12 - 2
    per_close = (large.bytes_read - small.bytes_read) / added
    assert per_close <= _BYTES_PER_CLOSE_CEILING, (
        f"one staging gate read {small.bytes_read} bytes over 2 closes and "
        f"{large.bytes_read} over 12: {per_close:.0f} bytes per extra close "
        f"exceeds the {_BYTES_PER_CLOSE_CEILING}-byte ceiling, so the read path is "
        "O(history) x O(#closes) in BYTES again"
    )
    assert per_close < large.ledger_bytes, (
        f"each extra close cost {per_close:.0f} bytes, at least a whole extra pass "
        f"over the campaign's {large.ledger_bytes} ledger bytes"
    )


@pytest.mark.heavy
def test_the_per_close_byte_slope_does_not_follow_history(tmp_path: Path) -> None:
    """The per-close BYTE cost is a constant, not history's shadow.

    The pre-fix per-close slope was itself proportional to history: tripling the
    campaign's history tripled what each extra close cost. A constant cannot do
    that, and this is the clause that tells the two apart.
    """

    lean_small, _ = _staging_gate_cost(tmp_path / "ls", rungs=2, history_rows=400)
    lean_large, _ = _staging_gate_cost(tmp_path / "ll", rungs=12, history_rows=400)
    fat_small, _ = _staging_gate_cost(tmp_path / "fs", rungs=2, history_rows=1200)
    fat_large, _ = _staging_gate_cost(tmp_path / "fl", rungs=12, history_rows=1200)
    lean_slope = (lean_large.bytes_read - lean_small.bytes_read) / 10
    fat_slope = (fat_large.bytes_read - fat_small.bytes_read) / 10
    assert fat_slope <= _BYTES_PER_CLOSE_CEILING, (
        f"the per-close byte cost rose to {fat_slope:.0f} when history tripled"
    )
    assert fat_slope <= 2 * max(lean_slope, 1024.0), (
        f"the per-close byte cost tracked history ({lean_slope:.0f} bytes per "
        f"close at H=400, {fat_slope:.0f} at H=1200): the product term is back"
    )


@pytest.mark.heavy
def test_one_gate_reads_campaign_history_a_bounded_number_of_times_in_bytes(
    tmp_path: Path,
) -> None:
    """The slope of BYTES against ledger bytes IS the number of passes made."""

    lean, _ = _staging_gate_cost(tmp_path / "lean", rungs=4, history_rows=400)
    fat, _ = _staging_gate_cost(tmp_path / "fat", rungs=4, history_rows=1200)
    passes = (fat.bytes_read - lean.bytes_read) / (fat.ledger_bytes - lean.ledger_bytes)
    assert passes <= _HISTORY_BYTE_PASS_CEILING, (
        f"one staging gate reads the campaign's bytes {passes:.2f} times "
        f"({lean.bytes_read} bytes at {lean.ledger_bytes} ledger bytes, "
        f"{fat.bytes_read} at {fat.ledger_bytes}); the ceiling is "
        f"{_HISTORY_BYTE_PASS_CEILING}"
    )


@pytest.mark.heavy
def test_twenty_thousand_rung_byte_projection_stays_under_its_derived_ceiling(
    tmp_path: Path,
) -> None:
    """Fit the measured BYTE model and project it to the 20k rung.

    Same method as the decoded-row projection above, on the metric round-3 found
    unbounded: measure the per-ledger-byte and per-close slopes on tractable
    ladders and evaluate the fitted model at the target. The pre-fix model
    projects 15 passes over a 20k ledger PLUS 40 closes x ~0.5 MB each; this
    asserts the fitted model has no such product term.
    """

    lean, lean_ladder = _staging_gate_cost(tmp_path / "lean", rungs=4, history_rows=400)
    fat, _ = _staging_gate_cost(tmp_path / "fat", rungs=4, history_rows=1200)
    short, _ = _staging_gate_cost(tmp_path / "short", rungs=2, history_rows=400)
    tall, _ = _staging_gate_cost(tmp_path / "tall", rungs=12, history_rows=400)

    bytes_per_ledger_byte = (fat.bytes_read - lean.bytes_read) / (
        fat.ledger_bytes - lean.ledger_bytes
    )
    bytes_per_close = (tall.bytes_read - short.bytes_read) / (12 - 2)
    # The target's ledger SIZE, from this rig's own measured bytes per row.
    target_ledger_bytes = _LADDER_TARGET_HISTORY * (
        lean.ledger_bytes / max(lean_ladder.history_rows, 1)
    )
    projected = (
        bytes_per_ledger_byte * target_ledger_bytes
        + bytes_per_close * _LADDER_TARGET_CLOSES
    )
    ceiling = (
        _HISTORY_BYTE_PASS_CEILING * target_ledger_bytes
        + _BYTES_PER_CLOSE_CEILING * _LADDER_TARGET_CLOSES
    )
    assert projected <= ceiling, (
        f"the fitted model ({bytes_per_ledger_byte:.2f} bytes read per ledger byte, "
        f"{bytes_per_close:.0f} per close) projects {projected:.0f} bytes for one "
        f"staging gate at the {_LADDER_TARGET_HISTORY}-model rung with "
        f"{_LADDER_TARGET_CLOSES} closes; the derived ceiling is {ceiling:.0f}"
    )


@pytest.mark.smoke
def test_inside_one_epoch_an_unchanged_extent_is_proven_once(tmp_path: Path) -> None:
    """A repeated derivation inside ONE decision reads ZERO further bytes.

    The crisp statement of the round-3 B1 fix, and of its teeth: the SAME 24
    repeats outside an epoch re-hash the whole ledger 24 times, which is exactly
    what one licensing decision was doing once per prior close per census source.
    """

    path = tmp_path / "models.jsonl"
    ledger_bytes = big_models_ledger(path, 400)
    close_reads_module.reset_incremental_reads()

    with close_reads_module.read_epoch():
        first = measure_read_cost(
            lambda: chain_state(path, scope="models.shard"), cold=False, peak=False
        )
        repeats = measure_read_cost(
            lambda: [
                chain_state(path, scope="models.shard") for _ in range(_EPOCH_REPEATS)
            ],
            cold=False,
            peak=False,
        )
    assert first.bytes_read >= ledger_bytes, "the first derivation must read the bytes"
    # The only bytes a repeat may still read are the ONE-BYTE torn-tail probe
    # every derivation of a recovering reader makes before it looks at anything.
    assert repeats.prefix_proofs == 0, (
        f"{_EPOCH_REPEATS} repeats of one derivation inside a single epoch executed "
        f"{repeats.prefix_proofs} prefix proofs over {repeats.prefix_proof_bytes} "
        "bytes; an unchanged extent is proven ONCE per epoch"
    )
    assert repeats.bytes_read <= 4 * _EPOCH_REPEATS, (
        f"{_EPOCH_REPEATS} repeats read {repeats.bytes_read} bytes, more than the "
        "one-byte torn-tail probe each is allowed"
    )
    assert repeats.epoch_proof_reuses == _EPOCH_REPEATS

    outside = measure_read_cost(
        lambda: [chain_state(path, scope="models.shard") for _ in range(_EPOCH_REPEATS)],
        cold=False,
        peak=False,
    )
    assert outside.bytes_read >= _EPOCH_REPEATS * ledger_bytes, (
        "outside an epoch every call must still prove the prefix from its bytes, "
        f"but {_EPOCH_REPEATS} calls read only {outside.bytes_read} bytes"
    )


@pytest.mark.smoke
def test_an_epoch_reproves_any_extent_that_moved(tmp_path: Path) -> None:
    """An epoch memoizes an EXTENT, never a path: a ledger that moved re-proves.

    The fail-closed half of the epoch, and its disclosure. Held state is still
    served only over bytes that reproduce their recorded digest; what an epoch
    removes is the REPETITION of that proof for an extent that has not moved.
    The memo key carries the device, inode, size, and mtime, so an append inside
    a decision's own window re-proves and extends, and a same-size REPLACEMENT is
    a different inode and re-derives. The one mutation an open epoch does not
    re-observe is an in-place rewrite that preserves both size and mtime inside a
    single decision -- the key's fields are asserted below so that residual can
    never widen unnoticed.
    """

    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"row": "a"}\n{"row": "b"}\n')
    close_reads_module.reset_incremental_reads()

    with close_reads_module.read_epoch():
        assert chain_state(path, scope="meta.test").rows == 2
        with path.open("ab") as handle:
            handle.write(b'{"row": "c"}\n')
        grown = measure_read_cost(
            lambda: chain_state(path, scope="meta.test"), cold=False, peak=False
        )
        assert grown.bytes_read > 0, "a grown ledger must re-prove its held prefix"
        assert chain_state(path, scope="meta.test").rows == 3, (
            "an appended row inside an epoch was answered from the stale extent"
        )

        # A REPLACED ledger of the SAME size is a different inode, so the epoch's
        # proof of a prefix of that length can never answer for it.
        replacement = tmp_path / "replacement.jsonl"
        replacement.write_bytes(
            b'{"row": "x"}\n{"row": "y"}\n{"row": "z"}\n'
        )
        assert replacement.stat().st_size == path.stat().st_size
        os.replace(replacement, path)
        replaced = chain_state(path, scope="meta.test")

    close_reads_module.reset_incremental_reads()
    assert replaced == chain_state(path, scope="meta.test"), (
        "a same-size replaced ledger inside an epoch was served from the prior "
        "extent's proof"
    )

    extent = close_reads_module.ledger_extent(path)
    assert extent is not None
    assert close_reads_module._epoch_key(path, extent, 5) == (  # noqa: SLF001
        str(path),
        extent.device,
        extent.inode,
        extent.size,
        extent.mtime_ns,
        5,
    ), "the epoch memo key must bind every observed extent field plus the length"


# --------------------------------------------------------------------------- #
# G -- peak resident memory and the held byte budget                          #
# --------------------------------------------------------------------------- #

#: Multiple of what a derivation RETAINS that its peak transient may reach, plus
#: the substrate's fixed streaming allowance. The Track A chain retains one head
#: per row (~0.3x its ledger's bytes) but the round-2 implementation materialized
#: every appended row as a decoded dict first, peaking at ~2.65x those bytes --
#: roughly ten times what it kept. A streaming derivation's peak is its retained
#: value plus a couple of chunks.
_PEAK_OVER_RETAINED_CEILING = 4
_STREAMING_ALLOWANCE_BYTES = 4 * close_reads_module._CHUNK_BYTES  # noqa: SLF001

#: Multiple of the held set's DECLARED weight that one whole staging gate's peak
#: may reach. Stated against retention rather than against the ledger on purpose:
#: the gate's peak is dominated by the models head fold, which retains a decoded
#: row per current head BY DESIGN, so "peak / ledger bytes" mostly measures that
#: intended retention. What this pins is composition -- no step of a decision
#: transiently allocates a multiple of everything the decision keeps.
_GATE_PEAK_OVER_HELD_CEILING = 2.5


@pytest.mark.heavy
def test_the_track_a_chain_derivation_streams_instead_of_materializing(
    tmp_path: Path,
) -> None:
    """Deriving a chain over a campaign's history keeps heads, not history.

    Round-3 B1's memory half. The projection is consumed ONLY as heads, so
    retaining the decoded rows put the whole models ledger in memory to answer
    "what was the head at row N?". The reference fold below is the round-2
    spelling -- decode every appended row, then link -- measured over the same
    bytes, so the ceiling has teeth rather than being chosen to pass.
    """

    path = tmp_path / "models.jsonl"
    ledger_bytes = big_models_ledger(path, 8000)
    close_reads_module.reset_incremental_reads()
    streaming = measure_read_cost(lambda: chain_state(path, scope="models.shard"))
    held = close_reads_module._HELD[(str(path), "chain.state:models.shard")]  # noqa: SLF001
    retained = deep_size(held.value)
    ceiling = _PEAK_OVER_RETAINED_CEILING * retained + _STREAMING_ALLOWANCE_BYTES

    assert streaming.peak_bytes <= ceiling, (
        f"deriving a {ledger_bytes}-byte ledger's chain peaked at "
        f"{streaming.peak_bytes} bytes while retaining {retained} (ceiling "
        f"{ceiling}): a streaming derivation peaks at its value plus a chunk, so "
        "this one is materializing rows it does not keep"
    )

    materializing = measure_read_cost(
        lambda: _materializing_reference_chain(path, scope="models.shard")
    )
    assert materializing.peak_bytes >= 3 * streaming.peak_bytes, (
        "the round-2 materializing reference fold peaked at "
        f"{materializing.peak_bytes} bytes against the streaming derivation's "
        f"{streaming.peak_bytes}: the ceiling above has no teeth"
    )


def _materializing_reference_chain(path: Path, *, scope: str) -> Any:
    """Derive one chain the round-2 way: decode the whole suffix, then link.

    Kept as the MEASURING STICK for the clause above, never used in production.
    Deliberately reads the same bytes through the same reader so the only
    difference measured is the materialization.
    """

    import menagerie.crawler.ledger_chain as chain_module

    def extend(value: Any, rows: Any) -> Any:
        decoded = tuple(
            chain_module._decode_row(path, number, line)  # noqa: SLF001
            for number, _offset, line in rows
        )
        return chain_module.verify_rows(decoded, scope=scope)

    return close_reads_module.derive(
        path,
        projection=f"reference.chain:{scope}",
        seed=lambda: None,
        extend=extend,
        weigh=lambda _value, byte_count: byte_count,
    )


@pytest.mark.smoke
def test_streaming_the_chain_keeps_naming_the_malformed_line_first(
    tmp_path: Path,
) -> None:
    """A malformed line still wins over a broken link BELOW it, while streaming.

    The refusal-ordering contract the round-2 materialization existed to
    reproduce: ``decode_ledger`` decoded every row before ``verify_rows`` linked
    any, so a ledger holding both faults named the malformed LINE. The streaming
    derivation defers a link violation and keeps decoding, so the same ledger
    still refuses with the same wording -- and a ledger whose ONLY fault is a
    broken link still names the link.
    """

    # The envelope's VALUES are machinery-shaped (PSLRE6 D6-5), so a forged link must at
    # least LOOK like one for this test to measure the link check rather than the shape
    # check -- two canonical digests that simply do not extend the preceding row.
    forged = (
        b'{"row": "a", "chain": {"seq": 0, "prev": "sha256:'
        + b"0" * 64
        + b'", "sha256": "sha256:'
        + b"1" * 64
        + b'"}}\n'
    )
    linked = tmp_path / "linked.jsonl"
    linked.write_bytes(forged)
    close_reads_module.reset_incremental_reads()
    with pytest.raises(LedgerChainError) as link_only:
        chain_state(linked, scope="meta.test")
    assert "fails its ledger chain at row 0" in str(link_only.value)

    both = tmp_path / "both.jsonl"
    both.write_bytes(forged + b"{not json}\n")
    close_reads_module.reset_incremental_reads()
    with pytest.raises(LedgerChainError) as both_faults:
        chain_state(both, scope="meta.test")
    assert "is not valid JSON" in str(both_faults.value), (
        "a broken link above a malformed line stole the refusal: streaming must "
        "defer the link violation and keep decoding"
    )
    assert "both.jsonl:2 is not valid JSON" in str(both_faults.value), (
        "the refusal must still name the malformed line's own row number"
    )


@pytest.mark.heavy
def test_one_staging_gate_peak_is_bounded_by_what_the_decision_retains(
    tmp_path: Path,
) -> None:
    """A whole gate's peak stays a small multiple of everything it keeps."""

    ladder = build_ladder(tmp_path, rungs=4, history_rows=1200)
    cost = measure_read_cost(
        lambda: stage_next_rung(ladder), records=ladder.records, peak=True
    )
    held_bytes = close_reads_module.held_derivation_bytes()
    assert held_bytes > 0, "the gate held nothing, so there is nothing to compare"
    ceiling = _GATE_PEAK_OVER_HELD_CEILING * held_bytes + _STREAMING_ALLOWANCE_BYTES
    assert cost.peak_bytes <= ceiling, (
        f"one staging gate over a {cost.ledger_bytes}-byte models ledger peaked at "
        f"{cost.peak_bytes} bytes while retaining {held_bytes} (ceiling "
        f"{ceiling:.0f}): some step allocates a multiple of what the decision keeps"
    )


@pytest.mark.smoke
def test_the_held_set_is_bounded_in_BYTES_not_in_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long run's held derivations stay inside a real memory budget.

    Round-3 B2: ``_MAX_HELD`` capped how MANY derivations were held over entries
    of unbounded size, so 256 models projections at the 20k rung was a resident
    set of gigabytes with the cap reading as satisfied. The budget below is spent
    in bytes, and the ladder's own projections are the units.
    """

    budget = 256 * 1024
    monkeypatch.setattr(close_reads_module, "_MAX_HELD_BYTES", budget)
    close_reads_module.reset_incremental_reads()
    for index in range(24):
        path = tmp_path / f"ledger-{index}.jsonl"
        big_models_ledger(path, 40, prefix=f"b{index}")
        decode_ledger(path)
        assert close_reads_module.held_derivation_bytes() <= budget, (
            f"the held set reached {close_reads_module.held_derivation_bytes()} "
            f"bytes against a {budget}-byte budget after {index + 1} ledgers"
        )
    assert close_reads_module.read_statistics().held_evictions > 0, (
        "24 ledgers fitted inside a 256 KiB budget without a single eviction: "
        "the budget is not being spent"
    )
    assert close_reads_module.held_derivation_count() < 24, (
        "no derivation was evicted, so the byte budget bounded nothing"
    )


@pytest.mark.smoke
def test_a_derivation_heavier_than_the_whole_budget_is_returned_but_not_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One over-budget projection is served to its caller and dropped, not kept."""

    path = tmp_path / "models.jsonl"
    big_models_ledger(path, 200)
    monkeypatch.setattr(close_reads_module, "_MAX_HELD_BYTES", 4096)
    close_reads_module.reset_incremental_reads()
    rows = decode_ledger(path)
    assert len(rows) == 200, "the caller must still get its whole derivation"
    assert close_reads_module.held_derivation_count() == 0
    assert close_reads_module.held_derivation_bytes() == 0


@pytest.mark.smoke
def test_the_budget_still_holds_the_heaviest_projection_at_the_20k_rung(
    tmp_path: Path,
) -> None:
    """The byte budget bounds the held set without DISABLING it at the target.

    The two halves of this fix pull against each other: B1 wants the models
    projection HELD across a decision (that is what makes a repeat read free),
    and B2 wants the held set bounded in bytes. A budget below the heaviest
    legitimate derivation would satisfy B2 by never holding anything and quietly
    re-open B1, so the arithmetic is pinned here: one 20k-rung models projection's
    CHARGE must fit the budget with room for the rest of a campaign's close path.
    """

    sample = tmp_path / "sample.jsonl"
    rows = 100
    bytes_per_row = big_models_ledger(sample, rows) / rows
    target_bytes = bytes_per_row * _LADDER_TARGET_HISTORY
    charged = (
        close_reads_module.PROJECTION_BASE_WEIGHT
        + close_reads_module.DECODED_ROW_WEIGHT_FACTOR * target_bytes
        + 2 * close_reads_module.LINE_SPAN_WEIGHT * _LADDER_TARGET_HISTORY
    )
    assert charged * 2 <= close_reads_module._MAX_HELD_BYTES, (  # noqa: SLF001
        f"a {_LADDER_TARGET_HISTORY}-model projection is charged {charged:.0f} bytes "
        f"against a {close_reads_module._MAX_HELD_BYTES}-byte budget: at the target "  # noqa: SLF001
        "rung the heaviest close-path derivation would be evicted on sight, so every "
        "decision would re-fold the campaign's history from byte zero"
    )


@pytest.mark.heavy
def test_declared_weights_bound_what_projections_retain(tmp_path: Path) -> None:
    """Every declared projection weight is an UPPER bound on real retention.

    The budget is only a memory bound if the weights are honest, so this measures
    the transitive footprint of every derivation a real ladder's staging gate
    leaves held and compares it with what that derivation was charged. A weight
    that under-states its value turns the byte budget back into the entry count it
    replaced -- silently.
    """

    ladder = build_ladder(tmp_path, rungs=4, history_rows=1200)
    close_reads_module.reset_incremental_reads()
    stage_next_rung(ladder)
    assert close_reads_module.held_derivation_count() > 0

    understated: list[str] = []
    for (path, projection), held in close_reads_module._HELD.items():  # noqa: SLF001
        retained = deep_size(held.value)
        if held.weight < retained:
            understated.append(
                f"{projection} over {Path(path).name}: charged {held.weight} bytes, "
                f"retains {retained}"
            )
    assert understated == [], (
        "these projections are charged less than they retain, so the held-set "
        "byte budget is not a memory bound: " + "; ".join(sorted(understated))
    )
