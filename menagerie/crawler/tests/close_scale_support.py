"""A real multi-close LADDER, and an instrument that counts what reading it costs.

The round-2 T4 rig. Two things the close/licensing suites had no way to express:

* a campaign with SEVERAL registered rungs, each closed honestly through the
  production derivation, each licensing the next -- the shape whose cost the
  200 -> 20k ladder actually pays, and the only shape in which an
  O(#closes)-per-read defect is visible at all; and
* a census of how many LEDGER ROWS one close derivation or one staging-gate
  evaluation decodes, so "bounded independent of #closes" is a measured number
  rather than a claim.

Every rung closes through :mod:`close_object_support`'s honest path (a real leg
generation, a real sealed WS-1 plan admitting exactly the members the rung
INTRODUCES, the production ``derive_partition_proof``), so the ladder is
byte-identical to what a driver would leave behind. Nothing here forges a close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, cast

from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.partitioner import emit_campaign_partitions
from menagerie.crawler.tests import rt3_support as rt

#: The ladder campaign. ``c3-classics`` is the roster half the rt3 rig widens.
LADDER_CAMPAIGN = "c3-classics"

#: Fixed provenance, so a ladder's digests are reproducible byte-for-byte.
LADDER_RUN_ID = "pytest-close"


@dataclass(frozen=True)
class Ladder:
    """One built ladder and the handles a gate evaluation needs.

    Parameters
    ----------
    repo:
        Repository root.
    records:
        Campaign canonical records directory.
    tip:
        ``StagedRung`` of the newest registered rung.
    tip_members:
        The tip's durable membership.
    rungs:
        How many rungs were staged and closed.
    history_rows:
        Non-empty row count of the models ledger.
    """

    repo: Path
    records: Path
    tip: Any
    tip_members: tuple[str, ...]
    rungs: int
    history_rows: int


def build_ladder_repo(tmp_path: Path, *, roster: int) -> Path:
    """Materialize the rt3 four-campaign repository widened by ``roster`` rows.

    Parameters
    ----------
    tmp_path:
        Scratch directory.
    roster:
        Extra ``unregistered-classics-pytorch`` roster rows beyond the rt3 set,
        so ``c3-classics`` is wide enough for a multi-rung ladder.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    repo_root = tmp_path / "repo"
    records_root = repo_root / "menagerie" / "crawler" / "records"
    rows = [
        *rt.ROSTER_ROWS,
        *(
            {
                "name": f"classic-{index:04d}",
                "zoo": "unregistered-classics-pytorch",
                "variant": "",
            }
            for index in range(roster)
        ),
    ]
    roster_path = tmp_path / "roster.jsonl"
    roster_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    emit_campaign_partitions(roster_path, records_root)
    rt.write_models_shard(records_root, [])
    return repo_root


def pad_models_history(records: Path, rows: int, *, prefix: str = "pad") -> None:
    """Append ``rows`` foreign pre-window heads to inflate campaign history.

    Foreign on purpose: they are not members of any rung, so they change no
    verdict and no digest the ladder's closes bind -- they exist only to make
    "did this read walk the whole history?" measurable.

    Parameters
    ----------
    records:
        Campaign canonical records directory.
    rows:
        How many heads to append.
    prefix:
        Stable-id namespace. A second call MUST use a fresh prefix: a repeated
        stable id would be a forked revision chain, which the models fold
        refuses (correctly).
    """

    if rows <= 0:
        return
    path = records / "models" / "current-shard.jsonl"
    with path.open("ab") as handle:
        handle.write(
            b"".join(
                canonical_json_bytes(rt.head_record(f"m_{prefix}{index:07d}")) + b"\n"
                for index in range(rows)
            )
        )


def close_rung_honestly(records: Path, repo: Path, staged: Any, *, closed_at: str) -> None:
    """Close one ladder rung: admit exactly the members it INTRODUCES.

    A ladder rung strictly widens its lineage tip, so its membership is
    "everything the tip already proved, plus the new members". The honest leg
    therefore admits only the introduced members to work (they get in-window
    heads and become ADJUDICATED) and CARRIES the inherited ones from the
    ancestor close that adjudicated them.

    Parameters
    ----------
    records, repo:
        Campaign records directory and repository root.
    staged:
        ``StagedRung`` for the rung being closed.
    closed_at:
        Fixed close timestamp.
    """

    from menagerie.crawler.leg_generation import record_leg_plan
    from menagerie.crawler.rung_staging import introduced_member_ids
    from menagerie.crawler.tests import close_object_support as support

    members = rt.members_of(repo, staged.registration.snapshot_id)
    introduced = sorted(introduced_member_ids(records, staged.registration))
    generation = support.open_fixture_leg(
        records,
        campaign_id=LADDER_CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        run_id=LADDER_RUN_ID,
    )
    support._seed_member_heads(
        records, introduced, staged.registration.snapshot_id, LADDER_RUN_ID
    )
    record_leg_plan(
        records,
        campaign_id=LADDER_CAMPAIGN,
        generation_id=generation.generation_id,
        run_id=LADDER_RUN_ID,
        admitted_ids=introduced,
        plan_payload={"probe": True, "admitted_ids": introduced},
        created_at=closed_at,
    )
    support.record_fixture_close(
        records,
        campaign_id=LADDER_CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        member_ids=members,
        seed_member_heads=False,
        closed_at=closed_at,
        run_id=LADDER_RUN_ID,
    )


def build_ladder(
    tmp_path: Path, *, rungs: int, history_rows: int = 0, first: int = 3, step: int = 1
) -> Ladder:
    """Stage and honestly close ``rungs`` widening rungs of one campaign.

    Parameters
    ----------
    tmp_path:
        Scratch directory.
    rungs:
        Number of rungs to stage and close.
    history_rows:
        Foreign models rows appended BEFORE the ladder, inflating the
        campaign's pre-window history.
    first, step:
        Width of the first rung and the widening step.

    Returns
    -------
    Ladder
        The built ladder.
    """

    repo = build_ladder_repo(tmp_path, roster=first + rungs * step + 2)
    records = rt.records_root_of(repo)
    pad_models_history(records, history_rows)
    staged = None
    for index in range(rungs):
        count = first + index * step
        staged = rt.stage(
            repo,
            LADDER_CAMPAIGN,
            count=count,
            derived_from=None if index == 0 else cast(Any, staged).registration.snapshot_id,
            label=f"rung-{count}",
        )
        close_rung_honestly(records, repo, staged, closed_at=f"2026-08-11T00:{index:02d}:00Z")
    assert staged is not None
    from menagerie.crawler.leg_generation import source_row_count

    return Ladder(
        repo=repo,
        records=records,
        tip=staged,
        tip_members=rt.members_of(repo, staged.registration.snapshot_id),
        rungs=rungs,
        history_rows=source_row_count(records / "models" / "current-shard.jsonl"),
    )


def stage_next_rung(ladder: Ladder, *, label: str = "probe-next") -> Any:
    """Stage the rung after a ladder's tip, driving the whole licensing gate."""

    return rt.stage(
        ladder.repo,
        LADDER_CAMPAIGN,
        count=len(ladder.tip_members) + 1,
        derived_from=ladder.tip.registration.snapshot_id,
        label=label,
    )


@dataclass
class ReadCensus:
    """How many ledger rows one close/staging evaluation decoded.

    Every decoder the close path can reach is counted, so a defect cannot hide
    by moving between them. ``rows`` is the number that must stay bounded: it
    counts one increment per LEDGER ROW parsed, wherever it was parsed.

    Parameters
    ----------
    rows:
        Total ledger rows decoded, across every decoder below.
    by_decoder:
        Per-decoder breakdown, for diagnosing which read regressed.
    """

    rows: int = 0
    by_decoder: dict[str, int] = field(default_factory=dict)

    def count(self, decoder: str, amount: int = 1) -> None:
        """Record ``amount`` rows decoded by ``decoder``."""

        self.rows += amount
        self.by_decoder[decoder] = self.by_decoder.get(decoder, 0) + amount


def install_read_census(monkeypatch: Any, *, cold: bool = True) -> ReadCensus:
    """Count every ledger row the close/staging read path decodes.

    Parameters
    ----------
    monkeypatch:
        Active pytest monkeypatch fixture.
    cold:
        Whether to drop every held derivation first, measuring a COLD
        evaluation -- the staging gate's real shape, since it runs in its own
        process. ``False`` measures a WARM evaluation inside a long-lived
        driver, where only an appended tail should cost anything.

    Returns
    -------
    ReadCensus
        Live census, updated as the measured call runs.
    """

    import menagerie.crawler.close_object as close_object_module
    import menagerie.crawler.close_reads as close_reads_module
    import menagerie.crawler.ledger_chain as ledger_chain_module
    import menagerie.crawler.leg_generation as leg_generation_module
    import menagerie.crawler.partition_proof as partition_proof_module
    import menagerie.crawler.recordio as recordio_module

    if cold:
        close_reads_module.reset_incremental_reads()
        partition_proof_module.reset_partition_proof_cache()
    census = ReadCensus()

    def wrap_row(module: Any, name: str, decoder: str) -> None:
        real = getattr(module, name)

        def counting(*args: Any, **kwargs: Any) -> Any:
            census.count(decoder)
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, counting)

    def wrap_sequence(module: Any, name: str, decoder: str) -> None:
        real = getattr(module, name)

        def counting(*args: Any, **kwargs: Any) -> Any:
            result = real(*args, **kwargs)
            census.count(decoder, len(result))
            return result

        monkeypatch.setattr(module, name, counting)

    wrap_row(ledger_chain_module, "_decode_row", "ledger_chain._decode_row")
    # Chain verification is per-ROW work too: one position-bound digest each. A
    # read path that stops DECODING history but still re-verifies its whole
    # chain per close has not been fixed, so the census counts links as rows.
    wrap_row(ledger_chain_module, "_verified_link", "ledger_chain._verified_link")
    wrap_row(close_object_module, "_decode_close_row", "close_object._decode_close_row")
    wrap_row(recordio_module, "_decode_validated_line", "recordio._decode_validated_line")
    wrap_sequence(recordio_module, "scan_jsonl", "recordio.scan_jsonl")
    wrap_sequence(ledger_chain_module, "decode_payload", "ledger_chain.decode_payload")
    wrap_sequence(leg_generation_module, "source_rows", "leg_generation.source_rows")
    real_window = leg_generation_module.source_row_window

    def counting_window(path: Path, start: int, stop: int) -> Any:
        result = real_window(path, start, stop)
        census.count("leg_generation.source_row_window", len(result))
        return result

    monkeypatch.setattr(leg_generation_module, "source_row_window", counting_window)
    monkeypatch.setattr(partition_proof_module, "source_row_window", counting_window)
    return census


@dataclass(frozen=True)
class ReadCost:
    """What one measured call physically cost the close/licensing read path.

    Rows decoded (:class:`ReadCensus`) was only half the question: round-2 made
    the DECODE incremental while every held derivation still re-hashed its whole
    prefix, so the same evaluation stayed ``O(#closes) x O(history)`` in BYTES.
    This is the other half, and it is measured from the substrate's OWN
    production counters plus ``tracemalloc`` -- no monkeypatch, so it cannot
    measure only the reads a patch happened to cover.

    Parameters
    ----------
    bytes_read:
        Ledger bytes physically read, to hash or to decode.
    prefix_proofs, prefix_proof_bytes:
        Executed prefix proofs and the bytes they covered.
    epoch_proof_reuses:
        Proofs served from an open read epoch instead of re-hashing.
    cold_derivations, extended_derivations:
        Derivations from byte zero, and extensions over an appended suffix.
    peak_bytes:
        Peak traced allocation above the pre-call baseline.
    ledger_bytes:
        Byte size of the campaign's models ledger, the scale variable.
    """

    bytes_read: int
    prefix_proofs: int
    prefix_proof_bytes: int
    epoch_proof_reuses: int
    cold_derivations: int
    extended_derivations: int
    peak_bytes: int
    ledger_bytes: int

    @property
    def history_passes(self) -> float:
        """Return bytes read expressed in whole models-ledger passes."""

        return self.bytes_read / max(self.ledger_bytes, 1)


def models_ledger_bytes(records: Path) -> int:
    """Return the byte size of one campaign's models ledger."""

    path = records / "models" / "current-shard.jsonl"
    return path.stat().st_size if path.is_file() else 0


def measure_read_cost(
    call: Any, *, records: Optional[Path] = None, cold: bool = True, peak: bool = True
) -> ReadCost:
    """Run ``call`` and return what it cost in BYTES and peak resident memory.

    Parameters
    ----------
    call:
        Zero-argument callable to measure.
    records:
        Campaign records directory, for the models-ledger scale variable.
    cold:
        Whether to drop every held derivation first (a fresh process's shape).
    peak:
        Whether to trace allocations. ``tracemalloc`` roughly doubles run time,
        so byte-only clauses turn it off.
    """

    import tracemalloc

    from menagerie.crawler import close_reads as close_reads_module
    from menagerie.crawler import partition_proof as partition_proof_module

    if cold:
        close_reads_module.reset_incremental_reads()
        partition_proof_module.reset_partition_proof_cache()
    close_reads_module.reset_read_statistics()
    peak_bytes = 0
    if peak:
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
    try:
        call()
    finally:
        if peak:
            peak_bytes = max(0, tracemalloc.get_traced_memory()[1] - baseline)
            tracemalloc.stop()
    statistics = close_reads_module.read_statistics()
    return ReadCost(
        bytes_read=statistics.bytes_read,
        prefix_proofs=statistics.prefix_proofs,
        prefix_proof_bytes=statistics.prefix_proof_bytes,
        epoch_proof_reuses=statistics.epoch_proof_reuses,
        cold_derivations=statistics.cold_derivations,
        extended_derivations=statistics.extended_derivations,
        peak_bytes=peak_bytes,
        ledger_bytes=0 if records is None else models_ledger_bytes(records),
    )


def deep_size(value: Any, seen: Optional[set[int]] = None) -> int:
    """Return the transitive CPython footprint of one object graph.

    The instrument behind "a declared projection weight is an UPPER bound on
    what that projection retains". Shared objects are counted once, which is
    what a resident-set question means.

    Parameters
    ----------
    value:
        Object graph to size.
    seen:
        Identities already counted, for recursion.
    """

    import sys

    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        for key, item in value.items():
            total += deep_size(key, seen) + deep_size(item, seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            total += deep_size(item, seen)
    elif hasattr(value, "__dict__"):
        total += deep_size(vars(value), seen)
    return total


def big_models_ledger(path: Path, rows: int, *, prefix: str = "chain") -> int:
    """Write a standalone models-shaped ledger and return its byte size.

    For the derivation-level memory clauses, which need a ledger of a REAL
    campaign's width without paying to close a ladder over it.

    Parameters
    ----------
    path:
        Ledger path to write.
    rows:
        Non-empty rows to write.
    prefix:
        Stable-id namespace.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            canonical_json_bytes(rt.head_record(f"m_{prefix}{index:07d}")) + b"\n"
            for index in range(rows)
        )
    )
    return path.stat().st_size


def derive_tip_proof(ladder: Ladder, *, generation_id: Optional[str] = None) -> Any:
    """Re-derive the tip rung's sealed proof through the production derivation.

    The CLOSE side of the cost question: ``proof_refusals`` runs exactly this
    recomputation, and the driver runs it to seal a close.

    Parameters
    ----------
    ladder:
        Built ladder.
    generation_id:
        Generation to derive for; defaults to the tip close's own.
    """

    from menagerie.crawler.close_object import close_object_path, load_close_objects
    from menagerie.crawler.leg_generation import find_leg_open_row
    from menagerie.crawler.partition_proof import derive_partition_proof

    closes = load_close_objects([close_object_path(ladder.records)])
    tip_close = next(
        row
        for row in reversed(closes)
        if str(row.get("intake_snapshot_id")) == ladder.tip.registration.snapshot_id
    )
    open_row = find_leg_open_row(ladder.records, str(tip_close["leg_open_row_sha256"]))
    assert open_row is not None
    return derive_partition_proof(
        ladder.records,
        campaign_id=LADDER_CAMPAIGN,
        intake_snapshot_id=str(tip_close["intake_snapshot_id"]),
        intake_snapshot_sha256=str(tip_close["intake_snapshot_sha256"]),
        phase=tip_close.get("phase"),
        close_kind="leg",
        generation_id=generation_id or str(tip_close["generation_id"]),
        baselines=dict(open_row["baselines"]),
        member_ids=sorted(ladder.tip_members),
    )


def ladder_digests(records: Path) -> dict[str, Any]:
    """Return every licensing digest a ladder commits, for equality proof.

    The semantics-preservation instrument: close content addresses, sealed
    partition-proof addresses, the same-campaign predecessor spine, each close's
    persisted Track A link and verified close-chain head, every census source's
    chain head, the three meta-ledger chain heads, the whole Track A refusal
    list, and each registration's licensing anchor.

    Parameters
    ----------
    records:
        Campaign canonical records directory.
    """

    from menagerie.crawler.close_object import close_object_path, load_close_objects
    from menagerie.crawler.ledger_chain import (
        chain_state,
        close_chain_head,
        ledger_chain_refusals,
    )
    from menagerie.crawler.leg_generation import close_ledger_sources, leg_events_path
    from menagerie.crawler.rung_staging import load_rung_registrations, rung_registry_path

    closes = load_close_objects([close_object_path(records)])
    sources = close_ledger_sources(records)
    chain_heads = {
        scope: chain_state(path, scope=scope).head_sha256 for scope, path in sources.items()
    }
    for scope, path in (
        ("meta.closes", close_object_path(records)),
        ("meta.leg-events", leg_events_path(records)),
        ("meta.registry", rung_registry_path(records)),
    ):
        chain_heads[scope] = chain_state(path, scope=scope).head_sha256
    return {
        "closes": [
            {
                "close_sha256": row["close_sha256"],
                "partition_proof_sha256": row["partition_proof_sha256"],
                "generation_id": row["generation_id"],
                "predecessor_close_sha256": row.get("predecessor_close_sha256"),
                "chain": row.get("chain"),
                "close_chain_head": close_chain_head(records, row["close_sha256"]),
            }
            for row in closes
        ],
        "chain_heads": chain_heads,
        "chain_refusals": list(ledger_chain_refusals(records)),
        "registrations": [
            {
                "snapshot_id": row.snapshot_id,
                "snapshot_sha256": row.snapshot_sha256,
                "licensed_by_close_sha256": row.licensed_by_close_sha256,
                "closes_chain_head": row.closes_chain_head,
            }
            for row in load_rung_registrations(records)
        ],
    }


def sorted_members(values: Sequence[str]) -> tuple[str, ...]:
    """Return one membership in canonical sorted order."""

    return tuple(sorted(str(value) for value in values))
