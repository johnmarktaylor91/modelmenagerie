"""Bind ad-hoc fixture intake snapshots into a real partitioned campaign repo.

Since the T4-T2 binding refusal (rung staging), every consumer that derives a
driver config or authority context from an intake snapshot REFUSES with
``UnboundIntakeSnapshotError`` unless a campaign manifest row or a durable rung
registration owns the snapshot -- defaults would silently select the wrong
author and checker models. Fixture repositories that drive the CLI must
therefore be partitioned and bound exactly like reality, using the same
machinery reality uses: :func:`emit_campaign_partitions` materializes the
four-campaign manifest over a roster containing the fixture's own rows, and
:func:`register_existing_snapshot` adopts the ad-hoc snapshot as a registered
rung of its campaign. Nothing here relaxes the refusal; an unbound snapshot
still refuses everywhere.
"""

from __future__ import annotations

from pathlib import Path

from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.intake import IntakeSnapshot
from menagerie.crawler.partitioner import emit_campaign_partitions
from menagerie.crawler.rung_staging import register_existing_snapshot

#: One roster row per non-c1 campaign: :func:`emit_campaign_partitions` refuses
#: an empty campaign partition, and fixture rows land in c1-mech by exclusion
#: (their zoos are none of the reserved campaign literals).
_FILLER_ROWS = (
    {"name": "filler-disco", "zoo": "discovered-pytorch", "variant": ""},
    {"name": "filler-classic", "zoo": "unregistered-classics-pytorch", "variant": ""},
    {"name": "filler-native", "zoo": "discovered-keras", "variant": ""},
)


def bind_snapshot_to_campaign(
    repo_root: Path,
    snapshot: IntakeSnapshot,
    campaign_id: str = "c1-mech",
) -> None:
    """Partition one fixture repository and bind a snapshot to its campaign.

    Emits a real ``campaigns.json`` (plus partitions and per-campaign root
    intake snapshots) under ``<repo_root>/menagerie/crawler/records`` from a
    roster made of the snapshot's own rows plus one filler row per other
    campaign, then registers the ad-hoc snapshot as a durable rung of
    ``campaign_id`` so ``find_campaign_binding`` / ``find_rung_registration``
    resolve it exactly as they do in production.

    Parameters
    ----------
    repo_root:
        Fixture repository root the test passes as ``--repo-root``.
    snapshot:
        Ad-hoc intake snapshot the test drives through the CLI.
    campaign_id:
        Owning frozen tier campaign. Fixture rows partition into ``c1-mech``
        by exclusion, so a different campaign requires roster rows whose zoos
        actually belong to it.
    """

    bind_snapshot_at_records_root(
        repo_root / "menagerie" / "crawler" / "records",
        snapshot,
        campaign_id,
        roster_path=repo_root / "fixture-roster.jsonl",
    )


def bind_snapshot_at_records_root(
    records_root: Path,
    snapshot: IntakeSnapshot,
    campaign_id: str = "c1-mech",
    *,
    roster_path: Path | None = None,
    extra_roster_rows: tuple[dict, ...] = (),
) -> None:
    """Partition one records root and bind a snapshot to its campaign.

    The records-root spelling of :func:`bind_snapshot_to_campaign` for driver
    rigs whose canonical records directory is not the CLI's
    ``<repo_root>/menagerie/crawler/records`` convention (the slice-f rigs use
    ``<tmp_path>/records``). Same machinery, same refusals.

    Parameters
    ----------
    records_root:
        Canonical crawler records directory the driver's ledgers resolve to.
    snapshot:
        Ad-hoc intake snapshot the test is about to drive.
    campaign_id:
        Owning frozen tier campaign (fixture rows land in ``c1-mech``).
    roster_path:
        Where to materialize the fixture roster; defaults next to the records
        root.
    extra_roster_rows:
        Additional roster rows for members a LATER widened rung will
        introduce: campaign membership is frozen at the root, so every future
        member must partition into the campaign from the start.
    """

    roster_rows = (
        [{"name": item.name, "zoo": item.zoo, "variant": item.variant} for item in snapshot.items]
        + list(extra_roster_rows)
        + list(_FILLER_ROWS)
    )
    roster = roster_path or records_root.parent / "fixture-roster.jsonl"
    roster.parent.mkdir(parents=True, exist_ok=True)
    roster.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in roster_rows))
    emit_campaign_partitions(roster, records_root)
    # A fixture snapshot is about to RUN, so it registers through the NEW-WORK
    # door (the historical door demands proof of recorded heads, which a
    # not-yet-driven fixture cannot have). AW-2 fails CLOSED on an absent
    # models ledger, so the empty shard states explicitly that no awards exist.
    models_shard = records_root / "models" / "current-shard.jsonl"
    if not models_shard.exists() and not (records_root / "models.jsonl").exists():
        models_shard.parent.mkdir(parents=True, exist_ok=True)
        models_shard.write_bytes(b"")
    register_existing_snapshot(
        records_root,
        snapshot.root,
        campaign_id=campaign_id,
        label="fixture-binding",
        registered_by="pytest",
        created_at="2026-08-11T00:00:00Z",
        historical_only=False,
    )


def adopt_successor_snapshot(
    records_root: Path,
    snapshot: IntakeSnapshot,
    campaign_id: str = "c1-mech",
    *,
    label: str = "fixture-successor",
) -> None:
    """Chain a later same-membership ad-hoc snapshot onto the registry tip.

    Multi-leg driver rigs that re-drive under CHANGED intake bytes must chain
    the changed snapshot to the registered lineage: the close-v2 partition
    proof carries an untouched member's head across legs only along that
    chain (BL-5), so an unregistered changed snapshot reads prior-leg
    evidence as foreign. The public staging doors refuse a same-membership
    successor ("a successor must strictly widen the lineage tip"), so this
    uses the registry-append door the close-v2 contract tests use, plus the
    durable snapshot copy ``introduced_member_ids`` re-verifies at close.
    """

    from menagerie.crawler.identity import atomic_replace_bytes
    from menagerie.crawler.rung_staging import (
        append_rung_registration,
        build_rung_registration,
        load_rung_registrations,
        rung_registry_path,
    )

    tip = [
        row for row in load_rung_registrations(records_root) if row.campaign_id == campaign_id
    ][-1]
    durable_root = records_root / "intake" / snapshot.snapshot_id
    if durable_root.resolve() != snapshot.root.resolve():
        for path in sorted(snapshot.root.rglob("*")):
            if path.is_file():
                atomic_replace_bytes(
                    durable_root / path.relative_to(snapshot.root), path.read_bytes()
                )
    append_rung_registration(
        rung_registry_path(records_root),
        build_rung_registration(
            campaign_id=campaign_id,
            snapshot_id=snapshot.snapshot_id,
            snapshot_sha256=snapshot.snapshot_sha256,
            derived_from_snapshot_id=tip.snapshot_id,
            member_count=len(snapshot.items),
            label=label,
            registered_by="pytest",
            created_at="2026-08-11T00:00:00Z",
        ),
    )
