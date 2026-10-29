"""BL-3: a checker-identity rotation must never brick frozen artifacts.

The B-05 sole-checker flip (`f08e8bdb`, JMT 2026-08-14 Terra retirement) rotated
``CAMPAIGN_SPECS[*].checker_model`` from ``gpt-5.6-terra`` to ``gpt-5.6-sol``.
Because the frozen campaign manifest was validated for EQUALITY against those
mutable in-code specs, one constant edit refused every reader of artifacts that
were perfectly intact:

* every catalog entry point tracebacked at manifest load (`stratify_rung`,
  `stage_rung`, `merge_host_shards`, the campaign-root binding);
* `campaign_merge` refused all 10 live c1-mech models -- including **m4334** and
  **m5273**, two of the three award-eating models -- for the crime of having been
  adjudicated by the model the manifest says adjudicated them;
* a rename hard-refused every operator command, `status` included, so the
  operator's first observation was "everything refuses" with a message naming a
  field and no lever;
* and one reader (`rung_staging._registered_prior_lineage`) swallowed the refusal
  into an EMPTY lineage, silently dropping the campaign root.

The fix is a kind split, not a re-freeze: BINDING fields address frozen bytes and
are verified exactly; PROJECTION fields are a read-time echo of the live spec and
are never compared for equality. The two identity-bearing projections are instead
constrained to a CLOSED declared history, so an unregistered rename still fails
closed -- and now names the remedy.

Pinned here, so the class cannot recur:

1. the entry points survive a rotated manifest, all the way down the OPS-04 stack;
2. rows recorded under a declared prior checker stay adjudicable;
3. an UNDECLARED label still refuses, and the refusal names the lever;
4. the frozen half is still verified byte-exactly (this is not a loosening);
5. every ``CampaignSpec`` field is classified as binding or projection;
6. every adjudication-steering projection is durably attested by the leg that ran
   under it, in the emitter AND in the schema that governs it;
7. no NEW load-bearing identity may be a non-versioned literal.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.campaign_merge import CampaignMergeError, _assert_author_binding
from menagerie.crawler.cli import _snapshot_driver_config
from menagerie.crawler.constants import (
    CAMPAIGN_SPEC_BINDING_FIELDS,
    CAMPAIGN_SPEC_IDENTITY_PROJECTIONS,
    CAMPAIGN_SPEC_LEG_ATTESTED_PROJECTIONS,
    CAMPAIGN_SPEC_PROJECTION_FIELDS,
    NON_VERSIONED_IDENTITY_LITERALS,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    TIER_CAMPAIGN_AUTHOR_MODEL_HISTORY,
    TIER_CAMPAIGN_CHECKER_MODEL_HISTORY,
)
from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.intake import load_intake_snapshot
from menagerie.crawler.models import JsonObject
from menagerie.crawler.partitioner import (
    CAMPAIGN_SPECS,
    CampaignBinding,
    CampaignPartitionError,
    campaign_binding_for_snapshot,
    declared_identity_history,
    emit_campaign_partitions,
    find_campaign_binding,
    load_campaign_bindings,
)
from menagerie.crawler.rung_staging import _campaign_root_binding
from menagerie.crawler.tools.audit_checker_identity_epochs import (
    CheckerIdentityAuditError,
    audit_records_root,
    main as audit_main,
)

#: Sub-second load-bearing coverage promoted into the mandatory smoke tier.
pytestmark = pytest.mark.smoke

_CRAWLER_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _CRAWLER_ROOT.parents[1]
_ROSTER = _REPOSITORY_ROOT / "menagerie" / "data" / "crawl_roster.jsonl"
_UNPINNED_INVENTORY_PATH = _CRAWLER_ROOT / "tests" / "unpinned_identity_inventory.json"

#: The prior sole checker. Named here exactly once, as the pre-flip epoch under
#: which the live pilot's 298 append-only revisions were adjudicated.
_PRIOR_CHECKER = "gpt-5.6-terra"

#: Campaigns whose live pilot manifest still records the prior checker.
_ROTATED_CAMPAIGNS = ("c1-mech", "c2-disco", "c4-native")


# ---------------------------------------------------------------------------
# Fixture: a real four-campaign records root, then a rotated manifest.
# ---------------------------------------------------------------------------


def _roster_rows() -> list[JsonObject]:
    """Return one real roster row per campaign classifier.

    Returns
    -------
    list[dict[str, Any]]
        Four production-shaped roster rows, one per frozen campaign.
    """

    rows = [json.loads(line) for line in _ROSTER.read_text(encoding="utf-8").splitlines() if line]
    zoos = (
        "torchvision",
        "discovered-pytorch",
        "unregistered-classics-pytorch",
        "discovered-tensorflow",
    )
    return [next(row for row in rows if row["zoo"] == zoo) for zoo in zoos]


def _emit_clone(tmp_path: Path) -> tuple[Path, Path]:
    """Emit a real four-campaign records root inside a clone-shaped repo root.

    Parameters
    ----------
    tmp_path:
        Temporary test root.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Clone repo root and its canonical records root.
    """

    repo_root = tmp_path / "clone"
    records_root = repo_root / "menagerie" / "crawler" / "records"
    roster_path = repo_root / "menagerie" / "data" / "crawl_roster.jsonl"
    roster_path.parent.mkdir(parents=True, exist_ok=True)
    roster_path.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in _roster_rows())
    )
    emit_campaign_partitions(roster_path, records_root)
    return repo_root, records_root


def _rotate_manifest_to_prior_checker(records_root: Path) -> Path:
    """Rewrite the manifest's checker projection to the PRE-FLIP binding.

    This reproduces the live pilot's frozen `campaigns.json` exactly: written
    2026-08-11 under Terra, read after the 2026-08-14 flip. Only the projection
    echo moves; every binding digest stays byte-identical.

    Parameters
    ----------
    records_root:
        Emitted records root to rotate in place.

    Returns
    -------
    pathlib.Path
        Manifest path.
    """

    manifest_path = records_root / "campaigns.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in value["campaigns"]:
        if row["campaign_id"] in _ROTATED_CAMPAIGNS:
            row["checker_model"] = _PRIOR_CHECKER
    manifest_path.write_bytes(canonical_json_bytes(value) + b"\n")
    return manifest_path


def _model_row(stable_id: str, *, checker_model: str, author_model: str) -> JsonObject:
    """Return the minimal provenance-bearing model row this boundary reads.

    Parameters
    ----------
    stable_id:
        Subject identity.
    checker_model, author_model:
        Recorded adjudicating identities.

    Returns
    -------
    dict[str, Any]
        Model revision row shaped for `_assert_author_binding`.
    """

    return {
        "stable_id": stable_id,
        "provenance": {
            "author_model": author_model,
            "author_version": "current",
            "checker_model": checker_model,
            "checker_version": "current",
        },
    }


def _binding_for(campaign_id: str) -> CampaignBinding:
    """Return a synthetic binding carrying the LIVE spec for one campaign.

    Parameters
    ----------
    campaign_id:
        Frozen campaign identifier.

    Returns
    -------
    CampaignBinding
        Binding whose spec is the current (post-flip) frozen spec.
    """

    spec = next(spec for spec in CAMPAIGN_SPECS if spec.campaign_id == campaign_id)
    return CampaignBinding(
        spec=spec,
        row_count=1,
        partition_path=f"partitions/{campaign_id}.jsonl",
        partition_sha256="sha256:" + "0" * 64,
        intake_path=f"intake/intake-{campaign_id}",
        intake_snapshot_id=f"intake-{campaign_id}",
        intake_snapshot_sha256="sha256:" + "1" * 64,
        review_cohort_size=spec.review_checkpoint_at,
    )


# ---------------------------------------------------------------------------
# 1. The entry points survive the rotation.
# ---------------------------------------------------------------------------


def test_checker_model_flip_does_not_brick_catalog_entry_points(tmp_path: Path) -> None:
    """Every reader of a rotated manifest still reads it, and says what drifted.

    The five frames in OPS-04's recorded traceback -- `find_campaign_binding`,
    `_optional_campaign_binding`, `_required_campaign_binding`,
    `_snapshot_driver_config`, `_status_command` -- are exercised through the
    real `cli._snapshot_driver_config` entry, plus the campaign-root binding
    `stratify_rung` and `stage_rung` both go through. None may raise.
    """

    repo_root, records_root = _emit_clone(tmp_path)
    manifest_path = _rotate_manifest_to_prior_checker(records_root)

    bindings = load_campaign_bindings(manifest_path)
    assert [binding.spec.campaign_id for binding in bindings] == [
        spec.campaign_id for spec in CAMPAIGN_SPECS
    ]
    drifted = {
        binding.spec.campaign_id: binding.projection_drift
        for binding in bindings
        if binding.projection_drift
    }
    assert drifted == {campaign: ("checker_model",) for campaign in _ROTATED_CAMPAIGNS}

    # The recorded echo is READABLE -- this is the fact a "re-freeze" would have
    # destroyed, and the only durable answer to "who adjudicated step 100?".
    c1 = next(binding for binding in bindings if binding.spec.campaign_id == "c1-mech")
    assert dict(c1.recorded_projection)["checker_model"] == _PRIOR_CHECKER
    # ...while the LIVE spec is what a new leg runs under (the Fable primary
    # since the 2026-09-01 seat swap).
    assert c1.spec.checker_model == "claude-fable-5-1"

    snapshot = load_intake_snapshot(records_root / c1.intake_path)
    assert find_campaign_binding(manifest_path, snapshot) == c1
    assert campaign_binding_for_snapshot(manifest_path, snapshot) == c1
    assert _campaign_root_binding(records_root, "c1-mech") == c1

    # The OPS-04 stack: `status` is a read-only command and must not be a
    # casualty of a rename. It resolves the CURRENT checker, not the echo.
    config = _snapshot_driver_config(repo_root, snapshot)
    assert config.checker_model == "claude-fable-5-1"
    assert config.author_model == c1.spec.author_model


def test_frozen_binding_fields_are_still_verified_exactly(tmp_path: Path) -> None:
    """The frozen half is unchanged: every binding mutation still refuses.

    Reading projections leniently is only safe if the BYTES stay strict. Each
    mutation below is a separate, load-bearing refusal that must survive.
    """

    _repo_root, records_root = _emit_clone(tmp_path)
    manifest_path = _rotate_manifest_to_prior_checker(records_root)
    pristine = json.loads(manifest_path.read_text(encoding="utf-8"))

    first_partition = records_root / str(pristine["campaigns"][0]["partition_path"])
    partition_bytes = first_partition.read_bytes()

    def refuses(mutate: Any) -> None:
        value = json.loads(json.dumps(pristine))
        mutate(value)
        manifest_path.write_bytes(canonical_json_bytes(value) + b"\n")
        with pytest.raises(CampaignPartitionError):
            load_campaign_bindings(manifest_path)
        # Every arm must refuse for its OWN reason, so restore the bytes an arm
        # may have tampered with before the next one runs.
        first_partition.write_bytes(partition_bytes)

    def set_first(field: str, new: Any) -> Any:
        return lambda value: value["campaigns"][0].__setitem__(field, new)

    # A wrong-but-well-formed digest is caught by HASHING the partition, which
    # the loader never did before BL-3: it checked the digest's shape only.
    refuses(set_first("partition_sha256", "sha256:" + "e" * 64))
    refuses(set_first("partition_sha256", "not-a-digest"))
    refuses(
        lambda value: first_partition.write_bytes(b'{"name":"tampered","zoo":"torchvision"}\n')
    )
    # An absent partition is disclosed by default (a merge source ships
    # `records/` without `partitions/`) and REFUSED under the strict reading.
    first_partition.unlink()
    assert len(load_campaign_bindings(manifest_path)) == len(CAMPAIGN_SPECS)
    with pytest.raises(CampaignPartitionError, match="partition bytes are absent"):
        load_campaign_bindings(manifest_path, require_frozen_bytes=True)
    first_partition.write_bytes(partition_bytes)

    refuses(set_first("intake_snapshot_sha256", "not-a-digest"))
    refuses(set_first("row_count", 999_999))
    refuses(set_first("intake_path", ""))
    refuses(lambda value: value["campaigns"][1].__setitem__(
        "intake_snapshot_id", value["campaigns"][0]["intake_snapshot_id"]
    ))
    refuses(lambda value: value["proof"].__setitem__("pairwise_disjoint", False))
    refuses(lambda value: value["roster"].__setitem__("sha256", "nope"))
    refuses(lambda value: value.__setitem__("format", "menagerie.crawler.campaigns.v0"))
    refuses(lambda value: value["campaigns"].pop())
    # Projections are read, not ignored: a malformed one is still malformed.
    refuses(set_first("phase", ""))
    refuses(set_first("review_checkpoint_at", -1))
    refuses(set_first("workload_class", 7))
    refuses(lambda value: value["campaigns"][0].__setitem__(
        "review_cohort", {"size": 3, "stratified_by": [], "source_order": "partition-prefix"}
    ))

    # And the sha256 digest mismatch is caught against BYTES, not against a
    # remembered constant: restore the digest and the manifest reads again.
    manifest_path.write_bytes(canonical_json_bytes(pristine) + b"\n")
    assert len(load_campaign_bindings(manifest_path)) == len(CAMPAIGN_SPECS)


# ---------------------------------------------------------------------------
# 2 + 3. Prior identities are adjudicable; undeclared ones are not.
# ---------------------------------------------------------------------------


def test_prior_checker_identity_rows_remain_adjudicable() -> None:
    """Rows recorded under a DECLARED prior checker pass the authority boundary.

    These are the live pilot's 10 c1-mech heads, m4334 and m5273 among them. A
    record that truthfully names the model that judged it is evidence, not drift.
    """

    binding = _binding_for("c1-mech")
    records = [
        _model_row("m4334", checker_model=_PRIOR_CHECKER, author_model="claude-fable-5"),
        _model_row("m5273", checker_model=_PRIOR_CHECKER, author_model="claude-sonnet"),
        _model_row("m9999", checker_model="gpt-5.6-sol", author_model="claude-fable-5"),
    ]

    assert _assert_author_binding("c1-mech", records, binding) == ("current", "current")


def test_undeclared_checker_identity_rows_still_refuse() -> None:
    """The closed set is the point: an unknown checker label still fails closed."""

    binding = _binding_for("c1-mech")
    records = [_model_row("m4334", checker_model="gpt-6-unknown", author_model="claude-fable-5")]

    with pytest.raises(CampaignMergeError, match="checker_model"):
        _assert_author_binding("c1-mech", records, binding)

    # c3-classics never ran under Terra, so Terra is undeclared THERE.
    assert _PRIOR_CHECKER not in TIER_CAMPAIGN_CHECKER_MODEL_HISTORY["c3-classics"]
    with pytest.raises(CampaignMergeError, match="checker_model"):
        _assert_author_binding(
            "c3-classics",
            [_model_row("m1", checker_model=_PRIOR_CHECKER, author_model="claude-opus-5")],
            _binding_for("c3-classics"),
        )


def test_unregistered_rename_refuses_and_names_the_remedy(tmp_path: Path) -> None:
    """An unregistered rename fails closed with the exact lever in the message.

    OPS-04's finding was not only that the refusal fired everywhere -- it was
    that the message named a field and no remedy, making the operator's first
    contact a research task.
    """

    _repo_root, records_root = _emit_clone(tmp_path)
    manifest_path = records_root / "campaigns.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["campaigns"][0]["checker_model"] = "gpt-7-unannounced"
    manifest_path.write_bytes(canonical_json_bytes(value) + b"\n")

    with pytest.raises(CampaignPartitionError) as refusal:
        load_campaign_bindings(manifest_path)
    message = str(refusal.value)
    assert "gpt-7-unannounced" in message
    assert "outside its closed declared history" in message
    assert "TIER_CAMPAIGN_CHECKER_MODEL_HISTORY" in message
    assert "stage_rung refresh-manifest" in message


def test_declared_history_agrees_with_the_live_specs() -> None:
    """Both closed histories cover every campaign and END at the live binding.

    "The current binding is always the last entry" is the invariant every reader
    of these tables assumes; a rotation that appends in the wrong place, or
    forgets to append at all, fails here rather than at a merge.
    """

    campaign_ids = {spec.campaign_id for spec in CAMPAIGN_SPECS}
    for field, table in (
        ("author_model", TIER_CAMPAIGN_AUTHOR_MODEL_HISTORY),
        ("checker_model", TIER_CAMPAIGN_CHECKER_MODEL_HISTORY),
    ):
        assert set(table) == campaign_ids, f"{field} history does not cover the campaigns"
        for spec in CAMPAIGN_SPECS:
            history = table[spec.campaign_id]
            assert history, f"{spec.campaign_id} {field} history is empty"
            assert len(history) == len(set(history)), f"{spec.campaign_id} {field} repeats"
            assert history[-1] == getattr(spec, field), (
                f"{spec.campaign_id} {field} history must end at the live binding "
                f"{getattr(spec, field)!r}, not {history[-1]!r}"
            )
            assert declared_identity_history(spec.campaign_id, field) == history

    # The pre-flip epoch the pilot's records carry must stay declared for the
    # campaigns whose frozen manifest names it -- deleting it re-bricks them.
    for campaign_id in _ROTATED_CAMPAIGNS:
        assert _PRIOR_CHECKER in TIER_CAMPAIGN_CHECKER_MODEL_HISTORY[campaign_id]

    with pytest.raises(CampaignPartitionError, match="not an identity projection"):
        declared_identity_history("c1-mech", "workload_class")


# ---------------------------------------------------------------------------
# 5 + 6. Field classification, and leg attestation of the projections.
# ---------------------------------------------------------------------------


def test_every_campaign_spec_field_is_classified() -> None:
    """A new spec field must be declared binding or projection, never neither.

    BL-3 happened because nobody had written down which manifest fields address
    frozen bytes and which are a mutable echo. An unclassified field would
    silently inherit whichever behavior the loader happened to give it.
    """

    spec_fields = set(CAMPAIGN_SPECS[0].__dataclass_fields__) - {"campaign_id"}
    binding_fields = set(CAMPAIGN_SPEC_BINDING_FIELDS)
    projection_fields = set(CAMPAIGN_SPEC_PROJECTION_FIELDS)

    assert spec_fields == projection_fields, (
        "every CampaignSpec field except campaign_id is a PROJECTION; a new one "
        "must be classified in menagerie.crawler.constants"
    )
    assert not binding_fields & projection_fields, "a field cannot be both kinds"
    assert set(CAMPAIGN_SPEC_IDENTITY_PROJECTIONS) <= projection_fields

    # The binding half is exactly the CampaignBinding's own frozen-byte fields.
    assert binding_fields <= set(CampaignBinding.__dataclass_fields__)


def _leg_config_detail_keys() -> frozenset[str]:
    """Return the detail keys `Driver._leg_config_details` emits, read from source.

    Read statically on purpose: this census must hold without constructing a
    driver, so a leg-config regression cannot hide behind an unavailable fixture.

    Returns
    -------
    frozenset[str]
        String constants used as dict keys inside the emitter.
    """

    tree = ast.parse((_CRAWLER_ROOT / "driver.py").read_text(encoding="utf-8"))
    emitter = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_leg_config_details"
    )
    return frozenset(
        key.value
        for node in ast.walk(emitter)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    )


def test_adjudication_steering_projections_are_leg_attested() -> None:
    """Every projection that steers adjudication is durably attested per leg.

    The manifest's own copy is a MUTABLE echo, so after a rotation the manifest
    alone cannot say which model adjudicated which rung. B-09's `leg-config`
    event is the durable answer, and both halves are pinned: the emitter must
    record these keys, and the schema that governs the event must REQUIRE them,
    so neither can be dropped without failing here.

    `workload_class` is deliberately not in the attested set: it is consumed by
    nothing but manifest emission and validation, and `campaign_id` -- which IS
    attested -- recovers it exactly.
    """

    attested = set(CAMPAIGN_SPEC_LEG_ATTESTED_PROJECTIONS)
    assert set(CAMPAIGN_SPEC_IDENTITY_PROJECTIONS) <= attested

    emitted = _leg_config_detail_keys()
    assert attested <= emitted, f"leg-config emitter drops: {sorted(attested - emitted)}"

    from menagerie.crawler.schema import load_schema

    schema = load_schema(OPERATIONAL_EVENT_SCHEMA_VERSION)
    leg_config_clauses = [
        clause
        for clause in schema["allOf"]
        if clause.get("if", {}).get("properties", {}).get("event_kind", {}).get("const")
        == "leg-config"
    ]
    assert len(leg_config_clauses) == 1
    required = set(leg_config_clauses[0]["then"]["properties"]["details"]["required"])
    assert attested <= required, f"leg-config schema does not require: {sorted(attested - required)}"

    # `workload_class` is the one projection that is NOT attested, and the reason
    # must stay true: the attested `campaign_id` determines it.
    assert "workload_class" not in attested
    assert len({spec.workload_class for spec in CAMPAIGN_SPECS}) == len(CAMPAIGN_SPECS)


# ---------------------------------------------------------------------------
# 7. The non-versioned-identity census.
# ---------------------------------------------------------------------------


_IDENTITY_FIELD_TOKENS = ("version", "identity", "model", "epoch", "revision", "build", "digest")


def _unpinned_identity_sites() -> tuple[dict[str, str], ...]:
    """Census every production site assigning a non-versioned literal to an identity.

    Scans the crawler package (excluding its own tests) for a string constant
    from :data:`NON_VERSIONED_IDENTITY_LITERALS` bound to an identity-shaped name,
    whether by assignment, annotated assignment, or keyword argument. Line
    numbers are deliberately NOT part of a site's key -- they churn on every
    unrelated edit -- so the inventory pins WHAT is unpinned, not where it moved.

    Returns
    -------
    tuple[dict[str, str], ...]
        Sorted ``{path, symbol, literal}`` site records.
    """

    sites: set[tuple[str, str, str]] = set()
    for path in sorted(_CRAWLER_ROOT.rglob("*.py")):
        relative = path.relative_to(_REPOSITORY_ROOT).as_posix()
        if "/tests/" in relative:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name: Any = None
            value: Any = None
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name, value = node.target.id, node.value
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                name, value = node.targets[0].id, node.value
            elif isinstance(node, ast.keyword) and node.arg:
                name, value = node.arg, node.value
            if not isinstance(name, str) or not isinstance(value, ast.Constant):
                continue
            if not isinstance(value.value, str) or value.value not in (
                NON_VERSIONED_IDENTITY_LITERALS
            ):
                continue
            if not any(token in name.lower() for token in _IDENTITY_FIELD_TOKENS):
                continue
            sites.add((relative, name, value.value))
    return tuple(
        {"path": path, "symbol": symbol, "literal": literal}
        for path, symbol, literal in sorted(sites)
    )


def test_no_new_load_bearing_identity_is_a_non_versioned_literal() -> None:
    """The unpinned-identity surface is closed and may only ever SHRINK.

    A sole authority cannot be identified over a multi-week run by an alias plus
    the word ``current``: all 344 lifetime gate rows and all 298 live model
    revisions carry ``checker_version="current"``, so a provider-side model
    update, a Codex CLI upgrade, or a prompt edit is indistinguishable from no
    change at all, and the W4 reuse key's version component is vacuous.

    The surviving sites are committed with the cluster that owns each file, and
    this test fails on any site the inventory does not name -- including a site
    the inventory names that no longer exists, so a fix must delete its row in
    the same reviewed change. The inventory grants nothing: it is a ceiling on
    an already-disclosed defect, never a licence to add another.
    """

    inventory = json.loads(_UNPINNED_INVENTORY_PATH.read_text(encoding="utf-8"))
    assert inventory["schema_version"] == "menagerie.crawler.unpinned-identity-inventory.v1"

    committed = {
        (row["path"], row["symbol"], row["literal"]): row for row in inventory["sites"]
    }
    observed = {(row["path"], row["symbol"], row["literal"]) for row in _unpinned_identity_sites()}

    new_sites = sorted(observed - set(committed))
    assert not new_sites, (
        "new unpinned load-bearing identity literal(s): "
        f"{new_sites}. Pin the identity; do not extend the inventory."
    )
    stale_rows = sorted(set(committed) - observed)
    assert not stale_rows, (
        f"inventory names site(s) that no longer exist: {stale_rows}. "
        "Delete the row in the same change that pinned the identity."
    )

    # Every surviving row must say who owns it and how it gets pinned, or the
    # inventory is a list of excuses rather than a work queue.
    for key, row in sorted(committed.items()):
        assert row["owner_cluster"], f"{key} has no owning cluster"
        assert row["remedy"], f"{key} has no remedy"
        assert row["literal"] in NON_VERSIONED_IDENTITY_LITERALS

    # And the census must actually see: a mutation-shaped probe reads as a site.
    assert ("menagerie/crawler/driver_contracts.py", "checker_version", "current") in committed


# ---------------------------------------------------------------------------
# The disclosure tool.
# ---------------------------------------------------------------------------


def _write_model_ledger(records_root: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write a raw model ledger the census can read without schema validation."""

    path = records_root / "models" / "current-shard.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def test_identity_epoch_census_discloses_prior_epochs_and_refuses_undeclared(
    tmp_path: Path,
) -> None:
    """The one-shot census reports epochs, unpinned components, and manifest drift.

    This is the disclosure half of BL-3's carve-out: no records migration ran,
    because rewriting `provenance.checker_model` would destroy the only durable
    record of which model actually adjudicated a row. Instead the prior epoch is
    declared and DISCLOSED, and an undeclared one exits 2.
    """

    _repo_root, records_root = _emit_clone(tmp_path)
    _rotate_manifest_to_prior_checker(records_root)
    _write_model_ledger(
        records_root,
        [
            _model_row("m4334", checker_model=_PRIOR_CHECKER, author_model="claude-fable-5"),
            _model_row("m5273", checker_model=_PRIOR_CHECKER, author_model="claude-sonnet"),
            _model_row("m9999", checker_model="gpt-5.6-sol", author_model="claude-fable-5"),
            _model_row("m9998", checker_model="claude-fable-5-1", author_model="claude-fable-5"),
        ],
    )

    census = audit_records_root(records_root, campaign_id="c1-mech")
    assert census["verdict"] == "all-identities-declared"
    assert not census["member_reasons"]["undeclared_identity"]
    classifications = {
        (row["epoch"]["author_model"], row["epoch"]["checker_model"]): row["classification"]
        for row in census["campaigns"][0]["epochs"]
    }
    assert classifications[("claude-fable-5", _PRIOR_CHECKER)]["checker_model"] == (
        "declared-prior"
    )
    # The 2026-09-01 seat swap made sol a DECLARED PRIOR epoch (it stays seated
    # as the K=2 confirmation, but the primary binding is Fable 5.1).
    assert classifications[("claude-fable-5", "gpt-5.6-sol")]["checker_model"] == (
        "declared-prior"
    )
    assert classifications[("claude-fable-5", "claude-fable-5-1")]["checker_model"] == (
        "declared-current"
    )
    assert classifications[("claude-sonnet", _PRIOR_CHECKER)]["author_model"] == "declared-prior"
    # The unpinned version components are disclosed, never silently pinned.
    unpinned = {row["axis"] for row in census["member_reasons"]["unpinned_identity_component"]}
    assert unpinned == {"author_version", "checker_version"}
    assert all(
        row["classification"]["checker_version"] == "unpinned"
        for row in census["campaigns"][0]["epochs"]
    )
    assert census["manifest"]["status"] == "readable"
    assert census["manifest"]["campaigns"]["c1-mech"]["projection_drift"] == ["checker_model"]
    assert audit_main(["--records-root", str(records_root), "--campaign", "c1-mech"]) == 0

    # An UNDECLARED epoch is a refusal, not a footnote.
    _write_model_ledger(
        records_root,
        [_model_row("m4334", checker_model="gpt-6-unknown", author_model="claude-fable-5")],
    )
    undeclared = audit_records_root(records_root, campaign_id="c1-mech")
    assert undeclared["verdict"] == "undeclared-identity"
    assert [row["axis"] for row in undeclared["member_reasons"]["undeclared_identity"]] == [
        "checker_model"
    ]
    assert audit_main(["--records-root", str(records_root), "--campaign", "c1-mech"]) == 2

    # An operator tool refuses typed; it never tracebacks.
    with pytest.raises(CheckerIdentityAuditError):
        audit_records_root(tmp_path / "absent")
    assert audit_main(["--records-root", str(tmp_path / "absent")]) == 2
