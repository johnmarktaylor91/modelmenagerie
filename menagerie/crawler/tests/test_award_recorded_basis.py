"""AW-2 — recorded-basis currency for ``runs`` heads (AUDIT100 SYNAUD_A section 2.2).

Incident evidence these tests are RED against (base ``2f22635f``): commit
``51cdae77`` rotated ``publication_policy_identity`` (``ccbef701 -> af2ea992``)
and the dependency-currency projection marked every settled award stale, so
the schedulers re-drove 20 recorded members (19 award holders) and three
verified awards were superseded by failures. An award is a claim about a
specific executed artifact under recorded closures; rotating live code changes
what NEW work would do, never what happened. The recorded-basis carve-out
already exists for review-held records; these tests pin its extension to
EVERY settled head. Since RC-2 the pin is UNCONDITIONAL: the projection reads
no rotation ledger, a refinement, soundness, or unmeasured move never stales
or holds a head, and the move's measured class feeds the recheck census
instead (``test_recheck_census.py``). The frozen operator registry
(``rotation_registry.py``) keeps its read-side resolution semantics here
because the RC-2 seed imports it; environment-generation resolution against
the ledger-attested set is unchanged (historical is not stale -- the
m9819/m9911 publication-seam refusals were exactly an attested generation read
as absent).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import project_dependency_current
from menagerie.crawler.rotation_registry import (
    RotationRegistrationError,
    append_rotation_registration,
    build_rotation_registration,
    load_rotation_registrations,
    registered_move_class,
    rotation_edge_census,
    rotation_registry_path,
)
from menagerie.crawler.tests.conftest import (
    bind_terminal_attempts,
    make_attempt,
    make_authority_context,
    make_failed_attempt,
    make_gate,
    make_model,
)
from menagerie.crawler.tests.test_reducer import CanonicalReducer

pytestmark = pytest.mark.smoke

_ROTATED = "sha256:" + "b" * 64


def _paths(tmp_path: Path) -> LedgerPaths:
    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def _award_head(paths: LedgerPaths) -> dict:
    """Append one released runs award and return the persisted head."""

    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_gate(make_gate(["m_example"]))
        reducer.append_attempt(make_attempt())
        return reducer.append_model(make_model(accepted=True)).record


def _failed_forward_head(
    paths: LedgerPaths, *, environment_generation: str | None = None
) -> dict:
    """Append one failed:forward terminal and return the persisted head."""

    attempt = make_failed_attempt(stage="forward", reason_code="exception")
    model = make_model(status_code="failed:forward")
    model["status"]["reason_code"] = "exception"
    if environment_generation is not None:
        # A real failed:forward ran IN an environment; the synthetic failure
        # fixture nulls the observed-execution facts, so restore exactly the
        # observed-environment minimum (env block + stream digests), attest
        # the generation the test needs, and rebuild the parent attestation
        # over the mutated observation with the fixture's own recipe.
        attempt["environment"] = make_attempt()["environment"]
        attempt["environment"]["family"] = "core"
        attempt["identities"]["environment"] = environment_generation
        attempt["supervisor_observation"].update(
            {"stdout_sha256": hash_bytes(b""), "stderr_sha256": hash_bytes(b"")}
        )
        parent = dict(attempt["parent_attestation"])
        parent.pop("attestation_sha256")
        parent.update(
            {
                "stdout_sha256": attempt["supervisor_observation"]["stdout_sha256"],
                "stderr_sha256": attempt["supervisor_observation"]["stderr_sha256"],
            }
        )
        parent["attestation_sha256"] = stable_hash(parent)
        attempt["parent_attestation"] = parent
        model["execution"]["env_generation"] = environment_generation
    bind_terminal_attempts(model, [attempt])
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        return reducer.append_model(model).record


def _registration(*, from_identity: str, to_identity: str, rotation_class: str) -> dict:
    return build_rotation_registration(
        axis="publication_policy_identity",
        from_identity=from_identity,
        to_identity=to_identity,
        rotation_class=rotation_class,
        rationale="test rotation",
        registered_by="operator-test",
        created_at="2026-08-16T00:00:00Z",
    )


def test_rotation_registry_rows_are_fail_closed_and_idempotent(tmp_path: Path) -> None:
    """The registry validates closed vocabularies and refuses conflicting bytes."""

    path = rotation_registry_path(tmp_path)
    row = _registration(
        from_identity="sha256:" + "a" * 64, to_identity=_ROTATED, rotation_class="refinement"
    )
    assert append_rotation_registration(path, row) is True
    assert append_rotation_registration(path, row) is False
    assert load_rotation_registrations(path) == (row,)

    conflicting = dict(row)
    conflicting["rationale"] = "different bytes under the same identity"
    with pytest.raises(RotationRegistrationError, match="conflicting bytes"):
        append_rotation_registration(path, conflicting)

    with pytest.raises(RotationRegistrationError, match="outside the closed"):
        build_rotation_registration(
            axis="not-an-axis",
            from_identity="sha256:" + "a" * 64,
            to_identity=_ROTATED,
            rotation_class="refinement",
            rationale="r",
            registered_by="o",
            created_at="2026-08-16T00:00:00Z",
        )
    with pytest.raises(RotationRegistrationError, match="unknown rotation class"):
        build_rotation_registration(
            axis="publication_policy_identity",
            from_identity="sha256:" + "a" * 64,
            to_identity=_ROTATED,
            rotation_class="cosmetic",
            rationale="r",
            registered_by="o",
            created_at="2026-08-16T00:00:00Z",
        )
    # An unreadable registry refuses loudly rather than degrading to
    # "nothing registered" (which would silently flip refinements to
    # fail-closed soundness).
    path.write_text(path.read_text() + '{"schema_version": "junk"}\n')
    with pytest.raises(RotationRegistrationError):
        load_rotation_registrations(path)


def test_refinement_chains_compose_and_a_soundness_link_poisons() -> None:
    """Classification follows registered edges: refinement chains, soundness taints."""

    a, b, c = ("sha256:" + ch * 64 for ch in "abc")
    refinements = (
        _registration(from_identity=a, to_identity=b, rotation_class="refinement"),
        _registration(from_identity=b, to_identity=c, rotation_class="refinement"),
    )
    key = {"axis": "publication_policy_identity"}
    assert registered_move_class(refinements, recorded=a, live=c, **key) == "refinement"
    assert registered_move_class(refinements, recorded=a, live=a, **key) == "refinement"
    mixed = (
        refinements[0],
        _registration(from_identity=b, to_identity=c, rotation_class="soundness"),
    )
    assert registered_move_class(mixed, recorded=a, live=c, **key) == "soundness"
    assert registered_move_class((), recorded=a, live=c, **key) is None
    # A different axis's registration governs nothing here.
    assert registered_move_class(refinements, axis="runner_identity", recorded=a, live=c) is None


def test_a_refinement_registered_rotation_keeps_a_runs_head_current(tmp_path: Path) -> None:
    """The incident axis: a registered refinement rotation never stales an award.

    RED at 2f22635f: 32 of 33 close-state awards were projection-stale under
    the campaign's own restored authority, and finish-8 re-authored fresh
    awards the same evening (m4334 seq 262 re-authored at 03:33Z).
    """

    paths = _paths(tmp_path)
    head = _award_head(paths)
    assert head["status"]["kind"] == "runs"
    context = make_authority_context(["m_example"])
    recorded = str(head["dependency_vector"]["publication_policy_identity"])
    rotated = replace(context, publication_policy_identity=_ROTATED)

    # RC-2: no registry row is needed -- the recorded basis pins unconditionally.
    assert recorded != _ROTATED
    after = project_dependency_current(paths, context=rotated)
    assert "m_example" in after.current_records, after.stale_reasons.get("m_example")
    assert after.current_records["m_example"]["record_revision"] == head["record_revision"]


def test_a_refinement_registered_rotation_keeps_a_failed_terminal_current(
    tmp_path: Path,
) -> None:
    """The leg-29 close shape: recorded-basis replay covers FAILED heads too.

    The first clean close after identity-component fixes (aefd46d4 et al.)
    read 139 recorded terminals as stale:terminal_rule_identity /
    stale:author_dispatcher_identity and refused the partition. A failed
    terminal is a claim adjudicated under recorded closures exactly like an
    award; a registered refinement rotation must keep it current, and an
    unregistered rotation must still fail closed (the sibling award tests pin
    that side).
    """

    paths = _paths(tmp_path)
    head = _failed_forward_head(paths)
    assert head["status"]["kind"] == "failed"
    context = make_authority_context(["m_example"])
    recorded = str(head["dependency_vector"]["terminal_rule_identity"])
    rotated = replace(context, terminal_policy_identity=_ROTATED)

    assert recorded != _ROTATED
    after = project_dependency_current(paths, context=rotated)
    assert "m_example" in after.current_records, after.stale_reasons.get("m_example")
    assert after.current_records["m_example"]["record_revision"] == head["record_revision"]


def test_an_evidence_append_never_flips_a_recorded_basis_settled_head(
    tmp_path: Path,
) -> None:
    """The D-47 digest-flap root cause: fast-path parity with the full derivation.

    The 2026-08-27 calibration legs flapped the stale-excluded census between
    26 and 31 members across consecutive launches over BYTE-IDENTICAL ledgers:
    the terminal-review sweep appended gate rows mid-leg, the projection's
    evidence-growth patch path re-replayed exactly the swept settled terminals
    -- but under the LIVE context, because the fast path pinned the recorded
    basis only for review-held records and ``runs`` awards -- and each
    old-basis terminal flipped transiently stale with the
    derivation-contradiction reason. The full derivation (next launch) read
    them current again, so the acknowledgment digest the operator was told to
    pass was refused on alternating legs. The patch path must be
    observationally identical to :func:`project_dependency_current`: an
    evidence append never changes a settled head's currency, two consecutive
    derivations agree, and the D-47 acknowledgment digest is stable across
    the append.
    """

    from menagerie.crawler.census_ack import stale_excluded_ack_digest
    from menagerie.crawler.reducer import CanonicalReducer as ProductionCanonicalReducer

    paths = _paths(tmp_path)
    head = _failed_forward_head(paths)
    context = make_authority_context(["m_example"])
    recorded = str(head["dependency_vector"]["terminal_rule_identity"])
    rotated = replace(context, terminal_policy_identity=_ROTATED)
    append_rotation_registration(
        rotation_registry_path(tmp_path),
        build_rotation_registration(
            axis="terminal_rule_identity",
            from_identity=recorded,
            to_identity=_ROTATED,
            rotation_class="refinement",
            rationale="test rotation",
            registered_by="operator-test",
            created_at="2026-08-27T00:00:00Z",
        ),
    )
    with ProductionCanonicalReducer(paths, rotated) as reducer:
        # Full derivation: the registered refinement pins the recorded basis.
        assert "m_example" in reducer.current_view, reducer.dependency_stale_reasons.get(
            "m_example"
        )
        digest_before = stale_excluded_ack_digest(dict(reducer.dependency_stale_reasons))
        # RED 1: two consecutive derivations agree.
        assert (
            stale_excluded_ack_digest(dict(reducer.dependency_stale_reasons))
            == digest_before
        )
        # RED 2: an evidence append naming the member (the terminal-review
        # sweep shape) routes the next projection through the patch fast path;
        # the settled head must stay current and the digest must not move.
        reducer.append_gate(make_gate(["m_example"]))
        assert "m_example" in reducer.current_view, reducer.dependency_stale_reasons.get(
            "m_example"
        )
        assert (
            stale_excluded_ack_digest(dict(reducer.dependency_stale_reasons))
            == digest_before
        )


def test_an_attested_slotless_generation_keeps_a_failed_terminal_current(
    tmp_path: Path,
) -> None:
    """The leg-29 close's third stale class: environment_generation.

    A failed terminal whose proof-attested generation lost its live context
    slot (family rebuilt or torn down mid-leg) must stay current through the
    ledger-attested generation set -- historical is not stale (AW-2) -- while
    a generation attested nowhere still refuses through the sibling award
    pins.
    """

    paths = _paths(tmp_path)
    rebuilt_generation = "sha256:" + "c" * 64
    head = _failed_forward_head(paths, environment_generation=rebuilt_generation)
    assert head["dependency_vector"]["environment_generation"] == rebuilt_generation
    # The live context maps a DIFFERENT generation for a different family:
    # the recorded generation has no slot anywhere, only ledger attestation.
    context = make_authority_context(["m_example"])
    projection = project_dependency_current(paths, context=context)
    assert "m_example" in projection.current_records, projection.stale_reasons.get(
        "m_example"
    )


def test_no_rotation_class_stales_or_holds_a_head_in_the_projection(tmp_path: Path) -> None:
    """RC-2: unmeasured, refinement, and soundness moves all project CURRENT.

    The award-release-hold is gone from the projection; the move's class is
    the recheck census's business (an unmeasured or soundness move lists the
    head as needs_recheck there, release-excluded, never held here).
    """

    from menagerie.crawler.recheck_census import recheck_census
    from menagerie.crawler.rotations import build_rotation

    paths = _paths(tmp_path)
    head = _award_head(paths)
    context = make_authority_context(["m_example"])
    recorded = str(head["dependency_vector"]["publication_policy_identity"])
    rotated = replace(context, publication_policy_identity=_ROTATED)
    projection = project_dependency_current(paths, context=rotated)
    assert projection.stale_reasons == {}
    assert projection.current_records["m_example"]["record_revision"] == head["record_revision"]
    heads = {"m_example": head}
    unmeasured = recheck_census(heads, (), rotated)
    assert unmeasured["counts"]["needs_recheck"] == 1
    assert unmeasured["needs_recheck"][0]["soundness_basis"] == "unmeasured"
    soundness = build_rotation(
        axis="publication_policy_identity",
        from_identity=recorded,
        to_identity=_ROTATED,
        rotation_class="soundness",
        reason="probe: 1 heads (1 awards) replayed; 1 flips",
        measured_by="predicate_replay",
        tree_sha="deadbeef",
        at="2026-09-01T00:00:00Z",
    )
    measured = recheck_census(heads, (soundness,), rotated)
    assert measured["needs_recheck"][0]["rotation_ids"] == [soundness["rotation_id"]]
    refinement = dict(soundness)
    refinement = build_rotation(
        axis="publication_policy_identity",
        from_identity=recorded,
        to_identity=_ROTATED,
        rotation_class="refinement",
        reason="probe: 1 heads (1 awards) replayed; 0 flips",
        measured_by="predicate_replay",
        tree_sha="deadbeef",
        at="2026-09-01T00:00:00Z",
    )
    assert recheck_census(heads, (refinement,), rotated)["counts"]["needs_recheck"] == 0
    # Sticky: the soundness row beside the refinement row still lists the head.
    assert recheck_census(heads, (refinement, soundness), rotated)["counts"]["needs_recheck"] == 1


def test_a_runs_head_contradicting_its_own_evidence_still_refuses(tmp_path: Path) -> None:
    """S2-2: the carve-out pins policy axes only; evidence cross-checks stay live.

    A runs head whose referenced attempt is gone from the ledger must read
    stale even when every rotation is refinement-registered -- the carve-out
    can never launder a record contradicting its own ledger.
    """

    paths = _paths(tmp_path)
    head = _award_head(paths)
    context = make_authority_context(["m_example"])
    recorded = str(head["dependency_vector"]["publication_policy_identity"])
    rotated = replace(context, publication_policy_identity=_ROTATED)
    assert recorded != _ROTATED
    paths.attempts.write_text("")
    projection = project_dependency_current(paths, context=rotated)
    assert "m_example" not in projection.current_records
    assert "m_example" in projection.stale_reasons


def test_attested_historical_environment_generation_is_not_stale(tmp_path: Path) -> None:
    """Environment resolution: ledger-attested historical generations stand.

    RED at 2f22635f: 31 of the 32 stale close-state awards were stale on
    ``environment_generation`` alone, every one of the five recorded
    generations PRESENT in the ledger-attested set; m9819/m9911 refused at
    the publication seam with "model environment generation is absent from
    active context" while their generation was attested by their own proof
    attempts.
    """

    paths = _paths(tmp_path)
    head = _award_head(paths)
    context = make_authority_context(["m_example"])
    rebuilt = replace(context, environment_generations={"env-test": "sha256:" + "e" * 64})

    projection = project_dependency_current(paths, context=rebuilt)
    assert "m_example" in projection.current_records, projection.stale_reasons.get("m_example")
    assert projection.current_records["m_example"]["record_revision"] == head["record_revision"]


def test_an_environment_generation_attested_nowhere_stays_fail_closed(
    tmp_path: Path,
) -> None:
    """The attested-set resolution refuses a generation no evidence attests."""

    from menagerie.crawler.reducer import _award_recorded_basis_context

    paths = _paths(tmp_path)
    head = _award_head(paths)
    context = make_authority_context(["m_example"])
    rebuilt = replace(context, environment_generations={"env-test": "sha256:" + "e" * 64})
    with pytest.raises(Exception, match="attested"):
        _award_recorded_basis_context(rebuilt, head, attested_bindings={})


# --- Same-edge class conflicts (P5 fix wave; fable-F3 mechanism / sol-04) ---
#
# RED against bd5c7837b: `registered_move_class` resolved refinement
# reachability FIRST, so a later parallel refinement row silently outranked a
# standing soundness registration on the SAME (axis, from, to) edge -- the
# 2026-08-29/30 checker_prompt pair (rotation-fe6dd6ce / rotation-72ec21e4)
# dissolved 46 designed review holds through exactly this channel, and
# `append_rotation_registration` refused nothing (only same-id byte
# conflicts).


def _write_row_raw(path: Path, row: dict) -> None:
    """Land one pre-law row as ledger bytes, bypassing the append door.

    The historical contradiction (both checker_prompt rows) predates the
    conflict law; these tests reproduce that ledger shape to pin that LOAD
    keeps reading it (rows are never deleted) while RESOLUTION reads the
    conflicted edge as soundness.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def test_same_edge_class_conflict_refuses_at_append(tmp_path: Path) -> None:
    """A second class on a registered edge refuses without a resolution row."""

    path = rotation_registry_path(tmp_path)
    a = "sha256:" + "a" * 64
    soundness = _registration(from_identity=a, to_identity=_ROTATED, rotation_class="soundness")
    assert append_rotation_registration(path, soundness) is True
    conflicting = _registration(
        from_identity=a, to_identity=_ROTATED, rotation_class="refinement"
    )
    with pytest.raises(RotationRegistrationError, match="resolution row"):
        append_rotation_registration(path, conflicting)
    # Same-class duplicates on one edge stay lawful (no contradiction).
    duplicate = build_rotation_registration(
        axis="publication_policy_identity",
        from_identity=a,
        to_identity=_ROTATED,
        rotation_class="soundness",
        rationale="re-affirmation",
        registered_by="operator-test",
        created_at="2026-08-17T00:00:00Z",
    )
    assert append_rotation_registration(path, duplicate) is True


def test_pre_law_parallel_class_conflict_reads_as_soundness(tmp_path: Path) -> None:
    """A persisted contradictory edge loads intact and resolves fail-closed.

    The 2026-08-30 corrective row stays in the ledger; its EFFECT is inert
    pending an explicit resolution row -- the edge classifies as soundness,
    never refinement-first.
    """

    path = rotation_registry_path(tmp_path)
    a = "sha256:" + "a" * 64
    soundness = _registration(from_identity=a, to_identity=_ROTATED, rotation_class="soundness")
    corrective = _registration(
        from_identity=a, to_identity=_ROTATED, rotation_class="refinement"
    )
    _write_row_raw(path, soundness)
    _write_row_raw(path, corrective)
    rows = load_rotation_registrations(path)
    assert len(rows) == 2  # never deleted, never refused at load
    key = {"axis": "publication_policy_identity"}
    assert registered_move_class(rows, recorded=a, live=_ROTATED, **key) == "soundness"
    (edge,) = rotation_edge_census(rows)
    assert edge["conflicted"] is True
    assert edge["resolved_class"] == "soundness"
    assert sorted(edge["live_classes"]) == ["refinement", "soundness"]


def test_resolution_row_reclassifies_and_can_retighten(tmp_path: Path) -> None:
    """An evidence-bound supersession row governs; either direction works."""

    path = rotation_registry_path(tmp_path)
    a = "sha256:" + "a" * 64
    soundness = _registration(from_identity=a, to_identity=_ROTATED, rotation_class="soundness")
    corrective = _registration(
        from_identity=a, to_identity=_ROTATED, rotation_class="refinement"
    )
    _write_row_raw(path, soundness)
    _write_row_raw(path, corrective)
    resolution = build_rotation_registration(
        axis="publication_policy_identity",
        from_identity=a,
        to_identity=_ROTATED,
        rotation_class="refinement",
        rationale="trilab ruling: closures acceptance-neutral over the artifact class",
        registered_by="operator-test",
        created_at="2026-08-18T00:00:00Z",
        supersedes=soundness["registration_id"],
        supersession_evidence="ruling-digest sha256:" + "d" * 64,
    )
    assert append_rotation_registration(path, resolution) is True
    rows = load_rotation_registrations(path)
    key = {"axis": "publication_policy_identity"}
    assert registered_move_class(rows, recorded=a, live=_ROTATED, **key) == "refinement"
    (edge,) = rotation_edge_census(rows)
    assert edge["conflicted"] is False
    assert edge["superseded_registration_ids"] == [soundness["registration_id"]]

    # Re-tightening is equally lawful: a later soundness resolution row
    # superseding the refinement rows flips the edge back fail-closed (the
    # v1 resolver was one-way; a soundness row could never re-tighten).
    retighten = build_rotation_registration(
        axis="publication_policy_identity",
        from_identity=a,
        to_identity=_ROTATED,
        rotation_class="soundness",
        rationale="re-verification found a possible prior false acceptance",
        registered_by="operator-test",
        created_at="2026-08-19T00:00:00Z",
        supersedes=resolution["registration_id"],
        supersession_evidence="re-verification sweep sha256:" + "e" * 64,
    )
    assert append_rotation_registration(path, retighten) is True
    rows = load_rotation_registrations(path)
    assert registered_move_class(rows, recorded=a, live=_ROTATED, **key) == "soundness"


def test_resolution_row_binding_is_closed(tmp_path: Path) -> None:
    """supersedes must name a same-edge row; evidence rides with it."""

    path = rotation_registry_path(tmp_path)
    a, c = ("sha256:" + ch * 64 for ch in "ac")
    soundness = _registration(from_identity=a, to_identity=_ROTATED, rotation_class="soundness")
    append_rotation_registration(path, soundness)
    other_edge = _registration(from_identity=c, to_identity=_ROTATED, rotation_class="soundness")
    append_rotation_registration(path, other_edge)

    # Naming a row on a DIFFERENT edge refuses.
    cross = build_rotation_registration(
        axis="publication_policy_identity",
        from_identity=a,
        to_identity=_ROTATED,
        rotation_class="refinement",
        rationale="r",
        registered_by="o",
        created_at="2026-08-18T00:00:00Z",
        supersedes=other_edge["registration_id"],
        supersession_evidence="e",
    )
    with pytest.raises(RotationRegistrationError, match="same .axis, from, to. edge"):
        append_rotation_registration(path, cross)

    # Naming a nonexistent registration refuses.
    dangling = build_rotation_registration(
        axis="publication_policy_identity",
        from_identity=a,
        to_identity=_ROTATED,
        rotation_class="refinement",
        rationale="r",
        registered_by="o",
        created_at="2026-08-18T00:00:00Z",
        supersedes="rotation-" + "0" * 24,
        supersession_evidence="e",
    )
    with pytest.raises(RotationRegistrationError, match="not a registered row"):
        append_rotation_registration(path, dangling)

    # A resolution row carries BOTH fields or refuses at build.
    with pytest.raises(RotationRegistrationError, match="supersession_evidence"):
        build_rotation_registration(
            axis="publication_policy_identity",
            from_identity=a,
            to_identity=_ROTATED,
            rotation_class="refinement",
            rationale="r",
            registered_by="o",
            created_at="2026-08-18T00:00:00Z",
            supersedes=soundness["registration_id"],
        )


def test_conflicted_edge_seeds_as_soundness_and_lists_the_award_for_recheck(tmp_path: Path) -> None:
    """Composed (RC-2): a pre-law contradictory edge seeds STICKY soundness.

    The exact production shape: an award recorded under the old identity, a
    soundness registration for the move, then a parallel corrective
    refinement row. The award projects CURRENT (no hold anywhere since RC-2)
    and the recheck census lists it as needs_recheck on the seeded soundness
    row -- never released refinement-first.
    """

    from menagerie.crawler.recheck_census import recheck_census
    from menagerie.crawler.rotations import load_rotations, seed_from_registry

    paths = _paths(tmp_path)
    head = _award_head(paths)
    context = make_authority_context(["m_example"])
    recorded = str(head["dependency_vector"]["publication_policy_identity"])
    rotated = replace(context, publication_policy_identity=_ROTATED)
    registry = rotation_registry_path(tmp_path)
    _write_row_raw(
        registry,
        _registration(from_identity=recorded, to_identity=_ROTATED, rotation_class="soundness"),
    )
    _write_row_raw(
        registry,
        _registration(from_identity=recorded, to_identity=_ROTATED, rotation_class="refinement"),
    )
    assert seed_from_registry(tmp_path, tree_sha="deadbeef")["appended"] == 2
    projection = project_dependency_current(paths, context=rotated)
    assert "m_example" in projection.current_records, projection.stale_reasons
    census = recheck_census({"m_example": head}, load_rotations(tmp_path), rotated)
    (row,) = census["needs_recheck"]
    assert row["soundness_basis"] == "measured"
    assert len(row["rotation_ids"]) == 1


def test_cross_edge_supersedes_is_inert(tmp_path: Path) -> None:
    """A hand-written supersedes naming a row on ANOTHER edge kills nothing.

    The append door refuses cross-edge supersession; a ledger row carrying one
    anyway (pre-law bytes, hand edit) must not dissolve a soundness row
    elsewhere -- that would be an upgrade channel through raw ledger bytes.
    """

    a, c = ("sha256:" + ch * 64 for ch in "ac")
    key = {"axis": "publication_policy_identity"}
    soundness_other_edge = _registration(
        from_identity=c, to_identity=_ROTATED, rotation_class="soundness"
    )
    refinement_other_edge = _registration(
        from_identity=c, to_identity=_ROTATED, rotation_class="refinement"
    )
    cross = build_rotation_registration(
        axis="publication_policy_identity",
        from_identity=a,
        to_identity=_ROTATED,
        rotation_class="refinement",
        rationale="r",
        registered_by="o",
        created_at="2026-08-18T00:00:00Z",
        supersedes=soundness_other_edge["registration_id"],
        supersession_evidence="e",
    )
    rows = (soundness_other_edge, refinement_other_edge, cross)
    # The other edge stays CONFLICTED -> soundness: the cross-edge
    # supersession never discharged its soundness row.
    assert registered_move_class(rows, recorded=c, live=_ROTATED, **key) == "soundness"
