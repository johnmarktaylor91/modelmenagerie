"""T1-06: dependency-refresh work generations bind the live authority epoch.

``refresh_work_id`` used to hash only rotation-invariant facts (kind, stable_id,
prior revision, intake snapshot), so a relaunch that rotated the author/checker/
policy authorities resumed the SAME campaign root, and gate history recorded
under the OLD world -- leg 9's persisted rejected metadata rounds are the live
example -- could cap or fingerprint-terminalize the NEW world's proposal before
any current gate adjudicated it. The fix binds
:func:`menagerie.crawler.authority.authority_epoch_identity` into the refresh
preimage: rotation now means a fresh generation, and old-world verdicts match
nothing in it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.authority import authority_epoch_identity
from menagerie.crawler.constants import MODEL_SCHEMA_VERSION_V3
from menagerie.crawler.driver_models import _terminal_metadata_gate
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import JsonlLedger
from menagerie.crawler.reducer import CanonicalReducer, ReductionError
from menagerie.crawler.tests.conftest import make_authority_context, make_model

pytestmark = pytest.mark.smoke


def _seed_prior_history(tmp_path: Path, snapshot: Any, stable_id: str) -> None:
    """Persist one prior model revision so refresh sees rejected history."""

    from menagerie.crawler.tests.test_slice_f_driver import _paths

    paths = _paths(tmp_path, snapshot)
    paths.ledgers.models.parent.mkdir(parents=True, exist_ok=True)
    record = make_model(stable_id, accepted=False, status_code="failed:source")
    record.pop("ledger_seq", None)
    record.pop("payload_sha256", None)
    with JsonlLedger(paths.ledgers.models, MODEL_SCHEMA_VERSION_V3) as ledger:
        ledger.append(record)


def _refresh_id(tmp_path: Path, *, checker_model: str) -> str:
    """Return the dependency-refresh work id one driver world derives."""

    from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    if not (tmp_path / "seeded").exists():
        _seed_prior_history(tmp_path, snapshot, stable_id)
        (tmp_path / "seeded").touch()
    driver = _driver(tmp_path, snapshot, checker_model=checker_model)
    work = driver._ordered_work(snapshot, {})
    assert len(work) == 1
    refresh = work[0].refresh_work_id
    assert refresh is not None
    return str(refresh)


def test_refresh_work_id_rotates_with_the_authority_epoch(tmp_path: Path) -> None:
    """Same history, rotated checker authority: the work generation must move.

    RED before the binding: both worlds derived the identical id from the
    rotation-invariant preimage.
    """

    world_a = _refresh_id(tmp_path, checker_model="gpt-5.6-sol")
    world_b = _refresh_id(tmp_path, checker_model="gpt-6-rotated")
    world_a_again = _refresh_id(tmp_path, checker_model="gpt-5.6-sol")
    assert world_a == world_a_again
    assert world_a != world_b


def test_epoch_identity_covers_every_authority_axis() -> None:
    """Rotating ANY contract identity moves the epoch; identity contexts agree."""

    context = make_authority_context(["m_example"])
    baseline = authority_epoch_identity(context)
    assert baseline == authority_epoch_identity(make_authority_context(["m_example"]))
    for field in (
        "author_prompt_identity",
        "author_model_identity",
        "author_schema_identity",
        "author_dispatcher_identity",
        "checker_prompt_identity",
        "checker_model_identity",
        "checker_schema_identity",
        "reducer_policy_identity",
        "runner_policy_identity",
        "terminal_policy_identity",
        "publication_policy_identity",
    ):
        rotated = replace(context, **{field: "sha256:" + "e" * 64})
        assert authority_epoch_identity(rotated) != baseline, field

    for field in ("author_model_fields", "checker_model_fields"):
        preimage = dict(getattr(context, field))
        for preimage_field in ("provider", "model", "version", "prompt_sha256"):
            forged = dict(preimage)
            forged[preimage_field] = f"rotated-{preimage_field}"
            rotated = replace(context, **{field: forged})
            assert authority_epoch_identity(rotated) != baseline, (
                field,
                preimage_field,
            )


def test_old_world_gates_cannot_terminalize_a_new_epoch_generation(
    tmp_path: Path,
) -> None:
    """The S2 repro, inverted: A-world rejected rounds die at the epoch boundary.

    Two identical rejected metadata rounds under the epoch-A campaign root are a
    fingerprint fast-terminal for THAT root. After the rotation issues a new
    generation, the same history must prove nothing about the new campaign root.
    """

    def rejected_round(gate_round: int, campaign_root: str) -> dict[str, Any]:
        return {
            "gate_id": f"gate-md-{gate_round}",
            "gate_kind": "metadata_batch",
            "gate_round": gate_round,
            "ledger_seq": gate_round,
            "items": [
                {
                    "stable_id": "m_example",
                    "work_id": campaign_root,
                    "campaign_root_work_id": campaign_root,
                    "verdict": "inaccurate",
                    "integrity": {"verdict": "accurate"},
                    "rung_check": {"verdict": "accurate"},
                    "required_repairs": ["identical complaint"],
                }
            ],
        }

    old_root = _refresh_id(tmp_path, checker_model="gpt-5.6-sol")
    new_root = _refresh_id(tmp_path, checker_model="gpt-6-rotated")
    assert old_root != new_root
    persisted = [rejected_round(1, old_root), rejected_round(2, old_root)]
    assert _terminal_metadata_gate(persisted, "m_example", old_root, max_repairs=2) is not None
    assert _terminal_metadata_gate(persisted, "m_example", new_root, max_repairs=2) is None


@pytest.mark.parametrize(
    "field",
    (
        "author_model_fields",
        "checker_model_fields",
        "prior_intake_snapshots",
        "prior_intake_member_ids",
    ),
)
def test_locked_reducer_refuses_non_environment_context_mutation(
    tmp_path: Path, field: str
) -> None:
    """A locked reducer may refresh only materialized environment generations."""

    context = make_authority_context(["m_example"])
    if field.endswith("_fields"):
        changed = dict(getattr(context, field))
        changed["version"] = "rotated-during-lock"
    elif field == "prior_intake_snapshots":
        changed = {"intake-prior": "sha256:" + "e" * 64}
    else:
        changed = frozenset({"m_prior"})
    paths = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )
    with CanonicalReducer(paths, context) as reducer:
        with pytest.raises(
            ReductionError,
            match="locked reducer context changed a non-environment authority axis",
        ):
            reducer.update_context(replace(context, **{field: changed}))
