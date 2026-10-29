"""AB-30 census: every Wave-2 batch-touched schema seam carries its matrix.

The committed declaration registry
(``mixed_era/wave2_batch_declarations.json``) enumerates every schema seam
the Wave-2 rotation batch touches, sourced from the batch lanes' design
notes. This census is the enforcement:

- a ``prior-era`` seam either LANDS its reader x era matrix module here
  (with digest-attested pre-batch fixtures captured at the freeze tree) or
  names the exact batch lane that owes it;
- a ``no-prior-era`` seam (a record class born inside the batch) owes a
  typed prior-artifact refusal proof, registered by node id, before the
  batch seal;
- the ``first-generation`` class is EXACTLY the four W16-owned surfaces of
  PLAN_v6's first-generation rule, each already carrying its typed-refusal
  proof node -- a fifth self-declared member fails this census.

The default run adjudicates the pre-batch obligations. With
``MENAGERIE_WAVE2_ASSEMBLY=1`` (the batch-assembly run) every owed matrix
and every empty refusal-proof list REFUSES: nothing rotation-bearing merges
with an undischarged mixed-era obligation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from menagerie.crawler.tests import mixed_era
from menagerie.crawler.tests.mixed_era import (
    MixedEraRegistryError,
    assembly_mode,
    fixture_path,
    load_manifest,
    load_registry,
)

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: PLAN_v6's first-generation rule is a closed FOUR-member enumeration.
# PLAN_v6 enumerated four first-generation W16 surfaces. RC-3 (certification by
# closure) retired three with the launch certificate -- the certificate-kind
# vocabulary, the certificate required-input manifest, and the blocking-tiers
# census membership -- so the closed class is now the one surviving surface.
FIRST_GENERATION_SEAMS = frozenset({"blocking-resolution-vocabulary"})


def test_registry_parses_strictly() -> None:
    """The registry is itself a closed vocabulary and parses strictly."""

    document = load_registry()
    assert document["freeze"]["tree_sha"]
    assert len(document["seams"]) >= 20


def test_registry_refuses_unknown_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreviewed registry shape refuses typed, never half-parses."""

    document = json.loads(
        (Path(mixed_era.__file__).parent / "wave2_batch_declarations.json").read_text()
    )
    mutations = [
        {"schema_version": "menagerie.crawler.wave2-mixed-era-registry.v99"},
        {"freeze": {}},
        {"seams": []},
    ]
    for mutation in mutations:
        mutated = {**document, **mutation}
        target = tmp_path / "registry.json"
        target.write_text(json.dumps(mutated))
        monkeypatch.setattr(mixed_era, "_REGISTRY_PATH", target)
        with pytest.raises(MixedEraRegistryError):
            load_registry()
    # unknown era class on one seam
    mutated = json.loads(json.dumps(document))
    mutated["seams"][0]["era_class"] = "era-from-the-future"
    target = tmp_path / "registry.json"
    target.write_text(json.dumps(mutated))
    monkeypatch.setattr(mixed_era, "_REGISTRY_PATH", target)
    with pytest.raises(MixedEraRegistryError):
        load_registry()


def test_every_prior_era_seam_carries_its_matrix() -> None:
    """Landed matrices exist with attested fixtures; owed rows name their lane."""

    document = load_registry()
    manifest = load_manifest()
    entries = dict(manifest["entries"])
    assert manifest["tree_sha"] == document["freeze"]["tree_sha"], (
        "fixture manifest was captured at a different tree than the declared freeze"
    )
    owed: list[str] = []
    for seam in document["seams"]:
        if seam["era_class"] != "prior-era":
            continue
        if seam["matrix_disposition"] == "owed-by-batch-lane":
            owed.append(f"{seam['seam_id']} (owed by {seam['owed_by']})")
            continue
        module = _REPO_ROOT / str(seam["matrix_module"])
        assert module.is_file(), f"{seam['seam_id']}: matrix module missing: {module}"
        assert seam.get("readers"), f"{seam['seam_id']}: landed matrix declares no readers"
        for name in seam.get("fixtures", ()):
            path = fixture_path(str(name))
            assert path.is_file(), f"{seam['seam_id']}: fixture missing: {name}"
            digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            assert entries.get(str(name)) == digest, (
                f"{seam['seam_id']}: fixture {name} contradicts the frozen manifest"
            )
    if assembly_mode():
        assert not owed, "batch assembly with undischarged mixed-era matrices: " + "; ".join(owed)


def test_no_prior_era_seams_owe_refusal_proofs() -> None:
    """A record class born in the batch registers its typed refusal proof."""

    document = load_registry()
    for seam in document["seams"]:
        if seam["era_class"] != "no-prior-era":
            continue
        nodes = list(seam.get("refusal_proof_nodes", ()))
        if assembly_mode():
            assert nodes, (
                f"{seam['seam_id']}: no-prior-era seam reached batch assembly without "
                "a registered typed prior-artifact refusal proof node"
            )
        for node in nodes:
            _assert_node_resolves(str(node), seam_id=str(seam["seam_id"]))


def test_first_generation_class_is_exactly_the_surviving_w16_surface() -> None:
    """Machine-check the closed first-generation enumeration (PLAN_v6 minus RC-3's retirements)."""

    document = load_registry()
    declared = {
        str(seam["seam_id"])
        for seam in document["seams"]
        if seam["era_class"] == "first-generation"
    }
    assert declared == FIRST_GENERATION_SEAMS
    for seam in document["seams"]:
        if seam["era_class"] != "first-generation":
            continue
        nodes = list(seam.get("refusal_proof_nodes", ()))
        assert nodes, f"{seam['seam_id']}: first-generation surface lacks its proof node"
        for node in nodes:
            _assert_node_resolves(str(node), seam_id=str(seam["seam_id"]))


def test_matrix_modules_and_registry_are_bijective() -> None:
    """Every test_mixed_era_* matrix module is declared by exactly one seam."""

    tests_dir = Path(__file__).resolve().parent
    on_disk = {
        str(path.relative_to(_REPO_ROOT))
        for path in tests_dir.glob("test_mixed_era_*.py")
        if path.name != Path(__file__).name
    }
    document = load_registry()
    declared: list[str] = [
        str(seam["matrix_module"]) for seam in document["seams"] if seam.get("matrix_module")
    ]
    assert sorted(declared) == sorted(set(declared)), "matrix module declared twice"
    assert set(declared) == on_disk, (
        f"matrix modules and registry disagree: only-declared="
        f"{sorted(set(declared) - on_disk)} only-on-disk={sorted(on_disk - set(declared))}"
    )


def test_ic9_pairing_consumer_is_wired() -> None:
    """The registry names the IC-9 batch run and the census tool imports."""

    document = load_registry()
    assert "schema_migration_census" in document["freeze"]["ic9_live_root_command"]
    from menagerie.crawler.tools.schema_migration_census import (  # noqa: F401
        schema_migration_census,
    )


def _assert_node_resolves(node: str, *, seam_id: str) -> None:
    """Assert a registered proof node names a real test in a real module."""

    module_path, _, test_name = node.partition("::")
    module = _REPO_ROOT / module_path
    assert module.is_file(), f"{seam_id}: proof node module missing: {module_path}"
    assert test_name and f"def {test_name.split('[')[0]}(" in module.read_text(), (
        f"{seam_id}: proof node {node} does not resolve to a test definition"
    )
