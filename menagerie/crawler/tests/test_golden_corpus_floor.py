"""W-7: the golden-corpus membership floor (rule-2 trilab consensus, 2026-08-29).

The calibration corpus had no membership lock, so "re-harvest a clearer
exemplar" was a one-line path around every zero bar in the calibration
contract (the m10470 temptation, Opus §7.3). These tests pin the committed
byte-digest manifest and its three consumers:

- name-set equality plus per-file sha256 identity: additions, deletions,
  renames, and in-place rewrites each fail with the named file;
- the PERMANENT retirement census: a disappearing member refuses ``write``
  unless retired with a reason, retirement preserves the digest forever, and
  a retired name reappearing on disk refuses without an explicit reactivation;
- the mint refuses over a corpus that does not match the committed manifest
  (a golden swap in the same change as a mint attempt), and the production
  activation gate enforces the same floor over the committed corpus.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from menagerie.crawler.checker_calibration import (
    CHECKER_CALIBRATION_RUN_ENV,
    CHECKER_GOLDEN_CORPUS_ENV,
    CheckerGoldenCorpusError,
    _require_receipt_coverage,
)
from menagerie.crawler.checker_calibration_mint import _require_corpus_membership_floor
from menagerie.crawler.tools import golden_corpus_floor as floor
from menagerie.crawler.tools.golden_corpus_floor import (
    CORPUS_RELPATH,
    MANIFEST_RELPATH,
    GoldenCorpusFloorError,
    enforce,
    enforce_committed_corpus,
    load_manifest,
    observe_corpus,
    write_manifest,
)

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[3]


def _staged_repo(tmp_path: Path) -> Path:
    """Copy the committed manifest + corpus into an isolated repo layout."""

    staged = tmp_path / "repo"
    (staged / CORPUS_RELPATH).parent.mkdir(parents=True)
    shutil.copytree(REPO_ROOT / CORPUS_RELPATH, staged / CORPUS_RELPATH)
    shutil.copyfile(REPO_ROOT / MANIFEST_RELPATH, staged / MANIFEST_RELPATH)
    return staged


def _manifest_data(repo_root: Path) -> dict:
    return json.loads((repo_root / MANIFEST_RELPATH).read_text(encoding="utf-8"))


def _write_manifest_data(repo_root: Path, data: dict) -> None:
    (repo_root / MANIFEST_RELPATH).write_text(
        json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# The committed corpus is exact, and every drift class fails with the file named.
# ---------------------------------------------------------------------------


def test_committed_corpus_matches_committed_manifest() -> None:
    assert enforce_committed_corpus(REPO_ROOT) == []


def test_in_place_rewrite_fails_with_the_named_file(tmp_path: Path) -> None:
    staged = _staged_repo(tmp_path)
    target = staged / CORPUS_RELPATH / "metadata_accept.json"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    failures = enforce_committed_corpus(staged)
    assert len(failures) == 1
    assert "metadata_accept.json bytes were replaced in place" in failures[0]


def test_removal_fails_and_names_the_retirement_path(tmp_path: Path) -> None:
    staged = _staged_repo(tmp_path)
    (staged / CORPUS_RELPATH / "fidelity_refusal.json").unlink()
    failures = enforce_committed_corpus(staged)
    assert len(failures) == 1
    assert "fidelity_refusal.json is in the committed manifest but not on disk" in failures[0]
    assert "--retire" in failures[0]


def test_addition_fails_as_unreviewed_membership(tmp_path: Path) -> None:
    staged = _staged_repo(tmp_path)
    (staged / CORPUS_RELPATH / "metadata_extra.json").write_text("{}", encoding="utf-8")
    failures = enforce_committed_corpus(staged)
    assert len(failures) == 1
    assert "metadata_extra.json is on disk but not in the committed manifest" in failures[0]


def test_retired_name_reappearing_requires_explicit_reactivation(tmp_path: Path) -> None:
    staged = _staged_repo(tmp_path)
    (staged / CORPUS_RELPATH / "metadata_mixed_batch.json").write_text("{}", encoding="utf-8")
    failures = enforce_committed_corpus(staged)
    assert len(failures) == 1
    assert "RETIRED in the committed census but present on disk" in failures[0]
    assert "--reactivate" in failures[0]


def test_class_minimum_breach_fails(tmp_path: Path) -> None:
    staged = _staged_repo(tmp_path)
    data = _manifest_data(staged)
    data["class_minimums"]["fidelity"] = 99
    _write_manifest_data(staged, data)
    failures = enforce_committed_corpus(staged)
    assert len(failures) == 1
    assert "under its own committed minimum 99" in failures[0]


def test_rename_fails_in_both_directions(tmp_path: Path) -> None:
    staged = _staged_repo(tmp_path)
    corpus = staged / CORPUS_RELPATH
    (corpus / "terminal_accept.json").rename(corpus / "terminal_accepted.json")
    failures = enforce_committed_corpus(staged)
    assert len(failures) == 2
    assert any("terminal_accept.json is in the committed manifest" in f for f in failures)
    assert any("terminal_accepted.json is on disk but not" in f for f in failures)


# ---------------------------------------------------------------------------
# The retirement census is permanent and explicit.
# ---------------------------------------------------------------------------


def test_write_refuses_a_disappearance_without_retirement(tmp_path: Path) -> None:
    staged = _staged_repo(tmp_path)
    (staged / CORPUS_RELPATH / "terminal_disputed.json").unlink()
    with pytest.raises(GoldenCorpusFloorError, match="without a retirement"):
        write_manifest(staged)


def test_write_retirement_appends_a_permanent_census_row(tmp_path: Path) -> None:
    staged = _staged_repo(tmp_path)
    before = load_manifest(staged)
    recorded_digest = before.members["terminal_disputed.json"]["sha256"]
    (staged / CORPUS_RELPATH / "terminal_disputed.json").unlink()
    with pytest.raises(GoldenCorpusFloorError, match="--retired-at"):
        write_manifest(staged, retire={"terminal_disputed.json": "superseded exemplar"})
    manifest = write_manifest(
        staged,
        retire={"terminal_disputed.json": "superseded exemplar"},
        retired_at="2026-08-29",
    )
    assert "terminal_disputed.json" not in manifest.members
    row = next(r for r in manifest.retired if r["file"] == "terminal_disputed.json")
    assert row["sha256"] == recorded_digest
    assert row["reason"] == "superseded exemplar"
    assert row["retired_at"] == "2026-08-29"
    # The founding census rows survive the rewrite untouched.
    assert any(r["file"] == "metadata_mixed_batch.json" for r in manifest.retired)
    # A retirement that breaches the lane minimum stays RED until the minimum
    # is lowered by hand in the same reviewed change -- shrinking a lane is a
    # decision the floor surfaces, never a silent consequence of a removal.
    failures = enforce_committed_corpus(staged)
    assert failures and "terminal_disposition members, under its own committed minimum" in (
        failures[0]
    )
    data = _manifest_data(staged)
    data["class_minimums"]["terminal_disposition"] = 2
    _write_manifest_data(staged, data)
    assert enforce_committed_corpus(staged) == []


def test_write_refuses_retiring_a_file_still_on_disk(tmp_path: Path) -> None:
    staged = _staged_repo(tmp_path)
    with pytest.raises(GoldenCorpusFloorError, match="not disappearing manifest members"):
        write_manifest(
            staged,
            retire={"metadata_accept.json": "still here"},
            retired_at="2026-08-29",
        )


def test_write_refuses_a_retired_name_on_disk_without_reactivate(tmp_path: Path) -> None:
    staged = _staged_repo(tmp_path)
    (staged / CORPUS_RELPATH / "metadata_mixed_batch.json").write_text("{}", encoding="utf-8")
    with pytest.raises(GoldenCorpusFloorError, match="--reactivate"):
        write_manifest(staged)
    manifest = write_manifest(staged, reactivate=("metadata_mixed_batch.json",))
    assert "metadata_mixed_batch.json" in manifest.members
    assert not any(r["file"] == "metadata_mixed_batch.json" for r in manifest.retired)
    # The other founding row is untouched.
    assert any(r["file"] == "metadata_mixed_disputed.json" for r in manifest.retired)


def test_write_preserves_committed_lane_minimums(tmp_path: Path) -> None:
    staged = _staged_repo(tmp_path)
    data = _manifest_data(staged)
    data["class_minimums"]["terminal_disposition"] = 2
    _write_manifest_data(staged, data)
    manifest = write_manifest(staged)
    assert manifest.class_minimums["terminal_disposition"] == 2


# ---------------------------------------------------------------------------
# Manifest loader refusals (fail-closed structure).
# ---------------------------------------------------------------------------


def _founding_member() -> dict:
    return {
        "file": "metadata_accept.json",
        "gate_kind": "metadata_batch",
        "sha256": "0" * 64,
    }


def _minimal_manifest() -> dict:
    return {
        "schema_version": floor.MANIFEST_SCHEMA_VERSION,
        "members": [_founding_member()],
        "class_minimums": {"metadata_batch": 1, "fidelity": 1, "terminal_disposition": 1},
        "retired": [],
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d.update(schema_version="v0"), "schema"),
        (lambda d: d.update(members=[]), "non-empty list"),
        (
            lambda d: d["members"].append(dict(_founding_member())),
            "twice",
        ),
        (
            lambda d: d["members"][0].update(sha256="zz"),
            "lowercase hex sha256",
        ),
        (
            lambda d: d["members"][0].update(file="mystery.json"),
            "matches no known lane prefix",
        ),
        (
            lambda d: d["members"][0].update(gate_kind="fidelity"),
            "filename prefix derives",
        ),
        (
            lambda d: d.update(class_minimums={"metadata_batch": 1}),
            "class_minimums must name exactly",
        ),
        (
            lambda d: d["class_minimums"].update(fidelity=0),
            "int >= 1",
        ),
        (
            lambda d: d.update(
                retired=[
                    {
                        "file": "metadata_accept.json",
                        "gate_kind": "metadata_batch",
                        "sha256": None,
                        "retired_at": "2026-08-29",
                        "reason": "overlap",
                    }
                ]
            ),
            "BOTH a member and retired",
        ),
        (
            lambda d: d.update(retired=[{"file": "metadata_old.json"}]),
            "must carry exactly",
        ),
    ],
)
def test_manifest_loader_refuses_malformed_shapes(
    tmp_path: Path, mutate, match: str
) -> None:
    staged = tmp_path / "repo"
    (staged / MANIFEST_RELPATH).parent.mkdir(parents=True)
    data = _minimal_manifest()
    mutate(data)
    _write_manifest_data(staged, data)
    with pytest.raises(GoldenCorpusFloorError, match=match):
        load_manifest(staged)


def test_missing_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(GoldenCorpusFloorError, match="missing committed golden corpus manifest"):
        load_manifest(tmp_path)


def test_empty_corpus_is_never_a_measurement(tmp_path: Path) -> None:
    empty = tmp_path / "corpus"
    empty.mkdir()
    with pytest.raises(GoldenCorpusFloorError, match="ZERO transcripts"):
        observe_corpus(empty)


def test_enforce_is_pure_over_observed_mapping() -> None:
    manifest = load_manifest(REPO_ROOT)
    observed = observe_corpus(REPO_ROOT / CORPUS_RELPATH)
    assert enforce(observed, manifest) == []
    tampered = dict(observed)
    tampered["metadata_accept.json"] = "f" * 64
    assert len(enforce(tampered, manifest)) == 1


# ---------------------------------------------------------------------------
# Consumer 2: the mint refuses over a drifted corpus.
# ---------------------------------------------------------------------------


def test_mint_floor_passes_on_the_committed_corpus() -> None:
    _require_corpus_membership_floor()


def test_mint_floor_refuses_a_tampered_corpus_dir(tmp_path: Path) -> None:
    staged = _staged_repo(tmp_path)
    target = staged / CORPUS_RELPATH / "metadata_accept.json"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CheckerGoldenCorpusError, match="does not match its committed manifest"):
        _require_corpus_membership_floor(staged / CORPUS_RELPATH)


# ---------------------------------------------------------------------------
# Consumer 3: the production activation gate enforces the floor.
# ---------------------------------------------------------------------------


def _stub_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.crawler.checker_calibration import live_checker_prompt_identity

    # The stub carries the live prompt epoch (R2-3): these tests exercise the
    # corpus-floor arm, and the epoch door sits upstream of it in the gate.
    monkeypatch.setattr(
        "menagerie.crawler.checker_calibration.load_checker_calibration_receipt",
        lambda path=None: {
            "checker_model": "gpt-5.6-sol",
            "checker_prompt_identity": live_checker_prompt_identity(),
        },
    )
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)


def test_activation_gate_enforces_the_floor_on_the_committed_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_receipt(monkeypatch)
    monkeypatch.delenv(CHECKER_GOLDEN_CORPUS_ENV, raising=False)
    # Green on the real committed corpus + manifest.
    coverage = _require_receipt_coverage("metadata_batch", "gpt-5.6-sol")
    assert coverage["checker_model"] == "gpt-5.6-sol"
    # Red the moment the floor reports drift.
    monkeypatch.setattr(
        "menagerie.crawler.tools.golden_corpus_floor.enforce_committed_corpus",
        lambda repo_root=None, corpus_dir=None: ["golden X bytes were replaced in place"],
    )
    with pytest.raises(CheckerGoldenCorpusError, match="membership floor"):
        _require_receipt_coverage("metadata_batch", "gpt-5.6-sol")


def test_activation_gate_floor_failure_is_typed_on_unmeasurable_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_receipt(monkeypatch)
    monkeypatch.delenv(CHECKER_GOLDEN_CORPUS_ENV, raising=False)

    def _boom(repo_root=None, corpus_dir=None):
        raise GoldenCorpusFloorError("manifest unreadable")

    monkeypatch.setattr(
        "menagerie.crawler.tools.golden_corpus_floor.enforce_committed_corpus", _boom
    )
    with pytest.raises(CheckerGoldenCorpusError, match="cannot be measured"):
        _require_receipt_coverage("metadata_batch", "gpt-5.6-sol")


def test_activation_gate_override_env_is_the_documented_bootstrap_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the operator corpus override active the floor arm is out of scope."""

    _stub_receipt(monkeypatch)
    corpus = tmp_path / "goldens"
    corpus.mkdir()
    (corpus / "metadata_boot.json").write_text(
        json.dumps({"checker_model": "gpt-5.6-sol"}), encoding="utf-8"
    )
    monkeypatch.setenv(CHECKER_GOLDEN_CORPUS_ENV, str(corpus))

    def _never(repo_root=None, corpus_dir=None):  # pragma: no cover - tripwire
        raise AssertionError("floor must not run under the override seam")

    monkeypatch.setattr(
        "menagerie.crawler.tools.golden_corpus_floor.enforce_committed_corpus", _never
    )
    coverage = _require_receipt_coverage("metadata_batch", "gpt-5.6-sol")
    assert coverage["checker_model"] == "gpt-5.6-sol"
