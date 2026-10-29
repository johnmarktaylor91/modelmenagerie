"""Tests for the atlas-prose batch runner: sources, refusals, retry, resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from menagerie.atlas_prose import (
    canonical_catalog_revision,
    current_entry,
    load_store_file,
    store_path,
)
from menagerie.tools.atlas_prose_runner import (
    ScriptedExecutor,
    SourceIndex,
    main,
    run_batch,
)

from tests.test_menagerie_atlas_prose import catalog_row, verified_record

PROMPT_TEXT = "Write warm, precise prose from verified facts only.\n"


def make_sources(tmp_path: Path) -> dict[str, Path]:
    """Write records, intake, and catalog fixture files.

    Parameters
    ----------
    tmp_path:
        Test scratch directory.

    Returns
    -------
    dict[str, Path]
        Paths keyed by fixture kind.
    """

    records = tmp_path / "models.jsonl"
    good = verified_record()
    bad = verified_record()
    bad["stable_id"] = "m8590"
    bad["accuracy_gate"] = {"verdict": "cannot-verify"}
    bad["authored_metadata_state"] = "failed"
    stale = verified_record()
    stale["record_seq"] = 1
    stale["authored_metadata_state"] = "failed"
    good["record_seq"] = 2
    with records.open("w") as handle:
        for record in (stale, good, bad):
            handle.write(json.dumps(record) + "\n")

    intake = tmp_path / "items.jsonl"
    intake.write_text(json.dumps({"stable_id": "m1573", "name": "LeNet-5"}) + "\n")

    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(json.dumps(catalog_row()) + "\n")

    prompt = tmp_path / "prompt.md"
    prompt.write_text(PROMPT_TEXT)
    return {"records": records, "intake": intake, "catalog": catalog, "prompt": prompt}


def scripted_result(stable_id: str, facts: list[str], evidence: list[str]) -> dict:
    """Build a scripted author result payload.

    Parameters
    ----------
    stable_id:
        Model the result describes.
    facts:
        facts_used paths.
    evidence:
        evidence_ids.

    Returns
    -------
    dict
        Author-result payload for the scripted executor.
    """

    paragraph = (
        f"{stable_id} is a fixture model whose summary is long enough to clear the "
        "character floor while staying inside the paragraph ceiling for the gate."
    )
    return {
        "summary_markdown": paragraph + "\n\n" + paragraph,
        "facts_used": facts,
        "evidence_ids": evidence,
    }


def write_script(
    directory: Path, stable_id: str, payload: dict, attempt: int | None = None
) -> None:
    """Write a scripted executor response file.

    Parameters
    ----------
    directory:
        Scripted-executor directory.
    stable_id:
        Target model id.
    payload:
        Author result payload.
    attempt:
        Attempt number for attempt-specific responses.
    """

    directory.mkdir(parents=True, exist_ok=True)
    name = f"{stable_id}.json" if attempt is None else f"{stable_id}.attempt{attempt}.json"
    (directory / name).write_text(json.dumps(payload))


class TestSourceIndex:
    """Stable-id resolution across record shards and catalog rows."""

    def test_prefers_latest_record_revision(self, tmp_path: Path) -> None:
        """The index resolves the highest record_seq revision."""

        paths = make_sources(tmp_path)
        index = SourceIndex.load(
            records=[paths["records"]], intakes=[paths["intake"]], catalogs=[paths["catalog"]]
        )
        kind, record = index.resolve("m9430")
        assert kind == "crawler-record"
        assert record["authored_metadata_state"] == "accepted"

    def test_resolves_catalog_row_via_intake(self, tmp_path: Path) -> None:
        """Ids absent from record shards fall through to the catalog."""

        paths = make_sources(tmp_path)
        index = SourceIndex.load(
            records=[paths["records"]], intakes=[paths["intake"]], catalogs=[paths["catalog"]]
        )
        kind, record = index.resolve("m1573")
        assert kind == "catalog-row"
        assert record["name"] == "LeNet-5"

    def test_unknown_id_is_typed(self, tmp_path: Path) -> None:
        """An unresolvable id raises a KeyError naming the id."""

        paths = make_sources(tmp_path)
        index = SourceIndex.load(
            records=[paths["records"]], intakes=[paths["intake"]], catalogs=[paths["catalog"]]
        )
        with pytest.raises(KeyError, match="m0"):
            index.resolve("m0")


class TestRunBatch:
    """End-to-end authoring through the scripted executor."""

    def run(self, tmp_path: Path, ids: list[str], *, resume: bool = False) -> dict:
        """Run a scripted batch over the standard fixtures.

        Parameters
        ----------
        tmp_path:
            Test scratch directory holding fixtures, scripts, and store.
        ids:
            Stable ids to author.
        resume:
            Whether to skip already-current entries.

        Returns
        -------
        dict
            Run report.
        """

        paths = make_sources(tmp_path)
        index = SourceIndex.load(
            records=[paths["records"]], intakes=[paths["intake"]], catalogs=[paths["catalog"]]
        )
        return run_batch(
            ids,
            sources=index,
            store_dir=tmp_path / "store",
            prompt_path=paths["prompt"],
            executor=ScriptedExecutor(tmp_path / "scripts"),
            author_model="claude-fable-5",
            resume=resume,
        )

    def test_authorship_lands_entries(self, tmp_path: Path) -> None:
        """Both source kinds author store entries with full provenance."""

        write_script(
            tmp_path / "scripts",
            "m9430",
            scripted_result("m9430", ["citation.title", "taxonomy.family"], ["ex-fpn-class"]),
        )
        write_script(
            tmp_path / "scripts",
            "m1573",
            scripted_result("m1573", ["name", "era", "recipe.body"], []),
        )
        report = self.run(tmp_path, ["m9430", "m1573"])
        assert report["authored"] == ["m9430", "m1573"]
        assert report["failed"] == [] and report["refused"] == []

        entry = current_entry(load_store_file(store_path(tmp_path / "store", "m9430")))
        assert entry["author_model"] == "claude-fable-5"
        assert entry["source_kind"] == "crawler-record"
        assert entry["record_revision"] == verified_record()["record_revision"]
        expected_digest = "sha256:" + hashlib.sha256(PROMPT_TEXT.encode()).hexdigest()
        assert entry["prompt_sha256"] == expected_digest

        classic = current_entry(load_store_file(store_path(tmp_path / "store", "m1573")))
        assert classic["source_kind"] == "catalog-row"
        assert classic["source_name"] == "LeNet-5"
        assert classic["record_revision"] == canonical_catalog_revision(catalog_row())

    def test_unverified_record_is_refused_not_failed(self, tmp_path: Path) -> None:
        """m8590-shaped records refuse with a typed code and no store file."""

        write_script(tmp_path / "scripts", "m8590", scripted_result("m8590", ["stable_id"], []))
        report = self.run(tmp_path, ["m8590"])
        assert report["authored"] == [] and report["failed"] == []
        assert report["refused"] == [{"stable_id": "m8590", "code": "record-not-verified"}]
        assert not store_path(tmp_path / "store", "m8590").exists()

    def test_gate_failure_retries_once_then_lands(self, tmp_path: Path) -> None:
        """An invalid first result triggers one typed retry."""

        write_script(
            tmp_path / "scripts",
            "m9430",
            scripted_result("m9430", ["citation.invented_field"], []),
            attempt=1,
        )
        write_script(
            tmp_path / "scripts",
            "m9430",
            scripted_result("m9430", ["citation.title"], ["ex-fpn-meta"]),
            attempt=2,
        )
        report = self.run(tmp_path, ["m9430"])
        assert report["authored"] == ["m9430"]
        assert report["retried"] == ["m9430"]

    def test_gate_failure_twice_fails(self, tmp_path: Path) -> None:
        """Two invalid results exhaust the retry budget and fail the id."""

        write_script(
            tmp_path / "scripts",
            "m9430",
            scripted_result("m9430", ["citation.invented_field"], []),
        )
        report = self.run(tmp_path, ["m9430"])
        assert report["authored"] == []
        assert [f["stable_id"] for f in report["failed"]] == ["m9430"]
        assert report["failed"][0]["code"] == "facts-path-unresolved"

    def test_resume_skips_current_entries(self, tmp_path: Path) -> None:
        """A second resumed run skips entries bound to the same revision+prompt."""

        write_script(
            tmp_path / "scripts",
            "m9430",
            scripted_result("m9430", ["citation.title"], ["ex-fpn-meta"]),
        )
        first = self.run(tmp_path, ["m9430"])
        assert first["authored"] == ["m9430"]
        second = self.run(tmp_path, ["m9430"], resume=True)
        assert second["authored"] == []
        assert second["skipped_current"] == ["m9430"]
        store = load_store_file(store_path(tmp_path / "store", "m9430"))
        assert len(store["entries"]) == 1

    def test_resume_reauthors_on_new_revision(self, tmp_path: Path) -> None:
        """A changed record revision supersedes the stored entry on resume."""

        write_script(
            tmp_path / "scripts",
            "m9430",
            scripted_result("m9430", ["citation.title"], ["ex-fpn-meta"]),
        )
        paths = make_sources(tmp_path)
        index = SourceIndex.load(
            records=[paths["records"]], intakes=[paths["intake"]], catalogs=[paths["catalog"]]
        )
        common = dict(
            store_dir=tmp_path / "store",
            prompt_path=paths["prompt"],
            executor=ScriptedExecutor(tmp_path / "scripts"),
            author_model="claude-fable-5",
        )
        assert run_batch(["m9430"], sources=index, **common)["authored"] == ["m9430"]

        revised = verified_record()
        revised["record_seq"] = 3
        revised["record_revision"] = "sha256:" + "e" * 64
        with paths["records"].open("a") as handle:
            handle.write(json.dumps(revised) + "\n")
        index = SourceIndex.load(
            records=[paths["records"]], intakes=[paths["intake"]], catalogs=[paths["catalog"]]
        )
        report = run_batch(["m9430"], sources=index, resume=True, **common)
        assert report["authored"] == ["m9430"]
        store = load_store_file(store_path(tmp_path / "store", "m9430"))
        assert len(store["entries"]) == 2
        assert current_entry(store)["record_revision"] == "sha256:" + "e" * 64

    def test_missing_script_is_failure(self, tmp_path: Path) -> None:
        """A missing scripted response is an executor failure, not a crash."""

        report = self.run(tmp_path, ["m9430"])
        assert [f["stable_id"] for f in report["failed"]] == ["m9430"]
        assert report["failed"][0]["code"] == "executor-error"

    def test_concurrent_batch_matches_serial(self, tmp_path: Path) -> None:
        """Concurrency changes wall-clock, never the report or the store."""

        write_script(
            tmp_path / "scripts",
            "m9430",
            scripted_result("m9430", ["citation.title"], ["ex-fpn-meta"]),
        )
        write_script(
            tmp_path / "scripts",
            "m1573",
            scripted_result("m1573", ["name", "era"], []),
        )
        paths = make_sources(tmp_path)
        index = SourceIndex.load(
            records=[paths["records"]], intakes=[paths["intake"]], catalogs=[paths["catalog"]]
        )
        report = run_batch(
            ["m9430", "m1573"],
            sources=index,
            store_dir=tmp_path / "store",
            prompt_path=paths["prompt"],
            executor=ScriptedExecutor(tmp_path / "scripts"),
            author_model="claude-fable-5",
            concurrency=4,
        )
        assert report["authored"] == ["m9430", "m1573"]


class TestCli:
    """The module CLI wires files to run_batch."""

    def test_cli_end_to_end(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """The CLI authors entries and prints the JSON report."""

        paths = make_sources(tmp_path)
        write_script(
            tmp_path / "scripts",
            "m9430",
            scripted_result("m9430", ["citation.title"], ["ex-fpn-meta"]),
        )
        rc = main(
            [
                "--stable-ids",
                "m9430",
                "--records",
                str(paths["records"]),
                "--intake",
                str(paths["intake"]),
                "--catalog",
                str(paths["catalog"]),
                "--prompt",
                str(paths["prompt"]),
                "--store",
                str(tmp_path / "store"),
                "--executor",
                "scripted",
                "--scripted-dir",
                str(tmp_path / "scripts"),
                "--author-model",
                "claude-fable-5",
            ]
        )
        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert report["authored"] == ["m9430"]
        assert store_path(tmp_path / "store", "m9430").exists()

    def test_cli_exit_code_on_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Failed ids drive a non-zero exit; refusals alone do not."""

        paths = make_sources(tmp_path)
        rc = main(
            [
                "--stable-ids",
                "m9430,m8590",
                "--records",
                str(paths["records"]),
                "--intake",
                str(paths["intake"]),
                "--catalog",
                str(paths["catalog"]),
                "--prompt",
                str(paths["prompt"]),
                "--store",
                str(tmp_path / "store"),
                "--executor",
                "scripted",
                "--scripted-dir",
                str(tmp_path / "scripts"),
                "--author-model",
                "claude-fable-5",
            ]
        )
        assert rc == 1
        report = json.loads(capsys.readouterr().out)
        assert [f["stable_id"] for f in report["failed"]] == ["m9430"]
        assert report["refused"] == [{"stable_id": "m8590", "code": "record-not-verified"}]
