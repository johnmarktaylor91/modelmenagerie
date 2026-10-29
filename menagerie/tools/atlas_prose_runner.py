"""Batch runner for the atlas-prose lane.

Authors reader-facing prose summaries for verified menagerie models, one store
entry per model, through a pluggable author executor:

- ``scripted`` reads prepared author results from a directory (tests, and any
  flow where a Fable session has already written its results to files);
- ``command`` pipes the prompt payload as JSON to an external command's stdin
  and parses the author result JSON from its stdout (production dispatch wraps
  a ``claude -p``-style session here; the runner never hardcodes a provider).

Discipline mirrors the crawler's executor lane at editorial weight: the write
root is the store directory, every result is validated by the traceability
gate before it lands, a gate failure earns one typed retry carrying the
failure back to the author, ineligible sources refuse with a typed code, and
``--resume`` skips entries already current for their record revision and
prompt digest. Refusals are honest terminals (exit 0); failures exit 1.

Example
-------
::

    PYTHONPATH=. python -m menagerie.tools.atlas_prose_runner \
        --stable-ids m9430,m1573 \
        --records /path/to/records/models/current-shard.jsonl \
        --intake menagerie/crawler/records/intake/<snapshot>/items.jsonl \
        --catalog menagerie/data/master_catalog.jsonl \
        --prompt menagerie/ATLAS_PROSE_PROMPT.md \
        --store menagerie/data/atlas_prose \
        --executor scripted --scripted-dir /path/to/results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from menagerie.atlas_prose import (
    AtlasProseGateError,
    append_entry,
    canonical_catalog_revision,
    catalog_row_is_trace_verified,
    crawler_record_is_verified,
    current_entry,
    load_store_file,
    store_path,
    validate_entry_against_record,
)

#: An executor maps (stable_id, prompt payload, attempt number) to an author
#: result object with summary_markdown / facts_used / evidence_ids.
AuthorExecutor = Callable[[str, Mapping[str, Any], int], Mapping[str, Any]]


class ExecutorError(RuntimeError):
    """The author executor failed to produce a parseable result."""


@dataclass
class SourceIndex:
    """Resolves stable ids to their prose source records.

    Crawler record shards win; ids absent from every shard fall through to the
    master catalog via the intake snapshot's stable-id-to-name binding.
    """

    records_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    intake_name_by_id: dict[str, str] = field(default_factory=dict)
    catalog_by_name: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        *,
        records: Sequence[Path | str] = (),
        intakes: Sequence[Path | str] = (),
        catalogs: Sequence[Path | str] = (),
    ) -> "SourceIndex":
        """Build an index from record shards, intake items, and catalogs.

        Parameters
        ----------
        records:
            Crawler model-record JSONL shards; the highest ``record_seq``
            revision per stable id wins.
        intakes:
            Intake ``items.jsonl`` files binding stable ids to roster names.
        catalogs:
            Master-catalog JSONL files keyed by row name.

        Returns
        -------
        SourceIndex
            Loaded resolver.
        """

        index = cls()
        for shard in records:
            for row in _read_jsonl(shard):
                stable_id = row.get("stable_id")
                if not isinstance(stable_id, str):
                    continue
                known = index.records_by_id.get(stable_id)
                if known is None or row.get("record_seq", 0) > known.get("record_seq", 0):
                    index.records_by_id[stable_id] = row
        for items in intakes:
            for row in _read_jsonl(items):
                stable_id, name = row.get("stable_id"), row.get("name")
                if isinstance(stable_id, str) and isinstance(name, str):
                    index.intake_name_by_id[stable_id] = name
        for catalog in catalogs:
            for row in _read_jsonl(catalog):
                name = row.get("name")
                if isinstance(name, str):
                    index.catalog_by_name.setdefault(name, row)
        return index

    def resolve(self, stable_id: str) -> tuple[str, dict[str, Any]]:
        """Resolve one stable id to its source kind and record.

        Parameters
        ----------
        stable_id:
            Model stable id.

        Returns
        -------
        tuple[str, dict]
            ``("crawler-record", record)`` or ``("catalog-row", row)``.

        Raises
        ------
        KeyError
            When no source knows the id.
        """

        record = self.records_by_id.get(stable_id)
        if record is not None:
            return "crawler-record", record
        name = self.intake_name_by_id.get(stable_id)
        if name is not None and name in self.catalog_by_name:
            return "catalog-row", self.catalog_by_name[name]
        raise KeyError(stable_id)


def _read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of objects.

    Parameters
    ----------
    path:
        JSONL file path.

    Returns
    -------
    list[dict]
        Parsed non-empty lines.
    """

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class ScriptedExecutor:
    """Author executor that reads prepared results from a directory.

    Looks for ``<stable_id>.attempt<N>.json`` first (attempt-specific replies,
    used to exercise the retry path) and falls back to ``<stable_id>.json``.

    Parameters
    ----------
    directory:
        Directory holding prepared author-result JSON files.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def __call__(self, stable_id: str, payload: Mapping[str, Any], attempt: int) -> dict[str, Any]:
        """Return the prepared result for one model attempt.

        Parameters
        ----------
        stable_id:
            Model stable id.
        payload:
            Prompt payload (unused; scripted replies are pre-authored).
        attempt:
            1-based attempt number.

        Returns
        -------
        dict
            Author result object.

        Raises
        ------
        ExecutorError
            When no prepared response exists or it is unparseable.
        """

        for name in (f"{stable_id}.attempt{attempt}.json", f"{stable_id}.json"):
            path = self.directory / name
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as err:
                    raise ExecutorError(f"unparseable scripted response {path}: {err}") from err
        raise ExecutorError(f"no scripted response for {stable_id} in {self.directory}")


class CommandExecutor:
    """Author executor that shells out to an external author command.

    The prompt payload is written to the command's stdin as JSON; the command
    must print the author result JSON object on stdout.

    Parameters
    ----------
    argv:
        Command argv to run per attempt.
    timeout:
        Per-attempt wall timeout in seconds.
    """

    def __init__(self, argv: Sequence[str], timeout: float = 1800.0) -> None:
        self.argv = list(argv)
        self.timeout = timeout

    def __call__(self, stable_id: str, payload: Mapping[str, Any], attempt: int) -> dict[str, Any]:
        """Run the author command for one model attempt.

        Parameters
        ----------
        stable_id:
            Model stable id.
        payload:
            Prompt payload piped to stdin as JSON.
        attempt:
            1-based attempt number.

        Returns
        -------
        dict
            Author result object parsed from stdout.

        Raises
        ------
        ExecutorError
            On non-zero exit, timeout, or unparseable stdout.
        """

        try:
            completed = subprocess.run(
                self.argv,
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as err:
            raise ExecutorError(f"author command failed for {stable_id}: {err}") from err
        if completed.returncode != 0:
            raise ExecutorError(
                f"author command exited {completed.returncode} for {stable_id}: "
                f"{completed.stderr.strip()[:500]}"
            )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as err:
            raise ExecutorError(f"unparseable author stdout for {stable_id}: {err}") from err


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 Z timestamp.

    Returns
    -------
    str
        ``YYYY-MM-DDTHH:MM:SSZ``.
    """

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _source_revision(source_kind: str, record: Mapping[str, Any]) -> str:
    """Compute the revision an entry must bind for its source.

    Parameters
    ----------
    source_kind:
        ``crawler-record`` or ``catalog-row``.
    record:
        The resolved source record.

    Returns
    -------
    str
        Revision digest.
    """

    if source_kind == "crawler-record":
        revision = record.get("record_revision")
        return revision if isinstance(revision, str) else ""
    return canonical_catalog_revision(record)


def _eligibility_refusal(source_kind: str, record: Mapping[str, Any]) -> str | None:
    """Return the typed refusal code for an ineligible source, if any.

    Parameters
    ----------
    source_kind:
        ``crawler-record`` or ``catalog-row``.
    record:
        The resolved source record.

    Returns
    -------
    str or None
        Refusal code, or None when the source is eligible for prose.
    """

    if source_kind == "crawler-record" and not crawler_record_is_verified(record):
        return "record-not-verified"
    if source_kind == "catalog-row" and not catalog_row_is_trace_verified(record):
        return "row-not-trace-verified"
    return None


@dataclass
class _Outcome:
    """One model's disposition inside a batch run."""

    stable_id: str
    disposition: str
    code: str | None = None
    detail: str | None = None
    retried: bool = False


def _author_one(
    stable_id: str,
    *,
    sources: SourceIndex,
    store_dir: Path,
    prompt_text: str,
    prompt_sha256: str,
    executor: AuthorExecutor,
    author_model: str,
    resume: bool,
    retries: int,
) -> _Outcome:
    """Author one model end to end: resolve, gate, retry, land.

    Parameters
    ----------
    stable_id:
        Model stable id.
    sources:
        Source resolver.
    store_dir:
        Atlas-prose store directory.
    prompt_text:
        Full prose prompt text handed to the author.
    prompt_sha256:
        Digest binding entries to the prompt version.
    executor:
        Author executor.
    author_model:
        Provenance value recorded on the entry.
    resume:
        Skip ids whose current entry already binds this revision and prompt.
    retries:
        Gate-failure retries granted after the first attempt.

    Returns
    -------
    _Outcome
        The model's disposition.
    """

    try:
        source_kind, record = sources.resolve(stable_id)
    except KeyError:
        return _Outcome(stable_id, "failed", "unknown-stable-id", "no source knows this id")

    refusal = _eligibility_refusal(source_kind, record)
    if refusal is not None:
        return _Outcome(stable_id, "refused", refusal)

    revision = _source_revision(source_kind, record)
    path = store_path(store_dir, stable_id)
    if resume and path.exists():
        entry = current_entry(load_store_file(path))
        if entry["record_revision"] == revision and entry["prompt_sha256"] == prompt_sha256:
            return _Outcome(stable_id, "skipped_current")

    gate_failure: dict[str, str] | None = None
    last_error: AtlasProseGateError | None = None
    for attempt in range(1, retries + 2):
        payload = {
            "stable_id": stable_id,
            "source_kind": source_kind,
            "record": record,
            "prompt": prompt_text,
            "gate_failure": gate_failure,
        }
        try:
            result = executor(stable_id, payload, attempt)
        except ExecutorError as err:
            return _Outcome(stable_id, "failed", "executor-error", str(err))
        row: dict[str, Any] = {
            "stable_id": stable_id,
            "summary_markdown": result.get("summary_markdown"),
            "facts_used": result.get("facts_used"),
            "evidence_ids": result.get("evidence_ids"),
            "author_model": author_model,
            "prompt_sha256": prompt_sha256,
            "record_revision": revision,
            "created_at": _utc_now(),
            "source_kind": source_kind,
        }
        if source_kind == "catalog-row":
            row["source_name"] = record.get("name")
        try:
            validate_entry_against_record(row, record)
        except AtlasProseGateError as err:
            last_error = err
            gate_failure = {"code": err.code, "detail": err.detail}
            continue
        append_entry(store_dir, row)
        return _Outcome(stable_id, "authored", retried=attempt > 1)
    assert last_error is not None
    return _Outcome(stable_id, "failed", last_error.code, last_error.detail)


def run_batch(
    stable_ids: Sequence[str],
    *,
    sources: SourceIndex,
    store_dir: Path | str,
    prompt_path: Path | str,
    executor: AuthorExecutor,
    author_model: str,
    resume: bool = False,
    retries: int = 1,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Author prose entries for a batch of models.

    Parameters
    ----------
    stable_ids:
        Models to author, in report order.
    sources:
        Source resolver.
    store_dir:
        Atlas-prose store directory.
    prompt_path:
        Prose prompt file; its digest binds every entry authored this run.
    executor:
        Author executor.
    author_model:
        Provenance value recorded on entries.
    resume:
        Skip ids whose current entry is already bound to this revision+prompt.
    retries:
        Gate-failure retries per id.
    concurrency:
        Bounded worker threads for executor calls.

    Returns
    -------
    dict
        Report with ``authored``, ``skipped_current``, ``retried``,
        ``refused`` (typed honest terminals), and ``failed`` lists.
    """

    prompt_text = Path(prompt_path).read_text(encoding="utf-8")
    prompt_sha256 = "sha256:" + hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    store = Path(store_dir)

    def work(stable_id: str) -> _Outcome:
        return _author_one(
            stable_id,
            sources=sources,
            store_dir=store,
            prompt_text=prompt_text,
            prompt_sha256=prompt_sha256,
            executor=executor,
            author_model=author_model,
            resume=resume,
            retries=retries,
        )

    if concurrency > 1 and len(stable_ids) > 1:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(stable_ids))) as pool:
            outcomes = list(pool.map(work, stable_ids))
    else:
        outcomes = [work(stable_id) for stable_id in stable_ids]

    report: dict[str, Any] = {
        "authored": [o.stable_id for o in outcomes if o.disposition == "authored"],
        "skipped_current": [o.stable_id for o in outcomes if o.disposition == "skipped_current"],
        "retried": [o.stable_id for o in outcomes if o.disposition == "authored" and o.retried],
        "refused": [
            {"stable_id": o.stable_id, "code": o.code}
            for o in outcomes
            if o.disposition == "refused"
        ],
        "failed": [
            {"stable_id": o.stable_id, "code": o.code, "detail": o.detail}
            for o in outcomes
            if o.disposition == "failed"
        ],
        "prompt_sha256": prompt_sha256,
        "author_model": author_model,
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the atlas-prose batch CLI.

    Parameters
    ----------
    argv:
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        0 on success (refusals included), 1 when any id failed.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stable-ids", help="comma-separated stable ids to author")
    parser.add_argument("--batch", type=Path, help="file with one stable id per line")
    parser.add_argument(
        "--records", action="append", default=[], type=Path, help="model-record JSONL shard"
    )
    parser.add_argument(
        "--intake", action="append", default=[], type=Path, help="intake items.jsonl"
    )
    parser.add_argument(
        "--catalog", action="append", default=[], type=Path, help="master-catalog JSONL"
    )
    parser.add_argument("--prompt", required=True, type=Path, help="prose prompt file")
    parser.add_argument("--store", required=True, type=Path, help="atlas-prose store directory")
    parser.add_argument("--executor", choices=("scripted", "command"), required=True)
    parser.add_argument("--scripted-dir", type=Path, help="scripted-executor response directory")
    parser.add_argument(
        "--command", nargs=argparse.REMAINDER, help="author command argv (command executor)"
    )
    parser.add_argument("--author-model", required=True, help="provenance author model name")
    parser.add_argument("--resume", action="store_true", help="skip already-current entries")
    parser.add_argument("--retries", type=int, default=1, help="gate-failure retries per id")
    parser.add_argument("--concurrency", type=int, default=1, help="bounded worker threads")
    parser.add_argument("--report", type=Path, help="also write the JSON report here")
    args = parser.parse_args(argv)

    stable_ids: list[str] = []
    if args.stable_ids:
        stable_ids.extend(part.strip() for part in args.stable_ids.split(",") if part.strip())
    if args.batch:
        stable_ids.extend(
            line.strip() for line in args.batch.read_text().splitlines() if line.strip()
        )
    if not stable_ids:
        parser.error("no stable ids given (use --stable-ids or --batch)")

    if args.executor == "scripted":
        if not args.scripted_dir:
            parser.error("--executor scripted requires --scripted-dir")
        executor: AuthorExecutor = ScriptedExecutor(args.scripted_dir)
    else:
        if not args.command:
            parser.error("--executor command requires --command ...")
        executor = CommandExecutor(args.command)

    sources = SourceIndex.load(records=args.records, intakes=args.intake, catalogs=args.catalog)
    report = run_batch(
        stable_ids,
        sources=sources,
        store_dir=args.store,
        prompt_path=args.prompt,
        executor=executor,
        author_model=args.author_model,
        resume=args.resume,
        retries=args.retries,
        concurrency=args.concurrency,
    )
    serialized = json.dumps(report, indent=1, ensure_ascii=False)
    print(serialized)
    if args.report:
        args.report.write_text(serialized + "\n", encoding="utf-8")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
