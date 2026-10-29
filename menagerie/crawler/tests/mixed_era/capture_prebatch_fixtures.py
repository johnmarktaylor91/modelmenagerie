"""Capture the frozen PRE-BATCH fixtures for the AB-30 mixed-era matrix.

Run once at the Wave-2 freeze tree, BEFORE any rotation-batch lane merges::

    PYTHONPATH=. python -m menagerie.crawler.tests.mixed_era.capture_prebatch_fixtures

Every fixture is minted through the CURRENT tree's real writers (the fixture
reducer, the canonical grant builder, the artifact transaction committer), so
the frozen bytes are genuine prior-era artifacts, not hand-written imitations.
The manifest records the capture tree and a sha256 per fixture; matrix reads
attest against it, so a silently edited fixture can never adjudicate an era
claim. Existing fixtures refuse to overwrite without ``--force`` -- the
capture is the era boundary, not a regeneratable cache.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from menagerie.crawler.tests.mixed_era import MANIFEST_SCHEMA_VERSION

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parents[3]


def _mint_terminal_ledgers(out: Path) -> None:
    """Freeze one models+attempts shard holding two lawful terminal heads."""

    from menagerie.crawler.reducer import LedgerPaths
    from menagerie.crawler.tests.conftest import (
        bind_terminal_attempts,
        make_failed_attempt,
        make_model,
    )
    from menagerie.crawler.tests.test_reducer import CanonicalReducer as FixtureReducer

    with tempfile.TemporaryDirectory() as scratch:
        base = Path(scratch)
        paths = LedgerPaths(
            models=base / "models.jsonl",
            attempts=base / "attempts.jsonl",
            gates=base / "gates.jsonl",
        )
        with FixtureReducer(paths, ["m_forward", "m_author"]) as reducer:
            forward_attempt = make_failed_attempt(
                "m_forward",
                attempt_id="attempt-forward-1",
                stage="forward",
                reason_code="exception",
            )
            reducer.append_attempt(forward_attempt)
            forward = make_model("m_forward", status_code="failed:forward")
            forward["status"]["reason_code"] = "exception"
            bind_terminal_attempts(forward, [forward_attempt])
            reducer.append_model(forward)

            author_attempt = make_failed_attempt(
                "m_author",
                attempt_id="attempt-author-1",
                stage="author",
                reason_code="effort-exhausted:author-sessions",
            )
            author_attempt["ledger_seq"] = 2
            reducer.append_attempt(author_attempt)
            author = make_model("m_author", status_code="failed:author")
            author["status"]["reason_code"] = "effort-exhausted:author-sessions"
            author.pop("record_seq", None)
            bind_terminal_attempts(author, [author_attempt])
            reducer.append_model(author)
        shutil.copyfile(paths.models, out / "terminal_models.jsonl")
        shutil.copyfile(paths.attempts, out / "terminal_attempts.jsonl")


def _mint_terminal_gates(out: Path) -> None:
    """Freeze the three adjudicated-gate era shapes for W13's discriminator."""

    from menagerie.crawler.tests.test_authority import _adjudicated_terminal_gate

    rendered = _adjudicated_terminal_gate("rejected")
    unrendered = copy.deepcopy(rendered)
    unrendered["items"][0]["terminal_disposition"]["verdict"] = None
    missing = copy.deepcopy(rendered)
    missing["items"][0].pop("terminal_disposition", None)
    for name, gate in (
        ("terminal_gate_rendered.json", rendered),
        ("terminal_gate_unrendered.json", unrendered),
        ("terminal_gate_missing_block.json", missing),
    ):
        (out / name).write_text(json.dumps(gate, sort_keys=True, indent=1) + "\n")


def _mint_requeue_grant(out: Path) -> None:
    """Freeze one canonical requeue-grant row exactly as built today."""

    from menagerie.crawler.checkpoint import build_canonical_requeue_grant

    with tempfile.TemporaryDirectory() as scratch:
        ledger = Path(scratch) / "requeue-grants.jsonl"
        grant = build_canonical_requeue_grant(
            ledger,
            stable_id="m_infra",
            stage="runner",
            reason="operator approved infra re-drive",
            attempts=1,
            granted_by="operator",
        )
    (out / "requeue_grant.json").write_text(json.dumps(grant, sort_keys=True, indent=1) + "\n")


def _mint_release_tree(out: Path, shape: str) -> None:
    """Freeze the three ledgers of one release-class adjudication shape."""

    from menagerie.crawler.tests.test_infra_terminal_grant_release import (
        STABLE_ID,
        _append_grant,
        _exhaustion_attempt,
        _paths,
        _terminal_record,
        _worker_crash_attempt,
        _write_attempts,
        _write_models,
    )
    from menagerie.crawler.checkpoint import canonical_requeue_grants_path
    from menagerie.crawler.ledger_chain import row_content_digest

    target = out / f"release_{shape}"
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        paths = _paths(Path(scratch))
        if shape == "worker_crash":
            witnesses = [
                _worker_crash_attempt(STABLE_ID, attempt_id="attempt-crash-1", attempt_no=1),
                _worker_crash_attempt(STABLE_ID, attempt_id="attempt-crash-2", attempt_no=2),
            ]
            record = _terminal_record(
                STABLE_ID, attempt_ids=("attempt-crash-1", "attempt-crash-2")
            )
            # A COMMITTED row must be an honest row: its declared
            # record_revision is its body derivation (the tree-wide
            # content-address census enforces this over every committed
            # .jsonl fixture).
            record["record_revision"] = row_content_digest(record)
            _write_attempts(paths, witnesses)
            _write_models(paths, [record])
            _append_grant(paths, STABLE_ID, stage="runner")
        elif shape == "exhaustion":
            witnesses = [
                _exhaustion_attempt(STABLE_ID, attempt_id="attempt-exh-1", attempt_no=1)
            ]
            record = _terminal_record(
                STABLE_ID,
                stage="author",
                code="failed:author",
                reason_code="effort-exhausted:author-sessions",
                attempt_ids=("attempt-exh-1",),
            )
            record["record_revision"] = row_content_digest(record)
            _write_attempts(paths, witnesses)
            _write_models(paths, [record])
            _append_grant(
                paths, STABLE_ID, stage="author", reason="operator approved exhaustion re-drive"
            )
        else:  # pragma: no cover - closed shape set
            raise ValueError(f"unknown release shape: {shape}")
        shutil.copyfile(paths.ledgers.models, target / "models.jsonl")
        shutil.copyfile(paths.ledgers.attempts, target / "attempts.jsonl")
        shutil.copyfile(
            canonical_requeue_grants_path(paths.ledgers.models),
            target / "requeue-grants.jsonl",
        )


def _mint_award_chain(out: Path) -> dict[str, Any]:
    """Freeze one finalized artifact chain plus its reconstruction document.

    The reconstruction document is stored under a portable filename (its
    canonical anchor path contains ``:``, which Windows checkouts refuse);
    the returned anchor metadata lets the matrix re-materialize the exact
    canonical layout at test time.
    """

    from menagerie.crawler.artifact_transactions import ArtifactEventLedger
    from menagerie.crawler.tests.test_artifact_transactions import (
        _commit_private,
        _context,
        _mirrors,
    )

    target = out / "award_chain"
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        repo = scratch_path / "repo"
        canonical = repo / "menagerie" / "crawler"
        shard = canonical / "records" / "artifacts" / "shard.jsonl"
        with ArtifactEventLedger(shard) as ledger:
            _commit_private(
                "m_prebatch",
                content=b"wave-2 pre-batch era artifact bytes",
                context=_context("m_prebatch"),
                mirrors=_mirrors(scratch_path / "mirrors"),
                ledger=ledger,
                canonical_root=canonical,
                repository_root=repo,
                evidence_id="ev-prebatch-freeze",
            )
        shutil.copyfile(shard, target / "shard.jsonl")
        docs = [path for path in repo.rglob("*.json") if "reconstruction" in path.parts]
        if len(docs) != 1:
            raise RuntimeError(f"expected exactly one reconstruction doc, found {docs}")
        doc = docs[0]
        shutil.copyfile(doc, target / "reconstruction_doc.json")
        return {
            "anchor_path": str(doc.relative_to(repo)),
            "anchor_sha256": "sha256:" + hashlib.sha256(doc.read_bytes()).hexdigest(),
        }


def main(argv: list[str] | None = None) -> int:
    """Capture every pre-batch fixture and write the attested manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing capture.")
    arguments = parser.parse_args(argv)
    out = _PACKAGE_ROOT / "fixtures"
    manifest_path = out / "MANIFEST.json"
    if manifest_path.exists() and not arguments.force:
        print("capture already exists; the freeze is not a regeneratable cache "
              "(--force to overwrite)", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    _mint_terminal_ledgers(out)
    _mint_terminal_gates(out)
    _mint_requeue_grant(out)
    _mint_release_tree(out, "worker_crash")
    _mint_release_tree(out, "exhaustion")
    award_anchor = _mint_award_chain(out)

    tree_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    entries = {
        str(path.relative_to(out)): "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "tree_sha": tree_sha,
        "captured_command": "python -m menagerie.crawler.tests.mixed_era.capture_prebatch_fixtures",
        "award_chain_anchor": award_anchor,
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=1) + "\n")
    print(json.dumps({"captured": len(entries), "tree_sha": tree_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
