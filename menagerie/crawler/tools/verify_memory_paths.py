"""D-65 (PM-14): verify durable-knowledge claims in the agent docs against the tree.

The 200a failure class this forecloses: durable operator knowledge — the
committed agent guides' repo paths, pytest markers, environment-lock family
names, and ``MENAGERIE_*`` environment variables — rots silently as the tree
moves (the round19-prefix memory incident). Every such claim is re-verified
here against the CURRENT tree, fail-closed:

- ``repo-path``: a repo-relative path named in a doc must exist (closed
  runtime/forward-reference allowlist below, each entry justified).
- ``lock-family``: a path under ``menagerie/crawler/envs/locks/`` must belong
  to the committed lock family (``round19-`` — the exact family prefix the
  release conftest independently enforces on ``MENAGERIE_PLATFORM_LOCK``).
- ``pytest-marker``: every marker used in a documented ``-m`` expression must
  be declared in ``pyproject.toml``.
- ``env-var``: every ``MENAGERIE_*`` variable a doc names must actually be
  read somewhere in the source/CI tree.

Tokens containing shell interpolations (``$``) are never claims.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

#: The committed agent documents whose claims are standing knowledge.
MEMORY_DOCUMENTS = ("CLAUDE.md", "AGENTS.md")

#: The committed environment-lock family. The release conftest enforces the
#: same prefix on MENAGERIE_PLATFORM_LOCK; a doc naming a lock outside this
#: family is stale knowledge even if the file were somehow present.
LOCK_FAMILY_PREFIX = "round19-"
_LOCKS_DIR = "menagerie/crawler/envs/locks/"

#: Closed allowlist of doc-named paths that legitimately do not exist in the
#: tree. Every entry carries its justification; widening this set is a
#: reviewed change, never a convenience.
RUNTIME_PATH_ALLOWLIST: dict[str, str] = {
    # CLAUDE.md/AGENTS.md name the user-facing glossary as a forward
    # reference ("when it ships"); the agent copy lives outside the repo.
    "docs/reference/glossary.md": "declared forward reference: ships later",
}

_TOP_DIRS = (
    "menagerie",
    "torchlens",
    "tests",
    "tools",
    "docs",
    "examples",
    "notebooks",
    "scripts",
    "benchmarks",
    ".github",
)

_PATH_RE = re.compile(
    r"(?<![\w$./@-])((?:" + "|".join(re.escape(d) for d in _TOP_DIRS) + r")/[A-Za-z0-9_./-]*[A-Za-z0-9_])"
)
_ENV_RE = re.compile(r"\b(MENAGERIE_[A-Z0-9_]+)\b")
_MARKER_EXPR_RE = re.compile(r"-m\s+(\"[^\"]+\"|'[^']+'|[A-Za-z_][A-Za-z0-9_]*)")
_MARKER_STOPWORDS = frozenset({"not", "and", "or"})

#: Source roots scanned for env-var reads (CI workflows included: some
#: variables are consumed only by the release runners).
_ENV_SCAN_ROOTS = ("menagerie", "tools", "tests", ".github")
_ENV_SCAN_SUFFIXES = {".py", ".yml", ".yaml", ".sh", ".toml"}


@dataclass(frozen=True)
class Claim:
    kind: str
    value: str
    line: int


def extract_claims(text: str) -> list[Claim]:
    """Extract every verifiable knowledge claim from one document body."""

    claims: list[Claim] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, value: str, line_no: int) -> None:
        key = (kind, value)
        if key in seen:
            return
        seen.add(key)
        claims.append(Claim(kind=kind, value=value, line=line_no))

    for line_no, line in enumerate(text.splitlines(), start=1):
        for match in _PATH_RE.finditer(line):
            value = match.group(1)
            tail = line[match.end() : match.end() + 1]
            if "$" in value or tail == "$":
                continue
            add("repo-path", value, line_no)
            if value.startswith(_LOCKS_DIR):
                add("lock-family", value, line_no)
        for match in _ENV_RE.finditer(line):
            add("env-var", match.group(1), line_no)
        for match in _MARKER_EXPR_RE.finditer(line):
            # `python -m <module>` is a module invocation, not a pytest
            # marker expression: skip when the word before `-m` is a python
            # interpreter (covers `python`, `python3`, `$PREFIX/bin/python`).
            preceding = line[: match.start()].rstrip().rsplit(None, 1)
            if preceding and re.search(r"python[0-9.]*[\"']?$", preceding[-1]):
                continue
            expression = match.group(1).strip("\"'")
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression):
                if token not in _MARKER_STOPWORDS:
                    add("pytest-marker", token, line_no)
    return claims


def _declared_markers(repo_root: Path) -> frozenset[str]:
    pyproject = repo_root / "pyproject.toml"
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return frozenset()
    rows = payload.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("markers", [])
    return frozenset(str(row).split(":", 1)[0].strip() for row in rows)


def _source_env_tokens(repo_root: Path) -> frozenset[str]:
    tokens: set[str] = set()
    for root_name in _ENV_SCAN_ROOTS:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in _ENV_SCAN_SUFFIXES:
                continue
            try:
                body = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            tokens.update(_ENV_RE.findall(body))
    return frozenset(tokens)


def verify_documents(repo_root: Path, documents: Sequence[Path]) -> dict:
    """Verify every claim in the given documents against the tree."""

    markers = _declared_markers(repo_root)
    env_tokens: Optional[frozenset[str]] = None
    rows: list[dict] = []

    for document in documents:
        body = document.read_text(encoding="utf-8")
        for claim in extract_claims(body):
            status = "present"
            detail = ""
            if claim.kind == "repo-path":
                if claim.value in RUNTIME_PATH_ALLOWLIST:
                    detail = f"allowlisted: {RUNTIME_PATH_ALLOWLIST[claim.value]}"
                elif not (repo_root / claim.value).exists():
                    status = "stale"
                    detail = "path does not exist in the tree"
            elif claim.kind == "lock-family":
                basename = claim.value.rsplit("/", 1)[-1]
                if not basename.startswith(LOCK_FAMILY_PREFIX):
                    status = "stale"
                    detail = (
                        f"lock {basename!r} is outside the committed "
                        f"{LOCK_FAMILY_PREFIX}* family the release conftest enforces"
                    )
            elif claim.kind == "pytest-marker":
                if claim.value not in markers:
                    status = "stale"
                    detail = "marker is not declared in pyproject.toml"
            elif claim.kind == "env-var":
                if env_tokens is None:
                    env_tokens = _source_env_tokens(repo_root)
                if claim.value not in env_tokens:
                    status = "stale"
                    detail = "variable is never read in the source/CI tree"
            rows.append(
                {
                    "document": str(document),
                    "line": claim.line,
                    "kind": claim.kind,
                    "claim": claim.value,
                    "status": status,
                    "detail": detail,
                }
            )

    status = "clean" if all(row["status"] == "present" for row in rows) else "stale"
    return {
        "status": status,
        "rows": rows,
        "documents": [str(document) for document in documents],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--doc",
        action="append",
        type=Path,
        default=None,
        help="document to verify (repeatable; default: the committed agent docs)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    documents = args.doc or [repo_root / name for name in MEMORY_DOCUMENTS]
    try:
        report = verify_documents(repo_root, documents)
    except OSError as exc:
        print(f"verify-memory-paths failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "clean" else 1


if __name__ == "__main__":
    raise SystemExit(main())
