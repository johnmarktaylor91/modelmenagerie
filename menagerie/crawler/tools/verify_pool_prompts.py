"""Verify the whole crawler prompt surface, pool fragments included, against PLAN pins.

``verify_prompts`` pins the two top-level agent prompts. The dispatch-brief fragments under
``prompts/pool/`` shape every author session just as authoritatively, so they carry literal
PLAN digests under the same section and are checked by the same oracle. This module is the
superset command: it verifies the two top-level prompts *and* every shipped pool fragment,
and additionally proves the pinned inventory and the shipped inventory are the same set, so
neither adding an unpinned fragment nor deleting a pinned one can hide.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.tools.verify_prompts import verify_prompts

#: Digest rows are read from exactly one PLAN section, so an unrelated bullet elsewhere in
#: the document can never be mistaken for a pin and a pin can never be parked outside it.
PIN_SECTION_HEADING = "## 18. Frozen agent prompt identities"

_PIN_ROW = re.compile(r"^- `([^`\n]+)`: `[^`\n]*`", re.MULTILINE)
_NEXT_HEADING = re.compile(r"^## ", re.MULTILINE)


def build_parser() -> argparse.ArgumentParser:
    """Build the full prompt-surface verification parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with overridable paths for tests and audits.
    """

    crawler_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=crawler_root / "PLAN.md")
    parser.add_argument(
        "--author-prompt",
        type=Path,
        default=crawler_root / "prompts" / "claude_crawler_author_v2.txt",
    )
    parser.add_argument(
        "--checker-prompt",
        type=Path,
        default=crawler_root / "prompts" / "codex_accuracy_checker_v2.txt",
    )
    parser.add_argument(
        "--pool-root",
        type=Path,
        default=crawler_root / "prompts" / "pool",
    )
    parser.add_argument(
        "--update-plan",
        action="store_true",
        help=(
            "Re-pin the literal PLAN digest of every already-pinned shipped prompt to its "
            "current bytes, then verify. This is the sanctioned rotation path for a "
            "deliberate prompt edit; inventory drift (an unpinned shipped prompt or a "
            "pinned prompt that is not shipped) still refuses, so adding or removing a "
            "prompt remains an explicit reviewed PLAN edit."
        ),
    )
    return parser


def pinned_prompt_names(plan_path: Path) -> tuple[str, ...]:
    """Return every prompt name pinned in the frozen-identity section, in file order.

    Parameters
    ----------
    plan_path:
        Committed PLAN.md path.

    Returns
    -------
    tuple[str, ...]
        Pinned prompt file names.

    Raises
    ------
    ValueError
        If the pinned section is absent or pins the same name twice. A missing section is
        a removed drift oracle, which must fail rather than silently verify nothing.
    """

    plan = plan_path.read_text(encoding="utf-8")
    start, end = _pinned_section_span(plan)
    names = tuple(match.group(1) for match in _PIN_ROW.finditer(plan[start:end]))
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate pinned prompt digests: {', '.join(duplicates)}")
    return names


def _pinned_section_span(plan: str) -> tuple[int, int]:
    """Return the start and end offsets of the frozen-identity PLAN section.

    Parameters
    ----------
    plan:
        Complete PLAN.md text.

    Returns
    -------
    tuple[int, int]
        Half-open ``[start, end)`` span of the pinned section.

    Raises
    ------
    ValueError
        If the pinned section is absent. A missing section is a removed drift
        oracle, which must fail rather than silently verify nothing.
    """

    start = plan.find(PIN_SECTION_HEADING)
    if start < 0:
        raise ValueError(f"missing pinned prompt section: {PIN_SECTION_HEADING!r}")
    after = _NEXT_HEADING.search(plan, start + len(PIN_SECTION_HEADING))
    return start, after.start() if after else len(plan)


def pool_prompt_paths(pool_root: Path) -> tuple[Path, ...]:
    """Return every shipped pool fragment, sorted by name.

    Parameters
    ----------
    pool_root:
        ``prompts/pool`` directory rendered into every author dispatch brief.

    Returns
    -------
    tuple[Path, ...]
        Fragment paths.

    Raises
    ------
    ValueError
        If the directory is missing or ships no fragment. An empty pool cannot dispatch,
        so reporting it as verified would be a false pass.
    """

    if not pool_root.is_dir():
        raise ValueError(f"missing prompt pool directory: {pool_root}")
    paths = tuple(
        sorted((path for path in pool_root.glob("*.md") if path.is_file()), key=lambda p: p.name)
    )
    if not paths:
        raise ValueError(f"prompt pool ships no fragment: {pool_root}")
    return paths


def _executor_fragments(executor_root: Path) -> tuple[Path, ...]:
    """Return the executor's own stage prompts, which drive the production author path.

    These were unpinned until 2026-07-29. The pool fragments serve the operator-mediated
    lane the headless executor replaced, so the *only* prompts still shaping every real
    author session were the two with no drift detection at all. Three separate
    prompt/contract drift defects had already cost a rung apiece by then.

    An empty or missing directory is an error for the same reason it is for the pool: an
    executor that ships no stage prompt cannot author, and reporting that as verified
    would be a false pass.
    """

    if not executor_root.is_dir():
        raise ValueError(f"missing executor prompt directory: {executor_root}")
    paths = tuple(
        sorted(
            (path for path in executor_root.glob("*.md") if path.is_file()), key=lambda p: p.name
        )
    )
    if not paths:
        raise ValueError(f"executor ships no stage prompt: {executor_root}")
    return paths


def verify_prompt_surface(
    plan_path: Path,
    author_path: Path,
    checker_path: Path,
    pool_root: Path,
) -> dict[str, str]:
    """Verify every top-level prompt and pool fragment against its literal PLAN digest.

    Parameters
    ----------
    plan_path, author_path, checker_path:
        Frozen plan and the two top-level prompt paths.
    pool_root:
        Directory of dispatch-brief fragments.

    Returns
    -------
    dict[str, str]
        Prompt filename to exact byte digest, covering the whole surface.

    Raises
    ------
    ValueError
        If a shipped prompt is unpinned, a pinned prompt is not shipped, or any prompt
        differs from its independently pinned PLAN digest.
    """

    shipped = _shipped_prompt_inventory(plan_path, author_path, checker_path, pool_root)
    fragments = tuple(
        path for name, path in shipped.items() if name not in {author_path.name, checker_path.name}
    )

    # verify_prompts is the single drift oracle: it hashes a file and compares it against
    # that file's own literal PLAN row. Pairing each fragment with the already-pinned
    # author prompt reuses that oracle verbatim instead of re-deriving a second comparison.
    digests = dict(verify_prompts(plan_path, author_path, checker_path))
    for fragment in fragments:
        digests.update(verify_prompts(plan_path, author_path, fragment))
    return digests


def _shipped_prompt_inventory(
    plan_path: Path,
    author_path: Path,
    checker_path: Path,
    pool_root: Path,
) -> dict[str, Path]:
    """Return the shipped prompt surface after proving it equals the pinned inventory.

    Parameters
    ----------
    plan_path, author_path, checker_path, pool_root:
        Frozen plan, the two top-level prompt paths, and the fragment pool.

    Returns
    -------
    dict[str, Path]
        Prompt filename to shipped path for the whole surface.

    Raises
    ------
    ValueError
        If names collide, a shipped prompt is unpinned, or a pinned prompt is
        not shipped.
    """

    fragments = pool_prompt_paths(pool_root)
    # The executor's stage prompts drive every production author session; the pool
    # fragments serve the lane it replaced. Verifying only the latter left the two most
    # load-bearing prompts undetectably driftable.
    executor_fragments = _executor_fragments(pool_root.parent / "executor")
    shipped = {
        path.name: path for path in (author_path, checker_path, *fragments, *executor_fragments)
    }
    if len(shipped) != 2 + len(fragments) + len(executor_fragments):
        raise ValueError("prompt file names collide across the pinned surface")
    pinned = set(pinned_prompt_names(plan_path))
    unpinned = sorted(set(shipped) - pinned)
    if unpinned:
        raise ValueError(f"shipped prompt is not pinned in PLAN: {', '.join(unpinned)}")
    unshipped = sorted(pinned - set(shipped))
    if unshipped:
        raise ValueError(f"pinned prompt is not shipped: {', '.join(unshipped)}")
    return shipped


def update_pinned_digests(
    plan_path: Path,
    author_path: Path,
    checker_path: Path,
    pool_root: Path,
) -> dict[str, str]:
    """Re-pin every already-pinned shipped prompt to its current byte digest.

    This is the sanctioned rotation path for a deliberate prompt edit: it rewrites
    ONLY the literal digest of existing pin rows inside the frozen-identity section,
    leaving every other PLAN byte untouched. Inventory drift refuses exactly like
    verification, so this can never pin a smuggled new prompt or paper over a
    deleted one; those remain explicit reviewed PLAN edits.

    Parameters
    ----------
    plan_path, author_path, checker_path, pool_root:
        Frozen plan, the two top-level prompt paths, and the fragment pool.

    Returns
    -------
    dict[str, str]
        Prompt filename to the freshly pinned byte digest.

    Raises
    ------
    ValueError
        If the pinned section is absent, the inventories differ, or a pin row
        is malformed.
    """

    shipped = _shipped_prompt_inventory(plan_path, author_path, checker_path, pool_root)
    plan = plan_path.read_text(encoding="utf-8")
    start, end = _pinned_section_span(plan)
    section = plan[start:end]
    digests: dict[str, str] = {}
    for name, path in shipped.items():
        digest = hash_bytes(path.read_bytes())
        row_prefix = f"- `{name}`: `"
        row_at = section.find(row_prefix)
        if row_at < 0:
            raise ValueError(f"missing or malformed pinned prompt digest: {name}")
        digest_at = row_at + len(row_prefix)
        digest_end = section.find("`", digest_at)
        if digest_end < 0:
            raise ValueError(f"missing or malformed pinned prompt digest: {name}")
        section = section[:digest_at] + digest + section[digest_end:]
        digests[name] = digest
    plan_path.write_text(plan[:start] + section + plan[end:], encoding="utf-8")
    return digests


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Check the whole frozen prompt surface and print its byte hashes.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero when every prompt matches its pin, otherwise non-zero.
    """

    args = build_parser().parse_args(argv)
    try:
        if args.update_plan:
            update_pinned_digests(
                args.plan,
                args.author_prompt,
                args.checker_prompt,
                args.pool_root,
            )
        digests = verify_prompt_surface(
            args.plan,
            args.author_prompt,
            args.checker_prompt,
            args.pool_root,
        )
    except (OSError, ValueError) as exc:
        print(f"prompt verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"prompt_sha256": digests}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
