"""Real streaming-envelope probe for the pinned author CLI (B1-XV12 / DV-4).

Every 200a author-lane weather decision keys on structured harness-envelope
fields (``is_error``, ``subtype``, ``terminal_reason``, the rate-limit block),
and a NEW harness release silently reshaping that envelope would make paid
deaths chargeable again with no alarm (union failure mode #34). This tool runs
ONE real ``claude -p`` round trip through the same binary resolution and
``--output-format json`` shape the executor uses, and freezes the REDACTED
envelope as a per-CLI-version fixture under
``menagerie/crawler/tests/fixtures/harness_envelopes/``.

Discipline (the pin-bump tripwire): a fixture must exist for the EXACT pinned
:data:`~menagerie.crawler.constants.CLAUDE_CLI_VERSION` --
``test_author_stream_envelope_fixture.py`` fails any pin bump until a fixture
is captured with the new binary, so an unattested envelope can never reach a
paid leg. Free-text fields are digested, never stored: the fixture is a shape
witness, not a transcript.

Usage::

    # capture (runs one real, tiny session against the pinned CLI)
    PYTHONPATH=. python -m menagerie.crawler.tools.author_stream_probe --capture

    # check only (no network, no session): the pinned version has a fixture
    PYTHONPATH=. python -m menagerie.crawler.tools.author_stream_probe --check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from menagerie.crawler.author_executor import (
    ExecutorConfig,
    _parse_harness_json,
    _verify_claude_cli_version,
)
from menagerie.crawler.constants import CLAUDE_CLI_VERSION
from menagerie.crawler.identity import hash_bytes, utc_now
from menagerie.crawler.models import JsonObject

FIXTURE_SCHEMA_VERSION = "menagerie.crawler.author-harness-envelope-fixture.v1"

#: Repository-relative fixture registry, one file per pinned CLI version.
FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "harness_envelopes"

#: Envelope free-text fields; their VALUES are digested, never stored.
_TEXT_FIELDS = frozenset({"result", "error", "message"})

_DEFAULT_PROMPT = "Reply with exactly: ok"


def fixture_path(version: str, root: Optional[Path] = None) -> Path:
    """Return the fixture path for one CLI version."""

    return (root or FIXTURE_ROOT) / f"v{version}.json"


def redact_envelope(envelope: Mapping[str, Any]) -> JsonObject:
    """Return the envelope with free text digested and structure preserved.

    Nested mappings keep their key structure (the rate-limit block's shape is
    exactly what the weather arms key on); strings in known free-text fields
    become ``sha256:...`` digests so a transcript can never ride into the
    repository through a fixture.
    """

    redacted: JsonObject = {}
    for key, value in envelope.items():
        if key in _TEXT_FIELDS and isinstance(value, str):
            redacted[key] = {
                "redacted_sha256": hash_bytes(value.encode("utf-8")),
                "chars": len(value),
            }
        elif isinstance(value, Mapping):
            redacted[key] = redact_envelope(value)
        else:
            redacted[key] = value
    return redacted


def capture_fixture(
    *,
    prompt: str = _DEFAULT_PROMPT,
    timeout_seconds: float = 240.0,
    root: Optional[Path] = None,
) -> Path:
    """Run one real pinned-CLI session and freeze its redacted envelope.

    Returns
    -------
    Path
        The written fixture path.

    Raises
    ------
    RuntimeError
        If the binary is not the pinned version, the session cannot run, or
        the stdout carries no parseable harness envelope.
    """

    config = ExecutorConfig.from_env()
    _verify_claude_cli_version(config.claude_command)
    argv = [
        *config.claude_command,
        "-p",
        prompt,
        "--setting-sources",
        "",
        "--output-format",
        "json",
    ]
    started = time.monotonic()
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    wall = time.monotonic() - started
    envelope = _parse_harness_json(completed.stdout or "")
    if envelope is None:
        raise RuntimeError(
            "the pinned CLI emitted no parseable harness JSON envelope "
            f"(rc={completed.returncode}, stdout tail: {(completed.stdout or '')[-400:]!r})"
        )
    fixture: JsonObject = {
        "fixture_version": FIXTURE_SCHEMA_VERSION,
        "cli_version": CLAUDE_CLI_VERSION,
        "captured_at": utc_now(),
        "probe_prompt": prompt,
        "returncode": completed.returncode,
        "wall_seconds": round(wall, 3),
        "envelope_top_level_keys": sorted(envelope.keys()),
        "envelope": redact_envelope(envelope),
        "raw_stdout_sha256": hash_bytes((completed.stdout or "").encode("utf-8")),
    }
    target = fixture_path(CLAUDE_CLI_VERSION, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def check_fixture(root: Optional[Path] = None) -> list[str]:
    """Return the failure list for the pinned version's fixture (empty = good)."""

    target = fixture_path(CLAUDE_CLI_VERSION, root)
    if not target.is_file():
        return [
            f"no captured harness-envelope fixture for pinned CLI {CLAUDE_CLI_VERSION} "
            f"(expected {target}); capture one with the NEW binary before bumping the pin: "
            "PYTHONPATH=. python -m menagerie.crawler.tools.author_stream_probe --capture"
        ]
    try:
        fixture = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"fixture unreadable: {exc}"]
    failures: list[str] = []
    if fixture.get("fixture_version") != FIXTURE_SCHEMA_VERSION:
        failures.append("fixture_version mismatch")
    if fixture.get("cli_version") != CLAUDE_CLI_VERSION:
        failures.append(
            f"fixture captured for {fixture.get('cli_version')}, pin is {CLAUDE_CLI_VERSION}"
        )
    if not isinstance(fixture.get("envelope"), Mapping):
        failures.append("fixture carries no envelope")
    return failures


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--capture", action="store_true", help="run one real probe session")
    action.add_argument("--check", action="store_true", help="verify the pinned fixture exists")
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--timeout", type=float, default=240.0)
    arguments = parser.parse_args(argv)
    if arguments.check:
        failures = check_fixture()
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1 if failures else 0
    target = capture_fixture(prompt=arguments.prompt, timeout_seconds=arguments.timeout)
    print(f"captured harness envelope fixture: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
