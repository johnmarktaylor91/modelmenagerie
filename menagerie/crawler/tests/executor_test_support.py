"""Shared support for author-executor tests: fake harness, fixtures, envelopes.

The fake ``claude`` binary honors the executor's real invocation contract: it
parses ``-p``/``--session-id``/``--resume``, extracts its output paths from the
rendered brief exactly as a real session would, records every invocation, and
prints a harness-shaped JSON result. Behavior is selected by environment
variables so kill/timeout/limit scenarios are scriptable.
"""

from __future__ import annotations

import json
import os
import sys

from menagerie.crawler.constants import CLAUDE_CLI_VERSION
from copy import deepcopy
from pathlib import Path

RESOLVED_SHA = "a" * 39 + "b"
COMMITS_URL = "https://api.github.com/repos/acme/widgets/commits/v1.0.0"
#: A ref the forge answers `404` for. This is what a fabricated SHA actually
#: looks like from GitHub, and the fixture says so explicitly rather than
#: leaving the URL unmapped: an unmapped URL is a transport failure, which is
#: *our* problem, and a test must not prove "bad ref" from that.
FABRICATED_SHA = "1" * 40
FABRICATED_COMMITS_URL = f"https://api.github.com/repos/acme/widgets/commits/{FABRICATED_SHA}"
RAW_URL = f"https://raw.githubusercontent.com/acme/widgets/{RESOLVED_SHA}/models/net.py"
IMPL_BODY = "class Net:\n    pass\n"
SUPPLEMENT_URL = "https://example.org/extra.txt"

DEFAULT_DISCOVERY = {
    "arm": "FOUND",
    "sources": [
        {
            "source_id": "impl-net",
            "kind": "forge-file",
            "repo": "github.com/acme/widgets",
            "path": "models/net.py",
            "ref": "v1.0.0",
            "requested_role": "implementation",
            "media_type_hint": "text/x-python",
            "basis": "The fixture repository owns the model implementation.",
        }
    ],
}

DEFAULT_RESULT = {
    "kind": "BLOCKED",
    "payload": {
        "stage": "source",
        "reason_code": "missing-material-source",
        "prerequisite_ids": ["source-needed"],
        "evidence_ids": ["evidence-gap"],
    },
}

_FAKE_CLAUDE_SOURCE = r"""
import json, os, re, sys, time, uuid

argv = sys.argv[1:]

if "--version" in argv:
    # Tracks the executor's pin via the environment the rig injects, so a
    # deliberate CLI adoption never breaks nine scenario tests (2.1.239 bump,
    # 2026-08-22).
    print(os.environ.get("FAKE_CLAUDE_VERSION", "0.0.0") + " (Claude Code)")
    raise SystemExit(0)


def flag(name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


prompt = flag("-p") or ""
mode = os.environ.get("FAKE_CLAUDE_MODE", "happy")


def extract(label):
    matched = re.search(re.escape(label) + r"[^`]*`([^`]+)`", prompt)
    return matched.group(1) if matched else None


discovery_path = extract("- DISCOVERY output path, exact:")
result_path = extract("- RESULT output path, exact:")
required_path = extract("- REQUIRED output path, exact:")
if "PRE-PUBLICATION VALIDATION FAILED" in prompt:
    stage = "repair"
elif "WALL CLAIM REFUSED" in prompt:
    stage = "wall-refusal"
elif discovery_path:
    stage = "stage1"
elif "SUPPLEMENTARY SOURCE ROUND" in prompt:
    stage = "supplement"
elif result_path:
    stage = "stage2"
elif required_path:
    stage = "probe"
else:
    stage = "unknown"

log_dir = os.environ.get("FAKE_CLAUDE_LOG")
if log_dir:
    with open(os.path.join(log_dir, "invocations.jsonl"), "a") as fh:
        fh.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "stage": stage,
                    "argv": argv,
                    "prompt": prompt,
                    "claude_config_dir": os.environ.get("CLAUDE_CONFIG_DIR"),
                    "pythonwarnings": os.environ.get("PYTHONWARNINGS"),
                }
            )
            + "\n"
        )
marker_dir = os.environ.get("FAKE_CLAUDE_MARKER")
if marker_dir:
    with open(os.path.join(marker_dir, stage + ".started"), "w") as fh:
        fh.write(str(os.getpid()))


def emit_harness(rc=0):
    sid = flag("--session-id") or str(uuid.uuid4())
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "duration_ms": 1234,
                "duration_api_ms": 987,
                "num_turns": 7,
                "session_id": sid,
                "total_cost_usd": 0.0123,
                "usage": {
                    "input_tokens": 111,
                    "output_tokens": 222,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 20,
                },
                "result": "done",
            }
        )
    )
    sys.exit(rc)


sleep_s = float(os.environ.get("FAKE_CLAUDE_SLEEP", "120"))

if mode == "sleep-" + stage:
    if os.environ.get("FAKE_CLAUDE_STDERR_EARLY"):
        print(os.environ["FAKE_CLAUDE_STDERR_EARLY"], file=sys.stderr, flush=True)
    time.sleep(sleep_s)
    sys.exit(0)
if mode == "late-write-stage2" and stage == "stage2":
    time.sleep(sleep_s)
    if result_path:
        # Land the late write atomically: tests wait on the file's existence,
        # so a create-then-write here races their read to an empty file.
        late_path = result_path + ".late-tmp"
        with open(late_path, "w") as fh:
            fh.write(os.environ.get("FAKE_CLAUDE_LATE_RESULT", '{"kind": "STALE"}'))
        os.replace(late_path, result_path)
    sys.exit(0)
if mode == "limit":
    # Real Claude Code result envelopes, not invented ones. FAKE_CLAUDE_LIMIT_SHAPE
    # selects which observed shape to replay; see _LIMIT_TERMINAL_REASONS in
    # author_executor.py for the citation of every field used here.
    shape = os.environ.get("FAKE_CLAUDE_LIMIT_SHAPE", "blocking_limit")
    payload = {
        "type": "result",
        "is_error": True,
        "session_id": str(uuid.uuid4()),
        "duration_ms": 1234,
        "num_turns": 1,
        "total_cost_usd": 0.0,
    }
    if shape == "blocking_limit":
        payload.update(subtype="error_during_execution", terminal_reason="blocking_limit")
    elif shape == "rapid_refill_breaker":
        payload.update(
            subtype="error_during_execution", terminal_reason="rapid_refill_breaker"
        )
    elif shape in ("five_hour", "seven_day"):
        payload.update(
            subtype="error_during_execution",
            terminal_reason="blocking_limit",
            rate_limit_info={
                "status": "rejected",
                "rateLimitType": shape,
                "resetsAt": int(os.environ["FAKE_CLAUDE_LIMIT_RESETS_AT"]),
            },
        )
    elif shape == "rate_limit_info_only":
        payload.update(
            subtype="error_during_execution",
            rate_limit_info={"status": "rejected", "rateLimitType": "seven_day"},
        )
    elif shape == "api_error_status":
        payload.update(subtype="success", api_error_status=429, result="")
    elif shape == "raw_api_error":
        payload = {"type": "error", "error": {"type": "rate_limit_error"}}
    elif shape == "generic_crash":
        # The generic failure subtype with NO limit terminal reason: an ordinary
        # session error that must stay a retry, never a campaign-wide pause.
        payload.update(subtype="error_during_execution", terminal_reason="model_error")
    else:
        raise SystemExit(f"unknown FAKE_CLAUDE_LIMIT_SHAPE {shape!r}")
    print(json.dumps(payload))
    sys.exit(1)
if mode == "outage":
    # The two REAL instant-crash shapes rung 7 and the clone-2 mini-rung
    # observed in the wild: the harness result document carries the cause in
    # its `result` text, the process exits 1 in under a second, and stderr is
    # empty. FAKE_CLAUDE_OUTAGE_SHAPE selects which observed shape to replay.
    shape = os.environ.get("FAKE_CLAUDE_OUTAGE_SHAPE", "not_logged_in")
    texts = {
        "not_logged_in": "Not logged in · Please run /login",
        "overloaded_529": "API Error: Repeated 529 Overloaded errors from the provider",
    }
    if shape not in texts:
        raise SystemExit(f"unknown FAKE_CLAUDE_OUTAGE_SHAPE {shape!r}")
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "session_id": str(uuid.uuid4()),
                "duration_ms": 618,
                "duration_api_ms": 114,
                "num_turns": 0,
                "total_cost_usd": 0.0,
                "result": texts[shape],
            }
        )
    )
    sys.exit(1)
if mode == "crash":
    print("boom, not json")
    print("GitHub API rate limit exceeded for 1.2.3.4", file=sys.stderr)
    sys.exit(3)
if mode == "resume-fail" and "--resume" in argv:
    sys.exit(5)
if mode == "content-refusal":
    # The A5-audit content-refusal shape: the session runs to completion
    # (envelope subtype success, rc=0), declines the work in prose carried in
    # `result`, and writes NO output file at all.
    refusal = os.environ.get(
        "FAKE_CLAUDE_REFUSAL_TEXT",
        "I can't help with authoring this model. The described architecture"
        " is purpose-built for generating non-consensual imagery of"
        " identifiable people, and providing a working implementation would"
        " directly enable that use.",
    )
    sid = flag("--session-id") or flag("--resume") or str(uuid.uuid4())
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "duration_ms": 61842,
                "duration_api_ms": 59000,
                "num_turns": 4,
                "session_id": sid,
                "total_cost_usd": 0.31,
                "usage": {
                    "input_tokens": 111,
                    "output_tokens": 222,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 20,
                },
                "result": refusal,
            }
        )
    )
    sys.exit(0)

if stage == "stage1" and discovery_path:
    payload = os.environ.get("FAKE_CLAUDE_DISCOVERY")
    if not payload:
        discovery = json.loads(os.environ["FAKE_CLAUDE_DEFAULT_DISCOVERY"])
        payload = json.dumps(discovery)
    with open(discovery_path, "w") as fh:
        fh.write(payload)
elif stage == "stage2" and result_path:
    if mode in (
        "supplement-request",
        "supplement-result-then-timeout",
        "supplement-result-then-crash",
    ):
        supp = os.path.join(os.path.dirname(result_path), "supplement-request.json")
        with open(supp, "w") as fh:
            fh.write(
                json.dumps(
                    {
                        "sources": [
                            {
                                "source_id": "supp-1",
                                "kind": "raw-url",
                                "url": os.environ["FAKE_CLAUDE_SUPPLEMENT_URL"],
                                "requested_role": "documentation",
                                "basis": "Supplementary fixture documentation.",
                            }
                        ],
                        "why": "one more file",
                    }
                )
            )
    else:
        with open(result_path, "w") as fh:
            fh.write(
                os.environ.get("FAKE_CLAUDE_RESULT")
                or os.environ["FAKE_CLAUDE_DEFAULT_RESULT"]
            )
elif stage == "supplement" and result_path:
    # The m8684 shape (F51 item 8, R-2b): the supplement session writes a
    # COMPLETE result atomically, then hangs until the external wall kill. The
    # result is on disk when SIGTERM lands, so result-on-disk-wins must settle
    # and publish it rather than discarding it as supplement-session-failed.
    body = os.environ.get("FAKE_CLAUDE_RESULT") or os.environ["FAKE_CLAUDE_DEFAULT_RESULT"]
    tmp_result = result_path + ".supp-tmp"
    with open(tmp_result, "w") as fh:
        fh.write(body)
    os.replace(tmp_result, result_path)
    if mode == "supplement-result-then-timeout":
        # Hang until the external wall kill: SIGTERM lands with the complete
        # result already on disk (the true m8684 shape).
        time.sleep(sleep_s)
    if mode == "supplement-result-then-crash":
        # The same `timed_out or returncode != 0` arm, reached instantly by a
        # non-zero exit with the complete result already on disk. No harness
        # document, so the session looks crashed to the executor.
        sys.exit(1)
elif stage == "repair" and result_path:
    # The pre-publication repair round: a scripted session either fixes its
    # result (FAKE_CLAUDE_REPAIRED_RESULT; FAKE_CLAUDE_REPAIRED_RESULT_N
    # scripts round N when a multi-round scenario needs distinct writes) or
    # rewrites the same broken one. The log already carries THIS invocation,
    # so the count of "repair" lines IS the current round number.
    round_number = 1
    if log_dir:
        try:
            with open(os.path.join(log_dir, "invocations.jsonl")) as fh:
                round_number = max(
                    1, sum(1 for line in fh if '"stage": "repair"' in line)
                )
        except OSError:
            round_number = 1
    with open(result_path, "w") as fh:
        fh.write(
            os.environ.get("FAKE_CLAUDE_REPAIRED_RESULT_%d" % round_number)
            or os.environ.get("FAKE_CLAUDE_REPAIRED_RESULT")
            or os.environ.get("FAKE_CLAUDE_RESULT")
            or os.environ["FAKE_CLAUDE_DEFAULT_RESULT"]
        )
elif stage == "wall-refusal" and result_path:
    # The refused-wall-claim round: a scripted session either resumes the work
    # (FAKE_CLAUDE_CONTINUED_RESULT) or insists by leaving its claim in place.
    continued = os.environ.get("FAKE_CLAUDE_CONTINUED_RESULT")
    if continued:
        with open(result_path, "w") as fh:
            fh.write(continued)
elif stage == "probe" and required_path:
    # Default: evidence shaped like a session that GENUINELY exercised all
    # three tools, derived from the prompt's own challenge facts so the
    # executor-side proof (nonce echo, live URLs, digest consistency,
    # version corroboration) can pass. FAKE_CLAUDE_PROBE=hollow writes the
    # evidence of a session that researched nothing, which the executor must
    # REFUSE to turn into a receipt.
    if os.environ.get("FAKE_CLAUDE_PROBE") == "hollow":
        evidence = {"challenge_id": "test", "tools": {}}
    else:
        import hashlib
        from datetime import datetime, timezone

        package = extract("- challenge package:") or "requests"
        metadata_url = extract("- challenge metadata URL:") or ""
        challenge_id = extract("- challenge_id:") or ""
        nonce_fact = extract("- probe nonce:") or ""
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        version = "9.9.9"
        serial = 20260728
        content = json.dumps(
            {"info": {"version": version}, "last_serial": serial, "pad": "x" * 900}
        )
        common = {"nonce": nonce_fact, "observed_at": now_iso}
        evidence = {
            "challenge_id": challenge_id,
            "tools": {
                "WebSearch": dict(
                    common,
                    registered_tool_name="WebSearch",
                    query=package + " pypi latest version",
                    reported_version=version,
                    results=[
                        {
                            "url": "https://pypi.org/project/" + package + "/",
                            "title": package + " on PyPI",
                            "excerpt": "latest release " + version,
                        },
                        {
                            "url": "https://libraries.io/pypi/" + package,
                            "title": package + " release history",
                            "excerpt": "current version " + version,
                        },
                    ],
                ),
                "mcp__exa__web_search_exa": dict(
                    common,
                    registered_tool_name="mcp__exa__web_search_exa",
                    query=package + " latest release",
                    reported_version=version,
                    results=[
                        {
                            "url": "https://snyk.io/advisor/python/" + package,
                            "title": package + " package health",
                            "text": (
                                "The package "
                                + package
                                + " has current released version "
                                + version
                                + ". "
                            )
                            * 8,
                        },
                        {
                            "url": "https://pypi.org/project/"
                            + package
                            + "/"
                            + version
                            + "/",
                            "title": package + " " + version,
                            "text": "release page",
                        },
                    ],
                ),
                "mcp__exa__web_fetch_exa": dict(
                    common,
                    registered_tool_name="mcp__exa__web_fetch_exa",
                    url=metadata_url,
                    content=content,
                    content_sha256=hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                    reported_version=version,
                    reported_last_serial=serial,
                ),
            },
        }
    with open(required_path, "w") as fh:
        fh.write(json.dumps(evidence))
if os.environ.get("FAKE_CLAUDE_STDERR_NOISE"):
    print(os.environ["FAKE_CLAUDE_STDERR_NOISE"], file=sys.stderr)
emit_harness()
"""


def write_fake_claude(directory: Path) -> Path:
    """Write the fake harness script and return its path."""

    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "fake_claude.py"
    script.write_text(_FAKE_CLAUDE_SOURCE, encoding="utf-8")
    return script


def write_broker_fixtures(directory: Path) -> Path:
    """Write the hermetic broker transport fixtures and return the root."""

    directory.mkdir(parents=True, exist_ok=True)
    index = {
        COMMITS_URL: {"status": 200, "body_text": json.dumps({"sha": RESOLVED_SHA})},
        RAW_URL: {"status": 200, "body_text": IMPL_BODY},
        SUPPLEMENT_URL: {"status": 200, "body_text": "extra documentation\n"},
        FABRICATED_COMMITS_URL: {
            "status": 404,
            "body_text": json.dumps({"message": "No commit found for SHA"}),
        },
    }
    (directory / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return directory


def executor_environment(
    monkeypatch,
    *,
    fake_claude: Path,
    fixtures: Path,
    log_dir: Path,
    campaign: str = "c1-mech",
) -> None:
    """Point the executor at the fake harness and hermetic broker fixtures."""

    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MENAGERIE_AUTHOR_CLAUDE_BIN", f'"{sys.executable}" "{fake_claude}"')
    monkeypatch.setenv("MENAGERIE_BROKER_FIXTURES", str(fixtures))
    monkeypatch.setenv("MENAGERIE_CAMPAIGN_ID", campaign)
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log_dir))
    monkeypatch.setenv("FAKE_CLAUDE_VERSION", CLAUDE_CLI_VERSION)
    monkeypatch.setenv("FAKE_CLAUDE_DEFAULT_DISCOVERY", json.dumps(DEFAULT_DISCOVERY))
    monkeypatch.setenv("FAKE_CLAUDE_DEFAULT_RESULT", json.dumps(DEFAULT_RESULT))
    monkeypatch.setenv("FAKE_CLAUDE_SUPPLEMENT_URL", SUPPLEMENT_URL)
    monkeypatch.delenv("FAKE_CLAUDE_MODE", raising=False)
    monkeypatch.delenv("FAKE_CLAUDE_PROBE", raising=False)
    monkeypatch.delenv("MENAGERIE_EXECUTOR_PAUSE_AFTER", raising=False)


def subprocess_environment(
    *,
    fake_claude: Path,
    fixtures: Path,
    log_dir: Path,
    campaign: str = "c1-mech",
    **extra: str,
) -> dict:
    """Build the environment for running the executor as a real subprocess."""

    log_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    env = {
        **os.environ,
        "PYTHONPATH": str(repo_root),
        "MENAGERIE_AUTHOR_CLAUDE_BIN": f'"{sys.executable}" "{fake_claude}"',
        "MENAGERIE_BROKER_FIXTURES": str(fixtures),
        "MENAGERIE_CAMPAIGN_ID": campaign,
        "FAKE_CLAUDE_LOG": str(log_dir),
        "FAKE_CLAUDE_VERSION": CLAUDE_CLI_VERSION,
        "FAKE_CLAUDE_DEFAULT_DISCOVERY": json.dumps(DEFAULT_DISCOVERY),
        "FAKE_CLAUDE_DEFAULT_RESULT": json.dumps(DEFAULT_RESULT),
        "FAKE_CLAUDE_SUPPLEMENT_URL": SUPPLEMENT_URL,
    }
    env.pop("MENAGERIE_EXECUTOR_PAUSE_AFTER", None)
    env.pop("FAKE_CLAUDE_MODE", None)
    env.update(extra)
    return env


def write_source_request(root: Path, stable_id: str) -> Path:
    """Write one lane-shaped source-request envelope under the author root."""

    root.mkdir(parents=True, exist_ok=True)
    request = {
        "envelope_version": "menagerie.crawler.author-source-request.v1",
        "work_id": f"work-{stable_id}",
        "stable_id": stable_id,
        "untrusted_hints": {"name": stable_id, "zoo": "test-zoo"},
        "required_output_path": str(root / "source-targets.json"),
        "max_sources": 8,
        "discovery_schema_version": "menagerie.crawler.source-discovery.v1",
    }
    path = root / "source-request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


#: The machine-held identity preimages every real author envelope discloses. The
#: executor stamps ``proposal.author`` from ``author`` and refuses a request without
#: it, so an envelope fixture omitting this block is invalid rather than minimal.
#: Single-sourced here so the several inline request stubs across the executor tests
#: cannot drift into disclosing different bindings.
AUTHOR_IDENTITY_INPUTS: dict[str, object] = {
    "author": {
        "provider": "anthropic",
        "model": "claude-sonnet",
        "version": "current",
        "prompt_sha256": "sha256:" + "4" * 64,
    },
    "checker": {
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "version": "current",
        "prompt_sha256": "sha256:" + "9" * 64,
    },
    "model_schema_version": "menagerie.crawler.model.v3",
}


def write_author_envelope(root: Path, stable_id: str) -> Path:
    """Write one minimal author envelope stub under the author root.

    The executor treats the envelope as opaque authority (the lane built and
    validated it); only the transport fields the executor reads are present.
    """

    root.mkdir(parents=True, exist_ok=True)
    (root / "model").mkdir(exist_ok=True)
    request = {
        "envelope_version": "menagerie.crawler.author-envelope.v3",
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "campaign_id": "c1-mech",
        "expected_result": {
            "schema_version": "menagerie.crawler.author-result.v4",
            "stable_id": stable_id,
            "work_id": f"work-{stable_id}",
            "campaign_id": "c1-mech",
            "author_identity": "sha256:" + "3" * 64,
            "prompt_identity": "sha256:" + "4" * 64,
            "dispatcher_identity": "sha256:" + "5" * 64,
            "source_manifest_identity": "sha256:" + "6" * 64,
            "intake_snapshot_id": "intake-test",
            "intake_snapshot_sha256": "sha256:" + "7" * 64,
            "intake_item_sha256": "sha256:" + "8" * 64,
        },
        "identity_inputs": deepcopy(AUTHOR_IDENTITY_INPUTS),
        "source_manifest": {
            "manifest_sha256": "sha256:" + "6" * 64,
            "sources": [{"source_id": "impl-net"}],
        },
        "required_output_path": str(root / "result.json"),
        "allowed_model_dir": str(root / "model"),
        "prompt": {"path": str(root / "prompts" / "author.txt")},
    }
    (root / "prompts").mkdir(exist_ok=True)
    (root / "prompts" / "author.txt").write_text("canonical author prompt", encoding="utf-8")
    path = root / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def read_invocations(log_dir: Path) -> list:
    """Return every recorded fake-harness invocation."""

    path = log_dir / "invocations.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
