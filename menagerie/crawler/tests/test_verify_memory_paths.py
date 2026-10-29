"""D-65 (PM-14) stale-knowledge tripwire: durable-doc path claims verify against the tree.

The 200a-era failure class: durable operator knowledge (agent guides, the
green-suite runner block, gate templates) names repo paths, pytest markers,
lock families, and MENAGERIE_* environment variables that silently rot as the
tree moves (the round19-prefix memory incident). ``verify_memory_paths`` is
the machine check: every such claim in the committed agent docs must resolve
against the CURRENT tree, fail-closed, with a closed runtime-path allowlist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.tools.verify_memory_paths import (
    LOCK_FAMILY_PREFIX,
    MEMORY_DOCUMENTS,
    extract_claims,
    main,
    verify_documents,
)

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_doc(tmp_path: Path, body: str) -> Path:
    doc = tmp_path / "doc.md"
    doc.write_text(body, encoding="utf-8")
    return doc


def test_stale_repo_path_claim_is_flagged(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path, "See `menagerie/crawler/does_not_exist_anywhere.py` for detail.\n")
    report = verify_documents(_REPO_ROOT, [doc])
    stale = [row for row in report["rows"] if row["status"] == "stale"]
    assert len(stale) == 1
    row = stale[0]
    assert row["kind"] == "repo-path"
    assert row["claim"] == "menagerie/crawler/does_not_exist_anywhere.py"
    assert report["status"] == "stale"


def test_present_repo_path_claim_is_clean(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path, "Run `menagerie/crawler/tools/gate_verdicts.py` first.\n")
    report = verify_documents(_REPO_ROOT, [doc])
    assert report["status"] == "clean"
    assert [row["status"] for row in report["rows"]] == ["present"]


def test_lock_family_claim_outside_committed_family_is_flagged(tmp_path: Path) -> None:
    doc = _write_doc(
        tmp_path,
        "--lock menagerie/crawler/envs/locks/round21-osx-arm64.lock\n",
    )
    report = verify_documents(_REPO_ROOT, [doc])
    stale = [row for row in report["rows"] if row["status"] == "stale"]
    assert stale, "a lock path outside the committed family must be flagged"
    assert any(LOCK_FAMILY_PREFIX in row["detail"] for row in stale)


def test_marker_claim_must_be_declared(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path, "run pytest -m round22_neon_real to check\n")
    report = verify_documents(_REPO_ROOT, [doc])
    stale = [row for row in report["rows"] if row["status"] == "stale"]
    assert [row["claim"] for row in stale] == ["round22_neon_real"]
    assert stale[0]["kind"] == "pytest-marker"


def test_known_marker_claim_is_clean(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path, "gated by -m round21_macos_real on macOS\n")
    report = verify_documents(_REPO_ROOT, [doc])
    assert report["status"] == "clean"


def test_env_var_claim_must_be_read_in_source(tmp_path: Path) -> None:
    # Assembled dynamically so this test file itself (which the source scan
    # covers) never contains the literal token.
    fake_var = "MENAGERIE_NO_SUCH_" + "VARIABLE_XYZ"
    doc = _write_doc(
        tmp_path,
        f"export {fake_var}=1 and MENAGERIE_RELEASE_GATE=1\n",
    )
    report = verify_documents(_REPO_ROOT, [doc])
    by_claim = {row["claim"]: row for row in report["rows"] if row["kind"] == "env-var"}
    assert by_claim[fake_var]["status"] == "stale"
    assert by_claim["MENAGERIE_RELEASE_GATE"]["status"] == "present"


def test_extract_claims_skips_shell_interpolations() -> None:
    claims = extract_claims('MENAGERIE_REAL_ENV_PREFIX="$PREFIX" tests/$NAME/x.py\n')
    assert all("$" not in claim.value for claim in claims)


def test_live_agent_documents_verify_clean() -> None:
    """The committed agent docs (CLAUDE.md, AGENTS.md) carry zero stale claims.

    This is the standing tripwire: any future rename that leaves the agent
    guides pointing at a vanished path, marker, lock family, or env var goes
    RED here instead of waiting for the next confused operator session.
    """

    docs = [_REPO_ROOT / name for name in MEMORY_DOCUMENTS]
    report = verify_documents(_REPO_ROOT, docs)
    stale = [row for row in report["rows"] if row["status"] == "stale"]
    assert stale == [], f"stale durable-knowledge claims: {json.dumps(stale, indent=2)}"
    assert report["status"] == "clean"


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    clean_doc = _write_doc(tmp_path, "see menagerie/crawler/tools/gate_verdicts.py\n")
    assert main(["--repo-root", str(_REPO_ROOT), "--doc", str(clean_doc)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "clean"

    stale_doc = tmp_path / "stale.md"
    stale_doc.write_text("see menagerie/nowhere/missing.py\n", encoding="utf-8")
    assert main(["--repo-root", str(_REPO_ROOT), "--doc", str(stale_doc)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "stale"
