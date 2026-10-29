"""BL-10 (e): the checker prompt identity is leg-pinned, never read live at seams.

Audit CH-F3: prompt bytes were hashed from the working tree at EVERY decision
seam (envelope build, adjudication identity, catch-up currency, identity
minting), so a mid-campaign prompt edit split the brain -- accepted gates stayed
durable while the award door demanded the new live hash, converting every
gated-but-unawarded model into full-cost unawardable work. The pilot carries
THREE prompt digests and prompt rotation was the proximate cause of the
award-eating incident; LADDER_PROTOCOL's "flips only at leg boundaries" was a
convention with no mechanism.

The mechanism, pinned here:

* one leg = one pin: the first checker decision of a driver process pins the
  prompt hash; every later decision consults the PIN;
* a live file that disagrees with the pin is a typed
  ``CheckerPromptRotatedError`` at the next dispatch -- mid-leg rotation is
  structurally impossible, not inadvisable;
* the pin is attested durably (``checker-prompt-pinned`` on the operational
  ledger) before any gate spend;
* an AST census over the checker cluster's modules proves the prompt file is
  read at exactly the sanctioned sites and nowhere else, so a new live-read
  seam cannot slip in silently.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import menagerie.crawler.checker_dispatch as checker_dispatch
from menagerie.crawler.checker_dispatch import (
    CheckerPromptRotatedError,
    _reset_checker_prompt_leg_pin_for_tests,
    leg_checker_prompt_sha256,
)
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import _driver, _paths, _snapshot

pytestmark = pytest.mark.smoke


@pytest.fixture()
def fresh_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Give the test its own prompt file and a clean pin, restoring both after."""

    prompt = tmp_path / "prompt.txt"
    prompt.write_bytes(b"frozen checker prompt v1\n")
    monkeypatch.setattr(checker_dispatch, "PROMPT_PATH", prompt)
    _reset_checker_prompt_leg_pin_for_tests()
    yield prompt
    _reset_checker_prompt_leg_pin_for_tests()


def test_first_use_pins_and_repeat_reads_agree(fresh_pin: Path) -> None:
    pinned = leg_checker_prompt_sha256()
    assert pinned == hash_bytes(b"frozen checker prompt v1\n")
    assert leg_checker_prompt_sha256() == pinned


def test_mid_leg_rotation_is_a_typed_refusal(fresh_pin: Path) -> None:
    """Editing the prompt under a running leg refuses at the next decision."""

    pinned = leg_checker_prompt_sha256()
    fresh_pin.write_bytes(b"frozen checker prompt v2 -- rotated mid-leg\n")
    with pytest.raises(CheckerPromptRotatedError, match="rotated mid-leg"):
        leg_checker_prompt_sha256()
    # The refusal is persistent, not once: every later seam keeps refusing.
    with pytest.raises(CheckerPromptRotatedError):
        leg_checker_prompt_sha256()
    # A fresh leg (new process, modeled by the test reset) re-pins cleanly.
    _reset_checker_prompt_leg_pin_for_tests()
    assert leg_checker_prompt_sha256() != pinned


def test_envelope_build_refuses_after_mid_leg_rotation(fresh_pin: Path, tmp_path: Path) -> None:
    """The dispatch seam itself refuses: no envelope is sealed under a split brain."""

    from menagerie.crawler.checker_dispatch import build_metadata_vet_envelope
    from menagerie.crawler.tests.conftest import make_gate
    from menagerie.crawler.tests.test_slice_d_checker_gates import _checker_item_pack

    item = _checker_item_pack(make_gate(["m_pin"])["items"][0])
    build = lambda: build_metadata_vet_envelope(  # noqa: E731
        [item],
        gate_round=1,
        output_path=tmp_path / "result.json",
        checker_model="gpt-5.6-sol",
        checker_version="current",
        request_nonce="pin-test",
        final_tail=True,
    )
    sealed = build()
    assert sealed["checker"]["prompt_sha256"] == leg_checker_prompt_sha256()
    fresh_pin.write_bytes(b"rotated\n")
    with pytest.raises(CheckerPromptRotatedError):
        build()


def test_leg_attests_its_prompt_pin_before_any_gate_spend(tmp_path: Path) -> None:
    """Every leg records ``checker-prompt-pinned`` on the operational ledger."""

    snapshot = _snapshot(tmp_path, count=1)
    result = _driver(tmp_path, snapshot).run()
    assert result.status == "complete"

    paths = _paths(tmp_path, snapshot)
    events = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition") == "checker-prompt-pinned"
    ]
    assert len(events) == 1
    details = events[0]["details"]
    assert details["checker_prompt_sha256"] == leg_checker_prompt_sha256()
    assert details["prompt_name"] == "codex_accuracy_checker_v2"


# ---------------------------------------------------------------------------
# The AST census: prompt reads happen at the sanctioned sites and nowhere else.
# ---------------------------------------------------------------------------

#: The checker cluster's decision-seam modules (BL-10's owner files that touch
#: checker identity). ``driver_models.py`` retains its own live reader outside
#: this cluster (BL-8 territory); every seam THESE modules own goes through the
#: pin.
_CENSUS_MODULES = (
    "checker_dispatch.py",
    "driver_admission.py",
    "operator_checker.py",
    "gate_reuse.py",
    "gates.py",
    "checker_lane.py",
    "operator_protocol.py",
    "metadata.py",
    "checker_calibration.py",
)

#: The ONLY functions allowed to open the prompt file: the pin's own reader,
#: and the wrapper's ``_build_prompt`` -- which VERIFIES the bytes it reads
#: against the envelope's pinned ``prompt.sha256`` and refuses typed on
#: disagreement, so its read carries no independent identity authority.
_SANCTIONED_READER = "_read_prompt"
_SANCTIONED_READ_SITES = {
    "checker_dispatch.py:_read_prompt",
    "operator_checker.py:_build_prompt",
}
_SANCTIONED_CALLERS = {"leg_checker_prompt_sha256"}


def _enclosing_function(tree: ast.Module, node: ast.AST) -> str:
    """Return the name of the function whose body contains ``node``."""

    for candidate in ast.walk(tree):
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(candidate):
                if child is node:
                    return candidate.name
    return "<module>"


def test_prompt_reads_census() -> None:
    """No decision seam reads the prompt file except through the leg pin.

    Two closed inventories, both derived from the AST of the checker cluster:

    * every attribute/name reference to ``PROMPT_PATH`` followed by a read
      (``read_bytes``/``read_text``) must live inside the sanctioned reader;
    * every call of the sanctioned reader must live inside the pin function.

    A new live-read seam -- the exact CH-F3 regression -- fails this census.
    """

    root = Path(checker_dispatch.__file__).parent
    read_sites: list[str] = []
    reader_calls: list[str] = []
    for name in _CENSUS_MODULES:
        source = (root / name).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("read_bytes", "read_text"):
                target = node.value
                if isinstance(target, ast.Name) and target.id == "PROMPT_PATH":
                    read_sites.append(f"{name}:{_enclosing_function(tree, node)}")
            if isinstance(node, ast.Call):
                func = node.func
                called = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else None
                )
                if called == _SANCTIONED_READER:
                    reader_calls.append(f"{name}:{_enclosing_function(tree, node)}")

    assert set(read_sites) == _SANCTIONED_READ_SITES, (
        f"unsanctioned prompt-file read sites: {read_sites}"
    )
    assert set(caller.split(":", 1)[1] for caller in reader_calls) <= _SANCTIONED_CALLERS, (
        f"unsanctioned callers of {_SANCTIONED_READER}: {reader_calls}"
    )


def test_no_seam_in_the_cluster_bypasses_the_pin_via_hash_bytes() -> None:
    """``hash_bytes(<prompt read>)`` inline is the historical bypass shape; gone."""

    root = Path(checker_dispatch.__file__).parent
    for name in _CENSUS_MODULES:
        source = (root / name).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=name)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != "hash_bytes":
                continue
            inner = ast.dump(node)
            assert "PROMPT_PATH" not in inner, (
                f"{name}: hash_bytes over a live prompt read bypasses the leg pin"
            )
