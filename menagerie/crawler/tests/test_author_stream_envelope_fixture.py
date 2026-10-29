"""B1-XV12: the pinned CLI's streaming envelope is fixture-attested per version.

Every author-lane weather decision keys on structured harness-envelope fields
(``is_error``, ``subtype``, ``terminal_reason``, the rate-limit block). A CLI
pin bump shipping a reshaped envelope would let paid deaths evade the closed
refund vocabulary with no alarm (200a union failure mode #34). These tests
make the pin and the captured envelope move TOGETHER: bumping
``CLAUDE_CLI_VERSION`` without capturing a fixture from the new binary is RED
here until someone runs

    PYTHONPATH=. python -m menagerie.crawler.tools.author_stream_probe --capture

against the new release.
"""

from __future__ import annotations

import json
from typing import Mapping

import pytest

from menagerie.crawler.author_executor import structured_limit_signal
from menagerie.crawler.constants import CLAUDE_CLI_VERSION
from menagerie.crawler.tools.author_stream_probe import (
    FIXTURE_SCHEMA_VERSION,
    check_fixture,
    fixture_path,
    redact_envelope,
)

#: The subtype union the pinned harness really emits (author_executor.py's
#: sdk.d.ts survey); an envelope outside it is an unattested-envelope alarm.
_KNOWN_SUBTYPES = frozenset(
    {
        "success",
        "error_during_execution",
        "error_max_turns",
        "error_max_budget_usd",
        "error_max_structured_output_retries",
    }
)


@pytest.mark.smoke
def test_pinned_cli_version_has_a_captured_envelope_fixture() -> None:
    """The pin-bump tripwire: pin and captured fixture move together."""

    assert check_fixture() == []


@pytest.mark.smoke
def test_captured_envelope_matches_the_classification_contract() -> None:
    """The frozen envelope's structured fields are ones the executor can read.

    A clean probe capture must parse as a non-error, non-limit envelope: a
    success that read as a pause (or an unknown subtype) means the harness
    reshaped the wire contract and the pin needs re-attestation, not trust.
    """

    fixture = json.loads(fixture_path(CLAUDE_CLI_VERSION).read_text(encoding="utf-8"))
    assert fixture["fixture_version"] == FIXTURE_SCHEMA_VERSION
    assert fixture["cli_version"] == CLAUDE_CLI_VERSION
    envelope = fixture["envelope"]
    assert isinstance(envelope, Mapping)
    assert envelope.get("is_error") is False
    assert envelope.get("subtype") in _KNOWN_SUBTYPES
    assert fixture["returncode"] == 0
    # A success capture may never be a pause authority.
    assert structured_limit_signal(dict(envelope)) is None


@pytest.mark.smoke
def test_fixture_redaction_digests_free_text() -> None:
    """No transcript rides into the repository through a fixture."""

    fixture = json.loads(fixture_path(CLAUDE_CLI_VERSION).read_text(encoding="utf-8"))
    result_field = fixture["envelope"].get("result")
    if result_field is not None:
        assert isinstance(result_field, Mapping)
        assert str(result_field.get("redacted_sha256", "")).startswith("sha256:")
    redacted = redact_envelope({"result": "secret text", "usage": {"input_tokens": 3}})
    assert redacted["result"]["redacted_sha256"].startswith("sha256:")
    assert redacted["usage"] == {"input_tokens": 3}
