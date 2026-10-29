"""Unit pins for environment-authority sealing (see composition suite for E2E)."""

from __future__ import annotations

from pathlib import Path


def test_seal_accepts_interpreter_reached_through_symlinked_prefix(tmp_path: Path) -> None:
    """A symlinked ancestor in the GIVEN prefix must not read as 'outside'.

    The pilot's env store lives behind a symlink (.crawl-local/envs -> SSD).
    The lexical containment check compares like with like; the security-bearing
    RESOLVED containment is exercised by the sibling refusal test below.
    """

    from menagerie.crawler.authority import _seal_environment_content

    real = tmp_path / "real-store" / "env"
    (real / "bin").mkdir(parents=True)
    interpreter = real / "bin" / "python"
    interpreter.write_bytes(b"#!/bin/sh\n")
    interpreter.chmod(0o755)
    link = tmp_path / "linked-store"
    link.symlink_to(tmp_path / "real-store")
    linked_prefix = link / "env"

    manifest = _seal_environment_content(linked_prefix, linked_prefix / "bin" / "python")
    assert manifest is not None


def test_seal_still_refuses_interpreter_truly_outside_prefix(tmp_path: Path) -> None:
    """The tripwire stays armed: a genuinely foreign interpreter refuses."""

    import pytest as _pytest
    from menagerie.crawler.authority import (
        AuthorityDerivationError,
        _seal_environment_content,
    )

    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    outside = tmp_path / "elsewhere" / "python"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"#!/bin/sh\n")
    outside.chmod(0o755)

    with _pytest.raises(AuthorityDerivationError, match="outside the canonical prefix"):
        _seal_environment_content(prefix, outside)
