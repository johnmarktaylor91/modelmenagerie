"""IC-7: the memo's implementation digests cover the validation import closure.

The environment identity that gates every memo hit hashed a HAND-LISTED four
implementation files; identity.py and mirrors.py (whose hashing and parsing a
memo hit skips re-running) were silently absent, so an edit there could keep
serving stale positive validations. The digest set is now DERIVED from the
package-local import closure of the validation entry modules -- a new
collaborator import joins automatically -- and the epoch bumped to v2 in the
same change (every pre-existing memo store invalidates).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.checkpoint_cache import (
    CHECKPOINT_CACHE_EPOCH,
    _VALIDATION_ENTRY_MODULES,
    _validation_import_closure,
    _validator_implementation_digests,
)

pytestmark = pytest.mark.smoke

CRAWLER_ROOT = Path(__file__).resolve().parents[1]


def test_digest_set_is_the_derived_import_closure() -> None:
    """Digest participants == the AST-derived closure, never a hand list."""

    closure = _validation_import_closure(CRAWLER_ROOT)
    digests = _validator_implementation_digests()
    assert set(digests) == set(closure)
    assert set(_VALIDATION_ENTRY_MODULES) <= set(closure)


def test_previously_absent_collaborators_are_covered() -> None:
    """The named IC-7 absentees that ARE import-reachable participate now."""

    digests = _validator_implementation_digests()
    for name in ("identity.py", "mirrors.py", "authority.py", "models.py", "constants.py"):
        assert name in digests, f"validation collaborator {name} missing from memo identity"
    for digest in digests.values():
        assert digest.startswith("sha256:") and len(digest) == 71


def test_epoch_bumped_with_the_closure_change() -> None:
    """The derived-closure change ships with its epoch bump: every v1 store
    invalidates rather than serving hits minted under the narrower identity."""

    assert CHECKPOINT_CACHE_EPOCH == "checkpoint-cache-epoch-v2"
