"""Dependency-free worker completion-line protocol (DX-3/RE-3 closure diet).

The v3 worker proves completion by emitting one canonical line naming its raw
award receipt. That line is built INSIDE the routed model environment, so the
functions that build it must import there: this module's only crawler
dependency is :mod:`menagerie.crawler.identity` (stdlib-only), keeping the
worker's bootstrap closure free of ``authority -> host_capacity -> recordio ->
schema`` and therefore of driver-side dependencies such as ``jsonschema``
(step-200 leg 7: that chain killed every worker in the routed envs, and the
2026-08-22 StrEnum recurrence proved the class was still open).

:mod:`menagerie.crawler.authority` re-exports everything here, so parent-side
callers keep their historical import surface. :class:`AuthorityDerivationError`
is DEFINED here for the same reason -- the completion-line builders raise it,
and its class identity must stay one object for every ``except`` clause in the
driver tree.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from menagerie.crawler.identity import stable_hash

WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V3 "

_HASH_PREFIX = "sha256:"


class AuthorityDerivationError(ValueError):
    """Raised when retained facts do not form the frozen replayable proof graph."""


class RelativeImportUnbindableError(AuthorityDerivationError):
    """The closure derivation cannot bind a relative import to a lookup package.

    Item 9 / author-uplift R-7: vendoring an upstream PACKAGE (relative
    imports inside a vendored directory) hits a limit of the crawler's own
    static import-closure derivation, not a fact about the model (R6b m243:
    ``modeling_bd3lm.py`` -> ``scheduler-member-quarantined`` /
    ``member-fault``). The subclass keeps the ``AuthorityDerivationError``
    identity and message every existing ``except`` clause relies on; the
    scheduler's member-fault catch-alls read the type to park the member as
    ``non-model-fault`` (machinery, no verdict) instead of ``member-fault``.
    """


class GenerationExhaustedError(AuthorityDerivationError):
    """No single-environment-generation terminal proof exists (AT-5).

    The member's per-mode decisive attempts span multiple environment
    generations and no one generation carries a complete satisfying set:
    the terminal proof is generation-exhausted. Raised BEFORE any chain is
    spent, so the award seam can retire loudly (quarantine + TW-GENSPAN
    disclosure) instead of publishing and then refusing at append
    ("terminal proof spans multiple environment generations" -- m12646)."""


def require_nonempty_string(value: object, field: str) -> str:
    """Return one required non-empty string.

    Parameters
    ----------
    value:
        Candidate value.
    field:
        Field name used in the failure.

    Returns
    -------
    str
        Validated string.

    Raises
    ------
    AuthorityDerivationError
        If the value is absent or empty.
    """

    if not isinstance(value, str) or not value:
        raise AuthorityDerivationError(f"{field} must be a non-empty string")
    return value


def require_hash(value: object, field: str) -> str:
    """Return one required prefixed SHA-256 identity.

    Parameters
    ----------
    value:
        Candidate value.
    field:
        Field name used in the failure.

    Returns
    -------
    str
        Validated prefixed digest.

    Raises
    ------
    AuthorityDerivationError
        If the value is not a canonical SHA-256 identity.
    """

    digest = require_nonempty_string(value, field)
    if len(digest) != 71 or not digest.startswith(_HASH_PREFIX):
        raise AuthorityDerivationError(f"{field} must be a prefixed SHA-256 identity")
    try:
        int(digest.removeprefix(_HASH_PREFIX), 16)
    except ValueError as exc:
        raise AuthorityDerivationError(f"{field} must be a prefixed SHA-256 identity") from exc
    if digest != digest.lower():
        raise AuthorityDerivationError(f"{field} must use lowercase hexadecimal")
    return digest


def canonical_completion_payload(payload: Mapping[str, Any]) -> str:
    """Serialize a completion payload in its single canonical representation.

    Parameters
    ----------
    payload:
        Closed completion payload.

    Returns
    -------
    str
        Canonical JSON text.
    """

    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def raw_award_receipt_sha256(raw_award_receipt: Mapping[str, Any]) -> str:
    """Hash the exact canonical raw award receipt.

    Parameters
    ----------
    raw_award_receipt:
        Closed v3 raw worker receipt without a self-referential digest field.

    Returns
    -------
    str
        Canonical receipt digest.
    """

    return stable_hash(dict(raw_award_receipt))


def completion_line_for_raw_award_receipt(raw_award_receipt: Mapping[str, Any]) -> str:
    """Build the canonical completion line naming one raw receipt.

    Parameters
    ----------
    raw_award_receipt:
        Closed v3 raw receipt.

    Returns
    -------
    str
        Canonical parent-visible completion line without its trailing newline.
    """

    payload = {
        "raw_award_receipt_sha256": raw_award_receipt_sha256(raw_award_receipt),
        "request_nonce": require_nonempty_string(
            raw_award_receipt.get("request_nonce"), "raw_award_receipt.request_nonce"
        ),
        "request_sha256": require_hash(
            raw_award_receipt.get("request_sha256"), "raw_award_receipt.request_sha256"
        ),
    }
    return WORKER_COMPLETION_PREFIX + canonical_completion_payload(payload)
