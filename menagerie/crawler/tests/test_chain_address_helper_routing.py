"""ONE helper computes every Track A content address, asserted structurally.

The globalized chain-address fix's claim is "one shared derivation across all twelve
chain writers and all twelve validators", and the point of the claim is that a writer
which cannot SPELL its own exclusion set cannot drift from the chain's. PSLRE6 acct F2
measured the claim and found it was a convention rather than an invariant: four Track A
writers still spelled ``row["row_sha256"] = stable_hash(row)`` and three validators
still spelled ``row.get("row_sha256") != stable_hash(body)``, each with its own private
notion of what the preimage excludes -- and NO census asserted the routing, so nothing
failed when one of them drifted.

Why a private preimage is a latent divergence and not a style nit: ``stable_hash(body)``
and ``row_content_address(body)`` agree exactly while the body carries no name from
``ROW_CONTENT_ADDRESS_EXCLUDED``, and diverge the moment it carries one. So a
self-derived writer mints rows the chain refuses (a permanent brick on that scope), and
a self-derived validator blesses rows the chain refuses or refuses rows it blesses. The
divergence is asserted below, on the same body, so the census can never be read as
tidiness.

The census is structural and total over the production tree: it is an AST pass, so a
comment cannot satisfy it and a new module is covered the moment it is written.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest

from menagerie.crawler.identity import (
    ROW_CONTENT_ADDRESS_EXCLUDED,
    row_content_address,
    stable_hash,
)

pytestmark = pytest.mark.smoke

_CRAWLER_ROOT = Path(__file__).resolve().parents[1]

#: The content-address field names a chained family may stamp. The envelope is not one
#: of them: it is attached by the chain, never stamped by a writer.
_ADDRESS_FIELDS = frozenset(ROW_CONTENT_ADDRESS_EXCLUDED - {"chain"})

#: The ONLY spellings that may PRODUCE a content address for one of those fields.
#: ``payload_hash`` is deliberately absent: it is the recordio families' shared
#: derivation, but stamping with it directly skips ``stamp_payload_digest``'s coupling
#: refusal (PSLRE6 D6-3), so a writer must use the stamping helper, not the hash.
_STAMP_HELPERS = frozenset({"stamp_row_digest", "stamp_payload_digest", "_stamped_record"})

#: The ONLY spellings that may RE-DERIVE a content address for comparison.
_DERIVE_HELPERS = frozenset(
    {"row_content_address", "row_content_digest", "payload_hash", "ladder_content_digest"}
)

#: Raw hashing primitives: correct for any digest that is not a chain content address,
#: and never correct for one that is.
_RAW_HASHES = frozenset({"stable_hash", "hash_bytes"})


def _production_modules() -> Iterator[Path]:
    """Yield every production crawler module, tests and caches excluded."""

    for path in sorted(_CRAWLER_ROOT.rglob("*.py")):
        if path.relative_to(_CRAWLER_ROOT).parts[0] in {"tests", "__pycache__"}:
            continue
        yield path


def _address_field(node: ast.expr) -> Optional[str]:
    """Return the content-address field one expression reads or writes, if any."""

    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value in _ADDRESS_FIELDS
    ):
        return str(node.slice.value)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"get", "pop"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value in _ADDRESS_FIELDS
    ):
        return str(node.args[0].value)
    return None


def _called_name(node: ast.expr) -> Optional[str]:
    """Return the name one expression calls, ``<computed>`` for an indirect call."""

    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return "<computed>"


def unrouted_address_sites(source: str, *, label: str = "<source>") -> list[str]:
    """Return every site that computes a Track A content address off the shared helper.

    THE scanner both the census and its own negative control run, so the census can
    never disagree with the check that proves it works. Two arms:

    * STAMP -- an assignment INTO a content-address field whose value is any call other
      than a stamping helper. Re-attaching a digest already held in a variable (the
      validators' ``validated["row_sha256"] = digest`` after a routed re-derivation) is
      not a call and is not a finding; computing one is.
    * DERIVE -- a comparison between a content-address field and a raw hashing
      primitive. Comparing two DECLARED digests to each other is an identity check, not
      a derivation, and is not a finding.
    """

    findings: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                field = _address_field(target)
                if field is None:
                    continue
                produced = _called_name(node.value)
                if produced is not None and produced not in _STAMP_HELPERS:
                    findings.append(
                        f"{label}:{node.lineno} stamps {field} with {produced}(): a "
                        "content address is produced by the shared stamping helper, so "
                        "no writer spells its own exclusion set"
                    )
        if isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            if not any(_address_field(side) for side in sides):
                continue
            for side in sides:
                produced = _called_name(side)
                if produced in _RAW_HASHES:
                    findings.append(
                        f"{label}:{node.lineno} re-derives a content address with "
                        f"{produced}(): a validator with a private preimage agrees with "
                        "the chain only while the body carries no excluded name"
                    )
    return findings


def routed_address_sites(source: str) -> int:
    """Return how many sites DO route through a shared helper, for anti-vacuity."""

    routed = 0
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            name = _called_name(node)
            if name in _STAMP_HELPERS | _DERIVE_HELPERS:
                routed += 1
    return routed


def test_every_chain_writer_and_validator_routes_through_the_shared_helper() -> None:
    """No production module computes a Track A content address of its own.

    Swept in ONE test that reports every offender at once, so a routing regression is
    never a single-module surprise that hides the rest of the sweep behind an early
    exit.
    """

    modules = list(_production_modules())
    assert len(modules) > 100, "the census must sweep the whole production tree"

    findings: list[str] = []
    routed = 0
    for module in modules:
        source = module.read_text(encoding="utf-8")
        findings.extend(
            unrouted_address_sites(source, label=str(module.relative_to(_CRAWLER_ROOT)))
        )
        routed += routed_address_sites(source)

    assert findings == [], "; ".join(findings)
    # The census cannot pass by there being nothing to route: the tree really does
    # compute these addresses, through the one helper, tens of times.
    assert routed >= 30, f"the shared-helper routing all but disappeared ({routed} sites)"


def test_the_routing_scanner_sees_every_spelling_it_must_forbid() -> None:
    """The scanner is not vacuous, on the EXACT seven sites PSLRE6 acct F2 found."""

    # The four writers' spelling, in both shapes they used.
    assert unrouted_address_sites('row["row_sha256"] = stable_hash(row)\n')
    assert unrouted_address_sites('row["row_sha256"] = stable_hash(body)\n')
    assert unrouted_address_sites('row["close_sha256"] = stable_hash(body)\n')
    assert unrouted_address_sites('record["record_revision"] = payload_hash(record)\n')

    # The three validators' spelling, in both directions.
    assert unrouted_address_sites('if row.get("row_sha256") != stable_hash(body):\n    pass\n')
    assert unrouted_address_sites('if stable_hash(canonical) != row["row_sha256"]:\n    pass\n')

    # An indirect stamp is a finding too: the census is about who OWNS the derivation.
    assert unrouted_address_sites('row["row_sha256"] = helpers[key](row)\n')

    # ...and the routed spellings are clean.
    assert unrouted_address_sites("row = stamp_row_digest(body)\n") == []
    assert unrouted_address_sites('record = stamp_payload_digest(body, field="payload_sha256")\n') == []
    assert (
        unrouted_address_sites('if row.get("row_sha256") != row_content_address(row):\n    pass\n')
        == []
    )
    # Re-attaching an already-verified digest is not a derivation.
    assert unrouted_address_sites('validated["row_sha256"] = digest\n') == []
    # Comparing two DECLARED digests is an identity check, not a derivation.
    assert (
        unrouted_address_sites(
            'if head.get("record_revision") != current.get("record_revision"):\n    pass\n'
        )
        == []
    )
    assert routed_address_sites("row = stamp_row_digest(body)\n") == 1


@pytest.mark.parametrize("name", sorted(_ADDRESS_FIELDS | {"chain"}))
def test_a_private_preimage_diverges_from_the_chain_on_the_same_body(name: str) -> None:
    """WHY the routing matters: the two derivations disagree, per excluded name.

    A self-derived site agrees with the chain exactly while the body carries no excluded
    name -- which is the condition nothing anywhere asserted. Naming the divergence here
    keeps the census from ever being downgraded to a formatting preference.
    """

    body: dict[str, Any] = {
        "schema_version": "menagerie.crawler.example.v1",
        "stable_id": "m_a",
        "disposition": "deferred",
    }
    assert stable_hash(body) == row_content_address(body)

    widened = {
        **body,
        name: (
            {"seq": 0, "prev": "sha256:" + "0" * 64, "sha256": "sha256:" + "1" * 64}
            if name == "chain"
            else "sha256:" + "0" * 64
        ),
    }
    assert stable_hash(widened) != row_content_address(widened), (
        f"a private preimage and the shared derivation agree on a body carrying {name}: "
        "the divergence this census prevents no longer exists, so re-derive the census"
    )
