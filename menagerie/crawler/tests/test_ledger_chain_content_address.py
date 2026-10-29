"""THE substrate tripwire: a chain address is DERIVED from the row body.

PSLRE5's CRITICAL was the shape of the address, not any one field name.
``ledger_chain.row_content_digest`` used to RETURN the first self-declared digest
field a row happened to carry -- ``close_sha256``, then ``row_sha256``, then
``record_revision``, then ``payload_sha256``.  So one extra field named
``close_sha256`` pinned the chain (and therefore every anchor, close pin, and
registry head derived from it) to an arbitrary attacker-chosen string while the
rest of the body was rewritten freely.  It was live on all twenty chain scopes.

This module holds the fix structurally, in the only spelling that cannot be
re-opened one field name over:

* **non-readability** -- adding a field of ANY name may never preserve the address
  across a change to another field (the ladder made that false for one name; a
  ban-list would make it true for the names someone remembered);
* **fused re-certification** -- every digest name EXCLUDED from the preimage is
  re-derived at the same chokepoint, so the fields consumers key on are evidence
  again without reintroducing selection;
* **transparency** -- for an honest row of every chain-bearing family, the derived
  address equals the value the deleted ladder read, so every persisted head, close
  pin, registry anchor, and paid anchor is byte-identical and no migration exists;
* **anti-substitution** -- the exclusion set's exact contents are pinned here, and
  the certified set is DERIVED from it, so a new unhashed region cannot be added
  without this file failing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler import ledger_chain as chain_module
from menagerie.crawler.identity import (
    PAYLOAD_HASH_EXCLUDED,
    PAYLOAD_HASH_TOLERATED_ADDRESS_FIELDS,
    ROW_CONTENT_ADDRESS_EXCLUDED,
    canonical_json_bytes,
    foreign_content_address_fields,
    payload_hash,
    row_content_address,
    stable_hash,
    stamp_payload_digest,
    stamp_row_digest,
    unpinned_address_fields,
)
from menagerie.crawler.ledger_chain import (
    CHAIN_ENVELOPE_FIELDS,
    FUSED_ROW_DIGEST_FIELDS,
    LedgerChainError,
    chain_envelope_value_findings,
    link_row,
    row_content_digest,
    verify_rows,
)
from menagerie.crawler.tests.durable_runtime_support import RUNTIME_DRIVERS, driven_rows
from menagerie.crawler.tools.chain_address_sweep import ladder_content_digest

pytestmark = pytest.mark.smoke

_SCOPE = "paid.sessions"

#: Field names an attacker may reach for, INCLUDING names no ban-list would ever
#: hold: the four historical ladder rungs, plausible future digest spellings, and
#: ordinary names.  The property under test is about all names at once.
_INJECTED_NAMES = (
    "close_sha256",
    "row_sha256",
    "record_revision",
    "payload_sha256",
    "envelope_sha256",
    "content_address",
    "sha256",
    "digest",
    "zzz_unlikely_name",
    "",
)


def _honest_body() -> dict[str, Any]:
    """Return one plain row body with no self-digest and no chain envelope."""

    return {
        "schema_version": "menagerie.crawler.example.v1",
        "event": "example",
        "campaign_id": "c1-mech",
        "observed": {"wall_seconds": 600.0, "sessions": 1},
        "created_at": "2026-08-18T12:00:00Z",
    }


# ---------------------------------------------------------------------------
# 1. The address is DERIVED: no field of any name can hold it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _INJECTED_NAMES)
def test_no_added_field_can_pin_the_address_across_a_body_edit(name: str) -> None:
    """Adding ``name=<address>`` may not preserve the address across a body edit.

    This is the PSLRE5 repro as a property, generalized over the field name: the
    forger's whole move was to add one field whose VALUE was the honest row's
    address and then rewrite the body.  Under a derived address the added field is
    just more body, so the address moves -- for every name, not for a ban-list.
    """

    honest = _honest_body()
    address = row_content_digest(honest)

    edited = {**honest, "observed": {"wall_seconds": 0.0, "sessions": 0}}
    hijacked = {**edited, name: address} if name else dict(edited)

    assert row_content_digest(edited) != address, "the body edit must move the address"
    assert row_content_digest(hijacked) != address, (
        f"a row carrying {name!r} = the honest address still holds that address: the "
        "chain is reading a field the row declares about itself again"
    )


def test_the_address_is_the_body_hash_over_the_pinned_exclusion_set() -> None:
    """The address is exactly ``stable_hash(row - the pinned exclusion set)``."""

    row = {
        **_honest_body(),
        "row_sha256": "sha256:whatever",
        "close_sha256": "sha256:whatever",
        "record_revision": "sha256:whatever",
        "payload_sha256": "sha256:whatever",
        "chain": {"seq": 7, "prev": "sha256:p", "sha256": "sha256:s"},
    }
    assert row_content_digest(row) == stable_hash(_honest_body())


def test_the_exclusion_set_is_pinned_and_the_certified_set_is_derived_from_it() -> None:
    """Anti-substitution: ``E`` is exactly these five names, and nothing more.

    Every member of ``E`` is an UNHASHED region on every Track A ledger at once, so
    growing it is a substrate-wide decision that must be deliberated in a reviewed
    change -- not a convenience while adding a family.  The certified set is derived
    from ``E`` rather than restated, so a new exclusion automatically comes under
    the fused re-certification instead of becoming an unchecked region.
    """

    assert set(ROW_CONTENT_ADDRESS_EXCLUDED) == {
        "chain",
        "close_sha256",
        "row_sha256",
        "record_revision",
        "payload_sha256",
    }
    assert set(FUSED_ROW_DIGEST_FIELDS) == set(ROW_CONTENT_ADDRESS_EXCLUDED) - {"chain"}
    assert set(CHAIN_ENVELOPE_FIELDS) == {"seq", "prev", "sha256"}


def test_the_second_narrower_exclusion_set_is_derived_from_the_shared_one() -> None:
    """PSLRE6 D6-3: the recordio families' exclusion is a DERIVATION, not a literal.

    8 of the 20 chain scopes -- the recordio fact ledgers -- stamp their content
    address with ``identity.payload_hash``, whose exclusion used to be a SECOND,
    narrower literal spelled inside that function. The two sets agreed only while no
    chain-bearing record family happened to carry ``chain``, ``close_sha256`` or
    ``row_sha256`` at top level, and NOTHING anywhere pinned that: this module's own
    docstring named the drift risk and no check held it.

    Now the narrow set is ``E`` minus a NAMED difference, so the halves partition ``E``
    by construction and a name added to ``E`` lands in exactly one of them. The
    difference is what the coupling refusal below is keyed on.
    """

    assert PAYLOAD_HASH_EXCLUDED | PAYLOAD_HASH_TOLERATED_ADDRESS_FIELDS == set(
        ROW_CONTENT_ADDRESS_EXCLUDED
    )
    assert not (PAYLOAD_HASH_EXCLUDED & PAYLOAD_HASH_TOLERATED_ADDRESS_FIELDS)
    # Anti-substitution: the partition's contents are pinned, so widening either half
    # is a deliberated change to this file rather than a silent digest move.
    assert set(PAYLOAD_HASH_EXCLUDED) == {"payload_sha256", "record_revision"}
    assert set(PAYLOAD_HASH_TOLERATED_ADDRESS_FIELDS) == {
        "chain",
        "close_sha256",
        "row_sha256",
    }
    # Both stamping helpers own exactly the digest names their own half excludes.
    assert set(FUSED_ROW_DIGEST_FIELDS) & set(PAYLOAD_HASH_EXCLUDED) == set(
        PAYLOAD_HASH_EXCLUDED
    )


@pytest.mark.parametrize("name", sorted(PAYLOAD_HASH_TOLERATED_ADDRESS_FIELDS))
def test_a_chained_record_carrying_a_tolerated_name_refuses_at_its_writer(name: str) -> None:
    """The coupling is enforced where the record is MINTED, not by every later read.

    A record body carrying one of the tolerated names has a ``payload_hash`` that is
    NOT its chain content address, so the chain's fused re-certification would refuse
    it -- correctly, and forever, on every read of that ledger for the rest of its
    life, with the honest producer's row bricking the scope. That is fail-closed but
    unexplained. The refusal now happens once, at the stamp, naming the coupling.
    """

    body = {"schema_version": "menagerie.crawler.example.v1", "stable_id": "m_a"}
    stamped = stamp_payload_digest(body, field="record_revision")
    assert stamped["record_revision"] == row_content_address(body)

    widened = {
        **body,
        name: (
            {"seq": 0, "prev": "sha256:" + "0" * 64, "sha256": "sha256:" + "1" * 64}
            if name == "chain"
            else "sha256:" + "0" * 64
        ),
    }
    # The divergence the refusal exists to prevent is real...
    assert payload_hash(widened) != row_content_address(widened)
    # ...and the writer refuses instead of persisting it.
    with pytest.raises(ValueError, match="content-address name"):
        stamp_payload_digest(widened, field="record_revision")
    assert unpinned_address_fields(widened) == (name,)
    assert unpinned_address_fields(body) == ()

    with pytest.raises(ValueError, match="excluded from the payload hash"):
        stamp_payload_digest(body, field="row_sha256")


def test_the_canonical_fact_writer_refuses_a_decoupled_record(tmp_path: Path) -> None:
    """The refusal is at the REAL chokepoint, in the ledger's typed vocabulary."""

    from menagerie.crawler.constants import MODEL_SCHEMA_VERSION_V3
    from menagerie.crawler.recordio import JsonlLedger
    from menagerie.crawler.schema import PayloadValidationError
    from menagerie.crawler.tests.conftest import make_model

    ledger = JsonlLedger(tmp_path / "models.jsonl", MODEL_SCHEMA_VERSION_V3)
    try:
        ledger.append(make_model())
        decoupled = {**make_model("m_second"), "close_sha256": "sha256:" + "0" * 64}
        decoupled.pop("record_seq", None)
        with pytest.raises(PayloadValidationError, match="content-address name"):
            ledger.append(decoupled)
    finally:
        ledger.close()


def test_the_shared_stamp_is_idempotent_and_refuses_an_unexcluded_field() -> None:
    """A stamped row and its bare body share one address, by construction."""

    body = _honest_body()
    stamped = stamp_row_digest(body)
    assert stamped["row_sha256"] == row_content_address(body)
    assert row_content_address(stamped) == row_content_address(body)
    assert stamp_row_digest(stamped) == stamped

    closed = stamp_row_digest(body, field="close_sha256")
    assert closed["close_sha256"] == row_content_address(body)

    with pytest.raises(ValueError, match="excluded from the content address"):
        stamp_row_digest(body, field="envelope_sha256")


# ---------------------------------------------------------------------------
# 2. Fused re-certification: a declared digest is evidence again.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", FUSED_ROW_DIGEST_FIELDS)
def test_a_declared_digest_that_is_not_the_body_derivation_refuses(field: str) -> None:
    """The address is UNCHANGED by the mutation, and the row is UNREADABLE.

    The structural inverse of the PSLRE5 repro, and both halves matter.  The
    address must NOT move (that is what makes a stored-digest shortcut impossible
    to sneak back in), and the row must nevertheless refuse (that is what keeps the
    close-head index, the predecessor spine, and every award revision reference
    reading evidence rather than a producer's string).
    """

    honest = stamp_row_digest(_honest_body())
    address = row_content_digest(honest)
    linked = link_row(honest, scope=_SCOPE, existing_rows=())

    forged = dict(linked)
    forged[field] = "sha256:" + "0" * 64
    assert row_content_digest(forged) == address, (
        "a digest-named field must not be part of the preimage, or honest heads move"
    )

    with pytest.raises(LedgerChainError, match="is not its own body content address"):
        verify_rows([forged], scope=_SCOPE)


@pytest.mark.parametrize("field", FUSED_ROW_DIGEST_FIELDS)
def test_a_declared_digest_equal_to_the_derivation_is_accepted(field: str) -> None:
    """The same field, holding the honest derivation, changes nothing and passes."""

    body = _honest_body()
    row = {**body, field: row_content_address(body)}
    linked = link_row(row, scope=_SCOPE, existing_rows=())
    state = verify_rows([linked], scope=_SCOPE)
    assert state.head_sha256 == linked["chain"]["sha256"]
    # And the address is the same one the bare body has.
    assert row_content_digest(linked) == row_content_address(body)


def test_a_null_digest_field_refuses_rather_than_reading_as_absent() -> None:
    """A JSON null cannot hold an address, so a chained row may not carry the key.

    This arm used to assert the OPPOSITE -- that a null read as "absent" and the row
    verified. That was the tolerant reading, and PSLRE6 D6-5 measured what it bought:
    three present-and-null foreign digest names are unhashed (excluded from the
    address) and unchecked (every authority tested ``is not None``), i.e. 65 bytes of
    address-invariant padding on any chained row. The strict reading costs nothing --
    no committed row in the corpus carries one -- and removes the donor.
    """

    body = {**_honest_body(), "close_sha256": None}
    with pytest.raises(LedgerChainError, match="declares a null close_sha256"):
        link_row(stamp_row_digest(body), scope=_SCOPE, existing_rows=())


def test_the_write_path_refuses_a_stamp_that_drifted_from_the_derivation() -> None:
    """A writer whose stamp disagrees refuses at its OWN append.

    The anti-drift half of the meta-fix: the writer and the chain share one
    derivation, so a family that stamped with a different exclusion set cannot
    persist rows that every later reader must refuse.
    """

    drifted = {**_honest_body(), "row_sha256": stable_hash({"a different": "preimage"})}
    with pytest.raises(LedgerChainError, match="is not its own body content address"):
        link_row(drifted, scope=_SCOPE, existing_rows=())


def test_the_write_path_refuses_a_row_that_already_carries_a_link() -> None:
    """A link is attached exactly once, by the chain, at one position."""

    linked = link_row(stamp_row_digest(_honest_body()), scope=_SCOPE, existing_rows=())
    with pytest.raises(LedgerChainError, match="already carries a chain envelope"):
        link_row(linked, scope=_SCOPE, existing_rows=())


def test_the_chain_envelope_vocabulary_is_closed() -> None:
    """An extra envelope key is covered by NEITHER the digest nor the link check.

    ``chain`` is excluded from the address and ``_verified_link`` only checks
    ``seq``/``prev``/``sha256``, so a free-form envelope key was an unhashed,
    unchecked region on every chained ledger -- a laundering channel with no honest
    producer (PSLRE5 acct observation).
    """

    linked = link_row(stamp_row_digest(_honest_body()), scope=_SCOPE, existing_rows=())
    smuggled = dict(linked)
    smuggled["chain"] = {**dict(linked["chain"]), "note": "arbitrary bytes"}
    with pytest.raises(LedgerChainError, match="outside its closed vocabulary"):
        verify_rows([smuggled], scope=_SCOPE)


#: Unbounded, unauthenticated, ADDRESS-INVARIANT ``seq`` padding that satisfies the
#: link check's numeric equality against row index ``0``: the envelope is excluded from
#: the content address, so none of it is hashed, and ``0.0 == -0.0 == False == 0`` in
#: Python (PSLRE6 D6-5, first two spellings). ``0.000...0`` grows without bound.
_ENVELOPE_SEQ_PADDING: tuple[tuple[str, Any], ...] = (
    ("a float that compares equal", 0.0),
    ("negative zero", -0.0),
    ("240 unauthenticated zeros", float("0." + "0" * 240)),
    ("a boolean, which Python counts as an int", False),
)


@pytest.mark.parametrize(("arm", "value"), _ENVELOPE_SEQ_PADDING)
def test_the_chain_envelope_values_must_conform_too(arm: str, value: Any) -> None:
    """Closing the envelope's KEYS left its VALUES an unauthenticated region.

    The link check compares the envelope's members for EQUALITY, and equality is
    NUMERIC: every value here satisfies ``chain["seq"] == 0`` for row 0 while carrying
    bytes nothing hashes and nothing checks. There is no consumer today -- which is
    precisely the state the extra-KEY half of this same channel was in before one
    appeared.

    The envelope is machinery, so its values have a SHAPE: a non-negative ``int`` row
    index and two canonical digests. Asserting the shape closes the channel for every
    row, whether or not an equality check also applies to it.
    """

    honest = dict(link_row(stamp_row_digest(_honest_body()), scope=_SCOPE, existing_rows=()))
    assert chain_envelope_value_findings(dict(honest["chain"])) == ()

    padded = dict(honest)
    padded["chain"] = {**dict(honest["chain"]), "seq": value}
    # The padding IS address-invariant and DOES satisfy the equality check...
    assert row_content_digest(padded) == row_content_digest(honest), arm
    assert padded["chain"]["seq"] == honest["chain"]["seq"], arm
    # ...and the shape check refuses it anyway.
    assert chain_envelope_value_findings(dict(padded["chain"])), arm
    with pytest.raises(LedgerChainError, match="values are not machinery-shaped"):
        verify_rows([padded], scope=_SCOPE)


def test_the_envelope_value_check_is_not_vacuous_on_the_shapes_it_admits() -> None:
    """The value check accepts exactly the machinery shapes and nothing adjacent."""

    digest = "sha256:" + "0" * 64
    assert chain_envelope_value_findings({"seq": 0, "prev": digest, "sha256": digest}) == ()
    assert chain_envelope_value_findings({"seq": 7, "prev": digest, "sha256": digest}) == ()
    assert chain_envelope_value_findings({"seq": -1, "prev": digest, "sha256": digest})
    assert chain_envelope_value_findings({"seq": 0, "prev": digest.upper(), "sha256": digest})
    assert chain_envelope_value_findings({"seq": 0, "prev": digest, "sha256": None})
    assert chain_envelope_value_findings({"seq": 0, "prev": digest, "sha256": digest[:-1]})


def test_a_null_declared_digest_is_a_refusal_not_an_absence() -> None:
    """PSLRE6 D6-5: ``"close_sha256": null`` was 65 bytes nothing hashed or checked.

    A null used to read as "absent": the fused re-certification skipped it and
    ``foreign_content_address_fields`` tested ``is not None``, so all three foreign
    digest names could be present-and-null on any chained row, unhashed (they are
    excluded from the address) and unchecked (no authority looked at them). A row
    DECLARES its own content address or OMITS the key.
    """

    body = {**_honest_body(), "close_sha256": None}
    assert row_content_address(body) == row_content_address(_honest_body())
    with pytest.raises(LedgerChainError, match="declares a null close_sha256"):
        link_row(stamp_row_digest(body), scope=_SCOPE, existing_rows=())

    padded = {
        **_honest_body(),
        "close_sha256": None,
        "record_revision": None,
        "payload_sha256": None,
    }
    assert foreign_content_address_fields(padded, own="row_sha256") == (
        "close_sha256",
        "payload_sha256",
        "record_revision",
    )
    assert foreign_content_address_fields(_honest_body(), own="row_sha256") == ()


def test_a_duplicate_object_key_is_hard_corruption() -> None:
    """PSLRE6 D6-5: repeated keys decode to the same row and add unbounded bytes.

    ``json.loads`` keeps the LAST occurrence, so eight extra copies of one field
    decode to a byte-identical row with a byte-identical address and head -- 192
    unauthenticated bytes on the line, on a ledger whose only extent evidence may be
    a byte count. The canonical writer emits no duplicate, so this is corruption.
    """

    honest = link_row(stamp_row_digest(_honest_body()), scope=_SCOPE, existing_rows=())
    line = canonical_json_bytes(honest)
    padded = line[:-1] + b',"campaign_id":"c1-mech"' * 8 + b"}"
    assert json.loads(padded) == json.loads(line), "the padding must be address-invariant"

    path = Path("ledger.jsonl")
    with pytest.raises(LedgerChainError, match="duplicate object key"):
        chain_module.decode_payload(padded + b"\n", path)
    assert chain_module.decode_payload(line + b"\n", path) == (dict(honest),)


# ---------------------------------------------------------------------------
# 3. Transparency: every honest family's own convention is reproduced exactly.
# ---------------------------------------------------------------------------


def _ladder_head(row: Mapping[str, Any], *, scope: str, field: str) -> str:
    """Return the head the DELETED ladder produced for one honest row.

    The ladder read the row's own declared digest and extended the chain with it.
    Reconstructing that rule here is the migration proof: for an honest row of
    every family the two answers agree, so no persisted head moves.
    """

    return chain_module._extend(  # noqa: SLF001 -- reconstructing the deleted rule
        scope, chain_module._genesis(scope), 0, str(row[field])  # noqa: SLF001
    )


#: One honest row per chain-bearing family, stamped with THAT family's own
#: historical rule (never the shared helper), as ``(scope, digest field, row)``.
def _family_rows() -> tuple[tuple[str, str, dict[str, Any]], ...]:
    """Return one honestly stamped row for each chain-bearing family shape."""

    families: list[tuple[str, str, dict[str, Any]]] = []

    money_body = {
        "schema_version": "menagerie.crawler.paid-session-settlement.v4",
        "event": "session_settled",
        "spend_id": "paid-spend-transparency",
        "observed": {"wall_seconds": 600.0},
    }
    families.append(
        ("paid.sessions", "row_sha256", {**money_body, "row_sha256": stable_hash(money_body)})
    )

    close_body = {
        "schema_version": "menagerie.crawler.leg-close.v2",
        "campaign_id": "c1-mech",
        "close_kind": "leg",
    }
    families.append(
        ("meta.closes", "close_sha256", {**close_body, "close_sha256": stable_hash(close_body)})
    )

    leg_body = {
        "schema_version": "menagerie.crawler.leg-event.v1",
        "event": "leg_plan",
        "admitted_ids": ["m_a", "m_b"],
    }
    families.append(
        ("meta.leg-events", "row_sha256", {**leg_body, "row_sha256": stable_hash(leg_body)})
    )

    registry_body = {
        "schema_version": "menagerie.crawler.rung-registration.v1",
        "snapshot_id": "rung-0001",
        "member_count": 12,
    }
    families.append(
        ("meta.registry", "row_sha256", {**registry_body, "row_sha256": stable_hash(registry_body)})
    )

    valve_body = {"schema_version": "menagerie.crawler.valve.v1", "stable_id": "m_a"}
    families.append(
        ("valve.capacity", "row_sha256", {**valve_body, "row_sha256": stable_hash(valve_body)})
    )

    # recordio records: BOTH lower rungs are identity.payload_hash of the record.
    model_record = {"stable_id": "m_a", "record_seq": 3, "kind": "model"}
    families.append(
        (
            "models.shard",
            "record_revision",
            {**model_record, "record_revision": payload_hash(model_record)},
        )
    )
    gate_record = {"stable_id": "m_a", "ledger_seq": 4, "kind": "gate"}
    families.append(
        (
            "gates.shard",
            "payload_sha256",
            {**gate_record, "payload_sha256": payload_hash(gate_record)},
        )
    )

    # A historic row carrying NO rung at all fell through to the body hash already.
    families.append(("meta.closes", "", {"schema_version": "legacy", "note": "no rung"}))
    return tuple(families)


@pytest.mark.parametrize(("scope", "field", "row"), _family_rows())
def test_every_honest_family_head_is_byte_identical_to_the_ladders(
    scope: str, field: str, row: dict[str, Any]
) -> None:
    """The derived address equals each family's own declared content address.

    Fable's algebra and Opus's 12782-row corpus census both concluded this: every
    family stamps its rung as the hash of its own body minus its own digest, under
    ONE canonicalization, and no honest row carries a second rung.  So the union
    exclusion is a no-op per family and global derivation is a transparent change.
    """

    derived = verify_rows([row], scope=scope).head_sha256
    if field:
        assert row_content_digest(row) == row[field], (
            f"{scope}: the family's own stamp is no longer the body derivation -- this "
            "is a real divergence, and it must be adjudicated as a decision rather "
            "than absorbed by relaxing the fused check"
        )
        assert derived == _ladder_head(row, scope=scope, field=field)
    else:
        assert derived == chain_module._extend(  # noqa: SLF001
            scope, chain_module._genesis(scope), 0, stable_hash(row)  # noqa: SLF001
        )


#: The driven writers whose real rows declare NO content-address field at all, and are
#: therefore addressed by the pure body hash (the historical "no rung" shape the
#: deleted ladder fell through to). DISCLOSED rather than silently skipped, and
#: asserted to declare no fused field, so one of them cannot quietly start declaring a
#: digest and land outside the transparency claim:
#:
#: * the paid ledger's external extent ANCHOR is an atomically replaced pin carrying
#:   ``anchor_sha256``, which is not a chain content-address name;
#: * the operational grant/void ledger's rows carry ``grant_id`` identity and no
#:   self-digest, so their address is their body and the two rules agree trivially.
_UNDECLARED_ADDRESS_WRITERS = frozenset(
    {
        ("paid_session_durability.py", "_write_anchor"),
        ("checkpoint.py", "append_canonical_requeue_grant"),
        ("checkpoint.py", "append_canonical_requeue_void"),
    }
)


@pytest.mark.parametrize(
    ("module_name", "function"), sorted(RUNTIME_DRIVERS), ids=lambda value: str(value)
)
def test_every_family_s_REAL_writer_output_is_addressed_by_its_body(
    tmp_path: Path, module_name: str, function: str
) -> None:
    """Transparency, on rows a PRODUCTION WRITER actually minted (PSLRE6 D6-6).

    The two transparency arms above assert the right property about rows this test
    file builds by hand: they reconstruct each family's historical stamping rule and
    check that the shared derivation reproduces it. That is the algebra, and it is
    worth having -- but no writer is in the loop, so a writer that drifts from the
    rule its own family's hand-built row encodes moves no head in those arms.

    This drives EVERY named ledger writer's real append (the same drivers the runtime
    durability arm uses) and asserts on the bytes that landed: for every fused digest
    field a persisted row declares, the value equals the body derivation, so the
    deleted ladder's answer and the derived address agree row by row -- and therefore
    head for head, on real writer output, per family.
    """

    ledger = RUNTIME_DRIVERS[(module_name, function)](tmp_path)
    rows = list(driven_rows(ledger))
    assert rows, f"{module_name}:{function} persisted no row to check"

    checked = 0
    for index, row in enumerate(rows):
        derived = row_content_digest(row)
        for field in FUSED_ROW_DIGEST_FIELDS:
            declared = row.get(field)
            if declared is None:
                continue
            checked += 1
            assert declared == derived, (
                f"{module_name}:{function} row {index} declares a {field} that is not "
                "its body derivation: the writer and the chain disagree about this "
                "family's content address, on real output"
            )
        # The ladder read the FIRST declared digest; the chain now derives one. On an
        # honest row the two are the same string, which is what makes every persisted
        # head byte-identical rather than migrated.
        assert ladder_content_digest(row) == derived, (
            f"{module_name}:{function} row {index}: the deleted ladder's address and "
            "the body derivation differ, so this family's persisted heads would move"
        )

    if (module_name, function) in _UNDECLARED_ADDRESS_WRITERS:
        assert checked == 0, (
            f"{module_name}:{function} now declares a chain content-address field; it "
            "is no longer body-addressed and must be moved out of the disclosed "
            "undeclared-address set so its stamp is covered here"
        )
    else:
        assert checked, (
            f"{module_name}:{function} persisted no declared content address: this "
            "family's real output is not covered by the transparency claim at all"
        )


def test_every_committed_fixture_row_keeps_its_address() -> None:
    """The committed corpus fixtures are honest rows, and none of them moves."""

    checked = 0
    for path in sorted(Path(__file__).resolve().parent.rglob("*.jsonl")):
        for line in path.read_bytes().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            declared = [
                name
                for name in FUSED_ROW_DIGEST_FIELDS
                if isinstance(row.get(name), str) and row[name]
            ]
            if not declared:
                continue
            checked += 1
            assert row_content_digest(row) == row[declared[0]], (
                f"{path.name}: committed fixture row's declared {declared[0]} is not its "
                "body derivation"
            )
    assert checked >= 23, "the committed fixture sweep lost its corpus"


def test_the_canonicalization_is_the_only_one_in_the_tree() -> None:
    """The address is canonical JSON: key order and spacing cannot move it."""

    body = _honest_body()
    permuted = dict(reversed(list(body.items())))
    assert row_content_address(permuted) == row_content_address(body)
    assert canonical_json_bytes(permuted) == canonical_json_bytes(body)
