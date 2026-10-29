"""One appended line may not turn the Track A integrity spine into an UNTYPED crash.

The module's declared contract is that a COMPLETE malformed line is hard corruption and
refuses with the typed ``LedgerChainError`` (torn-tail recovery is the only tolerant
path, and only for a non-newline-terminated final line). PSLRE6 D6-2 measured the
contract and found a gap the globalized chain-address fix widened: ``json.loads`` accepts
more than the one canonicalization can serialize, so a line the strict decoder ACCEPTED
now reached ``canonical_json_bytes`` inside the address derivation and died there.

The blast radius is the whole spine, because the exception is not the typed one. Before:

    decode_payload           : ACCEPTS the line
    verified_chain           : UNTYPED ValueError: Out of range float values ...
    verify_records_integrity : UNTYPED ValueError: Out of range float values ...

Nothing above ``verify_records_integrity`` catches a bare ``ValueError`` or a
``UnicodeEncodeError``, so one appended line takes out every close and staging decision
with a traceback instead of a refusal a caller can act on. The old preference ladder
never hit this: it RETURNED a declared digest field and never hashed the body at all.

The fix is not a list of the spellings that break: ``json.loads`` accepts ``NaN``,
``Infinity``, ``-Infinity``, any overflowing exponent (``1e400`` -> ``inf``, with no
``NaN`` token anywhere in the line), and ``\\udXXX`` lone surrogates. The decoder instead
PROVES the row is addressable, by round-tripping it through the one canonicalization.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.close_reads import reset_incremental_reads
from menagerie.crawler.ledger_chain import (
    LedgerChainError,
    decode_ledger,
    decode_payload,
    verified_chain,
    verify_records_integrity,
)

pytestmark = pytest.mark.smoke

_LEG_EVENT = '{"schema_version":"menagerie.crawler.leg-event.v1","event":"leg_plan"'

#: Complete lines ``json.loads`` accepts and canonical JSON cannot serialize. The fourth
#: carries no non-finite TOKEN at all -- it overflows to ``inf`` during parsing -- which
#: is why the refusal is a serializability proof rather than a token ban-list.
_POISON: tuple[tuple[str, bytes], ...] = (
    ("NaN literal", f'{_LEG_EVENT},"x":NaN}}\n'.encode()),
    ("Infinity literal", f'{_LEG_EVENT},"x":Infinity}}\n'.encode()),
    ("negative Infinity literal", f'{_LEG_EVENT},"x":-Infinity}}\n'.encode()),
    ("overflowing exponent, no literal", f'{_LEG_EVENT},"x":1e400}}\n'.encode()),
    ("lone surrogate escape", f'{_LEG_EVENT},"x":"\\ud800"}}\n'.encode()),
)


@pytest.mark.parametrize(("arm", "payload"), _POISON)
def test_an_unaddressable_line_refuses_typed_at_every_authority(
    tmp_path: Path, arm: str, payload: bytes
) -> None:
    """The strict decoder, the chain, and the whole spine all refuse TYPED."""

    records = tmp_path / "records"
    ledger = records / "intake" / "leg-events.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    (records / "models").mkdir(parents=True, exist_ok=True)
    ledger.write_bytes(payload)

    # 1. The strict decoder -- the module's declared hard-corruption authority.
    reset_incremental_reads()
    with pytest.raises(LedgerChainError, match="not canonically serializable"):
        decode_payload(payload, ledger)

    # 2. The streaming single-row decoder settles through the same judgement, so the
    #    whole-bytes and incremental readers cannot drift in what they accept.
    reset_incremental_reads()
    with pytest.raises(LedgerChainError, match="not canonically serializable"):
        decode_ledger(ledger)

    # 3. The chain.
    reset_incremental_reads()
    with pytest.raises(LedgerChainError):
        verified_chain(ledger, scope="meta.leg-events")

    # 4. The whole Track A integrity spine -- the arm that used to raise an UNTYPED
    #    exception straight through every close and staging decision.
    reset_incremental_reads()
    with pytest.raises(LedgerChainError):
        verify_records_integrity(records)


def test_an_honest_line_still_decodes(tmp_path: Path) -> None:
    """The refusal is not a blanket one: an addressable row decodes unchanged."""

    ledger = tmp_path / "leg-events.jsonl"
    honest = f'{_LEG_EVENT},"x":1.5}}\n'.encode()
    ledger.write_bytes(honest)
    reset_incremental_reads()
    rows = decode_payload(honest, ledger)
    assert rows == (
        {
            "schema_version": "menagerie.crawler.leg-event.v1",
            "event": "leg_plan",
            "x": 1.5,
        },
    )
    reset_incremental_reads()
    assert decode_ledger(ledger) == rows


def test_a_refusal_names_the_line_it_refused(tmp_path: Path) -> None:
    """The typed refusal carries the path and line, as every sibling refusal does."""

    ledger = tmp_path / "leg-events.jsonl"
    payload = f'{_LEG_EVENT},"x":1.0}}\n'.encode() + f'{_LEG_EVENT},"x":NaN}}\n'.encode()
    ledger.write_bytes(payload)
    reset_incremental_reads()
    with pytest.raises(LedgerChainError) as caught:
        decode_payload(payload, ledger)
    assert f"{ledger}:2" in str(caught.value)
