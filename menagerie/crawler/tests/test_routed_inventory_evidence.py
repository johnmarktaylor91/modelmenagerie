"""The routed-environment inventory is frozen, bindable, re-attestable evidence.

Pilot model m1359 (Highway Networks MLP) failed ``failed:fidelity`` after three
bounded rounds although six of seven material checks matched and the seventh
was declared nonmaterial: its "R1_LIBRARY unavailable, therefore R2_VENDOR is
highest applicable" claim rested on the routed environment's inventory, which
existed only as an envelope disclosure the author could QUOTE as proposal
prose. The checker verifies against frozen bytes, so every round's
``required_repairs`` demanded a frozen inventory no stage could produce -- an
unsatisfiable repair, the same defect class as the pre-disclosure blind R1 pin.

These tests pin the closing machinery end to end:

- the frozen document is deterministic, derived through the SAME disclosure
  the envelope publishes, and absent (never empty) when the machine holds no
  inventory;
- the manifest row publishes real CAS bytes at the canonical digest address;
- the round trip works against production validators: an author excerpt
  quoting the frozen inventory verbatim validates through
  ``validate_evidence`` for ``source_resolution.rung``, and a tampered
  excerpt is refused -- the exact mechanical re-attestation the checker's
  read roots expose;
- the reserved ``source_id`` cannot be claimed by an authored descriptor;
- the frozen document and a production author envelope agree on
  ``legal_distributions`` for the same packages (lockstep against silent
  derivation drift).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.environment_inventory import (
    ROUTED_INVENTORY_RECORD_VERSION,
    ROUTED_INVENTORY_SOURCE_ID,
    ROUTED_INVENTORY_SOURCE_KIND,
    render_routed_inventory_document,
    routed_inventory_source_row,
)
from menagerie.crawler.evidence import EvidenceValidationError, validate_evidence
from menagerie.crawler.fetcher import cas_path
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.source_broker import SourceBrokerError, _validated_descriptor
from menagerie.crawler.tests.test_routed_environment_disclosure import (
    _envelope,
    _real_inventory,
)

_INTENT = "core"


def _document() -> bytes:
    """Render the frozen inventory for the committed release lock."""

    content = render_routed_inventory_document(
        intent=_INTENT,
        packages=_real_inventory(),
        resolved_export_sha256="sha256:" + "e" * 64,
    )
    assert content is not None
    return content


def _row(tmp_path: Path) -> dict[str, Any]:
    """Publish the inventory into a sandbox CAS and return its manifest row."""

    row = routed_inventory_source_row(
        intent=_INTENT,
        packages=_real_inventory(),
        cas_root=tmp_path / "source-cas",
        resolved_export_sha256="sha256:" + "e" * 64,
    )
    assert row is not None
    return dict(row)


# -- the frozen document --------------------------------------------------------


def test_the_document_is_deterministic_and_json_parseable() -> None:
    """Unchanged environment, unchanged bytes: repair generations rebind cleanly."""

    first, second = _document(), _document()
    assert first == second
    parsed = json.loads(first.decode("utf-8"))
    assert parsed["record_version"] == ROUTED_INVENTORY_RECORD_VERSION
    assert parsed["intent"] == _INTENT
    assert parsed["inventory_disclosed"] is True
    assert parsed["resolved_export_sha256"] == "sha256:" + "e" * 64
    assert parsed["packages"], "the committed release lock is not empty"


def test_the_document_withholds_the_machine_derived_artifact_digests() -> None:
    """The frozen evidence discloses exactly what the envelope does -- no more."""

    serialized = _document().decode("utf-8")
    for entry in _real_inventory():
        assert str(entry["sha256"]) not in serialized


def test_no_inventory_means_no_document_never_an_empty_one() -> None:
    """An absent inventory must not freeze as a positive empty-inventory claim."""

    assert render_routed_inventory_document(intent=_INTENT, packages=()) is None
    assert render_routed_inventory_document(intent="", packages=_real_inventory()) is None


def test_the_document_agrees_with_the_envelope_disclosure(tmp_path: Path) -> None:
    """Frozen evidence and the author's capability sheet can never disagree.

    Both derive through ``inventory_disclosure``; this pins the derived
    ``legal_distributions`` lists byte-equal so a drift in either derivation
    is caught here rather than shipping as a checker-visible contradiction.
    """

    envelope = _envelope(
        tmp_path,
        routed_environment_intent=_INTENT,
        routed_environment_packages=_real_inventory(),
    )
    parsed = json.loads(_document().decode("utf-8"))
    assert parsed["legal_distributions"] == envelope["routed_environment"]["legal_distributions"]
    assert parsed["packages"] == envelope["routed_environment"]["packages"]


# -- the manifest row and its CAS bytes ------------------------------------------


def test_the_row_publishes_exact_bytes_at_the_canonical_cas_address(
    tmp_path: Path,
) -> None:
    """The row is a real, hash-covered source: bytes live at the digest address."""

    row = _row(tmp_path)
    assert row["source_id"] == ROUTED_INVENTORY_SOURCE_ID
    assert row["source_kind"] == ROUTED_INVENTORY_SOURCE_KIND
    assert row["retrieval_status"] == "machine-derived"
    assert row["routed_environment_intent"] == _INTENT
    digest = str(row["content_sha256"])
    # Self-describing URN over the frozen bytes, like discovery evidence.
    assert row["url"] == (
        f"urn:menagerie:routed-environment-inventory:{digest.removeprefix('sha256:')}"
    )
    assert isinstance(row["retrieved_at"], str) and row["retrieved_at"].endswith("Z")
    published = cas_path(tmp_path / "source-cas", digest)
    assert Path(row["cas_path"]) == published
    content = published.read_bytes()
    assert hash_bytes(content) == digest == str(row["revision"])
    assert len(content) == row["fetched_bytes_len"]
    assert content == _document()


def test_the_persisted_row_is_stable_while_the_inventory_is_unchanged(
    tmp_path: Path,
) -> None:
    """Repair generations and restarts rebind the identical row (and identity).

    ``retrieved_at`` is a real clock fact, so without persistence every
    rebuild would move the manifest identity and invalidate every cached
    author artifact on every driver restart. The sidecar freezes the minted
    row exactly like the broker pack freezes its rows; a changed inventory
    honestly re-mints.
    """

    cache = tmp_path / "routed-inventory-row.json"
    clocks = iter(["2026-08-09T00:00:00Z", "2026-08-09T01:00:00Z", "2026-08-09T02:00:00Z"])

    def mint(packages: Any) -> Any:
        return routed_inventory_source_row(
            intent=_INTENT,
            packages=packages,
            cas_root=tmp_path / "source-cas",
            resolved_export_sha256="sha256:" + "e" * 64,
            row_cache_path=cache,
            clock=lambda: next(clocks),
        )

    first = mint(_real_inventory())
    second = mint(_real_inventory())
    assert first == second
    assert second is not None and second["retrieved_at"] == "2026-08-09T00:00:00Z"

    # A changed environment moves the digest and honestly re-mints the row.
    changed = mint([{"name": "numpy", "version": "2.0.0"}])
    assert changed is not None
    assert changed["content_sha256"] != first["content_sha256"]
    assert changed["retrieved_at"] == "2026-08-09T01:00:00Z"


def test_an_unrouted_or_unlocked_target_produces_no_row(tmp_path: Path) -> None:
    """The manifest stays exactly as before when the machine holds no inventory."""

    assert (
        routed_inventory_source_row(
            intent=None, packages=_real_inventory(), cas_root=tmp_path / "source-cas"
        )
        is None
    )
    assert (
        routed_inventory_source_row(intent=_INTENT, packages=(), cas_root=tmp_path / "source-cas")
        is None
    )
    assert not (tmp_path / "source-cas").exists()


# -- the evidence round trip: author cites, checker re-attests -------------------


def _excerpt_text() -> str:
    """Return a verbatim run of the frozen document naming a real distribution."""

    document = _document().decode("utf-8")
    start = document.index('"legal_distributions"')
    end = document.index("]", start) + 1
    text = document[start:end]
    assert '"torch"' in text, "the committed release lock provides torch"
    return text


def _evidence(text: str) -> dict[str, Any]:
    """Wrap one inventory excerpt as a complete model evidence block."""

    return {
        "excerpts": [
            {
                "evidence_id": "ex-env-inventory",
                "source_id": ROUTED_INVENTORY_SOURCE_ID,
                "locator": "routed-environment-inventory#legal_distributions",
                "text": text,
                "text_sha256": hash_bytes(text.encode("utf-8")),
                "supports": ["source_resolution.rung"],
                "family_level": False,
            }
        ],
        "coverage": {
            "missing_support": [],
            "all_agent_fields_have_support": True,
        },
    }


def test_an_author_excerpt_of_the_frozen_inventory_re_attests(
    tmp_path: Path,
) -> None:
    """The full round trip: frozen row -> literal excerpt -> byte re-verification.

    ``validate_evidence`` with a governing ``cas_root`` is exactly the
    mechanical re-attestation surface the checker's read roots expose: the
    row's recorded path is ignored, the bytes are re-read digest-derived and
    re-hashed, and the excerpt must exist verbatim in them. This is the claim
    m1359 could never make checkable.
    """

    row = _row(tmp_path)
    report = validate_evidence(
        _evidence(_excerpt_text()),
        {"sources": [row]},
        required_claims=["source_resolution.rung"],
        cas_root=tmp_path / "source-cas",
    )
    assert "source_resolution.rung" in report.supported_claims


def test_a_tampered_inventory_excerpt_is_refused(tmp_path: Path) -> None:
    """Quoting a distribution the frozen inventory does not name still fails."""

    row = _row(tmp_path)
    forged = _excerpt_text().replace('"torch"', '"allennlp"')
    with pytest.raises(EvidenceValidationError, match="not verbatim"):
        validate_evidence(
            _evidence(forged),
            {"sources": [row]},
            required_claims=["source_resolution.rung"],
            cas_root=tmp_path / "source-cas",
        )


def test_swapped_cas_bytes_are_refused_by_the_digest(tmp_path: Path) -> None:
    """A CAS object that no longer hashes to the row's digest cannot ground anything."""

    row = _row(tmp_path)
    Path(row["cas_path"]).write_bytes(b'{"legal_distributions": ["allennlp"]}\n')
    with pytest.raises(EvidenceValidationError, match="does not match"):
        validate_evidence(
            _evidence(_excerpt_text()),
            {"sources": [row]},
            required_claims=["source_resolution.rung"],
            cas_root=tmp_path / "source-cas",
        )


# -- the reserved identifier ------------------------------------------------------


def test_an_authored_descriptor_cannot_claim_the_reserved_source_id() -> None:
    """Machine custody stays machine custody: the broker refuses the collision."""

    descriptor: Mapping[str, Any] = {
        "source_id": ROUTED_INVENTORY_SOURCE_ID,
        "kind": "raw-url",
        "url": "https://example.com/fake-inventory.json",
        "requested_role": "documentation",
        "basis": "impersonation attempt",
    }
    with pytest.raises(SourceBrokerError, match="reserved for a machine-materialized"):
        _validated_descriptor(descriptor, 1)


def test_an_ordinary_descriptor_still_validates() -> None:
    """The refusal is exactly the reserved identifier, nothing wider."""

    descriptor: Mapping[str, Any] = {
        "source_id": "doc-vendor-readme",
        "kind": "raw-url",
        "url": "https://example.com/readme.md",
        "requested_role": "documentation",
        "basis": "vendor readme",
    }
    validated = _validated_descriptor(descriptor, 1)
    assert validated["source_id"] == "doc-vendor-readme"


# -- the author's declared custody echo validates against the closed schema ------


def _source_row_validator() -> Any:
    """Build a validator for the closed declared-source row definition."""

    from jsonschema import Draft202012Validator, FormatChecker
    from referencing import Registry, Resource

    schema_dir = Path(__file__).resolve().parents[1] / "schemas"
    common = json.loads((schema_dir / "model-common.schema.json").read_text("utf-8"))
    resource = Resource.from_contents(common)
    registry = Registry().with_resource(common["$id"], resource)
    return Draft202012Validator(
        {"$ref": f"{common['$id']}#/$defs/source"},
        registry=registry,
        format_checker=FormatChecker(),
    )


def _declared_inventory_row(tmp_path: Path) -> dict[str, Any]:
    """Echo the manifest row exactly as the stage-2 conventions instruct."""

    row = _row(tmp_path)
    return {
        "source_id": str(row["source_id"]),
        "role": "documentation",
        "kind": "environment-inventory",
        "url": str(row["url"]),
        "revision_kind": "routed-environment-inventory-sha256",
        "revision": str(row["revision"]),
        "locator": "routed-environment-inventory-v1#/legal_distributions",
        "content_sha256": str(row["content_sha256"]),
        "byte_count": int(row["fetched_bytes_len"]),
        "media_type": str(row["media_type"]),
        "retrieved_at": str(row["retrieved_at"]),
        "fetch_recipe": "driver-materialized routed-environment inventory",
        "mirror_class": "private-machine-evidence",
        "mirror_digest": str(row["content_sha256"]),
    }


def test_the_declared_inventory_row_passes_the_closed_source_schema(
    tmp_path: Path,
) -> None:
    """The custody-echo shape the prompt instructs is actually schema-legal.

    The declared-source definition is closed (``additionalProperties: false``,
    enum ``kind``/``role``, required ``retrieved_at``), so an inexpressible
    machine row would be a NEW unsatisfiable rule: every proposal must mirror
    every manifest row exactly, and a row that cannot be declared kills every
    model on its route. This test is the tripwire against that.
    """

    _source_row_validator().validate(_declared_inventory_row(tmp_path))


def test_the_schema_still_polices_url_shapes_per_kind(tmp_path: Path) -> None:
    """The new kind branch narrows; it never loosens the http kinds."""

    from jsonschema.exceptions import ValidationError

    validator = _source_row_validator()
    urn_on_web_page = {**_declared_inventory_row(tmp_path), "kind": "web-page"}
    with pytest.raises(ValidationError):
        validator.validate(urn_on_web_page)
    http_on_inventory = {
        **_declared_inventory_row(tmp_path),
        "url": "https://example.com/inventory.json",
    }
    with pytest.raises(ValidationError):
        validator.validate(http_on_inventory)
