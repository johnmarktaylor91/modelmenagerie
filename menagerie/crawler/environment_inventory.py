"""Freeze the routed environment's inventory as citable machine evidence.

The author envelope has disclosed the routed intent's exact package inventory
since the rung-3 census (``package_namespace.inventory_disclosure``), and the
author prompt tells the author to adjudicate R1_LIBRARY availability against
it. What did NOT exist was a checker-facing form of that same fact: the
disclosure lived only on the dispatch envelope, so an author asserting
"R1_LIBRARY is unavailable, therefore R2_VENDOR is highest applicable" could
only QUOTE the inventory as proposal prose. The fidelity checker verifies
against frozen bytes -- an unfrozen negative proof is exactly what it must
refuse -- so the claim was structurally unverifiable no matter how faithfully
the author transcribed it. Pilot model m1359 (Highway Networks MLP) burned all
three bounded fidelity rounds on precisely this: six of seven material checks
matched, the seventh was declared nonmaterial, and every round's
``required_repairs`` still demanded a frozen routed-environment inventory no
stage could produce. An unsatisfiable repair is not a check.

This module closes the gap on the evidence side, mirroring the
``discovery-evidence-v1`` precedent: the driver freezes the SAME disclosure the
envelope carries into the model's source CAS as a machine-materialized source
row (``ROUTED_INVENTORY_SOURCE_ID``). The author can bind literal excerpts to
it like any other frozen source, and the checker re-attests the negative proof
from exact bytes under the manifest's ordinary hash discipline. The row is
deterministic -- no clock fields -- so an unchanged environment yields an
unchanged manifest identity across repair generations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from menagerie.crawler.fetcher import publish_cas_object
from menagerie.crawler.identity import hash_bytes, utc_now
from menagerie.crawler.models import JsonObject
from menagerie.crawler.package_namespace import inventory_disclosure

#: Reserved manifest identifier for the machine-materialized inventory source.
#: The broker refuses author-named descriptors that claim it
#: (``source_broker._validated_descriptor``), so the row can only ever be
#: machine custody.
ROUTED_INVENTORY_SOURCE_ID = "env-routed-inventory"

#: Closed record identity for the frozen document and its manifest row kind.
ROUTED_INVENTORY_RECORD_VERSION = "menagerie.crawler.routed-environment-inventory.v1"

#: Manifest row ``source_kind`` marker, parallel to ``discovery-evidence-v1``.
ROUTED_INVENTORY_SOURCE_KIND = "routed-environment-inventory-v1"


def render_routed_inventory_document(
    *,
    intent: str,
    packages: Sequence[Mapping[str, Any]],
    resolved_export_sha256: Optional[str] = None,
) -> Optional[bytes]:
    """Render the routed inventory as deterministic, excerpt-friendly bytes.

    The document body is derived through the SAME disclosure the author
    envelope publishes (:func:`inventory_disclosure`), and
    ``legal_distributions`` is derived from the disclosed rows by the same
    rule ``author_dispatch._assemble_author_envelope`` applies, so the frozen
    evidence and the envelope's capability sheet can never disagree about what
    the environment installs. The serialization is pretty-printed (sorted
    keys, two-space indent, trailing newline) rather than canonical-compact
    because authors quote verbatim byte runs out of it; determinism is what
    matters, and ``json.dumps`` with fixed options is deterministic.

    Parameters
    ----------
    intent:
        Routed environment intent name.
    packages:
        Exact resolved-export package rows for that intent
        (``driver_admission._routed_environment_packages`` output).
    resolved_export_sha256:
        Optional digest of the exact resolved-export lock artifact the rows
        came from, binding the document to the solve that produced it.

    Returns
    -------
    bytes | None
        Frozen document bytes, or ``None`` when there is nothing to disclose
        (no intent or an empty inventory). An absent document is the honest
        "the machine holds no inventory here" state and must never be frozen
        as an empty one -- an empty list would read as a positive claim that
        the environment installs nothing.
    """

    if not intent:
        return None
    disclosed = [dict(row) for row in inventory_disclosure(packages)]
    if not disclosed:
        return None
    legal_distributions = sorted(
        {
            str(distribution)
            for row in disclosed
            for distribution in row.get("provides", [])
            if isinstance(distribution, str) and distribution
        }
    )
    document: JsonObject = {
        "record_version": ROUTED_INVENTORY_RECORD_VERSION,
        "intent": intent,
        "inventory_disclosed": True,
        "legal_distributions": legal_distributions,
        "packages": disclosed,
        "resolved_export_sha256": resolved_export_sha256,
    }
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )


def routed_inventory_source_row(
    *,
    intent: Optional[str],
    packages: Sequence[Mapping[str, Any]],
    cas_root: Union[str, Path],
    resolved_export_sha256: Optional[str] = None,
    row_cache_path: Union[str, Path, None] = None,
    clock: Callable[[], str] = utc_now,
) -> Optional[JsonObject]:
    """Publish the routed inventory into the CAS and return its manifest row.

    The row rides inside the frozen stage-1 ``sources`` list, BEFORE the
    manifest identity is computed, so it participates in
    ``manifest_sha256 == stable_hash(sources)`` like every fetched source, is
    disclosed to the author on the envelope's ``source_manifest``, is staged
    into private custody by ``_stage_author_result``, and reaches the checker
    item verbatim.

    ``retrieved_at`` is a REAL clock fact (the instant this model's inventory
    was first frozen), because the declared-source schema requires the author
    to echo a timestamp and an invented one would be a false provenance claim.
    To keep the manifest identity stable across repair generations and driver
    restarts -- broker rows achieve this by freezing their pack once in
    ``source-targets.json`` -- the minted row is persisted at
    ``row_cache_path`` and reused verbatim for as long as the inventory bytes
    are unchanged. An environment change moves the digest, which honestly
    re-mints the row (and the manifest identity with it).

    Parameters
    ----------
    intent:
        Routed environment intent for this work item, when one is routed.
    packages:
        Exact resolved-export package rows for that intent.
    cas_root:
        The model's author-custody ``source-cas`` root.
    resolved_export_sha256:
        Optional lock-artifact digest binding the rows to their solve.
    row_cache_path:
        Durable sidecar recording the minted row for identity stability.
        ``None`` mints a fresh row every call (tests).
    clock:
        UTC timestamp source for a fresh mint.

    Returns
    -------
    dict[str, Any] | None
        Complete manifest row, or ``None`` when the machine holds no
        inventory for this route (unrouted intent, unlocked target, or empty
        export). ``None`` keeps the manifest exactly as before, which reads
        downstream as "no inventory held", never as an empty inventory.
    """

    if not isinstance(intent, str) or not intent:
        return None
    content = render_routed_inventory_document(
        intent=intent,
        packages=packages,
        resolved_export_sha256=resolved_export_sha256,
    )
    if content is None:
        return None
    digest = hash_bytes(content)
    destination = publish_cas_object(cas_root, digest, content)
    cached = _cached_row(row_cache_path, digest)
    if cached is not None:
        return cached
    row: JsonObject = {
        "source_id": ROUTED_INVENTORY_SOURCE_ID,
        # Self-describing URN over the frozen bytes, exactly like discovery
        # evidence: resolvable inside the campaign store, unforgeable outside
        # it. The intent rides in the row and in the frozen document itself.
        "url": f"urn:menagerie:routed-environment-inventory:{digest.removeprefix('sha256:')}",
        "revision": digest,
        "content_sha256": digest,
        "fetched_bytes_len": len(content),
        "retrieval_status": "machine-derived",
        "retrieved_at": clock(),
        "media_type": "application/json",
        "cas_path": str(destination),
        "source_kind": ROUTED_INVENTORY_SOURCE_KIND,
        "routed_environment_intent": intent,
        "resolved_export_sha256": resolved_export_sha256,
        "basis": (
            "Machine-materialized routed-environment inventory: the exact "
            "resolved package export the R1_LIBRARY availability judgment for "
            "this model is adjudicated against."
        ),
    }
    if row_cache_path is not None:
        _write_row_cache(Path(row_cache_path), row)
    return row


def _cached_row(row_cache_path: Union[str, Path, None], digest: str) -> Optional[JsonObject]:
    """Return the persisted row when it still describes the current inventory.

    Parameters
    ----------
    row_cache_path:
        Durable sidecar location, or ``None`` when persistence is disabled.
    digest:
        Digest of the freshly rendered inventory document.

    Returns
    -------
    dict[str, Any] | None
        The recorded row verbatim when its ``content_sha256`` equals the
        current digest, otherwise ``None``. A missing, unreadable, or stale
        sidecar simply re-mints -- the CAS digest check downstream refuses
        wrong bytes regardless, so this cache can never substitute content.
    """

    if row_cache_path is None:
        return None
    path = Path(row_cache_path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        isinstance(value, dict)
        and value.get("source_id") == ROUTED_INVENTORY_SOURCE_ID
        and value.get("content_sha256") == digest
    ):
        return value
    return None


def _write_row_cache(path: Path, row: Mapping[str, Any]) -> None:
    """Atomically persist one minted manifest row beside the author custody."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(row, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
