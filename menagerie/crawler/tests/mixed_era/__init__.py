"""AB-30 mixed-era reader matrix support (Wave-2 rotation batch, PLAN_v6).

Every schema seam the Wave-2 rotation batch touches gets a reader x era
matrix: pre-batch artifacts are captured and frozen BEFORE the batch merges,
and every declared reader of that schema must either keep parsing prior-era
rows with era-scoped semantics or refuse them TYPED. A prior-era row that
crashes a reader unclassified, or silently changes meaning, is exactly the
mixed-era breakage this matrix exists to catch.

The committed declaration registry (``wave2_batch_declarations.json``)
enumerates every batch-touched schema seam, sourced from the Wave-2 lanes'
design notes. ``test_mixed_era_matrix_census.py`` is the census asserting
every declared seam carries its obligation; the ``test_mixed_era_*`` modules
are the per-seam matrices. IC-9's schema-migration census
(``menagerie.crawler.tools.schema_migration_census``) is the pairing
consumer and runs against the same frozen pre-batch artifacts.

Phases: the default run adjudicates the PRE-BATCH baseline (fixtures minted
by the freeze tree must parse; future-vocabulary members must refuse typed).
Setting ``MENAGERIE_WAVE2_ASSEMBLY=1`` switches the matrix to batch-assembly
adjudication: readers changed by the batch must land in a declared outcome
class -- never an unclassified crash -- and seams whose matrix was owed by a
batch lane refuse the census until discharged.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

REGISTRY_SCHEMA_VERSION = "menagerie.crawler.wave2-mixed-era-registry.v1"
MANIFEST_SCHEMA_VERSION = "menagerie.crawler.wave2-prebatch-fixture-manifest.v1"

#: Closed era classes a declared seam may take.
ERA_CLASSES = frozenset({"prior-era", "no-prior-era", "first-generation"})
#: Closed matrix dispositions for prior-era seams.
MATRIX_DISPOSITIONS = frozenset({"landed-here", "owed-by-batch-lane"})
#: Closed post-batch expectations a declared reader may take.
POST_BATCH_EXPECTATIONS = frozenset(
    {"parse-era-scoped", "parse-identical-verdict", "refuse-typed", "flip-declared"}
)

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REGISTRY_PATH = _PACKAGE_ROOT / "wave2_batch_declarations.json"
_FIXTURES_ROOT = _PACKAGE_ROOT / "fixtures"
_MANIFEST_PATH = _FIXTURES_ROOT / "MANIFEST.json"

_SEAM_REQUIRED_KEYS = frozenset(
    {"seam_id", "owner", "plan_items", "design_note_source", "declared_change", "era_class"}
)
_SEAM_OPTIONAL_KEYS = frozenset(
    {
        "matrix_disposition",
        "matrix_module",
        "fixtures",
        "readers",
        "refusal_proof_nodes",
        "owed_by",
        "notes",
    }
)
_READER_REQUIRED_KEYS = frozenset({"reader_id", "post_batch_expectation"})
_READER_OPTIONAL_KEYS = frozenset({"expected_flip", "notes"})


class MixedEraRegistryError(ValueError):
    """Typed refusal for a malformed mixed-era declaration registry."""


def assembly_mode() -> bool:
    """Return whether the matrix is adjudicating the assembled Wave-2 batch."""

    return os.environ.get("MENAGERIE_WAVE2_ASSEMBLY") == "1"


def load_registry() -> Mapping[str, Any]:
    """Load and strictly validate the committed declaration registry.

    Returns
    -------
    Mapping[str, Any]
        The validated registry document.

    Raises
    ------
    MixedEraRegistryError
        On any unknown schema version, missing key, unknown key, or
        closed-vocabulary violation. The registry is itself a closed
        vocabulary: an unreviewed shape must refuse, never half-parse.
    """

    document = json.loads(_REGISTRY_PATH.read_text())
    if document.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise MixedEraRegistryError(
            f"unknown mixed-era registry schema: {document.get('schema_version')!r}"
        )
    freeze = document.get("freeze")
    if not isinstance(freeze, Mapping) or not str(freeze.get("tree_sha", "")):
        raise MixedEraRegistryError("registry lacks its freeze tree_sha")
    seams = document.get("seams")
    if not isinstance(seams, list) or not seams:
        raise MixedEraRegistryError("registry names no seams")
    seen: set[str] = set()
    for seam in seams:
        if not isinstance(seam, Mapping):
            raise MixedEraRegistryError("seam row is not an object")
        keys = frozenset(seam)
        missing = _SEAM_REQUIRED_KEYS - keys
        unknown = keys - _SEAM_REQUIRED_KEYS - _SEAM_OPTIONAL_KEYS
        if missing or unknown:
            raise MixedEraRegistryError(
                f"seam {seam.get('seam_id')!r} keys invalid: missing={sorted(missing)} "
                f"unknown={sorted(unknown)}"
            )
        seam_id = str(seam["seam_id"])
        if seam_id in seen:
            raise MixedEraRegistryError(f"duplicate seam: {seam_id}")
        seen.add(seam_id)
        era = seam["era_class"]
        if era not in ERA_CLASSES:
            raise MixedEraRegistryError(f"seam {seam_id}: unknown era_class {era!r}")
        if era == "prior-era":
            disposition = seam.get("matrix_disposition")
            if disposition not in MATRIX_DISPOSITIONS:
                raise MixedEraRegistryError(
                    f"seam {seam_id}: prior-era rows need a closed matrix_disposition, "
                    f"got {disposition!r}"
                )
            if disposition == "landed-here" and not str(seam.get("matrix_module", "")):
                raise MixedEraRegistryError(f"seam {seam_id}: landed-here without matrix_module")
            if disposition == "owed-by-batch-lane" and not str(seam.get("owed_by", "")):
                raise MixedEraRegistryError(f"seam {seam_id}: owed row must name owed_by")
        for reader in seam.get("readers", ()):  # type: ignore[union-attr]
            if not isinstance(reader, Mapping):
                raise MixedEraRegistryError(f"seam {seam_id}: reader row is not an object")
            rkeys = frozenset(reader)
            rmissing = _READER_REQUIRED_KEYS - rkeys
            runknown = rkeys - _READER_REQUIRED_KEYS - _READER_OPTIONAL_KEYS
            if rmissing or runknown:
                raise MixedEraRegistryError(
                    f"seam {seam_id} reader {reader.get('reader_id')!r} keys invalid: "
                    f"missing={sorted(rmissing)} unknown={sorted(runknown)}"
                )
            if reader["post_batch_expectation"] not in POST_BATCH_EXPECTATIONS:
                raise MixedEraRegistryError(
                    f"seam {seam_id} reader {reader['reader_id']!r}: unknown "
                    f"post_batch_expectation {reader['post_batch_expectation']!r}"
                )
            if reader["post_batch_expectation"] == "flip-declared" and not str(
                reader.get("expected_flip", "")
            ):
                raise MixedEraRegistryError(
                    f"seam {seam_id} reader {reader['reader_id']!r}: flip-declared "
                    "without a named expected_flip plan item"
                )
    return document


def load_manifest() -> Mapping[str, Any]:
    """Load and validate the frozen pre-batch fixture manifest."""

    document = json.loads(_MANIFEST_PATH.read_text())
    if document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise MixedEraRegistryError(
            f"unknown fixture manifest schema: {document.get('schema_version')!r}"
        )
    return document


def fixture_path(name: str) -> Path:
    """Return the absolute path of a committed fixture by manifest name."""

    return _FIXTURES_ROOT / name


def read_fixture_bytes(name: str) -> bytes:
    """Read one frozen fixture, refusing on any digest drift.

    Every matrix read is attested against the capture-time manifest so a
    silently edited fixture can never adjudicate an era claim.
    """

    manifest = load_manifest()
    entries = manifest.get("entries", {})
    if name not in entries:
        raise MixedEraRegistryError(f"fixture {name!r} is not in the frozen manifest")
    data = fixture_path(name).read_bytes()
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    if digest != entries[name]:
        raise MixedEraRegistryError(
            f"fixture {name!r} bytes contradict the frozen manifest: {digest} != {entries[name]}"
        )
    return data


def read_fixture_json(name: str) -> Any:
    """Read one frozen JSON fixture with digest attestation."""

    return json.loads(read_fixture_bytes(name))


def classify_reader(call: Callable[[], Any]) -> tuple[str, Any]:
    """Run one reader over a frozen artifact and classify the outcome.

    Returns
    -------
    tuple[str, Any]
        ``("parsed", value)`` when the reader accepted the artifact, or
        ``("refused-typed", exception)`` when it raised a crawler-typed
        refusal (an exception class defined inside ``menagerie.``).

    Raises
    ------
    BaseException
        Any non-crawler exception propagates: an unclassified crash on a
        prior-era row is a matrix FAILURE, never a lawful refusal.
    """

    try:
        return ("parsed", call())
    except Exception as exc:
        if type(exc).__module__.startswith("menagerie."):
            return ("refused-typed", exc)
        raise


def assert_era_outcome(
    *,
    reader_id: str,
    outcome: tuple[str, Any],
    prebatch_expected: str,
    expected_flip: Optional[str] = None,
) -> None:
    """Adjudicate one (reader, frozen-era-artifact) matrix cell.

    In the pre-batch phase the cell must land exactly on its captured
    baseline. In assembly phase (``MENAGERIE_WAVE2_ASSEMBLY=1``) a cell
    with a declared flip may land on either side -- the batch may or may
    not have carried the flipping item -- but it must always land in a
    CLASSIFIED outcome; ``classify_reader`` has already propagated any
    unclassified crash as a hard failure before this assertion runs.
    """

    kind = outcome[0]
    if assembly_mode() and expected_flip:
        assert kind in {"parsed", "refused-typed"}, (
            f"{reader_id}: unclassifiable assembly outcome {kind!r}"
        )
        return
    assert kind == prebatch_expected, (
        f"{reader_id}: expected {prebatch_expected!r} on the frozen pre-batch "
        f"artifact, observed {kind!r} ({outcome[1]!r})"
    )
