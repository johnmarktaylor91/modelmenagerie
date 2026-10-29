"""Honest disposition for a model routed to an environment that cannot serve it.

Routing is decided at intake from the immutable roster row (zoo and era); the
distribution a model actually needs is decided later, by the author, in the
declarative recipe. When the two disagree the model lands in an intent whose
package set does not contain its library at all, and
``recipe.resolve_environment_artifact_digest`` correctly refuses to invent a
digest for a distribution the routed environment does not install.

That refusal is right and stays exactly as it is. What was wrong is the RECORD
it produced. A DGL model routed to ``core`` -- because its roster zoo carried no
graph marker -- terminalized as ``failed:runner`` / ``protocol-violation``, which
asserts our pipeline broke on the model. Nothing broke. The catalog deliberately
keeps ``core`` free of DGL (conda-forge's ``dgl`` hard-pins torch to 2.3.1 and
drags in TensorFlow, so admitting it would downgrade torch and numpy for every
core model), and there is a ``graph`` intent that carries DGL precisely so those
models have a home. The truthful statement is "this needs the graph
environment", and it names the thing that would cover the model later.

This module derives that statement:

* :func:`assess_environment_coverage` compares the recipe's OWN declared
  distribution against the routed intent's exact resolved-export inventory --
  the same inventory, through the same namespace bridge, that the refusal reads
  -- so the verdict can never disagree with the refusal it explains.
* When the routed intent cannot serve it, the registry is searched for an intent
  that can, and the row records which intent and on what BASIS: a locked
  inventory that provably contains it, a declared dependency that asks for it,
  or the routing table that already maps that package to an intent.
* The result is a durable, append-only row carrying the declared distribution,
  the routed intent, the covering intent, and a recheck hint -- so a reviewer
  can act on it without re-running anything.

Three properties are deliberate, and they mirror
:mod:`menagerie.crawler.host_capacity` on purpose: these are the same kind of
fact ("this campaign cannot run this model here, and here is what would") and
they should read the same way.

**It is a refusal, not a relaxation.** Nothing here weakens a validation check.
A model whose library genuinely is not installed still never runs, still gets no
artifact digest, and still asserts no rung. It is simply withheld and recorded
instead of being blamed.

**It is not a canonical terminal.** Every canonical ``deferred:*`` model status
is proved either by a checker-adjudicated DEFER gate or by an accepted BLOCKED
gate; there is no third proof path. This disposition is machine-derived from an
inventory, has no adjudication behind it, and therefore does not claim one.
See :data:`COVERAGE_DEFERRAL_DISPOSITION`.

**It is fail-open toward attempting.** A model is withheld ONLY when the routed
environment publishes an inventory AND that inventory provably lacks the
declared distribution. No inventory, no declarative recipe, or an unreadable
recipe all mean "not assessable", which admits the model unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
try:  # python >= 3.11
    from enum import StrEnum
except ImportError:  # python 3.10: the oldest routed-env interpreter (legacy-torch
    # lock resolves py3.10). The worker closure must import there; the constants
    # shim is value-identical (str content == member value), differing only in
    # str()/format() rendering, which nothing identity-bearing uses (DX-3/RE-3).
    from menagerie.crawler.constants import StrEnum  # type: ignore[assignment]
import fcntl
import json
import os
import re
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional, Sequence

from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    foreign_content_address_fields,
    fsync_directory,
    row_content_address,
    stamp_row_digest,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.package_namespace import (
    canonical_distribution_name,
    dependency_spec_name,
    inventory_row_provides_distribution,
)
from menagerie.crawler.recordio import read_with_tail_recovery

COVERAGE_DEFERRAL_SCHEMA_VERSION = "menagerie.crawler.environment-coverage-deferral.v1"

COVERAGE_DEFERRAL_DISPOSITION = "deferred:needs-environment-coverage"
"""Greppable disposition for a model routed to an intent that cannot serve it.

It follows the established ``deferred:needs-<capability>`` shape
(``needs-cuda``, ``needs-x86``, ``needs-source-access``, ``needs-more-memory``):
a real, located, correctly authored model that this campaign cannot execute AS
ROUTED, naming the capability that would recover it. A different environment
intent is such a capability, and the row names the exact one.

It is deliberately NOT ``failed:*``. A ``failed:`` code asserts that our pipeline
broke on a model. Nothing broke: the author declared a real distribution, the
routed environment honestly does not install it, and the refusal that says so is
working. The defect is in the ROUTING, which is a campaign-level fact and is
recoverable by re-running the model against the intent that carries its library.

It is also deliberately not minted as a canonical model terminal, for exactly the
reason ``host_capacity.CAPACITY_DEFERRAL_DISPOSITION`` is not: a canonical
``deferred:*`` record must carry a checker-adjudicated terminal gate over frozen
sources (see ``authority._derive_deferral`` and
``reducer._derive_blocked_capability_deferral_proof``), and this disposition is
machine-derived from a package inventory with no such adjudication. Claiming
that terminal would mean fabricating a proof. The model instead stays
UNCOMPLETED for this campaign and is recorded here, which is the truthful
statement: not attempted, not failed, recoverable in a named environment.
"""

COVERAGE_DEFERRAL_RELATIVE_PATH = Path("intake-extensions") / "environment-coverage-deferrals.jsonl"

PACKAGE_INDEX_TOKENS = frozenset({"pypi", "conda", "conda-forge"})
"""Package-index and channel names that can never name a Python distribution.

This is the valve-side echo of the author-result contract
(``author_dispatch._validate_routed_distribution``: "Package indexes such as
'pypi' and 'conda' are not distributions"). That contract only runs when the
author envelope disclosed the routed inventory, so a proposal minted before the
contract existed -- or validated without a disclosed inventory -- can still
reach this valve declaring an index as its "distribution". Rung 10 held five
such rows: each recorded the remedy "declare distribution 'pypi' in an
environment intent", which no operator can ever action because 'pypi' is where
distributions live, not one of them.
"""


def distribution_vocabulary_rejection(distribution: str) -> Optional[str]:
    """Explain why a declared name cannot name a Python distribution, or ``None``.

    Parameters
    ----------
    distribution:
        The name a declarative recipe pins as its library distribution.

    Returns
    -------
    str | None
        Human-readable rejection for a package-index token, or ``None`` for a
        name that is at least plausibly a distribution.
    """

    if canonical_distribution_name(distribution) in PACKAGE_INDEX_TOKENS:
        return (
            f"{distribution!r} names a package index, not a Python distribution, "
            "so no environment intent could ever carry it"
        )
    return None


# ---------------------------------------------------------------------------
# EN-6: closed prerequisite-token grammar and the known-19 aggregation
# ---------------------------------------------------------------------------

PREREQUISITE_TOKEN_GRAMMAR_SCHEMA_VERSION = (
    "menagerie.crawler.environment-prerequisite-token-grammar.v1"
)
"""Schema identity of the closed prerequisite-token grammar (EN-6).

Born BEFORE the 200b census so every 200b prerequisite claim is classifiable
from its first row. The terminal/prerequisite vocabulary schema has ONE owner
(the W14 seam, per PLAN_v6 WS2); this module contributes the environment-side
grammar and consumes it -- the constant names the contract a schema move must
carry along.
"""

RUNTIME_DEPENDENCY_TOKEN_PREFIX = "runtime-dependency:"
"""Canonical forward grammar for an environment prerequisite token.

``runtime-dependency:<pep503>`` -- the marker, one colon, then the PEP 503
normalized distribution name. The pilot's five ad-hoc spellings (bare marker,
``runtime-dependency-<name>``, ``<name>-distribution``, ``<name>-package``,
bare names) are classified by :func:`parse_prerequisite_token` so the known-19
replay stays deterministic, but only this form is rendered going forward.
"""

_LEGACY_HYPHEN_PREFIX = "runtime-dependency-"
_LEGACY_TOKEN_SUFFIXES = ("-distribution", "-package")
_PEP503_NORMALIZED_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\Z")

PLATFORM_CLASS_DISTRIBUTIONS = frozenset(
    {"triton", "flash-attn", "spconv", "knn-cuda", "cuda-runtime"}
)
"""CLOSED set of CUDA-platform-bound prerequisite distributions (EN-6).

Every member is census-evidenced in the pilot known-19 corpus
(``tests/pilot_known19_prerequisites.jsonl``): a build that exists only for a
CUDA toolchain/host, so a prerequisite set consisting ENTIRELY of members is a
capability fact ("this model needs a CUDA host"), never a pipeline failure.
Extending this set is a reviewed vocabulary change (Wave-2-by-law), never a
runtime inference.
"""


class PrerequisiteTokenKind(StrEnum):
    """Closed classification of one author-named prerequisite token spelling."""

    RUNTIME_DEPENDENCY = "runtime-dependency"
    """Canonical ``runtime-dependency:<pep503>`` forward-grammar form."""

    LEGACY_HYPHEN = "legacy-hyphen"
    """Pilot ``runtime-dependency-<name>`` spelling; the name is extractable."""

    LEGACY_SUFFIXED = "legacy-suffixed"
    """Pilot ``<name>-distribution`` / ``<name>-package`` spelling."""

    BARE_DISTRIBUTION = "bare-distribution"
    """A plausible PEP 503 name carrying no marker at all."""

    CONTENT_FREE = "content-free"
    """A token naming NO distribution (m4954's bare ``runtime-dependency``)."""


@dataclass(frozen=True)
class PrerequisiteToken:
    """One classified prerequisite token: raw spelling, kind, and extraction."""

    raw: str
    kind: PrerequisiteTokenKind
    distribution: Optional[str]
    platform_class: bool

    def to_json(self) -> JsonObject:
        """Return the immutable JSON projection of this classification."""

        return {
            "raw": self.raw,
            "kind": self.kind.value,
            "distribution": self.distribution,
            "platform_class": self.platform_class,
        }


def canonical_prerequisite_token(distribution: str) -> str:
    """Render one distribution as its canonical forward-grammar token."""

    return f"{RUNTIME_DEPENDENCY_TOKEN_PREFIX}{canonical_distribution_name(distribution)}"


def _classified(raw: str, kind: PrerequisiteTokenKind, name: str) -> PrerequisiteToken:
    """Build one token classification, demoting an unusable name to content-free."""

    canonical = canonical_distribution_name(name)
    if not _PEP503_NORMALIZED_NAME.fullmatch(canonical):
        return PrerequisiteToken(
            raw=raw, kind=PrerequisiteTokenKind.CONTENT_FREE, distribution=None, platform_class=False
        )
    return PrerequisiteToken(
        raw=raw,
        kind=kind,
        distribution=canonical,
        platform_class=canonical in PLATFORM_CLASS_DISTRIBUTIONS,
    )


def parse_prerequisite_token(raw: str) -> PrerequisiteToken:
    """Classify one author-named prerequisite token under the closed grammar.

    Total over arbitrary strings: every input lands in exactly one
    :class:`PrerequisiteTokenKind`, and a token that names no distribution --
    including the bare ``runtime-dependency`` marker the pilot's m4954 emitted
    -- is CONTENT_FREE and can never steer a disposition.

    Parameters
    ----------
    raw:
        The verbatim token from a checked BLOCKED recommendation's
        ``prerequisite_ids``.

    Returns
    -------
    PrerequisiteToken
        Classification with the extracted PEP 503 distribution, when one exists.
    """

    stripped = raw.strip()
    lowered = stripped.casefold()
    marker = RUNTIME_DEPENDENCY_TOKEN_PREFIX[:-1]  # "runtime-dependency"
    if lowered == marker or lowered in (RUNTIME_DEPENDENCY_TOKEN_PREFIX, _LEGACY_HYPHEN_PREFIX):
        return PrerequisiteToken(
            raw=raw, kind=PrerequisiteTokenKind.CONTENT_FREE, distribution=None, platform_class=False
        )
    if lowered.startswith(RUNTIME_DEPENDENCY_TOKEN_PREFIX):
        name = stripped[len(RUNTIME_DEPENDENCY_TOKEN_PREFIX) :]
        return _classified(raw, PrerequisiteTokenKind.RUNTIME_DEPENDENCY, name)
    if lowered.startswith(_LEGACY_HYPHEN_PREFIX):
        name = stripped[len(_LEGACY_HYPHEN_PREFIX) :]
        return _classified(raw, PrerequisiteTokenKind.LEGACY_HYPHEN, name)
    for suffix in _LEGACY_TOKEN_SUFFIXES:
        if lowered.endswith(suffix) and len(lowered) > len(suffix):
            return _classified(
                raw, PrerequisiteTokenKind.LEGACY_SUFFIXED, stripped[: -len(suffix)]
            )
    if stripped:
        return _classified(raw, PrerequisiteTokenKind.BARE_DISTRIBUTION, stripped)
    return PrerequisiteToken(
        raw=raw, kind=PrerequisiteTokenKind.CONTENT_FREE, distribution=None, platform_class=False
    )


class PrerequisiteAggregateDisposition(StrEnum):
    """Closed aggregate outcomes for one BLOCKED verdict's prerequisite set."""

    PLATFORM_DEFERRAL = "platform-deferral"
    """Every unserved prerequisite is platform-class: a CUDA host recovers it."""

    BLOCKED_PREREQUISITE = "blocked-prerequisite"
    """The set is well-formed but a CUDA host alone does not recover it."""

    GRAMMAR_REFUSED = "grammar-refused"
    """A content-free token is present: the set asserts something it cannot name."""


@dataclass(frozen=True)
class PrerequisiteAggregation:
    """One complete prerequisite-set decision with its per-token evidence."""

    tokens: tuple[PrerequisiteToken, ...]
    disposition: PrerequisiteAggregateDisposition
    served: tuple[str, ...]
    platform: tuple[str, ...]
    covered: tuple[tuple[str, str], ...]
    unrecoverable: tuple[str, ...]
    explanation: str

    def to_json(self) -> JsonObject:
        """Return the immutable JSON projection of this aggregation."""

        return {
            "schema_version": PREREQUISITE_TOKEN_GRAMMAR_SCHEMA_VERSION,
            "tokens": [token.to_json() for token in self.tokens],
            "disposition": self.disposition.value,
            "served": list(self.served),
            "platform": list(self.platform),
            "covered": [list(pair) for pair in self.covered],
            "unrecoverable": list(self.unrecoverable),
            "explanation": self.explanation,
        }


def aggregate_prerequisite_tokens(
    raw_tokens: Sequence[str],
    *,
    routed_intent: Optional[str] = None,
    routed_packages: Sequence[Mapping[str, Any]] = (),
    registry: Any = None,
) -> PrerequisiteAggregation:
    """Aggregate one BLOCKED verdict's prerequisite set onto a closed disposition.

    The ladder is closed and deterministic:

    1. any CONTENT_FREE token refuses aggregation (``grammar-refused``) -- a set
       that asserts an unnamed prerequisite can never earn a deferral;
    2. tokens whose distribution the routed inventory provably installs are
       SERVED and discounted (they are present, so they cannot be the blocker);
    3. if the unserved remainder is non-empty and ENTIRELY platform-class, the
       set is CUDA-bound: ``platform-deferral``;
    4. everything else stays ``blocked-prerequisite``, with covering intents
       named per token (the coverage story the recheck consumes) and the
       residue listed as unrecoverable.

    Parameters
    ----------
    raw_tokens:
        Verbatim ``prerequisite_ids`` from the checked BLOCKED recommendation.
    routed_intent:
        Intent the model was routed to; excluded from covering-intent search.
    routed_packages:
        The routed intent's exact resolved-export rows (empty means unknown --
        nothing is discounted).
    registry:
        Loaded environment registry for the covering-intent search, or ``None``.

    Returns
    -------
    PrerequisiteAggregation
        Closed disposition with per-token evidence.
    """

    tokens = tuple(parse_prerequisite_token(raw) for raw in raw_tokens)
    if any(token.kind is PrerequisiteTokenKind.CONTENT_FREE for token in tokens):
        return PrerequisiteAggregation(
            tokens=tokens,
            disposition=PrerequisiteAggregateDisposition.GRAMMAR_REFUSED,
            served=(),
            platform=(),
            covered=(),
            unrecoverable=(),
            explanation=(
                "at least one prerequisite token names no distribution (the bare "
                "runtime-dependency marker); a set that cannot say what blocks it "
                "cannot steer a disposition"
            ),
        )
    served: list[str] = []
    platform: list[str] = []
    covered: list[tuple[str, str]] = []
    unrecoverable: list[str] = []
    for distribution in dict.fromkeys(
        token.distribution for token in tokens if token.distribution is not None
    ):
        if _inventory_provides(tuple(routed_packages), distribution) is not None:
            served.append(distribution)
            continue
        if distribution in PLATFORM_CLASS_DISTRIBUTIONS:
            platform.append(distribution)
            continue
        covering = find_covering_intents(distribution, registry, exclude=routed_intent)
        if covering:
            covered.append((distribution, covering[0].intent))
        else:
            unrecoverable.append(distribution)
    if platform and not covered and not unrecoverable:
        disposition = PrerequisiteAggregateDisposition.PLATFORM_DEFERRAL
        explanation = (
            "every unserved prerequisite is a CUDA-platform-class build "
            f"({', '.join(platform)}); a CUDA host recovers this model"
        )
    else:
        disposition = PrerequisiteAggregateDisposition.BLOCKED_PREREQUISITE
        explanation = (
            "the prerequisite set is not recoverable by a CUDA host alone; "
            f"served={list(served)}, platform={list(platform)}, "
            f"covered={[list(pair) for pair in covered]}, "
            f"unrecoverable={list(unrecoverable)}"
        )
    return PrerequisiteAggregation(
        tokens=tokens,
        disposition=disposition,
        served=tuple(served),
        platform=tuple(platform),
        covered=tuple(covered),
        unrecoverable=tuple(unrecoverable),
        explanation=explanation,
    )


def census_platform_class_environment_failures(
    rows: Sequence[tuple[str, Sequence[str]]],
) -> tuple[int, ...]:
    """Return the indices of failed:environment rows whose prerequisites are all platform-class.

    The EN-6 cross-check tripwire: after the terminal crossing learns the
    platform-deferral arm, an all-platform prerequisite set terminalized as
    ``failed:environment`` is a regression -- the census must stay EMPTY over
    every records set.

    Parameters
    ----------
    rows:
        ``(terminal_status_code, prerequisite_ids)`` pairs.

    Returns
    -------
    tuple[int, ...]
        Indices of violating rows; empty when the invariant holds.
    """

    violations: list[int] = []
    for index, (status_code, prerequisite_ids) in enumerate(rows):
        if status_code != "failed:environment":
            continue
        aggregation = aggregate_prerequisite_tokens(tuple(prerequisite_ids))
        if aggregation.disposition is PrerequisiteAggregateDisposition.PLATFORM_DEFERRAL:
            violations.append(index)
    return tuple(violations)


class CoverageVerdict(StrEnum):
    """Closed routed-environment coverage outcomes."""

    SERVED = "served"
    """The routed environment installs the declared distribution."""

    UNSERVABLE = "unservable"
    """The routed environment publishes an inventory that provably lacks it."""

    NOT_ASSESSABLE = "not-assessable"
    """No inventory or no declarative distribution; the model is admitted."""


class CoverageBasis(StrEnum):
    """How a covering intent was established, strongest evidence first."""

    LOCKED_INVENTORY = "locked-inventory"
    """The candidate intent's own locked resolved export contains the package."""

    DECLARED_DEPENDENCY = "declared-dependency"
    """The candidate intent's environment spec asks for the package by name."""

    ROUTING_TABLE = "routing-table"
    """``routing._PACKAGE_INTENTS`` already maps this package to the intent."""


class CoverageDeferralError(ValueError):
    """Raised when an environment-coverage deferral row is malformed or unpersistable."""


@dataclass(frozen=True)
class DeclaredLibrary:
    """The distribution and version one declarative recipe pins."""

    distribution: str
    version: str

    def to_json(self) -> JsonObject:
        """Return the immutable JSON projection of this pin."""

        return {"distribution": self.distribution, "version": self.version}


@dataclass(frozen=True)
class CoveringIntent:
    """One environment intent that would carry the declared distribution."""

    intent: str
    basis: CoverageBasis
    evidence: str

    def to_json(self) -> JsonObject:
        """Return the immutable JSON projection of this coverage claim."""

        return {"intent": self.intent, "basis": self.basis.value, "evidence": self.evidence}


@dataclass(frozen=True)
class CoverageAssessment:
    """One complete routed-environment coverage decision with its justification."""

    verdict: CoverageVerdict
    routed_intent: Optional[str]
    library: Optional[DeclaredLibrary]
    covering: tuple[CoveringIntent, ...]
    explanation: str

    @property
    def deferred(self) -> bool:
        """Return whether this assessment withholds the model from its route."""

        return self.verdict is CoverageVerdict.UNSERVABLE

    @property
    def preferred_intent(self) -> Optional[str]:
        """Return the strongest-evidence covering intent, when one exists."""

        return self.covering[0].intent if self.covering else None

    def to_json(self) -> JsonObject:
        """Return the immutable JSON projection of this decision."""

        return {
            "verdict": self.verdict.value,
            "routed_intent": self.routed_intent,
            "library": self.library.to_json() if self.library is not None else None,
            "covering_intents": [item.to_json() for item in self.covering],
            "explanation": self.explanation,
        }


def declared_library(implementation: Mapping[str, Any]) -> Optional[DeclaredLibrary]:
    """Return the distribution pin a declarative-library recipe declares.

    The guard conditions mirror ``recipe.bind_library_artifact_digest`` exactly,
    so this can only ever speak about recipes that binding actually resolves.

    Parameters
    ----------
    implementation:
        Proposal implementation block.

    Returns
    -------
    DeclaredLibrary | None
        The pinned distribution and version, or ``None`` when this recipe is not
        a declarative-library pin at all.
    """

    if implementation.get("recipe_type") != "declarative-library":
        return None
    recipe = implementation.get("library_recipe")
    if not isinstance(recipe, Mapping):
        return None
    distribution = recipe.get("distribution")
    version = recipe.get("version")
    if not isinstance(distribution, str) or not isinstance(version, str):
        return None
    if not distribution.strip():
        return None
    return DeclaredLibrary(distribution=distribution, version=version)


def _inventory_provides(packages: Sequence[Mapping[str, Any]], distribution: str) -> Optional[str]:
    """Return the inventory row name providing ``distribution``, if any."""

    for row in packages:
        if not isinstance(row, Mapping):
            continue
        name = row.get("name")
        if isinstance(name, str) and inventory_row_provides_distribution(name, distribution):
            return name
    return None


def _routing_table_intent(distribution: str) -> Optional[str]:
    """Return the intent ``routing._PACKAGE_INTENTS`` already maps this package to."""

    # Imported here rather than at module scope: `routing` is a peer of the
    # driver's import graph and this module is also loaded by the CLI, which has
    # no reason to pull the routing tables in.
    from menagerie.crawler.routing import _PACKAGE_INTENTS

    wanted = canonical_distribution_name(distribution)
    for intent, markers in _PACKAGE_INTENTS:
        if any(canonical_distribution_name(marker) == wanted for marker in markers):
            return intent
    return None


def find_covering_intents(
    distribution: str,
    registry: Any,
    *,
    exclude: Optional[str] = None,
) -> tuple[CoveringIntent, ...]:
    """Find every intent that would carry ``distribution``, strongest basis first.

    Three independent signals are consulted, in descending order of how directly
    each one proves the claim. The BASIS travels with the answer, so a reader can
    tell a solved lock ("this intent demonstrably installs it") from a declared
    dependency ("this intent asks for it") from the routing table ("this package
    is already meant to route there"), rather than seeing one undifferentiated
    recommendation.

    Parameters
    ----------
    distribution:
        Declared Python distribution the model needs.
    registry:
        Loaded :class:`menagerie.crawler.envs.EnvironmentRegistry`, or ``None``.
    exclude:
        Routed intent to omit; it is by construction not a candidate.

    Returns
    -------
    tuple[CoveringIntent, ...]
        One entry per covering intent, ordered by basis strength then name. Empty
        when nothing in this campaign declares the distribution.
    """

    found: dict[str, CoveringIntent] = {}
    intents = getattr(registry, "intents", None)
    if isinstance(intents, Mapping):
        for name in sorted(intents):
            if name == exclude:
                continue
            intent = intents[name]
            packages = _resolved_export_packages(intent)
            row_name = _inventory_provides(packages, distribution)
            if row_name is not None:
                found[name] = CoveringIntent(
                    intent=name,
                    basis=CoverageBasis.LOCKED_INVENTORY,
                    evidence=(
                        f"the locked resolved export for {name!r} installs package "
                        f"{row_name!r}, which provides {distribution!r}"
                    ),
                )
                continue
            for spec in getattr(intent, "dependencies", ()):  # declared, not solved
                spec_name = dependency_spec_name(str(spec))
                if spec_name and inventory_row_provides_distribution(spec_name, distribution):
                    found[name] = CoveringIntent(
                        intent=name,
                        basis=CoverageBasis.DECLARED_DEPENDENCY,
                        evidence=(
                            f"the {name!r} environment spec declares dependency "
                            f"{str(spec).strip()!r}"
                        ),
                    )
                    break
    routed = _routing_table_intent(distribution)
    if routed is not None and routed != exclude and routed not in found:
        found[routed] = CoveringIntent(
            intent=routed,
            basis=CoverageBasis.ROUTING_TABLE,
            evidence=(
                f"routing._PACKAGE_INTENTS maps package {distribution!r} to intent {routed!r}"
            ),
        )
    order = {
        CoverageBasis.LOCKED_INVENTORY: 0,
        CoverageBasis.DECLARED_DEPENDENCY: 1,
        CoverageBasis.ROUTING_TABLE: 2,
    }
    return tuple(sorted(found.values(), key=lambda item: (order[item.basis], item.intent)))


def _resolved_export_packages(intent: Any) -> tuple[Mapping[str, Any], ...]:
    """Return one intent's exact resolved-export package rows, or nothing."""

    lock = getattr(intent, "lock", None)
    export_bytes = getattr(lock, "export_bytes", None)
    if not isinstance(export_bytes, (bytes, bytearray)):
        return ()
    status = getattr(lock, "status", None)
    if isinstance(status, str) and status != "locked":
        # A superseded (or otherwise unreadable) lock is last run's solve
        # residue, not this campaign's inventory: claiming LOCKED_INVENTORY
        # coverage from it would prove a covering intent with stale evidence.
        # The declared-dependency and routing-table bases still apply.
        return ()
    # Imported lazily for the same reason as the routing table: a ledger reader
    # should not need the environment-exactness machinery to print rows.
    from menagerie.crawler.env_lifecycle import EnvironmentExactnessError, parse_resolved_export

    try:
        value = json.loads(parse_resolved_export(bytes(export_bytes)))
    except (EnvironmentExactnessError, UnicodeDecodeError, json.JSONDecodeError):
        return ()
    packages = value.get("packages") if isinstance(value, Mapping) else None
    if not isinstance(packages, list):
        return ()
    rows = tuple(row for row in packages if isinstance(row, Mapping))
    # A pinned pip overlay is installed into the same exact runtime the lock
    # realizes, so its rows count as this intent's inventory for the
    # covering-intent search too (duck-typed: a fake intent without the
    # attribute simply carries no overlay).
    from menagerie.crawler.envs import overlay_package_rows

    return rows + tuple(overlay_package_rows(intent))


def assess_environment_coverage(
    implementation: Mapping[str, Any],
    *,
    routed_intent: Optional[str],
    routed_packages: Sequence[Mapping[str, Any]],
    registry: Any = None,
) -> CoverageAssessment:
    """Decide whether the routed environment can serve one authored recipe.

    Parameters
    ----------
    implementation:
        Proposal implementation block carrying the declarative recipe.
    routed_intent:
        Environment intent this model was routed to at intake.
    routed_packages:
        Exact package rows from that intent's resolved export. Empty means the
        intent is unlocked, which is "unknown", never "absent".
    registry:
        Loaded environment registry used to name a covering intent.

    Returns
    -------
    CoverageAssessment
        Verdict, the declared pin, any covering intents, and a plain-language
        reason.
    """

    library = declared_library(implementation)
    if library is None:
        return CoverageAssessment(
            verdict=CoverageVerdict.NOT_ASSESSABLE,
            routed_intent=routed_intent,
            library=None,
            covering=(),
            explanation=(
                "the proposal pins no declarative library distribution, so routed "
                "coverage says nothing about it and the model is admitted"
            ),
        )
    vocabulary_rejection = distribution_vocabulary_rejection(library.distribution)
    if vocabulary_rejection is not None:
        # An index token would always read UNSERVABLE (no inventory row provides
        # 'pypi'), and the row minted from that verdict would carry a remedy no
        # operator can action. The truthful statement is that coverage says
        # nothing about a name that is not a distribution at all: the model is
        # admitted unchanged and the recipe's own validation decides. This does
        # not weaken any check -- the author-result contract still refuses the
        # vocabulary outright whenever the routed inventory is disclosed.
        return CoverageAssessment(
            verdict=CoverageVerdict.NOT_ASSESSABLE,
            routed_intent=routed_intent,
            library=library,
            covering=(),
            explanation=(
                f"the declared pin is not assessable: {vocabulary_rejection}; the "
                "proposal is defective and deferring on it would record an "
                "unactionable remedy, so the model is admitted unchanged"
            ),
        )
    if not routed_packages:
        return CoverageAssessment(
            verdict=CoverageVerdict.NOT_ASSESSABLE,
            routed_intent=routed_intent,
            library=library,
            covering=(),
            explanation=(
                f"the routed intent {routed_intent!r} publishes no package inventory, so "
                "whether it installs "
                f"{library.distribution!r} is unknown and the model is admitted"
            ),
        )
    row_name = _inventory_provides(routed_packages, library.distribution)
    if row_name is not None:
        return CoverageAssessment(
            verdict=CoverageVerdict.SERVED,
            routed_intent=routed_intent,
            library=library,
            covering=(),
            explanation=(
                f"the routed intent {routed_intent!r} installs package {row_name!r}, which "
                f"provides {library.distribution!r}"
            ),
        )
    covering = find_covering_intents(library.distribution, registry, exclude=routed_intent)
    if covering:
        names = ", ".join(repr(item.intent) for item in covering)
        remedy = f"the {names} environment carries it"
    else:
        remedy = "no environment intent in this campaign declares it"
    return CoverageAssessment(
        verdict=CoverageVerdict.UNSERVABLE,
        routed_intent=routed_intent,
        library=library,
        covering=covering,
        explanation=(
            f"the model pins distribution {library.distribution!r}, and the intent it "
            f"routes to ({routed_intent!r}) installs no package providing it; {remedy}"
        ),
    )


def coverage_deferral_path(records_root: Path) -> Path:
    """Return the canonical environment-coverage deferral ledger below one records root.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        Append-only environment-coverage deferral JSONL path.
    """

    return records_root / COVERAGE_DEFERRAL_RELATIVE_PATH


def build_coverage_deferral_row(
    *,
    stable_id: str,
    work_id: str,
    name: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
    assessment: CoverageAssessment,
    routed_export_sha256: Optional[str] = None,
) -> JsonObject:
    """Assemble one complete, self-describing environment-coverage deferral row.

    The row states the declared pin, the intent that could not serve it, and
    every intent that could -- with the evidence for each -- so a later reviewer
    can act on it without the code that made the call.

    Parameters
    ----------
    stable_id, work_id, name:
        Durable model identity and the active scheduled work generation.
    campaign_id, run_id, machine_id:
        Campaign and host provenance for the deferral.
    created_at:
        UTC timestamp of the decision.
    assessment:
        The withholding coverage decision.
    routed_export_sha256:
        Digest of the routed intent's resolved-export inventory the verdict
        was adjudicated against, when one is locked (BL-9d rotation-blast
        audit: coverage rows were content-blind to environment generations,
        so a census could never say WHICH setup rotation a deferral wave
        followed). Optional and additive -- historical rows without it keep
        validating and re-claiming exactly as before.

    Returns
    -------
    dict[str, Any]
        Content-addressed deferral row.

    Raises
    ------
    CoverageDeferralError
        If the assessment does not actually withhold the model.
    """

    if not campaign_id or campaign_id == str(None):
        # T2-12: the pilot froze campaign_id="None" (a stringified absence)
        # into 62 row digests. The absence must refuse at BUILD time; read-side
        # validators stay tolerant so those historical rows remain readable.
        raise ValueError("campaign_id must name a real campaign, never a stringified absence")
    if not assessment.deferred:
        raise CoverageDeferralError(
            "only an unservable coverage assessment may be recorded as a deferral"
        )
    assert assessment.library is not None  # guaranteed by CoverageVerdict.UNSERVABLE
    distribution = assessment.library.distribution
    vocabulary_rejection = distribution_vocabulary_rejection(distribution)
    if vocabulary_rejection is not None:
        # Belt-and-suspenders behind the assessment-side refusal above: the MINT
        # is the last place a bogus remedy can be stopped before it becomes a
        # durable ledger fact. The READ path (`validate_coverage_deferral_row`)
        # deliberately stays permissive so ledgers already holding pre-fix rows
        # keep loading; those rows are neutralized where they could act instead
        # (they can never re-claim a model on resume, and their models' terminals
        # supersede them in the completion partition).
        raise CoverageDeferralError(
            "refusing to mint an environment-coverage deferral whose remedy could "
            f"never be actioned: {vocabulary_rejection}"
        )
    if assessment.preferred_intent is not None:
        recheck_hint = (
            f"re-run this model routed to the {assessment.preferred_intent!r} environment "
            f"intent, which carries distribution {distribution!r}; the intent it routed to "
            f"({assessment.routed_intent!r}) deliberately does not"
        )
    else:
        recheck_hint = (
            f"declare distribution {distribution!r} in an environment intent (or add an "
            "intent that carries it) and re-run this model; no intent in this campaign "
            "declares it today"
        )
    row: JsonObject = {
        "schema_version": COVERAGE_DEFERRAL_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": work_id,
        "name": name,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "created_at": created_at,
        "disposition": COVERAGE_DEFERRAL_DISPOSITION,
        "coverage": assessment.to_json(),
        "recheck_hint": recheck_hint,
    }
    if routed_export_sha256 is not None:
        row["routed_export_sha256"] = routed_export_sha256
    row = stamp_row_digest(row)
    return row


def validate_coverage_deferral_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one environment-coverage deferral row and return its canonical copy.

    Parameters
    ----------
    row:
        Candidate deferral row.

    Returns
    -------
    dict[str, Any]
        Validated row.

    Raises
    ------
    CoverageDeferralError
        If a required field is missing, mistyped, or the digest disagrees.
    """

    required = (
        "schema_version",
        "stable_id",
        "work_id",
        "name",
        "campaign_id",
        "run_id",
        "machine_id",
        "created_at",
        "disposition",
        "coverage",
        "recheck_hint",
        "row_sha256",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise CoverageDeferralError(
            f"environment-coverage deferral row is missing {', '.join(sorted(missing))}"
        )
    if row["schema_version"] != COVERAGE_DEFERRAL_SCHEMA_VERSION:
        raise CoverageDeferralError(
            f"environment-coverage deferral row declares unknown schema {row['schema_version']!r}"
        )
    if row["disposition"] != COVERAGE_DEFERRAL_DISPOSITION:
        raise CoverageDeferralError(
            f"environment-coverage deferral row declares unknown disposition {row['disposition']!r}"
        )
    for field in ("stable_id", "work_id", "name", "campaign_id", "run_id", "machine_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise CoverageDeferralError(f"environment-coverage deferral row has an empty {field}")
    coverage = row["coverage"]
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("verdict") != CoverageVerdict.UNSERVABLE.value
    ):
        raise CoverageDeferralError(
            "environment-coverage deferral row does not record an unservable routing"
        )
    library = coverage.get("library")
    if not isinstance(library, Mapping) or not isinstance(library.get("distribution"), str):
        raise CoverageDeferralError(
            "environment-coverage deferral row must name the distribution it could not serve"
        )
    if not isinstance(coverage.get("covering_intents"), list):
        raise CoverageDeferralError(
            "environment-coverage deferral row must state its covering intents, even when empty"
        )
    if not isinstance(row["recheck_hint"], str) or not row["recheck_hint"].strip():
        raise CoverageDeferralError(
            "environment-coverage deferral row must name what would cover the model later"
        )
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise CoverageDeferralError(
            f"coverage deferral row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise CoverageDeferralError(
            "environment-coverage deferral row digest does not match its payload"
        )
    validated["row_sha256"] = digest
    return validated


def append_coverage_deferral_row(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one environment-coverage deferral durably and idempotently.

    Parameters
    ----------
    path:
        Destination environment-coverage deferral JSONL.
    row:
        Complete typed deferral row.

    Returns
    -------
    dict[str, Any]
        Validated persisted row.

    Raises
    ------
    CoverageDeferralError
        If the row is malformed.

    Notes
    -----
    The FIRST deferral recorded for a work generation is the record, and a later
    run that re-derives it is a no-op -- the same rule
    ``host_capacity.append_capacity_deferral_row`` follows, and for the same
    reason: ``created_at`` and ``run_id`` necessarily differ on every resume, so
    treating a re-derived deferral as a conflict would abort the driver on the
    second pass over the same model. A model that has stopped being unservable
    writes nothing here, so a stale row can only ever be superseded by the model
    completing normally.
    """

    validated = validate_coverage_deferral_row(row)
    key = (validated["stable_id"], validated["work_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _decode_coverage_lines(_ledger_bytes(path, handle), path)
        matching = [item for item in existing if (item["stable_id"], item["work_id"]) == key]
        if matching:
            return matching[0]
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def _ledger_bytes(path: Path, handle: Optional[BinaryIO] = None) -> bytes:
    """Read the whole ledger, recovering only a torn final line with evidence.

    A complete malformed line stays this valve's typed hard refusal: the
    recovery's prefix check is this module's own strict decoder.
    """

    return read_with_tail_recovery(
        path,
        prefix_check=lambda retained: _decode_coverage_lines(retained.read_bytes(), path),
        locked_handle=handle,
    )


def _decode_coverage_lines(payload: bytes, path: Path) -> tuple[JsonObject, ...]:
    """Decode and validate every row in one deferral ledger's bytes."""

    rows: list[JsonObject] = []
    for index, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CoverageDeferralError(
                f"environment-coverage deferral ledger {path} has an unreadable row at line {index}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise CoverageDeferralError(
                f"environment-coverage deferral ledger {path} has a non-object row at line {index}"
            )
        rows.append(validate_coverage_deferral_row(decoded))
    return tuple(rows)


def load_coverage_deferral_rows(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Load environment-coverage deferrals from one or more campaign roots.

    Parameters
    ----------
    paths:
        Candidate ledger paths. Missing files are empty ledgers.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Rows ordered by stable ID then work ID.
    """

    by_key: dict[tuple[str, str], JsonObject] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in _decode_coverage_lines(_ledger_bytes(path), path):
            by_key[(row["stable_id"], row["work_id"])] = row
    return tuple(by_key[key] for key in sorted(by_key))
