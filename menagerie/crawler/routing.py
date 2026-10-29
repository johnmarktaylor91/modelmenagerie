"""Dependency-aware intent routing and evidenced arm64 platform deferral."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Literal, Mapping, Optional

from menagerie.crawler.backend import (
    FRAMEWORK_CAPABILITIES,
    Framework,
    FrameworkVocabularyError,
    LaneStatus,
    framework_capability,
    normalize_framework,
)
from menagerie.crawler.constants import (
    BACKEND_PARITY_DEFERRAL_STATUS_CODE,
    EnvironmentPhase,
    PlatformRequirement,
)
from menagerie.crawler.effort import EffortTracker

#: Native environment intents, DERIVED from the one framework capability registry.
#:
#: This map used to be an independent literal, and it was one of the six drifting closed
#: framework vocabularies AREA_BACKENDS_opus B1 tabulated: it accepted ``"keras"`` while
#: ``standard_inputs._convert_array`` raised on it and ``worker._seed_frameworks`` silently
#: seeded nothing for it. Deriving the map means an alias cannot be routable and
#: unexecutable at the same time.
_NATIVE_INTENTS = {
    alias: capability.env_intent
    for capability in FRAMEWORK_CAPABILITIES.values()
    for alias in capability.aliases
    if capability.env_intent is not None
    and capability.lane_phase is EnvironmentPhase.NATIVE_TAIL
    and capability.lane_status is LaneStatus.EXECUTABLE
}

_PACKAGE_INTENTS: tuple[tuple[str, frozenset[str]], ...] = (
    ("detect-misc", frozenset({"detectron2", "mmdet", "ultralytics"})),
    ("mmlab", frozenset({"mmengine", "mmcv", "mmpretrain", "mmsegmentation"})),
    ("graph", frozenset({"torch-geometric", "torch_geometric", "dgl", "ogb", "pykeen"})),
    # speech precedes audio: speechbrain's supported torchaudio range cannot
    # ride the audio stack (see the registry split guidance), so its models
    # route to the bounded speech intent.
    ("speech", frozenset({"speechbrain"})),
    ("audio", frozenset({"torchaudio", "librosa"})),
    (
        "tabular-ts",
        frozenset({"pytorch-forecasting", "sktime"}),
    ),
    ("recsys", frozenset({"recbole", "d3rlpy", "bindsnet"})),
    # tsai's and pytorch-tabnet's only conda builds pin pytorch below the
    # tabular-ts >=2.3 floor, which the legacy stack fits (see the registry
    # split guidance for the solver evidence).
    ("legacy-torch", frozenset({"tsai", "pytorch-tabnet"})),
)

_ZooMatchKind = Literal["exact", "prefix", "contains", "pattern"]


@dataclass(frozen=True)
class ZooRequirementsRule:
    """One auditable zoo/era rule used to derive intake routing requirements.

    ``match="pattern"`` holds anchored regular expressions matched against the whole
    normalized zoo. It exists because repository-shaped zoo strings are a *form*, not a
    vocabulary: ``facebookresearch/esm``, ``dreamer-torch(repo)``, and ``nanodet-src`` name
    concrete repositories with no shared literal to key on, and enumerating them as exact
    rows would silently go stale on the next harvest.
    """

    match: _ZooMatchKind
    zoos: tuple[str, ...]
    packages: frozenset[str] = frozenset()
    exact_repository: bool = False
    legacy_before_year: Optional[int] = None


@dataclass(frozen=True)
class IntakeRoutingRequirements:
    """Dependency facts deterministically derived from one immutable roster row."""

    packages: frozenset[str] = frozenset()
    exact_repository: bool = False
    legacy_torch: bool = False


# Rules are normalized, ordered for audit readability, and cumulative. A zoo may
# legitimately require both an ecosystem package and exact-repository handling.
ZOO_ERA_REQUIREMENTS_TABLE: tuple[ZooRequirementsRule, ...] = (
    ZooRequirementsRule(
        match="exact",
        zoos=(
            "discovered-pytorch",
            "historical-classics",
            "unregistered-classics-pytorch",
        ),
        exact_repository=True,
    ),
    # Forge markers are matched anywhere in the string, not as a prefix. The roster
    # spells the same fact as `github:owner/name`, `github(owner/name)`,
    # `owner/name (github)`, and `github arcinstitute/evo2 (web)`; a prefix rule saw
    # only the first spelling.
    ZooRequirementsRule(
        match="contains",
        zoos=("github", "gitlab", "torch.hub", "torchhub"),
        exact_repository=True,
    ),
    # Bare `owner/repo`, the single most common repository spelling in the roster and
    # previously undetected entirely. The negative lookahead drops harvest-provenance
    # pseudo-paths (`ocr/document`, `torch-reference/from-scratch`, `nvidia-nemo/web`),
    # which name a lookup context rather than a repository.
    ZooRequirementsRule(
        match="pattern",
        zoos=(
            r"[a-z0-9][a-z0-9._-]*/"
            r"(?!(?:web|blog|arxiv|docs?|papers?|document|from-scratch|missing-package)$)"
            r"[a-z0-9][a-z0-9._-]*",
        ),
        exact_repository=True,
    ),
    # Explicit "the source is a repository checkout" markers: `dreamer-torch(repo)`,
    # `act(repo,tonyzhaozh)`, `nanodet-src`, `chosj95/MIMO-UNet source`.
    ZooRequirementsRule(
        match="contains",
        zoos=("(repo)", "(repo,"),
        exact_repository=True,
    ),
    ZooRequirementsRule(
        match="pattern",
        zoos=(r".*(?:-src|-source| source)",),
        exact_repository=True,
    ),
    # Paper-only rows have no packaged home either, and the same minimal base plus build
    # toolchain is the environment a faithful port or reimplementation needs.
    ZooRequirementsRule(
        match="pattern",
        zoos=(r"(?:paper/)?arxiv(?:[:(].*)?",),
        exact_repository=True,
    ),
    ZooRequirementsRule(
        match="contains",
        zoos=(
            "open-mmlab",
            "mmagic",
            "mmdet",
            "mmlab",
            "mmocr",
            "mmpose",
            "mmpretrain",
            "mmseg",
        ),
        packages=frozenset({"mmengine", "mmcv"}),
    ),
    ZooRequirementsRule(
        match="contains",
        zoos=("detectron2",),
        packages=frozenset({"detectron2"}),
    ),
    # DELIBERATE substring adjacency (S2 batch-6 verdict): the contains rule
    # also matches TG-ecosystem EXTENSIONS such as ``torch_geometric_temporal``
    # (m2991), deriving ``torch-geometric`` and routing them to the graph
    # intent -- their base ecosystem's environment. The extension's own
    # distribution (``torch-geometric-temporal``, absent from the graph lock)
    # stays subject to the coverage valve, which parks the model as
    # ``deferred:needs-environment-coverage`` rather than running it into an
    # import-error terminal. Pinned by
    # test_torch_geometric_temporal_rides_the_graph_intent_deliberately.
    ZooRequirementsRule(
        match="contains",
        zoos=("pytorch-geometric", "torch-geometric", "torch_geometric"),
        packages=frozenset({"torch-geometric"}),
    ),
    ZooRequirementsRule(
        match="contains",
        zoos=("dgl",),
        packages=frozenset({"dgl"}),
    ),
    ZooRequirementsRule(
        match="exact",
        zoos=("pykeen",),
        packages=frozenset({"pykeen"}),
    ),
    ZooRequirementsRule(
        match="exact",
        zoos=("e3nn",),
        packages=frozenset({"e3nn"}),
    ),
    ZooRequirementsRule(
        match="exact",
        zoos=("ogb",),
        packages=frozenset({"ogb"}),
    ),
    ZooRequirementsRule(
        match="prefix",
        zoos=("speechbrain",),
        packages=frozenset({"speechbrain"}),
    ),
    ZooRequirementsRule(
        match="prefix",
        zoos=("torchaudio",),
        packages=frozenset({"torchaudio"}),
    ),
    ZooRequirementsRule(
        match="exact",
        zoos=("pypi:pytorch-forecasting",),
        packages=frozenset({"pytorch-forecasting"}),
    ),
    ZooRequirementsRule(
        match="exact",
        zoos=("bindsnet",),
        packages=frozenset({"bindsnet"}),
    ),
    ZooRequirementsRule(
        match="exact",
        zoos=("d3rlpy",),
        packages=frozenset({"d3rlpy"}),
    ),
    ZooRequirementsRule(
        match="exact",
        zoos=("recbole",),
        packages=frozenset({"recbole"}),
    ),
    ZooRequirementsRule(
        match="exact",
        zoos=("zuko",),
        packages=frozenset({"zuko"}),
    ),
    # Ultralytics is spelled both bare (`ultralytics`) and repo-shaped
    # (`ultralytics/ultralytics`); a contains rule covers both, and overmatching
    # only adds a package fact for the intent that actually carries the library.
    ZooRequirementsRule(
        match="contains",
        zoos=("ultralytics",),
        packages=frozenset({"ultralytics"}),
    ),
    ZooRequirementsRule(
        match="exact",
        zoos=("tsai",),
        packages=frozenset({"tsai"}),
    ),
    ZooRequirementsRule(
        match="exact",
        zoos=("historical-classics",),
        legacy_before_year=2019,
    ),
)


@dataclass(frozen=True)
class NameRequirementsRule:
    """One auditable model-name rule used to derive intake routing requirements.

    The zoo is the primary routing authority, but a large legacy tier spells a
    generic zoo (``base-pytorch-compact``) and names the concrete library only in
    the MODEL NAME: ``DGL-TAGConv``, ``recbole.NFM``, ``d3rlpy_IQL``,
    ``bindsnet_diehlandcook2015v2``, ``detectron2_fast_rcnn_R_50_FPN_1x``. Those
    rows previously fell through to ``core``, which deliberately carries none of
    these libraries, and every one of them terminalized behind a missing import
    (the 2026-08-05 pilot rung recorded them as
    ``deferred:needs-environment-coverage``). Prefixes are anchored at the start
    of the normalized name, so an incidental substring can never reroute a model.
    """

    prefixes: tuple[str, ...]
    packages: frozenset[str]


# Name rules are consulted only for rows whose zoo is NOT repository-shaped:
# hand-built classics (`unregistered-classics-pytorch`) legitimately reuse
# library naming conventions (`bindsnet_ifnodes`, `recbole.XGBoost`) while being
# self-contained ports that belong in the pinned-repository base environment.
NAME_REQUIREMENTS_TABLE: tuple[NameRequirementsRule, ...] = (
    NameRequirementsRule(prefixes=("dgl-", "dgl_", "dgl."), packages=frozenset({"dgl"})),
    NameRequirementsRule(
        prefixes=("recbole.", "recbole_", "recbole-"), packages=frozenset({"recbole"})
    ),
    NameRequirementsRule(
        prefixes=("d3rlpy.", "d3rlpy_", "d3rlpy-"), packages=frozenset({"d3rlpy"})
    ),
    NameRequirementsRule(
        prefixes=("bindsnet.", "bindsnet_", "bindsnet-"), packages=frozenset({"bindsnet"})
    ),
    NameRequirementsRule(
        prefixes=("detectron2.", "detectron2_", "detectron2-"),
        packages=frozenset({"detectron2"}),
    ),
    NameRequirementsRule(prefixes=("zuko.", "zuko_", "zuko-"), packages=frozenset({"zuko"})),
    NameRequirementsRule(
        prefixes=("pykeen.", "pykeen_", "pykeen-"), packages=frozenset({"pykeen"})
    ),
    NameRequirementsRule(
        prefixes=("speechbrain.", "speechbrain_", "speechbrain-"),
        packages=frozenset({"speechbrain"}),
    ),
    NameRequirementsRule(
        prefixes=("ultralytics.", "ultralytics_", "ultralytics-"),
        packages=frozenset({"ultralytics"}),
    ),
)


def requirements_from_intake_hints(
    name: str, zoo: str, era: Optional[str] = None
) -> IntakeRoutingRequirements:
    """Derive routing requirements from every immutable intake hint.

    This is the intake-facing derivation: the zoo/era table facts plus the
    conservative model-name library markers. Name markers are additive package
    facts only -- they never flip ``exact_repository`` or ``legacy_torch`` --
    and they are skipped entirely for repository-shaped zoos, where the name is
    a homage rather than a dependency (a hand-built ``bindsnet_ifnodes`` classic
    must stay in the pinned-repository base, not be rerouted to the library
    stack).

    Parameters
    ----------
    name:
        Immutable roster model name.
    zoo:
        Immutable roster zoo identifier.
    era:
        Optional roster era; only an explicit four-digit year can activate a
        legacy rule.

    Returns
    -------
    IntakeRoutingRequirements
        Cumulative package, repository, and legacy-routing facts.
    """

    base = requirements_from_zoo_era(zoo, era)
    if base.exact_repository:
        return base
    normalized_name = name.strip().lower()
    packages = set(base.packages)
    for rule in NAME_REQUIREMENTS_TABLE:
        if normalized_name.startswith(rule.prefixes):
            packages.update(rule.packages)
    if packages == set(base.packages):
        return base
    return IntakeRoutingRequirements(
        packages=frozenset(packages),
        exact_repository=base.exact_repository,
        legacy_torch=base.legacy_torch,
    )


class RoutingError(ValueError):
    """Base class for invalid or unsupported routing decisions."""


class MissingPlatformEvidenceError(RoutingError):
    """Raised when code attempts a speculative CUDA/x86 deferral."""


class BackendParityUnavailableError(RoutingError):
    """Raised when a model's framework has no environment lane in this campaign.

    A distinct type from the generic :class:`RoutingError` on purpose. The generic error
    means "this string is not a framework"; this one means "this IS a framework, we simply
    cannot execute it here" -- an honest machinery gap whose terminal is
    ``deferred:needs-backend-parity``, never ``failed:forward``. Callers branch on the type
    rather than on the message.
    """

    def __init__(self, framework: Framework, detail: str) -> None:
        """Carry the framework and its typed no-lane reason on the exception itself."""

        self.framework = framework
        self.detail = detail
        self.status = BACKEND_PARITY_DEFERRAL_STATUS_CODE
        super().__init__(
            f"{framework.value} has no environment lane ({detail}); "
            f"the honest terminal is {BACKEND_PARITY_DEFERRAL_STATUS_CODE}"
        )


@dataclass(frozen=True)
class ModelRequirements:
    """Framework and direct package requirements used for deterministic routing."""

    stable_id: str
    framework: str
    packages: frozenset[str] = frozenset()
    exact_repository: bool = False
    legacy_torch: bool = False


def requirements_from_zoo_era(zoo: str, era: Optional[str] = None) -> IntakeRoutingRequirements:
    """Derive deterministic dependency requirements from immutable intake hints.

    Parameters
    ----------
    zoo:
        Immutable roster zoo identifier.
    era:
        Optional roster era. Only an explicit four-digit year can activate a
        legacy rule.

    Returns
    -------
    IntakeRoutingRequirements
        Cumulative package, repository, and legacy-routing facts.
    """

    normalized_zoo = zoo.strip().lower()
    years = tuple(int(match) for match in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", era or ""))
    earliest_year = min(years) if years else None
    packages: set[str] = set()
    exact_repository = False
    legacy_torch = False
    for rule in ZOO_ERA_REQUIREMENTS_TABLE:
        if not _zoo_rule_matches(normalized_zoo, rule):
            continue
        packages.update(rule.packages)
        exact_repository = exact_repository or rule.exact_repository
        legacy_torch = legacy_torch or (
            rule.legacy_before_year is not None
            and earliest_year is not None
            and earliest_year < rule.legacy_before_year
        )
    return IntakeRoutingRequirements(
        packages=frozenset(packages),
        exact_repository=exact_repository,
        legacy_torch=legacy_torch,
    )


def _zoo_rule_matches(zoo: str, rule: ZooRequirementsRule) -> bool:
    """Return whether one normalized zoo matches an auditable table rule.

    Parameters
    ----------
    zoo:
        Lowercase normalized zoo identifier.
    rule:
        Exact, prefix, or substring table row.

    Returns
    -------
    bool
        Whether the table row applies.
    """

    if rule.match == "exact":
        return zoo in rule.zoos
    if rule.match == "prefix":
        return zoo.startswith(rule.zoos)
    if rule.match == "pattern":
        return any(_compiled(pattern).fullmatch(zoo) is not None for pattern in rule.zoos)
    return any(marker in zoo for marker in rule.zoos)


@lru_cache(maxsize=None)
def _compiled(pattern: str) -> re.Pattern[str]:
    """Return one compiled table pattern, cached because the table is frozen."""

    return re.compile(pattern)


@dataclass(frozen=True)
class IntentRoute:
    """Selected environment intent and ordered campaign phase."""

    stable_id: str
    intent: str
    phase: EnvironmentPhase


@dataclass(frozen=True)
class PlatformEvidence:
    """Captured source/probe/runtime evidence for one platform requirement."""

    requirement: PlatformRequirement
    source: str
    detail: str
    package_identity: Optional[str] = None
    source_identity: Optional[str] = None

    def __post_init__(self) -> None:
        """Require literal, attributable evidence rather than a static hint."""

        if self.source not in {"source", "focused-probe", "runtime"}:
            raise ValueError("platform evidence source must be source, focused-probe, or runtime")
        if not self.detail.strip():
            raise ValueError("platform evidence detail must be non-empty")


@dataclass(frozen=True)
class DeferralDecision:
    """Closed deferred status backed by captured platform evidence."""

    stable_id: str
    status: str
    evidence: tuple[PlatformEvidence, ...]
    attempts: int


@dataclass
class PlatformEvidenceStore:
    """Learn evidence only for matching exact package/source identities."""

    _by_model: dict[str, list[PlatformEvidence]] = field(default_factory=dict)
    _by_identity: dict[str, list[PlatformEvidence]] = field(default_factory=dict)

    def collect(self, stable_id: str, evidence: PlatformEvidence) -> None:
        """Store one observed evidence item for a model and exact identities."""

        self._by_model.setdefault(stable_id, []).append(evidence)
        for identity in (evidence.package_identity, evidence.source_identity):
            if identity is not None:
                self._by_identity.setdefault(identity, []).append(evidence)

    def for_model(
        self, stable_id: str, *, identities: Iterable[str] = ()
    ) -> tuple[PlatformEvidence, ...]:
        """Return model-local plus matching exact-identity evidence."""

        collected = list(self._by_model.get(stable_id, ()))
        for identity in identities:
            collected.extend(self._by_identity.get(identity, ()))
        return tuple(dict.fromkeys(collected))


def route_model(model: ModelRequirements) -> IntentRoute:
    """Assign a model to an intent using framework and declared packages.

    Parameters
    ----------
    model:
        Model routing requirements.

    Returns
    -------
    IntentRoute
        Deterministic intent and phase.
    """

    framework = model.framework.strip().lower()
    native = _NATIVE_INTENTS.get(framework)
    if native is not None:
        return IntentRoute(model.stable_id, native, EnvironmentPhase.NATIVE_TAIL)
    try:
        resolved = normalize_framework(framework)
    except FrameworkVocabularyError as exc:
        raise RoutingError(f"unsupported declared framework: {model.framework!r}") from exc
    if resolved is not Framework.TORCH:
        # A real framework with no lane. Refusing typed here is what keeps the honest
        # outcome (deferred:needs-backend-parity) reachable: the shipped code sent every
        # unrecognised framework into the torch author lane, where the only available
        # outcomes were a torch reimplementation or a mis-typed model defect.
        capability = framework_capability(resolved)
        raise BackendParityUnavailableError(resolved, capability.no_lane_reason)
    normalized = frozenset(package.lower() for package in model.packages)
    for intent, markers in _PACKAGE_INTENTS:
        if normalized & markers:
            return IntentRoute(model.stable_id, intent, EnvironmentPhase.PYTORCH)
    if model.legacy_torch:
        intent = "legacy-torch"
    elif model.exact_repository:
        intent = "oddballs"
    else:
        intent = "core"
    return IntentRoute(model.stable_id, intent, EnvironmentPhase.PYTORCH)


def phase_routes(routes: Iterable[IntentRoute]) -> tuple[IntentRoute, ...]:
    """Order all PyTorch routes before the native-framework tail."""

    order = {EnvironmentPhase.PYTORCH: 0, EnvironmentPhase.NATIVE_TAIL: 1}
    return tuple(
        sorted(routes, key=lambda route: (order[route.phase], route.intent, route.stable_id))
    )


def defer_for_platform(
    stable_id: str,
    requirement: PlatformRequirement | str,
    evidence: Iterable[PlatformEvidence],
    *,
    attempts: int = 0,
) -> DeferralDecision:
    """Create a CUDA/x86 deferral only from matching captured evidence."""

    required = PlatformRequirement(requirement)
    if required is PlatformRequirement.BACKEND_PARITY:
        # One mint point, because this capability's proof is a re-derivation rather than a
        # captured observation: `backend_parity_deferral` checks the framework really has
        # no lane. Passing evidence in here would let a caller assert the conclusion.
        raise MissingPlatformEvidenceError(
            "backend parity is minted by routing.backend_parity_deferral, whose proof is "
            "re-derived from the framework capability registry"
        )
    matching = tuple(item for item in evidence if item.requirement is required)
    if not matching:
        raise MissingPlatformEvidenceError(
            f"cannot defer {stable_id} for {required.value} without captured evidence"
        )
    return DeferralDecision(
        stable_id=stable_id,
        status=f"deferred:needs-{required.value}",
        evidence=matching,
        attempts=attempts,
    )


def backend_parity_deferral(
    stable_id: str,
    framework: Framework | str,
    *,
    detail: str,
    attempts: int = 0,
) -> DeferralDecision:
    """Mint the backend-parity deferral for a framework this campaign cannot execute.

    The proof is MACHINE-derived and re-derived on every call: the framework's own
    capability row must actually say it has no lane. A caller cannot mint this terminal for
    a framework that is executable, and cannot mint it without a literal reason -- the two
    ways a capability deferral becomes a fabricated proof.

    Parameters
    ----------
    stable_id:
        Model whose work is being deferred.
    framework:
        Canonical framework or any accepted spelling.
    detail:
        Literal, attributable reason. Never empty.
    attempts:
        Attempts already spent on this model.

    Returns
    -------
    DeferralDecision
        Deferral carrying ``deferred:needs-backend-parity`` and its evidence.

    Raises
    ------
    ValueError
        When the detail is empty, or when the named framework HAS a lane -- in which case
        the honest outcome is a real failure, not a deferral.
    """

    resolved = normalize_framework(framework)
    capability = framework_capability(resolved)
    if not detail.strip():
        raise ValueError(f"cannot defer {stable_id} for backend parity without a literal reason")
    if capability.lane_status is LaneStatus.EXECUTABLE:
        raise ValueError(
            f"{resolved.value} is executable in this campaign; a backend-parity deferral "
            "would be a fabricated proof"
        )
    evidence = PlatformEvidence(
        requirement=PlatformRequirement.BACKEND_PARITY,
        source="source",
        detail=f"{capability.no_lane_reason}; {detail}",
    )
    return DeferralDecision(
        stable_id=stable_id,
        status=BACKEND_PARITY_DEFERRAL_STATUS_CODE,
        evidence=(evidence,),
        attempts=attempts,
    )


class Arm64HeavyBuildAttempts:
    """Bound detectron2/mmcv CPU source builds to two recorded attempts."""

    def __init__(self, effort: EffortTracker) -> None:
        """Use the shared effort tracker for environment attempt accounting."""

        self._effort = effort
        self._attempts: dict[str, int] = {}

    def record_failure(
        self, stable_id: str, evidence: PlatformEvidence
    ) -> Optional[DeferralDecision]:
        """Record one heavy-build failure and defer after the second x86 failure.

        Parameters
        ----------
        stable_id, evidence:
            Affected model and evidence captured by its build attempt.

        Returns
        -------
        DeferralDecision or None
            Deferral on the second attempt, otherwise ``None``.
        """

        self._effort.consume("environment", attempts=1)
        count = self._attempts.get(stable_id, 0) + 1
        self._attempts[stable_id] = count
        if evidence.requirement is PlatformRequirement.CUDA:
            return defer_for_platform(stable_id, evidence.requirement, (evidence,), attempts=count)
        if count == 2:
            return defer_for_platform(
                stable_id, PlatformRequirement.X86, (evidence,), attempts=count
            )
        return None

    def attempts(self, stable_id: str) -> int:
        """Return the number of recorded heavy-build attempts for a model."""

        return self._attempts.get(stable_id, 0)


def intent_capabilities() -> Mapping[str, frozenset[str]]:
    """Return package markers used by the deterministic router."""

    return {intent: packages for intent, packages in _PACKAGE_INTENTS}
