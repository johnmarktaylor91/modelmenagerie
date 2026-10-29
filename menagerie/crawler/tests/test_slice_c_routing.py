"""Slice C framework routing and platform-evidence tests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from menagerie.crawler.constants import (
    EnvironmentPhase,
    PlatformRequirement,
    framework_from_markers,
)
from menagerie.crawler.envs import DEFAULT_ENVS_ROOT
from menagerie.crawler.effort import EffortCapExceeded, EffortTracker, StageCap
from menagerie.crawler.routing import (
    Arm64HeavyBuildAttempts,
    MissingPlatformEvidenceError,
    ModelRequirements,
    PlatformEvidence,
    defer_for_platform,
    phase_routes,
    requirements_from_intake_hints,
    requirements_from_zoo_era,
    route_model,
)


def test_torch_model_routes_by_declared_package_need() -> None:
    """A PyTorch graph dependency selects the graph intent in phase one."""

    route = route_model(
        ModelRequirements("m_graph", "pytorch", frozenset({"torch", "torch-geometric"}))
    )
    assert route.intent == "graph"
    assert route.phase is EnvironmentPhase.PYTORCH


@pytest.mark.parametrize(
    ("zoo", "era", "expected_intent"),
    (
        ("open-mmlab/mmdetection", None, "mmlab"),
        ("torch_geometric", None, "graph"),
        ("historical-classics", "2018", "legacy-torch"),
        ("discovered-pytorch", None, "oddballs"),
    ),
)
def test_zoo_era_table_populates_complete_router_requirements(
    zoo: str, era: str | None, expected_intent: str
) -> None:
    """Immutable zoo and era hints select the dependency-capable intent."""

    requirements = requirements_from_zoo_era(zoo, era)
    route = route_model(
        ModelRequirements(
            "m_intake",
            "pytorch",
            requirements.packages,
            requirements.exact_repository,
            requirements.legacy_torch,
        )
    )
    assert route.intent == expected_intent


def test_torch_geometric_temporal_rides_the_graph_intent_deliberately() -> None:
    """The TG-extension substring adjacency is a documented decision, not drift.

    ``torch_geometric_temporal`` (m2991's zoo) contains the ``torch_geometric``
    substring, so the contains-rule derives ``torch-geometric`` and routes the
    model to the graph intent -- the TG ecosystem environment (S2 batch-6
    verdict, the "substring adjacency"). That is the DELIBERATE outcome: the
    temporal extension rides its base ecosystem's intent, while its own
    distribution (``torch-geometric-temporal``, absent from the graph lock)
    stays subject to the coverage valve, which parks the model as
    ``deferred:needs-environment-coverage`` instead of running it into an
    import-error terminal. If this rule ever changes, the routing of the
    step-100 rows m1115/m2991 changes with it -- re-adjudicate, don't drift.
    """

    requirements = requirements_from_zoo_era("torch_geometric_temporal", None)
    assert requirements.packages == frozenset({"torch-geometric"})
    route = route_model(
        ModelRequirements(
            "m2991",
            "pytorch",
            requirements.packages,
            requirements.exact_repository,
            requirements.legacy_torch,
        )
    )
    assert route.intent == "graph"


def test_real_roster_routing_distribution_cannot_collapse_to_one_intent() -> None:
    """The complete launch roster exercises every shipped environment intent."""

    roster_path = Path(__file__).resolve().parents[2] / "data" / "crawl_roster.jsonl"
    rows = [
        json.loads(line)
        for line in roster_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts: Counter[str] = Counter()
    targeted: dict[str, Counter[str]] = {
        "open-mmlab/mmdetection": Counter(),
        "torch_geometric": Counter(),
    }
    for index, row in enumerate(rows):
        zoo = str(row["zoo"])
        requirements = requirements_from_intake_hints(str(row["name"]), zoo, row.get("era"))
        route = route_model(
            ModelRequirements(
                str(index),
                framework_from_markers(zoo, str(row["name"])),
                requirements.packages,
                requirements.exact_repository,
                requirements.legacy_torch,
            )
        )
        counts[route.intent] += 1
        if zoo in targeted:
            targeted[zoo][route.intent] += 1

    shipped_intents = {path.parent.name for path in DEFAULT_ENVS_ROOT.glob("*/environment.yml")}
    assert len(rows) == 28_482
    assert set(counts) == shipped_intents
    assert counts["core"] < len(rows) * 0.8
    assert max(counts.values()) < len(rows) * 0.8
    assert targeted["open-mmlab/mmdetection"] == {"mmlab": 151}
    assert targeted["torch_geometric"] == {"graph": 69}


@pytest.mark.parametrize(
    ("name", "zoo", "expected_intent"),
    (
        # The 2026-08-05 pilot rung: generic-zoo rows whose names carry the
        # library, previously stranded in `core` behind a missing import.
        ("DGL-TAGConv", "base-pytorch-compact", "graph"),
        ("recbole.NFM", "base-pytorch-compact", "recsys"),
        ("recbole.LDiffRec", "base-pytorch-compact", "recsys"),
        ("d3rlpy_IQL", "base-pytorch-compact", "recsys"),
        ("bindsnet_diehlandcook2015v2", "base-pytorch-compact", "recsys"),
        ("detectron2_fast_rcnn_R_50_FPN_1x", "base-pytorch-compact", "detect-misc"),
        # Library-named zoos that previously matched no table row at all.
        ("bindsnet_diehl_cook_network", "bindsnet", "recsys"),
        ("ultralytics_yolov8n_det", "ultralytics", "detect-misc"),
        ("detectron2_mask_rcnn", "detectron2.model_zoo", "detect-misc"),
        # zuko lives in core (noarch, solver-proven no pin movement).
        ("zuko_MAF", "zuko", "core"),
        ("zuko_gf", "base-pytorch-compact", "core"),
        # tsai's and pytorch-tabnet's conda builds pin pytorch below the
        # tabular-ts floor; both live in the legacy stack.
        ("tsai_InceptionTime", "tsai", "legacy-torch"),
    ),
)
def test_library_named_models_reach_the_intent_that_carries_their_library(
    name: str, zoo: str, expected_intent: str
) -> None:
    """Name and zoo library markers route to a covering environment, not core."""

    requirements = requirements_from_intake_hints(name, zoo)
    route = route_model(
        ModelRequirements(
            "m_named",
            "pytorch",
            requirements.packages,
            requirements.exact_repository,
            requirements.legacy_torch,
        )
    )
    assert route.intent == expected_intent


def test_name_markers_never_reroute_a_hand_built_classic() -> None:
    """A classics-zoo row keeps its pinned-repository base despite a library name.

    ``bindsnet_ifnodes`` and ``recbole.XGBoost`` under
    ``unregistered-classics-pytorch`` are self-contained faithful ports; the name
    is a homage, not a dependency, and rerouting them to the library stack would
    strand them without their build toolchain.
    """

    for name in ("bindsnet_ifnodes", "recbole.XGBoost", "dgl-classic-port"):
        requirements = requirements_from_intake_hints(name, "unregistered-classics-pytorch")
        assert requirements.exact_repository
        assert not requirements.packages
        route = route_model(
            ModelRequirements(
                "m_classic",
                "pytorch",
                requirements.packages,
                requirements.exact_repository,
                requirements.legacy_torch,
            )
        )
        assert route.intent == "oddballs"


def test_name_markers_are_anchored_prefixes_not_substrings() -> None:
    """An incidental substring can never add a library package fact."""

    for name in ("edge-dgl-lite", "my_recbole_wrapper", "not_d3rlpy"):
        assert requirements_from_intake_hints(name, "base-pytorch-compact").packages == frozenset()


@pytest.mark.parametrize(
    ("framework", "intent"),
    (("tensorflow", "tf-keras-arm64"), ("jax", "jax-flax-arm64"), ("paddle", "paddle-arm64")),
)
def test_native_frameworks_route_to_phase_two(framework: str, intent: str) -> None:
    """Native TensorFlow, JAX, and Paddle models remain in the ordered tail."""

    route = route_model(ModelRequirements(f"m_{framework}", framework))
    assert route.intent == intent
    assert route.phase is EnvironmentPhase.NATIVE_TAIL
    ordered = phase_routes((route, route_model(ModelRequirements("m_torch", "torch"))))
    assert [item.phase for item in ordered] == [
        EnvironmentPhase.PYTORCH,
        EnvironmentPhase.NATIVE_TAIL,
    ]


def test_evidenced_cuda_failure_defers_instead_of_failing() -> None:
    """Literal focused-probe CUDA evidence supports the closed deferred status."""

    evidence = PlatformEvidence(
        PlatformRequirement.CUDA,
        "focused-probe",
        "extension load reports that CUDA_HOME is required",
    )
    decision = defer_for_platform("m_cuda", PlatformRequirement.CUDA, (evidence,))
    assert decision.status == "deferred:needs-cuda"
    assert decision.evidence == (evidence,)


@pytest.mark.parametrize(
    "zoo",
    (
        # Bare `owner/repo` -- the roster's most common repository spelling, and the form
        # the prefix-only table missed entirely.
        "facebookresearch/esm",
        "NVlabs/RADIO",
        "thuml/Time-Series-Library",
        # Forge markers anywhere, not only as a prefix.
        "github:tobran/DF-GAN",
        "github(ShoufaChen/DiffusionDet)",
        "gnobitab/RectifiedFlow (github)",
        "github arcinstitute/evo2 (web)",
        "torch.hub:intel-isl/MiDaS",
        # Explicit repository-checkout markers.
        "dreamer-torch(repo)",
        "act(repo,tonyzhaozh)",
        "nanodet-src",
        "chosj95/MIMO-UNet source",
        # Paper-only rows, which also have no packaged home.
        "arxiv",
        "arxiv:2606.05116",
        "paper/arxiv:2502.05171",
    ),
)
def test_repository_shaped_zoo_reaches_the_exact_repository_environment(zoo: str) -> None:
    """Every shipped repository spelling routes to the pinned-repository base."""

    requirements = requirements_from_zoo_era(zoo)
    assert requirements.exact_repository
    route = route_model(
        ModelRequirements(
            "m_repo",
            "pytorch",
            requirements.packages,
            requirements.exact_repository,
            requirements.legacy_torch,
        )
    )
    assert route.intent == "oddballs"


@pytest.mark.parametrize(
    "zoo",
    (
        # Harvest-provenance pseudo-paths. These name where a human looked, not a
        # repository, so promoting them would strand a library model in a base env.
        "ocr/document",
        "torch-reference/from-scratch",
        "reference/missing-package",
        "nvidia-nemo/web",
        "deeplab-public-ver2/web",
        # Ordinary library zoos must stay library zoos.
        "timm",
        "huggingface_transformers",
        "base-pytorch-compact",
        "web",
        "source",
    ),
)
def test_non_repository_zoo_is_not_promoted_to_exact_repository(zoo: str) -> None:
    """A lookup-context or library label never counts as a pinned repository."""

    assert not requirements_from_zoo_era(zoo).exact_repository


def test_repository_shape_never_overrides_a_declared_package_stack() -> None:
    """A repo-spelled ecosystem zoo carries both facts and keeps its package intent."""

    requirements = requirements_from_zoo_era("open-mmlab/mmdetection")

    assert requirements.exact_repository
    assert requirements.packages == frozenset({"mmcv", "mmengine"})
    assert (
        route_model(
            ModelRequirements(
                "m_mm",
                "pytorch",
                requirements.packages,
                requirements.exact_repository,
                requirements.legacy_torch,
            )
        ).intent
        == "mmlab"
    )


def test_c1_repository_detection_is_not_stuck_at_the_prefix_only_rate() -> None:
    """The residual c1 tier's repository rows must not collapse back to 231 rows.

    The prefix-only table flagged 231 of c1-mech's 6,968 rows as ``exact_repository``
    while roughly a thousand are repository-shaped, so every one of the rest was
    provisioned the general library environment instead of the pinned-repository base.
    This is the regression guard on that fix; it is deliberately a floor, not an exact
    count, so a later honest harvest can add repository rows without editing a test.
    """

    partition = Path(__file__).resolve().parents[1] / "records" / "partitions" / "c1-mech.jsonl"
    rows = [
        json.loads(line)
        for line in partition.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    detected = sum(
        1
        for row in rows
        if requirements_from_zoo_era(str(row["zoo"]), row.get("era")).exact_repository
    )

    assert len(rows) == 6_968
    assert detected > 900


def test_detectron2_mmcv_pool_gets_exactly_two_arm64_attempts() -> None:
    """The first x86 build failure retries and the second creates a deferral."""

    tracker = EffortTracker({"environment": StageCap(attempts=2)})
    attempts = Arm64HeavyBuildAttempts(tracker)
    evidence = PlatformEvidence(
        PlatformRequirement.X86,
        "focused-probe",
        "compiler rejects the arm64 target in the same detectron2 source build",
    )
    assert attempts.record_failure("m_detect", evidence) is None
    decision = attempts.record_failure("m_detect", evidence)
    assert decision is not None
    assert decision.status == "deferred:needs-x86"
    assert decision.attempts == 2
    assert attempts.attempts("m_detect") == 2
    with pytest.raises(EffortCapExceeded):
        attempts.record_failure("m_detect", evidence)
    assert tracker.failures[-1].metric == "attempts"


def test_deferral_without_matching_platform_evidence_is_refused() -> None:
    """A static package name cannot speculate a platform deferral."""

    with pytest.raises(MissingPlatformEvidenceError):
        defer_for_platform("m_guess", PlatformRequirement.CUDA, ())
    x86_evidence = PlatformEvidence(
        PlatformRequirement.X86, "source", "upstream documents x86-only assembly"
    )
    with pytest.raises(MissingPlatformEvidenceError):
        defer_for_platform("m_guess", PlatformRequirement.CUDA, (x86_evidence,))
