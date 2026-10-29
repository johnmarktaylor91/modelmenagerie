"""BL-13's end-to-end tripwire: a misrouted model defers, and completed work survives.

The BACKEND MISROUTE TRIAD is three defects that compose into a pipeline for destroying
finished work inside a torch-only rung:

1. **B6/F6** -- the lane is chosen by a substring match on the roster row's zoo+name,
   before anyone reads the source, with a measured 13% false-positive rate, and everything
   unrecognised is silently called "PyTorch".
2. **BK-1** -- the award chain's dtype normalizer strips only the ``torch.`` prefix, so a
   completed ``paddle.float32`` / ``<dtype: 'float32'>`` forward fails the
   contract-signature conjunct and is published as an evidence-free ``mode-run`` failure
   (reproduced in the shipped ledger: m7960, four attempts, forward demonstrably fine).
3. **B7** -- ``deferred:needs-backend-parity`` did not exist, so the honest outcome
   "this backend has no lane here" had to land as ``failed:forward``: a machinery gap
   published as a permanent model defect that no backend lift can mechanically recover.

Each test below is anchored to live evidence from the pilot, and the dtype goldens carry
the exact per-framework spellings from the finding's table.
"""

from __future__ import annotations

import pytest

from menagerie.crawler.backend import (
    NATIVE_LANE_DISPOSITIONS,
    Framework,
    LaneStatus,
    NativeLaneUndisclosedError,
    framework_capability,
    reconcile_execution_framework,
    require_native_lane_disposition,
)
from menagerie.crawler.constants import (
    BACKEND_PARITY_DEFERRAL_STATUS_CODE,
    TERMINAL_STATUS_CODES,
    framework_from_markers,
)
from menagerie.crawler.modes import output_signature
from menagerie.crawler.payload_codec import (
    CanonicalDtype,
    DtypeVocabularyError,
    canonical_dtype,
    canonical_dtype_text,
    dtype_matches,
)
from menagerie.crawler.routing import (
    ModelRequirements,
    RoutingError,
    backend_parity_deferral,
    route_model,
)


class _FakeTensor:
    """A framework tensor stand-in whose ``dtype`` reprs exactly like the real one."""

    def __init__(self, dtype_text: str, shape: tuple[int, ...] = (2, 3, 224, 224)) -> None:
        self._dtype_text = dtype_text
        self.shape = shape

    @property
    def dtype(self) -> object:
        return _FakeDtype(self._dtype_text)


class _FakeDtype:
    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


# The exact per-framework observed spellings from AREA_BACKENDS_fable BK-1's table.
_FRAMEWORK_DTYPE_SPELLINGS = (
    ("torch", "torch.float32", CanonicalDtype.FLOAT32),
    ("numpy/jax", "float32", CanonicalDtype.FLOAT32),
    ("paddle", "paddle.float32", CanonicalDtype.FLOAT32),
    ("tensorflow", "<dtype: 'float32'>", CanonicalDtype.FLOAT32),
    ("mlx", "mlx.core.float32", CanonicalDtype.FLOAT32),
    ("torch-bf16", "torch.bfloat16", CanonicalDtype.BFLOAT16),
    ("tensorflow-bf16", "<dtype: 'bfloat16'>", CanonicalDtype.BFLOAT16),
    ("torch-alias", "torch.long", CanonicalDtype.INT64),
    ("numpy-char", "<f4", CanonicalDtype.FLOAT32),
    ("paddle-int", "paddle.int64", CanonicalDtype.INT64),
)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("label", "spelling", "expected"),
    _FRAMEWORK_DTYPE_SPELLINGS,
    ids=[row[0] for row in _FRAMEWORK_DTYPE_SPELLINGS],
)
def test_every_framework_dtype_spelling_canonicalizes(
    label: str, spelling: str, expected: CanonicalDtype
) -> None:
    """ONE dtype vocabulary: the cross-framework golden parity test BK-1 asks for."""

    assert canonical_dtype(spelling) is expected


@pytest.mark.smoke
def test_a_completed_non_torch_forward_matches_its_authored_contract() -> None:
    """BK-1's kill: `paddle.float32` vs an authored `float32` must MATCH, not destroy.

    m7960's four published ``failed:forward / mode-run`` terminals -- with
    ``mode_error: null`` and a work dir proving the forward completed -- are exactly this
    comparison returning False.
    """

    # An authored contract dtype in the CANONICAL spelling matches every framework's
    # observed spelling of the same dtype. This is the conjunct m7960 failed.
    for _label, observed, canonical in _FRAMEWORK_DTYPE_SPELLINGS:
        assert dtype_matches(canonical.value, observed), f"{canonical.value} vs {observed}"
    # The specific ledger-reproduced pair, spelled both ways round.
    assert dtype_matches("float32", "paddle.float32")
    assert dtype_matches("torch.float32", "<dtype: 'float32'>")
    assert dtype_matches("paddle.float32", "float32")
    # And the tripwire still bites on a REAL mismatch: canonicalization must not launder.
    assert not dtype_matches("float32", "paddle.float64")
    assert not dtype_matches("int64", "torch.int32")


@pytest.mark.smoke
def test_capture_records_dtypes_canonically_so_the_ledger_is_framework_neutral() -> None:
    """BK-1 fix (1): record canonically AT CAPTURE, not at each comparison site.

    Every downstream comparer -- including the torch-only normalizer still live in the
    checker-owned metadata module -- then matches for every framework, because the
    observed side never carries a framework prefix again.
    """

    for _label, spelling, canonical in _FRAMEWORK_DTYPE_SPELLINGS:
        signature = output_signature(_FakeTensor(spelling))
        recorded = signature["leaves"][0]["dtype"]
        assert recorded == canonical.value, f"{spelling} recorded as {recorded}"
        assert "torch." not in str(recorded)
        assert "paddle." not in str(recorded)
        assert "<dtype" not in str(recorded)


@pytest.mark.smoke
def test_an_unmappable_dtype_is_preserved_verbatim_never_laundered() -> None:
    """Honesty over totality: an unknown spelling is recorded as itself, and refuses."""

    with pytest.raises(DtypeVocabularyError):
        canonical_dtype("quantum.float3")
    assert canonical_dtype_text("quantum.float3") == "quantum.float3"
    signature = output_signature(_FakeTensor("quantum.float3"))
    assert signature["leaves"][0]["dtype"] == "quantum.float3"
    assert not dtype_matches("float32", "quantum.float3")


@pytest.mark.smoke
def test_the_backend_parity_deferral_exists_and_is_terminal() -> None:
    """B7: the honest outcome is representable at all."""

    assert BACKEND_PARITY_DEFERRAL_STATUS_CODE == "deferred:needs-backend-parity"
    assert BACKEND_PARITY_DEFERRAL_STATUS_CODE in TERMINAL_STATUS_CODES


@pytest.mark.smoke
def test_a_no_lane_framework_defers_instead_of_failing_forward() -> None:
    """B7's core: backend incompleteness never publishes as a model defect."""

    no_lane = [
        framework
        for framework in Framework
        if framework_capability(framework).lane_status is LaneStatus.NO_ENVIRONMENT_LANE
    ]
    assert no_lane, "the census is vacuous if every framework has a lane"
    for framework in no_lane:
        decision = backend_parity_deferral(
            "m7960",
            framework,
            detail="no environment intent exists for this framework on this platform",
        )
        assert decision.status == BACKEND_PARITY_DEFERRAL_STATUS_CODE
        assert decision.status != "failed:forward"
        assert decision.evidence, "a deferral with no evidence is a fabricated proof"
        with pytest.raises(RoutingError):
            route_model(ModelRequirements(stable_id="m7960", framework=framework.value))


@pytest.mark.smoke
def test_a_deferral_cannot_be_minted_without_evidence() -> None:
    """The proof rule is machine-derived; an evidence-free defer is refused."""

    with pytest.raises(ValueError):
        backend_parity_deferral("m1", Framework.MLX, detail="   ")


@pytest.mark.smoke
def test_the_measured_13_percent_false_positive_class_is_gone() -> None:
    """F6: 8 of 63 native-tail rows are provably PyTorch, and their own zoo says so.

    The live roster rows from TRACKB_opus F6. A name substring must never outrank the
    row's own zoo declaration.
    """

    false_positives = (
        "paddleseg_stfpm",
        "paddleseg_dnlnet",
        "paddleseg_uhrnet",
        "paddleseg_upernet",
        "paddleseg_vit_adapter",
        "paddleseg_mscale_ocrnet",
        "paddledet_ppyoloe_plus_s",
        "paddledet_yolof",
    )
    for name in false_positives:
        assert (
            framework_from_markers("unregistered-classics-pytorch", name) == Framework.TORCH.value
        )
    # A genuine native row still routes native.
    assert framework_from_markers("discovered-tensorflow", "movinet") == Framework.TENSORFLOW.value
    assert framework_from_markers("keras-applications", "resnet50") == Framework.TENSORFLOW.value


@pytest.mark.smoke
def test_marker_precedence_is_specificity_not_tuple_order() -> None:
    """F6: "first hit in tuple order wins" made precedence arbitrary."""

    # "tf-keras" names Keras on TensorFlow; the more specific marker must win, and the
    # answer must not depend on which marker happens to come first in a tuple.
    assert framework_from_markers("google", "tf_keras_efficientnet") == Framework.TENSORFLOW.value
    assert framework_from_markers("google", "keras_tf_efficientnet") == Framework.TENSORFLOW.value
    # jax+flax both present resolves to the one canonical framework either implies.
    assert framework_from_markers("google", "flax_jax_vit") == Framework.JAX.value


@pytest.mark.smoke
def test_the_marker_output_is_always_a_canonical_framework_spelling() -> None:
    """B1: only ONE spelling per framework may reach an identity preimage."""

    for zoo, name in (
        ("timm", "resnet"),
        ("discovered-keras", "model"),
        ("discovered-flax", "model"),
        ("discovered-tensorflow", "model"),
        ("discovered-paddle", "model"),
        ("github:google-research/vision_transformer", "vit"),
    ):
        derived = framework_from_markers(zoo, name)
        assert derived in {framework.value for framework in Framework}


@pytest.mark.smoke
def test_original_framework_drift_requires_a_disclosed_native_lane_disposition() -> None:
    """B6 (2): the settled native-first decision becomes mechanically enforceable.

    11 of 67 settled pilot heads carry ``original_framework != run_framework`` with NOTHING
    in the record distinguishing m10470 (an exact maintained torch library for a
    TF-origin architecture -- legitimate) from an undisclosed torch reimplementation,
    which is the 2026-07-01 1,029-row deletion class.
    """

    with pytest.raises(NativeLaneUndisclosedError):
        require_native_lane_disposition(
            original_framework="tensorflow", run_framework="pytorch", disposition=None
        )
    for disposition in NATIVE_LANE_DISPOSITIONS:
        assert (
            require_native_lane_disposition(
                original_framework="tensorflow",
                run_framework="pytorch",
                disposition=disposition,
            )
            is not None
        )
    # No disclosure is owed when the frameworks agree.
    assert (
        require_native_lane_disposition(
            original_framework="pytorch", run_framework="pytorch", disposition=None
        )
        is None
    )


@pytest.mark.smoke
def test_the_routed_lane_and_the_declared_framework_must_agree() -> None:
    """B1 (4): today nothing compares the declared framework with the routed intent."""

    torch_route = route_model(ModelRequirements(stable_id="m1", framework="pytorch"))
    assert (
        reconcile_execution_framework(
            declared_run_framework="pytorch", routed_intent=torch_route.intent
        ).agrees
        is True
    )
    verdict = reconcile_execution_framework(
        declared_run_framework="tensorflow", routed_intent=torch_route.intent
    )
    assert verdict.agrees is False
    assert verdict.disposition == BACKEND_PARITY_DEFERRAL_STATUS_CODE
