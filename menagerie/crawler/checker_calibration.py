"""Checker calibration receipts: measured sol capacity, fail-closed activation.

BL-10 (a)+(i) / audits CH-F5, opus B-5, sol B02/B03. The all-sol checker
rotation (``f08e8bdb``) pinned ``gpt-5.6-sol`` as the sole adjudication
authority for all three gate kinds while ZERO lifetime sol sessions existed on
``metadata_batch`` and ``terminal_disposition``: every calibration constant
(batch cap 5, the 600s attempt cap, the refusal baseline) is terra-measured,
and the committed golden transcripts for both lanes are terra traffic. The
launch plan also carried no stated checker budget at all -- the pilot measured
51.15h of checker wall for 67 models, 28% of it rendering no verdict, which
straight-lines to ~15,000h at 20k.

This module makes checker activation a TYPED TRANSACTION instead of a constants
edit. The real command lane refuses to buy ANY session for a gate kind until a
calibration receipt exists that:

1. names the exact pinned checker model for that kind;
2. carries a MEASURED per-lane sample of at least
   :data:`CALIBRATION_MIN_SESSIONS` real sessions (latency p95/p99, output-token
   p99, refusal rate) -- so the receipt cannot be a guess;
3. proves the NEXT rung fits inside its declared wall (and, when declared,
   token/dollar) budgets with :data:`CHECKER_CAPACITY_MARGIN` margin -- the
   governor fails closed BEFORE overspend, not after;
4. is corroborated by RE-RECORDED golden transcripts: the committed golden
   corpus must contain at least one real transcript for the (gate kind, model)
   pair, so a receipt alone can never activate a lane whose goldens still name
   a retired checker.

KERNEL_DECISION (2026-08-29, tri-lab unanimous rule-2 resolution) -- CHECKER
CALIBRATION: DECIDEDNESS IS PROVED BY CLOSING QUESTIONS, NOT BY CLASSIFYING
ITEMS. The receipt schema is v2 and STRICTLY HARDER to mint than v1:

* The raw same-epoch zero bar is NOT loosened, split, rate-bounded, per-lane
  scoped, or relocated onto a lane-adjudicated outcome. A same-epoch
  disagreement on the raw per-item verdict projection remains an
  unconditional refusal. No CONTESTED class exists: every acceptance flip in
  the round-19/21 matrix corpus root-caused to a defect the contract closed
  (m10470's pole was a draw asserting 21 mechanically refuted line claims --
  0/7,031 real mislocations exist; m10759's was the negative/meta inventory
  arm plus a whitespace-equivalence authority mismatch).
* Three NEW zero bars: (i) ``mechanical_claims`` -- every machine-decidable
  checker claim in the matrix must be discharged by the typed machine
  attestation, zero refuted, zero coverage gaps; (ii) ``locator_attestation``
  -- the whole corpus's excerpt locators machine-verified with zero failures,
  ``attested_by`` structurally const ``"machine"`` so model-adjudicated
  locators are unrepresentable, and the keyset digest verifier-recomputable
  from the corpus bytes; (iii) ``acceptance_witness`` -- the production
  acceptance depth K=2 (unconditional, every gate kind) is BOUND into the
  receipt with every accept-side golden carrying two agreeing same-epoch
  witnesses, so K can never fall back to 1 through a constants edit.
* Width-invariance degeneracy at production width 1 must be DECLARED
  (``degenerate_at_production_width``) and its extra draw folds into the
  same-epoch count; a vacuous comparison can never present as a passed
  invariance proof.
* ``disagreement_diagnosis`` is a required diagnostic array over closed
  vocabularies (pole class, mechanical refutation, defect class). It derives
  nothing and admits nothing: any row at all still refuses through the
  unchanged same-epoch bar, and a ``refuted`` row refuses on its own.
* Era: ZERO v1 receipts were ever minted (the width-1 mint attempt refused
  PRE-WRITE by design, evidence filed in the checker-width1 sweep artifacts;
  ``git log --all`` over the assets path is empty). v1-shaped receipts refuse
  typed with the era named; there is nothing to migrate and nothing is
  grandfathered. Refused receipts stay refused and stay disclosed.

The receipt is a committed, reviewed artifact
(:data:`CHECKER_CALIBRATION_RECEIPT_PATH`); operators may point
``MENAGERIE_CHECKER_CALIBRATION_RECEIPT`` at an alternative receipt file (e.g.
a freshly minted one during the calibration rung itself), and the calibration
capture run opts out explicitly with ``MENAGERIE_CHECKER_CALIBRATION_RUN=1`` --
that is the ONE sanctioned way to buy the ~25 measurement sessions the receipt
is minted from. Calibration mode WITHHOLDS awards structurally (W3): while the
flag is set, the canonical reducer refuses to issue any ``runs`` award head,
refuses award restoration, and refuses to mark any record release-eligible
(:class:`CheckerCalibrationWithheldError`); every leg attests the flag and the
override env reads in its leg-config row
(:func:`calibration_environment_snapshot`); and every gate verdict rendered
without receipt coverage is disclosed by the derived
:func:`calibration_bypass_census` arm (PD-9), which a retroactively minted
receipt can never bless. Absence of a valid receipt is always a REFUSAL,
never a downgrade.

W3's THIRD leg mode is the shadow-accuracy rehearsal
(``MENAGERIE_SHADOW_REHEARSAL_RUN=<rehearsal-generation-namespace>``): a
RECEIPT-BEARING mode that withholds awards exactly like calibration, stamps
its namespace as the structural ``shadow_rehearsal`` marker onto every record
the leg writes (:func:`menagerie.crawler.identity.stamp_shadow_rehearsal`),
and is disclosed by the segregated :func:`shadow_rehearsal_census` arm.
Declaring calibration and shadow-rehearsal together is an ambiguous mode and
refuses (:class:`AmbiguousLegModeError`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from menagerie.crawler.constants import CHECKER_PROMPT_NAME, METADATA_BATCH_MAX, GateKind
from menagerie.crawler.evidence import TEXT_LINE_LOCATOR_BASIS
from menagerie.crawler.identity import hash_bytes, is_shadow_rehearsal_record
from menagerie.crawler.operator_protocol import (
    CHECKER_SESSION_WALL_SECONDS,
    OPERATOR_ATTEMPT_TIMEOUT_SECONDS,
)

#: The pre-rule-2 schema. ZERO instances were ever minted (the width-1 mint
#: attempt refused pre-write), so there is no migration: a v1-shaped receipt
#: refuses typed, naming the era.
CHECKER_CALIBRATION_RECEIPT_SCHEMA_V1 = "menagerie.crawler.checker-calibration-receipt.v1"

CHECKER_CALIBRATION_RECEIPT_SCHEMA = "menagerie.crawler.checker-calibration-receipt.v2"

#: Closed disagreement-diagnosis vocabularies (Opus 6.2). Extending either set
#: is a reviewed vocabulary change, never a config edit.
POLE_CLASS_ORDERED_ABSTENTION = "ordered-abstention"
POLE_CLASS_CONTRADICTION = "contradiction"
DIAGNOSIS_POLE_CLASSES = frozenset({POLE_CLASS_ORDERED_ABSTENTION, POLE_CLASS_CONTRADICTION})
MECHANICAL_REFUTATION_VALUES = frozenset({"none-attempted", "refuted", "unrefuted"})
DIAGNOSIS_DEFECT_CLASSES = frozenset(
    {
        "underdetermined-contract-question",
        "fabricated-evidence-claim",
        "unclosed-vocabulary-overlap",
        "attention-dilution",
        "residual-evidence-sufficiency",
    }
)

#: acceptance_witness consts (Sol H-5 / fable A-4): production acceptance
#: depth is K=2 unconditionally for every gate kind; a confirmation conflict
#: parks, an unfunded confirmation parks, and the contested census arm is
#: fixed. The receipt BINDS these so production depth is a typed transaction.
ACCEPTANCE_WITNESS_MINIMUM = 2
ACCEPTANCE_DISAGREEMENT_OUTCOME = "checker-verdict-conflict-park"
ACCEPTANCE_UNFUNDED_OUTCOME = "checker-accept-unconfirmed"
CONTESTED_CENSUS_ARM = "checker-verdict-conflict"

#: locator_attestation const: the field exists to make model-adjudicated
#: locators structurally unrepresentable in a mintable receipt.
LOCATOR_ATTESTED_BY = "machine"

#: The batch width every calibration measurement and blind-replay quantile is
#: pinned to: the PRODUCTION metadata cap. A receipt measured at any other
#: width is measuring a different lane and refuses (fable U-F2/C8).
CALIBRATION_PRODUCTION_BATCH_WIDTH = METADATA_BATCH_MAX

#: Committed receipt location. Absent until the sol calibration rung runs and
#: its measured receipt lands in a reviewed change -- which is exactly the
#: fail-closed step-200 gate: no receipt, no sol session.
CHECKER_CALIBRATION_RECEIPT_PATH = (
    Path(__file__).with_name("assets") / "checker-calibration-receipt.json"
)

#: Operator override naming an alternative receipt file (bootstrap and tests).
CHECKER_CALIBRATION_RECEIPT_ENV = "MENAGERIE_CHECKER_CALIBRATION_RECEIPT"

#: Overhaul RULING B.2 (checker row): the per-epoch calibration receipt and its
#: 42-ask re-mint are REPLACED by an activation matrix ONCE per new (checker
#: model, gate kind) at the live prompt epoch; thereafter the production K=2
#: conflict census per (kind, model, epoch) measures drift. The committed
#: activation record is that matrix. It is consulted BEFORE the receipt door:
#: a (kind, model) pair with an activation row at the live epoch is activated
#: without a receipt; every other pair keeps the receipt+goldens door exactly
#: as before (the sol seat's receipt stays what it was).
CHECKER_ACTIVATION_PATH = Path(__file__).with_name("assets") / "checker-activation.json"
CHECKER_ACTIVATION_ENV = "MENAGERIE_CHECKER_ACTIVATION"
CHECKER_ACTIVATION_SCHEMA = "menagerie.crawler.checker-activation.v1"
#: The activation floor: fewer witnessed sessions than this on a (kind, model)
#: is not a matrix, it is a probe. Three draws per golden over the smallest
#: kind's two goldens is six; the bench's smallest kind ran exactly that.
CHECKER_ACTIVATION_MIN_SESSIONS = 6

#: Explicit opt-out for the calibration capture run itself: the measurement
#: sessions must be purchasable before a receipt can exist.
CHECKER_CALIBRATION_RUN_ENV = "MENAGERIE_CHECKER_CALIBRATION_RUN"

#: W3's THIRD leg mode (frozen PLAN_v6 addition): the shadow-accuracy
#: rehearsal. The variable's NON-EMPTY VALUE is the declared rehearsal
#: generation namespace (e.g. ``"r6b-gen-1"``) -- never a bare ``1`` -- and
#: that exact string is what the write-side producer stamps into every record
#: the leg writes (:func:`menagerie.crawler.identity.stamp_shadow_rehearsal`).
#: The leg-mode vocabulary is CLOSED at exactly three -- normal, calibration,
#: shadow-rehearsal -- and declaring calibration AND shadow-rehearsal together
#: is an AMBIGUOUS mode that refuses (:class:`AmbiguousLegModeError`); a
#: fourth mode cannot appear without its own reviewed change here. Unlike
#: calibration, shadow-rehearsal is RECEIPT-BEARING: a verdict-rendering
#: shadow leg still requires the valid committed receipt (the mode never
#: substitutes for it), so shadow+receipt is NOT ambiguous.
SHADOW_REHEARSAL_RUN_ENV = "MENAGERIE_SHADOW_REHEARSAL_RUN"

#: Committed golden transcript corpus (real recorded checker traffic).
CHECKER_GOLDEN_CORPUS_DIR = Path(__file__).with_name("tests") / "golden_checker_transcripts"

#: Corpus override for tests exercising the gate mechanics.
CHECKER_GOLDEN_CORPUS_ENV = "MENAGERIE_CHECKER_GOLDEN_CORPUS"

#: Minimum MEASURED sessions per lane before a receipt row counts as evidence
#: rather than a guess. Metadata needs the cap-width distribution (CH-F5's ~20
#: frozen step-100 packs); the single-item lanes need enough to see a tail.
CALIBRATION_MIN_SESSIONS: Mapping[str, int] = {
    GateKind.METADATA_BATCH.value: 20,
    GateKind.FIDELITY.value: 5,
    GateKind.TERMINAL_DISPOSITION.value: 5,
}

#: A lane refusing more than this often is not calibrated -- it is broken.
CALIBRATION_MAX_REFUSAL_RATE = 0.2

#: Projected next-rung usage must fit inside this fraction of every declared
#: budget. The governor fails closed at the margin, before overspend.
CHECKER_CAPACITY_MARGIN = 0.8

#: Per-lane measured fields a receipt must carry (all non-negative numbers).
#: ``sessions`` counts RENDERED verdicts (non-reused gate rows);
#: ``sessions_bought`` is the paid-ledger denominator (settled
#: provider-accepted reservations), so the refusal rate can never hide bought
#: sessions that rendered nothing (sol P2-12 denominator honesty). The
#: ``attempt_*`` trio carries the completed-attempt latency tail from the
#: wrapper telemetry -- the evidence the per-attempt grant is derived from.
_LANE_MEASURED_FIELDS = (
    "sessions",
    "sessions_bought",
    "latency_p95_seconds",
    "latency_p99_seconds",
    "output_tokens_p99",
    "refusal_rate",
    "attempt_latency_p99_seconds",
    "attempt_samples",
    "attempt_censoring_rate",
)

#: (projected, budget) capacity pairs. Wall, sessions, and dollars are
#: REQUIRED (W2: the receipt DECLARES its dollar+session budget); tokens are
#: enforced whenever declared.
_CAPACITY_PAIRS = (
    ("projected_wall_hours", "wall_budget_hours", True),
    ("projected_sessions", "session_budget", True),
    ("projected_output_tokens", "token_budget", False),
    ("projected_dollars", "dollar_budget", True),
)

#: Golden corpus filename prefix per gate kind (the corpus keys entries by
#: lane-prefixed names; see ``tests/golden_checker_transcripts/README.md``).
_GOLDEN_PREFIX: Mapping[str, str] = {
    GateKind.METADATA_BATCH.value: "metadata_",
    GateKind.FIDELITY.value: "fidelity_",
    GateKind.TERMINAL_DISPOSITION.value: "terminal_",
}


class CheckerCalibrationError(RuntimeError):
    """The checker lane is not calibrated for a dispatch it was asked to buy.

    Deliberately NOT a ``CheckerDispatchError``: a missing calibration is a
    LAUNCH-ELIGIBILITY refusal for the whole lane, never a per-batch lane
    failure the downshift ladder should absorb into per-model parks.
    """


class CheckerCalibrationMissingError(CheckerCalibrationError):
    """No calibration receipt exists for the pinned checker model."""


class CheckerCalibrationInvalidError(CheckerCalibrationError):
    """A receipt exists but is malformed or under the measurement floor."""


class CheckerCapacityMarginError(CheckerCalibrationError):
    """The measured capacity projection does not fit the budgets with margin."""


class CheckerGoldenCorpusError(CheckerCalibrationError):
    """The committed golden corpus does not corroborate the receipt's model."""


class CheckerCalibrationEpochError(CheckerCalibrationError):
    """The receipt's checker prompt epoch is not the shipping prompt epoch.

    P5 round-2 R2-3: a prompt rotation moved the shipping checker out from
    under the committed receipt and NOTHING refused — the receipt recorded its
    epoch only as free text inside ``source``, and the coverage gate compared
    ``checker_model`` alone. Every verdict-rendering lane now refuses, typed,
    when the receipt carries no structured ``checker_prompt_identity`` or
    carries one that is not the live prompt's content hash: a receipt minted
    at another epoch measured a different checker.
    """


class CheckerCalibrationWithheldError(CheckerCalibrationError):
    """An award or release-eligibility marking was attempted in calibration mode.

    The sanctioned calibration capture run may buy sessions and render gate
    verdicts (that is what it is for), but it may never ISSUE a ``runs``
    award head, restore one, or mark any record release-eligible: calibration
    verdicts are measurements, not adjudications the campaign may bank. The
    withhold is a refusal at the canonical append seam, never a downgrade.
    The shadow-rehearsal mode (W3's third mode) withholds identically:
    rehearsal verdicts are attested measurements, never bankable adjudications.
    """


class AmbiguousLegModeError(CheckerCalibrationError):
    """Calibration AND shadow-rehearsal are both declared: an ambiguous mode.

    The leg-mode vocabulary is closed at exactly three -- normal, calibration,
    shadow-rehearsal -- and a process declaring two of them at once has no
    single sanctioned disposition. Every DIRECT mode reader refuses here,
    fail-closed, before any mode-conditioned behavior runs; the W2 preflight
    XOR arm surfaces the same refusal as an evidence-shaped refusal row.
    """


def _ambiguous_mode_declared() -> bool:
    """Return whether calibration and shadow-rehearsal are both declared (raw)."""

    return (
        os.environ.get(CHECKER_CALIBRATION_RUN_ENV, "") == "1"
        and bool(os.environ.get(SHADOW_REHEARSAL_RUN_ENV, "").strip())
    )


def _refuse_ambiguous_mode() -> None:
    """Refuse, typed, when calibration and shadow-rehearsal are both declared."""

    if _ambiguous_mode_declared():
        raise AmbiguousLegModeError(
            f"ambiguous leg mode: {CHECKER_CALIBRATION_RUN_ENV}=1 and "
            f"{SHADOW_REHEARSAL_RUN_ENV} are both declared; the mode vocabulary "
            "is closed at normal/calibration/shadow-rehearsal and a leg holds "
            "exactly one -- unset one of the two variables"
        )


def calibration_run_active() -> bool:
    """Return whether this process is the sanctioned calibration capture run.

    Raises
    ------
    AmbiguousLegModeError
        If the shadow-rehearsal mode is declared alongside calibration: a
        direct mode reader never adjudicates under an ambiguous declaration.
    """

    _refuse_ambiguous_mode()
    return os.environ.get(CHECKER_CALIBRATION_RUN_ENV, "") == "1"


def shadow_rehearsal_namespace() -> Optional[str]:
    """Return the declared rehearsal generation namespace, or ``None``.

    The namespace is :data:`SHADOW_REHEARSAL_RUN_ENV`'s non-empty value (e.g.
    ``"r6b-gen-1"``): the exact string the write-side producer stamps as the
    typed shadow marker on every record the leg writes.

    Raises
    ------
    AmbiguousLegModeError
        If calibration mode is declared alongside shadow-rehearsal.
    """

    _refuse_ambiguous_mode()
    return os.environ.get(SHADOW_REHEARSAL_RUN_ENV, "").strip() or None


def shadow_rehearsal_run_active() -> bool:
    """Return whether this process is a declared shadow-rehearsal leg (W3).

    Raises
    ------
    AmbiguousLegModeError
        If calibration mode is declared alongside shadow-rehearsal.
    """

    return shadow_rehearsal_namespace() is not None


def calibration_environment_snapshot() -> dict[str, Any]:
    """Return the complete verdict-steering calibration env-read inventory.

    Exactly the four environment variables this module reads can steer how a
    gate verdict is adjudicated (W3/TE-3): the calibration-mode flag, the
    shadow-rehearsal mode namespace, the receipt override, and the
    golden-corpus override. Every leg attests this snapshot in its
    ``leg-config`` row so a bypass-adjudicated leg is distinguishable from an
    attested one in the ledgers, not just in an operator's shell history.

    Deliberately TOTAL: the snapshot reports raw declarations and never
    raises, even for the ambiguous both-declared mode -- attestation records
    what was set, and the direct mode readers are where ambiguity refuses.

    Returns
    -------
    dict[str, Any]
        ``calibration_run_active``, ``shadow_rehearsal_run`` (the declared
        rehearsal namespace, ``None`` when unset), plus the two override
        paths (``None`` when unset -- the committed artifacts are in effect).
    """

    receipt_override = os.environ.get(CHECKER_CALIBRATION_RECEIPT_ENV, "").strip()
    corpus_override = os.environ.get(CHECKER_GOLDEN_CORPUS_ENV, "").strip()
    return {
        "calibration_run_active": os.environ.get(CHECKER_CALIBRATION_RUN_ENV, "") == "1",
        "shadow_rehearsal_run": os.environ.get(SHADOW_REHEARSAL_RUN_ENV, "").strip() or None,
        "calibration_receipt_override": receipt_override or None,
        "golden_corpus_override": corpus_override or None,
    }


def live_checker_prompt_identity() -> str:
    """Return the SHIPPING checker prompt's content identity (the live epoch).

    The same derivation :func:`menagerie.crawler.authority.build_authority_context`
    stamps into every dependency vector's ``checker_prompt_identity`` axis:
    ``sha256:`` over the exact shipped prompt bytes. Computed fresh on every
    call so no cached value can survive a prompt rotation.

    Raises
    ------
    CheckerCalibrationInvalidError
        If the shipped prompt file cannot be read: an unmeasurable live epoch
        must refuse the lane, never default it.
    """

    prompt_path = Path(__file__).resolve().parent / "prompts" / f"{CHECKER_PROMPT_NAME}.txt"
    try:
        return hash_bytes(prompt_path.read_bytes())
    except OSError as exc:
        raise CheckerCalibrationInvalidError(
            f"the shipped checker prompt is unreadable at {prompt_path}: {exc}; the "
            "live prompt epoch cannot be derived, so no receipt can be proven current"
        ) from exc


def _receipt_path() -> Path:
    """Resolve the active receipt path (operator override, else committed)."""

    override = os.environ.get(CHECKER_CALIBRATION_RECEIPT_ENV, "").strip()
    return Path(override) if override else CHECKER_CALIBRATION_RECEIPT_PATH


def _golden_corpus_dir() -> Path:
    """Resolve the active golden corpus directory (override, else committed)."""

    override = os.environ.get(CHECKER_GOLDEN_CORPUS_ENV, "").strip()
    return Path(override) if override else CHECKER_GOLDEN_CORPUS_DIR


def _require_number(value: Any, *, field: str, source: str) -> float:
    """Return one required non-negative finite number, fail-closed."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: {field} must be a number, got {value!r}"
        )
    number = float(value)
    if not number >= 0.0 or number != number or number == float("inf"):
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: {field} must be finite and non-negative"
        )
    return number


def load_checker_calibration_receipt(path: Optional[Path] = None) -> dict[str, Any]:
    """Load and structurally validate the calibration receipt, fail-closed.

    Parameters
    ----------
    path:
        Explicit receipt path; defaults to the operator override or the
        committed artifact.

    Returns
    -------
    dict[str, Any]
        The validated receipt object.

    Raises
    ------
    CheckerCalibrationMissingError
        If no receipt file exists at the resolved path.
    CheckerCalibrationInvalidError
        If the receipt is unreadable, malformed, or under-measured.
    """

    resolved = path if path is not None else _receipt_path()
    if not resolved.is_file():
        raise CheckerCalibrationMissingError(
            f"no checker calibration receipt exists at {resolved}; the sole checker "
            "has never been measured on this lane's production contract. Run the "
            "calibration rung (MENAGERIE_CHECKER_CALIBRATION_RUN=1) over frozen "
            "step-100 packs, mint the receipt, and land it in a reviewed change "
            "before any full rung buys sessions"
        )
    source = str(resolved)
    try:
        receipt = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source} is unreadable: {exc}"
        ) from exc
    if not isinstance(receipt, dict):
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source} must be one JSON object"
        )
    if receipt.get("schema_version") == CHECKER_CALIBRATION_RECEIPT_SCHEMA_V1:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source} declares the retired v1 schema "
            f"({CHECKER_CALIBRATION_RECEIPT_SCHEMA_V1}). Zero v1 receipts were ever "
            "minted -- the width-1 attempt refused pre-write -- so a v1-shaped "
            "artifact is not history, it is an attempt to mint under the pre-rule-2 "
            "bars; re-mint under the v2 zero bars"
        )
    if receipt.get("schema_version") != CHECKER_CALIBRATION_RECEIPT_SCHEMA:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source} declares unknown schema "
            f"{receipt.get('schema_version')!r}"
        )
    for field in ("checker_model", "captured_at", "source"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: {field} must be a non-empty string"
            )
    if "checker_prompt_identity" in receipt and (
        not isinstance(receipt["checker_prompt_identity"], str)
        or not receipt["checker_prompt_identity"].startswith("sha256:")
    ):
        # Presence is enforced at the activation door (R2-3), so the committed
        # pre-epoch receipt stays LOADABLE for census and diagnosis while it
        # can never activate a verdict-rendering lane; a present-but-garbage
        # epoch refuses even the load.
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: checker_prompt_identity must be a "
            "sha256: content digest of the exact prompt bytes the calibration ran under"
        )
    width = receipt.get("production_batch_width")
    if not isinstance(width, int) or isinstance(width, bool):
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: production_batch_width must be an "
            "integer -- the width every measurement was captured at"
        )
    if width != CALIBRATION_PRODUCTION_BATCH_WIDTH:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: measured at batch width {width}, but "
            f"production runs metadata batches at {CALIBRATION_PRODUCTION_BATCH_WIDTH}; a "
            "calibration at another width measured a different lane (fable U-F2/C8)"
        )
    lanes = receipt.get("lanes")
    if not isinstance(lanes, Mapping):
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source} must carry per-lane measurements"
        )
    for kind in GateKind:
        lane = lanes.get(kind.value)
        if not isinstance(lane, Mapping):
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source} has no measured lane for "
                f"{kind.value!r}; every gate kind must be measured before activation"
            )
        for field in _LANE_MEASURED_FIELDS:
            _require_number(lane.get(field), field=f"lanes.{kind.value}.{field}", source=source)
        if "pre_genesis_rendered" in lane:
            # Epoch-scope disclosure: verdicts rendered before the paid
            # ledger's genesis anchor (settled on a predecessor ledger).
            # Disclosed, never counted -- it feeds no floor or rate, so it is
            # optional for receipts minted before the window was scoped, but
            # garbage disclosure still refuses.
            _require_number(
                lane.get("pre_genesis_rendered"),
                field=f"lanes.{kind.value}.pre_genesis_rendered",
                source=source,
            )
        sessions = int(lane["sessions"])
        floor = CALIBRATION_MIN_SESSIONS[kind.value]
        if sessions < floor:
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: lane {kind.value} measured only "
                f"{sessions} sessions (floor {floor}); an under-measured lane is a guess"
            )
        if float(lane["refusal_rate"]) > CALIBRATION_MAX_REFUSAL_RATE:
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: lane {kind.value} measured a "
                f"{lane['refusal_rate']} refusal rate (max {CALIBRATION_MAX_REFUSAL_RATE}); "
                "a lane refusing that often is broken, not calibrated"
            )
        if int(lane["sessions_bought"]) < sessions:
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: lane {kind.value} claims "
                f"{sessions} rendered sessions from only {lane['sessions_bought']} bought; "
                "the paid-ledger denominator cannot be smaller than the verdicts it paid for"
            )
        if float(lane["attempt_censoring_rate"]) > CALIBRATION_MAX_REFUSAL_RATE:
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: lane {kind.value} measured a "
                f"{lane['attempt_censoring_rate']} attempt censoring rate (max "
                f"{CALIBRATION_MAX_REFUSAL_RATE}); a p99 quoted over censored attempts "
                "understates the tail by exactly the amount the cap is too small"
            )
        # The 600s lesson (2026-08-21): the provisional 600s attempt grant
        # censored EVERY sol metadata attempt. A timing constant below the
        # measured p99 truncates sessions from outside, so the receipt refuses
        # the lane until the constant is re-derived from this measurement in
        # the same reviewed change -- wall >= measured p99, never hand-set.
        attempt_p99 = float(lane["attempt_latency_p99_seconds"])
        if attempt_p99 > float(OPERATOR_ATTEMPT_TIMEOUT_SECONDS):
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: lane {kind.value} measured a "
                f"{attempt_p99}s completed-attempt p99 above the "
                f"{OPERATOR_ATTEMPT_TIMEOUT_SECONDS}s attempt grant (the 600s lesson: a "
                "grant below the measured p99 censors attempts); re-derive "
                "OPERATOR_ATTEMPT_TIMEOUT_SECONDS from this receipt before activation"
            )
        session_p99 = float(lane["latency_p99_seconds"])
        if session_p99 > float(CHECKER_SESSION_WALL_SECONDS):
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: lane {kind.value} measured a "
                f"{session_p99}s session p99 above the {CHECKER_SESSION_WALL_SECONDS}s "
                "session wall (the 600s lesson one level up); re-derive the wall ladder "
                "from this receipt before activation"
            )
    _validate_blind_replay(receipt.get("blind_replay"), source=source)
    capacity = receipt.get("capacity")
    if not isinstance(capacity, Mapping):
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source} must carry a next-rung capacity plan"
        )
    _require_number(
        capacity.get("next_rung_models"), field="capacity.next_rung_models", source=source
    )
    for projected_field, budget_field, required in _CAPACITY_PAIRS:
        projected = capacity.get(projected_field)
        budget = capacity.get(budget_field)
        if projected is None and budget is None and not required:
            continue
        projected_value = _require_number(
            projected, field=f"capacity.{projected_field}", source=source
        )
        budget_value = _require_number(budget, field=f"capacity.{budget_field}", source=source)
        if projected_value > CHECKER_CAPACITY_MARGIN * budget_value:
            raise CheckerCapacityMarginError(
                f"checker calibration receipt {source}: {projected_field}="
                f"{projected_value} exceeds {CHECKER_CAPACITY_MARGIN:.0%} of "
                f"{budget_field}={budget_value}; the next rung does not fit with "
                "margin, so the governor refuses before overspend"
            )
    return receipt


def _validate_blind_replay(matrix: Any, *, source: str) -> None:
    """Validate the B1-XV2 blind-replay matrix attestation, fail-closed.

    The matrix is W2's acceptance instrument: every golden replayed at least
    3x under the same model/prompt epoch with ZERO disagreement, one more
    replay after a deliberate rotation (its disagreement count is RECORDED,
    not constrained -- rotation may lawfully flip), batch-width invariance
    between width 1 and the production width, and the p99 measured AT
    production width before the receipt was minted. ``report_sha256`` binds
    the full matrix report artifact the numbers were derived from.
    """

    if not isinstance(matrix, Mapping):
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source} must carry the blind-replay matrix "
            "attestation (B1-XV2) -- the receipt's acceptance instrument"
        )
    goldens = _require_number(matrix.get("goldens"), field="blind_replay.goldens", source=source)
    if goldens < 1:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: the blind-replay matrix ran over zero "
            "goldens; an empty matrix attests nothing"
        )
    replays = _require_number(
        matrix.get("replays_per_golden"), field="blind_replay.replays_per_golden", source=source
    )
    if replays < 3:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: {replays} replays per golden is under "
            "the B1-XV2 floor of 3 same-epoch replays"
        )
    disagreements = _require_number(
        matrix.get("same_epoch_disagreements"),
        field="blind_replay.same_epoch_disagreements",
        source=source,
    )
    if disagreements != 0:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: {disagreements} same-epoch blind-replay "
            "disagreement(s); a checker that disagrees with itself under one epoch is not "
            "calibrated (the m4976 shape)"
        )
    if matrix.get("rotation_replay_ran") is not True:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: the post-rotation replay (B1-XV2's "
            "4th) did not run"
        )
    _require_number(
        matrix.get("rotation_disagreements"),
        field="blind_replay.rotation_disagreements",
        source=source,
    )
    invariance = matrix.get("width_invariance")
    if not isinstance(invariance, Mapping):
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: blind_replay.width_invariance must "
            "record the width-1 vs production-width comparison"
        )
    degenerate = invariance.get("degenerate_at_production_width")
    if not isinstance(degenerate, bool):
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: width_invariance must DECLARE "
            "degenerate_at_production_width -- at production width 1 the arm compares "
            "nothing, and a vacuous comparison silently presented as a passed "
            "invariance proof is a disarmed tripwire (Opus 6.6)"
        )
    widths = invariance.get("widths")
    width_disagreements = _require_number(
        invariance.get("disagreements"),
        field="blind_replay.width_invariance.disagreements",
        source=source,
    )
    if CALIBRATION_PRODUCTION_BATCH_WIDTH == 1:
        if width_disagreements != 0:
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: a degenerate width arm compares "
                f"nothing and cannot record {width_disagreements} disagreement(s); its "
                "extra draw's disagreement belongs in the same-epoch count"
            )
        if (
            degenerate is not True
            or not isinstance(widths, (list, tuple))
            or list(widths) != [1]
            or invariance.get("folded_into_same_epoch") is not True
        ):
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: at production width 1 the width "
                "arm is degenerate by construction -- it must declare "
                "degenerate_at_production_width true, widths [1], and "
                "folded_into_same_epoch true (its extra draw is a same-epoch draw and "
                "counts there, where it belongs)"
            )
    else:
        if degenerate:
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: width_invariance claims "
                "degeneracy while the production width is "
                f"{CALIBRATION_PRODUCTION_BATCH_WIDTH}; a real second width exists"
            )
        if (
            not isinstance(widths, (list, tuple))
            or 1 not in widths
            or CALIBRATION_PRODUCTION_BATCH_WIDTH not in widths
        ):
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: width invariance must compare "
                f"width 1 against the production width "
                f"{CALIBRATION_PRODUCTION_BATCH_WIDTH} (got {widths!r})"
            )
        if width_disagreements != 0:
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: {width_disagreements} batch-width "
                "invariance disagreement(s); a verdict that changes with batch width is "
                "not an adjudication of the item"
            )
    _require_number(
        matrix.get("p99_at_production_width_seconds"),
        field="blind_replay.p99_at_production_width_seconds",
        source=source,
    )
    report_digest = matrix.get("report_sha256")
    if not isinstance(report_digest, str) or not report_digest.strip():
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: blind_replay.report_sha256 must bind "
            "the matrix report artifact the attestation was derived from"
        )
    _validate_disagreement_diagnosis(matrix.get("disagreement_diagnosis"), source=source)
    _validate_mechanical_claims(matrix.get("mechanical_claims"), source=source)
    _validate_locator_attestation_block(matrix.get("locator_attestation"), source=source)
    _validate_acceptance_witness(matrix.get("acceptance_witness"), source=source)


def _validate_disagreement_diagnosis(rows: Any, *, source: str) -> None:
    """Require the closed-vocabulary diagnosis array (zero bar #1's carrier).

    The array is REQUIRED (an omitted field is indistinguishable from an
    unexamined matrix) and usually empty, because any same-epoch disagreement
    already refuses the mint through the unchanged zero bar. Each row is a
    derived diagnostic over closed vocabularies; a ``refuted`` row refuses on
    its own -- a draw that asserted a mechanically false claim about frozen
    bytes is a defect no receipt may be minted over (the bar that would have
    caught m10470 se2 on 2026-08-28 instead of promoting it to a contract
    question).
    """

    if not isinstance(rows, list):
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: blind_replay.disagreement_diagnosis "
            "must be an array (empty when the matrix has zero same-epoch disagreements)"
        )
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: disagreement_diagnosis[{index}] "
                "must be an object"
            )
        for field in ("golden", "item", "field"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise CheckerCalibrationInvalidError(
                    f"checker calibration receipt {source}: "
                    f"disagreement_diagnosis[{index}].{field} must be a non-empty string"
                )
        poles = row.get("poles")
        if not isinstance(poles, list) or len(poles) < 2:
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: "
                f"disagreement_diagnosis[{index}].poles must list the observed enum poles"
            )
        if row.get("pole_class") not in DIAGNOSIS_POLE_CLASSES:
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: "
                f"disagreement_diagnosis[{index}].pole_class must be one of "
                f"{sorted(DIAGNOSIS_POLE_CLASSES)}"
            )
        refutation = row.get("mechanical_refutation")
        if refutation not in MECHANICAL_REFUTATION_VALUES:
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: "
                f"disagreement_diagnosis[{index}].mechanical_refutation must be one of "
                f"{sorted(MECHANICAL_REFUTATION_VALUES)}"
            )
        if refutation == "refuted":
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: disagreement_diagnosis[{index}] "
                "records a mechanically REFUTED draw claim; a draw asserting a false "
                "fact about frozen bytes is a defect and no receipt may be minted over "
                "one (zero bar #1)"
            )
        defect = row.get("defect_class")
        if defect is not None and defect not in DIAGNOSIS_DEFECT_CLASSES:
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: "
                f"disagreement_diagnosis[{index}].defect_class must be null or one of "
                f"{sorted(DIAGNOSIS_DEFECT_CLASSES)}"
            )


def _validate_mechanical_claims(block: Any, *, source: str) -> None:
    """Zero bar over the machine-decidable claim inventory (Sol 4.3).

    Every machine-decidable claim any same-epoch draw made must be discharged
    by the typed machine attestation, none refuted, none escaping coverage --
    across ALL draws, not only the disagreeing ones (all draws repeating one
    mechanically false claim would otherwise produce no disagreement row and
    mint).
    """

    if not isinstance(block, Mapping):
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: blind_replay.mechanical_claims must "
            "carry the machine-decidable claim inventory (required/discharged/refuted/"
            "coverage_gaps)"
        )
    values = {
        field: _require_number(
            block.get(field), field=f"blind_replay.mechanical_claims.{field}", source=source
        )
        for field in ("required", "discharged", "refuted", "coverage_gaps")
    }
    if values["refuted"] != 0:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: {values['refuted']} mechanically "
            "refuted checker claim(s); a draw disputing machine arithmetic is a typed "
            "defect and refuses the mint (zero bar #1)"
        )
    if values["coverage_gaps"] != 0:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: {values['coverage_gaps']} "
            "machine-decidable claim(s) escaped attestation coverage; an unexamined "
            "claim is not a discharged one"
        )
    if values["required"] != values["discharged"]:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: mechanical_claims required="
            f"{values['required']} != discharged={values['discharged']}; every "
            "machine-decidable claim must be discharged by the typed attestation"
        )


def _validate_locator_attestation_block(block: Any, *, source: str) -> None:
    """Zero bar #2: corpus-total machine locator attestation, no sampling.

    ``attested_by`` is const ``"machine"`` so a model-adjudicated locator is
    structurally unrepresentable; ``basis`` is the one declared line basis;
    ``failures != 0`` refuses; ``keyset_sha256`` binds the sorted per-excerpt
    keyset so the verifier can recompute it from the corpus bytes (fable A-3:
    a producer-declared total alone would be self-referential).
    """

    if not isinstance(block, Mapping):
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: blind_replay.locator_attestation "
            "must carry the corpus-total machine locator attestation (zero bar #2)"
        )
    checks = _require_number(
        block.get("checks"), field="blind_replay.locator_attestation.checks", source=source
    )
    if checks < 1:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: locator_attestation covered zero "
            "excerpts; an empty attestation attests nothing"
        )
    failures = _require_number(
        block.get("failures"), field="blind_replay.locator_attestation.failures", source=source
    )
    if failures != 0:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: {failures} machine locator "
            "failure(s) in the golden corpus; a corpus carrying a refuted locator "
            "cannot calibrate anything (zero bar #2)"
        )
    _require_number(
        block.get("unattestable"),
        field="blind_replay.locator_attestation.unattestable",
        source=source,
    )
    if block.get("attested_by") != LOCATOR_ATTESTED_BY:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: locator_attestation.attested_by "
            f"must be {LOCATOR_ATTESTED_BY!r}; the field exists to make "
            "model-adjudicated locators structurally unrepresentable"
        )
    if block.get("basis") != TEXT_LINE_LOCATOR_BASIS:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: locator_attestation.basis must be "
            f"{TEXT_LINE_LOCATOR_BASIS!r}"
        )
    keyset = block.get("keyset_sha256")
    if not isinstance(keyset, str) or not keyset.strip():
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: locator_attestation.keyset_sha256 "
            "must bind the sorted per-excerpt attestation keyset"
        )


def _validate_acceptance_witness(block: Any, *, source: str) -> None:
    """Zero bar #3: bind the production acceptance depth K into the receipt.

    K=2 is UNCONDITIONAL for every gate kind (fable A-4 / Sol 4.5): deriving
    K from the presence of an abstention diagnosis would let a prompt repair
    silently lower publication depth back to 1 against the measured
    bad-bless floor. Every accept-side golden must carry at least two
    agreeing same-epoch witnesses; a missing or disagreeing confirmation
    refuses.
    """

    if not isinstance(block, Mapping):
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: blind_replay.acceptance_witness "
            "must bind the production acceptance depth (zero bar #3)"
        )
    required = _require_number(
        block.get("required_witnesses_per_accept"),
        field="blind_replay.acceptance_witness.required_witnesses_per_accept",
        source=source,
    )
    if required < ACCEPTANCE_WITNESS_MINIMUM:
        raise CheckerCalibrationInvalidError(
            f"checker calibration receipt {source}: acceptance_witness requires "
            f"{required} witnesses per accept, under the unconditional production "
            f"minimum {ACCEPTANCE_WITNESS_MINIMUM}; the matrix-derived requirement may "
            "only ever RAISE it"
        )
    for field in (
        "metadata_accept_witnesses",
        "fidelity_accept_witnesses",
        "terminal_accept_witnesses",
    ):
        _require_number(
            block.get(field), field=f"blind_replay.acceptance_witness.{field}", source=source
        )
    for field in ("missing_confirmations", "confirmation_disagreements"):
        value = _require_number(
            block.get(field), field=f"blind_replay.acceptance_witness.{field}", source=source
        )
        if value != 0:
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: acceptance_witness.{field}="
                f"{value}; every accept-side golden needs "
                f"{ACCEPTANCE_WITNESS_MINIMUM} agreeing same-epoch witnesses"
            )
    consts = {
        "disagreement_outcome": ACCEPTANCE_DISAGREEMENT_OUTCOME,
        "unfunded_confirmation_outcome": ACCEPTANCE_UNFUNDED_OUTCOME,
        "contested_census_arm": CONTESTED_CENSUS_ARM,
    }
    for field, expected in consts.items():
        if block.get(field) != expected:
            raise CheckerCalibrationInvalidError(
                f"checker calibration receipt {source}: acceptance_witness.{field} must "
                f"be {expected!r} -- the park-not-publish disposition is bound into the "
                "receipt, never a runtime choice"
            )


def _golden_corpus_models(kind: str) -> set[str]:
    """Return every checker model the committed goldens record for one kind."""

    corpus = _golden_corpus_dir()
    prefix = _GOLDEN_PREFIX.get(kind)
    if prefix is None or not corpus.is_dir():
        return set()
    models: set[str] = set()
    for entry in sorted(corpus.glob(f"{prefix}*.json")):
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        model = data.get("checker_model") if isinstance(data, Mapping) else None
        if isinstance(model, str) and model:
            models.add(model)
    return models


#: Derived census arm naming (PD-9): gate verdicts adjudicated without
#: calibration-receipt coverage, disclosed and release-ineligible.
CALIBRATION_BYPASS_CENSUS_ARM = "gates_adjudicated_under_calibration_bypass"


def calibration_bypass_census(
    gate_records: Iterable[Mapping[str, Any]],
    receipt: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Derive the PD-9 retroactive bypass-disclosure census arm.

    A gate verdict is RECEIPT-ATTESTED only when a structurally valid
    calibration receipt (a) names the exact checker model the row records and
    (b) was captured BEFORE the row finished -- so minting a receipt later can
    never retroactively bless history. Everything else -- no receipt, model
    mismatch, a row that predates the receipt, or a row/receipt without a
    comparable timestamp -- counts as bypass-adjudicated, fail-closed. The
    arm follows waiver-doctrine part 4: a derived, segregated disclosure that
    is recomputed from ledger rows plus the committed receipt artifact on
    every call (zero producer input steers it), never counted as attested,
    and release eligibility keeps excluding it.

    Parameters
    ----------
    gate_records:
        Gate-ledger rows (``menagerie.crawler.gate.v3`` shape; only the
        ``checker`` block and ``gate_id`` are read).
    receipt:
        A receipt already validated by
        :func:`load_checker_calibration_receipt`, or ``None`` when no valid
        receipt exists (an invalid receipt attests nothing).

    Returns
    -------
    dict[str, Any]
        The census arm name, the segregated bypass/attested counts, and the
        bypass gate IDs.
    """

    receipt_model = receipt.get("checker_model") if isinstance(receipt, Mapping) else None
    captured_at = receipt.get("captured_at") if isinstance(receipt, Mapping) else None
    bypass_gate_ids: list[str] = []
    attested = 0
    for row in gate_records:
        checker = row.get("checker") if isinstance(row, Mapping) else None
        checker = checker if isinstance(checker, Mapping) else {}
        finished_at = checker.get("finished_at")
        covered = bool(
            isinstance(receipt_model, str)
            and isinstance(captured_at, str)
            and checker.get("model") == receipt_model
            and isinstance(finished_at, str)
            and finished_at >= captured_at
        )
        if covered:
            attested += 1
        else:
            bypass_gate_ids.append(str(row.get("gate_id", "<unidentified-gate>")))
    return {
        "census_arm": CALIBRATION_BYPASS_CENSUS_ARM,
        "gates_adjudicated_under_calibration_bypass": len(bypass_gate_ids),
        "receipt_attested": attested,
        "bypass_gate_ids": bypass_gate_ids,
    }


#: Derived census arm naming (W3): gate verdicts rendered by a shadow-rehearsal
#: leg, disclosed and segregated. DISTINCT from the PD-9 bypass arm by design:
#: a shadow leg is RECEIPT-BEARING, so its verdicts are attested measurements
#: of a rehearsal world, not bypass-adjudicated production verdicts -- and they
#: are never counted as production-adjudicated (waiver-doctrine part 4 shape).
SHADOW_REHEARSAL_CENSUS_ARM = "gates_rendered_under_shadow_rehearsal"


def shadow_rehearsal_census(gate_records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive the W3 shadow-rehearsal disclosure census arm.

    Recomputed from ledger rows on every call, keyed exclusively on each
    row's own structural ``shadow_rehearsal`` marker
    (:func:`menagerie.crawler.identity.is_shadow_rehearsal_record`) -- never
    on the live environment flag, so the census reads identically after the
    rehearsal env is long gone. Marked rows are DISCLOSED, segregated from
    the production-adjudicated count, and release eligibility keeps excluding
    them (the read-side W44 exclusion plus the award-seam withhold).

    Parameters
    ----------
    gate_records:
        Gate-ledger rows (only the marker field and ``gate_id`` are read).

    Returns
    -------
    dict[str, Any]
        The census arm name, the segregated shadow/production counts, and
        the shadow gate IDs.
    """

    shadow_gate_ids: list[str] = []
    production_adjudicated = 0
    for row in gate_records:
        if isinstance(row, Mapping) and is_shadow_rehearsal_record(row):
            shadow_gate_ids.append(str(row.get("gate_id", "<unidentified-gate>")))
        else:
            production_adjudicated += 1
    return {
        "census_arm": SHADOW_REHEARSAL_CENSUS_ARM,
        "gates_rendered_under_shadow_rehearsal": len(shadow_gate_ids),
        "production_adjudicated": production_adjudicated,
        "shadow_gate_ids": shadow_gate_ids,
    }


def require_checker_calibration(gate_kind: str, checker_model: str) -> dict[str, Any]:
    """Refuse, fail-closed, unless (gate kind, model) is measured and corroborated.

    The one activation gate the real command lane consults before buying any
    session. The sanctioned calibration capture run
    (``MENAGERIE_CHECKER_CALIBRATION_RUN=1``) bypasses it explicitly -- that is
    how the receipt's own measurement sessions are bought.

    Parameters
    ----------
    gate_kind:
        ``metadata_batch``, ``fidelity``, or ``terminal_disposition``.
    checker_model:
        The pinned production checker model for that kind.

    Returns
    -------
    dict[str, Any]
        The validated receipt (empty dict during the calibration run).

    Raises
    ------
    CheckerCalibrationMissingError
        No receipt, or the receipt names a different checker model.
    CheckerCalibrationInvalidError
        Receipt malformed or under the measurement floors.
    CheckerCapacityMarginError
        The next rung does not fit the declared budgets with margin.
    CheckerGoldenCorpusError
        The golden corpus has no real transcript for (gate kind, model).
    """

    if calibration_run_active():
        return {}
    return _require_receipt_coverage(gate_kind, checker_model)


def _activation_path() -> Path:
    """Resolve the active activation record (operator override, else committed)."""

    override = os.environ.get(CHECKER_ACTIVATION_ENV, "").strip()
    return Path(override) if override else CHECKER_ACTIVATION_PATH


def _validate_activation_row(row: Any, *, source: str, index: int) -> dict[str, Any]:
    """Validate one activation-matrix row's closed shape, fail-closed."""

    label = f"checker activation {source}[{index}]"
    if not isinstance(row, dict):
        raise CheckerCalibrationInvalidError(f"{label} is not an object")
    for field in ("checker_model", "gate_kind", "checker_prompt_identity", "activated_at", "source"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise CheckerCalibrationInvalidError(f"{label}: {field} must be a non-empty string")
    if not row["checker_prompt_identity"].startswith("sha256:"):
        raise CheckerCalibrationInvalidError(
            f"{label}: checker_prompt_identity must be a sha256: digest"
        )
    for field in ("draws", "sessions_ok", "model_witnessed"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CheckerCalibrationInvalidError(f"{label}: {field} must be a non-negative integer")
    if not (row["model_witnessed"] <= row["sessions_ok"] <= row["draws"]):
        raise CheckerCalibrationInvalidError(
            f"{label}: model_witnessed <= sessions_ok <= draws must hold"
        )
    return dict(row)


def load_checker_activations(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Load and structurally validate the committed activation matrix rows.

    A missing file is an EMPTY matrix (every pair falls through to the
    receipt door); a present but malformed file refuses, because a record
    that cannot be read cannot be allowed to activate anything.
    """

    resolved = path if path is not None else _activation_path()
    if not resolved.is_file():
        return []
    source = str(resolved)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckerCalibrationInvalidError(f"checker activation {source} is unreadable: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKER_ACTIVATION_SCHEMA:
        raise CheckerCalibrationInvalidError(
            f"checker activation {source} must declare schema {CHECKER_ACTIVATION_SCHEMA}"
        )
    rows = payload.get("activations")
    if not isinstance(rows, list):
        raise CheckerCalibrationInvalidError(f"checker activation {source} lacks its activations list")
    return [_validate_activation_row(row, source=source, index=index) for index, row in enumerate(rows)]


def checker_activation_for(gate_kind: str, checker_model: str) -> Optional[dict[str, Any]]:
    """Return the live-epoch activation row covering one (kind, model), if any.

    Closed comparison: exact kind, exact model, exact structured
    ``checker_prompt_identity`` equal to the live shipped epoch, every ok
    session model-witnessed, and at least :data:`CHECKER_ACTIVATION_MIN_SESSIONS`
    of them. A row at another epoch activates nothing (a prompt rotation is a
    different checker).
    """

    live_epoch = live_checker_prompt_identity()
    for row in load_checker_activations():
        if row["gate_kind"] != gate_kind or row["checker_model"] != checker_model:
            continue
        if row["checker_prompt_identity"] != live_epoch:
            continue
        if row["sessions_ok"] < CHECKER_ACTIVATION_MIN_SESSIONS:
            continue
        if row["model_witnessed"] != row["sessions_ok"]:
            continue
        return row
    return None


def _require_receipt_coverage(gate_kind: str, checker_model: str) -> dict[str, Any]:
    """The activation-or-receipt coverage check, WITHOUT the calibration-run bypass.

    Shared by :func:`require_checker_calibration` (which bypasses only for the
    sanctioned capture run) and the W2 preflight XOR arm (which must evaluate
    the receipt side of the XOR independently of the mode flag -- a
    calibration run holding a live valid receipt is an AMBIGUOUS mode, not a
    doubly-authorized one). The activation matrix (RULING B.2) is consulted
    first; a pair it covers at the live epoch is activated and the receipt
    door is not consulted for it.
    """

    activation = checker_activation_for(gate_kind, checker_model)
    if activation is not None:
        return {"activation": activation}
    receipt = load_checker_calibration_receipt()
    if receipt["checker_model"] != checker_model:
        raise CheckerCalibrationMissingError(
            f"checker calibration receipt measures {receipt['checker_model']!r} but the "
            f"{gate_kind} lane pins {checker_model!r}; every checker flip re-calibrates "
            "before its first rung"
        )
    # R2-3: the receipt-epoch door. The comparison is CLOSED — exactly the
    # structured field against exactly the live content hash; a free-text
    # epoch inside `source` is prose, not evidence.
    live_epoch = live_checker_prompt_identity()
    recorded_epoch = receipt.get("checker_prompt_identity")
    if not isinstance(recorded_epoch, str) or not recorded_epoch.strip():
        raise CheckerCalibrationEpochError(
            f"checker calibration receipt carries no structured checker_prompt_identity; "
            f"the shipping checker prompt epoch is {live_epoch} and nothing proves the "
            "receipt measured it — re-mint the receipt at the live epoch (a prompt "
            "rotation without a re-calibration is exactly the silent staleness this "
            "door exists to refuse)"
        )
    if recorded_epoch != live_epoch:
        raise CheckerCalibrationEpochError(
            f"checker calibration receipt was minted at prompt epoch {recorded_epoch} "
            f"but the shipping checker prompt epoch is {live_epoch}; a receipt at "
            "another epoch measured a different checker — register the rotation and "
            "re-calibrate/re-mint at the live epoch before any verdict-rendering leg"
        )
    golden_models = _golden_corpus_models(gate_kind)
    if checker_model not in golden_models:
        raise CheckerGoldenCorpusError(
            f"the committed golden transcript corpus has no {gate_kind} entry recorded "
            f"under {checker_model!r} (found {sorted(golden_models) or 'none'}); "
            "re-record the goldens under the pinned checker "
            "(tests/golden_checker_transcripts/capture_golden_checker_transcripts.py) "
            "in the same reviewed change as the calibration receipt"
        )
    if not os.environ.get(CHECKER_GOLDEN_CORPUS_ENV, "").strip():
        # W-7 corpus membership floor: when production activates on the
        # COMMITTED corpus (no operator override staging a bootstrap copy),
        # the corpus must byte-match its committed manifest -- a swapped,
        # edited, or silently removed golden refuses the whole checker lane,
        # exactly as removing a smoke test without moving the inventory lock
        # refuses CI. The override env is the established test/bootstrap seam
        # (same trust class as CHECKER_CALIBRATION_RECEIPT_ENV) and the mint
        # enforces the same floor unconditionally.
        from menagerie.crawler.tools.golden_corpus_floor import (
            GoldenCorpusFloorError,
            enforce_committed_corpus,
        )

        try:
            floor_failures = enforce_committed_corpus()
        except GoldenCorpusFloorError as exc:
            raise CheckerGoldenCorpusError(
                f"golden corpus membership floor (W-7) cannot be measured: {exc}"
            ) from exc
        if floor_failures:
            raise CheckerGoldenCorpusError(
                "golden corpus membership floor (W-7): the committed corpus does not "
                "match its committed manifest -- " + "; ".join(floor_failures)
            )
    return receipt


#: The W2 preflight arm's census name and refusal shape (F-O19 scoping: the
#: arm binds legs that RENDER GATE VERDICTS; a verdict-free leg passes it
#: vacuously, and that scoping is stated here and in DX-22).
CHECKER_CALIBRATION_MODE_ARM = "checker-calibration-mode"
CHECKER_CALIBRATION_MODE_UNPROVEN = "checker-calibration-mode-unproven"


def evaluate_checker_calibration_mode(
    *,
    renders_gate_verdicts: bool,
    pinned_models: Mapping[str, str],
) -> dict[str, Any]:
    """Evaluate W2's receipt-XOR preflight arm for one launch.

    A leg that renders gate verdicts must hold EXACTLY ONE of the two
    sanctioned calibration dispositions: a valid receipt (with golden-corpus
    corroboration) covering every pinned ``(gate kind, model)`` pair, XOR the
    declared calibration capture mode (``MENAGERIE_CHECKER_CALIBRATION_RUN=1``).
    Both asserted is an AMBIGUOUS mode and refuses; neither is the
    uncalibrated launch BL-10 exists to prevent. A leg that renders no gate
    verdict (the closing licensing leg, final-sweep step 3) passes vacuously
    with that fact disclosed (F-O19). W3's shadow-rehearsal third mode is
    RECEIPT-BEARING by design: it satisfies the RECEIPT side of this XOR, so
    the arm never admits a second receiptless verdict-rendering mode -- the
    W3 change extends the ``calibration_run_active`` disjunct, never adds a
    receiptless one. Concretely: shadow mode alone with a covering receipt
    passes (the mode rides the receipt side), shadow mode without a receipt
    refuses exactly like the uncalibrated launch (the mode never substitutes
    for the receipt), and declaring calibration AND shadow-rehearsal together
    is the ambiguous mode this arm refuses as a typed refusal row.

    Parameters
    ----------
    renders_gate_verdicts:
        Whether the launch's legs render gate verdicts.
    pinned_models:
        ``gate_kind -> pinned checker model`` for every lane the launch
        adjudicates with (the caller derives it from
        ``required_checker_model``).

    Returns
    -------
    dict[str, Any]
        ``{"arm", "ok", "renders_gate_verdicts", "calibration_run_active",
        "shadow_rehearsal_active", "environment", "lanes", "refusals"}`` --
        evidence-shaped for the W4 preflight registry.
    """

    ambiguous = _ambiguous_mode_declared()
    # Raw reads under ambiguity: the arm reports the refusal as an
    # evidence-shaped refusal row instead of propagating the direct readers'
    # typed AmbiguousLegModeError, so the preflight registry keeps a record.
    calibration = os.environ.get(CHECKER_CALIBRATION_RUN_ENV, "") == "1"
    shadow = bool(os.environ.get(SHADOW_REHEARSAL_RUN_ENV, "").strip())
    lanes: list[dict[str, Any]] = []
    receipt_refusals: list[str] = []
    for gate_kind in sorted(pinned_models):
        model = pinned_models[gate_kind]
        try:
            coverage = _require_receipt_coverage(gate_kind, model)
        except CheckerCalibrationError as exc:
            lanes.append({"gate_kind": gate_kind, "model": model, "receipt": "refused"})
            receipt_refusals.append(f"{gate_kind}: {exc}")
        else:
            lanes.append(
                {
                    "gate_kind": gate_kind,
                    "model": model,
                    # Disclosed per lane: `activated` rides the RULING's
                    # activation matrix, `attested` the receipt door.
                    "receipt": "activated" if "activation" in coverage else "attested",
                }
            )
    receipt_covers_all = not receipt_refusals and bool(pinned_models)
    result: dict[str, Any] = {
        "arm": CHECKER_CALIBRATION_MODE_ARM,
        "renders_gate_verdicts": bool(renders_gate_verdicts),
        "calibration_run_active": calibration,
        "shadow_rehearsal_active": shadow,
        "environment": calibration_environment_snapshot(),
        "lanes": lanes,
        "refusals": [],
    }
    if ambiguous:
        # The closed-mode refusal outranks vacuity: an ambiguous declaration
        # is a defective launch regardless of whether this leg renders
        # verdicts, and every direct mode reader downstream would refuse too.
        result["vacuous"] = False
        result["refusals"].append(
            f"{CHECKER_CALIBRATION_MODE_UNPROVEN}: ambiguous leg mode -- "
            f"{CHECKER_CALIBRATION_RUN_ENV}=1 and {SHADOW_REHEARSAL_RUN_ENV} are "
            "both declared; the mode vocabulary is closed at "
            "normal/calibration/shadow-rehearsal -- unset one of the two variables"
        )
        result["ok"] = False
        return result
    if not renders_gate_verdicts:
        result["ok"] = True
        result["vacuous"] = True
        return result
    result["vacuous"] = False
    if calibration and receipt_covers_all:
        result["refusals"].append(
            f"{CHECKER_CALIBRATION_MODE_UNPROVEN}: the calibration capture mode is "
            "declared while a valid receipt already covers every pinned lane; the "
            "receipt XOR calibration-mode arm refuses an ambiguous mode -- unset "
            f"{CHECKER_CALIBRATION_RUN_ENV} or run under the receipt"
        )
    elif not calibration and not receipt_covers_all:
        # Shadow-rehearsal is deliberately NOT a disjunct here: the mode is
        # receipt-bearing, so a shadow leg without receipt coverage refuses
        # exactly like an uncalibrated launch (it never substitutes).
        details = " | ".join(receipt_refusals) or "no lanes pinned"
        qualifier = (
            " (the declared shadow-rehearsal mode is receipt-bearing and does not "
            "substitute for the receipt)"
            if shadow
            else ""
        )
        result["refusals"].append(
            f"{CHECKER_CALIBRATION_MODE_UNPROVEN}: this launch renders gate verdicts "
            "with NEITHER a valid calibration receipt covering every pinned lane NOR "
            f"the declared calibration capture mode{qualifier}: " + details
        )
    result["ok"] = not result["refusals"]
    return result
