"""Pre-admission host-capacity refusal for models this machine cannot instantiate.

Some catalog models cannot be built on a small host at all. Full-config
Mixtral-8x7B declares roughly 46.5B parameters; at fp32 random init that is about
186 GB of parameter tensors alone, and on a 16 GiB machine the macOS OOM killer
takes the worker down after minutes of futile allocation -- after an agentic
author session and an environment solve have already been spent.

This module refuses those models EARLY, CHEAPLY, and HONESTLY:

* :func:`assess_model_capacity` derives a *lower bound* on the model's parameter
  count from the authored recipe's own declared configuration, without importing
  torch, without importing the model's library, and without allocating anything.
* The bound is compared against a ceiling derived from the host's real physical
  memory, read at run time. On a bigger machine the same catalog admits the same
  models with no code change.
* A refusal is recorded as a durable, append-only row carrying the estimate AND
  the threshold that produced it, so a future reviewer can tell whether the call
  was right.

Three properties are deliberate:

**It is a refusal, not a relaxation.** Nothing here weakens a validation check.
A deferred model simply never reaches the environment lane, so no attempt, gate,
or record is asserted about it.

**It is biased toward attempting.** The agentic authoring stage runs ONCE per
model, so a model wrongly refused becomes a permanent dead record, while a model
wrongly admitted costs one OOM'd run. Every lever is therefore set the generous
way: the parameter estimate is a strict LOWER bound (it omits biases, norms,
positional tables, and the third gated feed-forward projection), and the host
ceiling is deliberately set ABOVE what the machine can really hold
(:data:`OVERCOMMIT_ALLOWANCE`). Only a model whose own configuration puts it
beyond that generous ceiling is refused.

**It is honest about coverage.** The estimate is derivable only when the recipe
declares a recognizable layer-stack configuration. When it is not derivable the
verdict is :data:`CapacityVerdict.NOT_DERIVABLE` and the model is ADMITTED, and
the row-level basis says so rather than pretending to completeness.
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
from pathlib import Path
import subprocess  # noqa: S404 -- fixed-argv sysctl probe, no shell, no user input
import sys
from typing import Any, BinaryIO, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    DECLARED_CONFIG_CAPACITY_ESTIMATE_BASIS,
    HOST_CAPABILITY_DEFERRAL_STATUS_CODE,
    WORKER_OBSERVED_CAPACITY_ESTIMATE_ORIGIN,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    foreign_content_address_fields,
    fsync_directory,
    row_content_address,
    stamp_row_digest,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.recordio import read_with_tail_recovery

CAPACITY_DEFERRAL_SCHEMA_VERSION = "menagerie.crawler.host-capacity-deferral.v1"

CAPACITY_DEFERRAL_DISPOSITION = HOST_CAPABILITY_DEFERRAL_STATUS_CODE
"""Greppable disposition for a model refused because this host is too small.

It follows the established ``deferred:needs-<capability>`` shape
(``needs-cuda``, ``needs-x86``, ``needs-opus-tier``, ``needs-source-access``):
a real, located model that this campaign cannot execute, naming the exact
capability that would recover it. Host memory is such a capability.

It is deliberately NOT ``failed:*``. A ``failed:`` code asserts that our pipeline
broke on a model; nothing broke here. The model is fine, the proposal is fine,
and the machine is simply too small -- a fact about the HOST, recoverable by
running the identical campaign somewhere bigger.

Since WS-2 it IS a model terminal -- but under its OWN machine-derived proof rule,
never a borrowed one. The gate-based rules (``authority._derive_deferral`` for the
platform handoff, ``reducer._derive_blocked_capability_deferral_proof`` for an
accepted BLOCKED verdict) still cannot prove it, and claiming either would mean
fabricating an adjudication this refusal does not have. What the reducer proves
instead is exactly what the machine measured: the host's physical-memory figure and
the model's own declared-configuration parameter bound, recomputed on every replay
(:func:`host_capability_claim_refusal`). That keeps the head honest AND scoped -- the
same record re-derived on a bigger host does not refuse, so the deferral re-opens
where the remedy says it should.

Only the static :data:`~menagerie.crawler.constants.DECLARED_CONFIG_CAPACITY_ESTIMATE_BASIS`
origin mints. A ``worker-observed-rss`` refusal stays exactly as it was -- durable row,
no head -- because it speaks about one run in one environment at one moment, not about
the model.
"""

CAPACITY_DEFERRAL_RELATIVE_PATH = Path("intake-extensions") / "host-capacity-deferrals.jsonl"

BYTES_PER_PARAMETER = 4
"""Worst-case parameter width for a random-init forward-only capture.

The crawler instantiates every model from a constructor at random init, on CPU,
in the default dtype. That is fp32: four bytes per parameter, no sharding, no
quantization, and no pretrained checkpoint to stream. Autograd state and
optimizer state are not involved -- the capture is a single forward pass -- so
parameter bytes, not training footprint, is the right basis.
"""

OVERCOMMIT_ALLOWANCE = 2.0
"""Multiple of physical memory a model's parameters may nominally claim.

This is the margin, and it is set generously ON PURPOSE. A host does not die the
instant a process's parameter bytes reach physical RAM: macOS compresses and
swaps, and the enforced metric the OOM killer watches
(``max(ri_resident_size, ri_phys_footprint)``, see ``worker_supervisor``) climbs
well past ``hw.memsize`` before a signal 9 arrives. The practical instantiation
ceiling measured on the 16 GiB host is nearer 4B parameters; this allowance puts
the REFUSAL threshold at roughly 8.6B parameters there, more than double the
practical ceiling.

So the band between "probably will not fit" and "is refused without trying" is
wide, and everything in it is attempted. That is the intended asymmetry: an OOM
costs one wasted run, a false refusal costs a catalog entry forever.
"""

HOST_MEMORY_ENV_VAR = "MENAGERIE_HOST_MEMORY_BYTES"
"""Operator override for the host physical-memory figure, in bytes.

Present so the check is testable and so an operator can model a larger machine
before moving a campaign to it. Raising it admits strictly more models (the
biased-toward-attempting direction); lowering it defers strictly more. It cannot
weaken any validation check either way, because this gate only decides whether a
model is attempted at all.
"""

_HIDDEN_SIZE_KEYS = ("hidden_size", "d_model", "n_embd", "embed_dim", "model_dim")
_DEPTH_KEYS = (
    "num_hidden_layers",
    "num_layers",
    "n_layer",
    "n_layers",
    "num_encoder_layers",
    "depth",
)
_INTERMEDIATE_KEYS = ("intermediate_size", "ffn_dim", "d_ff", "n_inner", "mlp_dim")
_VOCAB_KEYS = ("vocab_size", "n_vocab")
_HEADS_KEYS = ("num_attention_heads", "n_head", "num_heads")
_KV_HEADS_KEYS = ("num_key_value_heads", "num_kv_heads")
_EXPERT_KEYS = ("num_local_experts", "num_experts", "moe_num_experts", "n_routed_experts")


class CapacityVerdict(StrEnum):
    """Closed pre-admission capacity outcomes."""

    ADMIT = "admit"
    """The model fits the host ceiling, or its size could not be bounded."""

    DEFER = "defer"
    """The model's own declared configuration exceeds the host ceiling."""

    NOT_DERIVABLE = "not-derivable"
    """No parameter bound could be derived; the model is admitted regardless."""


class CapacityDeferralError(ValueError):
    """Raised when a host-capacity deferral row is malformed or cannot persist."""


@dataclass(frozen=True)
class ParameterEstimate:
    """A strict lower bound on one model's instantiated parameter count."""

    parameter_count_lower_bound: int
    basis: str
    config_path: str
    terms: JsonObject

    def to_json(self) -> JsonObject:
        """Return the immutable JSON projection of this estimate."""

        return {
            "parameter_count_lower_bound": self.parameter_count_lower_bound,
            "basis": self.basis,
            "config_path": self.config_path,
            "terms": dict(self.terms),
            "bytes_per_parameter": BYTES_PER_PARAMETER,
            "estimated_parameter_bytes": (self.parameter_count_lower_bound * BYTES_PER_PARAMETER),
        }


@dataclass(frozen=True)
class HostCapacity:
    """The host memory figure and the parameter ceiling derived from it."""

    physical_memory_bytes: int
    memory_source: str

    @property
    def admissible_parameter_bytes(self) -> int:
        """Return the generous parameter-byte budget this host will attempt."""

        return int(self.physical_memory_bytes * OVERCOMMIT_ALLOWANCE)

    @property
    def admissible_parameter_ceiling(self) -> int:
        """Return the parameter count above which a model is refused outright."""

        return self.admissible_parameter_bytes // BYTES_PER_PARAMETER

    def to_json(self) -> JsonObject:
        """Return the immutable JSON projection of this threshold."""

        return {
            "host_physical_memory_bytes": self.physical_memory_bytes,
            "host_memory_source": self.memory_source,
            "overcommit_allowance": OVERCOMMIT_ALLOWANCE,
            "admissible_parameter_bytes": self.admissible_parameter_bytes,
            "admissible_parameter_ceiling": self.admissible_parameter_ceiling,
        }


@dataclass(frozen=True)
class CapacityAssessment:
    """One complete pre-admission capacity decision with its full justification."""

    verdict: CapacityVerdict
    host: HostCapacity
    estimate: Optional[ParameterEstimate]
    explanation: str

    @property
    def deferred(self) -> bool:
        """Return whether this assessment refuses the model on this host."""

        return self.verdict is CapacityVerdict.DEFER

    def to_json(self) -> JsonObject:
        """Return the immutable JSON projection of this decision."""

        payload: JsonObject = {
            "verdict": self.verdict.value,
            "threshold": self.host.to_json(),
            "explanation": self.explanation,
            "estimate": self.estimate.to_json() if self.estimate is not None else None,
        }
        if self.estimate is not None and self.host.admissible_parameter_ceiling > 0:
            payload["overage_ratio"] = round(
                self.estimate.parameter_count_lower_bound / self.host.admissible_parameter_ceiling,
                4,
            )
        return payload


def host_capacity(*, environ: Optional[Mapping[str, str]] = None) -> HostCapacity:
    """Read this host's real physical memory and derive its parameter ceiling.

    The figure is TOTAL physical memory rather than momentarily-free memory. Free
    memory fluctuates with whatever else the machine is doing, and a fluctuating
    threshold would make the same model refuse on one run and admit on the next
    -- unacceptable when the authoring stage runs once per model. Total physical
    memory is stable, machine-derived, and moves automatically when the campaign
    is re-run on bigger hardware, which is the whole point.

    Parameters
    ----------
    environ:
        Environment mapping to consult for the operator override. Defaults to
        the live process environment.

    Returns
    -------
    HostCapacity
        Host memory figure, its provenance, and the derived ceiling.

    Raises
    ------
    CapacityDeferralError
        If the override is present but not a positive integer.
    """

    source_environ = os.environ if environ is None else environ
    override = source_environ.get(HOST_MEMORY_ENV_VAR)
    if override is not None:
        try:
            override_bytes = int(override)
        except ValueError as exc:
            raise CapacityDeferralError(
                f"{HOST_MEMORY_ENV_VAR} must be an integer byte count, got {override!r}"
            ) from exc
        if override_bytes <= 0:
            raise CapacityDeferralError(f"{HOST_MEMORY_ENV_VAR} must be positive")
        return HostCapacity(
            physical_memory_bytes=override_bytes,
            memory_source=f"env:{HOST_MEMORY_ENV_VAR}",
        )
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        pages = -1
        page_size = -1
    if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
        return HostCapacity(
            physical_memory_bytes=pages * page_size,
            memory_source="sysconf:SC_PHYS_PAGES*SC_PAGE_SIZE",
        )
    if sys.platform == "darwin":
        try:
            observed = subprocess.run(  # noqa: S603 -- fixed argv, no shell
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                capture_output=True,
                check=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            memsize = int(observed)
        except (OSError, subprocess.SubprocessError, ValueError):
            memsize = 0
        if memsize > 0:
            return HostCapacity(
                physical_memory_bytes=memsize,
                memory_source="sysctl:hw.memsize",
            )
    raise CapacityDeferralError(
        f"host physical memory is unreadable; set {HOST_MEMORY_ENV_VAR} to declare it explicitly"
    )


def _positive_int(candidate: Any) -> Optional[int]:
    """Return a strictly positive integer value, or ``None`` for anything else."""

    if isinstance(candidate, bool) or not isinstance(candidate, int):
        return None
    return candidate if candidate > 0 else None


def _first_key(
    config: Mapping[str, Any], keys: Sequence[str]
) -> tuple[Optional[str], Optional[int]]:
    """Return the first declared positive integer among ``keys``."""

    for key in keys:
        value = _positive_int(config.get(key))
        if value is not None:
            return key, value
    return None, None


def _candidate_configs(node: Any, path: str) -> list[tuple[str, Mapping[str, Any]]]:
    """Collect every nested keyword mapping in a recipe, with its dotted path.

    Recipes nest constructor payloads (``{"__construct__": {"kwargs": {...}}}``)
    to arbitrary depth, and different rungs put the configuration in different
    places. Rather than hardcode one shape, every mapping in the recipe is a
    candidate and the caller picks the one that actually declares a layer stack.

    Parameters
    ----------
    node:
        Recipe fragment being walked.
    path:
        Dotted path of ``node`` within the recipe, for provenance.

    Returns
    -------
    list[tuple[str, Mapping[str, Any]]]
        Every nested mapping, outermost first, paired with its dotted path.
    """

    found: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(node, Mapping):
        found.append((path, node))
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            found.extend(_candidate_configs(value, child))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            found.extend(_candidate_configs(value, f"{path}[{index}]"))
    return found


def estimate_parameter_count(recipe: Any) -> Optional[ParameterEstimate]:
    """Derive a strict lower bound on a recipe's instantiated parameter count.

    The bound is assembled from the configuration the AUTHOR already declared --
    no import, no construction, no allocation. Every term is deliberately
    understated so the result can only ever be smaller than the real model:

    * Embeddings count ``vocab_size * hidden`` once, doubled only when the config
      explicitly says the output head is untied.
    * Attention counts the four projections at their declared head geometry, and
      falls back to square projections when head counts are absent.
    * The feed-forward block counts TWO projections per expert, never three, even
      though gated architectures (Mixtral, Llama, and every SwiGLU descendant)
      have three. For Mixtral this alone understates the model by about a third.
    * Biases, normalization scales, positional tables, and every auxiliary head
      are omitted entirely.

    Parameters
    ----------
    recipe:
        Authored recipe fragment, typically
        ``proposal["proposed_facts"]["implementation"]["library_recipe"]``.

    Returns
    -------
    ParameterEstimate | None
        Lower bound and its provenance, or ``None`` when the recipe declares no
        recognizable layer stack. ``None`` means "unknown", never "small".
    """

    for path, config in _candidate_configs(recipe, ""):
        hidden_key, hidden = _first_key(config, _HIDDEN_SIZE_KEYS)
        depth_key, depth = _first_key(config, _DEPTH_KEYS)
        if hidden is None or depth is None:
            continue
        terms: JsonObject = {
            "hidden_size": {"key": hidden_key, "value": hidden},
            "depth": {"key": depth_key, "value": depth},
        }

        embedding = 0
        vocab_key, vocab = _first_key(config, _VOCAB_KEYS)
        if vocab is not None:
            # Untied output heads double the embedding cost. `tie_word_embeddings`
            # defaults to True across the config families this recognizes, so an
            # absent flag counts the table ONCE -- the smaller, safer reading.
            untied = config.get("tie_word_embeddings") is False
            embedding = vocab * hidden * (2 if untied else 1)
            terms["embedding"] = {
                "key": vocab_key,
                "vocab_size": vocab,
                "untied_output_head": untied,
                "parameters": embedding,
            }

        _heads_key, heads = _first_key(config, _HEADS_KEYS)
        _kv_key, kv_heads = _first_key(config, _KV_HEADS_KEYS)
        head_dim = _positive_int(config.get("head_dim"))
        if head_dim is None and heads is not None and hidden % heads == 0:
            head_dim = hidden // heads
        if head_dim is not None and heads is not None:
            kv_dim = (kv_heads if kv_heads is not None else heads) * head_dim
            attention = (
                hidden * (heads * head_dim) + (heads * head_dim) * hidden + 2 * (hidden * kv_dim)
            )
        else:
            # No head geometry declared: four square projections is the standard
            # shape and remains a lower bound for grouped-query variants.
            attention = 4 * hidden * hidden
        terms["attention_per_layer"] = {
            "heads": heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "parameters": attention,
        }

        intermediate_key, intermediate = _first_key(config, _INTERMEDIATE_KEYS)
        if intermediate is None:
            # Transformer feed-forward blocks are conventionally 4x the model
            # width; using that ratio keeps a stack without a declared
            # intermediate size in scope while staying a lower bound for the
            # wider designs that omit the field.
            intermediate = 4 * hidden
        _expert_key, experts = _first_key(config, _EXPERT_KEYS)
        expert_count = experts if experts is not None else 1
        feedforward = expert_count * 2 * hidden * intermediate
        terms["feedforward_per_layer"] = {
            "key": intermediate_key,
            "intermediate_size": intermediate,
            "intermediate_declared": intermediate_key is not None,
            "experts": expert_count,
            "projections_counted": 2,
            "parameters": feedforward,
        }

        total = embedding + depth * (attention + feedforward)
        terms["per_layer_parameters"] = attention + feedforward
        return ParameterEstimate(
            parameter_count_lower_bound=total,
            basis="declared-config-layer-stack-lower-bound",
            config_path=path or "<recipe>",
            terms=terms,
        )
    return None


def assess_model_capacity(
    recipe: Any,
    *,
    host: Optional[HostCapacity] = None,
) -> CapacityAssessment:
    """Decide whether this host should attempt to instantiate one recipe.

    Parameters
    ----------
    recipe:
        Authored recipe fragment carrying the model's declared configuration.
    host:
        Host capacity to compare against. Read from the machine when omitted.

    Returns
    -------
    CapacityAssessment
        Verdict, the host threshold, the estimate, and a plain-language reason.
    """

    resolved_host = host_capacity() if host is None else host
    estimate = estimate_parameter_count(recipe)
    if estimate is None:
        return CapacityAssessment(
            verdict=CapacityVerdict.NOT_DERIVABLE,
            host=resolved_host,
            estimate=None,
            explanation=(
                "the recipe declares no recognizable layer stack, so no parameter "
                "bound could be derived; the model is admitted and will be attempted"
            ),
        )
    ceiling = resolved_host.admissible_parameter_ceiling
    if estimate.parameter_count_lower_bound > ceiling:
        return CapacityAssessment(
            verdict=CapacityVerdict.DEFER,
            host=resolved_host,
            estimate=estimate,
            explanation=(
                f"declared configuration bounds the model below by "
                f"{estimate.parameter_count_lower_bound:,} parameters "
                f"({estimate.parameter_count_lower_bound * BYTES_PER_PARAMETER:,} bytes at "
                f"fp32 random init), above this host's ceiling of {ceiling:,} parameters "
                f"({resolved_host.physical_memory_bytes:,} bytes of physical memory "
                f"times an allowance of {OVERCOMMIT_ALLOWANCE})"
            ),
        )
    return CapacityAssessment(
        verdict=CapacityVerdict.ADMIT,
        host=resolved_host,
        estimate=estimate,
        explanation=(
            f"declared configuration bounds the model below by "
            f"{estimate.parameter_count_lower_bound:,} parameters, within this host's "
            f"ceiling of {ceiling:,}"
        ),
    )


def capacity_deferral_path(records_root: Path) -> Path:
    """Return the canonical host-capacity deferral ledger below one records root.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        Append-only host-capacity deferral JSONL path.
    """

    return records_root / CAPACITY_DEFERRAL_RELATIVE_PATH


def build_capacity_deferral_row(
    *,
    stable_id: str,
    work_id: str,
    name: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
    assessment: CapacityAssessment,
) -> JsonObject:
    """Assemble one complete, self-describing host-capacity deferral row.

    The row states BOTH the estimate and the threshold that produced it, so a
    later reviewer can re-derive the decision and judge whether it was right,
    without needing the code that made it.

    Parameters
    ----------
    stable_id, work_id, name:
        Durable model identity and the active scheduled work generation.
    campaign_id, run_id, machine_id:
        Campaign and host provenance for the refusal.
    created_at:
        UTC timestamp of the decision.
    assessment:
        The refusing capacity decision.

    Returns
    -------
    dict[str, Any]
        Content-addressed deferral row.

    Raises
    ------
    CapacityDeferralError
        If the assessment does not actually refuse the model.
    """

    if not campaign_id or campaign_id == str(None):
        # T2-12: the pilot froze campaign_id="None" (a stringified absence)
        # into 62 row digests. The absence must refuse at BUILD time; read-side
        # validators stay tolerant so those historical rows remain readable.
        raise ValueError("campaign_id must name a real campaign, never a stringified absence")
    if not assessment.deferred:
        raise CapacityDeferralError(
            "only a refusing capacity assessment may be recorded as a deferral"
        )
    assert assessment.estimate is not None  # guaranteed by CapacityVerdict.DEFER
    required_bytes = assessment.estimate.parameter_count_lower_bound * BYTES_PER_PARAMETER
    row: JsonObject = {
        "schema_version": CAPACITY_DEFERRAL_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": work_id,
        "name": name,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "created_at": created_at,
        "disposition": CAPACITY_DEFERRAL_DISPOSITION,
        "capacity": assessment.to_json(),
        "recheck_hint": (
            "re-run this campaign on a host with at least "
            f"{required_bytes / OVERCOMMIT_ALLOWANCE / 2**30:.1f} GiB of physical memory, "
            "or set "
            f"{HOST_MEMORY_ENV_VAR} to that machine's byte count"
        ),
    }
    row = stamp_row_digest(row)
    return row


def validate_capacity_deferral_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one host-capacity deferral row and return its canonical copy.

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
    CapacityDeferralError
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
        "capacity",
        "recheck_hint",
        "row_sha256",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise CapacityDeferralError(
            f"host-capacity deferral row is missing {', '.join(sorted(missing))}"
        )
    if row["schema_version"] != CAPACITY_DEFERRAL_SCHEMA_VERSION:
        raise CapacityDeferralError(
            f"host-capacity deferral row declares unknown schema {row['schema_version']!r}"
        )
    if row["disposition"] != CAPACITY_DEFERRAL_DISPOSITION:
        raise CapacityDeferralError(
            f"host-capacity deferral row declares unknown disposition {row['disposition']!r}"
        )
    for field in ("stable_id", "work_id", "name", "campaign_id", "run_id", "machine_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise CapacityDeferralError(f"host-capacity deferral row has an empty {field}")
    capacity = row["capacity"]
    if not isinstance(capacity, Mapping) or capacity.get("verdict") != CapacityVerdict.DEFER.value:
        raise CapacityDeferralError("host-capacity deferral row does not record a refusal")
    estimate = capacity.get("estimate")
    threshold = capacity.get("threshold")
    if not isinstance(estimate, Mapping) or not isinstance(threshold, Mapping):
        raise CapacityDeferralError(
            "host-capacity deferral row must state both its estimate and its threshold"
        )
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise CapacityDeferralError(
            f"host-capacity deferral row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise CapacityDeferralError("host-capacity deferral row digest does not match its payload")
    validated["row_sha256"] = digest
    return validated


def append_capacity_deferral_row(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one host-capacity deferral durably and idempotently.

    Parameters
    ----------
    path:
        Destination host-capacity deferral JSONL.
    row:
        Complete typed deferral row.

    Returns
    -------
    dict[str, Any]
        Validated persisted row.

    Raises
    ------
    CapacityDeferralError
        If the row is malformed.

    Notes
    -----
    The FIRST refusal recorded for a work generation is the record, and a later
    run that re-derives a refusal for the same generation is a no-op. This is
    deliberate: ``created_at`` (and possibly ``run_id``) necessarily differ on
    every resume, so treating a re-derived refusal as a conflict would abort the
    driver on the second pass over the same oversized model. A model that has
    stopped being too large is not deferred at all and writes nothing here, so
    a stale row can only ever be superseded by the model completing normally.
    """

    validated = validate_capacity_deferral_row(row)
    key = (validated["stable_id"], validated["work_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _decode_capacity_lines(_ledger_bytes(path, handle), path)
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
        prefix_check=lambda retained: _decode_capacity_lines(retained.read_bytes(), path),
        locked_handle=handle,
    )


def _decode_capacity_lines(payload: bytes, path: Path) -> tuple[JsonObject, ...]:
    """Decode and validate every row in one deferral ledger's bytes."""

    rows: list[JsonObject] = []
    for index, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CapacityDeferralError(
                f"host-capacity deferral ledger {path} has an unreadable row at line {index}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise CapacityDeferralError(
                f"host-capacity deferral ledger {path} has a non-object row at line {index}"
            )
        rows.append(validate_capacity_deferral_row(decoded))
    return tuple(rows)


def load_capacity_deferral_rows(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Load host-capacity deferrals from one or more campaign roots.

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
        for row in _decode_capacity_lines(_ledger_bytes(path), path):
            by_key[(row["stable_id"], row["work_id"])] = row
    return tuple(by_key[key] for key in sorted(by_key))


# ---------------------------------------------------------------------------
# WS-2: the host-capability model terminal
# ---------------------------------------------------------------------------

HOST_CAPABILITY_CLAIM_FIELDS: tuple[str, ...] = (
    "machine_id",
    "host_physical_memory_bytes",
    "host_memory_source",
    "overcommit_allowance",
    "bytes_per_parameter",
    "admissible_parameter_ceiling",
    "parameter_count_lower_bound",
    "required_physical_memory_bytes",
    "estimate_basis",
    "deferral_row_sha256",
)
"""The closed field set of a host-capability terminal claim.

Every field is an INPUT to the refusal arithmetic or to the scope of the fact, so the
reducer can recompute the whole decision from the claim alone. Nothing here is a verdict
a producer asserts: the verdict is re-derived (:func:`host_capability_claim_refusal`),
which is why a forged field fails closed instead of laundering a head.
"""


def capacity_estimate_provenance(row: Mapping[str, Any]) -> Optional[str]:
    """Classify one capacity row's estimate against the closed origin vocabulary.

    The two live origins spell themselves differently in the ledger: the static
    assessment writes ``estimate.basis`` (:meth:`ParameterEstimate.to_json`) and the
    worker-observed path writes ``estimate.origin``
    (``driver_receipts._worker_observed_rss_deferral_row``). Both spellings are read
    here, so the classification is a fact about the ROW rather than about which writer
    produced it, and anything outside the closed set is ``None`` -- unknown provenance
    never mints.

    Parameters
    ----------
    row:
        Candidate host-capacity deferral row.

    Returns
    -------
    str | None
        The closed origin token, or ``None`` when the row's provenance is not one this
        campaign knows how to adjudicate.
    """

    capacity = row.get("capacity")
    if not isinstance(capacity, Mapping):
        return None
    estimate = capacity.get("estimate")
    if not isinstance(estimate, Mapping):
        return None
    if estimate.get("basis") == DECLARED_CONFIG_CAPACITY_ESTIMATE_BASIS:
        return DECLARED_CONFIG_CAPACITY_ESTIMATE_BASIS
    if estimate.get("origin") == WORKER_OBSERVED_CAPACITY_ESTIMATE_ORIGIN:
        return WORKER_OBSERVED_CAPACITY_ESTIMATE_ORIGIN
    return None


def _is_sha256_digest(candidate: str) -> bool:
    """Return whether one string is a canonical ``sha256:<64 hex>`` digest."""

    prefix, _, hexits = candidate.partition(":")
    if prefix != "sha256" or len(hexits) != 64:
        return False
    return all(character in "0123456789abcdef" for character in hexits)


def host_capability_claim_row_mismatch(
    claim: Mapping[str, Any], row: Mapping[str, Any]
) -> Optional[str]:
    """Return why a claim does not belong to this exact deferral row, if it does not.

    Well-formedness is not identity: a claim carrying some OTHER member's digest
    recomputes perfectly on its own terms. Every mint and every replay therefore
    compares the claim against the row it is supposed to project, so a claim can never
    travel between members or between generations of one member.

    Parameters
    ----------
    claim:
        Candidate machine-derived capability claim.
    row:
        The durable deferral row the claim must project.

    Returns
    -------
    str | None
        Closed mismatch reason, or ``None`` when the pair agrees exactly.
    """

    refusal = host_capability_claim_refusal(claim)
    if refusal is not None:
        return refusal
    if not mints_host_capability_terminal(row):
        return "row is not a static declared-config host-capacity refusal"
    if claim["deferral_row_sha256"] != str(row.get("row_sha256")):
        return "host-capability claim does not bind this deferral row"
    expected = build_host_capability_claim(row)
    if dict(claim) != expected:
        return "host-capability claim does not project this deferral row"
    return None


def _positive_number(candidate: Any) -> Optional[float]:
    """Return a strictly positive real number, or ``None`` for anything else."""

    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        return None
    value = float(candidate)
    if value != value or value <= 0.0:  # NaN or non-positive
        return None
    return value


def required_physical_memory_bytes(row: Mapping[str, Any]) -> Optional[int]:
    """Return the physical memory this row's own remedy asks for, in bytes.

    This is the LEDGER's figure -- the ``recheck_hint`` number -- not the fp32
    parameter-byte total. The hint is ``parameter_bytes / OVERCOMMIT_ALLOWANCE``: the
    smallest host whose generous ceiling would have admitted the model, which is the
    honest routing number. The fp32 total (exactly twice it) is the conservative
    ceiling: useful for headroom, wrong for "which machine can run this". The step-100
    labs' apparent ~2x disagreement about these five was this identity, not a dispute.

    Parameters
    ----------
    row:
        Host-capacity deferral row carrying a static parameter bound.

    Returns
    -------
    int | None
        Required physical bytes, or ``None`` when the row states no static bound.
    """

    if capacity_estimate_provenance(row) != DECLARED_CONFIG_CAPACITY_ESTIMATE_BASIS:
        return None
    estimate = row["capacity"]["estimate"]
    bound = _positive_number(estimate.get("parameter_count_lower_bound"))
    if bound is None:
        return None
    width = _positive_number(estimate.get("bytes_per_parameter", BYTES_PER_PARAMETER)) or float(
        BYTES_PER_PARAMETER
    )
    return int(bound * width / OVERCOMMIT_ALLOWANCE)


def mints_host_capability_terminal(row: Mapping[str, Any]) -> bool:
    """Return whether this exact row may mint a host-capability model head.

    The trigger is deliberately NARROW -- two independent conditions, both required:

    * the disposition is the host-capacity one. Every other valve disposition
      (environment rebuild, coverage, host disk, checker lane, the two park kinds)
      stays "withheld, not settled". Reconciling settlement to the leg's whole stash
      would have settled out 55 members whose deferrals were transient, five of whom
      went on to earn awards;
    * the estimate is the STATIC declared-config bound. The ``worker-observed-rss``
      origin speaks about one run in one environment, and its exemplar (m7637) holds 21
      attempts, 5 gates, 13 artifacts and a real failure head; terminalizing that class
      freezes recoverable members out of later generations.

    Parameters
    ----------
    row:
        Candidate deferral row, already digest-validated by its loader.

    Returns
    -------
    bool
        Whether a head is derivable from this row.
    """

    if row.get("disposition") != CAPACITY_DEFERRAL_DISPOSITION:
        return False
    if capacity_estimate_provenance(row) != DECLARED_CONFIG_CAPACITY_ESTIMATE_BASIS:
        return False
    return required_physical_memory_bytes(row) is not None


def build_host_capability_claim(row: Mapping[str, Any]) -> JsonObject:
    """Project one capacity row into its machine-derived terminal claim.

    Parameters
    ----------
    row:
        Host-capacity deferral row that :func:`mints_host_capability_terminal` accepts.

    Returns
    -------
    dict[str, Any]
        Closed claim carrying every input the reducer recomputes.

    Raises
    ------
    CapacityDeferralError
        If the row may not mint a terminal at all.
    """

    if not mints_host_capability_terminal(row):
        raise CapacityDeferralError(
            "only a static declared-config host-capacity refusal may claim a terminal"
        )
    capacity = row["capacity"]
    estimate = capacity["estimate"]
    threshold = capacity["threshold"]
    required = required_physical_memory_bytes(row)
    assert required is not None  # guaranteed by mints_host_capability_terminal
    return {
        "machine_id": str(row["machine_id"]),
        "host_physical_memory_bytes": int(threshold["host_physical_memory_bytes"]),
        "host_memory_source": str(threshold["host_memory_source"]),
        "overcommit_allowance": float(threshold.get("overcommit_allowance", OVERCOMMIT_ALLOWANCE)),
        "bytes_per_parameter": int(estimate.get("bytes_per_parameter", BYTES_PER_PARAMETER)),
        "admissible_parameter_ceiling": int(threshold["admissible_parameter_ceiling"]),
        "parameter_count_lower_bound": int(estimate["parameter_count_lower_bound"]),
        "required_physical_memory_bytes": required,
        "estimate_basis": DECLARED_CONFIG_CAPACITY_ESTIMATE_BASIS,
        "deferral_row_sha256": str(row["row_sha256"]),
    }


def host_capability_claim_refusal(claim: Mapping[str, Any]) -> Optional[str]:
    """Recompute a host-capability claim and return its refusal reason, if any.

    This is the replay half of the proof: the head is never trusted on the claim's own
    say-so. The ceiling is re-derived from the stated host figure and the stated bound
    is re-checked against it, so a tampered field refuses the head instead of
    laundering one -- lowering the bound, inflating the host figure, inflating the
    ceiling, erasing the machine scope, forging the row binding, and smuggling the
    observed-RSS basis all land here.

    Parameters
    ----------
    claim:
        Candidate machine-derived capability claim.

    Returns
    -------
    str | None
        Closed refusal reason, or ``None`` when the claim recomputes exactly.
    """

    missing = [field for field in HOST_CAPABILITY_CLAIM_FIELDS if field not in claim]
    if missing:
        return f"host-capability claim is missing {', '.join(sorted(missing))}"
    if set(claim) - set(HOST_CAPABILITY_CLAIM_FIELDS):
        return "host-capability claim carries fields outside its closed set"
    if claim["estimate_basis"] != DECLARED_CONFIG_CAPACITY_ESTIMATE_BASIS:
        return "host-capability claim does not rest on the static declared-config bound"
    machine_id = claim["machine_id"]
    if not isinstance(machine_id, str) or not machine_id.strip():
        return "host-capability claim is not scoped to a named machine"
    row_digest = claim["deferral_row_sha256"]
    if not isinstance(row_digest, str) or not _is_sha256_digest(row_digest):
        return "host-capability claim does not bind its durable deferral row"
    source = claim["host_memory_source"]
    if not isinstance(source, str) or not source.strip():
        return "host-capability claim does not name how the host figure was measured"
    memory = _positive_number(claim["host_physical_memory_bytes"])
    allowance = _positive_number(claim["overcommit_allowance"])
    width = _positive_number(claim["bytes_per_parameter"])
    ceiling = _positive_number(claim["admissible_parameter_ceiling"])
    bound = _positive_number(claim["parameter_count_lower_bound"])
    required = _positive_number(claim["required_physical_memory_bytes"])
    if (
        memory is None
        or allowance is None
        or width is None
        or ceiling is None
        or bound is None
        or required is None
    ):
        return "host-capability claim carries a non-positive measurement"
    recomputed_ceiling = int(memory * allowance) // int(width)
    if int(ceiling) != recomputed_ceiling:
        return "host-capability claim ceiling does not recompute from its host figure"
    if int(bound) <= recomputed_ceiling:
        return "host-capability claim does not actually refuse the model on this host"
    if int(required) != int(bound * width / allowance):
        return "host-capability claim remedy does not recompute from its own bound"
    return None


def host_capability_still_refuses(
    claim: Mapping[str, Any], *, host: Optional[HostCapacity] = None
) -> bool:
    """Return whether a recorded capability refusal still holds on THIS host.

    The one route out of the head that needs no operator at all: run the identical
    campaign somewhere bigger and the recorded fact simply stops being true. Everything
    errs toward RE-OPENING -- an unreadable host figure or an unrecomputable claim
    releases the model rather than freezing it past its own remedy.

    Parameters
    ----------
    claim:
        Recorded machine-derived capability claim.
    host:
        Host to judge against. Read from the machine when omitted.

    Returns
    -------
    bool
        Whether the model is still refused here.
    """

    if host_capability_claim_refusal(claim) is not None:
        return False
    try:
        resolved = host_capacity() if host is None else host
    except CapacityDeferralError:
        return False
    bound = _positive_number(claim["parameter_count_lower_bound"])
    if bound is None:
        return False
    return int(bound) > resolved.admissible_parameter_ceiling


def host_capability_terminal_refusal(
    row: Mapping[str, Any], *, source_resolved: bool
) -> Optional[str]:
    """Return why this row may not become a model head yet, if it may not.

    Beyond the row's own eligibility there is one record-level precondition: a
    ``deferred:*`` model revision must still carry an exact public primary source link,
    so a member whose source was never resolved has no honest head available. That is
    exactly the carried five's situation -- their intake rows say
    ``legacy-source-unresolved`` and no author ever published a resolution -- and it is
    why WS-2 is FORWARD-LOOKING: the five are ROUTED to a capable host at step-200
    staging, never back-filled with a head no evidence supports.

    Parameters
    ----------
    row:
        Candidate host-capacity deferral row.
    source_resolved:
        Whether an author-published exact primary source exists for the member.

    Returns
    -------
    str | None
        Closed refusal reason, or ``None`` when the head may be minted.
    """

    if not mints_host_capability_terminal(row):
        return "row is not a static declared-config host-capacity refusal"
    if not source_resolved:
        return "member has no author-published exact primary source to record"
    return host_capability_claim_refusal(build_host_capability_claim(row))


def host_capability_preflight(
    target_ids: Sequence[str],
    *,
    deferral_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    host: Optional[HostCapacity] = None,
) -> JsonObject:
    """Census which members of a leg's target set this host cannot run at all.

    The cheapest diagnostic nobody ran in three days. Eight finishing legs (~23-28h
    wall) chased a halt condition whose blocker was already printed in a
    ``recheck_hint`` about 200 times; this prints it ONCE, at leg start, in
    needs-vs-actual form, and reports whether the whole target set is unrunnable so the
    caller can refuse to launch before buying a single author session.

    Parameters
    ----------
    target_ids:
        The leg's scheduled stable IDs.
    deferral_rows:
        Durable valve rows per stable ID (the driver's own validated read).
    host:
        Host to judge against. Read from the machine when omitted.

    Returns
    -------
    dict[str, Any]
        Census: the host figure, target/deferred counts, per-member needs-vs-actual, and
        the ``fully_host_deferred`` launch-refusal flag.
    """

    try:
        resolved = host_capacity() if host is None else host
    except CapacityDeferralError:
        resolved = None
    members: list[JsonObject] = []
    for stable_id in target_ids:
        for row in deferral_rows.get(stable_id, ()):
            if not mints_host_capability_terminal(row):
                continue
            claim = build_host_capability_claim(row)
            if resolved is not None and not host_capability_still_refuses(claim, host=resolved):
                continue
            members.append(
                {
                    "stable_id": stable_id,
                    "name": str(row.get("name", "")),
                    "machine_id": claim["machine_id"],
                    "required_physical_memory_bytes": claim["required_physical_memory_bytes"],
                    "host_physical_memory_bytes": (
                        resolved.physical_memory_bytes if resolved is not None else None
                    ),
                    "parameter_count_lower_bound": claim["parameter_count_lower_bound"],
                    "recheck_hint": str(row.get("recheck_hint", "")),
                }
            )
            break
    ordered = sorted(members, key=lambda member: str(member["stable_id"]))
    target_count = len(tuple(target_ids))
    return {
        "host_physical_memory_bytes": (
            resolved.physical_memory_bytes if resolved is not None else None
        ),
        "host_memory_source": None if resolved is None else resolved.memory_source,
        "admissible_parameter_ceiling": (
            None if resolved is None else resolved.admissible_parameter_ceiling
        ),
        "target_count": target_count,
        "host_deferred_count": len(ordered),
        "members": ordered,
        # A leg with nothing it can run is not a leg. Refusing here costs one
        # message; NOT refusing cost step-100 eight legs and ~25 hours of wall.
        "fully_host_deferred": bool(target_count) and len(ordered) == target_count,
    }
