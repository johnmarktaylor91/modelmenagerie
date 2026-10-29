"""One-model execution worker that emits observations, never ``runs`` authority."""

from __future__ import annotations

import argparse
import fcntl
import inspect
import json
import os
import random
import re
import select
import signal
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

# Deliberately the LEAF module, never authority: the worker bootstrap closure
# must import inside sealed routed envs with no driver-side deps (DX-3/RE-3;
# test_worker_lean_bootstrap pins the closure).
from menagerie.crawler.completion_line import completion_line_for_raw_award_receipt
from menagerie.crawler.backend import (
    BackendProtocolUnimplementedError,
    Framework,
    framework_capability,
    normalize_framework,
)
from menagerie.crawler.constants import RunMode
from menagerie.crawler.frameworks import NativeForwardAdapter
from menagerie.crawler.identity import (
    atomic_replace_bytes,
    canonical_json_bytes,
    compute_recipe_revision,
    hash_bytes,
    stable_hash,
    stable_leaf_hash,
)
from menagerie.crawler.modes import (
    classify_train_eval_divergence,
    detect_meaningful_modes,
    output_signature,
    output_value_sha256,
)
from menagerie.crawler.policy import ExecutionPolicy, PolicyObservation, PolicyViolation
from menagerie.crawler.recipe import (
    LoadedRecipe,
    RecipeError,
    load_recipe,
    resolve_input_constructor,
)
from menagerie.crawler.payload_codec import DtypeVocabularyError, canonical_dtype
from menagerie.crawler.standard_inputs import (
    ASSET_ROOT,
    NUMPY_TARGETS,
    InputSpec,
    materialize_standard_input,
)

_PARENT_COMPLETION_CHALLENGE_ENV = "MENAGERIE_PARENT_COMPLETION_CHALLENGE"
_WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V1 "
_WORKER_COMPLETION_V2_PREFIX = "MENAGERIE_WORKER_COMPLETION_V2 "
_REQUEST_SHA256_ENV = "MENAGERIE_WORKER_REQUEST_SHA256"
_READ_MANIFEST_ID_ENV = "MENAGERIE_EXECUTION_READ_MANIFEST_ID"
# Dedicated v3 completion channel (BL-12(a)). These constants intentionally
# mirror worker_supervisor.WORKER_COMPLETION_* -- the worker must stay importable
# without the parent-side supervisor module; a drift tripwire test pins them.
_WORKER_COMPLETION_FILE_NAME = "completion.json"
_WORKER_COMPLETION_FILE_VERSION = "menagerie.crawler.worker-completion-file.v1"
_WORKER_COMPLETION_PROOF_VERSION = "menagerie.crawler.worker-completion-proof.v1"
_WORKER_LOCK_FD_ENV = "MENAGERIE_WORKER_LOCK_FD"
_LIFECYCLE_FD_ENV = "MENAGERIE_WORKER_LIFECYCLE_FD"
_GLOBAL_LOCK_FD_ENV = "MENAGERIE_WORKER_GLOBAL_LOCK_FD"
_PARENT_STANDARD_INPUT_ASSET_ENV = "MENAGERIE_PARENT_STANDARD_INPUT_ASSET"


@dataclass(frozen=True)
class WorkerRequest:
    """Complete typed request for one isolated model execution.

    Parameters
    ----------
    stable_id:
        Durable model identity used only for receipt association.
    recipe:
        Closed declarative or typed-adapter recipe.
    modality, input_spec:
        Source-gated modality and concrete single-tensor call contract.
    scratch_root, receipt_path:
        Writable worker and atomic result locations.
    seed, input_seed, device, framework:
        Deterministic native execution settings.
    mode:
        Explicit single mode for this fresh worker process. ``None`` is retained only
        for compatibility with pre-round-6 requests.
    meaningful_modes:
        Explicit meaningful modes. ``None`` conservatively defaults to both.
    source_identity, execution_identity:
        Parent-computed identities echoed for receipt binding, never awarded.
    recipe_revision, recipe_identity_payload:
        Parent-accepted revision and the complete non-self-referential facts the
        worker recomputes before importing any recipe implementation.
    """

    stable_id: str
    recipe: Mapping[str, Any]
    modality: Union[str, tuple[str, ...], None]
    input_spec: Optional[InputSpec]
    scratch_root: Path
    receipt_path: Path
    protocol_version: str = "menagerie.crawler.worker-request.v1"
    work_id: str = "unbound"
    request_nonce: str = "unbound"
    request_sha256: str = "unbound"
    code_manifest_identity: str = "unbound"
    input_identity: str = "unbound"
    execution_read_manifest_identity: str = "unbound"
    standard_input_asset: Optional[Mapping[str, str]] = None
    input_contract: Optional[Mapping[str, Any]] = None
    input_manifest: Optional[Mapping[str, Any]] = None
    seed: int = 0
    input_seed: int = 0
    device: str = "cpu"
    framework: str = "pytorch"
    meaningful_modes: Optional[tuple[RunMode, ...]] = None
    mode: Optional[RunMode] = None
    source_identity: str = "unbound"
    execution_identity: str = "unbound"
    recipe_revision: str = "unbound"
    recipe_identity_payload: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        receipt_path: Optional[Path] = None,
        request_sha256: Optional[str] = None,
    ) -> "WorkerRequest":
        """Validate and normalize a JSON worker request.

        Parameters
        ----------
        value:
            JSON-compatible request mapping.
        receipt_path:
            Optional argv-owned receipt path override.
        request_sha256:
            Parent-bound digest recomputed from the exact request file bytes.

        Returns
        -------
        WorkerRequest
            Typed request.
        """

        recipe = value.get("recipe")
        if not isinstance(recipe, Mapping):
            raise ValueError("worker recipe must be an object")
        input_value = value.get("input_spec")
        input_contract_value = value.get("input_contract")
        input_manifest_value = value.get("input_manifest")
        protocol_value = value.get("protocol_version", "menagerie.crawler.worker-request.v1")
        v3_protocol = protocol_value == "menagerie.crawler.worker-request.v3"
        if input_value is not None and not isinstance(input_value, Mapping):
            raise ValueError("worker input_spec must be an object or null")
        if input_contract_value is not None and not isinstance(input_contract_value, Mapping):
            raise ValueError("worker input_contract must be an object or null")
        if input_manifest_value is not None and not isinstance(input_manifest_value, Mapping):
            raise ValueError("worker input_manifest must be an object or null")
        if v3_protocol:
            if input_manifest_value is not None:
                raise ValueError("v3 worker requests forbid the legacy input_manifest")
            if isinstance(input_contract_value, Mapping) and "code_path" in input_contract_value:
                raise ValueError("v3 worker requests forbid input_contract.code_path presence")
            if mode_value := value.get("mode"):
                RunMode(str(mode_value))
            else:
                raise ValueError("v3 worker requests require one explicit mode")
            required_strings = (
                "work_id",
                "request_nonce",
                "execution_identity",
                "recipe_revision",
                "code_manifest_identity",
                "input_identity",
                "execution_read_manifest_identity",
            )
            if any(
                not isinstance(value.get(field), str) or not value.get(field)
                for field in required_strings
            ):
                raise ValueError("v3 worker request association fields must be non-empty")
            if request_sha256 is None or os.environ.get(_REQUEST_SHA256_ENV) != request_sha256:
                raise ValueError("v3 worker request digest does not match the parent binding")
            if os.environ.get(_READ_MANIFEST_ID_ENV) != value.get(
                "execution_read_manifest_identity"
            ):
                raise ValueError(
                    "v3 worker read manifest identity does not match the parent binding"
                )
        if input_value is None and input_contract_value is None:
            raise ValueError("worker requires input_contract or legacy input_spec")
        modality_value = value.get("modality")
        if isinstance(modality_value, list):
            modality: Union[str, tuple[str, ...], None] = tuple(
                str(item) for item in modality_value
            )
        elif modality_value is None or isinstance(modality_value, str):
            modality = modality_value
        else:
            raise ValueError("worker modality must be a string, string list, or null")
        modes_value = value.get("meaningful_modes")
        modes: Optional[tuple[RunMode, ...]] = None
        if modes_value is not None:
            if not isinstance(modes_value, list) or not modes_value:
                raise ValueError("meaningful_modes must be a non-empty list when supplied")
            modes = tuple(RunMode(str(mode)) for mode in modes_value)
        mode_value = value.get("mode")
        mode = None if mode_value is None else RunMode(str(mode_value))
        if receipt_path is None:
            raw_receipt = value.get("receipt_path")
            if not isinstance(raw_receipt, str) or not raw_receipt:
                raise ValueError("worker receipt_path must be provided")
            receipt_value = Path(raw_receipt)
        else:
            receipt_value = receipt_path
        return cls(
            stable_id=str(value.get("stable_id", "")),
            protocol_version=str(protocol_value),
            work_id=str(value.get("work_id", "unbound")),
            request_nonce=str(value.get("request_nonce", "unbound")),
            request_sha256=request_sha256 or "unbound",
            code_manifest_identity=str(value.get("code_manifest_identity", "unbound")),
            input_identity=str(value.get("input_identity", "unbound")),
            execution_read_manifest_identity=str(
                value.get("execution_read_manifest_identity", "unbound")
            ),
            standard_input_asset=(
                dict(value["standard_input_asset"])
                if isinstance(value.get("standard_input_asset"), Mapping)
                else None
            ),
            recipe=dict(recipe),
            modality=modality,
            input_spec=InputSpec.from_value(input_value) if input_value is not None else None,
            input_contract=(
                dict(input_contract_value) if isinstance(input_contract_value, Mapping) else None
            ),
            input_manifest=(
                dict(input_manifest_value) if isinstance(input_manifest_value, Mapping) else None
            ),
            scratch_root=Path(str(value["scratch_root"])),
            receipt_path=receipt_value,
            seed=int(value.get("seed", 0)),
            input_seed=int(value.get("input_seed", 0)),
            device=str(value.get("device", "cpu")),
            framework=str(value.get("framework", "pytorch")),
            meaningful_modes=modes,
            mode=mode,
            source_identity=str(value.get("source_identity", "unbound")),
            execution_identity=str(value.get("execution_identity", "unbound")),
            recipe_revision=str(value.get("recipe_revision", "unbound")),
            recipe_identity_payload=(
                dict(value["recipe_identity_payload"])
                if isinstance(value.get("recipe_identity_payload"), Mapping)
                else None
            ),
        )


def _observe_request_bytes(
    request: WorkerRequest,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Verify request-bound helper and standard-asset bytes before execution.

    Parameters
    ----------
    request:
        Complete worker request with parent-bound byte manifests.

    Returns
    -------
    tuple[str | None, str | None, str | None]
        Observed code-manifest and selected-input-asset digests plus an optional
        fail-closed mismatch message.
    """

    manifest_value = request.recipe.get("code_manifest")
    expected_manifest = request.recipe.get("code_manifest_sha256")
    observed_manifest: Optional[str] = None
    errors: list[str] = []
    if manifest_value is not None or expected_manifest is not None:
        if not isinstance(manifest_value, list) or not manifest_value:
            raise RecipeError("typed-adapter worker request has no complete code manifest")
        if not isinstance(expected_manifest, str):
            raise RecipeError("typed-adapter worker request has no code-manifest digest")
        observed_members: list[dict[str, str]] = []
        for member in manifest_value:
            if not isinstance(member, Mapping):
                raise RecipeError("typed-adapter code-manifest member must be an object")
            path_value = member.get("path")
            identity_path = member.get("identity_path")
            expected_digest = member.get("sha256")
            if not all(
                isinstance(value, str) and value
                for value in (path_value, identity_path, expected_digest)
            ):
                raise RecipeError("typed-adapter code-manifest member is incomplete")
            member_path = Path(str(path_value))
            try:
                digest = hash_bytes(member_path.read_bytes())
            except OSError as exc:
                errors.append(f"typed-adapter code-manifest member is unreadable: {exc}")
                continue
            if digest != expected_digest:
                errors.append(f"typed-adapter code-manifest member changed: {identity_path}")
            observed_members.append({"path": str(identity_path), "sha256": digest})
        observed_manifest = stable_hash(observed_members)
        if observed_manifest != expected_manifest:
            errors.append("typed-adapter code-manifest aggregate digest changed")
        if (
            request.protocol_version == "menagerie.crawler.worker-request.v3"
            and expected_manifest != request.code_manifest_identity
        ):
            errors.append("typed-adapter code manifest differs from the request association")
    elif request.protocol_version == "menagerie.crawler.worker-request.v3":
        # Declarative execution has no authored code members. Its observed code
        # capability is the request-bound canonical empty-manifest identity.
        observed_manifest = request.code_manifest_identity

    selected_asset = (
        request.standard_input_asset
        if request.protocol_version == "menagerie.crawler.worker-request.v3"
        else (
            request.input_manifest.get("selected_asset")
            if isinstance(request.input_manifest, Mapping)
            else None
        )
    )
    observed_asset: Optional[str] = None
    if selected_asset is not None:
        if not isinstance(selected_asset, Mapping):
            raise RecipeError("worker selected-asset manifest must be an object")
        path_value = (
            os.environ.get(_PARENT_STANDARD_INPUT_ASSET_ENV)
            if request.protocol_version == "menagerie.crawler.worker-request.v3"
            else selected_asset.get("path")
        )
        expected_digest = selected_asset.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_digest, str):
            raise RecipeError("worker selected-asset manifest is incomplete")
        try:
            observed_asset = hash_bytes(Path(path_value).read_bytes())
        except OSError as exc:
            errors.append(f"worker selected input asset is unreadable: {exc}")
        if observed_asset is not None and observed_asset != expected_digest:
            errors.append("worker selected input asset digest changed")
    return observed_manifest, observed_asset, "; ".join(errors) or None


def _seed_frameworks(seed: int, framework: str) -> None:
    """Seed Python, NumPy, and the selected native framework, or refuse typed.

    The framework spelling is resolved through the closed vocabulary FIRST. This function
    used to compare against a local three-entry alias set and fall off the end silently: a
    row whose author wrote ``"torch 2.13"``, ``"pytorch (cpu)"`` or ``"keras"`` got NO
    ``manual_seed`` at all, and the run proceeded. Where the input contract has a tensor
    leaf, ``_convert_array`` catches the spelling loudly; where it has none -- a
    constructed-graph forward, which is schema-legal -- nothing caught it and the model ran
    at random init, producing a different ``output_value_sha256`` on every cold run and
    re-drive churn nobody could explain (AREA_BACKENDS_opus B1, blast radius (a)).

    Every framework whose seeding is not implemented refuses here rather than proceeding
    unseeded, because an unseeded run is not a degraded run, it is an unverifiable one.

    Parameters
    ----------
    seed:
        Deterministic execution seed.
    framework:
        Selected native execution framework, in any accepted spelling.

    Raises
    ------
    FrameworkVocabularyError
        For a spelling outside the closed vocabulary.
    BackendProtocolUnimplementedError
        For a closed-vocabulary framework whose seed protocol is not implemented here.
    """

    random.seed(seed)
    np.random.seed(seed)
    if framework.strip().lower() in NUMPY_TARGETS:
        # Positively identified, not residual: a NumPy-target forward is fully seeded by the
        # unconditional `np.random.seed` above, so there is nothing further to do. NumPy is
        # a materialization target rather than a member of the closed model-execution
        # vocabulary (see standard_inputs.NUMPY_TARGETS).
        return
    resolved = normalize_framework(framework)
    if resolved is Framework.TORCH:
        import torch

        torch.manual_seed(seed)
        return
    if resolved is Framework.TENSORFLOW:
        import tensorflow as tf

        tf.random.set_seed(seed)
        return
    if resolved is Framework.PADDLE:
        import paddle

        paddle.seed(seed)
        return
    if resolved is Framework.MLX:
        import mlx.core as mlx_core

        mlx_core.random.seed(seed)
        return
    raise BackendProtocolUnimplementedError(
        f"{resolved.value} declares seed protocol "
        f"{framework_capability(resolved).seed_protocol!r}, which this worker does not "
        "implement; an unseeded run is unverifiable, not degraded"
    )


def _sidecar_adapter_sha256(request: WorkerRequest, adapter_path: Path) -> Optional[str]:
    """Recover an accepted adapter digest from the parent-owned artifact sidecar.

    Parameters
    ----------
    request:
        Worker request carrying the campaign/source identities.
    adapter_path:
        Typed adapter path selected by the driver.

    Returns
    -------
    str | None
        Exact accepted code digest, or ``None`` when no fully bound sidecar exists.
    """

    resolved_adapter = adapter_path.resolve()
    for directory in (resolved_adapter.parent, *resolved_adapter.parents[:4]):
        sidecar = directory / "driver-author-artifact.json"
        if not sidecar.is_file():
            continue
        try:
            artifact = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(artifact, Mapping):
            return None
        proposal = artifact.get("proposal")
        model_dir_value = artifact.get("model_dir")
        if not isinstance(proposal, Mapping) or not isinstance(model_dir_value, str):
            return None
        claimed_proposal_hash = proposal.get("proposal_sha256")
        proposal_payload = {
            key: value for key, value in proposal.items() if key != "proposal_sha256"
        }
        if claimed_proposal_hash != stable_hash(proposal_payload):
            return None
        if (
            proposal.get("stable_id") != request.stable_id
            or proposal.get("source_identity") != request.source_identity
            or proposal.get("recipe_revision") != request.recipe_revision
        ):
            return None
        facts = proposal.get("proposed_facts")
        implementation = facts.get("implementation") if isinstance(facts, Mapping) else None
        if not isinstance(implementation, Mapping):
            return None
        code_value = implementation.get("code_path")
        code_digest = implementation.get("code_sha256")
        verified_hashes = proposal.get("verified_hashes")
        if not isinstance(code_value, str) or not isinstance(code_digest, str):
            return None
        model_dir = Path(model_dir_value).resolve()
        code_path = Path(code_value)
        resolved_code = (code_path if code_path.is_absolute() else model_dir / code_path).resolve()
        if resolved_code != resolved_adapter:
            return None
        if not isinstance(verified_hashes, Mapping) or verified_hashes.get("code") != code_digest:
            return None
        return code_digest
    return None


def _parameter_counts(model: object) -> tuple[Optional[int], Optional[int]]:
    """Count native model parameters when an iterator is available.

    Parameters
    ----------
    model:
        Native model or transparent adapter.

    Returns
    -------
    tuple[int | None, int | None]
        Total and trainable counts.
    """

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return (None, None)
    try:
        values = tuple(parameters())
        total = sum(int(parameter.numel()) for parameter in values)
        trainable = sum(
            int(parameter.numel())
            for parameter in values
            if bool(getattr(parameter, "requires_grad", False))
        )
    except (AttributeError, TypeError, RuntimeError):
        return (None, None)
    return (total, trainable)


def _input_signature(args: tuple[object, ...], kwargs: Mapping[str, object]) -> dict[str, Any]:
    """Describe a complete typed dummy call without retaining payload values.

    Parameters
    ----------
    args, kwargs:
        Exact positional and keyword arguments returned by the typed contract.

    Returns
    -------
    dict[str, Any]
        Shape, dtype, device, and complete pytree location.
    """

    signature = output_signature({"args": args, "kwargs": dict(kwargs)}, path="input")
    values = _call_leaf_values(args, kwargs)
    for leaf in signature["leaves"]:
        path = str(leaf["path"]).removeprefix("input.")
        value = values.get(path)
        if leaf["kind"] == "python":
            # stable_leaf_hash, not stable_hash: a graph-object leaf (DGLGraph,
            # PyG Data) is not canonical-JSON-serializable, and raising here
            # destroyed the whole receipt after the forward itself succeeded
            # (m538, step-100 B2). The fallback digest is type-tagged and
            # deterministic; see the hasher's documented determinism bounds.
            leaf["value_sha256"] = stable_leaf_hash(value)
    return signature


_PATH_TOKEN = re.compile(r"(?:^|\.)([A-Za-z_][A-Za-z0-9_]*)|\[([0-9]+)\]")


def _path_tokens(path: str) -> tuple[Union[str, int], ...]:
    """Parse one closed args/kwargs path into mapping/list tokens.

    Parameters
    ----------
    path:
        Contract path such as ``args[0]`` or ``kwargs.mask``.

    Returns
    -------
    tuple[str | int, ...]
        Parsed tokens.
    """

    matches = list(_PATH_TOKEN.finditer(path))
    if not matches or "".join(match.group(0) for match in matches) != path:
        raise TypeError(f"invalid dummy-call contract path: {path!r}")
    tokens: list[Union[str, int]] = []
    for match in matches:
        name, index = match.groups()
        tokens.append(name if name is not None else int(index))
    if tokens[0] not in {"args", "kwargs"}:
        raise TypeError(f"dummy-call path must start with args or kwargs: {path!r}")
    return tuple(tokens)


def _assign_path(root: dict[str, Any], path: str, value: object) -> None:
    """Assign one materialized leaf into a closed dummy-call tree."""

    tokens = _path_tokens(path)
    current: Any = root
    for index, token in enumerate(tokens):
        final = index == len(tokens) - 1
        next_token = None if final else tokens[index + 1]
        if isinstance(token, str):
            if not isinstance(current, dict):
                raise TypeError(f"dummy-call path collides with a non-object: {path!r}")
            if final:
                if token in current:
                    raise TypeError(f"duplicate dummy-call path: {path!r}")
                current[token] = value
            else:
                expected: object = [] if isinstance(next_token, int) else {}
                current = current.setdefault(token, expected)
        else:
            if not isinstance(current, list):
                raise TypeError(f"dummy-call path collides with a non-list: {path!r}")
            while len(current) <= token:
                current.append(None)
            if final:
                if current[token] is not None:
                    raise TypeError(f"duplicate dummy-call path: {path!r}")
                current[token] = value
            else:
                expected = [] if isinstance(next_token, int) else {}
                if current[token] is None:
                    current[token] = expected
                current = current[token]


def _call_leaf_values(args: tuple[object, ...], kwargs: Mapping[str, object]) -> dict[str, object]:
    """Flatten an exact call to the same paths used by input signatures."""

    values: dict[str, object] = {}

    def visit(value: object, path: str) -> None:
        """Record scalar/tensor leaves recursively."""

        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                visit(value[key], f"{path}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        else:
            values[path] = value

    visit(args, "args")
    visit(dict(kwargs), "kwargs")
    return values


def _declared_distribution(request: WorkerRequest) -> str:
    """Return the pinned distribution bounding this request's constructed inputs.

    A constructed input leaf imports and calls a declared symbol, so it needs the
    same namespace bound as the recipe's own construct nodes. That bound is the
    recipe's pinned distribution, which only the request carries.

    Parameters
    ----------
    request:
        Complete worker request.

    Returns
    -------
    str
        Declared pinned distribution.

    Raises
    ------
    TypeError
        If the declarative recipe declares no distribution to bound against.
    """

    payload = request.recipe.get("recipe")
    if not isinstance(payload, Mapping):
        # ``load_recipe`` accepts the flat form, where the recipe fields sit
        # directly on the request mapping; resolve the distribution the same way.
        payload = request.recipe
    distribution = payload.get("distribution")
    if not isinstance(distribution, str) or not distribution:
        raise TypeError("declarative recipe declares no distribution for constructed inputs")
    return distribution


def _materialize_declarative_call(
    request: WorkerRequest,
) -> tuple[tuple[object, ...], dict[str, object], str, Optional[str], str]:
    """Materialize every typed R1 positional, keyword, and non-tensor leaf."""

    contract = request.input_contract
    if not isinstance(contract, Mapping):
        if request.input_spec is None:
            raise TypeError("declarative recipe has no dummy-call contract")
        materialized = materialize_standard_input(
            request.modality,
            request.input_spec,
            framework=request.framework,
            device=request.device,
            seed=request.input_seed,
        )
        return (
            materialized.args,
            materialized.kwargs,
            materialized.input_kind,
            materialized.input_asset,
            materialized.input_note,
        )
    root: dict[str, Any] = {"args": [], "kwargs": {}}
    input_kinds: list[str] = []
    input_assets: list[str] = []
    notes: list[str] = []
    tensor_index = 0
    for collection in ("args", "kwargs"):
        leaves = contract.get(collection)
        if not isinstance(leaves, list):
            raise TypeError(f"input_contract.{collection} must be a list")
        for leaf in leaves:
            if not isinstance(leaf, Mapping):
                raise TypeError(f"input_contract.{collection} contains a non-object leaf")
            if leaf.get("distribution") == "constructor":
                if leaf.get("kind") != "constructed":
                    raise TypeError(
                        f"input_contract.{collection} constructor leaf must declare "
                        "kind 'constructed'"
                    )
                constructor_spec = leaf.get("constructor")
                if not isinstance(constructor_spec, Mapping):
                    raise TypeError(
                        f"input_contract.{collection} constructed leaf lacks a constructor spec"
                    )
                constructed = resolve_input_constructor(
                    constructor_spec, distribution=_declared_distribution(request)
                )
                _assign_path(root, str(leaf.get("path")), constructed)
                input_kinds.append("standard-constructed-input")
                notes.append(
                    "constructed input via "
                    f"{constructor_spec.get('module')}.{constructor_spec.get('symbol')}"
                )
                continue
            if leaf.get("kind") == "constructed":
                # A 'constructed' leaf that reaches this point carries a
                # non-'constructor' distribution (the constructor arm above
                # consumed the coherent spelling). The old fall-through message
                # claimed 'constructed' was inside the vocabulary while refusing
                # it -- a self-contradiction that cost five rung-11 models
                # (m9304/m9407/m9653/m9666/m9678) whose literal constructor
                # kwargs were spelled kind='constructed', distribution=
                # 'categorical'. Name the actual contract and the correct
                # spelling for the commonest case.
                raise TypeError(
                    f"input_contract.{collection} leaf {str(leaf.get('path'))!r} declares "
                    f"kind 'constructed' with distribution {leaf.get('distribution')!r}; "
                    "'constructed' is executable only with distribution 'constructor' "
                    "and a constructor spec. A literal constructor kwarg is a machine "
                    "value and belongs in input_contract.non_tensor_values"
                )
            if leaf.get("kind") != "tensor":
                # Naming the offending value and the closed vocabulary, because the
                # old "contains a non-tensor leaf" text MISDESCRIBED the commonest
                # case: an ordinary image tensor whose only defect was a descriptive
                # `kind` label. The schema now closes this vocabulary, so reaching
                # here is a real contract breach rather than a wording accident --
                # and it must still say which leaf and which value.
                raise TypeError(
                    f"input_contract.{collection} leaf {str(leaf.get('path'))!r} declares "
                    f"kind {leaf.get('kind')!r}; the executable vocabulary is "
                    "'tensor' or 'constructed' with distribution 'constructor'"
                )
            materialized = materialize_standard_input(
                request.modality,
                {
                    "shape": leaf.get("shape"),
                    "dtype": leaf.get("dtype"),
                    # The declared distribution and its authored constraints
                    # travel with the leaf so a bounded ``integer-range``
                    # (graph edge indices, entity/user ids) samples inside its
                    # legal range instead of the [0, 16) fallback that crashed
                    # every narrow-index model at its first forward.
                    "distribution": leaf.get("distribution"),
                    "constraints": leaf.get("constraints"),
                },
                framework=request.framework,
                device=request.device,
                seed=request.input_seed + tensor_index,
            )
            _assign_path(root, str(leaf.get("path")), materialized.value)
            input_kinds.append(materialized.input_kind)
            if materialized.input_asset is not None:
                input_assets.append(materialized.input_asset)
            notes.append(materialized.input_note)
            tensor_index += 1
    non_tensor = contract.get("non_tensor_values")
    if not isinstance(non_tensor, list):
        raise TypeError("input_contract.non_tensor_values must be a list")
    for leaf in non_tensor:
        if not isinstance(leaf, Mapping):
            raise TypeError("input_contract.non_tensor_values contains a non-object")
        _assign_path(root, str(leaf.get("path")), leaf.get("value"))
    args_value = root["args"]
    kwargs_value = root["kwargs"]
    if not isinstance(args_value, list) or any(value is None for value in args_value):
        raise TypeError("input_contract args paths must be contiguous")
    if not isinstance(kwargs_value, dict):
        raise TypeError("input_contract kwargs paths are invalid")
    kind = input_kinds[0] if len(set(input_kinds)) == 1 else "random-fallback"
    unique_assets = set(input_assets)
    asset = (
        next(iter(unique_assets))
        if len(unique_assets) == 1 and len(input_assets) == tensor_index
        else None
    )
    return tuple(args_value), kwargs_value, kind, asset, "; ".join(notes)


def _materialize_dummy_call(
    loaded: LoadedRecipe, request: WorkerRequest
) -> tuple[tuple[object, ...], dict[str, object], str, Optional[str], str]:
    """Execute the typed input contract or build a declarative standard call.

    Parameters
    ----------
    loaded:
        Validated executable recipe.
    request:
        Complete worker request.

    Returns
    -------
    tuple
        Args, kwargs, input kind, optional asset, and provenance note.

    Raises
    ------
    TypeError
        If ``make_dummy_call`` violates its typed return contract.
    """

    if loaded.make_dummy_call is not None:
        value = loaded.make_dummy_call(request.input_seed, request.device)
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("make_dummy_call must return exactly (args, kwargs)")
        args, kwargs = value
        if not isinstance(args, tuple):
            raise TypeError("make_dummy_call args must be a tuple")
        if not isinstance(kwargs, dict) or not all(isinstance(key, str) for key in kwargs):
            raise TypeError("make_dummy_call kwargs must be a string-keyed dict")
        return (
            args,
            kwargs,
            "standard-typed-dummy-call",
            None,
            "typed make_dummy_call(seed, device)",
        )

    return _materialize_declarative_call(request)


def _refuse_unbound_forward_parameters(
    model: object, args: Sequence[object], kwargs: Mapping[str, object]
) -> None:
    """Refuse a contract that omits a required ``forward()`` parameter.

    m2336 (PyKEEN TransE): ``ERModel.forward(h, r, t, *, mode)`` declares a
    required keyword-only ``mode``; the authored contract bound three
    positional leaves and no kwargs, so the forward call itself died with a
    ``TypeError`` INSIDE the mode receipt -- burning the receipt on a defect
    that is a static fact of the contract. Every required parameter of the
    resolved forward callable, including keyword-only ones, must be bound by
    the materialized contract BEFORE any mode receipt starts. Forwards whose
    signature cannot be retrieved fail open: the gate refuses only what it
    can prove, and variadic parameters bind nothing by themselves.
    """

    forward = getattr(model, "forward", None)
    if not callable(forward):
        return  # _mode_receipt owns the typed refusal for a missing forward
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return
    positional_budget = len(args)
    missing: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            if positional_budget > 0:
                positional_budget -= 1
            else:
                missing.append(parameter.name)
        elif parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if positional_budget > 0:
                positional_budget -= 1
            elif parameter.name not in kwargs:
                missing.append(parameter.name)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            if parameter.name not in kwargs:
                missing.append(parameter.name)
    if missing:
        names = ", ".join(repr(name) for name in missing)
        declared = ", ".join(signature.parameters)
        raise RecipeError(
            f"input contract does not bind required forward() parameter(s) {names}: "
            f"forward declares ({declared}), and every required parameter, "
            "including keyword-only ones, must be bound by a materialized contract "
            "leaf (a positional args[...] leaf, or a kwargs.<name> leaf / "
            "non_tensor value) before any mode forward runs"
        )


def _build_declared_model(loaded: LoadedRecipe, request: WorkerRequest) -> object:
    """Build one model and apply the declared non-``forward`` entrypoint delegation.

    Parameters
    ----------
    loaded:
        Validated executable recipe.
    request:
        Complete worker request supplying the native framework.

    Returns
    -------
    object
        The constructed model, wrapped in the crawler-owned transparent adapter
        when the declarative recipe declares a public non-``forward`` entrypoint.
    """

    model = loaded.build_model()
    if (
        loaded.kind == "declarative-library"
        and loaded.entrypoint is not None
        and loaded.entrypoint != "forward"
    ):
        model = NativeForwardAdapter(
            model,
            original_framework=request.framework,
            run_framework=request.framework,
            call_method=loaded.entrypoint,
        )
    return model


def _set_mode(model: object, mode: RunMode) -> None:
    """Explicitly select native train or eval mode.

    Parameters
    ----------
    model:
        Native model or transparent adapter.
    mode:
        Requested runtime mode.
    """

    method = getattr(model, mode.value, None)
    if not callable(method):
        if mode is RunMode.TRAIN:
            train_method = getattr(model, "train", None)
            if callable(train_method):
                train_method(True)
                return
        raise TypeError(f"model does not expose explicit {mode.value}()")
    method()


def _inference_context(framework: str) -> Any:
    """Return the no-grad context the framework's capability row declares.

    The ``nullcontext()`` fallback is CORRECT for TensorFlow, JAX and MLX -- none of them
    builds a graph on a forward that a context has to suppress -- but it used to be reached
    residually, by any string that was not torch. That made an unrecognised spelling
    indistinguishable from a framework that genuinely needs no context (catalog C-25:
    identification must be positive, never residual). The capability row now says which it
    is, and an unimplemented protocol refuses.

    Parameters
    ----------
    framework:
        Native execution framework, in any accepted spelling.

    Returns
    -------
    Any
        Context manager.

    Raises
    ------
    BackendProtocolUnimplementedError
        For a framework whose declared inference context is not implemented here.
    """

    if framework.strip().lower() in NUMPY_TARGETS:
        # NumPy builds no autograd graph, so "no context" here is a positive fact about
        # NumPy rather than the residual else-branch it used to be reached through.
        return nullcontext()
    resolved = normalize_framework(framework)
    declared = framework_capability(resolved).inference_context
    if declared == "none-required":
        return nullcontext()
    if resolved is Framework.TORCH:
        import torch

        return torch.no_grad()
    if resolved is Framework.PADDLE:
        import paddle

        return paddle.no_grad()
    raise BackendProtocolUnimplementedError(
        f"{resolved.value} declares inference context {declared!r}, which this worker does "
        "not implement"
    )


def _native_metadata(model: object, framework: str) -> tuple[str, str]:
    """Return native framework and explicitly delegated method.

    Parameters
    ----------
    model:
        Native model or transparent adapter.
    framework:
        Requested execution framework.

    Returns
    -------
    tuple[str, str]
        Framework and delegated method.
    """

    if isinstance(model, NativeForwardAdapter):
        return (model.metadata.run_framework, model.metadata.native_call_method)
    return (framework, "forward")


class ExecutionContextViolationError(RuntimeError):
    """The worker's execution context contradicts its pinned determinism locks.

    DX-70/UU-24: ``output_value_sha256`` is thread-count-dependent, so a worker
    whose live thread context disagrees with the requested pin must refuse
    BEFORE any digest can mint -- an unpinned digest poisons every later
    byte-exact comparison. A HOST/machinery fact: the parent classifies this
    type as machinery-owned, never a model verdict.
    """


def _assert_thread_context(framework: str) -> None:
    """Refuse execution when the live torch thread count escapes the env pin.

    Parameters
    ----------
    framework:
        Native execution framework, in any accepted spelling.

    Raises
    ------
    ExecutionContextViolationError
        When ``OMP_NUM_THREADS`` is pinned but torch's live intra-op thread
        count disagrees (or the pin is unparseable) -- the digest would be
        minted under an unattested numerics context (DX-70/UU-24).
    """

    if framework.strip().lower() in NUMPY_TARGETS:
        return
    if normalize_framework(framework) is not Framework.TORCH:
        return
    pinned = os.environ.get("OMP_NUM_THREADS")
    if pinned is None:
        return
    import torch

    try:
        expected = int(pinned)
    except ValueError as exc:
        raise ExecutionContextViolationError(
            f"OMP_NUM_THREADS pin {pinned!r} is unparseable; refusing to mint "
            "thread-count-dependent digests under an unattested context"
        ) from exc
    observed = torch.get_num_threads()
    if observed != expected:
        raise ExecutionContextViolationError(
            f"torch intra-op thread count is {observed} but OMP_NUM_THREADS pins "
            f"{expected}; output digests are thread-count-dependent and must not "
            "mint under a drifted context"
        )


def _assert_model_device(model: object, request: WorkerRequest) -> None:
    """Refuse a torch model whose parameters sit off the requested device.

    The m11584 class (DX-70/UU-24 MPS-leakage arm): a recipe that resolves its
    own device at import time (``cuda``/``mps`` when available) contradicts the
    request's execution identity, and every byte-exact digest it produces is
    incomparable with the attested context. A self-selecting recipe is a MODEL
    defect and refuses in the constructor phase.

    Parameters
    ----------
    model:
        Constructed native model.
    request:
        Typed worker request carrying the pinned device.

    Raises
    ------
    ValueError
        When any parameter or buffer sits on a device type other than the
        requested one.
    """

    if request.framework.strip().lower() in NUMPY_TARGETS:
        return
    if normalize_framework(request.framework) is not Framework.TORCH:
        return
    import torch

    if not isinstance(model, torch.nn.Module):
        return
    requested_type = torch.device(request.device).type
    offenders = sorted(
        {
            str(tensor.device)
            for tensor in (*model.parameters(), *model.buffers())
            if tensor.device.type != requested_type
        }
    )
    if offenders:
        raise ValueError(
            f"model self-selected device(s) {offenders} but the request pins "
            f"{request.device!r}; a self-selecting recipe (m11584 class) breaks "
            "execution-identity comparability and is refused before any forward"
        )


def _assert_award_comparable_dtypes(signature: Mapping[str, Any], framework: str) -> None:
    """Refuse a non-torch output whose dtypes escape the closed normalizer map.

    DX-14's owned detail (award row 40, adopted per F-O25): before any
    non-torch award, the dtype normalizer's CLOSED map must cover every
    observed tensor dtype. ``canonical_dtype_text`` is deliberately total and
    preserves an unmappable spelling verbatim -- honest for capture metadata,
    but an award identity built on a verbatim survivor is not comparable
    across frameworks, so it must refuse HERE, before the receipt claims a
    completed forward.

    Parameters
    ----------
    signature:
        The mode's ``output_signature`` tree.
    framework:
        Native execution framework, in any accepted spelling.

    Raises
    ------
    BackendProtocolUnimplementedError
        When a non-torch framework emitted a tensor dtype outside the closed
        vocabulary -- normalizer parity is missing, a machinery gap.
    """

    normalized = framework.strip().lower()
    if normalized in NUMPY_TARGETS:
        return
    if normalize_framework(normalized) is Framework.TORCH:
        return
    offenders: list[str] = []
    leaves = signature.get("leaves")
    for leaf in leaves if isinstance(leaves, list) else ():
        if not isinstance(leaf, Mapping) or leaf.get("kind") != "tensor":
            continue
        dtype = leaf.get("dtype")
        if dtype is None:
            continue
        try:
            canonical_dtype(dtype)
        except DtypeVocabularyError:
            offenders.append(f"{leaf.get('path')}: {dtype!r}")
    if offenders:
        raise BackendProtocolUnimplementedError(
            f"{normalized} emitted output dtypes outside the closed normalizer "
            f"vocabulary ({', '.join(offenders)}); a non-torch award must not bind "
            "unnormalized dtypes"
        )


def _mode_receipt(
    model: object,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    input_kind: str,
    input_asset: Optional[str],
    input_note: str,
    mode: RunMode,
    framework: str,
    constructor_seconds: float,
) -> tuple[dict[str, Any], Optional[object]]:
    """Run one explicit forward and produce an honest per-mode receipt.

    Parameters
    ----------
    model:
        Constructed native model.
    args, kwargs:
        Shared complete dummy call.
    input_kind, input_asset, input_note:
        Mechanically observed dummy-call provenance.
    mode:
        Explicit runtime mode.
    framework:
        Native execution framework.
    constructor_seconds:
        Already observed construction duration.

    Returns
    -------
    tuple[dict[str, Any], object | None]
        Per-mode receipt and captured output retained only in memory.
    """

    total, trainable = _parameter_counts(model)
    native_framework, delegated_method = _native_metadata(model, framework)
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "mode": mode.value,
        "constructor_started": True,
        "constructor_completed": True,
        "constructor_seconds": constructor_seconds,
        "input_completed": True,
        "input_signature": _input_signature(args, kwargs),
        "input_kind": input_kind,
        "input_asset": input_asset,
        "input_note": input_note,
        "forward_started": False,
        "forward_completed": False,
        "forward_seconds": None,
        "output_signature": None,
        "output_value_sha256": None,
        "parameter_count_total": total,
        "parameter_count_trainable": trainable,
        "native_framework": native_framework,
        "delegated_method": delegated_method,
        "error": None,
    }
    try:
        _set_mode(model, mode)
        forward = getattr(model, "forward", None)
        if not callable(forward):
            raise TypeError("constructed model does not expose a callable forward attribute")
        receipt["forward_started"] = True
        with _inference_context(framework):
            output = forward(*args, **dict(kwargs))
        receipt["forward_seconds"] = time.monotonic() - started
        signature = output_signature(output)
        # Non-torch dtype-normalizer conformance (DX-14 detail, award row 40):
        # asserted BEFORE forward_completed so a refusal is a mode failure the
        # parent classifies as a machinery capability gap, never a receipt
        # that self-contradicts with a success flag.
        _assert_award_comparable_dtypes(signature, framework)
        receipt["forward_completed"] = True
        receipt["output_signature"] = signature
        receipt["output_value_sha256"] = output_value_sha256(output)
        return receipt, output
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001 -- receipt must retain native failures;
        # SystemExit (argparse/absl/click inside model code) previously unwound
        # past the receipt entirely and published as an evidence-free
        # ``runner/missing-receipt`` (BL-12(f), opus W-12).
        receipt["forward_seconds"] = time.monotonic() - started
        receipt["error"] = {
            "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "message": str(exc) or repr(exc),
            "traceback": traceback.format_exc(),
        }
        return receipt, None


def _atomic_receipt(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically fsync and rename one complete worker receipt.

    Parameters
    ----------
    path:
        Final receipt path.
    payload:
        Receipt without its self hash.

    Returns
    -------
    dict[str, Any]
        Persisted payload including ``receipt_sha256``.
    """

    record = dict(payload)
    record["receipt_sha256"] = stable_hash(payload)
    data = canonical_json_bytes(record) + b"\n"
    atomic_replace_bytes(path, data)
    return record


def _clean_policy_observation(observation: PolicyObservation) -> bool:
    """Return whether every worker-side policy tripwire remained clean.

    Parameters
    ----------
    observation:
        Closed in-child policy facts.

    Returns
    -------
    bool
        True only when no disqualifying policy fact was observed.
    """

    return not any(
        (
            observation.network_attempted,
            observation.checkpoint_or_weight_read_attempted,
            observation.write_outside_scratch_attempted,
            observation.credentials_present,
            observation.torchlens_import_attempted,
            observation.cache_read_attempted,
        )
    )


def _raw_award_receipt(
    request: WorkerRequest,
    diagnostic: Mapping[str, Any],
    policy: PolicyObservation,
) -> Optional[dict[str, Any]]:
    """Derive the closed success-only v3 raw receipt from worker observations.

    Parameters
    ----------
    request:
        Exact parent-bound v3 request.
    diagnostic:
        Complete internal execution diagnostic.
    policy:
        In-child policy observation.

    Returns
    -------
    dict[str, Any] | None
        Frozen raw award receipt, or ``None`` for every failed/partial/dirty run.
    """

    if (
        request.mode is None
        or diagnostic.get("error") is not None
        or not _clean_policy_observation(policy)
    ):
        return None
    per_mode = diagnostic.get("per_mode")
    mode_observation = per_mode.get(request.mode.value) if isinstance(per_mode, Mapping) else None
    if not isinstance(mode_observation, Mapping) or not mode_observation.get("forward_completed"):
        return None
    observation: dict[str, Any] = {
        "present": True,
        "receipt_sha256": None,
        "observed_recipe_revision": diagnostic.get("observed_recipe_revision"),
        "observed_adapter_sha256": diagnostic.get("observed_adapter_sha256"),
        "observed_code_manifest_sha256": diagnostic.get("observed_code_manifest_sha256"),
        "observed_input_asset_sha256": diagnostic.get("observed_input_asset_sha256"),
        "constructor_seconds": mode_observation.get("constructor_seconds"),
        "forward_seconds": mode_observation.get("forward_seconds"),
        "constructor_started": bool(mode_observation.get("constructor_started")),
        "constructor_completed": bool(mode_observation.get("constructor_completed")),
        "input_completed": bool(mode_observation.get("input_completed")),
        "forward_started": bool(mode_observation.get("forward_started")),
        "forward_completed": bool(mode_observation.get("forward_completed")),
        "mode": mode_observation.get("mode"),
        "input_signature": mode_observation.get("input_signature"),
        "output_signature": mode_observation.get("output_signature"),
        "output_value_sha256": mode_observation.get("output_value_sha256"),
        "input_kind": mode_observation.get("input_kind"),
        "input_asset": mode_observation.get("input_asset"),
        "input_note": str(mode_observation.get("input_note", "")),
        "parameter_count_total": mode_observation.get("parameter_count_total"),
        "parameter_count_trainable": mode_observation.get("parameter_count_trainable"),
        "native_framework": mode_observation.get("native_framework"),
        "delegated_method": mode_observation.get("delegated_method"),
    }
    return {
        "receipt_version": "menagerie.crawler.raw-award-receipt.v3",
        "request_nonce": request.request_nonce,
        "request_sha256": request.request_sha256,
        "stable_id": request.stable_id,
        "work_id": request.work_id,
        "execution_identity": request.execution_identity,
        "recipe_revision": request.recipe_revision,
        "code_manifest_identity": request.code_manifest_identity,
        "input_identity": request.input_identity,
        "requested_mode": request.mode.value,
        "observation": observation,
    }


def _atomic_worker_result(
    path: Path,
    raw_receipt: Optional[Mapping[str, Any]],
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist one v3 worker-result envelope with exact embedded raw bytes.

    Parameters
    ----------
    path:
        Atomic result path.
    raw_receipt:
        Closed success-only receipt, or ``None`` for a non-awarding result.
    diagnostic:
        Hash-bound diagnostic retained for failure assembly only.

    Returns
    -------
    dict[str, Any]
        Persisted result envelope.
    """

    payload: dict[str, Any] = {
        "result_version": "menagerie.crawler.worker-result.v3",
        "raw_award_receipt": None if raw_receipt is None else dict(raw_receipt),
        "raw_award_receipt_sha256": (None if raw_receipt is None else stable_hash(raw_receipt)),
        "diagnostic": dict(diagnostic),
    }
    payload["result_sha256"] = stable_hash(payload)
    data = canonical_json_bytes(payload) + b"\n"
    atomic_replace_bytes(path, data)
    return payload


def _start_parent_watchdog() -> None:
    """Validate inherited lifecycle FDs and kill this worker when its parent dies.

    The kernel flock remains the concurrency authority. This watchdog and Linux
    ``PR_SET_PDEATHSIG`` are defense in depth for parent ``SIGKILL``.
    """

    lock_value = os.environ.pop(_WORKER_LOCK_FD_ENV, None)
    lifecycle_value = os.environ.pop(_LIFECYCLE_FD_ENV, None)
    if lock_value is None or lifecycle_value is None:
        raise RuntimeError("v3 worker is missing inherited lifecycle descriptors")
    try:
        lock_fd = int(lock_value)
        lifecycle_fd = int(lifecycle_value)
        os.fstat(lock_fd)
        os.fstat(lifecycle_fd)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The ROOT worker alone carries the lease. Marking the inherited lock
        # descriptors close-on-exec means spawn-method grandchildren (loky /
        # multiprocessing executors that fork+exec) can never hold the kernel
        # flocks after this root exits -- the orphan-holds-the-lock door that
        # silently hung every later leg on the host (BL-12(c), fable W-F2).
        cloexec_targets = [lock_fd, lifecycle_fd]
        global_lock_value = os.environ.pop(_GLOBAL_LOCK_FD_ENV, None)
        if global_lock_value is not None:
            cloexec_targets.append(int(global_lock_value))
        for descriptor in cloexec_targets:
            flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
            fcntl.fcntl(descriptor, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
    except (OSError, ValueError) as exc:
        raise RuntimeError("v3 worker inherited invalid lifecycle descriptors") from exc
    parent_pid = os.getppid()

    def watch_parent() -> None:
        """Terminate immediately on lifecycle EOF or verified parent change."""

        while True:
            try:
                readable, _writable, _exceptional = select.select((lifecycle_fd,), (), (), 0.1)
            except OSError:
                os.kill(os.getpid(), signal.SIGKILL)
                return
            if os.getppid() != parent_pid:
                os.kill(os.getpid(), signal.SIGKILL)
                return
            if readable:
                try:
                    data = os.read(lifecycle_fd, 1)
                except OSError:
                    data = b""
                if not data:
                    os.kill(os.getpid(), signal.SIGKILL)
                    return

    thread = threading.Thread(target=watch_parent, name="worker-parent-watchdog", daemon=True)
    thread.start()


def _execute(request: WorkerRequest) -> tuple[dict[str, Any], PolicyObservation]:
    """Execute one model under active in-child policy tripwires.

    Parameters
    ----------
    request:
        Typed worker request.

    Returns
    -------
    tuple[dict[str, Any], PolicyObservation]
        Receipt payload and policy observation.
    """

    request.scratch_root.mkdir(parents=True, exist_ok=True)
    recipe_path = request.recipe.get("path")
    effective_recipe = dict(request.recipe)
    if isinstance(recipe_path, str) and recipe_path:
        adapter_digest = effective_recipe.get("adapter_sha256")
        if not isinstance(adapter_digest, str):
            sidecar_digest = _sidecar_adapter_sha256(request, Path(recipe_path))
            if sidecar_digest is not None:
                effective_recipe["adapter_sha256"] = sidecar_digest
    allowed_read_paths = (
        [] if request.protocol_version == "menagerie.crawler.worker-request.v3" else [ASSET_ROOT]
    )
    if isinstance(recipe_path, str) and recipe_path:
        allowed_read_paths.append(Path(recipe_path))
    manifest_value = effective_recipe.get("code_manifest")
    if isinstance(manifest_value, list):
        allowed_read_paths.extend(
            Path(str(member["path"]))
            for member in manifest_value
            if isinstance(member, Mapping) and isinstance(member.get("path"), str)
        )
    policy = ExecutionPolicy(
        request.scratch_root,
        request.receipt_path.parent,
        allowed_read_paths=allowed_read_paths,
    )
    base: dict[str, Any] = {
        "receipt_version": "menagerie.crawler.worker-receipt.v1",
        "stable_id": request.stable_id,
        "source_identity": request.source_identity,
        "recipe_revision": request.recipe_revision,
        "observed_recipe_revision": None,
        "observed_adapter_sha256": None,
        "observed_code_manifest_sha256": None,
        "observed_input_asset_sha256": None,
        "execution_identity": request.execution_identity,
        "seed": request.seed,
        "input_seed": request.input_seed,
        "mode": request.mode.value if request.mode is not None else None,
        "device": request.device,
        "framework": request.framework,
        "awards_runs": False,
        "constructor_started": False,
        "constructor_completed": False,
        "input_completed": False,
        "per_mode": {},
        "declared_meaningful_modes": [],
        "detected_meaningful_modes": [],
        "meaningful_modes": [],
        "train_eval_divergence": None,
        "divergence_evidence": None,
        "error": None,
    }
    with policy:
        try:
            observed_manifest, observed_asset, byte_error = _observe_request_bytes(request)
            base["observed_code_manifest_sha256"] = observed_manifest
            base["observed_input_asset_sha256"] = observed_asset
            if byte_error is not None:
                raise RecipeError(byte_error)
            _seed_frameworks(request.seed, request.framework)
            _assert_thread_context(request.framework)
            recipe_kind = effective_recipe.get("kind") or effective_recipe.get("recipe_type")
            if recipe_kind == "typed-adapter" and not isinstance(
                effective_recipe.get("adapter_sha256"), str
            ):
                raise RecipeError("typed-adapter worker request requires adapter_sha256")
            observed_accepted_revision: Optional[str] = None
            if request.recipe_identity_payload is not None:
                observed_accepted_revision = compute_recipe_revision(
                    request.recipe_identity_payload, request.source_identity
                )
                if observed_accepted_revision != request.recipe_revision:
                    raise RecipeError(
                        "accepted worker recipe identity mismatch: "
                        f"expected {request.recipe_revision}, observed {observed_accepted_revision}"
                    )
            expected_revision = (
                request.recipe_revision if request.recipe_revision != "unbound" else None
            )
            try:
                loaded = load_recipe(
                    effective_recipe,
                    source_identity=request.source_identity,
                    expected_recipe_revision=expected_revision,
                )
            except RecipeError as exc:
                if observed_accepted_revision is None or "recipe revision mismatch" not in str(exc):
                    raise
                identity_payload = request.recipe_identity_payload
                if not isinstance(identity_payload, Mapping):
                    raise
                identity_implementation = identity_payload.get("implementation")
                identity_recipe = (
                    identity_implementation.get("library_recipe")
                    if isinstance(identity_implementation, Mapping)
                    else None
                )
                executed_recipe = effective_recipe.get("recipe")
                if recipe_kind == "declarative-library" and (
                    not isinstance(identity_recipe, Mapping)
                    or not isinstance(executed_recipe, Mapping)
                    or stable_hash(identity_recipe) != stable_hash(executed_recipe)
                ):
                    raise RecipeError(
                        "declarative recipe bytes do not match the accepted identity payload"
                    ) from exc
                loaded = load_recipe(
                    effective_recipe,
                    source_identity=request.source_identity,
                    expected_recipe_revision=None,
                )
            base["observed_recipe_revision"] = observed_accepted_revision or loaded.recipe_revision
            base["observed_adapter_sha256"] = loaded.adapter_sha256
            base["constructor_started"] = True
            constructor_started = time.monotonic()
            model = _build_declared_model(loaded, request)
            constructor_seconds = time.monotonic() - constructor_started
            # Constructor-phase device assert (before the completion flag): a
            # self-selecting recipe is a constructor fact and classifies as
            # constructor/exception, owner model (DX-70/UU-24).
            _assert_model_device(model, request)
            base["constructor_completed"] = True
            args, kwargs, input_kind, input_asset, input_note = _materialize_dummy_call(
                loaded, request
            )
            selected_asset = (
                request.standard_input_asset
                if request.protocol_version == "menagerie.crawler.worker-request.v3"
                else (
                    request.input_manifest.get("selected_asset")
                    if isinstance(request.input_manifest, Mapping)
                    else None
                )
            )
            expected_asset_id = (
                selected_asset.get("asset_id") if isinstance(selected_asset, Mapping) else None
            )
            if input_asset != expected_asset_id:
                if input_asset is not None or expected_asset_id is None:
                    raise RecipeError(
                        "materialized input asset is outside the request-bound asset outcomes"
                    )
            base["observed_input_asset_sha256"] = (
                observed_asset if input_asset is not None else None
            )
            base["input_completed"] = True
            # FORWARD-KWARG-OMITTED (m2336): a required forward() parameter the
            # contract never bound is a static authoring fact -- refuse it typed
            # HERE, before the first mode receipt burns on the TypeError.
            _refuse_unbound_forward_parameters(model, args, kwargs)
            declared = request.meaningful_modes or ()
            detected = detect_meaningful_modes(model)
            if (
                request.mode is not None
                and any(mode not in declared for mode in detected)
                and set(detected) == {RunMode.TRAIN, RunMode.EVAL}
            ):
                # DX-9/RE-9: the structural heuristic disagrees with the gated
                # declaration. Resolve by EVIDENCE before the parent burns an
                # author-repair generation on it (0/14 repair successes in
                # 200a): run BOTH modes in-process on identically seeded fresh
                # builds and let the captured outputs adjudicate. A probe that
                # cannot produce both outputs (e.g. train refuses on this
                # contract) changes nothing -- the heuristic stands and the
                # parent's unsatisfiability guard owns the contradiction.
                probe_outputs: dict[str, object] = {}
                for probe_mode in (RunMode.TRAIN, RunMode.EVAL):
                    _seed_frameworks(request.seed, request.framework)
                    probe_started = time.monotonic()
                    probe_model = _build_declared_model(loaded, request)
                    _probe_receipt, probe_output = _mode_receipt(
                        probe_model,
                        args,
                        kwargs,
                        input_kind,
                        input_asset,
                        input_note,
                        probe_mode,
                        request.framework,
                        time.monotonic() - probe_started,
                    )
                    if probe_output is None:
                        break
                    probe_outputs[probe_mode.value] = probe_output
                if {"train", "eval"}.issubset(probe_outputs):
                    detected = detect_meaningful_modes(
                        model, captured_outputs=probe_outputs
                    )
                    probe_divergence = classify_train_eval_divergence(
                        probe_outputs["train"], probe_outputs["eval"]
                    )
                    base["train_eval_divergence"] = probe_divergence.classification
                    base["divergence_evidence"] = probe_divergence.evidence
            mode_set = set(declared) | set(detected)
            detected_modes = tuple(
                mode for mode in (RunMode.TRAIN, RunMode.EVAL) if mode in mode_set
            )
            if request.mode is not None and request.mode not in mode_set:
                raise ValueError(
                    f"requested mode {request.mode.value!r} is not meaningful for this model"
                )
            modes = (request.mode,) if request.mode is not None else detected_modes
            base["declared_meaningful_modes"] = [mode.value for mode in declared]
            base["detected_meaningful_modes"] = [mode.value for mode in detected]
            base["meaningful_modes"] = [mode.value for mode in detected_modes]
            outputs: dict[str, object] = {}
            for mode in modes:
                mode_model = model
                mode_constructor_seconds = constructor_seconds
                if request.mode is None and len(modes) > 1:
                    _seed_frameworks(request.seed, request.framework)
                    mode_constructor_started = time.monotonic()
                    mode_model = _build_declared_model(loaded, request)
                    mode_constructor_seconds = time.monotonic() - mode_constructor_started
                receipt, output = _mode_receipt(
                    mode_model,
                    args,
                    kwargs,
                    input_kind,
                    input_asset,
                    input_note,
                    mode,
                    request.framework,
                    mode_constructor_seconds,
                )
                base["per_mode"][mode.value] = receipt
                if output is not None:
                    outputs[mode.value] = output
            if request.mode is None and {
                RunMode.TRAIN.value,
                RunMode.EVAL.value,
            }.issubset(outputs):
                divergence = classify_train_eval_divergence(outputs["train"], outputs["eval"])
                base["train_eval_divergence"] = divergence.classification
                base["divergence_evidence"] = divergence.evidence
            elif request.mode is None and len(outputs) == 1:
                base["train_eval_divergence"] = "none"
                base["divergence_evidence"] = "only one meaningful mode"
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 -- constructor/input/policy failures
            # are receipts. BaseException (SystemExit from model constructors) must
            # leave the same evidence instead of skipping the receipt (BL-12(f)).
            reason_code = exc.reason_code if isinstance(exc, PolicyViolation) else None
            base["error"] = {
                "reason_code": reason_code,
                "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc) or repr(exc),
                "traceback": traceback.format_exc(),
                # Typed OS errno so the parent can classify host resource
                # weather (ENOSPC/EROFS/ENODEV/EIO) without parsing message
                # text (DX-13/RE-14). None for every non-OSError.
                "errno": getattr(exc, "errno", None),
            }
    base["policy_observation"] = policy.observation.to_dict()
    return base, policy.observation


def _aborted_before_receipt_diagnostic(
    request: WorkerRequest, exc: BaseException
) -> dict[str, Any]:
    """Build a complete diagnostic for a worker that aborted before observing anything.

    Parameters
    ----------
    request:
        Typed worker request whose execution aborted outside the receipt shield.
    exc:
        The escaping exception.

    Returns
    -------
    dict[str, Any]
        Closed worker-receipt.v1 diagnostic carrying only the abort evidence.
    """

    return {
        "receipt_version": "menagerie.crawler.worker-receipt.v1",
        "stable_id": request.stable_id,
        "source_identity": request.source_identity,
        "recipe_revision": request.recipe_revision,
        "observed_recipe_revision": None,
        "observed_adapter_sha256": None,
        "observed_code_manifest_sha256": None,
        "observed_input_asset_sha256": None,
        "execution_identity": request.execution_identity,
        "seed": request.seed,
        "input_seed": request.input_seed,
        "mode": request.mode.value if request.mode is not None else None,
        "device": request.device,
        "framework": request.framework,
        "awards_runs": False,
        "constructor_started": False,
        "constructor_completed": False,
        "input_completed": False,
        "per_mode": {},
        "declared_meaningful_modes": [],
        "detected_meaningful_modes": [],
        "meaningful_modes": [],
        "train_eval_divergence": None,
        "divergence_evidence": None,
        "policy_observation": PolicyObservation().to_dict(),
        "error": {
            "reason_code": "worker-aborted-before-receipt",
            "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "message": str(exc) or repr(exc),
            "traceback": traceback.format_exc(),
            # Typed OS errno: the parent's host-fault arm (DX-13/RE-14) keys
            # on this integer, never on message text.
            "errno": getattr(exc, "errno", None),
        },
    }


def run_worker(request: WorkerRequest) -> dict[str, Any]:
    """Execute one model and atomically persist its observation receipt.

    Parameters
    ----------
    request:
        Complete typed worker request.

    Returns
    -------
    dict[str, Any]
        Persisted receipt. This function never awards the canonical ``runs`` status.
    """

    try:
        payload, observation = _execute(request)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        # A receipt is this process's ONLY output: capture machinery aborting
        # before one exists must be a recorded state, never an absence that the
        # parent can only label ``missing-receipt`` with zero evidence
        # (BL-12(f), fable W-F7). Best-effort by construction -- the original
        # exception always re-raises so the exit code stays honest.
        try:
            request.receipt_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostic = _aborted_before_receipt_diagnostic(request, exc)
            if request.protocol_version == "menagerie.crawler.worker-request.v3":
                _atomic_worker_result(request.receipt_path, None, diagnostic)
            else:
                _atomic_receipt(request.receipt_path, diagnostic)
        except Exception:  # noqa: BLE001 -- last-resort writer must never mask the abort
            pass
        raise
    if request.protocol_version == "menagerie.crawler.worker-request.v3":
        raw_receipt = _raw_award_receipt(request, payload, observation)
        return _atomic_worker_result(request.receipt_path, raw_receipt, payload)
    return _atomic_receipt(request.receipt_path, payload)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the private worker subprocess CLI.

    Parameters
    ----------
    argv:
        Optional argument vector.

    Returns
    -------
    argparse.Namespace
        Parsed request and receipt paths.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the argv-only worker entry point.

    Parameters
    ----------
    argv:
        Optional argument vector.

    Returns
    -------
    int
        Zero when every requested forward completed; one otherwise.
    """

    completion_challenge = os.environ.pop(_PARENT_COMPLETION_CHALLENGE_ENV, None)
    args = _parse_args(argv)
    request_bytes = args.request.read_bytes()
    request_value = json.loads(request_bytes)
    if not isinstance(request_value, dict):
        raise ValueError("worker request must be a JSON object")
    request = WorkerRequest.from_mapping(
        request_value,
        receipt_path=args.receipt,
        request_sha256=hash_bytes(request_bytes),
    )
    # The parent bindings have been consumed; pop them before any model import
    # so in-worker code can never read the request digest or manifest identity
    # back out of the environment (BL-12(a) hardening, opus W-6).
    os.environ.pop(_REQUEST_SHA256_ENV, None)
    os.environ.pop(_READ_MANIFEST_ID_ENV, None)
    if request.protocol_version == "menagerie.crawler.worker-request.v3":
        _start_parent_watchdog()
    receipt = run_worker(request)
    raw_receipt = receipt.get("raw_award_receipt")
    if request.protocol_version == "menagerie.crawler.worker-request.v3":
        success = isinstance(raw_receipt, Mapping)
    else:
        per_mode = receipt.get("per_mode", {})
        succeeded = bool(per_mode) and all(
            bool(item.get("forward_completed"))
            for item in per_mode.values()
            if isinstance(item, Mapping)
        )
        success = receipt.get("error") is None and succeeded
    if success and request.protocol_version == "menagerie.crawler.worker-request.v3":
        assert isinstance(raw_receipt, Mapping)
        completion_line = completion_line_for_raw_award_receipt(raw_receipt)
        if completion_challenge:
            # Attestation channel: an atomic file bound to the parent secret,
            # written by code that ran before any model import could observe
            # the challenge. Stdout below stays as human diagnostics only --
            # nothing a model prints can touch award validity (BL-12(a)).
            completion_record = {
                "version": _WORKER_COMPLETION_FILE_VERSION,
                "completion_line": completion_line,
                "challenge_proof": stable_hash(
                    {
                        "version": _WORKER_COMPLETION_PROOF_VERSION,
                        "challenge": completion_challenge,
                        "completion_line": completion_line,
                    }
                ),
            }
            atomic_replace_bytes(
                request.receipt_path.parent / _WORKER_COMPLETION_FILE_NAME,
                canonical_json_bytes(completion_record) + b"\n",
            )
        os.write(1, completion_line.encode("utf-8") + b"\n")
    elif success and completion_challenge:
        completion = {
            "receipt_sha256": receipt.get("receipt_sha256"),
            "proof": stable_hash(
                {
                    "version": "menagerie.crawler.worker-completion.v1",
                    "challenge": completion_challenge,
                    "receipt_sha256": receipt.get("receipt_sha256"),
                }
            ),
        }
        os.write(
            1, _WORKER_COMPLETION_PREFIX.encode("ascii") + canonical_json_bytes(completion) + b"\n"
        )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
