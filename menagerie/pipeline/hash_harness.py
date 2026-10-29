"""The hash harness: whether ``architecture`` can become the headline and dedup key.

Owned by the hash lane (rung 2). Over 30 traced models it checks ``trace_summary``'s hash family
(``skeleton``, ``architecture``, ``model``), read from the worker's own capture
(``worker._hash_capture``), so it checks the hashes the ledger records:

- ``same_process``: two fresh builds in one process hash the same at every level;
- ``fresh_process``: a build in a second process hashes the same at every level;
- ``wrapper``: the model under two wrapper modules keeps its skeleton and architecture, and the
  two wrappers' ``model`` hashes differ (module organisation is what ``model`` adds);
- ``batch`` and ``resolution``: twice the batch, and twice the spatial size or sequence length,
  keep the architecture. When the skeleton changes too the topology really depends on the input
  (an unrolled loop, a size-dependent branch): that is recorded as an exception, never forced
  into a collision;
- ``kernel``: the first odd-kernel convolution grown by two (padding keeps every shape) changes
  the architecture and keeps the skeleton;
- ``width``: a real pair of library variants that differ only in width
  (``convnext_atto``, ``convnext_femto``) has different architectures and one skeleton;
- constructed cases, never named models: one computation in two module organisations, an MLP at
  two widths, a pooling kernel that only an argument carries, and attention at two head counts.

``structure_only`` (the meta-device capture keeps the full capture's architecture) and the head
count are reported and do not gate. A variant the model refuses (a fixed input size, a meta build
that cannot run) is skipped with its reason. The harness passes when no gating check fails and
every gating per-model kind was checked on at least ``MIN_CHECKED`` models.

Each model runs in its own child process (a crash or a leak stays there); a second child is the
fresh process. The verdict (``verdict.json`` under ``out``; committed as
``menagerie/data/hash_harness.json``) is what ``headline_basis`` reads.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, now, read_json

# Thirty models that traced and passed at rung 1 and build in ``torch-main`` with no extras or
# vendor snapshot, chosen for spread: convolutional, attention, hybrid and multi-branch image
# models, text, detection and segmentation heads, recurrent cells, an adaptive ODE solver and
# adapter-built classics. (Graph networks need the ``torch_geometric`` extra, and vendored
# adapters a staged snapshot, so neither is here.)
HARNESS_IDS = (
    "m8839", "m7341", "m3882", "m3888", "m3825", "m3711", "m3877", "m3730", "m3739", "m4184",
    "m4194", "m3963", "m4447", "m10921", "m7325", "m3638", "m5228", "m9427", "m11497", "m10036",
    "m413", "m2061", "m1799", "m4331", "m1835", "m1643", "m3297", "m14347", "m3971", "m7836",
)
WIDTH_PAIR = ("m3882", "m3888")  # convnext_atto and convnext_femto: one depth, two widths
LEVELS = ("skeleton", "architecture", "model")
GATING = ("same_process", "fresh_process", "wrapper", "batch", "resolution", "kernel")
MIN_CHECKED = 10
# The backends whose captures the harness checks. TorchLens preview backends (TensorFlow, JAX,
# Paddle, MLX) are not run here, so their hashes never count under the architecture headline.
HASHED_BACKENDS = ("torch",)
VERDICT_FILE = "hash_harness.json"
CHILD_TIMEOUT_S = 1800

# What each per-model case compares: (variant, variant, levels that must agree, levels that must
# differ). An invariance case whose skeletons differ is the input-dependent-topology exception.
PAIRS: dict[str, tuple[str, str, tuple[str, ...], tuple[str, ...]]] = {
    "same_process": ("default", "again", LEVELS, ()),
    "fresh_process": ("default", "fresh", LEVELS, ()),
    "wrapper": ("wrap_a", "wrap_b", ("skeleton", "architecture"), ("model",)),
    "batch": ("default", "batch", ("architecture",), ()),
    "resolution": ("default", "resolution", ("architecture",), ()),
    "kernel": ("default", "kernel", ("skeleton",), ("architecture", "model")),
    "structure_only": ("default", "meta", ("architecture",), ()),
}
INVARIANT = ("batch", "resolution", "structure_only")


def harness(ids: Sequence[str] = (), *, data: DataLayout, out: Path | None = None) -> dict[str, Any]:
    """Run the hash harness and report each case.

    Inputs: stable ids (default: the harness's 30), the data root and a report folder.
    Returns: ``{"passed": bool, "cases": [{"case", "id", "hash", "ok", "why"}]}`` (``hash`` names
    the level that decided the case). Side effects: ``report.json`` and ``verdict.json`` under
    ``out`` when given.
    """

    ids = tuple(ids) or HARNESS_IDS
    probes = {sid: _probe_pair(sid, data) for sid in ids}
    cases = [case for sid in ids for case in model_cases(sid, *probes[sid])]
    cases += width_cases(probes) + constructed_cases()
    report = {"passed": passed(cases), "cases": cases}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        verdict = summarize(cases, probes)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8")
    return report


def passed(cases: Sequence[Mapping[str, Any]]) -> bool:
    """No gating check failed, and each gating per-model kind was checked often enough."""

    gating = [c for c in cases if c.get("gates", True)]
    if any(not c["ok"] for c in gating):
        return False
    checked = Counter(c["case"] for c in gating if c["ok"] is True and not c.get("skipped"))
    return all(checked[kind] >= MIN_CHECKED for kind in GATING)


def summarize(cases: Sequence[Mapping[str, Any]], probes: Mapping[str, Any]) -> dict[str, Any]:
    """The committed verdict: pass, versions, counts per case and outcome, and the failures."""

    counts: dict[str, Counter[str]] = {}
    for case in cases:
        counts.setdefault(case["case"], Counter())[_outcome(case)] += 1
    versions = next((p[0]["versions"] for p in probes.values() if p[0].get("versions")), {})
    failures = [c for c in cases if not c["ok"]]
    return {
        "schema": "hash-harness/1", "at": now(), "passed": passed(cases), **versions,
        "backends": list(HASHED_BACKENDS),
        "models": len(probes), "min_checked": MIN_CHECKED,
        "counts": {k: dict(sorted(v.items())) for k, v in sorted(counts.items())},
        "failures": [{k: c.get(k) for k in ("case", "id", "hash", "why", "gates")}
                     for c in failures[:20]],
    }


def _outcome(case: Mapping[str, Any]) -> str:
    if case.get("skipped"):
        return "skipped"
    if case.get("exception"):
        return "exception"
    return "ok" if case["ok"] else "failed"


def headline_basis(data: DataLayout) -> dict[str, Any]:
    """Which hash counts distinct architectures: ``architecture`` once a committed verdict passed.

    Returns ``{"basis": "architecture" | "graph_shape_hash", "schema", "backends"}``: the hash
    schema the harness passed at and the backends it checked; only those hashes count under it.
    Reads ``hash_harness.json`` under the data root; no torch (publish runs on the hub).
    """

    legacy = {"basis": "graph_shape_hash", "schema": None, "backends": []}
    path = data.root / VERDICT_FILE
    if not path.exists():
        return legacy
    verdict = json.loads(path.read_text(encoding="utf-8"))
    if verdict.get("passed") is not True:
        return legacy
    return {"basis": "architecture", "schema": int(verdict["hash_schema"]),
            "backends": list(verdict.get("backends") or HASHED_BACKENDS)}


def dedup_key(hashes: Mapping[str, Any] | None, graph_shape_hash: str | None,
              basis: Mapping[str, Any], backend: str = "torch") -> str:
    """A record's "same architecture" key (``dedup_architecture_key``) on the headline basis.

    Under ``architecture`` only a hash at the passing schema and from a backend the harness
    checked counts; any other record has no key (``""``), as a preview trace has no legacy
    ``graph_shape_hash`` either.
    """

    if basis["basis"] != "architecture":
        return graph_shape_hash or ""
    hashes = hashes or {}
    checked = hashes.get("schema") == basis["schema"] and backend in basis["backends"]
    return hashes.get("architecture") or "" if checked else ""


def basis_label(basis: Mapping[str, Any]) -> str:
    if basis["basis"] != "architecture":
        return "legacy graph_shape_hash"
    return f"architecture hash, schema {basis['schema']}"


# --- comparing probes ---------------------------------------------------------------


def model_cases(sid: str, probe: Mapping[str, Any], fresh: Mapping[str, Any]) -> list[dict]:
    """The per-model cases from one child's variants and the fresh child's default."""

    variants = {**probe.get("variants", {}), "fresh": fresh.get("variants", {}).get("default")}
    if probe.get("error") or not _hashed(variants.get("default")):
        why = probe.get("error") or _why(variants.get("default"))
        return [_case("default", sid, None, False, f"the default build did not hash: {why}")]
    return [compare(kind, sid, variants) for kind in PAIRS]


def compare(kind: str, sid: str, variants: Mapping[str, Any]) -> dict[str, Any]:
    """One case: the two variants' hashes against what the case expects."""

    first, second, same, differ = PAIRS[kind]
    a, b = variants.get(first), variants.get(second)
    gates = kind != "structure_only"
    for variant in (a, b):
        if not _hashed(variant):
            return _case(kind, sid, None, True, f"skipped: {_why(variant)}", gates, skipped=True)
    if kind == "structure_only" and a["hashes"]["skeleton"] != b["hashes"]["skeleton"]:
        why = (f"the meta capture records another graph ({a['n_ops']} ops, then {b['n_ops']}; "
               f"{_first_diff(a, b)})")
        return _case(kind, sid, "skeleton", False, why, gates)
    if kind in INVARIANT and a["hashes"]["skeleton"] != b["hashes"]["skeleton"]:
        why = f"topology depends on the input ({a['n_ops']} ops, then {b['n_ops']})"
        return _case(kind, sid, "skeleton", True, why, gates, exception=True)
    for level in same:
        if a["hashes"][level] != b["hashes"][level]:
            return _case(kind, sid, level, False, f"{level} differs: {_first_diff(a, b)}", gates)
    for level in differ:
        if a["hashes"][level] == b["hashes"][level]:
            return _case(kind, sid, level, False, f"{level} collides", gates)
    return _case(kind, sid, (differ or same)[-1], True, "as expected", gates)


def _case(kind: str, sid: str, level: str | None, ok: bool, why: str, gates: bool = True,
          **flags: bool) -> dict[str, Any]:
    return {"case": kind, "id": sid, "hash": level, "ok": ok, "why": why, "gates": gates,
            **{k: v for k, v in flags.items() if v}}


def _hashed(variant: Mapping[str, Any] | None) -> bool:
    return bool(variant) and "hashes" in variant


def _why(variant: Mapping[str, Any] | None) -> str:
    if not variant:
        return "not run"
    return variant.get("skipped") or variant.get("error") or "no hashes"


def _first_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    """Where two architecture payloads first differ: the op index, its function and the keys."""

    left, right = a.get("payload") or [], b.get("payload") or []
    for index, (x, y) in enumerate(zip(left, right)):
        if x != y:
            keys = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
            detail = {k: [x.get(k), y.get(k)] for k in keys}
            return f"op {index} ({x.get('func')}): {json.dumps(detail, default=str)[:400]}"
    return f"payload lengths {len(left)} and {len(right)}"


def width_cases(probes: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The real width pair: different architectures, one skeleton."""

    narrow, wide = WIDTH_PAIR
    if narrow not in probes or wide not in probes:
        return []
    variants = {"default": probes[narrow][0].get("variants", {}).get("default"),
                "wide": probes[wide][0].get("variants", {}).get("default")}
    return [_pair_case("width", f"{narrow}/{wide}", variants["default"], variants["wide"])]


def _pair_case(kind: str, label: str, a: Any, b: Any, *, gates: bool = True) -> dict[str, Any]:
    """A discrimination pair: one skeleton, different architectures."""

    if not (_hashed(a) and _hashed(b)):
        return _case(kind, label, None, False, f"not hashed: {_why(a)}; {_why(b)}", gates)
    if a["hashes"]["skeleton"] != b["hashes"]["skeleton"]:
        return _case(kind, label, "skeleton", False, "not a pure change: the skeletons differ",
                     gates)
    if a["hashes"]["architecture"] == b["hashes"]["architecture"]:
        return _case(kind, label, "architecture", False, "architecture collides", gates)
    return _case(kind, label, "architecture", True, "as expected", gates)


# --- constructed cases (no named model; run in this process) -------------------------


def constructed_cases() -> list[dict[str, Any]]:
    """One computation in two organisations, and three discrimination pairs on tiny modules."""

    import torch

    from menagerie.pipeline import worker

    def hashed(model: Any, *inputs: Any) -> dict[str, Any]:
        return _hash_subject(worker.Subject(model.eval(), inputs, {}))

    worker._wrap_torch()
    nets = _constructed_modules()
    vec, image, seq = torch.ones(2, 8), torch.ones(1, 3, 12, 12), torch.ones(1, 6, 16)
    seq_net, flat_net = hashed(nets["sequential"](), vec), hashed(nets["functional"](), vec)
    cases = [_organisation_case(seq_net, flat_net)]
    for kind, (make, x), gates in (("width_mlp", (nets["mlp"], vec), True),
                                   ("pool_kernel", (nets["pool"], image), True),
                                   ("heads", (nets["attention"], seq), False)):
        cases.append(_pair_case(kind, "constructed", hashed(make(0), x), hashed(make(1), x),
                                gates=gates))
    return cases


def _organisation_case(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    for level in ("skeleton", "architecture"):
        if a["hashes"][level] != b["hashes"][level]:
            return _case("organisation", "constructed", level, False,
                         f"{level} differs: {_first_diff(a, b)}")
    if a["hashes"]["model"] == b["hashes"]["model"]:
        return _case("organisation", "constructed", "model", False, "model collides")
    return _case("organisation", "constructed", "architecture", True, "as expected")


def _constructed_modules() -> dict[str, Callable[..., Any]]:
    """Tiny modules for the constructed cases; ``make(0)`` and ``make(1)`` are the two sides."""

    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(0)

    class Functional(nn.Module):
        """The sequential net's computation with its parameters on one module."""

        def __init__(self) -> None:
            super().__init__()
            self.w1, self.b1 = nn.Parameter(torch.randn(16, 8)), nn.Parameter(torch.zeros(16))
            self.w2, self.b2 = nn.Parameter(torch.randn(4, 16)), nn.Parameter(torch.zeros(4))

        def forward(self, x: Any) -> Any:
            return F.linear(F.relu(F.linear(x, self.w1, self.b1)), self.w2, self.b2)

    class Pool(nn.Module):
        def __init__(self, kernel: int) -> None:
            super().__init__()
            self.conv, self.kernel = nn.Conv2d(3, 4, 3, padding=1), kernel

        def forward(self, x: Any) -> Any:
            return F.max_pool2d(self.conv(x), self.kernel)

    class Attention(nn.Module):
        def __init__(self, heads: int) -> None:
            super().__init__()
            self.qkv, self.proj, self.heads = nn.Linear(16, 48), nn.Linear(16, 16), heads

        def forward(self, x: Any) -> Any:
            b, n, c = x.shape
            q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(
                2, 0, 3, 1, 4).unbind(0)
            out = F.scaled_dot_product_attention(q, k, v)
            return self.proj(out.transpose(1, 2).reshape(b, n, c))

    return {
        "sequential": lambda: nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)),
        "functional": Functional,
        "mlp": lambda side: nn.Sequential(nn.Linear(8, (16, 32)[side]), nn.ReLU(),
                                          nn.Linear((16, 32)[side], 4)),
        "pool": lambda side: Pool((2, 3)[side]),
        "attention": lambda side: Attention((2, 4)[side]),
    }


# --- the child process -----------------------------------------------------------------


def _probe_pair(sid: str, data: DataLayout) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a model's variants in one child and its default again in a fresh child."""

    return _child(sid, "all", data), _child(sid, "default", data)


def _child(sid: str, which: str, data: DataLayout) -> dict[str, Any]:
    env = {**os.environ, "MENAGERIE_DATA": str(data.root)}
    cmd = [sys.executable, "-m", "menagerie.pipeline.hash_harness", sid, which]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False,
                              timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"error": f"child timed out after {CHILD_TIMEOUT_S} s"}
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode != 0 or not lines:
        tail = (done.stderr.strip().splitlines() or ["no output"])[-1][:300]
        return {"error": f"child exited {done.returncode}: {tail}"}
    return json.loads(lines[-1])


def probe(sid: str, which: str, data: DataLayout) -> dict[str, Any]:
    """In the child: hash a model's variants (``which`` is ``all`` or ``default``)."""

    from menagerie.pipeline.records import ModelRecord

    recipe = read_json(ModelRecord, data.model(sid)).recipe
    names = ("default",) if which == "default" else VARIANTS
    return {"id": sid, "versions": _versions(),
            "variants": {name: _variant(recipe, name) for name in names}}


def _versions() -> dict[str, Any]:
    import torch
    import torchlens as tl

    from menagerie import trace_summary

    return {"hash_schema": trace_summary.HASH_SCHEMA_VERSION, "torchlens": tl.__version__,
            "torch": torch.__version__, "trace_summary": trace_summary.TRACE_SUMMARY_VERSION}


VARIANTS = ("default", "again", "wrap_a", "wrap_b", "batch", "resolution", "kernel", "meta")


def _variant(recipe: Any, name: str) -> dict[str, Any]:
    """Build the variant, capture it as the worker does and hash it; a refusal is recorded."""

    import gc

    from menagerie.pipeline import worker

    try:
        subject = _subject(recipe, name)
        if isinstance(subject, str):
            return {"skipped": subject}
        return _hash_subject(subject, structure_only=name == "meta")
    except worker.Failure as failure:
        return {"error": f"{failure.cls}: {failure.msg}"[:300]}
    except Exception as exc:  # noqa: BLE001 -- a variant the model refuses is a recorded skip
        return {"skipped": f"{type(exc).__name__}: {str(exc)[:200]}"}
    finally:
        gc.collect()


def _subject(recipe: Any, name: str) -> Any:
    """The variant's fresh build, or a string saying why the variant does not apply."""

    from menagerie.pipeline import worker

    if name in ("batch", "resolution"):
        shape = _varied_shape(recipe.input, name)
        if isinstance(shape, str):
            return shape
        recipe = replace(recipe, input=replace(recipe.input, shape=shape))
    subject = worker._build(recipe, device="meta" if name == "meta" else "cpu")
    if name in ("wrap_a", "wrap_b"):
        return replace(subject, model=_wrapped(subject.model, name))
    if name == "kernel":
        return subject if _grow_first_kernel(subject.model) else "no odd-kernel convolution"
    return subject


def _varied_shape(spec: Any, name: str) -> tuple[int, ...] | str:
    """Twice the batch, or twice the spatial size (image, 4-D and 5-D random) or sequence."""

    if spec.shape is None:
        return f"the {spec.preset.partition(':')[0]} preset fixes the input"
    shape = list(spec.shape)
    if name == "batch":
        return (shape[0] * 2, *shape[1:])
    preset = spec.preset.partition(":")[0]
    if preset == "text" and len(shape) == 2:
        return (shape[0], shape[1] * 2)
    if preset in ("image", "randn") and len(shape) in (4, 5):
        return (*shape[:-2], shape[-2] * 2, shape[-1] * 2)
    return f"no size axis known for a {len(shape)}-D {preset} input"


def _wrapped(model: Any, name: str) -> Any:
    """The model under one of two wrapper modules that add no computation."""

    from torch import nn

    class WrapA(nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return self.inner(*args, **kwargs)

    class WrapB(nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.outer = nn.ModuleDict({"core": inner})

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return self.outer["core"](*args, **kwargs)

    return (WrapA if name == "wrap_a" else WrapB)(model)


def _grow_first_kernel(model: Any) -> bool:
    """Grow the first odd, square, undilated, zero-padded 'same'-style kernel by two, in place.

    Padding grows by one with it, so every output shape and the topology stay as they were.
    """

    from torch import nn

    for parent in model.modules():
        for child_name, conv in parent.named_children():
            if not isinstance(conv, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) or not _growable(conv):
                continue
            k = conv.kernel_size[0] + 2
            grown = type(conv)(conv.in_channels, conv.out_channels, k, stride=conv.stride,
                               padding=k // 2, groups=conv.groups, bias=conv.bias is not None,
                               dtype=conv.weight.dtype)
            setattr(parent, child_name, grown)
            return True
    return False


def _growable(conv: Any) -> bool:
    k = conv.kernel_size
    return (len(set(k)) == 1 and k[0] % 2 == 1 and set(conv.dilation) == {1}
            and not isinstance(conv.padding, str) and set(conv.padding) == {k[0] // 2}
            and conv.padding_mode == "zeros")


def _hash_subject(subject: Any, *, structure_only: bool = False) -> dict[str, Any]:
    from menagerie import trace_summary
    from menagerie.pipeline import worker

    trace = worker._hash_capture(subject, structure_only=structure_only)
    family = trace_summary.hash_family(trace)
    return {"hashes": {level: family[level] for level in LEVELS}, "n_ops": len(trace.layer_list),
            "payload": trace_summary.hash_payload(trace, with_module=False)}


def _main(argv: Sequence[str]) -> int:
    sid, which = argv
    result = probe(sid, which, DataLayout.from_env())
    print(json.dumps(result, default=str, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
