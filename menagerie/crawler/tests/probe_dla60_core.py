"""RECON PROBE (untracked, delete after): timm dla60 R1 in a freshly solved core env.

End-to-end recon of the execution half against a core-intent environment that was
REALLY solved and materialised today by the shipped environment operator:

  solve:  ``operator_environment solve envs/core/environment.yml osx-arm64``
  create: staged CAS + offline ``mamba create`` (conda 26.3.2 offline create is
          broken -- reported separately)

The lock family (explicit.lock / resolved.json / resolved.sha256) is arranged in
the production ``envs/<intent>/locks/<target>.*`` layout in a scratch envs root,
loaded with the production ``load_environment_registry``, probed with the REAL
``operator_environment probe`` subcommand, and bound with the production
``bind_materialized_environment``. No fixture forgery: the round19 release
fixture is not used at all.

EXPERIMENTAL OVERRIDE (recon-only, clearly declared): the generated macOS
Seatbelt profile is extended with the confined ``.dist-info``/``.egg-info``
metadata-name read grant for the worker's source root, mirroring the shipped
no-manifest pattern and the Linux classifier (defect already mapped; fix in
branch probe/r1-macos-metadata-grant).

Hand-written inputs (declared honestly):
- library recipe fields: ``timm``/``timm.create_model``, model_name dla60,
  pretrained False, version taken from the solved lock's timm row;
- input contract shape [1, 3, 224, 224];
- mode declarations train+eval with divergence "statistical" (dla60 carries
  BatchNorm, so train/eval genuinely diverge);
- the checker gate is the same stubbed FakeChecker the shipped typed-adapter
  real-execution test uses.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import menagerie.crawler.driver as driver_module
from menagerie.crawler.authority import derive_mode_summary
from menagerie.crawler.driver import (
    SupervisedForwardLane,
    bind_materialized_environment,
)
from menagerie.crawler.driver_admission import EnvironmentAuthorityCache
from menagerie.crawler.env_lifecycle import ProbeResult, parse_resolved_export
from menagerie.crawler.envs import load_environment_registry
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.recipe import bind_library_artifact_digest
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakeChecker,
    _driver,
    _paths,
    _rebind_fake_author_result,
    _refresh_proposal_identities,
    _snapshot,
    _test_authority_context,
)

_SC = Path(
    "/private/tmp/claude-501/-Users-jmt-projects-torchlens/"
    "a574bfe5-b8da-4e13-a75a-43ff4080f286/scratchpad"
)
_SOLVE_ROOT = _SC / "env-operator-state/solves/core/osx-arm64"
_PREFIX = _SC / "prefixes/core-mamba-test"
_REPO_ENVS = Path(__file__).resolve().parents[1] / "envs"
_DUMP_ROOT = _SC / "probe_dla60_evidence"


def _stage(name: str) -> None:
    print(f"\n=== PROBE STAGE: {name} ===", flush=True)


def _scratch_envs_root() -> Path:
    """Arrange the real solve products in the production envs layout."""

    root = _SC / "envs-root-core"
    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(
        _REPO_ENVS,
        root,
        ignore=shutil.ignore_patterns("locks", "__pycache__"),
    )
    lock_dir = root / "core" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_SOLVE_ROOT / "explicit.lock", lock_dir / "osx-arm64.lock")
    export_bytes = (_SOLVE_ROOT / "resolved.json").read_bytes()
    (lock_dir / "osx-arm64.resolved.json").write_bytes(export_bytes)
    (lock_dir / "osx-arm64.resolved.sha256").write_text(
        hash_bytes(export_bytes) + "\n", encoding="utf-8"
    )
    return root


def _real_probe_results(probes_json: str) -> tuple[ProbeResult, ...]:
    """Run the REAL environment-operator probe subcommand on the fresh prefix."""

    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "menagerie.crawler.operator_environment",
            "--state-root",
            str(_SC / "env-operator-state"),
            "probe",
            str(_PREFIX),
            probes_json,
        ),
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    payload = json.loads(completed.stdout)
    return tuple(
        ProbeResult(str(row["name"]), bool(row["passed"]), str(row["detail"]))
        for row in payload["results"]
    )


def test_probe_dla60_core_declarative_r1(tmp_path: Path, monkeypatch: object) -> None:
    """Drive timm dla60 down the real declarative path in the fresh core env."""

    import menagerie.crawler.policy as policy_module
    import menagerie.crawler.worker_supervisor as supervisor_module

    shipped_generate = supervisor_module.generate_macos_sandbox_profile
    source_root = Path.cwd().resolve()

    def _patched_generate(*args: object, **kwargs: object) -> str:
        profile = shipped_generate(*args, **kwargs)
        escaped = re.escape(str(source_root))
        descendant = f"{escaped}/(.*/)?"
        metadata_names = "|".join(
            re.escape(name) for name in sorted(policy_module._RUNTIME_METADATA_NAMES)
        )
        pattern = f"^{descendant}[^/]*\\.(dist-info|egg-info)/(.*/)?({metadata_names})$"
        extended = profile + (
            "; EXPERIMENTAL OVERRIDE (recon probe): confined source-root metadata reads\n"
            f"(allow file-read-data (regex {policy_module._sbpl_regex_literal(pattern)}))\n"
        )
        _DUMP_ROOT.mkdir(parents=True, exist_ok=True)
        (_DUMP_ROOT / "generated-profile.sb").write_text(extended, encoding="utf-8")
        return extended

    monkeypatch.setattr(  # type: ignore[attr-defined]
        supervisor_module, "generate_macos_sandbox_profile", _patched_generate
    )

    _stage("load solved core intent through the production registry loader")
    registry = load_environment_registry(_scratch_envs_root(), target="osx-arm64")
    intent = registry.intents["core"]
    assert intent.lock_status == "locked", intent.lock_status
    print("core intent lock status:", intent.lock_status)

    _stage("real operator probe of the fresh prefix")
    probes_json = json.dumps(
        {
            "imports": list(intent.probes.imports),
            "source_build": [],
            "export_checks": [
                {"module": check.module, "attribute": check.attribute}
                for check in intent.probes.export_checks
            ],
        }
    )
    probe_results = _real_probe_results(probes_json)
    for result in probe_results:
        print("probe:", result.name, result.passed)
    assert all(result.passed for result in probe_results)

    # Persist the REAL observed probe results as the durable canonical receipt
    # the strict binder requires beside the lock family (honest observation,
    # not a fabricated attestation).
    from menagerie.crawler.env_lifecycle import canonical_probe_receipt_bytes

    receipt_path = _SC / "envs-root-core" / "core" / "locks" / "osx-arm64.probes.json"
    receipt_path.write_bytes(canonical_probe_receipt_bytes(probe_results))

    _stage("production strict binding of the materialized prefix")
    binding = bind_materialized_environment(
        intent,
        _PREFIX,
        probe_results,
        authority_cache=EnvironmentAuthorityCache(),
    )
    print("bound family:", binding.family, "generation:", binding.env_generation)

    _stage("author fixture + real dla60 recipe substitution")
    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    context = _test_authority_context(snapshot, driver.config)
    artifact = FakeAuthor().author(item, paths.work_root, driver.config, context)

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    implementation = facts["implementation"]

    export_bytes = intent.lock.export_bytes
    assert export_bytes is not None
    packages = json.loads(parse_resolved_export(export_bytes))["packages"]
    timm_rows = [row for row in packages if row["name"] == "timm"]
    assert len(timm_rows) == 1, "core lock must install exactly one timm"
    timm_version = timm_rows[0]["version"]
    print("core lock timm version:", timm_version)

    implementation["library_recipe"] = {
        "distribution": "timm",
        "version": str(timm_version),
        "module": "timm",
        "symbol": "create_model",
        "kwargs": {"model_name": "dla60", "pretrained": False},
        "pretrained_disable_fields": ["pretrained"],
    }
    changed = bind_library_artifact_digest(implementation, packages)
    print(
        "derived artifact_sha256:",
        implementation["library_recipe"].get("artifact_sha256"),
        f"(changed={changed})",
    )
    assert implementation["library_recipe"].get("artifact_sha256") is not None

    facts["input_contract"]["args"][0]["shape"] = [1, 3, 224, 224]
    facts["modes"]["meaningful_modes"] = ["train", "eval"]
    facts["modes"]["train_eval_divergence"] = "statistical"
    facts["external_metadata"]["modes"]["meaningful_modes"] = ["train", "eval"]
    facts["external_metadata"]["modes"]["train_eval_divergence"] = "statistical"
    _refresh_proposal_identities(
        proposal,
        checker_model=driver.config.checker_model,
        checker_version=driver.config.checker_version,
    )
    artifact = _rebind_fake_author_result(artifact)

    context = replace(
        context,
        environment_generations={
            **context.environment_generations,
            binding.family: binding.env_generation,
        },
    )

    with CanonicalReducer(paths.ledgers, context) as reducer:
        _stage("stage author result (private custody)")
        artifact = driver._stage_author_result(item, artifact, reducer)
        print("staged ok")

        _stage("checker gate (stubbed FakeChecker, as in the typed-adapter map test)")
        gate = FakeChecker().check_metadata([artifact], paths.work_root, driver.config).gate
        assert gate is not None
        reducer.append_gate(gate)
        print("gate appended")

        _stage("supervised forward lane (real subprocess worker in fresh core prefix)")
        attempts = SupervisedForwardLane(timeout_seconds=900, cwd=Path.cwd()).forward(
            artifact,
            binding,
            1,
            paths.work_root,
            worker_lock_path=paths.worker_lock,
            worker_lease_path=paths.worker_lease,
            run_id=driver.config.run_id,
            attempt_sink=reducer.append_attempt,
        )
        for attempt in attempts:
            print(
                "attempt:",
                attempt.get("mode"),
                attempt.get("result"),
                "error:",
                json.dumps(attempt.get("error"))[:1500],
            )

        _stage("evidence dump (probe-only, before autouse scratch cleanup)")
        _DUMP_ROOT.mkdir(parents=True, exist_ok=True)
        for diagnostic_path in sorted(tmp_path.rglob("diagnostics/*.json")):
            shutil.copy2(diagnostic_path, _DUMP_ROOT / diagnostic_path.name)
        for receipt_path in sorted(tmp_path.rglob("receipt.json")):
            rel = "__".join(receipt_path.parts[-6:])
            shutil.copy2(receipt_path, _DUMP_ROOT / rel)
        for audit_path in sorted(tmp_path.rglob("*.ndjson")):
            if audit_path.is_file():
                rel = "__".join(audit_path.parts[-6:])
                try:
                    shutil.copy2(audit_path, _DUMP_ROOT / rel)
                except OSError:
                    pass
        for mode in ("train", "eval"):
            result_path = (
                paths.work_root
                / item.stable_id
                / "forward"
                / "cold-1"
                / mode
                / "result"
                / "receipt.json"
            )
            if not result_path.is_file():
                print(f"[{mode}] no worker result file at {result_path}")
                continue
            wrapper = json.loads(result_path.read_text(encoding="utf-8"))
            diag = wrapper.get("diagnostic", {})
            if diag.get("error") is not None:
                print(f"[{mode}] diagnostic error:", json.dumps(diag["error"])[:3000])
            for mode_name, receipt in (diag.get("per_mode") or {}).items():
                print(
                    f"[{mode}] per_mode[{mode_name}]:",
                    "forward_completed=",
                    receipt.get("forward_completed"),
                    "params=",
                    receipt.get("parameter_count_total"),
                    "ctor_s=",
                    round(receipt.get("constructor_seconds") or -1, 2),
                    "fwd_s=",
                    round(receipt.get("forward_seconds") or -1, 2),
                    "output=",
                    json.dumps(receipt.get("output_signature"))[:220],
                )
        assert all(attempt["result"] == "succeeded" for attempt in attempts), [
            (attempt["result"], attempt["error"]) for attempt in attempts
        ]

        _stage("mode summary + model assembly + reducer award")
        persisted = scan_jsonl(paths.ledgers.attempts)
        assert len(persisted) == 2
        by_mode = {str(attempt["mode"]): attempt for attempt in persisted}
        summary = derive_mode_summary(by_mode["train"], by_mode["eval"])
        print(
            "mode summary:",
            summary.comparison_state,
            summary.classification,
            summary.compared_fields,
        )
        model = driver_module._assemble_run_model(
            item,
            artifact,
            persisted,
            [gate],
            driver.config,
        )
        appended = reducer.append_model(reducer.prepare_model(model))

    _stage("final record")
    print("status:", appended.record["status"]["code"])
    print("divergence:", appended.record["modes"]["train_eval_divergence"])
    assert appended.record["status"]["code"] == "runs"
    print("\nPROBE RESULT: timm dla60 R1 recipe reached a runs record in the fresh core env")
