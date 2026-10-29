"""RECON PROBE (untracked, delete after): declarative R1 with metadata-read override.

Identical to ``probe_declarative_r1.py`` except for ONE clearly-declared
EXPERIMENTAL OVERRIDE: the generated macOS Seatbelt profile is extended with the
confined ``.dist-info``/``.egg-info`` metadata-name read grant for the worker's
source root (cwd), exactly mirroring the pattern the shipped
``_macos_runtime_root_read_patterns`` already emits for no-manifest runs and the
unconditional allowance the Linux classifier ``_runtime_import_metadata_path_allowed``
grants. The shipped generator is NOT modified; the override is a monkeypatched
wrapper, recon-only, to learn what breaks NEXT after the metadata-read denial.

Hand-written inputs (declared honestly): same as probe_declarative_r1.py --
library recipe fields for ``pytorch``/``torch.nn.Conv2d``, mode declarations.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import menagerie.crawler.driver as driver_module
from menagerie.crawler.authority import derive_mode_summary
from menagerie.crawler.driver import SupervisedForwardLane
from menagerie.crawler.env_lifecycle import parse_resolved_export
from menagerie.crawler.recipe import bind_library_artifact_digest
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer
from menagerie.crawler.tests.conftest import RealEnvironmentFixture
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

_DUMP_ROOT = Path(
    "/private/tmp/claude-501/-Users-jmt-projects-torchlens/"
    "a574bfe5-b8da-4e13-a75a-43ff4080f286/scratchpad/probe_r1c_evidence"
)


def _stage(name: str) -> None:
    print(f"\n=== PROBE STAGE: {name} ===", flush=True)


def test_probe_declarative_r1_torch_name(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
    monkeypatch: object,
) -> None:
    """Declarative R1 with the cwd metadata-read grant restored experimentally."""

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
        extra = (
            "; EXPERIMENTAL OVERRIDE (recon probe): confined source-root metadata reads\n"
            f"(allow file-read-data (regex {policy_module._sbpl_regex_literal(pattern)}))\n"
        )
        return profile + extra

    monkeypatch.setattr(  # type: ignore[attr-defined]
        supervisor_module, "generate_macos_sandbox_profile", _patched_generate
    )

    _stage("author fixture + real recipe substitution")
    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    context = _test_authority_context(snapshot, driver.config)
    artifact = FakeAuthor().author(item, paths.work_root, driver.config, context)

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    implementation = facts["implementation"]

    export_bytes = real_environment_fixture.intent.lock.export_bytes
    assert export_bytes is not None
    packages = json.loads(parse_resolved_export(export_bytes))["packages"]
    torch_rows = [row for row in packages if row["name"] == "pytorch"]
    assert len(torch_rows) == 1, "fixture env must install exactly one pytorch"
    torch_version = torch_rows[0]["version"]
    print(f"fixture pytorch version: {torch_version}")

    implementation["library_recipe"] = {
        "distribution": "torch",
        "version": str(torch_version),
        "module": "torch.nn",
        "symbol": "Conv2d",
        "kwargs": {"in_channels": 3, "out_channels": 4, "kernel_size": 3},
        "pretrained_disable_fields": [],
    }
    changed = bind_library_artifact_digest(implementation, packages)
    print(
        "bind_library_artifact_digest changed recipe:",
        changed,
        "-> artifact_sha256:",
        implementation["library_recipe"].get("artifact_sha256"),
    )

    facts["modes"]["meaningful_modes"] = ["train", "eval"]
    facts["modes"]["train_eval_divergence"] = "none"
    facts["external_metadata"]["modes"]["meaningful_modes"] = ["train", "eval"]
    facts["external_metadata"]["modes"]["train_eval_divergence"] = "none"
    _refresh_proposal_identities(
        proposal,
        checker_model=driver.config.checker_model,
        checker_version=driver.config.checker_version,
    )
    artifact = _rebind_fake_author_result(artifact)

    environment = real_environment_fixture.binding
    context = replace(
        context,
        environment_generations={
            **context.environment_generations,
            environment.family: environment.env_generation,
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

        _stage("supervised forward lane (real subprocess worker in real prefix)")
        attempts = SupervisedForwardLane(timeout_seconds=120, cwd=Path.cwd()).forward(
            artifact,
            environment,
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
                json.dumps(attempt.get("error"))[:2000],
            )

        _stage("evidence dump (probe-only, before autouse scratch cleanup)")
        import shutil as _shutil

        _shutil.rmtree(_DUMP_ROOT, ignore_errors=True)
        _DUMP_ROOT.mkdir(parents=True, exist_ok=True)
        for diagnostic_path in sorted(tmp_path.rglob("diagnostics/*.json")):
            _shutil.copy2(diagnostic_path, _DUMP_ROOT / diagnostic_path.name)
        for receipt_path in sorted(tmp_path.rglob("receipt.json")):
            rel = "__".join(receipt_path.parts[-6:])
            _shutil.copy2(receipt_path, _DUMP_ROOT / rel)
        for audit_path in sorted(tmp_path.rglob("*.ndjson")):
            if audit_path.is_file():
                rel = "__".join(audit_path.parts[-6:])
                try:
                    _shutil.copy2(audit_path, _DUMP_ROOT / rel)
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
                    "output=",
                    json.dumps(receipt.get("output_signature"))[:300],
                    "params=",
                    receipt.get("parameter_count_total"),
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
    print("status:", appended.record["status"])
    print("modes:", json.dumps(appended.record["modes"])[:600])
    assert appended.record["status"]["code"] == "runs"
    assert driver_module._current_run_is_fresh(
        appended.record,
        artifact,
        environment,
        [gate],
    )
    print("\nPROBE RESULT: declarative-library R1 recipe reached a runs record")
