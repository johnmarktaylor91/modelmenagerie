"""RECON PROBE (untracked, delete after): timm dla60 R1 recipe, real core env.

Same composition as ``probe_declarative_r1.py`` but the library recipe is the
caller-requested candidate: ``timm.create_model("dla60", pretrained=False)``,
executed inside a prefix genuinely solved and created by the shipped
environment operator (``core`` intent, osx-arm64). Point
``MENAGERIE_REAL_ENV_PREFIX`` at that prefix before running.

Hand-written inputs (declared honestly):
- the library recipe fields (distribution/version/module/symbol/kwargs);
- the input contract shape [1, 3, 224, 224];
- the proposal mode declarations (train+eval, divergence "statistical" --
  dla60 carries BatchNorm, so train/eval outputs genuinely diverge).
"""

from __future__ import annotations

import json
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


def test_probe_dla60_declarative_r1(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
    monkeypatch: object,
) -> None:
    """Drive timm dla60 down the real declarative execution path.

    EXPERIMENTAL OVERRIDE (recon-only, clearly declared): extends the
    parent-side benign-OS-metadata read allowlist by exactly one path,
    ``/System/Library/CoreServices/SystemVersion.plist`` (read by
    ``platform.mac_ver()`` in the huggingface_hub/timm import chain), to
    learn whether that sandbox trip is the sole blocker. The shipped
    ``_SYSTEM_READ_FILES`` is NOT modified.
    """

    import menagerie.crawler.worker_supervisor as supervisor_module

    monkeypatch.setattr(  # type: ignore[attr-defined]
        supervisor_module,
        "_SYSTEM_READ_FILES",
        frozenset(
            supervisor_module._SYSTEM_READ_FILES
            | {Path("/System/Library/CoreServices/SystemVersion.plist")}
        ),
    )

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
    timm_rows = [row for row in packages if row["name"] == "timm"]
    assert len(timm_rows) == 1, "core prefix must install exactly one timm"
    timm_version = timm_rows[0]["version"]
    print(f"\ncore prefix timm version: {timm_version}", flush=True)

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

    environment = real_environment_fixture.binding
    context = replace(
        context,
        environment_generations={
            **context.environment_generations,
            environment.family: environment.env_generation,
        },
    )

    with CanonicalReducer(paths.ledgers, context) as reducer:
        artifact = driver._stage_author_result(item, artifact, reducer)
        gate = FakeChecker().check_metadata([artifact], paths.work_root, driver.config).gate
        assert gate is not None
        reducer.append_gate(gate)
        attempts = SupervisedForwardLane(timeout_seconds=600, cwd=Path.cwd()).forward(
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
                json.dumps(attempt.get("error"))[:1500],
            )
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
                print(f"[{mode}] no worker result at {result_path}")
                continue
            wrapper = json.loads(result_path.read_text(encoding="utf-8"))
            diag = wrapper.get("diagnostic", {})
            if diag.get("error") is not None:
                print(f"[{mode}] diagnostic error:", json.dumps(diag["error"])[:3000])
            for mode_name, receipt in diag.get("per_mode", {}).items():
                print(
                    f"[{mode}] per_mode[{mode_name}]:",
                    "forward_completed=",
                    receipt.get("forward_completed"),
                    "params=",
                    receipt.get("parameter_count_total"),
                    "input_kind=",
                    receipt.get("input_kind"),
                    "input_asset=",
                    receipt.get("input_asset"),
                    "ctor_s=",
                    round(receipt.get("constructor_seconds") or -1, 2),
                    "fwd_s=",
                    round(receipt.get("forward_seconds") or -1, 2),
                    "output=",
                    json.dumps(receipt.get("output_signature"))[:220],
                )
                if receipt.get("error"):
                    print(f"[{mode}] mode error:", json.dumps(receipt["error"])[:3000])
        for attempt in attempts:
            print("FULL ATTEMPT:", json.dumps(attempt, default=str)[:8000])
        for diagnostic_path in sorted(tmp_path.rglob("diagnostics/*.json")):
            print("DIAGNOSTIC", diagnostic_path.name)
            print(diagnostic_path.read_text(encoding="utf-8")[:4000])
        assert all(attempt["result"] == "succeeded" for attempt in attempts), [
            (attempt["result"], attempt["error"]) for attempt in attempts
        ]

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

    print("status:", appended.record["status"]["code"])
    print("divergence:", appended.record["modes"]["train_eval_divergence"])
    assert appended.record["status"]["code"] == "runs"
    print("\nPROBE RESULT: timm dla60 R1 recipe reached a runs record")
