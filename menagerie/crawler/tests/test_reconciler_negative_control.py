"""D-1/DX-75/B2-X-V12: the constant-accept reconciler class is dead in code.

200a ran its whole rung with a constant-accept provider adapter wired
(ambiguity-as-spent), and the receipt-backed evidence adapter that replaced it
was observationally identical on every ledger row (1547/1547 ``accepted:
true``), so NO ledger evidence could falsify which reconciler answered. Three
independent code defenses land here, each pinned in this file:

1. the exact retired stub spelling refuses at ``CommandProviderReconciler``
   construction (the stale persisted campaign-config / launchd resurrection
   vector fires typed instead of silently);
2. the canary-nonce negative control refuses ANY adapter that answers a
   well-formed key no reservation names -- the behavioral kill for impostors
   under any filename, standing at every admission-boundary reconcile;
3. the adapter status contract requires a digest-covered ``adapter_identity``
   stamp, so every future settlement receipt proves WHICH adapter answered
   and every legacy three-field adapter refuses typed.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.driver_admission import AdmissionEnvironmentMixin
from menagerie.crawler.paid_session_lifecycle import (
    BudgetPolicy,
    CommandProviderReconciler,
    PaidSessionAuthority,
    PaidSessionError,
    ProviderReconcileRefusal,
    ReconcilerNegativeControlError,
    verify_reconciler_negative_control,
)

pytestmark = pytest.mark.smoke

#: The retired launch-night stub's answering body, VERBATIM (2026-08-19,
#: crawler-launch-sprint/provider_reconcile_accept.py). The negative control
#: must go red against exactly this behavior whatever the file is called.
RETIRED_STUB_BODY = textwrap.dedent(
    """\
    import json
    import sys

    if len(sys.argv) < 2 or not sys.argv[-1]:
        print("usage: provider_reconcile_accept.py <idempotency-key>", file=sys.stderr)
        raise SystemExit(2)

    print(json.dumps({
        "idempotency_key": sys.argv[-1],
        "accepted": True,
        "observed_wall_seconds": None,
    }))
    """
)


def _adapter(tmp_path: Path, body: str, name: str = "adapter.py") -> CommandProviderReconciler:
    """Return a command reconciler over one scripted adapter body."""

    script = tmp_path / name
    script.write_text(body, encoding="utf-8")
    return CommandProviderReconciler((sys.executable, str(script)))


def _scope(stable_id: str = "m_one") -> dict[str, Any]:
    """Return one author reservation scope carrying both accounting axes."""

    return {
        "campaign_id": "c1-mech",
        "generation_id": "generation-open",
        "lane": "author",
        "dispatch_kind": "author_session",
        "work_generations": ((stable_id, f"work-{stable_id}"),),
        "gate_kind": None,
        "checker_identity": None,
        "run_id": "run-one",
        "machine_id": "host-one",
    }


def test_the_exact_launch_stub_body_dies_at_the_contract_whatever_its_name(
    tmp_path: Path,
) -> None:
    """The retired stub body, under any filename, refuses typed at the contract.

    The exact launch-night body answers the old three-field contract, so it
    cannot even parse under the identity-bearing contract -- its kill is the
    typed field-contract refusal, before the canary is needed.
    """

    reconciler = _adapter(tmp_path, RETIRED_STUB_BODY, name="innocuous_name.py")
    with pytest.raises(PaidSessionError, match="invalid field contract"):
        verify_reconciler_negative_control(reconciler)


def test_a_modern_constant_accept_impostor_fails_the_negative_control(
    tmp_path: Path,
) -> None:
    """A contract-fluent constant-accept (identity and all) dies at the canary.

    The behavioral kill: an adapter that upgrades to the new field set but
    still answers every key is exactly the class the standing probe exists to
    falsify -- no filename, contract, or identity claim can save it.
    """

    body = RETIRED_STUB_BODY.replace(
        '"observed_wall_seconds": None,',
        '"observed_wall_seconds": None,\n    "adapter_identity": "impostor/1",',
    )
    reconciler = _adapter(tmp_path, body, name="innocuous_name.py")
    with pytest.raises(ReconcilerNegativeControlError, match="canary nonce"):
        verify_reconciler_negative_control(reconciler)


def test_a_constant_decline_also_fails_the_negative_control(tmp_path: Path) -> None:
    """Answering ``accepted: false`` about an unknowable key is equally dead.

    A blanket decline is the money-NEGATIVE failure mode (it zeroes charges
    the host cannot see); the control refuses any ANSWER, not just accepts.
    """

    body = RETIRED_STUB_BODY.replace('"accepted": True', '"accepted": False').replace(
        '"observed_wall_seconds": None,',
        '"observed_wall_seconds": None,\n    "adapter_identity": "constant-decline/1",',
    )
    reconciler = _adapter(tmp_path, body)
    with pytest.raises(ReconcilerNegativeControlError):
        verify_reconciler_negative_control(reconciler)


def test_the_retired_stub_spelling_refuses_at_construction(tmp_path: Path) -> None:
    """A legacy config naming provider_reconcile_accept.py cannot even wire."""

    script = tmp_path / "provider_reconcile_accept.py"
    script.write_text(RETIRED_STUB_BODY, encoding="utf-8")
    with pytest.raises(ValueError, match="retired constant-accept"):
        CommandProviderReconciler(("/usr/bin/python3", str(script)))


def test_a_legacy_three_field_adapter_refuses_typed_at_the_contract(tmp_path: Path) -> None:
    """An identity-less answer is a typed contract refusal, not a pass."""

    reconciler = _adapter(tmp_path, RETIRED_STUB_BODY)
    with pytest.raises(PaidSessionError, match="invalid field contract"):
        reconciler("sha256:" + "ab" * 32)


def test_an_adapter_outage_propagates_typed_and_is_not_a_pass(tmp_path: Path) -> None:
    """A query that cannot complete raises typed; silence never reads as held."""

    reconciler = CommandProviderReconciler((str(tmp_path / "does-not-exist"),))
    with pytest.raises(PaidSessionError, match="did not complete"):
        verify_reconciler_negative_control(reconciler)


def test_the_evidence_adapter_holds_the_negative_control(tmp_path: Path) -> None:
    """The shipped evidence adapter refuses the canary against a real ledger."""

    authority = PaidSessionAuthority(
        tmp_path / "records" / "operational" / "spend.jsonl",
        runtime_root=tmp_path / ".crawl-local",
        clock=lambda: "2026-08-24T12:00:00Z",
    )
    authority.reserve(
        scope=_scope(),
        policy=BudgetPolicy(
            quota_sessions=100,
            quota_wall_seconds=100_000.0,
            runaway_sessions=100,
            runaway_wall_seconds=100_000.0,
            reservation_wall_seconds=100.0,
        ),
    )
    reconciler = CommandProviderReconciler(
        (
            sys.executable,
            "-m",
            "menagerie.crawler.tools.provider_reconcile_evidence",
            "--records-root",
            str(tmp_path / "records"),
        )
    )
    nonce = verify_reconciler_negative_control(reconciler)
    assert nonce.startswith("sha256:")


def test_the_evidence_adapter_stamps_its_identity_into_the_receipt(tmp_path: Path) -> None:
    """A real answer carries the adapter identity, digest-covered."""

    from menagerie.crawler.tools.provider_reconcile_evidence import ADAPTER_IDENTITY

    authority = PaidSessionAuthority(
        tmp_path / "records" / "operational" / "spend.jsonl",
        runtime_root=tmp_path / ".crawl-local",
        clock=lambda: "2026-08-24T12:00:00Z",
    )
    reservation = authority.reserve(
        scope=_scope(),
        policy=BudgetPolicy(
            quota_sessions=100,
            quota_wall_seconds=100_000.0,
            runaway_sessions=100,
            runaway_wall_seconds=100_000.0,
            reservation_wall_seconds=100.0,
        ),
    )
    reconciler = CommandProviderReconciler(
        (
            sys.executable,
            "-m",
            "menagerie.crawler.tools.provider_reconcile_evidence",
            "--records-root",
            str(tmp_path / "records"),
        )
    )
    receipt = reconciler(reservation.provider_idempotency_key)
    assert receipt.adapter_identity == ADAPTER_IDENTITY


def test_receipt_digests_differ_across_adapter_identities(tmp_path: Path) -> None:
    """Identity is inside the digested preimage, not decorative."""

    key = "sha256:" + "cd" * 32
    body = (
        "import json, sys\n"
        "print(json.dumps({'idempotency_key': sys.argv[-1], 'accepted': False,\n"
        "                  'observed_wall_seconds': None,\n"
        "                  'adapter_identity': 'IDENTITY'}))\n"
    )
    first = _adapter(tmp_path, body.replace("IDENTITY", "adapter-a/1"), name="a.py")(key)
    second = _adapter(tmp_path, body.replace("IDENTITY", "adapter-b/1"), name="b.py")(key)
    assert first.receipt_sha256 != second.receipt_sha256


def test_an_honest_refusal_holds_the_control(tmp_path: Path) -> None:
    """A nonzero-exit unknown-key refusal is the one passing behavior."""

    body = "import sys\nprint('unknown key', file=sys.stderr)\nraise SystemExit(3)\n"
    reconciler = _adapter(tmp_path, body)
    with pytest.raises(ProviderReconcileRefusal):
        reconciler("sha256:" + "ef" * 32)
    assert verify_reconciler_negative_control(reconciler).startswith("sha256:")


def test_the_standing_probe_is_wired_at_the_admission_boundary() -> None:
    """The admission-boundary reconcile runs the negative control (wiring pin)."""

    source = textwrap.dedent(
        inspect.getsource(
            AdmissionEnvironmentMixin._reconcile_paid_sessions_before_admission
        )
    )
    loaded = {
        node.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert {
        "verify_reconciler_negative_control",
        "CommandProviderReconciler",
        "ReconcilerNegativeControlError",
    } <= loaded
