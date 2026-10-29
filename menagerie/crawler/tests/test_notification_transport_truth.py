"""OB-M3 + D-38 (ack half): notify-transport truth and the critical-send outbox.

The 200a pilot muted every page identically (``MENAGERIE_NOTIFY_TRANSPORT`` =
``/usr/bin/true``), and the receipt shim wrote its receipt on exit 0, so the
doctor read the muted transport as healthy. These tests pin the fix surface:

* a closed no-op transport denylist that REFUSES to execute, typed dispositions;
* delivery receipts carrying ``transport_echoed_key`` -- transport stdout must
  contain the idempotency key, which an arg-ignoring no-op can never fake;
* the doctor requiring the echoed key, not just exit 0;
* critical-class sends durably recorded in the outbox: intent before transport,
  settled ONLY by an echoed receipt, refusals typed;
* the monitor rendering "notifications: DISCARDED" and alarming on a dead
  transport and on unechoed pending critical sends.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from menagerie.crawler.driver_progress import CommandNotifier
from menagerie.crawler.notification_outbox import (
    default_outbox_path,
    pending_critical_sends,
    read_outbox_rows,
    record_send_intent,
    record_send_settled,
)
from menagerie.crawler.notification_policy import NotificationClass
from menagerie.crawler.operator_notify import (
    EXIT_TRANSPORT_DISCARDING,
    NOOP_TRANSPORT_BASENAMES,
    deliver,
    evaluate_notification_transport,
    transport_disposition,
)
from menagerie.crawler.tools.launch_monitor import (
    MonitorState,
    Snapshot,
    Thresholds,
    evaluate_alerts,
    render_snapshot,
)

pytestmark = pytest.mark.smoke


def _script(path: Path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _snapshot(**overrides) -> Snapshot:
    base = dict(
        generated_at="2026-08-24T00:00:00+00:00",
        roster_label="step-200",
        roster_source="registry",
        roster_total=10,
        done=1,
        runs=1,
        failed=0,
        skipped=0,
        deferred=0,
        parked=0,
        in_flight=9,
        in_flight_started=1,
        block_reasons={},
        corrupt_lines=0,
        overall_rate_per_hour=None,
        rolling_rate_per_hour=None,
        eta_seconds=None,
        recent_error_rate=None,
        recent_error_window=0,
        quota_paused=False,
        disk_free_gb={},
        last_activity_age_minutes=1.0,
    )
    base.update(overrides)
    return Snapshot(**base)


class TestTransportDisposition:
    def test_closed_denylist_matrix(self) -> None:
        assert transport_disposition(None) == "absent"
        assert transport_disposition(()) == "absent"
        for base in sorted(NOOP_TRANSPORT_BASENAMES):
            assert transport_disposition((base,)) == "discarding"
            assert transport_disposition((f"/usr/bin/{base}",)) == "discarding"
            assert transport_disposition((f"/bin/{base}", "arg")) == "discarding"
        assert transport_disposition(("/bin/echo",)) == "live"
        assert transport_disposition(("send-to-jmt.sh",)) == "live"

    def test_usr_bin_true_is_denylisted(self) -> None:
        # The exact 200a mute.
        assert transport_disposition(("/usr/bin/true",)) == "discarding"

    def test_evaluate_arm_refuses_non_live(self, monkeypatch) -> None:
        monkeypatch.setenv("MENAGERIE_NOTIFY_TRANSPORT", "/usr/bin/true")
        verdict = evaluate_notification_transport()
        assert verdict["disposition"] == "discarding"
        assert verdict["ok"] is False
        assert "DISCARDED" in verdict["reason"]
        monkeypatch.setenv("MENAGERIE_NOTIFY_TRANSPORT", "/bin/echo")
        assert evaluate_notification_transport()["ok"] is True


class TestDeliverRefusalAndEcho:
    def test_deliver_refuses_noop_transport_without_executing(
        self, tmp_path: Path
    ) -> None:
        marker = tmp_path / "executed"
        transport = _script(tmp_path / "true", f"touch {marker}\nexit 0")
        # Basename "true" is denylisted even though the script would exit 0.
        receipt = tmp_path / "receipt.json"
        code = deliver(
            "critical page",
            transport=(transport,),
            receipt_path=receipt,
            nonce="nonce-1",
        )
        assert code == EXIT_TRANSPORT_DISCARDING
        assert not marker.exists()
        assert not receipt.exists()

    def test_echoing_transport_writes_echoed_receipt(self, tmp_path: Path) -> None:
        transport = _script(tmp_path / "echoer.sh", 'echo "$1"')
        receipt = tmp_path / "receipt.json"
        code = deliver(
            "page containing nonce-2",
            transport=(transport,),
            receipt_path=receipt,
            nonce="nonce-2",
        )
        assert code == 0
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        assert payload["nonce"] == "nonce-2"
        assert payload["transport_echoed_key"] is True
        assert "nonce-2" in payload["transport_stdout_head"]

    def test_silent_exit_zero_transport_writes_unechoed_receipt(
        self, tmp_path: Path
    ) -> None:
        transport = _script(tmp_path / "silent.sh", "exit 0")
        receipt = tmp_path / "receipt.json"
        assert (
            deliver(
                "page", transport=(transport,), receipt_path=receipt, nonce="nonce-3"
            )
            == 0
        )
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        assert payload["transport_echoed_key"] is False


class TestOutbox:
    def test_pending_projection_settles_only_on_echo(self, tmp_path: Path) -> None:
        outbox = tmp_path / "outbox.jsonl"
        record_send_intent(
            outbox, nonce="n1", notification_class="circuit-open", message="page one"
        )
        record_send_intent(
            outbox, nonce="n2", notification_class="completion", message="page two"
        )
        record_send_settled(outbox, nonce="n1", outcome="sent-unechoed")
        record_send_settled(outbox, nonce="n2", outcome="sent-echoed")
        pending = pending_critical_sends(read_outbox_rows(outbox))
        assert [send.nonce for send in pending] == ["n1"]
        assert pending[0].attempts == 1
        assert pending[0].last_outcome == "sent-unechoed"

    def test_unknown_outcome_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            record_send_settled(
                tmp_path / "outbox.jsonl", nonce="n", outcome="handled"
            )

    def test_crash_between_intent_and_settlement_reads_pending(
        self, tmp_path: Path
    ) -> None:
        outbox = tmp_path / "outbox.jsonl"
        record_send_intent(
            outbox, nonce="n1", notification_class="circuit-open", message="page"
        )
        pending = pending_critical_sends(read_outbox_rows(outbox))
        assert len(pending) == 1 and pending[0].attempts == 0

    def test_torn_final_line_is_tolerated(self, tmp_path: Path) -> None:
        outbox = tmp_path / "outbox.jsonl"
        record_send_intent(
            outbox, nonce="n1", notification_class="circuit-open", message="page"
        )
        with open(outbox, "ab") as handle:
            handle.write(b'{"kind": "send-settl')
        assert len(read_outbox_rows(outbox)) == 1


class TestCommandNotifierCriticalClasses:
    def test_critical_class_refuses_absent_transport(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from menagerie.crawler import operator_notify

        monkeypatch.setattr(operator_notify, "resolve_transport", lambda *a: None)
        notifier = CommandNotifier(
            None,
            receipt_root=tmp_path / "receipts",
            outbox_path=tmp_path / "outbox.jsonl",
        )
        assert notifier.command is None
        assert (
            notifier.notify(
                "circuit page",
                idempotency_key="crit-1",
                notification_class=NotificationClass.CIRCUIT_OPEN.value,
            )
            is False
        )
        rows = read_outbox_rows(tmp_path / "outbox.jsonl")
        assert [row["kind"] for row in rows] == ["send-intent", "send-settled"]
        assert rows[1]["outcome"] == "refused-transport-absent"

    def test_critical_send_settles_only_on_echo(self, tmp_path: Path) -> None:
        echoer = _script(
            tmp_path / "echoer.sh",
            'echo "$MENAGERIE_NOTIFICATION_IDEMPOTENCY_KEY"',
        )
        wrapped = (
            f"{os.sys.executable} -m menagerie.crawler.operator_notify"
            f" --transport {echoer}"
        )
        notifier = CommandNotifier(
            wrapped,
            receipt_root=tmp_path / "receipts",
            outbox_path=tmp_path / "outbox.jsonl",
        )
        assert (
            notifier.notify(
                "circuit page",
                idempotency_key="crit-2",
                notification_class=NotificationClass.CIRCUIT_OPEN.value,
            )
            is True
        )
        rows = read_outbox_rows(tmp_path / "outbox.jsonl")
        assert rows[-1]["outcome"] == "sent-echoed"
        assert not pending_critical_sends(rows)

    def test_critical_send_unechoed_stays_pending(self, tmp_path: Path) -> None:
        silent = _script(tmp_path / "silent.sh", "exit 0")
        wrapped = (
            f"{os.sys.executable} -m menagerie.crawler.operator_notify"
            f" --transport {silent}"
        )
        notifier = CommandNotifier(
            wrapped,
            receipt_root=tmp_path / "receipts",
            outbox_path=tmp_path / "outbox.jsonl",
        )
        assert (
            notifier.notify(
                "circuit page",
                idempotency_key="crit-3",
                notification_class=NotificationClass.CIRCUIT_OPEN.value,
            )
            is False
        )
        rows = read_outbox_rows(tmp_path / "outbox.jsonl")
        assert rows[-1]["outcome"] == "sent-unechoed"
        assert len(pending_critical_sends(rows)) == 1

    def test_critical_send_to_noop_transport_is_typed_refusal(
        self, tmp_path: Path
    ) -> None:
        wrapped = (
            f"{os.sys.executable} -m menagerie.crawler.operator_notify"
            f" --transport /usr/bin/true"
        )
        notifier = CommandNotifier(
            wrapped,
            receipt_root=tmp_path / "receipts",
            outbox_path=tmp_path / "outbox.jsonl",
        )
        assert (
            notifier.notify(
                "circuit page",
                idempotency_key="crit-4",
                notification_class=NotificationClass.CIRCUIT_OPEN.value,
            )
            is False
        )
        rows = read_outbox_rows(tmp_path / "outbox.jsonl")
        assert rows[-1]["outcome"] == "refused-transport-discarding"

    def test_non_essential_class_never_touches_the_outbox(
        self, tmp_path: Path
    ) -> None:
        notifier = CommandNotifier(
            None,
            receipt_root=tmp_path / "receipts",
            outbox_path=tmp_path / "outbox.jsonl",
        )
        assert (
            notifier.notify(
                "progress",
                idempotency_key="prog-1",
                notification_class=NotificationClass.PROGRESS.value,
            )
            is True
        )
        assert not (tmp_path / "outbox.jsonl").exists()

    def test_default_outbox_sits_beside_the_receipt_root(self, tmp_path: Path) -> None:
        assert default_outbox_path(tmp_path / "x" / "notification-receipts") == (
            tmp_path / "x" / "notification-outbox.jsonl"
        )


class TestMonitorTransportTruth:
    def test_discarding_transport_renders_discarded_and_alarms(self) -> None:
        snapshot = _snapshot(notify_transport_disposition="discarding")
        rendered = render_snapshot(snapshot)
        assert "notifications: DISCARDED" in rendered
        kinds = {kind for kind, _ in evaluate_alerts(snapshot, MonitorState(), Thresholds())}
        assert "notify-transport" in kinds

    def test_absent_transport_renders_and_alarms(self) -> None:
        snapshot = _snapshot(notify_transport_disposition="absent")
        assert "NO TRANSPORT INSTALLED" in render_snapshot(snapshot)
        kinds = {kind for kind, _ in evaluate_alerts(snapshot, MonitorState(), Thresholds())}
        assert "notify-transport" in kinds

    def test_pending_critical_sends_render_and_alarm(self) -> None:
        snapshot = _snapshot(
            notify_pending_critical=2,
            notify_oldest_pending_at="2026-08-24T00:00:00+00:00",
        )
        assert "2 critical" in render_snapshot(snapshot)
        kinds = {kind for kind, _ in evaluate_alerts(snapshot, MonitorState(), Thresholds())}
        assert "notify-pending" in kinds

    def test_live_transport_and_empty_outbox_stay_quiet(self) -> None:
        snapshot = _snapshot(notify_pending_critical=0)
        rendered = render_snapshot(snapshot)
        assert "notifications:" not in rendered
        kinds = {kind for kind, _ in evaluate_alerts(snapshot, MonitorState(), Thresholds())}
        assert "notify-transport" not in kinds and "notify-pending" not in kinds


class TestDoctorEchoRequirement:
    def test_doctor_requires_transport_echo(self, tmp_path: Path, monkeypatch) -> None:
        from menagerie.crawler import doctor as doctor_module

        silent = _script(tmp_path / "silent.sh", "exit 0")
        echoer = _script(
            tmp_path / "echoer.sh",
            'echo "$MENAGERIE_NOTIFICATION_IDEMPOTENCY_KEY"',
        )

        class _Config:
            # The probe launches the operator_notify shim with cwd=repo_root;
            # a relative PYTHONPATH must still resolve the package there.
            repo_root = Path.cwd()
            runtime_root = tmp_path / "runtime"

        probe = object.__new__(doctor_module.SystemDoctorProbes)
        probe.config = _Config()

        monkeypatch.setenv("MENAGERIE_NOTIFY_TRANSPORT", silent)
        assert probe.notifier_delivery() is False

        monkeypatch.setenv("MENAGERIE_NOTIFY_TRANSPORT", echoer)
        assert probe.notifier_delivery() is True

        monkeypatch.setenv("MENAGERIE_NOTIFY_TRANSPORT", "/usr/bin/true")
        assert probe.notifier_delivery() is False
