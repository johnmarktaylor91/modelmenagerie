"""Class-routed notification policy: essential classes page, chatter settles.

The step-200 pilot silenced its transport bluntly (``MENAGERIE_NOTIFY_TRANSPORT``
= ``/usr/bin/true``), erasing circuit-open pages and completion alongside the
crash-restart chatter that motivated the silence. Pinned here:

* the class enum is closed and the essential subset is exactly circuit-open,
  completion, unrecoverable-stall;
* every existing alert-event identity classifies through the closed table;
* only essential classes pass to the transport by default; the firehose env
  override restores everything; unclassified NEVER suppresses (fail-open);
* the real ``CommandNotifier`` applies the policy BEFORE the transport --
  a suppressed class settles without spawning it, an essential class reaches
  it -- and the supervisor labels its alerts with the derived class.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.driver_progress import CommandNotifier
from menagerie.crawler.notification_policy import (
    ESSENTIAL_NOTIFICATION_CLASSES,
    FIREHOSE_ENV,
    NotificationClass,
    classify_notification_event,
    should_deliver,
)

pytestmark = pytest.mark.smoke


def test_the_class_enum_and_essential_subset_are_closed() -> None:
    assert {member.value for member in NotificationClass} == {
        "crash-restart-recovered",
        "circuit-open",
        "usage-pause",
        "completion",
        "stall",
        "unrecoverable-stall",
        "progress",
    }
    assert ESSENTIAL_NOTIFICATION_CLASSES == frozenset(
        {
            NotificationClass.CIRCUIT_OPEN,
            NotificationClass.COMPLETION,
            NotificationClass.UNRECOVERABLE_STALL,
        }
    )


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ("driver-crash", NotificationClass.CRASH_RESTART_RECOVERED),
        ("flat-restart-circuit-armed", NotificationClass.CIRCUIT_OPEN),
        ("flat-restart-circuit-open", NotificationClass.CIRCUIT_OPEN),
        ("crash-circuit-open", NotificationClass.CIRCUIT_OPEN),
        ("usage-pause", NotificationClass.USAGE_PAUSE),
        ("review-pause", NotificationClass.UNRECOVERABLE_STALL),
        ("review-checkpoint", NotificationClass.UNRECOVERABLE_STALL),
        ("operator-outage", NotificationClass.UNRECOVERABLE_STALL),
        ("campaign-complete", NotificationClass.COMPLETION),
        ("indeterminate-stop", NotificationClass.UNRECOVERABLE_STALL),
        ("lock-contention", NotificationClass.UNRECOVERABLE_STALL),
        ("author-queue-stall", NotificationClass.STALL),
        ("progress-milestone", NotificationClass.PROGRESS),
    ],
)
def test_every_existing_alert_event_classifies(event: str, expected: NotificationClass) -> None:
    assert classify_notification_event(event) is expected


def test_an_unknown_event_is_unclassified_and_delivers() -> None:
    """Suppression is earned by explicit classification only."""

    assert classify_notification_event("brand-new-alert") is None
    assert should_deliver(None, env={}) is True
    assert should_deliver("not-a-real-class", env={}) is True


def test_default_routing_passes_exactly_the_essential_classes() -> None:
    for member in NotificationClass:
        assert should_deliver(member, env={}) is (member in ESSENTIAL_NOTIFICATION_CLASSES)
        # The string spelling routes identically to the enum member.
        assert should_deliver(member.value, env={}) is (
            member in ESSENTIAL_NOTIFICATION_CLASSES
        )


def test_the_firehose_override_delivers_every_class() -> None:
    for member in NotificationClass:
        assert should_deliver(member, env={FIREHOSE_ENV: "1"}) is True
    assert should_deliver(NotificationClass.PROGRESS, env={FIREHOSE_ENV: "false"}) is False


def test_command_notifier_settles_suppressed_classes_before_the_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A suppressed class never spawns the transport and reports settled.

    The transport here is ``/usr/bin/false``: if the suppressed class reached
    it, notify would return False and the retry ladders would escalate a
    deliberately silenced page as a delivery failure.
    """

    monkeypatch.delenv(FIREHOSE_ENV, raising=False)
    notifier = CommandNotifier("/usr/bin/false", receipt_root=tmp_path)
    assert (
        notifier.notify(
            "routine crash chatter",
            idempotency_key="sha256:" + "a" * 64,
            notification_class=NotificationClass.CRASH_RESTART_RECOVERED.value,
        )
        is True
    )
    # An essential class DOES reach the transport: /usr/bin/false exits 1, so
    # the delivery honestly fails instead of being silently settled.
    assert (
        notifier.notify(
            "circuit opened",
            idempotency_key="sha256:" + "b" * 64,
            notification_class=NotificationClass.CIRCUIT_OPEN.value,
        )
        is False
    )
    # Unclassified attempts delivery exactly like before the policy existed.
    assert (
        notifier.notify(
            "unclassified alert",
            idempotency_key="sha256:" + "c" * 64,
        )
        is False
    )


def test_command_notifier_firehose_env_restores_full_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FIREHOSE_ENV, "1")
    notifier = CommandNotifier("/usr/bin/false", receipt_root=tmp_path)
    assert (
        notifier.notify(
            "routine crash chatter",
            idempotency_key="sha256:" + "d" * 64,
            notification_class=NotificationClass.CRASH_RESTART_RECOVERED.value,
        )
        is False
    ), "under the firehose the non-essential class reaches the transport"


def test_supervisor_labels_alerts_with_their_derived_class(tmp_path: Path) -> None:
    """The supervisor passes the classified event to the notification lane."""

    from menagerie.crawler.tests.test_slice_f_supervisor import (
        FakeNotifier,
        FakeProcessFactory,
        _config,
    )
    from menagerie.crawler.supervisor import EXIT_PAUSED, CrawlerSupervisor

    notifier = FakeNotifier()
    factory = FakeProcessFactory([EXIT_PAUSED])
    supervisor = CrawlerSupervisor(
        _config(tmp_path),
        notifier,
        process_factory=factory,
        sleep=lambda _seconds: None,
    )
    assert supervisor.run() == 0
    assert notifier.notification_classes == [NotificationClass.USAGE_PAUSE.value]
