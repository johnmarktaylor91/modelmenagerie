"""Class-routed operator notification policy.

The step-200 pilot silenced its notifier bluntly: ``MENAGERIE_NOTIFY_TRANSPORT``
pointed at ``/usr/bin/true``, so EVERY class of message -- circuit-open pages
that mean the campaign will never restart itself, completion, and routine
crash-restart chatter alike -- vanished identically. The opposite default (the
full firehose) is what forced that setting: recoverable restarts and progress
milestones paged a human for events the machinery already owned.

This module is the middle: every notification names a CLASS from a closed
enum, and only the ESSENTIAL classes -- the ones where silence loses the
campaign or the operator's one required action -- pass to the transport by
default. The rest are logged and settled. ``MENAGERIE_NOTIFY_FIREHOSE=1``
restores full delivery for operators who want everything.

Fail-open, never fail-silent: an event no mapping row names, or a call site
that supplies no class at all, DELIVERS. Suppression is earned by explicit
classification only, so a new alert added without a policy row pages until
someone deliberately classifies it.
"""

from __future__ import annotations

from enum import Enum
import os
from typing import Mapping, Optional, Union

__all__ = [
    "ESSENTIAL_NOTIFICATION_CLASSES",
    "FIREHOSE_ENV",
    "NotificationClass",
    "classify_notification_event",
    "should_deliver",
]

FIREHOSE_ENV = "MENAGERIE_NOTIFY_FIREHOSE"
"""Environment override restoring full-firehose delivery for every class."""

_FIREHOSE_TRUE = frozenset({"1", "true", "yes", "on"})


class NotificationClass(str, Enum):
    """Closed vocabulary of operator-notification classes."""

    #: A driver crash the supervisor itself restarts. The machinery owns the
    #: recovery; the page is chatter unless the crashes accumulate into a
    #: circuit, which is its own class below.
    CRASH_RESTART_RECOVERED = "crash-restart-recovered"
    #: A circuit opened (or armed a launch refusal): the campaign will not
    #: restart itself and is silently dead until an operator acts.
    CIRCUIT_OPEN = "circuit-open"
    #: A durable provider-usage pause whose wake episode owns the resume.
    USAGE_PAUSE = "usage-pause"
    #: Verified campaign completion.
    COMPLETION = "completion"
    #: A stall the machinery still owns (a queue job past its stall clock);
    #: watchers and the next round re-drive it.
    STALL = "stall"
    #: A stop nothing will ever re-drive without an operator: an indeterminate
    #: exit, an exhausted lock contention, a blocking review checkpoint, a
    #: dead operator lane.
    UNRECOVERABLE_STALL = "unrecoverable-stall"
    #: Routine progress milestones; the monitoring cadence owns these.
    PROGRESS = "progress"


ESSENTIAL_NOTIFICATION_CLASSES: frozenset[NotificationClass] = frozenset(
    {
        NotificationClass.CIRCUIT_OPEN,
        NotificationClass.COMPLETION,
        NotificationClass.UNRECOVERABLE_STALL,
    }
)
"""Classes that pass to the transport with no firehose override."""

#: Closed mapping from every existing alert-event identity to its class. An
#: event absent from this table classifies as ``None`` and DELIVERS (fail-open).
_EVENT_CLASSES: Mapping[str, NotificationClass] = {
    "driver-crash": NotificationClass.CRASH_RESTART_RECOVERED,
    "flat-restart-circuit-armed": NotificationClass.CIRCUIT_OPEN,
    "flat-restart-circuit-open": NotificationClass.CIRCUIT_OPEN,
    "crash-circuit-open": NotificationClass.CIRCUIT_OPEN,
    "usage-pause": NotificationClass.USAGE_PAUSE,
    # The campaign blocks until a human passes --after-review: silence here
    # is an unbounded hold, not a skipped courtesy.
    "review-pause": NotificationClass.UNRECOVERABLE_STALL,
    "review-checkpoint": NotificationClass.UNRECOVERABLE_STALL,
    "operator-outage": NotificationClass.UNRECOVERABLE_STALL,
    "campaign-complete": NotificationClass.COMPLETION,
    "indeterminate-stop": NotificationClass.UNRECOVERABLE_STALL,
    "lock-contention": NotificationClass.UNRECOVERABLE_STALL,
    "author-queue-stall": NotificationClass.STALL,
    "progress-milestone": NotificationClass.PROGRESS,
}


def classify_notification_event(event: str) -> Optional[NotificationClass]:
    """Return the closed class for one alert-event identity.

    Parameters
    ----------
    event:
        Alert event identity exactly as the supervisor or driver names it.

    Returns
    -------
    NotificationClass | None
        Closed class, or ``None`` for an event no policy row names -- which
        :func:`should_deliver` treats as deliverable, never as silent.
    """

    return _EVENT_CLASSES.get(event)


def should_deliver(
    notification_class: Union[NotificationClass, str, None],
    *,
    env: Mapping[str, str] = os.environ,
) -> bool:
    """Decide whether one classified notification passes to the transport.

    Parameters
    ----------
    notification_class:
        Closed class, its string value, or ``None`` for an unclassified
        notification.
    env:
        Environment carrying the optional firehose override.

    Returns
    -------
    bool
        True for every essential class, every unclassified notification
        (fail-open), every unknown class string (same rule: suppression is
        earned by explicit classification only), and everything under the
        firehose override; False only for a known, non-essential class.
    """

    if str(env.get(FIREHOSE_ENV, "")).strip().lower() in _FIREHOSE_TRUE:
        return True
    if notification_class is None:
        return True
    if isinstance(notification_class, NotificationClass):
        resolved = notification_class
    else:
        try:
            # NOT str(...): Python 3.11 str-mixin enums stringify to the
            # member repr, which would silently fail-open every enum input.
            resolved = NotificationClass(notification_class)
        except ValueError:
            return True
    return resolved in ESSENTIAL_NOTIFICATION_CLASSES
