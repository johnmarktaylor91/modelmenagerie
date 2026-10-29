"""Typed wake-continuation contract: ONE flag vocabulary, three consumers (W15).

The wake/resume contract was a hand-enumerated partial COPY in four places --
the driver's callback argv builder, the supervisor's run parser, the
supervisor's verbatim forwarding list, and the crawler CLI -- and each carried
a different subset (caaeb362 patched one spelling; the supervisor branch still
dropped the environment-readiness receipt, so the first real supervised pause
of a production campaign resumed into the readiness refusal: three flat
restarts, circuit open, rung dead). This module is the single vocabulary:

* :data:`CONTINUATION_FLAG_SPECS` names every continuation flag once, with its
  ``DriverConfig`` attribute and rendering kind;
* :func:`render_continuation_flags` renders the COMPLETE contract from a
  config (both callback branches consume it, so the supervisor branch can no
  longer diverge from the direct one);
* :func:`forwarded_continuation_argv` re-emits the contract verbatim from the
  supervisor's parsed namespace (typed and forwarded, never interpreted).

``parse(render(config)) == config`` over this vocabulary is pinned by test;
adding a continuation-relevant flag ANYWHERE without adding it here fails the
closure test rather than shipping another partial copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from menagerie.crawler.work_selection import selection_continuation_identity


@dataclass(frozen=True)
class ContinuationFlagSpec:
    """One continuation flag: its spelling, config attribute, and kind."""

    flag: str
    #: ``DriverConfig`` attribute (and, underscored, the argparse dest).
    attribute: str
    #: ``value`` (emit str when set), ``value-always`` (emit str always),
    #: ``bool`` (emit bare flag when true), ``csv`` (join a tuple).
    kind: str


#: The closed continuation vocabulary, in canonical render order. The
#: redrive triple renders under the D-54 rule (manifest preferred; the inline
#: CSV and its reason only without one); everything else renders per kind.
CONTINUATION_FLAG_SPECS: tuple[ContinuationFlagSpec, ...] = (
    ContinuationFlagSpec("--scheduler", "scheduler", "value-always"),
    ContinuationFlagSpec("--phase", "phase", "value"),
    ContinuationFlagSpec("--only-unsettled", "only_unsettled", "bool"),
    ContinuationFlagSpec("--redrive-manifest", "redrive_manifest_path", "value"),
    ContinuationFlagSpec("--redrive-targets", "redrive_targets", "csv"),
    ContinuationFlagSpec("--redrive-reason", "redrive_reason", "value"),
    ContinuationFlagSpec("--gate-catchup-targets", "gate_catchup_targets", "csv"),
    # Shape 4: the declared closed leg purpose is part of the launching
    # config -- a wake resuming a licensing_close generation must replay the
    # declaration (the leg-open row refuses a mismatch either way; this keeps
    # the callback honest instead of relying on the refusal).
    ContinuationFlagSpec("--leg-purpose", "leg_purpose", "value"),
    ContinuationFlagSpec("--ack-stale-excluded", "ack_stale_excluded", "bool"),
    ContinuationFlagSpec("--acknowledge-award-holds", "acknowledge_award_holds", "bool"),
    # D-47: the digest companions ride the continuation so a resumed leg
    # replays the SAME bound acknowledgment (and refuses a drifted census).
    ContinuationFlagSpec("--award-holds-digest", "award_holds_digest", "value"),
    ContinuationFlagSpec("--stale-excluded-digest", "stale_excluded_digest", "value"),
    ContinuationFlagSpec("--author-concurrency", "author_concurrency", "value-always"),
    ContinuationFlagSpec("--checker-concurrency", "checker_concurrency", "value-always"),
    # Computed, never read back off the config attribute (the resumed leg
    # re-derives and compares it -- WS-1 clause 7).
    ContinuationFlagSpec("--continuation-identity", "", "computed-identity"),
    # The environment-readiness receipt IS part of the continuation contract:
    # a production leg cannot pass the CLI readiness gate without it, and a
    # callback branch missing it strands every wake at the door (the
    # adjudicated D1 supervisor-branch defect).
    ContinuationFlagSpec(
        "--environment-readiness-receipt", "environment_readiness_receipt_path", "value"
    ),
)


def render_continuation_flags(config: Any) -> tuple[str, ...]:
    """Render the COMPLETE continuation contract from one driver config."""

    flags: List[str] = []
    manifest_path = getattr(config, "redrive_manifest_path", None)
    for spec in CONTINUATION_FLAG_SPECS:
        if spec.kind == "computed-identity":
            flags.extend((spec.flag, selection_continuation_identity(config)))
            continue
        value = getattr(config, spec.attribute, None)
        if spec.attribute == "redrive_targets" and manifest_path is not None:
            # D-54: an over-cap list is only legal via the digested manifest;
            # replaying the expanded inline CSV would refuse at the resumed
            # leg's own guard.
            continue
        if spec.attribute == "redrive_reason" and manifest_path is not None:
            # The manifest carries its own reason.
            continue
        if spec.kind == "value-always":
            flags.extend((spec.flag, str(value)))
        elif spec.kind == "value":
            if value is not None:
                flags.extend((spec.flag, str(value)))
        elif spec.kind == "bool":
            if value:
                flags.append(spec.flag)
        elif spec.kind == "csv":
            if value:
                flags.extend((spec.flag, ",".join(value)))
        else:  # pragma: no cover - closed kind vocabulary
            raise ValueError(f"unknown continuation flag kind {spec.kind!r}")
    return tuple(flags)


def forwarded_continuation_argv(namespace: Any) -> tuple[str, ...]:
    """Re-emit the contract verbatim from a supervisor-parsed namespace.

    The supervisor types and forwards; it never interprets (its values are
    plain strings, so ``value-always`` and ``value`` collapse together and
    the redrive preference was already applied by the renderer upstream).
    """

    argv: List[str] = []
    for spec in CONTINUATION_FLAG_SPECS:
        # The argparse dest is derived from the SPELLING, so the supervisor
        # namespace and this forwarder can never disagree about a name.
        dest = spec.flag.lstrip("-").replace("-", "_")
        value = getattr(namespace, dest, None)
        if spec.kind == "bool":
            if value:
                argv.append(spec.flag)
        elif value is not None and value != ():
            argv.extend((spec.flag, str(value)))
    return tuple(argv)


def supervisor_run_flag_names() -> tuple[str, ...]:
    """Return every flag spelling the supervisor's run parser must accept."""

    return tuple(spec.flag for spec in CONTINUATION_FLAG_SPECS)
