"""D-47 (OS-3+PM-4): digest-bound census acknowledgments.

200a launched with blanket ``--acknowledge-award-holds`` / ``--ack-stale-
excluded`` booleans frozen into every continuation: the operator said "yes"
once to whatever census any later leg would compute. The acknowledgment now
BINDS the exact census it acknowledges: the driver computes the census
digest at the door, the typed refusal and the census event both carry it
(the ceremony is paid once, legibly), and a launch acknowledging a DIFFERENT
census than the one standing refuses instead of proceeding.

One digest authority for both doors; the driver and every test/operator
ceremony compute through these two functions, never a re-spelling.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

from menagerie.crawler.identity import stable_hash

__all__ = [
    "award_holds_ack_digest",
    "is_ack_digest",
    "stale_excluded_ack_digest",
]

#: Exact shape of every census acknowledgment digest. The doors validate the
#: OPERATOR-SUPPLIED acknowledgment against this before comparing, so a legacy
#: free-form acknowledgment string ("yes", a blanket note, a pasted reason)
#: refuses with its own typed message instead of masquerading as a digest
#: mismatch against a census the operator never read.
_ACK_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def is_ack_digest(value: object) -> bool:
    """Return whether one supplied acknowledgment is a well-formed digest."""

    return isinstance(value, str) and _ACK_DIGEST_PATTERN.fullmatch(value) is not None


def award_holds_ack_digest(
    stale_awards: Iterable[str], award_park_row_shas: Iterable[str]
) -> str:
    """Digest of one award-holds census: exactly what the operator acknowledges.

    Parameters
    ----------
    stale_awards:
        Stable ids of projection-stale ``runs`` awards.
    award_park_row_shas:
        ``row_sha256`` of every live award-supersession park row (byte-exact:
        a re-minted park row is a NEW fact needing a new acknowledgment).
    """

    return stable_hash(
        {
            "census": "award-holds",
            "stale_awards": sorted(str(item) for item in stale_awards),
            "award_park_rows": sorted(str(item) for item in award_park_row_shas),
        }
    )


def stale_excluded_ack_digest(stale_members: Mapping[str, str]) -> str:
    """Digest of one stale-excluded census (the dependency-stale settled set).

    Binds CANONICAL census content only: member ids plus each member's CLOSED
    stale classification (:func:`menagerie.crawler.status.stale_axis`) --
    never the free-form projection reason string, which legitimately embeds
    revision hashes and live-derived identities and therefore flapped the
    digest across launches without any membership or reason-class change
    (the 2026-08-27 calibration-leg 26/31 flap). The digest is invariant
    under unrelated ledger appends and across consecutive derivations, and
    changes exactly when the stale MEMBERSHIP or a member's reason CLASS
    changes -- the D-47 point.

    Parameters
    ----------
    stale_members:
        Stable-ID keyed stale reasons. Values may be raw projection reason
        strings (classified here through the one closed classifier) or
        already-closed class names (bound verbatim); anything else a raw
        reason classifies to lands in its closed bucket, so no free-form
        content ever reaches the preimage. The legacy ids-only iterable
        ceremony refuses typed: an acknowledgment that binds no reason
        classes cannot notice a reason-class change.
    """

    if isinstance(stale_members, str) or not isinstance(stale_members, Mapping):
        raise TypeError(
            "stale_excluded_ack_digest binds member ids AND closed reason "
            "classes (D-47): pass the stable-ID -> stale-reason mapping "
            "(e.g. plan.stale_reasons), not a bare member list -- the "
            "ids-only ceremony is retired because it cannot notice a "
            "reason-class change"
        )
    from menagerie.crawler.status import (  # noqa: PLC0415 -- lazy: keeps census_ack import-light
        closed_stale_classes,
        stale_axis,
    )

    closed = closed_stale_classes()
    members = {
        str(stable_id): (
            str(reason) if str(reason) in closed else stale_axis(str(reason))
        )
        for stable_id, reason in stale_members.items()
    }
    return stable_hash(
        {
            "census": "stale-excluded",
            "schema": "member-ids-and-closed-reason-classes-v2",
            "members": members,
        }
    )
