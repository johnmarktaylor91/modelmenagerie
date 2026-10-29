"""Stubs for the speed-up build's verbs (BUILD-PLAN section 1, item 1) until their lanes fill them.

A stub raises ``NotBuilt``; ``cli.main`` prints its message and returns ``NOT_BUILT``, so a
caller sees "not built yet" and a non-zero status. A lane that fills a stub replaces its body and
adds the verb to ``FILLED_SP`` in ``tests/pipeline/test_records.py`` in the same commit.
"""

from __future__ import annotations

from typing import NoReturn

NOT_BUILT = 3  # the exit status of a verb whose body is not built yet


class NotBuilt(NotImplementedError):
    """A verb whose speed-up lane has not built its body yet."""


def not_built(verb: str, lane: str) -> NoReturn:
    """Refuse ``verb``: name it and the lane that builds it."""

    raise NotBuilt(f"{verb}: not built yet (speed-up lane {lane})")
