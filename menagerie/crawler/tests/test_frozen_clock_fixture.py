"""Self-tests for the shared frozen-clock fixture (T5-5 shared infrastructure).

Fixture self-tests are part of the RED set: a fake that silently drifts or
formats wrongly would invalidate every timing test built on it.
"""

from __future__ import annotations

import pytest

from menagerie.crawler.tests.conftest import NOW, FrozenClock

#: Sub-second load-bearing coverage in the mandatory smoke tier (T5-7).
pytestmark = pytest.mark.smoke


def test_frozen_clock_starts_at_the_shared_instant(frozen_clock: FrozenClock) -> None:
    """The fixture starts frozen at the conftest `NOW` and stays put.

    Parameters
    ----------
    frozen_clock:
        Shared frozen-clock fixture.
    """

    assert frozen_clock() == NOW
    assert frozen_clock() == NOW


def test_frozen_clock_advances_exactly_and_formats_driver_strings() -> None:
    """Advancing moves the instant precisely and keeps the ISO-8601 Z shape."""

    clock = FrozenClock("2026-07-14T12:00:00Z")
    clock.advance(90)
    assert clock() == "2026-07-14T12:01:30Z"
    clock.advance(0)
    assert clock() == "2026-07-14T12:01:30Z"
    with pytest.raises(ValueError):
        clock.advance(-1)


def test_frozen_clock_is_accepted_as_a_driver_clock() -> None:
    """The callable satisfies the `DriverDependencies.clock` contract shape."""

    clock = FrozenClock()
    value = clock()
    assert isinstance(value, str)
    assert value.endswith("Z")
