"""Valve rows carry a REAL campaign identity, never a stringified absence.

Docket T2-12: all 62 pilot valve rows froze ``campaign_id="None"`` -- the
literal ``str(None)`` -- into their ``row_sha256``, because every emission site
spelled ``str(config.campaign_id)`` over an unbound ``Optional`` and the
non-empty tripwire accepts any non-empty string. The absence now refuses typed
at BOTH boundaries: the config resolver every emission site routes through,
and each row builder. Read-side validators stay tolerant, so the 62 historical
rows remain readable (their tolerance is disclosed in the docket, not here).
"""

from __future__ import annotations

import inspect

import pytest

from menagerie.crawler import driver as driver_module
from menagerie.crawler import driver_admission, driver_receipts
from menagerie.crawler.disk_floor import build_disk_floor_deferral_row
from menagerie.crawler.driver_contracts import (
    DriverConfig,
    DriverIntegrationError,
    bound_campaign_id,
)
from menagerie.crawler.environment_coverage import build_coverage_deferral_row
from menagerie.crawler.grant_parks import build_grant_park_row

NOW = "2026-08-12T08:00:00Z"


@pytest.mark.smoke
def test_bound_campaign_id_refuses_the_absence_and_passes_the_binding() -> None:
    """The resolver is the ONE spelling between config and durable rows."""

    assert bound_campaign_id(DriverConfig(campaign_id="c1-mech")) == "c1-mech"
    with pytest.raises(DriverIntegrationError, match="bound tier campaign_id"):
        bound_campaign_id(DriverConfig())


@pytest.mark.smoke
def test_no_emission_site_stringifies_the_campaign_binding() -> None:
    """The ``str(config.campaign_id)`` spelling is extinct in the producers.

    Source-binding pin: one surviving cast re-mints ``"None"`` rows the moment
    a config is unbound, which is exactly how the 62 pilot rows happened.
    """

    for module in (driver_module, driver_admission, driver_receipts):
        source = inspect.getsource(module)
        assert "str(self.config.campaign_id)" not in source, module.__name__
        assert "str(driver.config.campaign_id)" not in source, module.__name__


@pytest.mark.smoke
def test_valve_builders_refuse_the_stringified_absence() -> None:
    """Each builder refuses ``"None"`` at BUILD time, before any digest forms."""

    with pytest.raises(ValueError, match="stringified absence"):
        build_grant_park_row(
            stable_id="m_example",
            work_id="work-m_example",
            name="ExampleNet",
            campaign_id="None",
            run_id="run-test",
            machine_id="machine-test",
            created_at=NOW,
            blocked_parent_revision="sha256:" + "a" * 64,
            blocked_stage="accuracy-gate",
            pending_model={},
        )
    with pytest.raises(ValueError, match="stringified absence"):
        build_disk_floor_deferral_row(
            stable_id="m_example",
            work_id="work-m_example",
            name="ExampleNet",
            campaign_id="",
            run_id="run-test",
            machine_id="machine-test",
            created_at=NOW,
            intent="core",
            target="osx-arm64",
            env_root="/tmp/envs",
            floor_bytes=1,
            free_bytes=0,
        )
    with pytest.raises(ValueError, match="stringified absence"):
        build_coverage_deferral_row(
            stable_id="m_example",
            work_id="work-m_example",
            name="ExampleNet",
            campaign_id="None",
            run_id="run-test",
            machine_id="machine-test",
            created_at=NOW,
            assessment=None,
        )
