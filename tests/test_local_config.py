"""menagerie.local_config: environment, then the private module, then the public fallback."""

from __future__ import annotations

import sys
import types

import pytest

from menagerie import local_config


@pytest.fixture
def private(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    defaults: dict[str, str] = {}
    module = types.ModuleType("menagerie._local_config")
    module.DEFAULTS = defaults  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "menagerie._local_config", module)
    return defaults


def test_fallback_without_private_value(private: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MENAGERIE_SOME_KEY", raising=False)
    assert local_config.local_value("SOME_KEY", "~/fallback") == "~/fallback"


def test_private_value_beats_fallback(private: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MENAGERIE_SOME_KEY", raising=False)
    private["SOME_KEY"] = "~/private"
    assert local_config.local_value("SOME_KEY", "~/fallback") == "~/private"


def test_environment_beats_private_value(private: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    private["SOME_KEY"] = "~/private"
    monkeypatch.setenv("MENAGERIE_SOME_KEY", "/env/value")
    assert local_config.local_value("SOME_KEY", "~/fallback") == "/env/value"


def test_absent_private_module_gives_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "menagerie._local_config", None)  # import raises ModuleNotFoundError
    monkeypatch.delenv("MENAGERIE_SOME_KEY", raising=False)
    assert local_config.local_value("SOME_KEY", "~/fallback") == "~/fallback"
