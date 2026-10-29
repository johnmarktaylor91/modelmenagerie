"""Machine-specific default locations that stay out of the public code export.

A few tools and replay tests default to locations on the operator's own machines (a monitor's
state folder, a synced heartbeat file, frozen campaign archives). Those values live in the
optional private module ``menagerie._local_config`` (a ``DEFAULTS`` dict), which the public
export leaves out. ``MENAGERIE_<KEY>`` in the environment overrides both; without either, the
public fallback below each call site applies.
"""

from __future__ import annotations

import importlib
import os


def _private_defaults() -> dict[str, str]:
    """The private module's ``DEFAULTS``, or an empty dict where it is absent."""
    try:
        module = importlib.import_module("menagerie._local_config")
    except ModuleNotFoundError as err:
        if err.name != "menagerie._local_config":
            raise
        return {}
    return dict(getattr(module, "DEFAULTS", {}))


def local_value(key: str, fallback: str) -> str:
    """The configured location for ``key``: environment, then private default, then ``fallback``.

    Returned as written (``~`` unexpanded), so each caller keeps its own path handling.
    """
    return os.environ.get(f"MENAGERIE_{key}") or _private_defaults().get(key) or fallback
