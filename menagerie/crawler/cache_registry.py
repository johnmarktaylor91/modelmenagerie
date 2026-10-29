"""W9 cache-LAYER registry: admission-steering caches consult IC-4's inventory.

The IC-4 closed cache inventory (``cache_inventory.json``) is enforced in CI by
the AST census; this module enforces it at RUN time. Every admission-steering
on-disk cache is consulted through a registry chokepoint that refuses when the
cache is not registered with a named invalidation authority -- an unregistered
cache is a startup refusal, not a silent new authority layer (R16: the work-dir
scratch cache survived ledger invalidation precisely because nothing bound its
lifecycle to a registered authority).

The registry is read-only over the packaged inventory file; registration
changes happen in ``cache_inventory.json`` under IC-4's census, never here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

__all__ = [
    "CacheRegistryError",
    "assert_admission_cache_registered",
    "author_result_cache_path",
    "startup_cache_registration_check",
]


class CacheRegistryError(RuntimeError):
    """An admission-steering cache was consulted outside the closed registry."""


_INVENTORY_PATH = Path(__file__).with_name("cache_inventory.json")

#: Registry rows by cache ``name``, loaded once per process. The inventory is
#: a packaged, reviewed file: reloading per consult would buy nothing but IO.
_REGISTRY: Optional[dict[str, Mapping[str, Any]]] = None

AUTHOR_RESULT_SCRATCH_CACHE = "work-dir-author-result-scratch-cache"


def _registry() -> dict[str, Mapping[str, Any]]:
    global _REGISTRY
    if _REGISTRY is None:
        data = json.loads(_INVENTORY_PATH.read_text(encoding="utf-8"))
        _REGISTRY = {str(row["name"]): row for row in data["caches"]}
    return _REGISTRY


def assert_admission_cache_registered(name: str) -> Mapping[str, Any]:
    """Return one registered cache row or refuse the consultation typed.

    Parameters
    ----------
    name:
        The inventory ``name`` of the cache about to be consulted.

    Raises
    ------
    CacheRegistryError
        If the cache is unregistered or names no invalidation authority.
    """

    row = _registry().get(name)
    if row is None:
        raise CacheRegistryError(
            f"admission-steering cache {name!r} is not registered in the IC-4 "
            "closed cache inventory (cache_inventory.json); register it with "
            "an invalidation authority before consulting it"
        )
    authority = row.get("invalidation_authority")
    if not isinstance(authority, str) or not authority.strip():
        raise CacheRegistryError(
            f"registered cache {name!r} names no invalidation authority"
        )
    return row


def author_result_cache_path(work_root: Path, stable_id: str) -> Path:
    """Return the C2 scratch-cache path through the registry chokepoint.

    Every reader and writer of ``driver-author-artifact.json`` routes here, so
    the cache cannot be consulted without its registration (and therefore its
    W9 invalidation authority) standing.
    """

    assert_admission_cache_registered(AUTHOR_RESULT_SCRATCH_CACHE)
    return work_root / stable_id / "driver-author-artifact.json"


def startup_cache_registration_check() -> tuple[str, ...]:
    """Verify every registered on-disk cache names its invalidation authority.

    Called before the first admission of a run; a violation refuses startup
    rather than letting an authority-less durable cache steer admissions.

    Returns
    -------
    tuple[str, ...]
        The sorted names of the verified on-disk cache layers.
    """

    verified: list[str] = []
    for name, row in _registry().items():
        if not str(row.get("kind", "")).startswith("on-disk"):
            continue
        authority = row.get("invalidation_authority")
        if not isinstance(authority, str) or not authority.strip():
            raise CacheRegistryError(
                f"on-disk cache {name!r} is registered without an invalidation "
                "authority; refusing startup (W9 cache-layer registry)"
            )
        verified.append(name)
    return tuple(sorted(verified))
