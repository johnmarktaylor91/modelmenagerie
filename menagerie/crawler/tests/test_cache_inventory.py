"""IC-4: the closed, machine-enforced cache census.

cache_inventory.json is the ONE inventory of every admission-steering cache
(the trilab union: 22 numbered caches + wave-state maps + lru sites), each row
naming its key axes, generation binding, and invalidation authority. W9's
invalidation design consumes those rows as its closed eviction scope.

The census here is the class tripwire: it AST-enumerates every
``functools.lru_cache``/``functools.cache`` decorator site and every
cache/memo-named binding (module level and ``self.`` attributes) across
``menagerie/crawler`` and requires EXACT set equality with the inventory's
declared sites plus its explicit non-cache exclusions. A new cache without a
registered invalidation authority -- or a silent removal -- fails CI until the
inventory is updated in the same reviewed change.

Disclosed boundary: an in-memory dict whose NAME avoids the cache/memo pattern
(e.g. reducer._final_artifact_transactions, C9) cannot be machine-discovered
by this census; such rows are declared in the inventory without sites and are
covered by W9's eviction tests instead.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

CRAWLER_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = CRAWLER_ROOT / "cache_inventory.json"

_NAME_HITS = ("cache", "memo")
_NAME_EXCLUDES = ("memory",)


def _pattern_hit(name: str) -> bool:
    low = name.lower()
    if any(exclude in low for exclude in _NAME_EXCLUDES):
        return False
    return any(hit in low for hit in _NAME_HITS)


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return f"{_decorator_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _assign_names(node: ast.stmt) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return list(node.targets)
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return [node.target]
    return []


def enumerate_cache_sites() -> tuple[set[str], set[str]]:
    """Return (lru decorator sites, cache/memo-named binding sites)."""

    lru_sites: set[str] = set()
    named_sites: set[str] = set()
    for path in sorted(CRAWLER_ROOT.rglob("*.py")):
        rel = path.relative_to(CRAWLER_ROOT.parents[1]).as_posix()
        if "/tests/" in rel:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    name = _decorator_name(decorator)
                    if "lru_cache" in name or name in {"cache", "functools.cache"}:
                        lru_sites.add(f"{rel}::{node.name}")
        # Module-level bindings: direct statements of the module body only.
        for statement in tree.body:
            for target in _assign_names(statement):
                if isinstance(target, ast.Name) and _pattern_hit(target.id):
                    named_sites.add(f"{rel}::{target.id}")
        # self-attribute bindings anywhere in the module.
        for node in ast.walk(tree):
            for target in _assign_names(node) if isinstance(node, ast.stmt) else []:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and _pattern_hit(target.attr)
                ):
                    named_sites.add(f"{rel}::self.{target.attr}")
    return lru_sites, named_sites


def _load_inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text())


def test_inventory_rows_are_complete_and_authority_bearing() -> None:
    """Every inventory row names non-empty key axes, generation binding, and
    an invalidation authority: an authority-less cache row fails CI."""

    inventory = _load_inventory()
    assert inventory["schema_version"] == "menagerie.crawler.cache-inventory.v1"
    seen_ids: set[str] = set()
    for row in inventory["caches"]:
        for field in ("id", "name", "kind", "location", "key_axes", "generation_binding"):
            assert isinstance(row.get(field), str) and row[field].strip(), (
                f"cache row {row.get('id')!r} lacks {field}"
            )
        authority = row.get("invalidation_authority")
        assert isinstance(authority, str) and authority.strip(), (
            f"cache row {row['id']} has NO invalidation authority -- register one "
            "or remove the cache"
        )
        assert row["id"] not in seen_ids, f"duplicate cache id {row['id']}"
        seen_ids.add(row["id"])
    for exclusion in inventory["non_cache_matches"]:
        assert exclusion.get("site") and exclusion.get("reason"), (
            "non-cache exclusions must name the site and the reason"
        )


def test_lru_census_matches_inventory_exactly() -> None:
    """Every lru_cache decorator site is a declared inventory site, and every
    declared lru site still exists (set equality, both directions)."""

    inventory = _load_inventory()
    lru_sites, _ = enumerate_cache_sites()
    declared = {
        site
        for row in inventory["caches"]
        if row["kind"] == "lru"
        for site in row["sites"]
    }
    assert lru_sites == declared, (
        "lru census drift -- register new caches with an invalidation authority, "
        f"or prune removed rows: new={sorted(lru_sites - declared)} "
        f"gone={sorted(declared - lru_sites)}"
    )


def test_named_binding_census_matches_inventory_exactly() -> None:
    """Every cache/memo-named module-level or self-attribute binding is either
    a declared cache site or an explicit, reasoned non-cache exclusion."""

    inventory = _load_inventory()
    _, named_sites = enumerate_cache_sites()
    declared = {
        site
        for row in inventory["caches"]
        if row["kind"] != "lru"
        for site in row["sites"]
    }
    excluded = {row["site"] for row in inventory["non_cache_matches"]}
    assert not (declared & excluded), "a site cannot be both a cache and an exclusion"
    assert named_sites == declared | excluded, (
        "cache-named binding census drift -- register the new cache (with an "
        "invalidation authority) or add a reasoned non-cache exclusion: "
        f"new={sorted(named_sites - declared - excluded)} "
        f"gone={sorted((declared | excluded) - named_sites)}"
    )


def test_w9_eviction_scope_rows_are_present() -> None:
    """The rows W9's invalidation design must evict (sol's in-memory maps and
    the work-dir scratch cache) stay present under their stable ids."""

    inventory = _load_inventory()
    ids = {row["id"] for row in inventory["caches"]}
    for required in ("C1", "C2", "C9", "C10", "C24", "C25"):
        assert required in ids, f"W9 eviction-scope row {required} missing from inventory"
