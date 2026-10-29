"""Stand-in detection: a substitute never publishes under the real model's name (decisions.md,
2026-10-03), whatever its label says.

A recipe is a stand-in, typed and refused, when

- its revision was typed ``standin:*`` for any model: a catalog row can share another row's
  recipe, revision and all, under its own id (m1778 ran m12019's typed dense recipe);
- it calls itself one: ``stand-in`` in the model's name or its delta.

A recipe whose adapter text is the adapter of a recipe typed ``standin:dense`` (under any file
name) but at another revision is flagged for a reader, not typed: one family adapter can build
the stand-in for one member and the library's own path for another (``f-schnet.py`` builds PyG's
SchNet on a hand-written radius graph and SchNetPack's SchNet on its own). Adapter source and
input notes are not searched for the word: import shims call themselves stand-ins too
(PlainMamba's mmcv glue), and a seeded input stands in for real data in every recipe.

``publish`` refuses every typed hit; ``sweep`` lists the hits not yet typed and the flags, and
with ``apply`` types the hits (a ledger row at the recipe's revision, and the record's give-up
when it holds none).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from menagerie.pipeline import DataLayout, now, read_json, write_json
from menagerie.pipeline.records import GiveUp, LedgerLine, LedgerRow, ModelRecord

SELF_DESCRIBED = re.compile(r"\bstand[- ]?ins?\b", re.IGNORECASE)
DENSE = "standin:dense"  # a self-described stand-in's class: a hand-written equivalent


@dataclass(frozen=True)
class Known:
    """What earlier typing marked: typed revisions and dense stand-in adapters' text digests."""

    revs: Mapping[str, tuple[str, str]]  # recipe revision -> (class, the model typed at it)
    adapters: Mapping[str, str]  # sha256 of a dense stand-in adapter's text -> its model


def _adapter_digest(model: ModelRecord, data: DataLayout, cache: dict[str, str | None]) -> str | None:
    """The sha256 of an adapter recipe's adapter file, read once per path; ``None`` without one."""

    recipe = model.recipe
    if recipe is None or recipe.kind != "adapter" or not recipe.adapter:
        return None
    if recipe.adapter not in cache:
        path = data.root / recipe.adapter
        cache[recipe.adapter] = (hashlib.sha256(path.read_bytes()).hexdigest()
                                 if path.is_file() else None)
    return cache[recipe.adapter]


def known(models: Iterable[ModelRecord], rows: Iterable[LedgerLine], *, data: DataLayout,
          cache: dict[str, str | None] | None = None) -> Known:
    """Every typed stand-in revision (ledger give-ups and records) and dense adapter digest."""

    cache = {} if cache is None else cache
    revs: dict[str, tuple[str, str]] = {}
    for row in rows:
        if isinstance(row, LedgerRow) and row.outcome == "giveup" and (
                row.cls or "").startswith("standin"):
            revs.setdefault(row.recipe_rev, (str(row.cls), row.id))
    models = [m for m in models if m.recipe is not None]
    for model in models:
        give = model.give_up
        if give is not None and give.cls.startswith("standin"):
            revs.setdefault(model.recipe.rev, (give.cls, model.id))  # type: ignore[union-attr]
    adapters: dict[str, str] = {}
    for model in models:
        typed = revs.get(model.recipe.rev)  # type: ignore[union-attr]
        if typed and typed[0] == DENSE and (digest := _adapter_digest(model, data, cache)):
            adapters.setdefault(digest, model.id)
    return Known(revs, adapters)


def _described(model: ModelRecord) -> str | None:
    texts = {"name": model.name, "delta": json.dumps(model.delta, sort_keys=True)}
    for where, text in texts.items():
        if found := SELF_DESCRIBED.search(text or ""):
            return f"its {where} calls it a {found[0].lower()}: {text[:160]!r}"
    return None


def detect(model: ModelRecord, found: Known) -> tuple[str, str] | None:
    """``(class, why)`` when the model's current recipe is a stand-in, else ``None``."""

    if model.recipe is None:
        return None
    rev = model.recipe.rev
    if rev in found.revs:
        cls, sid = found.revs[rev]
        return cls, f"recipe revision {rev} was typed {cls} for {sid}"
    why = _described(model)
    return (DENSE, why) if why else None


def flag(model: ModelRecord, found: Known, *, data: DataLayout,
         cache: dict[str, str | None] | None = None) -> str | None:
    """Why a reader should judge the recipe: it shares a dense stand-in's adapter text."""

    digest = _adapter_digest(model, data, {} if cache is None else cache)
    if digest is None or digest not in found.adapters or detect(model, found) is not None:
        return None
    return (f"its adapter {model.recipe.adapter} is the adapter of {found.adapters[digest]}'s "  # type: ignore[union-attr]
            "dense stand-in recipe; check which path this member builds")


def sweep(*, data: DataLayout, apply: bool = False) -> dict[str, list[dict[str, Any]]]:
    """``typed``: stand-ins not typed at their current revision yet (with ``apply``, each gets a
    give-up row at that revision and the record's give-up when it holds none); ``flagged``:
    recipes sharing a dense stand-in's adapter, for a reader."""

    from menagerie.pipeline import ledger, runner

    rows = ledger.read_all(data=data)
    models = [read_json(ModelRecord, p) for p in sorted(data.root.glob("models/*/*.json"))]
    cache: dict[str, str | None] = {}
    found = known(models, rows, data=data, cache=cache)
    done = {(r.id, r.recipe_rev) for r in rows if isinstance(r, LedgerRow)
            and r.outcome == "giveup" and (r.cls or "").startswith("standin")}
    typed, flagged, new_rows = [], [], []
    for model in models:
        if (why := flag(model, found, data=data, cache=cache)) is not None:
            flagged.append({"id": model.id, "rev": model.recipe.rev, "why": why})  # type: ignore[union-attr]
        hit = detect(model, found)
        own = model.give_up is not None and model.give_up.cls.startswith("standin")
        if hit is None or own or (model.id, model.recipe.rev) in done:  # type: ignore[union-attr]
            continue
        cls, why = hit
        recipe = model.recipe
        assert recipe is not None  # detect returns None without a recipe
        give = GiveUp(cls=cls, reason=f"stand-in: {why}", at=now(), by="script")
        typed.append({"id": model.id, "rev": recipe.rev, "class": cls, "why": why,
                      "record": model.give_up is None})
        if not apply:
            continue
        new_rows.append(LedgerRow(id=model.id, stage="forward", at=give.at,
                                  host=runner.host_label(), env=recipe.env, recipe_rev=recipe.rev,
                                  outcome="giveup", cls=cls, msg=give.reason[:300]))
        if model.give_up is None:
            write_json(replace(model, give_up=give), data.model(model.id))
    if new_rows:
        ledger.append(new_rows, data=data)
    return {"typed": typed, "flagged": flagged}
