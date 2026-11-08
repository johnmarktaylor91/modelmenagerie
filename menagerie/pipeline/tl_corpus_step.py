"""The tick step ``torchlens_corpus``: refresh the TorchLens failure corpus at most daily, tell
TorchLens's Lead about new clusters only, and queue the corpus for a re-trace on a new pin.

Config (the tick's ``config.json``): ``"torchlens_corpus": {"enabled": true, "out": <folder>}``.
State (``tick`` state ``torchlens_corpus``): ``refreshed`` (UTC date), ``seen`` (every cluster id
ever published; the first refresh seeds it without a message), ``pin`` and ``queued`` (the ids
listed for a re-trace at that pin, so each is listed once per pin).

Re-trace on pin: when the corpus has rows whose failing trace ran at another TorchLens than the
pin, the step appends those not yet listed for this pin to the re-trace feed's ``pin.txt``
(``retrace``), which feeds them trace-only into runner B ahead of ``todo.txt``. Rows on the
worklist also re-trace through ``stage3_queue`` at a new pin (no trace at the pin is the next
step); the feed skips any already traced at the pin. No runner is restarted for this.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, tl_corpus
from menagerie.pipeline import tl_attribution as attrib

KEY = "torchlens_corpus"
NOTIFY = (attrib.TORCHLENS, attrib.UNATTRIBUTED)
MARGIN_S = 90  # left in the pass after the raise-site lookups, for the steps after this one
LISTED = 5


def config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    given = cfg.get(KEY) or {}
    if not isinstance(given, Mapping):
        raise ValueError(f'{KEY} must be an object such as {{"enabled": true, "out": "<folder>"}}')  # noqa: TRY004 -- config errors are ValueError, as the tick's others
    out = Path(given["out"]).expanduser() if given.get("out") else work_dir("torchlens-corpus")
    return {"enabled": given.get("enabled") is True, "out": out}


def new_clusters(groups: Sequence[Mapping[str, Any]], seen: set[str] | None
                 ) -> list[Mapping[str, Any]]:
    """Clusters TorchLens may own that no earlier refresh published (none on the first)."""

    if seen is None:
        return []
    return [g for g in groups if g["id"] not in seen and g["attribution"] in NOTIFY]


def fact(groups: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], out: Path) -> str:
    """The FACT for TorchLens's Lead about new clusters."""

    shown = "; ".join(f"{g['id']} {g['signature']} ({g['count']} models, e.g. "
                      f"{', '.join(g['representatives'][:2])}; {g['attribution']})"
                      for g in groups[:LISTED])
    more = f"; and {len(groups) - LISTED} more" if len(groups) > LISTED else ""
    n = summary["counts"]
    return (f"Menagerie TorchLens failure corpus: {len(groups)} new cluster(s): {shown}{more}. "
            f"Corpus now {n['models']} models in {n['clusters']} clusters "
            f"({n[attrib.TORCHLENS]} attributed to TorchLens): {out / 'clusters.md'} "
            "(reproduce commands inside; every row in corpus.jsonl).")


def stale(rows: Sequence[Mapping[str, Any]], pin: str, listed: Sequence[str]) -> list[str]:
    """Corpus ids not yet listed for ``pin`` whose failing trace ran at another pin, or that never
    traced at their recipe (a typed TorchLens give-up)."""

    done = set(listed)
    return [r["id"] for r in rows if r.get("tl") != pin
            and r["attribution"] != attrib.MENAGERIE and r["id"] not in done]


def list_for_retrace(folder: Path, ids: Sequence[str]) -> list[str]:
    """Append ``ids`` not already in the feed's ``pin.txt``; returns those appended."""

    from menagerie.pipeline import retrace

    path = folder / retrace.PIN_LIST
    have = set(retrace._ids(path))  # the feed's own reader
    new = [i for i in dict.fromkeys(ids) if i not in have]
    if new:
        folder.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.writelines(f"{i}\n" for i in new)
    return new


def step(data: DataLayout, cfg: Mapping[str, Any], state: dict[str, Any], *, now: datetime,
         dry_run: bool, deadline: float | None = None, pin: str | None = None,
         build: Any = None) -> tuple[dict[str, Any], list[str]]:
    """One tick pass's refresh: (result, FACT texts for TorchLens's Lead)."""

    from menagerie.pipeline import retrace
    from menagerie.pipeline.publish import torchlens_pin

    conf = config(cfg)
    if not conf["enabled"]:
        return {"skipped": "off"}, []
    mine = dict(state.get(KEY) or {})
    pin = pin or torchlens_pin()
    today = now.date().isoformat()
    if mine.get("refreshed") == today and mine.get("pin") == pin:
        return {"fresh": today}, []
    if dry_run:
        return {"would_refresh": str(conf["out"])}, []
    index = attrib.SourceIndex(attrib.source_root(), work_dir("torchlens-corpus-cache"),
                               deadline=None if deadline is None else deadline - MARGIN_S)
    try:
        built = (build or tl_corpus.build)(data=data, out=conf["out"], index=index, now=now)
    except TimeoutError as exc:
        index.save()
        return {"held": f"{exc}; it continues next pass"}, []
    groups, seen = built["clusters"], mine.get("seen")
    new = new_clusters(groups, set(seen) if seen is not None else None)
    queued = list((mine.get("queued") or {}).get(pin, []))
    due = stale(built["rows"], pin, queued)
    added = list_for_retrace(retrace.config(cfg)["folder"], due)
    state[KEY] = {"refreshed": today, "pin": pin,
                  "seen": sorted(set(seen or ()) | {g["id"] for g in groups}),
                  "queued": {pin: queued + due}}
    result = {"models": built["counts"]["models"], "clusters": len(groups),
              "new": [g["id"] for g in new], "listed_for_retrace": len(added),
              "seeded": seen is None, "out": str(conf["out"])}
    return result, [fact(new, built, conf["out"])] if new else []
