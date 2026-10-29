"""``describe write`` and ``describe check``: describe records from saved text (lane r3-describe).

Runs on the hub. ``write`` takes the validated families without a record, oldest packet first:
a family with no saved text at all is typed ``describe:no_text`` at once; the rest are laid out
ten to a batch (``describe_text.gather``), the enabled vocabulary fields are filled by jev
(``describe_jev.classify``) and one ``model-judgment`` lane writes the prose, taxonomy and
lineage under ``prompts/describe.md``. When the lane ends, each draft becomes a ``DescribeRecord``
whose lineage targets are resolved against roster names and packet identities, and ``verify``
checks it: every quote verbatim in its source after normalization, every summary sentence
grounded, vocabulary values in their lists, lineage targets resolving. A failing family goes back
to one return lane with its findings; still failing, it is written ``describe:ungrounded``.
``write`` is idempotent like the review verbs: call it until every family is settled. The tick
calls it through ``describe_stage`` with the stage's knobs in the environment: the seat cap
(``MENAGERIE_LANES_DESCRIBE``), hub slots a seat leaves free (``HUB_RESERVE``) and whether
families validated only at an earlier TorchLens pin follow (``EARLIER_PINS``) and which families
may ground on the cited paper's saved abstract when they have no full text (``ABSTRACTS``). A
record names the strongest kind of text it read (``basis``); a stronger kind it did not read, landing
later in the packet, makes the family due again. With the stage's worker switch
(``describe_workers.WORKERS``) a seat runs on a worker runtime when one has room after 2b, with no
hub-slot gate; the hub, behind its gate, is the fallback. ``lane.json`` then names the worker and
lease, and the lane has ended once its outputs are ingested (``describe_workers.landed``).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import (
    DataLayout,
    StoreLayout,
    describe_jev,
    describe_text,
    describe_workers,
    ledger,
    read_json,
    review,
    roster,
    tick_full,
    tree_sha256,
    write_json,
)
from menagerie.pipeline import drive as drive_mod
from menagerie.pipeline.describe import _norm
from menagerie.pipeline.records import DescribeRecord, RecordError, from_dict, to_dict, vocab

ROLE = "model-judgment"
KIND = "describe"
PROMPT = Path(__file__).resolve().parents[2] / "prompts" / "describe.md"
BATCH = 10
DEFAULT_SEATS = 4  # trial D passed (decisions.md 2026-10-16); MENAGERIE_LANES_DESCRIBE=1 reverts
HUB_RESERVE = "MENAGERIE_DESCRIBE_HUB_RESERVE"  # hub slots a new seat leaves free (default none)
EARLIER_PINS = "MENAGERIE_DESCRIBE_EARLIER_PINS"  # "1": earlier-pin families follow (``due``)
# "all", or comma-separated family ids (a trial, described first): ground on a saved abstract.
ABSTRACTS = "MENAGERIE_DESCRIBE_ABSTRACTS"
BASIS = {"full": "full", "abstract": "abstract", "docs": "docs", "readme-only": "readme",
         "none": "none"}
SUMMARY_CHARS = (250, 4000)
QUOTE_WORDS = (3, 80)  # a grounding quote: at least three words, at most eighty
COVER = 0.7  # a grounding sentence may stand for a longer split sentence only past this share
DRAFT_FIELDS = ("domain", "field", "subfield", "task", "tasks", "modality", "modalities",
                "modality_output", "output_type", "paradigm", "arch_class", "arch_subclass",
                "is_generative", "tags", "keywords", "labs", "lineage", "members", "tagline",
                "summary_markdown", "key_contribution", "grounding")
# Sentence ends: a terminal mark, then a capital, digit or opening quote; common abbreviations
# (Fig., et al., e.g., i.e., vs., No.) do not end one. A split the lane drew differently still
# passes `_grounded`, which matches by containment either way.
SENTENCE = re.compile(r"(?<!\bFig\.)(?<!\bal\.)(?<!e\.g\.)(?<!i\.e\.)(?<!\bvs\.)(?<!\bNo\.)"
                      r"(?<=[.!?])\s+(?=[A-Z0-9\"'(“])")


def prompt_version() -> str:
    return tree_sha256(PROMPT)[:8]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Folder writers hold their lock; discovery readers need a whole manifest at every instant.
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --- the check ------------------------------------------------------------------------


def sentences(text: str | None) -> list[str]:
    """The sentences of a Markdown prose field, paragraphs split first."""

    if not text:
        return []
    return [s.strip() for para in text.split("\n") for s in SENTENCE.split(para.strip())
            if s.strip()]


def _grounded(sentence: str, grounding: Sequence[Mapping[str, str]]) -> bool:
    """Whether a prose sentence has a grounding entry: the entry's sentence equal to it or
    containing it (the splitter cut finer than the lane), or contained in it while covering at
    least ``COVER`` of it (the splitter joined two). The quote itself is checked separately."""

    want = _norm(sentence)
    for g in grounding:
        have = _norm(g["sentence"])
        if not have:
            continue
        if have == want or want in have or (have in want and len(have) >= COVER * len(want)):
            return True
    return False


def _quote_findings(record: DescribeRecord, norm: Mapping[str, str]) -> list[str]:
    found = []
    for g in [*(to_dict(g) for g in record.grounding), *(to_dict(item) for item in record.lineage)]:
        words = len(_norm(g["quote"]).split())
        if not QUOTE_WORDS[0] <= words <= QUOTE_WORDS[1]:
            found.append(f"quote of {words} words, not {QUOTE_WORDS[0]} to {QUOTE_WORDS[1]}: "
                         f"{g['quote'][:60]!r}")
        elif g["sid"] not in norm:
            found.append(f"quote cites {g['sid']!r}, not a saved text: {g['quote'][:60]!r}")
        elif _norm(g["quote"]) not in norm[g["sid"]]:
            found.append(f"quote not verbatim in {g['sid']}: {g['quote'][:80]!r}")
    return found


def _prose_findings(record: DescribeRecord, grounding: Sequence[Mapping[str, str]], mode: str
                    ) -> list[str]:
    if mode == "readme-only":
        return (["readme-only family: no summary or key contribution allowed"]
                if record.summary_markdown or record.key_contribution else [])
    if not record.summary_markdown:
        return ["no summary"]
    found = []
    if not SUMMARY_CHARS[0] <= len(record.summary_markdown) <= SUMMARY_CHARS[1]:
        found.append(f"summary is {len(record.summary_markdown)} characters, "
                     f"not {SUMMARY_CHARS[0]} to {SUMMARY_CHARS[1]}")
    for s in [*sentences(record.summary_markdown), *sentences(record.key_contribution)]:
        if not _grounded(s, grounding):
            found.append(f"sentence not grounded: {s[:80]!r}")
    return found


def _member_findings(record: DescribeRecord, grounding: Sequence[Mapping[str, str]],
                     member_ids: Sequence[str]) -> list[str]:
    found = []
    for member, item in record.members.items():
        if member not in member_ids:
            found.append(f"members names {member}, not a packet member")
        found += [f"member {member} delta not grounded: {s[:80]!r}"
                  for s in sentences(item.delta_markdown) if not _grounded(s, grounding)]
        for name in ("output_type", "modality_output"):
            value = getattr(item, name)
            if value is not None and value not in vocab(name):
                found.append(f"member {member} {name} {value!r} is outside its vocabulary")
    return found


def _vocab_findings(record: DescribeRecord) -> list[str]:
    return [f"{name} {getattr(record, name)!r} is outside its vocabulary"
            for name in describe_jev.VOCAB_FIELDS
            if getattr(record, name) is not None and getattr(record, name) not in vocab(name)]


def verify(record: DescribeRecord, texts: Mapping[str, str], *, mode: str,
           member_ids: Sequence[str], resolves: Callable[[str], bool]) -> list[str]:
    """Every finding that keeps a record from ``grounded``; empty when it passes."""

    norm = {sid: _norm(t) for sid, t in texts.items()}
    grounding = [to_dict(g) for g in record.grounding]
    found = _quote_findings(record, norm) + _prose_findings(record, grounding, mode)
    found += _member_findings(record, grounding, member_ids)
    found += [f"lineage target id {item.target_id} does not resolve"
              for item in record.lineage if item.target_id and not resolves(item.target_id)]
    found += _vocab_findings(record)
    found += [f"lab {lab!r} is not named in any saved text" for lab in record.labs
              if not any(_norm(lab) in t for t in norm.values())]
    return found


class _Resolver:
    """Lineage targets to catalogue ids: roster names and aliases to stable ids, packet
    identities to family ids; punctuation-insensitive."""

    def __init__(self, data: DataLayout) -> None:
        self.ids: dict[str, str] = {}
        self.known: set[str] = set()
        for row in roster.read(data=data):
            if row.stable_id:
                self.known.add(row.stable_id)
                for name in (row.name, *row.aliases):
                    self.ids.setdefault(roster.norm(name), row.stable_id)
        for path in sorted((data.root / "research").glob("*/*.json")):
            packet = _json(path)
            self.known.add(packet["family"])
            identity = packet.get("identity") or {}
            for name in (identity.get("canonical"), *(identity.get("aliases") or [])):
                if name:
                    self.ids.setdefault(roster.norm(name), packet["family"])

    def resolve(self, target: str) -> str | None:
        return self.ids.get(roster.norm(target)) if target else None

    def resolves(self, target_id: str) -> bool:
        return target_id in self.known


# --- records from drafts --------------------------------------------------------------


def _record(draft: Mapping[str, Any] | None, facts: Mapping[str, Any], *, family: str,
            prompt: str, resolver: _Resolver, cls: str | None = None) -> DescribeRecord:
    """A ``DescribeRecord`` from a lane draft (or a typed one from the facts alone)."""

    body: dict[str, Any] = {k: v for k, v in (draft or {}).items() if k in DRAFT_FIELDS}
    for name, value in (facts.get("jev") or {}).items():
        body[name] = value
    body["lineage"] = [{**item, "target_id": item.get("target_id") or resolver.resolve(item.get("target", ""))}
                       for item in body.get("lineage") or []]
    sids = list(facts.get("sids") or [])
    if facts.get("institutions_sid"):
        sids.append(facts["institutions_sid"])
    if facts.get("mode") == "none":  # a textless record lists what it looked at (``_undescribed``)
        sids = list(facts.get("considered") or [])
    by = "readme-only" if facts.get("mode") == "readme-only" else f"{ROLE}@describe"
    return from_dict(DescribeRecord, {
        "schema": DescribeRecord.SCHEMA, "family": family, "by": by if draft else "describe",
        "era": facts.get("era"), "institutions": list(facts.get("institutions") or []),
        **body, "inputs": sorted(set(sids)), "prompt": prompt, "class": cls,
        "grounding_state": "unchecked", "basis": BASIS.get(facts.get("mode"))})


def _settle(family: str, folder: Path, lane: Mapping[str, Any], *, data: DataLayout,
            resolver: _Resolver) -> tuple[DescribeRecord | None, list[str]]:
    """One family's record from its draft, with the check's findings (``None`` with no draft)."""

    facts = _json(folder / "families" / family / "facts.json")
    trial_values = folder / "trial-values.json"
    if trial_values.is_file():
        facts = {**facts, "jev": _json(trial_values).get(family, {})}
    out = folder / "out" / f"{family}.json"
    if not out.is_file():
        return None, ["the lane wrote no draft"]
    try:
        record = _record(_json(out), facts, family=family, prompt=lane["prompt"], resolver=resolver)
    except (RecordError, TypeError, ValueError, KeyError) as exc:
        return None, [f"draft does not parse as a describe record: {str(exc)[:200]}"]
    texts = {sid: (folder / "families" / family / "source" / name).read_text("utf-8", errors="replace")
             for sid, name in _text_files(folder / "families" / family).items()}
    findings = verify(record, texts, mode=facts["mode"],
                      member_ids=[m["id"] for m in facts_members(folder, family)], resolves=resolver.resolves)
    return record, findings


def _text_files(family_folder: Path) -> dict[str, str]:
    """sid -> file name of each saved text in a family folder."""

    found: dict[str, str] = {}
    source = family_folder / "source"
    for path in sorted(source.iterdir()) if source.is_dir() else []:
        if path.suffix == ".txt":
            found[path.stem.rsplit("-", 1)[0]] = path.name
    return found


def facts_members(folder: Path, family: str) -> list[dict[str, Any]]:
    return _json(folder / "families" / family / "members.json")


def _write_record(record: DescribeRecord, state: str, *, data: DataLayout, cls: str | None = None,
                  write: bool) -> dict[str, Any]:
    final = from_dict(DescribeRecord, {**to_dict(record), "grounding_state": state,
                                       "class": cls or record.cls})
    if write:
        write_json(final, data.describe(record.family))
    return {"family": record.family, "verdict": state if cls is None else cls, "written": write}


# --- batches and lanes ----------------------------------------------------------------


def _launch(folder: Path, families: Sequence[str], role: str, *, data: DataLayout,
            returned: bool = False) -> str:
    # The return lane gets its own brief, so its result file is new: the launcher accepts a lane
    # only when the result file beside its brief was written by that run. A worker seat takes no
    # hub slot, so with workers on the hub gate is read only for the hub fallback.
    workers = describe_workers.seats() > 0
    if not workers:
        _hub_gate()
    if sum(lane["kind"] == KIND for lane in drive_mod.lanes(data=data)) >= _cap():
        raise drive_mod.Refused("describe_full", "no free describe seat")
    brief = folder / ("BRIEF-r2.md" if returned else "BRIEF.md")
    head = [f"# Describe lane, role `{role}`", "",
            f"Family folders: `{folder}/families/<family>/`. Write `{folder}/out/<family>.json` for",
            f"each family below; the result file is `{folder}/{brief.stem}.result.md`.", "",
            "Families: " + ", ".join(f"`{f}`" for f in families), "", ""]
    if (folder / "trial.json").is_file():
        head += ["Trial: answer every vocabulary field yourself from the source texts. Do not read",
                 "the classifier journal or trial-values.json. Those values are compared later.", ""]
    brief.write_text("\n".join(head) + PROMPT.read_text(encoding="utf-8"), encoding="utf-8")
    (folder / "out").mkdir(exist_ok=True)
    placed = describe_workers.launch(brief, families, role=role) if workers else None
    where: dict[str, str] = {}
    if placed is None:
        if workers:
            _hub_gate()
        run = drive_mod.launch(KIND, role, brief, keys=tuple(families), parent=os.environ.get("FLEET_RUN_ID"),
                               label=f"describe-{folder.name}{'-r2' if returned else ''}", lab="claude")
    else:
        run, where["placement"], where["label"] = placed
    _dump(folder / "lane.json", {"run": run, "role": role, "families": list(families),
                                 "prompt": prompt_version(), "returned": returned, **where})
    return run


def _hub_gate() -> None:
    """A hub seat leaves ``HUB_RESERVE`` hub slots free for 2a and review lanes."""

    if tick_full.hub_slots() <= int(os.environ.get(HUB_RESERVE) or 0):
        raise drive_mod.Refused("hub_full", "no free describe slot")


def _return(folder: Path, failing: Mapping[str, Sequence[str]], role: str, *, data: DataLayout) -> str:
    for family, findings in failing.items():
        (folder / "families" / family / "RETURN.md").write_text(
            "# The mechanical check returned your draft\n\nFix exactly these and rewrite "
            f"`out/{family}.json`:\n\n" + "".join(f"- {f}\n" for f in findings), encoding="utf-8")
    return _launch(folder, sorted(failing), role, data=data, returned=True)


def _collect(folder: Path, lane: Mapping[str, Any], *, data: DataLayout, resolver: _Resolver,
             write: bool) -> list[dict[str, Any]]:
    """Settle an ended lane's families: write grounded records, return the rest once, then type."""

    results, failing = [], {}
    for family in lane["families"]:
        record, findings = _settle(family, folder, lane, data=data, resolver=resolver)
        _dump(folder / "check" / f"{family}.json", {"findings": findings})
        if record is not None and not findings:
            results.append(_write_record(record, "grounded", data=data, write=write))
        else:
            failing[family] = findings
    if failing and not lane.get("returned") and write:
        run = _return(folder, failing, lane["role"], data=data)
        results += [{"family": f, "verdict": "returned", "run": run, "findings": fs}
                    for f, fs in failing.items()]
        return results
    for family, findings in failing.items():
        facts = _json(folder / "families" / family / "facts.json")
        draft = folder / "out" / f"{family}.json"
        try:
            record = _record(_json(draft) if draft.is_file() else None, facts, family=family,
                             prompt=lane["prompt"], resolver=resolver)
        except (RecordError, TypeError, ValueError, KeyError):
            record = _record(None, facts, family=family, prompt=lane["prompt"], resolver=resolver)
        row = _write_record(record, "ungrounded", data=data, cls="describe:ungrounded", write=write)
        results.append({**row, "findings": findings})
    if write:
        _dump(folder / "done.json", {"families": lane["families"]})
    return results


def _pins(data: DataLayout, tl: str, rows: Sequence[Any] | None = None) -> dict[str, str]:
    """Each describable family and the pin its facts are read at (``describe_text.validated_pins``);
    earlier pins only with ``EARLIER_PINS``."""

    return describe_text.validated_pins(data=data, tl=tl, rows=rows,
                                        earlier=os.environ.get(EARLIER_PINS) == "1")


def abstracts_for(family: str) -> bool:
    """Whether ``family`` may ground on its cited paper's saved abstract (``ABSTRACTS``)."""

    given = os.environ.get(ABSTRACTS, "").strip()
    return given == "all" or family in given.split(",")


def _trial() -> set[str]:
    given = os.environ.get(ABSTRACTS, "").strip()
    return set() if given in ("", "all") else set(given.split(","))


def due(families: Sequence[str] | None, *, data: DataLayout, tl: str) -> list[str]:
    """The families to describe: the given ones, else every validated family with a packet and no
    describe record (or one a stronger text has reached since, ``_undescribed``): an abstract
    trial's families first, then those validated at ``tl`` before those validated only at an
    earlier pin, each oldest packet first."""

    valid = _pins(data, tl)
    if families:
        return list(dict.fromkeys(f for f in families if f in valid))
    todo = [f for f in valid if data.packet(f).is_file() and _undescribed(f, data)]
    trial = _trial()
    return sorted(todo, key=lambda f: (f not in trial, valid[f] != tl,
                                       _json(data.packet(f)).get("written") or "", f))


def _basis(record: Mapping[str, Any]) -> str | None:
    """The kind of text a record rests on; a record from before ``basis`` is told by its class
    and ``by``. ``None`` for an old textless record, which never listed what it looked at."""

    if record.get("basis"):
        return record["basis"]
    if record.get("class") == "describe:no_text":
        return None
    return "readme" if record.get("by") == "readme-only" else "full"


def _undescribed(family: str, data: DataLayout) -> bool:
    """No record yet, or the packet now offers a kind of text stronger than the record's basis
    that the record did not read (the paper pass lands full texts after an abstract-grounded or
    textless record; ``describe_text.RANK``). An old textless record re-opens on any candidate."""

    path = data.describe(family)
    if not path.exists():
        return True
    record = _json(path)
    found = describe_text.candidates(_json(data.packet(family)), abstracts=abstracts_for(family))
    basis = _basis(record)
    if basis is None:
        return bool(found)
    read, rank = set(record.get("inputs") or ()), describe_text.RANK[basis]
    return any(describe_text.RANK[kind] > rank and sid not in read for sid, kind in found.items())


def _extend(folder: Path, todo: Sequence[str], count: int, *, data: DataLayout) -> list[str]:
    """Backfill a reserved seat without taking another folder's or lane's families, or one an
    earlier seat of this pass already typed (a staged folder keeps only its text families)."""

    live = drive_mod.lanes(data=data)
    with _locked():
        held = {f for p in _folders() for f in _json(p / "families.json")}
        held.update(k for lane in live for k in lane.get("keys", []))
        extra = [f for f in todo if f not in held and _undescribed(f, data)][:count]
        _dump(folder / "families.json", _json(folder / "families.json") + extra)
        return extra


def _gather_batch(folder: Path, todo: Sequence[str], *, data: DataLayout,
                  store: StoreLayout, tl: str, resolver: _Resolver
                  ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Keep ten text-bearing families, typing missing text before requesting an agent."""

    results, batch, states = [], [], []
    vocabularies = describe_jev.vocabularies()
    rows = ledger.read_all(data=data)  # once: ``gather`` would read the whole ledger per family
    valid = _pins(data, tl, rows)
    pending = _json(folder / "families.json")
    while pending:
        for family in pending:
            if family not in valid:
                results.append({"family": family, "verdict": "not_validated"})
                continue
            inputs = describe_text.gather(family, data=data, store=store, tl=valid[family], rows=rows,
                                          folder=folder / "families" / family, vocab=vocabularies,
                                          abstracts=abstracts_for(family))
            if inputs.mode == "none":
                record = _record(None, inputs.facts, family=family, prompt=prompt_version(), resolver=resolver)
                results.append(_write_record(record, "unchecked", data=data, cls="describe:no_text", write=True))
            else:
                batch.append(family)
                states.append({"id": family, "state": describe_jev.state_of(inputs)})
        if len(batch) >= BATCH or (folder / "trial.json").is_file():
            break
        pending = _extend(folder, todo, BATCH - len(batch), data=data)
    return results, batch, states


def _stage(folder: Path, todo: Sequence[str], *, data: DataLayout, store: StoreLayout, tl: str,
           resolver: _Resolver) -> list[dict[str, Any]]:
    """Fill a reserved folder outside the discovery lock; a folder lock owns its preparation."""

    results, batch, states = _gather_batch(folder, todo, data=data, store=store, tl=tl, resolver=resolver)
    values = describe_jev.classify(folder / "jev", states, data=data)
    if (folder / "trial.json").is_file():
        _dump(folder / "trial-values.json", values)
    else:
        for family, fields in values.items():
            facts = folder / "families" / family / "facts.json"
            _dump(facts, {**_json(facts), "jev": fields})
    _prior(folder, batch, data)
    _dump(folder / "families.json", batch)
    _dump(folder / "ready.json", {})
    if not batch:
        _dump(folder / "done.json", {})
    return results


def _folders() -> list[Path]:
    return sorted(p for p in work_dir(KIND).glob("*")
                  if p.is_dir() and not p.name.startswith(".")
                  and (p / "families.json").is_file() and not (p / "done.json").is_file())


def _cap() -> int:
    return max(0, min(drive_mod.lane_cap(KIND), int(os.environ.get("MENAGERIE_LANES_DESCRIBE", DEFAULT_SEATS))))


def _reserve(todo: Sequence[str], *, data: DataLayout) -> Path | None:
    """Reserve a seat and its family set; discovery is the only global critical section."""

    live = drive_mod.lanes(data=data)
    with _locked():
        folders = _folders()
        held = {f for p in folders for f in _json(p / "families.json")}
        held.update(k for lane in live for k in lane.get("keys", []))
        local_runs = {_json(p / "lane.json")["run"] for p in folders if (p / "lane.json").is_file()}
        external = sum(lane["kind"] == KIND and lane["run"] not in local_runs for lane in live)
        if len(folders) + external >= _cap():
            return None
        batch = [f for f in todo if f not in held][:BATCH]
        if not batch:
            return None
        folder = work_dir(KIND) / hashlib.sha256(",".join(batch).encode()).hexdigest()[:8]
        # An explicit re-description must never reuse a completed batch or its result file.
        if folder.exists():
            folder = work_dir(KIND) / f"{folder.name}-{os.getpid()}-{len(list(work_dir(KIND).glob('*')))}"
        folder.mkdir()
        _dump(folder / "families.json", batch)
        _dump(folder / "pending.json", {})
        if os.environ.get("MENAGERIE_DESCRIBE_TRIAL") == "1":
            _dump(folder / "trial.json", {"families": batch})
        return folder


def _prior(folder: Path, families: Sequence[str], data: DataLayout) -> None:
    """Copy each family's earlier published prose (the atlas regression set) beside its inputs."""

    for path in sorted((data.root / "atlas_prose").glob("*.json")) if (data.root / "atlas_prose").is_dir() else []:
        for entry in _json(path)["entries"]:
            model = data.model(entry["stable_id"])
            if model.is_file() and _json(model)["family"] in families:
                prior = folder / "families" / _json(model)["family"] / "prior.md"
                with prior.open("a", encoding="utf-8") as handle:
                    handle.write(f"<!-- {entry['stable_id']} -->\n{entry['summary_markdown']}\n\n")


def _selected_folders(folders: Sequence[Path], todo: Sequence[str], *,
                      only_staged: bool, only_matching: bool) -> list[Path]:
    """Select the requested batches separately from processing their state transitions."""

    selected = []
    wanted = set(todo)
    for folder in folders:
        matches = bool(set(_json(folder / "families.json")) & wanted)
        if only_matching and not matches:
            continue
        if only_staged and ((folder / "lane.json").is_file() or not matches):
            continue
        selected.append(folder)
    return selected


def _sweep(todo: Sequence[str], *, role: str, data: DataLayout, store: StoreLayout,
           tl: str, resolver: _Resolver, write: bool, only_staged: bool = False,
           fill_from: Sequence[str] | None = None, only_matching: bool = False
           ) -> tuple[list[dict[str, Any]], set[str]]:
    """Process independent folders under their own locks; family leases guard every launch."""

    results: list[dict[str, Any]] = []
    held: set[str] = set()
    folders = _folders()
    all_live = drive_mod.lanes(data=data)
    live = [lane for lane in all_live if lane["kind"] == KIND]
    held.update(k for lane in all_live for k in lane.get("keys", []))
    local_active = sum((p / "lane.json").is_file() and
                       review._fleet_run(_json(p / "lane.json")["run"]).get("state") != "TERMINAL"
                       for p in folders)
    room = max(0, _cap() - max(len(live), local_active))
    staged = [p for p in folders if not (p / "lane.json").is_file()][:room]
    selected = _selected_folders(folders, todo, only_staged=only_staged, only_matching=only_matching)
    for folder in selected:
        batch = _json(folder / "families.json")
        held.update(batch)
        more = _sweep_folder(folder, todo if fill_from is None else fill_from,
                             can_start=folder in staged, role=role, data=data,
                             store=store, tl=tl, resolver=resolver, write=write)
        results += more
        held.update(row["family"] for row in more)
        held.update(_json(folder / "families.json"))
    return results, held


def _sweep_folder(folder: Path, todo: Sequence[str], *, can_start: bool, role: str, data: DataLayout,
                  store: StoreLayout, tl: str, resolver: _Resolver, write: bool) -> list[dict[str, Any]]:
    """Own one folder's prepare/launch/collect transition without holding the discovery lock."""

    with (folder / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return []
        if (folder / "done.json").is_file():
            return []
        if not (folder / "lane.json").is_file() and not can_start:
            return []
        results = []
        batch = _json(folder / "families.json")
        try:
            if write and (folder / "pending.json").is_file() and not (folder / "ready.json").is_file():
                results += _stage(folder, todo, data=data, store=store, tl=tl, resolver=resolver)
                batch = _json(folder / "families.json")
            if batch:
                results += _batch_rows(folder, batch, role=role, data=data, resolver=resolver, write=write)
        except drive_mod.Refused as refused:
            results += [{"family": f, "verdict": "waiting", "why": refused.result} for f in batch]
        return results


def _batch_rows(folder: Path, batch: Sequence[str], *, role: str, data: DataLayout,
                resolver: _Resolver, write: bool) -> list[dict[str, Any]]:
    """One open batch's rows: launched (a staged folder), running, or settled by ``_collect``."""

    if not (folder / "lane.json").is_file():
        run = _launch(folder, batch, role, data=data) if write else "dry-run"
        return [{"family": f, "verdict": "launched", "run": run} for f in batch]
    lane = _json(folder / "lane.json")
    status = review._fleet_run(lane["run"])
    if status.get("state") != "TERMINAL" or not describe_workers.landed(lane):
        return [{"family": f, "verdict": "running", "run": lane["run"]} for f in lane["families"]]
    if status.get("admission_state") not in (None, "admitted") and write:
        # The hub's queue ended the lane before it ever ran: it did no work, so it starts again
        # rather than spending the families' one return.
        run = _launch(folder, lane["families"], lane["role"], data=data,
                      returned=bool(lane.get("returned")))
        return [{"family": f, "verdict": "relaunched", "run": run} for f in lane["families"]]
    return _collect(folder, lane, data=data, resolver=resolver, write=write)


def write(families: Sequence[str] | None, *, data: DataLayout, store: StoreLayout,
          role: str | None = None, write: bool = True) -> list[dict[str, Any]]:
    """Describe the due families; return one ``{"family", "verdict", ...}`` per family touched.

    Verdicts: ``grounded`` and ``describe:no_text`` (records written), ``launched``, ``running``,
    ``relaunched`` (its lane was never admitted) and ``returned`` (a lane holds the family), ``describe:ungrounded`` (typed after the return),
    ``waiting`` (a lane is active, so no new one starts), ``no_packet``, ``due`` (a dry run). Each
    seat holds at most ``BATCH`` families; discovery reserves disjoint batches under a short lock.
    """

    from menagerie.pipeline import publish

    role, tl = role or ROLE, publish.torchlens_pin()
    return _write(families, data=data, store=store, role=role, tl=tl, write=write)


@contextlib.contextmanager
def _locked() -> Iterator[None]:
    """Serialize only discovery and reservation of staged family sets."""

    work_dir(KIND).mkdir(parents=True, exist_ok=True)
    with (work_dir(KIND) / ".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _reserve_seats(todo: Sequence[str], *, data: DataLayout) -> tuple[set[str], list[str]]:
    """Reserve available disjoint batches and return the families still waiting for a seat."""

    rest = list(todo)
    new: set[str] = set()
    for _ in range(_cap()):
        if not rest:
            break
        folder = _reserve(rest, data=data)
        if folder is None:
            break
        batch = _json(folder / "families.json")
        new.update(batch)
        rest = [f for f in rest if f not in batch]
    return new, rest


def _unheld(todo: Sequence[str], held: set[str], data: DataLayout
            ) -> tuple[list[str], list[dict[str, Any]]]:
    """Split unleased families into those with packets and those with missing inputs."""

    remaining = [f for f in todo if f not in held]
    rest = [f for f in remaining if data.packet(f).is_file()]
    missing = [{"family": f, "verdict": "no_packet"} for f in remaining if not data.packet(f).is_file()]
    return rest, missing


def _fill_seats(todo: Sequence[str], *, data: DataLayout, store: StoreLayout, role: str,
                tl: str, resolver: _Resolver) -> list[dict[str, Any]]:
    """Reserve and start this pass's new batches exactly once, leaving unscheduled rows waiting."""

    new, rest = _reserve_seats(todo, data=data)
    results, seen = [], set()
    if new:
        more, _ = _sweep(sorted(new), role=role, data=data, store=store, tl=tl,
                          resolver=resolver, write=True, only_staged=True, fill_from=todo)
        results += more
        seen = {r["family"] for r in more}
        results += [{"family": f, "verdict": "waiting"} for f in sorted(new - seen)]
    return results + [{"family": f, "verdict": "waiting"} for f in rest if f not in seen]


def _write(families: Sequence[str] | None, *, data: DataLayout, store: StoreLayout, role: str,
           tl: str, write: bool) -> list[dict[str, Any]]:
    todo = due(families, data=data, tl=tl)
    resolver = _Resolver(data)
    results = [{"family": f, "verdict": "not_validated"} for f in dict.fromkeys(families or ()) if f not in todo]
    swept, held = _sweep(todo, role=role, data=data, store=store, tl=tl,
                         resolver=resolver, write=write, only_matching=bool(families) and _cap() <= 1)
    results += swept
    seen = {r["family"] for r in results}
    results += [{"family": f, "verdict": "waiting"} for f in todo if f in held and f not in seen]
    rest, missing = _unheld(todo, held, data)
    results += missing
    if not write:
        return results + [{"family": f, "verdict": "due"} for f in rest]
    return results + _fill_seats(rest, data=data, store=store, role=role, tl=tl, resolver=resolver)


def _check_one(family: str, *, data: DataLayout, store: StoreLayout, resolver: _Resolver
               ) -> dict[str, Any]:
    path = data.describe(family)
    if not path.is_file():
        return {"family": family, "verdict": "no_record", "findings": []}
    record = read_json(DescribeRecord, path)
    if record.cls == "describe:no_text":
        return {"family": family, "verdict": "typed", "findings": []}
    folder = work_dir(KIND) / "check" / family
    shutil.rmtree(folder, ignore_errors=True)
    packet = _json(data.packet(family))
    basis = _basis(to_dict(record))
    texts = {sid: p.read_text("utf-8", errors="replace")
             for sid, p in describe_text._texts(packet, folder, store, abstracts=basis == "abstract").items()
             if sid in record.inputs}
    findings = verify(record, texts, mode="readme-only" if basis == "readme" else "full",
                      member_ids=[m["id"] for m in packet.get("members", [])],
                      resolves=resolver.resolves)
    state = "grounded" if not findings else "ungrounded"
    if state != record.grounding_state:
        write_json(from_dict(DescribeRecord, {**to_dict(record), "grounding_state": state}), path)
    return {"family": family, "verdict": state, "findings": findings}


def check(families: Sequence[str] | None, *, data: DataLayout, store: StoreLayout
          ) -> list[dict[str, Any]]:
    """Re-run the mechanical check on written records (default: every one); update each record's
    ``grounding_state`` when it changed. Returns ``{"family", "verdict", "findings"}`` per record:
    ``grounded``, ``ungrounded``, ``typed`` (a ``describe:no_text`` record) or ``no_record``."""

    resolver = _Resolver(data)
    if families:
        names = list(families)
    else:
        root = data.root / "describe"
        names = sorted(p.stem for p in root.glob("*/*.json")) if root.is_dir() else []
    return [_check_one(f, data=data, store=store, resolver=resolver) for f in names]


def seats(*, data: DataLayout) -> list[dict[str, Any]]:
    """Live describe seats with their family leases and the current cap."""

    return [{**lane, "cap": _cap(), "families": lane.get("keys", [])}
            for lane in drive_mod.lanes(data=data) if lane["kind"] == KIND]
