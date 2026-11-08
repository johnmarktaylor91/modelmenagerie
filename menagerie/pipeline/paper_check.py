"""Whether a saved paper is the family's own paper, before any describe prose rests on it.

Two checks, both deterministic and read from the packet and the saved text alone:

- ``refused_link``: the packet's paper link is one paper harvest refused (its last ``paper-harvest``
  amendment says ``shared docstring``: a classics module naming several papers, so its link names
  none of them for this family). ``research.complete`` and ``paper_texts`` never fetch such a link.
- ``matches``: the family's names (``names``: its id, canonical name and aliases and its members'
  naming words) appear in the paper's opening: its title, abstract and the start of its
  introduction. A paper whose opening never names the family is taken as another family's paper
  (a backbone, a baseline, a library paper or a wrong link); the family describes from its other
  texts.

A paper or page text that fails is retired (an amendment by ``bib_harvest.RETIRED`` naming its
sids; a later ``CONFIRMED`` amendment, a person's reading, clears it), so it never grounds a describe record, ``paper_texts`` never fetches it again and a describe
record that read it re-opens and stops publishing. ``sweep`` (``python -m
menagerie.pipeline.paper_check``) runs the check over every packet with a paper or page text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, StoreLayout, bib_harvest, now, read_json, write_json
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.records import Amendment, Packet, RecordError, from_dict

HEAD_CHARS = 6000
UNDONE = "paper-check"  # a citation or source change undone because its paper names another family
CONFIRMED = "paper-confirmed"  # a person read these sources against the family: the check passes
TEXT_ROLES = ("paper", "page")
GENERIC = frozenset({
    "the", "deep", "neural", "network", "networks", "model", "models", "net", "base", "hf", "torch",
    "pytorch", "tf", "keras", "small", "large", "tiny", "mini", "official", "simple", "basic",
    "classic", "classics", "vanilla", "learned", "learning", "image", "video", "graph", "text",
    "audio", "multi", "conv", "attention", "transformer", "lstm", "rnn", "cnn", "gan", "vae", "mmdet",
    "mmpretrain", "monai", "timm", "fuxictr", "compressai", "basicsr", "original", "variant",
    "human", "point", "spatial", "temporal", "dynamic", "efficient", "fast", "light", "lightweight",
    "real", "single", "dual", "hybrid", "residual", "recurrent", "convolutional", "generative",
    "adversarial", "variational", "implicit", "unified", "universal", "general", "sparse", "dense",
    "neuro", "spike", "spiking", "quantum", "graph", "depth", "stereo", "super", "semantic",
    "wongkinyiu", "facebook", "google", "microsoft", "nvidia", "openai", "huggingface",
    "encoder", "decoder", "head", "backbone", "block", "layer", "module", "classifier", "generator",
    "discriminator", "pretrained", "giant", "huge", "benchmark", "reference", "driven", "sphere",
})
VARIANT = re.compile(r"(?:v\d+|x\d+|\d+[a-z]?)$")


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def tokens(names: Iterable[str]) -> frozenset[str]:
    """Each name's naming words: its first word, any later word with a digit (``yolov7``), the
    first two words joined, and a word's stem without a version suffix; generic words never."""

    out: set[str] = set()
    for name in names:
        words = _words(re.sub(r"\s*\([^)]*\)", "", name))
        named = [w for n, w in enumerate(words) if n == 0 or (any(c.isdigit() for c in w)
                                                                 and not VARIANT.fullmatch(w))]
        for word in named:
            if word in GENERIC or word.isdigit():
                continue
            if len(word) >= 3 or (len(word) == 2 and any(c.isdigit() for c in word)):
                out.add(word)
            stem = VARIANT.sub("", word)
            if len(stem) >= 4 and stem not in GENERIC:
                out.add(stem)
        if len(words) > 1 and len("".join(words[:2])) >= 5:
            out.add("".join(words[:2]))
    return frozenset(out)


def names(packet: Packet, data: DataLayout) -> frozenset[str]:
    """The family's tokens from its id and identity, and from each member's name the words after
    its first (a member name often starts with its library or repository: ``DyGLib TGN``)."""

    own = tokens([packet.family[2:], packet.identity.canonical, *packet.identity.aliases])
    later: list[str] = []
    for member in packet.members:
        path = data.model(member.id)
        if path.is_file():
            name = str(json.loads(path.read_text(encoding="utf-8")).get("name") or "")
            later.append(" ".join(_words(re.sub(r"\s*\([^)]*\)", "", name))[1:]))
    return own | tokens(n for n in later if n)


def matches(names_: Iterable[str], text: str, title: str | None = None) -> bool:
    """Whether a name is a word of the text's opening (title, abstract, start of the
    introduction), or, five characters or more, inside its hyphen-free form (``ST-GCN``); or the
    cited ``title``, whose initials are a name (``DLA``: Deep Layer Aggregation), opens the text."""

    head = _words(text[:HEAD_CHARS])
    head_set, head_run = set(head), "".join(head)
    names_ = frozenset(names_)
    if any(name in head_set or (len(name) >= 5 and name in head_run) for name in names_):
        return True
    words = _words(title or "")
    initials = "".join(w[0] for w in words)
    return len(words) >= 3 and initials in names_ and " ".join(words) in " ".join(head)


def refused_link(packet: Packet) -> bool:
    """The last paper harvest refused the packet's paper link as a shared docstring's."""

    harvests = [a.why for a in packet.amendments if a.by == "paper-harvest"]
    return bool(harvests) and "shared docstring" in harvests[-1]


def retired(packet: Packet) -> frozenset[str]:
    """Sources that must not ground a claim: a held fill's (``held_sources``) and retired ones,
    less those a later ``CONFIRMED`` amendment names (a person read the paper against the family)."""

    gone = set(bib_harvest.held_sources(packet))
    for amendment in packet.amendments:
        if amendment.by == CONFIRMED:
            gone -= set(amendment.evidence)
        elif amendment.by == bib_harvest.RETIRED:
            gone |= set(amendment.evidence)
    return frozenset(gone)


def confirmed(packet: Packet) -> frozenset[str]:
    return frozenset(sid for a in packet.amendments if a.by == CONFIRMED for sid in a.evidence)


def retired_raw(raw: Any) -> frozenset[str]:
    """``retired`` for a packet read as plain JSON (describe's view); nothing for a malformed one."""

    try:
        return retired(from_dict(Packet, raw))
    except (RecordError, TypeError, ValueError):
        return frozenset()


def retire(packet: Packet, sids: Sequence[str], why: str) -> Packet:
    """The packet with an amendment retiring ``sids`` (``bib_harvest.RETIRED``)."""

    amendment = Amendment(at=now(), why=f"paper check: {why}"[:300], evidence=tuple(sids),
                          changed=("sources",), by=bib_harvest.RETIRED)
    return replace(packet, amendments=(*packet.amendments, amendment))


def _text_of(sha: str, store: StoreLayout, tmp: Path) -> str:
    dest = tmp / sha
    if not dest.exists():
        store_mod.materialize(sha, dest, store=store)
    return dest.read_text(encoding="utf-8", errors="replace")[:HEAD_CHARS]


def _paper_of(sid: str) -> str:
    return sid.removesuffix("-pdf")


def check(packet: Packet, *, data: DataLayout, store: StoreLayout, tmp: Path
          ) -> dict[str, str]:
    """Each paper or page text not yet retired -> ``ok``, ``mismatch`` or ``unreadable: ...``.
    A paper is judged whole: its abstract page and its PDF (``<sid>-pdf``) pass together when
    either names the family."""

    gone, sure = retired(packet), confirmed(packet)
    tokens_ = names(packet, data)
    title = packet.citation.title if packet.citation else None
    out: dict[str, str] = {}
    for src in packet.sources:
        if src.role not in TEXT_ROLES or not src.text or src.sid in gone:
            continue
        if src.sid in sure:
            out[src.sid] = "ok"
            continue
        try:
            text = _text_of(src.text, store, tmp)
        except (OSError, ValueError) as exc:
            out[src.sid] = f"unreadable: {type(exc).__name__}"
            continue
        out[src.sid] = "ok" if matches(tokens_, text, title) else "mismatch"
    named = {_paper_of(sid) for sid, verdict in out.items() if verdict == "ok"}
    return {sid: "ok" if verdict == "mismatch" and _paper_of(sid) in named else verdict
            for sid, verdict in out.items()}


def check_family(family: str, *, data: DataLayout, store: StoreLayout, tmp: Path,
                 write: bool) -> dict[str, Any]:
    """Check one packet's texts; with ``write`` retire the mismatched ones."""

    path = data.packet(family)
    packet = read_json(Packet, path)
    verdicts = check(packet, data=data, store=store, tmp=tmp)
    bad = sorted(sid for sid, verdict in verdicts.items() if verdict == "mismatch")
    if bad and write:
        fresh = read_json(Packet, path)
        write_json(retire(fresh, bad, "the saved text never names the family"), path)
    return {"family": family, "verdicts": verdicts, "retired": bad if write else [],
            "refused_link": refused_link(packet)}


def sweep(families: Sequence[str] | None, *, data: DataLayout, store: StoreLayout,
          write: bool) -> list[dict[str, Any]]:
    paths = ([data.packet(f) for f in families] if families is not None
             else sorted(data.root.glob("research/*/*.json")))
    rows = []
    with tempfile.TemporaryDirectory(prefix="paper-check-") as tmp:
        for path in paths:
            raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if not any(s.get("role") in TEXT_ROLES and s.get("text") for s in raw.get("sources") or ()):
                continue
            rows.append(check_family(str(raw["family"]), data=data, store=store, tmp=Path(tmp),
                                     write=write))
            for leftover in Path(tmp).iterdir():
                leftover.unlink()
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="paper_check")
    parser.add_argument("families", nargs="*")
    parser.add_argument("--write", action="store_true", help="retire mismatched texts")
    args = parser.parse_args(argv)
    rows = sweep(args.families or None, data=DataLayout.from_env(), store=StoreLayout.from_env(),
                 write=args.write)
    for row in rows:
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
