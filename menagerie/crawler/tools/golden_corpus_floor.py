"""ONE authority for the golden checker-transcript corpus membership gate.

W-7 (rule-2 trilab consensus, 2026-08-29): the calibration corpus had no
membership lock — "re-harvest a clearer exemplar" was a one-line path around
every zero bar in the calibration contract (the m10470 temptation the Opus
adjudication names in §7.3). This module is the committed-manifest set-equality
pattern (the former smoke inventory floor, retired by RC-3) applied to the
corpus itself:

- ``golden_corpus_manifest.json`` is a committed **byte-digest manifest**: every
  golden's exact sha256, its lane, per-lane minimum counts, and a PERMANENT
  retirement census. Enforcement is set equality over file names plus digest
  equality per file — additions, deletions, renames, and in-place rewrites all
  fail with the named files.
- :func:`enforce` is the single comparison authority. Three consumers route
  through it: the in-suite anti-substitution test, the production activation
  gate (:func:`menagerie.crawler.checker_calibration._require_receipt_coverage`),
  and the receipt mint/verify path
  (:mod:`menagerie.crawler.checker_calibration_mint`) — so replacing or removing
  a golden in the same change as a mint attempt refuses.
- ``write`` regenerates the member rows from corpus bytes and PRESERVES the
  retirement census and lane minimums. Retiring a golden requires naming it
  (``--retire file=reason``) in the same reviewed change that deletes it; every
  retirement leaves a permanent census row (name, lane, digest, date, reason).
  Reactivating a retired name (a future width increase re-recording the
  width-retired classes) requires the explicit ``--reactivate`` flag.

The manifest deliberately lives BESIDE the corpus directory, not inside it:
``load_golden_corpus_entries`` and the capture tool glob ``*.json`` inside the
corpus dir, and the manifest is not a transcript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

MANIFEST_SCHEMA_VERSION = "menagerie.crawler.golden-corpus-manifest.v1"
MANIFEST_RELPATH = Path("menagerie/crawler/tests/golden_corpus_manifest.json")
CORPUS_RELPATH = Path("menagerie/crawler/tests/golden_checker_transcripts")

#: Filename prefix -> gate kind. The corpus names its lane in the filename and
#: :data:`menagerie.crawler.checker_calibration._GOLDEN_PREFIX` derives models
#: from the same prefixes; this map is the manifest-side spelling of that
#: convention and the manifest loader refuses a row whose declared gate kind
#: contradicts its own filename.
GOLDEN_KIND_PREFIXES: tuple[tuple[str, str], ...] = (
    ("metadata_", "metadata_batch"),
    ("fidelity_", "fidelity"),
    ("terminal_", "terminal_disposition"),
)

_GATE_KINDS = tuple(kind for _prefix, kind in GOLDEN_KIND_PREFIXES)
_RETIRED_ROW_FIELDS = ("file", "gate_kind", "sha256", "retired_at", "reason")


class GoldenCorpusFloorError(RuntimeError):
    """Raised when the manifest is malformed or the corpus cannot be measured."""


@dataclass(frozen=True)
class CorpusManifest:
    """Validated committed corpus membership + retirement census."""

    members: Mapping[str, Mapping[str, str]]
    class_minimums: Mapping[str, int]
    retired: tuple[Mapping[str, Optional[str]], ...]


def golden_kind_for_filename(name: str) -> Optional[str]:
    """Return the gate kind a corpus filename's prefix declares, if any."""

    for prefix, kind in GOLDEN_KIND_PREFIXES:
        if name.startswith(prefix):
            return kind
    return None


def _require_plain_name(name: object, *, where: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or "/" in name
        or "\\" in name
        or not name.endswith(".json")
    ):
        raise GoldenCorpusFloorError(
            f"golden corpus manifest {where} names an invalid corpus file {name!r}: "
            "entries are plain *.json filenames inside the corpus directory"
        )
    return name


def _require_digest(value: object, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise GoldenCorpusFloorError(
            f"golden corpus manifest {where} must carry a lowercase hex sha256, got {value!r}"
        )
    return value


def load_manifest(repo_root: Path) -> CorpusManifest:
    """Load and fail-closed validate the committed corpus manifest."""

    path = repo_root / MANIFEST_RELPATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GoldenCorpusFloorError(
            f"missing committed golden corpus manifest: {MANIFEST_RELPATH} — regenerate "
            "with `python -m menagerie.crawler.tools.golden_corpus_floor write`"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GoldenCorpusFloorError(
            f"unparseable golden corpus manifest {MANIFEST_RELPATH}: {exc}"
        ) from exc
    if not isinstance(data, Mapping) or data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise GoldenCorpusFloorError(
            f"golden corpus manifest schema {data.get('schema_version')!r} != "
            f"{MANIFEST_SCHEMA_VERSION!r}"
        )
    raw_members = data.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise GoldenCorpusFloorError("golden corpus manifest members must be a non-empty list")
    members: dict[str, dict[str, str]] = {}
    for index, row in enumerate(raw_members):
        where = f"members[{index}]"
        if not isinstance(row, Mapping) or set(row) != {"file", "gate_kind", "sha256"}:
            raise GoldenCorpusFloorError(
                f"golden corpus manifest {where} must carry exactly file/gate_kind/sha256"
            )
        name = _require_plain_name(row.get("file"), where=where)
        if name in members:
            raise GoldenCorpusFloorError(f"golden corpus manifest lists {name} twice")
        derived = golden_kind_for_filename(name)
        if derived is None:
            raise GoldenCorpusFloorError(
                f"golden corpus manifest {where}: {name} matches no known lane prefix "
                f"({', '.join(prefix for prefix, _kind in GOLDEN_KIND_PREFIXES)})"
            )
        if row.get("gate_kind") != derived:
            raise GoldenCorpusFloorError(
                f"golden corpus manifest {where}: {name} declares gate_kind "
                f"{row.get('gate_kind')!r} but its filename prefix derives {derived!r}"
            )
        members[name] = {
            "gate_kind": derived,
            "sha256": _require_digest(row.get("sha256"), where=where),
        }
    minimums = data.get("class_minimums")
    if not isinstance(minimums, Mapping) or set(minimums) != set(_GATE_KINDS):
        raise GoldenCorpusFloorError(
            "golden corpus manifest class_minimums must name exactly "
            f"{sorted(_GATE_KINDS)}"
        )
    bound_minimums: dict[str, int] = {}
    for kind in _GATE_KINDS:
        value = minimums.get(kind)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise GoldenCorpusFloorError(
                f"golden corpus manifest class_minimums[{kind}] must be an int >= 1"
            )
        bound_minimums[kind] = value
    raw_retired = data.get("retired")
    if not isinstance(raw_retired, list):
        raise GoldenCorpusFloorError("golden corpus manifest retired must be a list")
    retired: list[dict[str, Optional[str]]] = []
    seen_retired: set[str] = set()
    for index, row in enumerate(raw_retired):
        where = f"retired[{index}]"
        if not isinstance(row, Mapping) or set(row) != set(_RETIRED_ROW_FIELDS):
            raise GoldenCorpusFloorError(
                f"golden corpus manifest {where} must carry exactly "
                f"{'/'.join(_RETIRED_ROW_FIELDS)}"
            )
        name = _require_plain_name(row.get("file"), where=where)
        if name in seen_retired:
            raise GoldenCorpusFloorError(f"golden corpus manifest retires {name} twice")
        seen_retired.add(name)
        if name in members:
            raise GoldenCorpusFloorError(
                f"golden corpus manifest lists {name} as BOTH a member and retired; "
                "the census and the active roster are disjoint by construction"
            )
        digest = row.get("sha256")
        if digest is not None:
            _require_digest(digest, where=where)
        for field in ("gate_kind", "retired_at", "reason"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise GoldenCorpusFloorError(
                    f"golden corpus manifest {where} must carry a non-empty {field}"
                )
        retired.append({key: row[key] for key in _RETIRED_ROW_FIELDS})
    return CorpusManifest(
        members=members, class_minimums=bound_minimums, retired=tuple(retired)
    )


def observe_corpus(corpus_dir: Path) -> dict[str, str]:
    """Measure the corpus: every ``*.json`` transcript's exact byte digest."""

    if not corpus_dir.is_dir():
        raise GoldenCorpusFloorError(f"golden corpus directory missing: {corpus_dir}")
    observed: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        observed[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not observed:
        raise GoldenCorpusFloorError(
            f"golden corpus directory {corpus_dir} holds ZERO transcripts — refusing to "
            "treat an empty corpus as a measurement"
        )
    return observed


def enforce(observed: Mapping[str, str], manifest: CorpusManifest) -> list[str]:
    """THE comparison authority: name-set equality plus per-file byte identity.

    Returns failure messages; empty means the corpus on disk IS the committed
    membership. Every consumer — the in-suite test, the production activation
    gate, the mint/verify path — routes through this function.
    """

    failures: list[str] = []
    observed_names = set(observed)
    member_names = set(manifest.members)
    for name in sorted(member_names - observed_names):
        failures.append(
            f"golden {name} is in the committed manifest but not on disk — a removal or "
            "rename must retire it (`golden_corpus_floor write --retire "
            f"{name}=<reason>`) in the same reviewed change"
        )
    for name in sorted(observed_names - member_names):
        if any(row["file"] == name for row in manifest.retired):
            failures.append(
                f"golden {name} is RETIRED in the committed census but present on disk — "
                "reactivating a retired class is an explicit reviewed change "
                "(`golden_corpus_floor write --reactivate {name}`)".replace("{name}", name)
            )
        else:
            failures.append(
                f"golden {name} is on disk but not in the committed manifest — corpus "
                "membership moves only in a reviewed manifest change"
            )
    for name in sorted(member_names & observed_names):
        expected = manifest.members[name]["sha256"]
        if observed[name] != expected:
            failures.append(
                f"golden {name} bytes were replaced in place: manifest sha256 {expected} "
                f"!= observed {observed[name]} — a re-recorded exemplar is a reviewed "
                "manifest change, never a silent swap"
            )
    counts: dict[str, int] = {kind: 0 for kind in _GATE_KINDS}
    for name, row in manifest.members.items():
        counts[row["gate_kind"]] += 1
    for kind in _GATE_KINDS:
        if counts[kind] < manifest.class_minimums[kind]:
            failures.append(
                f"golden corpus manifest carries {counts[kind]} {kind} members, under its "
                f"own committed minimum {manifest.class_minimums[kind]}"
            )
    return failures


def default_repo_root() -> Path:
    """Return the repo root this module ships in (tools/ is three levels down)."""

    return Path(__file__).resolve().parents[3]


def enforce_committed_corpus(
    repo_root: Optional[Path] = None, corpus_dir: Optional[Path] = None
) -> list[str]:
    """Load the committed manifest and enforce it over one corpus directory."""

    root = repo_root if repo_root is not None else default_repo_root()
    manifest = load_manifest(root)
    directory = corpus_dir if corpus_dir is not None else root / CORPUS_RELPATH
    return enforce(observe_corpus(directory), manifest)


def _parse_retire(values: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        name, separator, reason = value.partition("=")
        if not separator or not name or not reason:
            raise GoldenCorpusFloorError(
                f"--retire takes file=reason, got {value!r}"
            )
        parsed[name] = reason
    return parsed


def write_manifest(
    repo_root: Path,
    *,
    corpus_dir: Optional[Path] = None,
    retire: Optional[Mapping[str, str]] = None,
    reactivate: Sequence[str] = (),
    retired_at: Optional[str] = None,
) -> CorpusManifest:
    """Regenerate member rows from corpus bytes; preserve census + minimums.

    A previously-manifested member absent from disk REFUSES unless named in
    ``retire`` (each retirement appends a permanent census row carrying the
    digest the manifest last recorded). A retired name present on disk refuses
    unless named in ``reactivate``. Lane minimums carry over from the committed
    manifest (seeded from the observed counts on the founding write) — raising
    or lowering one is a hand-reviewed manifest edit by design.
    """

    directory = corpus_dir if corpus_dir is not None else repo_root / CORPUS_RELPATH
    observed = observe_corpus(directory)
    retire = dict(retire or {})
    reactivating = set(reactivate)
    previous: Optional[CorpusManifest]
    try:
        previous = load_manifest(repo_root)
    except GoldenCorpusFloorError:
        previous = None
    retired_rows: list[dict[str, Optional[str]]] = (
        [dict(row) for row in previous.retired] if previous is not None else []
    )
    if previous is not None:
        for name in sorted(set(previous.members) - set(observed)):
            reason = retire.pop(name, None)
            if reason is None:
                raise GoldenCorpusFloorError(
                    f"golden {name} disappeared from the corpus without a retirement — "
                    f"pass --retire {name}=<reason> in the same reviewed change"
                )
            if retired_at is None or not retired_at:
                raise GoldenCorpusFloorError(
                    "retiring a golden requires --retired-at <YYYY-MM-DD>"
                )
            retired_rows.append(
                {
                    "file": name,
                    "gate_kind": previous.members[name]["gate_kind"],
                    "sha256": previous.members[name]["sha256"],
                    "retired_at": retired_at,
                    "reason": reason,
                }
            )
    if retire:
        raise GoldenCorpusFloorError(
            "--retire names files that are not disappearing manifest members: "
            + ", ".join(sorted(retire))
        )
    retired_names = {str(row["file"]) for row in retired_rows}
    for name in sorted(set(observed) & retired_names):
        if name not in reactivating:
            raise GoldenCorpusFloorError(
                f"golden {name} is retired in the committed census but present on disk — "
                f"pass --reactivate {name} to re-arm it as an explicit reviewed change"
            )
        retired_rows = [row for row in retired_rows if row["file"] != name]
    members: list[dict[str, str]] = []
    counts: dict[str, int] = {kind: 0 for kind in _GATE_KINDS}
    for name in sorted(observed):
        kind = golden_kind_for_filename(name)
        if kind is None:
            raise GoldenCorpusFloorError(
                f"corpus file {name} matches no known lane prefix; the corpus directory "
                "holds only lane-prefixed transcripts"
            )
        counts[kind] += 1
        members.append({"file": name, "gate_kind": kind, "sha256": observed[name]})
    minimums = (
        dict(previous.class_minimums)
        if previous is not None
        else {kind: max(counts[kind], 1) for kind in _GATE_KINDS}
    )
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "comment": (
            "Committed byte-digest membership manifest for the golden checker-transcript "
            "corpus (W-7 anti-substitution floor). Enforced by name-set equality plus "
            "per-file sha256 identity in "
            "menagerie.crawler.tools.golden_corpus_floor.enforce(); consumed by the "
            "in-suite anti-substitution test, the production calibration activation "
            "gate, and the receipt mint/verify path. Regenerate ONLY via `python -m "
            "menagerie.crawler.tools.golden_corpus_floor write`; never hand-edit. The "
            "retired list is a PERMANENT census — rows are appended by --retire and "
            "removed only by an explicit --reactivate reviewed change."
        ),
        "members": members,
        "class_minimums": {kind: minimums[kind] for kind in sorted(minimums)},
        "retired": sorted(retired_rows, key=lambda row: str(row["file"])),
    }
    path = repo_root / MANIFEST_RELPATH
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return load_manifest(repo_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("enforce", "write"),
        help="enforce the committed manifest, or regenerate it from corpus bytes",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument(
        "--retire",
        action="append",
        default=[],
        metavar="FILE=REASON",
        help="retire a disappearing member into the permanent census (write only)",
    )
    parser.add_argument(
        "--reactivate",
        action="append",
        default=[],
        metavar="FILE",
        help="re-arm a retired golden present on disk (write only)",
    )
    parser.add_argument("--retired-at", default=None, help="YYYY-MM-DD for --retire rows")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        if args.command == "write":
            manifest = write_manifest(
                repo_root,
                corpus_dir=args.corpus_dir,
                retire=_parse_retire(args.retire),
                reactivate=tuple(args.reactivate),
                retired_at=args.retired_at,
            )
            print(
                f"golden_corpus_floor: wrote {MANIFEST_RELPATH} — "
                f"{len(manifest.members)} members, {len(manifest.retired)} retired"
            )
            return 0
        failures = enforce_committed_corpus(repo_root, args.corpus_dir)
    except GoldenCorpusFloorError as exc:
        print(f"golden_corpus_floor: FAIL-CLOSED — {exc}")
        return 2
    if failures:
        print("golden_corpus_floor: CORPUS DRIFT (byte identity vs committed manifest)")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    manifest = load_manifest(repo_root)
    print(
        f"golden_corpus_floor: corpus exact — {len(manifest.members)} members match the "
        "committed manifest"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
