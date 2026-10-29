"""EN-1 route census: per-row framework-reroute deltas over the composition corpora.

The 200a misroute class (m10502, m4714) existed because TWO framework derivations
coexisted and every census consulted the weaker one. This tool is the durable form of
the pre-flip adjudication: for every row of each corpus it reports

* the LEGACY driver derivation (bare ``tf`` substring, no torch-zoo veto),
* the LEGACY plan derivation (``cli._framework``: full-word markers, marker spelling
  returned raw), and
* the UNIFIED derivation (``constants.framework_from_markers`` as shipped),

and emits one JSON row per delta with the reason class, so a reviewer can adjudicate
every reroute rather than trusting an aggregate. Run it read-only any time; nothing
here mutates records.

Usage::

    PYTHONPATH=. python -m menagerie.crawler.tools.env_route_census \
        [--corpus PATH ...] [--deltas-only]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from menagerie.crawler.constants import (
    _FRAMEWORK_MARKER_ALIASES,
    TORCH_NATIVE_ZOO_VETO,
    framework_from_markers,
)

EXIT_OK = 0
EXIT_USAGE = 64

_DEFAULT_CORPORA = (
    Path("menagerie/data/master_catalog.jsonl"),
    Path("menagerie/data/crawl_roster.jsonl"),
)

_LEGACY_ALIASES_BY_SPECIFICITY = tuple(
    sorted(_FRAMEWORK_MARKER_ALIASES, key=lambda item: (-len(item[0]), item[0]))
)
_LEGACY_PLAN_MARKERS = ("tensorflow", "keras", "jax", "flax", "paddle")
_PLAN_CANONICAL = {
    "tensorflow": "tensorflow",
    "keras": "tensorflow",
    "jax": "jax",
    "flax": "jax",
    "paddle": "paddle",
}
_TF_TOKEN = re.compile(r"(?<![a-z0-9])tf(?![a-z0-9])")


def legacy_driver_derivation(zoo: str, name: str) -> str:
    """Replay the pre-EN-1 driver derivation: substring aliases, no zoo veto."""

    def marker(text: str) -> str | None:
        for alias, framework in _LEGACY_ALIASES_BY_SPECIFICITY:
            if alias in text:
                return framework
        return None

    declared = marker(str(zoo).strip().lower())
    if declared is not None:
        return declared
    named = marker(str(name).strip().lower())
    if named is not None:
        return named
    return "pytorch"


def legacy_plan_derivation(name: str, zoo: str) -> str:
    """Replay the deleted ``cli._framework``, canonicalized for comparability."""

    lowered = f"{name} {zoo}".lower()
    for marker in _LEGACY_PLAN_MARKERS:
        if marker in lowered:
            return _PLAN_CANONICAL[marker]
    return "pytorch"


def _delta_reason(zoo: str, name: str, legacy: str, unified: str) -> str:
    """Classify one driver reroute for the adjudication census."""

    zoo_text = str(zoo).strip().lower()
    if zoo_text in TORCH_NATIVE_ZOO_VETO:
        return "torch-zoo-veto"
    lowered = f"{zoo_text} {str(name).strip().lower()}"
    if "tf" in lowered and _TF_TOKEN.search(lowered) is None:
        return "tf-token-boundary"
    return "other"


def _corpus_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def census(corpus: Path) -> dict[str, Any]:
    """Compute the reroute census for one corpus.

    Returns
    -------
    dict[str, Any]
        Row counts, plan-vs-driver divergence counts before and after
        unification, and the per-row driver reroute deltas with reasons.
    """

    rows = 0
    legacy_plan_vs_driver = 0
    driver_deltas: list[dict[str, str]] = []
    reasons: Counter[str] = Counter()
    for row in _corpus_rows(corpus):
        name, zoo = str(row.get("name", "")), str(row.get("zoo", ""))
        rows += 1
        legacy = legacy_driver_derivation(zoo, name)
        unified = framework_from_markers(zoo, name)
        if legacy != legacy_plan_derivation(name, zoo):
            legacy_plan_vs_driver += 1
        if legacy != unified:
            reason = _delta_reason(zoo, name, legacy, unified)
            reasons[reason] += 1
            driver_deltas.append(
                {
                    "name": name,
                    "zoo": zoo,
                    "legacy_driver": legacy,
                    "unified": unified,
                    "reason": reason,
                }
            )
    return {
        "corpus": str(corpus),
        "rows": rows,
        "legacy_plan_vs_driver_divergence": legacy_plan_vs_driver,
        # Structurally zero after EN-1: plan and driver call the same function.
        "unified_plan_vs_driver_divergence": 0,
        "driver_reroute_deltas": len(driver_deltas),
        "delta_reasons": dict(sorted(reasons.items())),
        "deltas": driver_deltas,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print the reroute census for each corpus as one JSON document."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        type=Path,
        default=None,
        help="corpus JSONL path (repeatable; defaults to both composition corpora)",
    )
    parser.add_argument(
        "--deltas-only",
        action="store_true",
        help="emit only the per-row delta lists, one JSON row per line",
    )
    args = parser.parse_args(argv)
    corpora = tuple(args.corpus) if args.corpus else _DEFAULT_CORPORA
    missing = [str(path) for path in corpora if not path.is_file()]
    if missing:
        print(f"corpus files missing: {', '.join(missing)}", file=sys.stderr)
        return EXIT_USAGE
    results = [census(path) for path in corpora]
    if args.deltas_only:
        for result in results:
            for delta in result["deltas"]:
                print(json.dumps({"corpus": result["corpus"], **delta}, sort_keys=True))
        return EXIT_OK
    print(json.dumps(results, indent=2, sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
