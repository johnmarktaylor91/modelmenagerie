"""Capture the golden checker transcript corpus from a real campaign tree.

This is the RE-RECORD tool for the corpus consumed by
``test_checker_envelope_golden.py``. Run it whenever the checker rotates
(model, prompt, or provider change) or when a campaign produces transcript
shapes the corpus lacks, then commit the refreshed JSON files it writes next
to itself.

The corpus is REAL data, deliberately not synthesized by any test builder:

* ``request.json`` / ``result.json`` pairs are the exact frozen bytes the
  production checker dispatch wrote and the checker subprocess answered with,
  lifted verbatim from the campaign work tree.
* Each entry is tied back to the durable gates ledger by its
  ``result_envelope_sha256`` (or, for pre-append refusals, recorded as
  ledger-absent), so a corpus entry is a recorded adjudication, not a fixture.

Usage (read-only against the campaign tree; writes only next to itself)::

    PYTHONPATH=<repo> python menagerie/crawler/tests/golden_checker_transcripts/\
capture_golden_checker_transcripts.py \
        --campaign-root CAMPAIGN_ROOT

Selection policy: for every (gate_kind, role) class the SMALLEST qualifying
pair is captured, where role ``accept`` means every item verdict is accurate
(fidelity: match + accurate rung) and ``refusal`` means at least one item is
not. One additional metadata entry records a pair the CURRENT validator
refuses purely for component-identity drift -- a real recorded drift refusal.

Two RICHEST-of-class entries are captured on top of that, because the
smallest-of-role policy is systematically blind to them:

* ``role`` is a BATCH-level binary, so a heterogeneous batch (some items
  accurate, some not) collapses into ``refusal`` alongside uniform rejections,
  and "smallest" then reliably picks a narrow uniform pair. Real sessions are
  routinely cap-width with all three item verdicts at once, and per-item
  routing is exactly what a batched lane has to get right.
* ``integrity`` -- whether the checker could re-attest the evidence at all --
  is orthogonal to the item verdict and absent from the role classification
  entirely. Every smallest-of-role pilot pair happens to carry
  ``integrity.verdict == "accurate"``, so the corpus had no real transcript in
  which the checker DISPUTED the evidence, on any lane.

Those two classes therefore select the WIDEST/richest qualifying pair rather
than the smallest, and their eligibility is the DURABLE replay (restamp ->
validate), which is what the golden tests actually assert: verbatim
component-identity drift is tolerated by name there, so it must not gate
selection or the corpus can only ever hold pairs whose shipped component bytes
have not moved since capture.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.constants import METADATA_BATCH_MAX

CORPUS_DIR = Path(__file__).resolve().parent
CHECKER_KINDS = ("metadata_batch", "fidelity", "terminal_disposition")

#: Hard ceiling on one written corpus entry. The repo's ``check-added-large-files``
#: pre-commit hook refuses a staged file over 500 KB and the corpus is
#: deliberately NOT in its exclude list (those exclusions are for the intentional
#: data catalogs), so this is a constraint of the venue, not a preference: a
#: cap-width metadata request embeds five full proposals with their evidence and
#: lands near it. The richest-of-class selection filters on this rather than
#: emitting an entry that cannot be committed, with headroom for a re-record
#: against a slightly larger session.
MAX_ENTRY_BYTES = 480 * 1024


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _integrity_disputed(item: Mapping[str, Any]) -> bool:
    """Whether the checker could NOT re-attest this item's evidence.

    Orthogonal to the content verdict: an item may be judged accurate while its
    excerpt locators, hashes, or excerpts failed to re-derive.
    """

    integrity = item.get("integrity")
    if not isinstance(integrity, Mapping):
        return False
    return integrity.get("verdict") not in (None, "accurate")


def _spread(items: Sequence[Mapping[str, Any]], key: str) -> int:
    """How many distinct values of ``key`` one recorded session carries."""

    if key == "integrity":
        return len({(item.get("integrity") or {}).get("verdict") for item in items})
    return len({item.get(key) for item in items})


def _item_accepted(kind: str, item: Mapping[str, Any]) -> bool:
    if kind == "fidelity":
        return (
            item.get("fidelity", {}).get("verdict") in {"match", "minor-drift"}
            and item.get("rung_check", {}).get("verdict") == "accurate"
        )
    if kind == "terminal_disposition":
        return item.get("terminal_disposition", {}).get("verdict") == "accepted"
    return item.get("verdict") == "accurate"


def _classify(kind: str, result: Mapping[str, Any]) -> str:
    items = result.get("items", [])
    return "accept" if all(_item_accepted(kind, item) for item in items) else "refusal"


def _ledger_index(
    gates_path: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    """Index ledger rows by result digest and by pre-normalization gate id.

    Metadata and fidelity results are bound to their durable repair generation
    before append (``_normalize_gate_generation``): the appended row carries
    ``<original gate_id>-generation-<n>`` and a resealed self-hash, so the raw
    transcript digest only matches terminal rows directly.
    """

    by_digest: dict[str, Mapping[str, Any]] = {}
    by_original_gate_id: dict[str, Mapping[str, Any]] = {}
    if not gates_path.is_file():
        return by_digest, by_original_gate_id
    for line in gates_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        digest = row.get("result_envelope_sha256")
        if isinstance(digest, str):
            by_digest[digest] = row
        gate_id = str(row.get("gate_id", ""))
        original = gate_id.split("-generation-", 1)[0]
        by_original_gate_id.setdefault(original, row)
    return by_digest, by_original_gate_id


def _committed_corpus_stable_ids() -> set[str]:
    """Every stable_id already asked by a committed corpus entry.

    The additive-exemplar rule (Opus W-7 anti-substitution discipline): a NEW
    class entry must be a NEW question. Re-capturing an item the corpus
    already interrogates under another name would duplicate the question
    while looking like added coverage.
    """

    ids: set[str] = set()
    for path in sorted(CORPUS_DIR.glob("*.json")):
        try:
            entry = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        request = entry.get("request") if isinstance(entry, Mapping) else None
        for item in (request.get("items") or []) if isinstance(request, Mapping) else []:
            if isinstance(item, Mapping):
                ids.add(str(item.get("stable_id")))
    return ids


def _machine_attestation_all_valid(kind: str, request: Mapping[str, Any]) -> bool:
    """Round-3 accept-exemplar screen: every excerpt machine-attests valid.

    An accept exemplar exists to witness the ACCEPT path. A pack carrying
    ``unattestable`` rows adjudicates as at-most cannot-verify per claim under
    the round-3 epoch (never an accept), and an ``invalid`` row refuses the
    replay envelope before any session exists -- either way the entry could
    never yield an acceptance witness, so it is screened out here rather than
    discovered dead in a paid matrix.
    """

    from menagerie.crawler.checker_dispatch import _derive_locator_attestation
    from menagerie.crawler.tools.blind_replay_matrix import golden_item_excerpts

    for item in request.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        attestation = _derive_locator_attestation(
            golden_item_excerpts(kind, item),
            item.get("source_manifest"),
            model_dir=item.get("model_dir"),
        )
        if not attestation["all_valid"]:
            return False
    return True


def capture(
    campaign_root: Path,
    *,
    dry_run: bool = False,
    checker_model: Optional[str] = None,
    only: Optional[set[str]] = None,
) -> None:
    from menagerie.crawler.checker_dispatch import (
        DETERMINISTIC_GATE_SCAFFOLD_FIELDS,
        CheckerDispatchError,
        CheckerPartialResultError,
        apply_machine_owned_gate_fields,
        compute_result_envelope_sha256,
        validate_checker_result_mapping,
    )

    machine_owned = (
        *DETERMINISTIC_GATE_SCAFFOLD_FIELDS,
        "checker",
        "result_envelope_sha256",
    )

    def durable_replays(result: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
        """Whether the DURABLE golden replay passes -- the corpus contract.

        Mirrors ``test_checker_envelope_golden._restamp``: strip the fields the
        production wrapper owns, re-stamp them on the current tree, validate,
        and require the checker-authored adjudication to survive verbatim.
        """

        raw = {key: deepcopy(value) for key, value in result.items() if key not in machine_owned}
        try:
            restamped = apply_machine_owned_gate_fields(
                raw,
                request,
                started_at=str(result["checker"]["started_at"]),
                finished_at=str(result["checker"]["finished_at"]),
            )
            validated = validate_checker_result_mapping(restamped, request)
        except (CheckerDispatchError, CheckerPartialResultError, KeyError, TypeError):
            return False
        return validated.get("items") == result.get("items")

    work = campaign_root / ".crawl-local" / "work"
    gates = campaign_root / "menagerie" / "crawler" / "records" / "gates" / "current-shard.jsonl"
    by_digest, by_original_gate_id = _ledger_index(gates)

    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    drift_candidates: list[dict[str, Any]] = []
    #: Every pair whose DURABLE replay passes and whose row the ledger recorded,
    #: regardless of verbatim component drift -- the pool the richest-of-class
    #: entries are drawn from.
    durable_candidates: list[dict[str, Any]] = []
    for request_path in sorted(work.rglob("request.json")):
        result_path = request_path.with_name("result.json")
        if not result_path.is_file():
            continue
        request = _load(request_path)
        kind = request.get("gate_kind")
        if kind not in CHECKER_KINDS:
            continue
        if (
            checker_model is not None
            and request.get("checker", {}).get("model") != checker_model
        ):
            # Post-rotation re-record must converge on the PINNED checker:
            # without this, retired-model pairs win smallest-of-class forever.
            continue
        if (
            kind == "metadata_batch"
            and len(request.get("items") or []) != METADATA_BATCH_MAX
        ):
            # The corpus pins the PRODUCTION metadata shape: a pair recorded at
            # another batch width measured a different lane (fable U-F2/C8),
            # cannot re-seal through the current envelope builder, and would
            # make the blind-replay matrix replay a question production never
            # asks. Width rotations re-record; they never inherit.
            continue
        result = _load(result_path)
        if result.get("result_envelope_sha256") != compute_result_envelope_sha256(result):
            print(f"SKIP (stale self-hash vs current digest fn): {result_path}")
            continue
        ledger_row = by_digest.get(result["result_envelope_sha256"]) or by_original_gate_id.get(
            str(result.get("gate_id", ""))
        )
        entry = {
            "kind": kind,
            "checker_model": request.get("checker", {}).get("model"),
            "prompt_sha256": request.get("checker", {}).get("prompt_sha256"),
            "source": {
                "request": str(request_path),
                "result": str(result_path),
            },
            "ledger_gate_id": ledger_row["gate_id"] if ledger_row else None,
            "request": request,
            "result": result,
            "ledger_row": ledger_row,
            "size": len(request_path.read_bytes()) + len(result_path.read_bytes()),
        }
        is_durable = durable_replays(result, request)
        if ledger_row is not None and is_durable:
            durable_candidates.append(entry)
        try:
            validate_checker_result_mapping(result, request)
        except (CheckerPartialResultError, CheckerDispatchError) as exc:
            message = str(exc)
            if "is not the machine-owned value for its envelope" in message:
                # Component-identity drift is what the golden verbatim test
                # tolerates BY NAME, so a drifting pair is still a valid corpus
                # entry (its durable replay is the contract) AND the exemplar the
                # drift class records. Gating role eligibility on the verbatim
                # result instead made re-recording a one-way ratchet: on any tree
                # whose shipped dispatcher/author-schema components have moved
                # since capture -- i.e. after any author_dispatch change -- every
                # accept class went MISSING and the corpus could only ever get
                # thinner, exactly when a rotation is why you re-record.
                drift_candidates.append({**entry, "refusal": message})
            elif not is_durable:
                print(f"SKIP ({message[:90]}): {result_path}")
                continue
        if not is_durable:
            continue
        role = _classify(kind, result)
        candidates.setdefault((kind, role), []).append(entry)

    wanted: dict[str, tuple[str, str]] = {
        "metadata_accept": ("metadata_batch", "accept"),
        "metadata_refusal": ("metadata_batch", "refusal"),
        "fidelity_accept": ("fidelity", "accept"),
        "fidelity_refusal": ("fidelity", "refusal"),
        "terminal_accept": ("terminal_disposition", "accept"),
        "terminal_refusal": ("terminal_disposition", "refusal"),
        # ADDITIVE round-3 class (2026-08-29): the committed metadata_accept
        # pair (m10470) is a deterministic NON-accept under the round-3 epoch
        # (its lineage evidence is exactly the architecture-class excerpt the
        # closed lineage rule refuses), so the metadata lane needed a fresh
        # accept exemplar that can actually witness the accept path. The class
        # takes the smallest recorded accept pair that (a) asks a stable_id no
        # committed entry already asks (anti-substitution: an added question,
        # never a re-dressed one) and (b) machine-attests every excerpt valid
        # (a pack with non-valid rows can never replay as an accept). The
        # retired-name entry stays committed: its recorded adjudication and
        # its deterministic-abstention replay are both still measurements.
        "metadata_accept_lineage_grounded": ("metadata_batch", "accept"),
    }
    manifest: dict[str, Any] = {}
    committed_ids = _committed_corpus_stable_ids()
    for name, key in wanted.items():
        if only is not None and name not in only:
            continue
        pool = candidates.get(key, [])
        if name == "metadata_accept_lineage_grounded":
            pool = [
                entry
                for entry in pool
                if {
                    str(item.get("stable_id"))
                    for item in entry["request"].get("items") or []
                    if isinstance(item, Mapping)
                }.isdisjoint(committed_ids)
                and _machine_attestation_all_valid(entry["kind"], entry["request"])
            ]
        if key == ("metadata_batch", "accept") and METADATA_BATCH_MAX >= 2:
            # The partial-result golden test truncates a multi-item batch; the
            # filter has no consumer at a single-item production width (the
            # truncation tripwire then drops the ONLY item instead).
            pool = [entry for entry in pool if entry["result"].get("batch_size", 0) >= 2]
        # A transcript the durable ledger actually recorded beats an orphaned
        # one (crashed or superseded leg) of any size.
        recorded = [entry for entry in pool if entry["ledger_row"] is not None]
        pool = recorded or pool
        if not pool:
            print(f"MISSING corpus class: {name}")
            continue
        manifest[name] = _write_entry(name, min(pool, key=lambda entry: entry["size"]), dry_run)

    if only is not None:
        # --only is the additive-capture mode: untouched classes are exactly
        # that, so the richest-of-class and drift-exemplar rewrites are skipped
        # unless explicitly asked for.
        return

    _capture_richest(durable_candidates, manifest, dry_run)

    # The committed entry is named (and documented) for the metadata lane, so the
    # exemplar is drawn from it: an unscoped smallest-of-all pick re-recorded a
    # TERMINAL pair into metadata_component_drift.json, which is a file whose name
    # then lies about its own provenance.
    metadata_drift = [entry for entry in drift_candidates if entry["kind"] == "metadata_batch"]
    if metadata_drift:
        chosen = min(metadata_drift, key=lambda entry: entry["size"])
        _write_entry("metadata_component_drift", chosen, dry_run)
        print(f"  drift refusal: {chosen['refusal'][:90]}")
    else:
        print("no component-drift refusal recorded (fine on a freshly rotated tree)")


def _serialize(chosen: Mapping[str, Any]) -> str:
    """Render one corpus entry exactly as it is committed."""

    payload = {key: value for key, value in chosen.items() if key != "size"}
    return json.dumps(payload, indent=1, sort_keys=True) + "\n"


def _entry_bytes(chosen: Mapping[str, Any]) -> int:
    """Committed byte size of one corpus entry, for the large-file ceiling."""

    return len(_serialize(chosen).encode())


def _write_entry(name: str, chosen: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
    """Write one corpus entry (or report what a real run would write).

    The chosen entry is never mutated: one pair can qualify for several classes,
    and a capture that popped bookkeeping fields off the shared object corrupted
    the selection of every class after it.
    """

    payload = {key: value for key, value in chosen.items() if key != "size"}
    summary = {
        "gate_id": payload["ledger_gate_id"],
        "checker_model": payload["checker_model"],
        "batch_size": payload["result"].get("batch_size"),
        "in_ledger": payload["ledger_row"] is not None,
        "source": Path(str(chosen["source"]["result"])).parent.name,
        "entry_kb": round(_entry_bytes(chosen) / 1024, 1),
    }
    if dry_run:
        print(f"WOULD capture {name}: {summary}")
        return summary
    (CORPUS_DIR / f"{name}.json").write_text(_serialize(chosen))
    print(f"captured {name}: {summary}")
    return summary


def _rung_adverse_matches_kernel_contract(items: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the first adverse item carries the identity-dispute shape.

    Mirrors ``test_golden_rung_adverse_item_derives_the_kernel_reason``: the
    first item whose verdict or rung check is not accurate must itself carry a
    disputed integrity verdict and derive ``identity-mismatch`` from the shared
    kernel, or the captured pair cannot witness that precedence.
    """

    from menagerie.crawler.constants import metadata_gate_failure_reason

    adverse = [
        item
        for item in items
        if item.get("verdict") != "accurate"
        or (item.get("rung_check") or {}).get("verdict") != "accurate"
    ]
    if not adverse:
        return False
    first = adverse[0]
    if (first.get("integrity") or {}).get("verdict") == "accurate":
        return False
    try:
        return metadata_gate_failure_reason(first) == "identity-mismatch"
    except Exception:  # noqa: BLE001 -- an underivable reason is a non-match
        return False


def _capture_richest(pool: list[dict[str, Any]], manifest: dict[str, Any], dry_run: bool) -> None:
    """Capture the richest-of-class entries the role policy cannot reach.

    ``metadata_mixed_disputed`` is the widest real metadata session carrying
    BOTH per-item verdict heterogeneity and at least one disputed-integrity
    item: cap-width batching, per-item routing, and unattestable evidence in one
    recorded adjudication. ``terminal_disputed`` is the terminal lane's
    disputed-evidence arm, which no smallest-of-role pair carries.

    Selection maximizes richness (verdict spread, then integrity spread, then
    batch width) so the entry keeps covering the widest real shape available
    even when a later campaign records more of them; ties break on the smallest
    byte size, since these entries are the corpus's largest.

    Candidates over ``MAX_ENTRY_BYTES`` are dropped BEFORE that ranking, and what
    they cost is logged rather than swallowed: the richest real pair on this tree
    carried all three integrity verdicts at 614 KB and could not have been
    committed at all. Silently emitting an unstageable entry, or widening the
    repo's large-file guard to fit a fixture, are both worse than taking the
    richest pair that fits and saying which shape that gave up.
    """

    def richest(entries: list[dict[str, Any]]) -> dict[str, Any]:
        return max(
            entries,
            key=lambda entry: (
                _spread(entry["result"]["items"], "verdict"),
                _spread(entry["result"]["items"], "integrity"),
                len(entry["result"]["items"]),
                -entry["size"],
            ),
        )

    written_pairs: dict[str, str] = {}
    classes: tuple[tuple[str, Any], ...] = (
        (
            "metadata_mixed_disputed",
            # The cap-width per-item-routing golden's contract
            # (test_golden_cap_width_batch_routes_per_item_not_per_batch): a
            # CAP-WIDTH session whose fully-accepted set (verdict AND integrity
            # AND rung all accurate) is non-empty beside at least one
            # disputed-integrity item -- verdict spread alone admits a batch
            # with zero accepted siblings, which that test rejects as vacuous.
            lambda entry: (
                entry["kind"] == "metadata_batch"
                and len(entry["result"]["items"]) == METADATA_BATCH_MAX
                and any(_integrity_disputed(item) for item in entry["result"]["items"])
                and any(
                    item.get("verdict") == "accurate"
                    and (item.get("integrity") or {}).get("verdict") == "accurate"
                    and (item.get("rung_check") or {}).get("verdict") == "accurate"
                    for item in entry["result"]["items"]
                )
            ),
        ),
        (
            "terminal_disputed",
            lambda entry: (
                entry["kind"] == "terminal_disposition"
                and any(_integrity_disputed(item) for item in entry["result"]["items"])
            ),
        ),
        # A session mixing ACCEPTED with non-accepted items -- the per-item route
        # independence witness. Distinct from ``metadata_mixed_disputed``, whose
        # spread is over the raw verdict vocabulary: a batch can spread
        # inaccurate/cannot-verify without carrying a single accepted sibling,
        # and it is the accepted-beside-rejected pair that pins no verdict
        # bleeding across items in one session.
        (
            "metadata_mixed_batch",
            lambda entry: (
                entry["kind"] == "metadata_batch"
                and {True, False}
                <= {_item_accepted(entry["kind"], item) for item in entry["result"]["items"]}
            ),
        ),
        # An adverse RUNG check. The pilot's dominant adverse class is
        # cannot-verify, and a rung dispute beside an integrity dispute is the
        # shape behind the recorded ``identity-mismatch`` terminals (m4066).
        (
            "metadata_rung_adverse",
            # The kernel-reason golden's contract
            # (test_golden_rung_adverse_item_derives_the_kernel_reason): the
            # FIRST adverse item (verdict or rung_check not accurate, in item
            # order) must itself carry a disputed integrity verdict and derive
            # ``identity-mismatch`` from the one shared kernel.
            lambda entry: (
                entry["kind"] == "metadata_batch"
                and _rung_adverse_matches_kernel_contract(entry["result"]["items"])
            ),
        ),
        # Not capturable from the pilot tree as of 2026-08-12 (no surviving
        # cannot-verify terminal pair); the class is declared so the next
        # campaign leg that records one folds it in with zero code changes.
        (
            "terminal_cannot_verify",
            lambda entry: (
                entry["kind"] == "terminal_disposition"
                and any(
                    (item.get("terminal_disposition") or {}).get("verdict") == "cannot-verify"
                    for item in entry["result"]["items"]
                )
            ),
        ),
    )
    for name, predicate in classes:
        shaped = [entry for entry in pool if predicate(entry)]
        qualifying = [entry for entry in shaped if _entry_bytes(entry) <= MAX_ENTRY_BYTES]
        for dropped in sorted(
            (entry for entry in shaped if entry not in qualifying),
            key=_entry_bytes,
        ):
            print(
                f"  {name}: dropped {Path(str(dropped['source']['result'])).parent.name} "
                f"({round(_entry_bytes(dropped) / 1024)} KB > "
                f"{MAX_ENTRY_BYTES // 1024} KB ceiling), "
                f"verdicts={_spread(dropped['result']['items'], 'verdict')} "
                f"integrity={_spread(dropped['result']['items'], 'integrity')} "
                f"n={len(dropped['result']['items'])}"
            )
        if not qualifying:
            print(f"MISSING corpus class: {name} (no recorded pair of this shape)")
            continue
        # One rich pair routinely satisfies SEVERAL of these classes -- a
        # cap-width session with a verdict spread and a disputed item answers
        # mixed_disputed, mixed_batch and rung_adverse at once -- so the richest
        # pick must be deduped against what earlier classes already took.
        # Without this the corpus silently collapses onto one pair, paying ~480
        # KB per duplicate for zero new witness. Preferring the richest pair not
        # yet taken keeps the classes on DISTINCT recorded sessions where the
        # traffic allows, and only when every qualifying pair is already
        # committed does the class report itself as covered rather than
        # duplicating bytes.
        untaken = [
            entry for entry in qualifying if str(entry["source"]["result"]) not in written_pairs
        ]
        if not untaken:
            covered_by = written_pairs[str(richest(qualifying)["source"]["result"])]
            print(f"{name}: covered by {covered_by} (same recorded pair)")
            continue
        chosen = richest(untaken)
        written_pairs[str(chosen["source"]["result"])] = name
        manifest[name] = _write_entry(name, chosen, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "report the pair each class would capture without rewriting the "
            "committed corpus -- how to check a re-record before it lands"
        ),
    )
    parser.add_argument(
        "--checker-model",
        default=None,
        help=(
            "only capture pairs recorded under this checker model (the pinned "
            "adjudication authority); a post-rotation re-record must converge "
            "on it or retired-model pairs win smallest-of-class"
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "comma-separated class names to (re-)capture, leaving every other "
            "committed entry untouched -- the additive-capture mode; richest-of-"
            "class and drift-exemplar rewrites are skipped unless named"
        ),
    )
    args = parser.parse_args()
    capture(
        args.campaign_root.resolve(),
        dry_run=args.dry_run,
        checker_model=args.checker_model,
        only=(
            {name.strip() for name in args.only.split(",") if name.strip()}
            if args.only
            else None
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
