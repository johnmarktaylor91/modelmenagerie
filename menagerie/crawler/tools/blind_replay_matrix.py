"""B1-XV2 blind-replay matrix: W2's calibration acceptance instrument.

Every golden checker question is replayed >= 3x under the SAME model/prompt
epoch (zero disagreement required), once more after a DELIBERATE rotation
(its disagreement is recorded, not constrained -- rotation may lawfully
flip), and across batch widths (width 1 vs the production width must agree:
a verdict that changes with batch width is not an adjudication of the item).
Latencies of every production-width call feed the receipt's
``p99_at_production_width_seconds``, measured BEFORE the receipt is minted.

This module owns the matrix ORCHESTRATION, the verdict-projection comparison,
and the report artifact (``menagerie.crawler.blind-replay-report.v2``, the
exact input :func:`menagerie.crawler.checker_calibration_mint.
derive_blind_replay_block` re-derives the receipt block from). The
adjudicator is INJECTED: one callable per identity epoch that answers one
request with a verdict projection and a latency. The R6 calibration leg binds
it to the real pinned checker (the production envelope builders + the
operator wrapper subprocess, under ``MENAGERIE_CHECKER_CALIBRATION_RUN=1``);
tests bind scripted adjudicators. The harness never reads env, never buys
sessions itself, and never mutates the corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from menagerie.crawler.checker_calibration import CALIBRATION_PRODUCTION_BATCH_WIDTH
from menagerie.crawler.checker_calibration_mint import BLIND_REPLAY_REPORT_SCHEMA
from menagerie.crawler.models import JsonObject

#: One adjudication answer: the per-item verdict projection this harness
#: compares, plus the session latency. ``verdicts`` maps stable_id -> the
#: item's verdict-relevant projection (any JSON-comparable value).
Adjudication = Mapping[str, Any]

#: An adjudicator answers one golden question (identified by name + kind +
#: batch width) with an :data:`Adjudication`.
Adjudicator = Callable[[Mapping[str, Any]], Adjudication]


class BlindReplayMatrixError(ValueError):
    """Raised on a malformed matrix input or adjudicator answer."""


def _projection(answer: Any, *, context: str) -> tuple[JsonObject, float, JsonObject]:
    """Validate one adjudicator answer, fail-closed.

    Returns the canonical projection, the latency, and the v2 per-draw
    disclosure extras (result digest, session root, typed integrity-dispute
    count) with fail-closed defaults for scripted adjudicators.
    """

    if not isinstance(answer, Mapping):
        raise BlindReplayMatrixError(f"{context}: adjudicator answer must be a mapping")
    verdicts = answer.get("verdicts")
    if not isinstance(verdicts, Mapping) or not verdicts:
        raise BlindReplayMatrixError(f"{context}: adjudicator answer carries no verdict projection")
    latency = answer.get("latency_seconds")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0:
        raise BlindReplayMatrixError(
            f"{context}: adjudicator answer carries no non-negative latency_seconds"
        )
    canonical = json.loads(json.dumps(dict(verdicts), sort_keys=True))
    extras: JsonObject = {
        "result_sha256": answer.get("result_sha256"),
        "session_root": answer.get("session_root"),
        "integrity_dispute_entries": int(answer.get("integrity_dispute_entries") or 0),
    }
    return canonical, float(latency), extras


def golden_item_excerpts(kind: str, item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return one golden item's excerpt rows for locator attestation.

    Terminal packs carry machine re-derived excerpts in ``evidence_pack``;
    metadata/fidelity packs carry the proposal's authored evidence excerpts.
    Both share the ``{evidence_id, source_id, locator, text}`` projection the
    attestation consumes.
    """

    if kind == "terminal_disposition":
        pack = item.get("evidence_pack")
        excerpts = pack.get("excerpts") if isinstance(pack, Mapping) else None
    else:
        proposal = item.get("proposal")
        facts = proposal.get("proposed_facts") if isinstance(proposal, Mapping) else None
        evidence = facts.get("evidence") if isinstance(facts, Mapping) else None
        excerpts = evidence.get("excerpts") if isinstance(evidence, Mapping) else None
    return [row for row in (excerpts if isinstance(excerpts, list) else []) if isinstance(row, Mapping)]


def derive_golden_locator_attestation(entry: Mapping[str, Any]) -> JsonObject:
    """Freeze one golden's machine locator attestation rows for the report.

    The same derivation the envelope builder stamps at dispatch (one
    spelling, deferred import to keep this harness light for tests).
    """

    from menagerie.crawler.checker_dispatch import _derive_locator_attestation

    request = entry.get("request")
    if not isinstance(request, Mapping):
        # The request is opaque to this harness (scripted-adjudicator tests
        # pass sentinels); an item-less request has zero excerpts to attest,
        # honestly disclosed as an empty row set -- the receipt's corpus-total
        # zero bar still refuses a whole corpus that attests nothing.
        return {"rows": [], "licensed": False}
    rows: list[JsonObject] = []
    licensed = False
    for item in request.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        attestation = _derive_locator_attestation(
            golden_item_excerpts(str(entry.get("kind")), item),
            item.get("source_manifest"),
            model_dir=item.get("model_dir"),
        )
        rows.extend(attestation["rows"])
        licensed = licensed or not attestation["all_valid"]
    return {"rows": rows, "licensed": licensed}


def run_blind_replay_matrix(
    goldens: Sequence[Mapping[str, Any]],
    adjudicate: Adjudicator,
    *,
    rotation_adjudicate: Optional[Adjudicator] = None,
    replays_per_golden: int = 3,
    production_batch_width: int = CALIBRATION_PRODUCTION_BATCH_WIDTH,
    checker_model: str,
    captured_at: str,
) -> JsonObject:
    """Run the full B1-XV2 matrix and return the raw-facts report.

    Parameters
    ----------
    goldens:
        One entry per golden question:
        ``{"name": str, "kind": str, "width": int, "request": <opaque>,
        "width_variants": {int: <opaque request>} | absent}``. The ``request``
        objects are passed to the adjudicator verbatim (opaque to the
        harness); ``width`` declares the batch width the base request carries.
    adjudicate:
        The same-epoch adjudicator (pinned model + prompt).
    rotation_adjudicate:
        The post-rotation adjudicator (a DIFFERENT identity epoch), or
        ``None`` when the rotation leg has not run -- the report then records
        ``rotation.ran = False`` per golden and the receipt validator refuses,
        which is exactly the fail-closed default.
    replays_per_golden:
        Same-epoch replay count (the B1-XV2 floor is 3; the receipt validator
        enforces it, this harness just runs what it is asked).
    production_batch_width, checker_model, captured_at:
        Report metadata; production-width latencies aggregate into
        ``latency_seconds_at_production_width``.

    Returns
    -------
    dict[str, Any]
        A ``menagerie.crawler.blind-replay-report.v2`` document of per-golden
        RAW facts. Aggregates are deliberately absent: the receipt mint
        re-derives them (zero producer-written summary to trust).
    """

    if not goldens:
        raise BlindReplayMatrixError("a blind-replay matrix over zero goldens attests nothing")
    rows: list[JsonObject] = []
    production_latencies: list[float] = []
    for golden in goldens:
        name = str(golden.get("name"))
        kind = str(golden.get("kind"))
        width = int(golden.get("width", 1))
        request = golden.get("request")
        attestation = derive_golden_locator_attestation(golden)
        licensed = bool(attestation.pop("licensed"))
        projections: list[JsonObject] = []
        draws: list[JsonObject] = []
        for replay in range(int(replays_per_golden)):
            projection, latency, extras = _projection(
                adjudicate(golden), context=f"golden {name!r} replay {replay + 1}"
            )
            projections.append(projection)
            draws.append(
                {
                    "ask": f"{name}:se{replay + 1}",
                    "mode": "same_epoch",
                    "projection": projection,
                    "latency_seconds": latency,
                    "licensed": licensed,
                    **extras,
                }
            )
            if width == production_batch_width:
                production_latencies.append(latency)
        row: JsonObject = {
            "name": name,
            "kind": kind,
            "width": width,
            "request": request if isinstance(request, (str, int, float)) else None,
            "locator_attestation": attestation,
        }
        if rotation_adjudicate is not None:
            rotated, latency, extras = _projection(
                rotation_adjudicate(golden), context=f"golden {name!r} rotation replay"
            )
            draws.append(
                {
                    "ask": f"{name}:rot",
                    "mode": "rotation",
                    "projection": rotated,
                    "latency_seconds": latency,
                    "licensed": licensed,
                    **extras,
                }
            )
            row["rotation"] = {
                "ran": True,
                "projection": rotated,
                "disagreement": rotated != projections[0],
            }
        else:
            row["rotation"] = {"ran": False}
        variants = golden.get("width_variants")
        if isinstance(variants, Mapping) and variants:
            # The arm is DEGENERATE when no variant carries a genuinely
            # different width: its "comparison" re-asks the same request, so
            # each extra draw is a same-epoch draw and folds there (Opus 6.6).
            degenerate = all(int(value) == width for value in variants)
            variant_projections: dict[str, JsonObject] = {str(width): projections[0]}
            disagreement = False
            for variant_width, variant_request in sorted(variants.items()):
                variant_golden = dict(golden)
                variant_golden["width"] = int(variant_width)
                variant_golden["request"] = variant_request
                projection, latency, extras = _projection(
                    adjudicate(variant_golden),
                    context=f"golden {name!r} width {variant_width}",
                )
                draws.append(
                    {
                        "ask": f"{name}:w{variant_width}",
                        "mode": "width_variant",
                        "projection": projection,
                        "latency_seconds": latency,
                        "licensed": licensed,
                        **extras,
                    }
                )
                if int(variant_width) == production_batch_width:
                    production_latencies.append(latency)
                variant_projections[str(variant_width)] = projection
                if degenerate:
                    # At production width 1 the "variant" is one more real
                    # session on the same single-item request: a SAME-EPOCH
                    # draw. It folds into the same-epoch projections, where a
                    # flip counts against the zero bar (Opus 6.6), and the
                    # invariance arm is declared degenerate rather than
                    # presented as a vacuously passed proof.
                    projections.append(projection)
                    continue
                # Width invariance compares the verdicts of the ITEMS the two
                # widths share; extra batch-mates are not part of the question.
                shared = set(projection) & set(projections[0])
                if not shared or any(projection[key] != projections[0][key] for key in shared):
                    disagreement = True
            row["width_invariance"] = {
                "widths": (
                    [width]
                    if degenerate
                    else sorted({width, *(int(value) for value in variants)})
                ),
                "projections": variant_projections,
                "disagreement": False if degenerate else disagreement,
                "degenerate_at_production_width": degenerate,
                "folded_into_same_epoch": degenerate,
            }
        row["same_epoch"] = {
            "replays": len(projections),
            "projections": projections,
            "disagreement": any(
                projection != projections[0] for projection in projections[1:]
            ),
        }
        row["draws"] = draws
        rows.append(row)
    return {
        "schema_version": BLIND_REPLAY_REPORT_SCHEMA,
        "checker_model": checker_model,
        "captured_at": captured_at,
        "production_batch_width": int(production_batch_width),
        "goldens": rows,
        "latency_seconds_at_production_width": production_latencies,
        # The in-tree harness treats an adjudicator failure as fatal, so a
        # completed report has zero typed authority conflicts by construction;
        # journal-based orchestrators surface theirs here.
        "authority_conflicts": [],
    }


def load_golden_corpus_entries(corpus_dir: Path) -> list[JsonObject]:
    """Load the committed golden transcripts as matrix entries.

    Each committed transcript's frozen ``request`` envelope becomes the
    opaque question; ``width`` derives from the envelope's item count. The
    R6 adjudicator re-dispatches these envelopes (fresh deadline/output
    path, same items/prompt) through the production wrapper.
    """

    entries: list[JsonObject] = []
    for path in sorted(corpus_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BlindReplayMatrixError(f"golden transcript {path} is unreadable: {exc}")
        if not isinstance(data, Mapping):
            continue
        request = data.get("request")
        items = request.get("items") if isinstance(request, Mapping) else None
        entries.append(
            {
                "name": path.stem,
                "kind": str(data.get("kind")),
                "width": len(items) if isinstance(items, list) and items else 1,
                "request": request,
            }
        )
    if not entries:
        raise BlindReplayMatrixError(f"golden corpus {corpus_dir} carries no transcripts")
    return entries


def write_report(report: Mapping[str, Any], path: Path) -> Path:
    """Write one matrix report artifact canonically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path
