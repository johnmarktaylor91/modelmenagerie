# Checker calibration leg (the sanctioned calibration-mode run)

The ONE legitimate use of `MENAGERIE_CHECKER_CALIBRATION_RUN=1`. It exists to
measure the pinned checker on its production contract and mint the activation
receipt (`menagerie/crawler/assets/checker-calibration-receipt.json`) that the
command lane requires before buying any real session. Everything below is
fail-closed: an under-measured, over-refusing, timing-incompatible, or
non-re-derivable receipt refuses at mint or verify, never activates.

## Invariants (enforced in code, restated here)

- **Calibration mode withholds awards structurally.** While the flag is set,
  the reducer refuses any `runs` award head, award restoration, or
  release-eligibility marking (`CheckerCalibrationWithheldError`), every leg
  attests the flag in its leg-config row, and every verdict rendered without
  receipt coverage lands in the `gates_adjudicated_under_calibration_bypass`
  census arm. Nothing a calibration leg renders is award- or
  release-eligible; re-run anything you need to cite under the minted receipt.
- **Receipt XOR mode.** A verdict-rendering launch passes the
  `checker-calibration-mode` preflight arm with a valid receipt XOR the
  declared calibration mode — never both, never neither. Verdict-free legs
  declare `--renders-no-gate-verdicts` (vacuous pass, disclosed).
- **Never lower `CALIBRATION_MIN_SESSIONS`.** The floors (20/5/5) are the
  measurement contract. Plan buys for the observed verdict yield (200a
  lifetime: 7 rendered / 11 bought), so ~16–18 metadata buys with a
  stop-at-21-rendered-rows condition clears the 20-floor under any counting
  rule.
- **Timing constants are derivation-backed.** The receipt refuses any lane
  whose measured completed-attempt p99 exceeds `OPERATOR_ATTEMPT_TIMEOUT_SECONDS`
  or whose session p99 exceeds `CHECKER_SESSION_WALL_SECONDS` (the 600s
  lesson). If it refuses, re-derive the constant from the measurement in the
  same reviewed change; never hand-set a wall below the evidence.

## Procedure

1. **Run the calibration leg** over frozen highest-value pending packs at
   PRODUCTION batch width (metadata batches of `METADATA_BATCH_MAX`),
   stratified across long-pack, disputed, censored, line-ending, and
   native-extension members:

   ```bash
   MENAGERIE_CHECKER_CALIBRATION_RUN=1 <the leg launch command>
   ```

   Buy until every lane clears its floor in RENDERED verdicts (gate rows).
   The wall doubles as adjudication practice, but nothing it renders may be
   cited before a re-run under the receipt (bypass-census rule above).

2. **Run the blind-replay matrix** (B1-XV2, the acceptance instrument):
   every golden replayed 3x under the same model/prompt epoch with zero
   disagreement, a 4th replay after a deliberate rotation (recorded, not
   constrained), and width-1 vs production-width invariance. Orchestrate with
   `menagerie.crawler.tools.blind_replay_matrix.run_blind_replay_matrix`
   (adjudicator = the production envelope builders + the operator wrapper
   subprocess, still under the calibration flag) and write the report with
   `write_report`. The production-width latencies in the report become the
   receipt's `p99_at_production_width_seconds`, measured BEFORE the mint.

3. **Re-record the goldens** from the leg's frozen request/result pairs:

   ```bash
   PYTHONPATH=. python menagerie/crawler/tests/golden_checker_transcripts/\
   capture_golden_checker_transcripts.py --campaign-root <campaign> --dry-run
   # inspect, then re-run without --dry-run
   ```

   The corpus must end sol-only for every lane; the activation gate refuses a
   receipt whose goldens still record a retired checker.

4. **Mint the receipt** (tool-minted, never hand-written):

   ```bash
   PYTHONPATH=. python -m menagerie.crawler.tools.calibration_receipt mint \
       --records-root <campaign>/menagerie/crawler/records \
       --work-root <campaign>/.crawl-local/work \
       --checker-model gpt-5.6-sol \
       --blind-replay-report <report.json> \
       --source "<leg identifier>" \
       --next-rung-models <N> \
       --projected-wall-hours <..> --wall-budget-hours <..> \
       --projected-sessions <..> --session-budget <..> \
       --projected-dollars <..> --dollar-budget <..> \
       --out menagerie/crawler/assets/checker-calibration-receipt.json
   ```

   Budgets/projections are the declared capacity plan; the capacity governor
   enforces the 0.8 margin at load, and the mint round-trips the written file
   through the activation loader (a refused receipt is deleted, not parked).

5. **Verify** (the leg's exit instrument — recompute-and-refuse):

   ```bash
   PYTHONPATH=. python -m menagerie.crawler.tools.calibration_receipt verify \
       --receipt menagerie/crawler/assets/checker-calibration-receipt.json \
       --records-root <...> --work-root <...> \
       --blind-replay-report <report.json>
   ```

6. **Land receipt + re-recorded goldens in the SAME reviewed change**, with
   any timing-constant re-derivation the receipt demanded. Unset the
   calibration flag; the next launch passes the XOR arm on the receipt side.
