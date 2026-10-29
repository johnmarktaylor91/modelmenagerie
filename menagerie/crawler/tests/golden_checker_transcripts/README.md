# Golden checker transcript corpus

Real recorded checker traffic, NOT test-builder output. Each `*.json` entry is
one exact frozen `request.json`/`result.json` pair the production checker
dispatch wrote and the pinned checker model (gpt-5.6-sol, every lane) answered,
captured verbatim from a campaign work tree and tied to its durable
gates-ledger row (`ledger_row` / `ledger_gate_id`). The metadata entries are
recorded at the PRODUCTION batch width (`METADATA_BATCH_MAX`, width 1 since the
2026-08-29 rotation): a pair recorded at another width measured a different
lane, cannot re-seal through the current envelope builder, and is refused by
the capture tool.

Consumed by `../test_checker_envelope_golden.py` (production
`validate_checker_result` + routing over real transcripts, plus the
armed-tripwire corruption goldens), `../test_gate_reuse_semantic_drift.py`
(REAL retired identities for the envelope-drift arms), and the B1-XV2
blind-replay matrix (`tools/blind_replay_matrix.py` /
`tools/checker_calibration_runner.py`), whose replay of every committed entry
is the calibration receipt's acceptance instrument.

Entries:

| file | lane | recorded outcome |
|---|---|---|
| `metadata_accept.json` | metadata_batch (sol, width 1) | single-item batch, accurate (m10470; replays as a DETERMINISTIC non-accept under the round-3 lineage rule -- kept as the recorded adjudication and abstention measurement) |
| `metadata_accept_lineage_grounded.json` | metadata_batch (sol, width 1) | single-item batch, accurate (m10949; additive round-3 accept exemplar: every excerpt machine-attests valid, new question) |
| `metadata_refusal.json` | metadata_batch (sol, width 1) | recorded rejection |
| `metadata_rung_adverse.json` | metadata_batch (sol, width 1) | adverse rung check (the reason kernel's identity-mismatch arm) |
| `metadata_component_drift.json` | metadata_batch (sol, width 1) | real pair the CURRENT validator refuses purely for retired component identity |
| `fidelity_accept.json` | fidelity (sol) | match + accurate rung |
| `fidelity_refusal.json` | fidelity (sol) | recorded fidelity block |
| `terminal_accept.json` | terminal_disposition (sol) | accepted disposition |
| `terminal_refusal.json` | terminal_disposition (sol) | rejected disposition |
| `terminal_disputed.json` | terminal_disposition (sol) | rejected disposition over evidence the checker could NOT re-attest |

## Width-retired classes

`metadata_mixed_batch` (accepted-beside-rejected in one session) and
`metadata_mixed_disputed` (cap-width heterogeneity with a disputed item) are
MULTI-ITEM shapes: at the calibrated width-1 production config no recorded
session can ever populate them, so their entries are retired from the
committed corpus. The capture tool keeps their class predicates and
`test_checker_envelope_golden.py` keeps their tests behind a
`METADATA_BATCH_MAX < 2` skip -- any future width increase re-records the
classes and re-arms both in the same reviewed change. The disputed-integrity
coverage floor stays armed at every width (it now scans all metadata entries
instead of the retired mixed_disputed exemplar).

One corpus entry must also stay under the repo's `check-added-large-files`
pre-commit ceiling (`MAX_ENTRY_BYTES` in the capture tool); the tool drops
over-ceiling candidates before ranking and logs what that gave up, rather than
writing an entry that cannot be committed.

Envelope bodies are hash-bound (`envelope_sha256`), so nothing inside an entry
may be edited -- including the capture-host absolute paths they embed. To
refresh the corpus (checker rotation, width rotation, new transcript shapes),
re-run `capture_golden_checker_transcripts.py --checker-model <pinned model>`
against a campaign tree and commit the rewritten files (two-pass where richer
traffic exists elsewhere: pilot pass first, then the calibration root so its
pairs overwrite shared classes). Check a re-record with `--dry-run` FIRST: it
prints the pair every class would take without touching the committed corpus.
