# Ladder protocol: running rung after rung

A **rung** is one ladder step of a campaign: a content-addressed intake subset of
the campaign's bound full-partition snapshot, registered in
`records/intake/registered-rungs.jsonl` with explicit `derived_from_snapshot_id`
lineage. The ladder walks widening rungs (`mini-10`, `step-100`, `step-200`, ...)
so a surprise class costs one rung, not one full run.

Two rules set the shape of everything below:

* **Rung 1 is a prefix; every rung from `step-200` onward is STRATIFIED** --
  deliberately salted with tail-class representatives (2026-08-11 directive), so
  surprise classes surface at rung width. See `rung_stratification.py`.
* **A rung is bound to exactly one campaign spec.** Staging goes through
  `stage_rung`, never by hand: an unbound snapshot silently adopts the default
  `DriverConfig` -- the wrong author model for any campaign whose binding differs
  (T4-T2), and every consumer refuses it.

## 0. Before the first rung of a campaign

The intake-currency carve-out must be present or staging refuses typed
(`CurrencyCarveOutMissingError`): `menagerie.crawler.authority` must declare
`PRIOR_SNAPSHOT_TERMINAL_CURRENCY_CARVE_OUT = True`. Without it a new active
snapshot id projects every prior terminal dependency-stale and the ladder cost
goes quadratic (T4-T1).

## 1. Compose the rung

`stage_rung` takes members two ways, and which one you use is the ladder step:

```bash
# Rung 1 -- the unstratified partition prefix.
PYTHONPATH=. python -m menagerie.crawler.tools.stage_rung stage \
  --records-root menagerie/crawler/records --campaign c1-mech \
  --count 100 --label step-100
```

```bash
# step-200 and every rung after it -- stratified, two commands.
PYTHONPATH=. python -m menagerie.crawler.tools.stratify_rung \
  --records-root menagerie/crawler/records --campaign c1-mech \
  --size 200 --seed 200 \
  --out-members step-200-members.txt --out-report step-200-report.json

# Every pair the report lists under award_authorities_required, verbatim.
AUTHORITIES=$(python - <<'PY'
import json
report = json.load(open("step-200-report.json"))
print(" ".join(
    f"--award-authority {stable_id}={revision}"
    for stable_id, revision in sorted(report["award_authorities_required"].items())
))
PY
)

PYTHONPATH=. python -m menagerie.crawler.tools.stage_rung stage \
  --records-root menagerie/crawler/records --campaign c1-mech \
  --label step-200 --stable-ids-file step-200-members.txt \
  --composition-attestation step-200-report.json $AUTHORITIES
```

**`--size` is the NEW width, and the member list is CUMULATIVE.** A rung is a
successor of the campaign's registry tip and must strictly widen it, so its
membership is everything the tip already proved plus what this composition
bought. The composer carries the tip explicitly as the `carried-tip` stratum
(C-2): the report's `carried_tip` block names the tip, its `member_ids`, the
`member_count`, which of them this rung `carried_forward`, and which a tail
stratum selected on its own evidence (a re-attempt of the hard tail keeps its own
evidence -- it is carried either way). Those labels ride inside the composition
digest, so the carry is attested, not merely reported, and the staging door
refuses any member CLAIMING `lineage-tip` provenance the registry tip does not
contain. Carrying is mandatory: `carried-tip` takes no `--share`, cannot be
acknowledged away, and is never backfilled from `ladder-bulk`.

A carried award is still operator property, so the AW-2 door demands an
`--award-authority stable_id=revision` for every carried `runs` holder, naming
its exact current revision. `award_authorities_required` in the report is that
list, derived by the door's own predicate -- pass it through like the
attestation, never a hand tally.

Pass the report through with `--composition-attestation` -- never a hand-copied
member list or snapshot sha. The report carries the composer's attestation
(campaign, member/stratum pairs, and the `(snapshot_id, snapshot_sha256)` those
members compose to); staging recomputes all three from what it is actually about
to stage and refuses on any mismatch (B-03). The S200-PREP incident is the reason:
a snapshot sha carried forward by hand does not produce a corrupt snapshot, it
produces a perfectly valid one that is not the rung anybody reviewed. Both doors
accept the flag (`register` too), and the check runs before any durable byte
exists, so a refusal leaves no orphan snapshot behind.

Record the `--seed` with the label: `(records state, seed, size)` determines the
rung exactly. Both commands are read-only against the ledgers and run in a plain
repo checkout -- no scientific runtime needed (pinned by
`test_rung_boundary_protocol.py::test_staging_tools_import_without_the_fat_runtime`).

### When the stratifier refuses

* **`EmptyStratumError`** -- a stratum with a positive share has zero
  candidates. This is the point of the salting: a vanished class is information.
  Investigate what emptied it. When it is genuinely gone, acknowledge it:

  ```bash
  ... --acknowledge-empty-stratum r2-vendored --acknowledge-empty-stratum cuda-deferred
  ```

  Each acknowledgement is PROVEN empty against the current records state (a live
  stratum refuses with `StratumNotEmptyError`), zeroes that stratum's share,
  moves the reclaimed share to `ladder-bulk`, and is disclosed in the report's
  `acknowledged_empty_strata`. Do NOT reach for `--share <stratum>=0`: shares
  must still sum to 1.0, so that alone earns a second refusal. Expect several
  acknowledgements on a campaign's FIRST stratified rung -- four tail strata
  (`r2-vendored`, `cuda-deferred`, `content-refusal`, `hard-tail`) are derived
  from prior heads and attempts, so they are empty until a rung has run.
* **"the campaign partition holds only N fresh rows"** -- ladder bulk is
  exhausted. Reduce `--size`; the campaign is near its end.
* **any records-read refusal** (torn ledger, unverifiable prior rung) -- staging
  happens between rungs, so an unreadable input is a surprise to surface, never
  to skip.

## 2. Preflight, then run the rung

A rung's members span whichever scheduler phases their routing derives, which is
NOT the campaign's declared phase. `c1-mech` declares `pytorch` and its partition
carries 141 `paddle-arm64` rows (the ERNIE / PaddleOCR class); `c3-classics`
carries 120. A plain `--count 100` prefix of `c1-mech` already spans both phases
(7 of its first 100 rows are native-tail).

**So run a rung with `--phase` OMITTED.** `_ordered_work` routes the whole
snapshot and enforces global phase order regardless, so one invocation walks
pytorch then native-tail and closes the WHOLE rung:

```bash
INTAKE=menagerie/crawler/records/intake/<rung-snapshot-id>
CRAWLER_RUNTIME="$(pwd)/.crawl-local"

PYTHONPATH=. python -m menagerie.crawler.tools.launch_preflight \
  --records-root menagerie/crawler/records --envs-root menagerie/crawler/envs \
  --env-root "$CRAWLER_RUNTIME/envs" \
  --state-root "$CRAWLER_RUNTIME/environment-operator" \
  --tmp-root "$CRAWLER_RUNTIME/tmp"

python -m menagerie.crawler plan --intake "$INTAKE" --target osx-arm64
python -m menagerie.crawler run  --intake "$INTAKE" --target osx-arm64 --sequential-envs
```

`launch_preflight` is MANDATORY before any leg that promises to release rebuild
parks, and refuses (`rotation-did-not-rotate`) when a promised rotation did not
actually rotate the four committed setup artifacts. Full detail: `RUN.md`.

Passing `--phase` narrows both the work set and the completion close to that one
phase. That is the right spelling for a whole-CAMPAIGN two-stage crawl (`RUN.md`
phase 1 / phase 2) and the wrong one for a rung, because it leaves the other
phase's members unworked while `status --verify-partition` -- which is NOT
phase-scoped -- reports them in `missing`. If you do run a rung one phase at a
time, `--phase native-tail` refuses outright while any pytorch row is still
unworked, and `rung_report`'s `legs_owed` is what tells you a leg remains.

## 3. The rung boundary: tally and spend

```bash
PYTHONPATH=. python -m menagerie.crawler.tools.rung_report \
  --records-root menagerie/crawler/records --campaign c1-mech --label step-200
```

One read-only report, scoped to one registered rung whose membership is
re-verified against its registered digest:

* `phases.<phase>` -- members, `by_status`, `converted`, `valve_claimed`,
  `unaccounted` (+ ids), and `closable`. A phase is closable when nothing is
  unaccounted: the driver's own partition rule.
* `legs_owed` -- every phase not yet closable. **Empty `legs_owed` is the rung
  boundary.** A phase-omitted run should empty it in one invocation; a non-empty
  `legs_owed` after such a run is a real stop, not a scoping artifact.
* `spend` -- `author_sessions_used`, `gate_rounds_used`, `explicit_grants` from
  the rung's own model heads; `checker.sessions_bought` vs `verdicts_reused`;
  attempts and wall seconds by stage.
* `valves` -- per valve ledger, `accountable_members` and `rejected_rows` kept
  SEPARATE. A row the completion partition's own check rejects claims nothing
  here either, so a malformed valve row can never make a rung look closable.
* `records_state` -- content digests of every ledger read, so a tally is
  re-derivable and comparable rung over rung.

Read alongside it, for trend rather than tally:

```bash
PYTHONPATH=. python -m menagerie.crawler.tools.throughput_report --campaign c1-mech=.
PYTHONPATH=. python -m menagerie.crawler.tools.checker_latency --help
```

`throughput_report` is campaign-scoped, not rung-scoped: use it for the R5
positive-slope alert (per-model final-authority time growing with terminal
count), which is a decision about whether to widen the next rung at all.

## 4. Clear the operator decisions the rung parked

A rung does not close over live operator debt. Each of these lists, and voids
without rewriting history:

```bash
PYTHONPATH=. python -m menagerie.crawler.tools.grant_parks --help    # review holds
PYTHONPATH=. python -m menagerie.crawler.tools.refusal_parks --help  # terminal-refusal parks
PYTHONPATH=. python -m menagerie.crawler.tools.requeue --help        # bounded effort grants
PYTHONPATH=. python -m menagerie.crawler.tools.reap_disk --help      # report-only disk GC
```

`status` surfaces live `review_holds` and `refusal_parks` unconditionally, so
nobody has to know to ask.

## 5. Widen: back to step 1

**The next rung crosses the boundary on the tip's close object, and on nothing
else.** Once a campaign has any registered rung, both staging doors demand an
immutable close for the campaign's REGISTRY TIP -- the lineage comes from the
state, so `--derived-from` naming the campaign root or an older rung refuses
instead of routing around the gate. The close is bound to that tip by campaign
id and by its registered snapshot digest, and (close-object v2, 2026-08-16
tri-lab contract) it binds by digest a generation-scoped sealed
`PartitionProofV2` that the gate RECOMPUTES from the pinned bytes: every tip
member cites ONE exact evidence item -- a head recorded IN THIS LEG'S WINDOW
and recorded-under this snapshot, a CARRIED delegation by digest to a named
ancestor close that resolved it, or a LIVE valve/park row appended in this
leg's window. The leg window is a half-open interval over the ONE source
census (every models/attempts/gates/artifacts layout plus the seven valve/park
ledgers, each slot attested present-or-absent), floored by the durable
`leg_open` row in `records/intake/leg-events.jsonl`, prefix-hash verified, and
anchored against the campaign's previous close so a forged floor proves
nothing. Evidence from an earlier leg's era, a stale park, or an unattested
shadow layout licenses nothing; a leg that recorded nothing can close only by
honestly CARRYING what a real ancestor close resolved, and the close discloses
`adjudicated_here: 0` when it does. `leg-close.v1` rows stay readable history
and license nothing -- `tools/backfill_close.py` is the only mint path for
pre-machinery legs (the pilot mini-rung + step-100), deriving the era-honest
proof from recorded-under evidence, never from an operator's arms. The same
tool is the production writer of the `leg-close.v1` HISTORY attestation the
era gate demands (`--attest-v1-history terminally-closed-pre-machinery`,
identity from the registry row, era-scoped, idempotent on identity, refused
for any rung with leg machinery or zero recorded evidence), and heads
re-adjudicated under a NAMED non-chain snapshot resolve only through
`--recorded-under-authority STABLE_ID=SNAPSHOT_ID:REVISION`, disclosed in the
close and dead the moment the head or the named snapshot claim moves. The root
exemption survives for exactly one case -- a campaign with zero registered
rungs, where no rung can have closed.

Composition dedups automatically -- a converted model is never re-crawled as
WORK (`runs`-holders are excluded from every composing stratum by construction,
verified in the rehearsal: 0 of 55 leaked into a 200-member rung), and
`ladder-bulk` skips every prior rung member, head-holder, and valve-claimed
model. Prior-rung members re-enter as work ONLY through the tail strata, which is
the intended re-attempt of the hard tail. The lineage TIP's members are a
separate matter: they are carried into the successor's membership by construction
(`carried-tip`), because membership is cumulative -- carried, not re-crawled, and
their awards are demanded at the door.

## Gate inventory

Every gate this protocol names, and what enforces it. A named gate with no tool
or no test is a promise nobody keeps.

| Gate | Tool | Test |
| --- | --- | --- |
| currency carve-out present | `rung_staging._require_currency_carve_out` | `test_prior_snapshot_currency_carve_out.py` |
| stratified composition | `tools.stratify_rung` | `test_rung_stratification.py`, `test_rung_boundary_protocol.py` |
| empty-stratum refusal + acknowledgement | `rung_stratification.EmptyStratumError` / `StratumNotEmptyError` | `test_rung_boundary_protocol.py` |
| lockless salting set matches disk | `rung_stratification.DEFAULT_ODDBALL_INTENTS` | `test_rung_boundary_protocol.py` |
| rung staging + lineage + registry | `tools.stage_rung` | `test_rung_staging.py`, `test_partitioner.py` |
| staged rung is the composed rung | `rung_staging.verify_composition_attestation` | `test_rung_composition_attestation.py` |
| composition carries the lineage tip (cumulative successor) | `rung_stratification` `carried-tip` + `rung_staging.lineage_tip_membership` | `test_rung_carried_tip_succession.py` |
| carry provenance is recomputed, never trusted | `rung_staging._verify_carried_tip_claim` | `test_rung_carried_tip_succession.py` |
| boundary crossed only on the TIP's close | `rung_staging._require_closed_lineage` + `close_object.require_closed_lineage` | `test_close_object.py`, `test_rung_staging.py` |
| sealed partition proof recomputes from pinned bytes | `partition_proof.proof_refusals` over `leg_generation.close_ledger_sources` | `test_close_v2_contract.py` |
| pre-machinery closes minted honestly | `tools.backfill_close` | `test_backfill_close.py` |
| v1 history attested era-scoped, from the registry row | `tools.backfill_close --attest-v1-history` | `test_backfill_close.py` |
| campaign-scoped membership census + durable acknowledgments | `rung_stratification._reconcile_membership` + `append_membership_acknowledgment` | `test_membership_census_scope.py` |
| root binding VERIFIED on every composition/staging path | `rung_staging.resolve_verified_root` | `test_rung_carried_tip_succession.py`, `test_close_v2_contract.py` |
| staging dry-run against the committed epoch (CI) | crawler-smoke job | `test_rung_carried_tip_succession.py` (two-generation compose→stage→register→compose→stage) |
| one receipt removed -> every door refuses, zero bytes | `rung_staging._require_closed_lineage` (all entry paths) | `test_rung_staging.py`, `test_close_object.py` |
| award custody at the staging door (every ledger layout) | `rung_staging._refuse_unauthorized_runs_holders` + `_models_ledger_paths` | `test_rung_staging.py` |
| subset rung over superset history | `authority` carve-out + reducer membership | `test_subset_rung_reducer_membership.py`, `test_subset_rung_requeue_grants.py` |
| rotation really rotated | `tools.launch_preflight`, `tools.rotate_setup_artifacts` | `test_launch_preflight.py`, `test_rotation_transaction.py`, `test_committed_locks_as_inputs.py` |
| phase-scoped completion partition | `driver` + `status.assert_partition` | `test_status.py`, `test_slice_f_driver.py` |
| rung tally + spend + legs owed | `tools.rung_report` | `test_rung_boundary_protocol.py` |
| throughput / R5 slope | `tools.throughput_report` | `test_gate_verdict_reuse.py` |
| checker latency | `tools.checker_latency` | `test_checker_latency_report.py`, `test_operator_checker.py` |
| review holds | `tools.grant_parks` | `test_grant_pending_parks.py` |
| terminal-refusal parks | `tools.refusal_parks` (CLI over `refusal_parks.py`) | `test_refusal_parks.py` (module-level) |
| effort grants | `tools.requeue` | `test_slice_g_tools_procedures.py` |
| disk GC | `tools.reap_disk` | `test_disk_reaper.py` |
| derived views rebuild | `tools.rebuild_views` | `test_checkpoint_transaction.py`, `test_slice_g_tools_procedures.py` |
| corpus integrity | `tools.verify_corpus`, `tools.promote_corpus` | `test_text_corpus.py` |
| prompt surface pins | `tools.verify_prompts`, `tools.verify_pool_prompts` | `test_executor_prompt_pins.py`, `test_acceptance_gate.py` |
| public/private byte boundary | `tools.license_sweep` | `test_slice_g_tools_procedures.py` |
| anti-substitution inventories + closure census | `tools.anti_substitution` + `tools.gate_closure` over `required_gate_set.json` (v3) + `tools.collection_gate` | `test_anti_substitution_inventories.py`, `test_gate_verdicts.py` |
| release attestation | `tools.release_lock` + `round21_macos_real` gate | `CLAUDE.md` "Green suite", conftest `pytest_sessionfinish` |
| host shard merge / handoff | `tools.merge_host_shards`, `tools.handoff_bundle` | `test_merge_host_shards.py`, `test_handoff_bundle.py` |
| campaign merge (end of ladder) | `tools.merge_campaigns` (CLI over `campaign_merge.py`) | `test_campaign_merge.py` (module-level) |

Two entries are marked module-level: `tools/refusal_parks.py` and
`tools/merge_campaigns.py` are thin operator CLIs whose logic is tested through
their modules, not through the CLI. Neither is on the rung-boundary path.

## Kernel-caller audit rule (standing)

Any diff touching a money/records KERNEL file — `paid_session_custody.py`,
`paid_session_*` modules, `ledger_chain.py`, `recordio.py`, the reducer close
ranges of `reducer.py`, the admission/pacing ranges of `scheduler.py` /
`driver_admission.py`, or `provider_reconcile_evidence.py` — triggers, in the
SAME review: (1) enumeration of every caller of the changed seam, and (2) a
failure-arm composition drive (each caller exercised through the changed
seam with its typed failure arms). A kernel diff whose review carries no
caller-audit row is an open GAP in the audit register and blocks its
cluster from closing.
