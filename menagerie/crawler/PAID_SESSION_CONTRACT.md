# Paid-session lifecycle contract

The paid-session surface is THREE separated concerns with zero shared mutable state.
That separation is the anti-regression rule, not a stylistic preference: six incremental
rounds of this surface each fixed one concern and regressed a neighbour, because
accounting, custody, and durability were coupled through one module.

| concern | module | owns | must never |
|---|---|---|---|
| **A. accounting** | `paid_session_accounting.py` | budgets, instruments, row validation, the wall rule | open a file, take a lock, spawn a process, read an environment variable |
| **B. custody** | `paid_session_custody.py` | kernel liveness, credential lifetime | import a budget symbol |
| **C/D. durability** | `paid_session_durability.py` | append-only chain, extent anchor, `F_FULLFSYNC` | know any money vocabulary |
| *(seam)* | `paid_session_lifecycle.py` | composes the three; `PaidSessionAuthority`, `Reservation` | hold budget arithmetic or kernel logic of its own |

`test_checker_wall_breaker.py::_assert_cross_concern_separation` is the permanent AST
census over those boundaries.

## Instruments (closed set, one axis each)

* **QUOTA** — cumulative, per unit of work, never resets. Axis: `campaign_id` plus the
  row's own `work_generations`.
* **RUNAWAY_BREAKER** — epoch-scoped, resets only on durable evidence the epoch ended.
  Axis: `campaign_id` plus the row's own `generation_id`.

Every money row stamps BOTH coordinates, so a consumer is windowed by construction.
There is no mutable spend counter and no hydration entry point anywhere: a budget is a
pure function of durable rows evaluated inside the same exclusive lock that would write
the row it authorizes, so a refused buy mints nothing.

## The two settlement axes, and where each gets its truth

**Sessions** — provider-attested in both directions, writer-minted only. A
provider-consummated session weighs 1 even when this host failed locally; a
provider-declined one weighs 0. `local_outcome` is recorded for disclosure and is read
by no projection, so no local flag can mint or dodge a session charge.

**Wall** — `min(reserved, max(machine_floor, provider_attested))`, one pure function
(`charged_wall_seconds`) applied twice: by the writer when it mints the settlement and
by the validator when it re-derives that settlement on every replay. A runaway is a
LOCAL phenomenon, so the driver's own `time.monotonic` measurement is a floor a remote
answer can raise but never delete; the reservation is a ceiling the provider cannot
exceed. The provider's RAW attestation is persisted unclamped in
`provider.attested_wall_seconds`, so a clamp is visible to a replay instead of silently
overwriting the number it claims to quote.

Both axes are keyed on the settlement SCHEMA string (the current schema is the one that
re-derives the wall and reads the session weight from provider attestation), so a schema
downgrade used to skip both and read the charge from a self-describing label instead
(PSLRE4 F2). Legacy settlement schemas are now admitted only where they can exist:

* a legacy schema is NEVER admissible for a row of this ledger, because every row of this
  ledger is post-cutover BY CONSTRUCTION: the `accounting_genesis` row must be the
  ledger's first row and it pins the pre-machine-floor prefix as a SEPARATE object (path,
  digest, byte count) the fold never ingests. The genesis row's `legacy_settlement_schemas`
  field is validated DISCLOSURE about that other object and licenses nothing here;
* the ladder `v1 < v2 < v3 < v4` is MONOTONE per ledger, so a settlement can never fold
  below the highest schema the ledger has already settled with — defense in depth that
  keeps holding when the ladder grows a new current schema.

Both gates are derived from the ledger's STRUCTURE with zero producer input. Treating the
declaration itself as the gate was the same "self-describing label" pattern one level up
(PSLRE5 acct F2): the genesis row is written by the same producer, with the same unkeyed
authority, as the settlements it would license, so minting it cost one inserted row — and
because the ladder is monotone rather than strictly increasing, a UNIFORM downgrade of
every settlement never moved below rank 0 and zeroed a whole ledger. A forged declaration
now grants nothing, and a uniform downgrade refuses on its FIRST row.

## `--provider-reconcile-command` adapter contract

The command receives ONE paid-session idempotency key as its final argument and must
print exactly this JSON object on stdout, exiting zero:

```json
{"idempotency_key": "<the key it was given>", "accepted": true, "observed_wall_seconds": 12.5, "adapter_identity": "provider-reconcile-evidence/2"}
```

* `idempotency_key` — REQUIRED, and checked against the queried key. Without the echo the
  receipt digest attests only itself; with it the answer is an attestation about a
  specific reservation. This is the same nonce-echo the notification lane already
  requires of a third party (`driver_progress.py`).
* `accepted` — boolean. Whether the provider consummated and therefore bills the session.
* `observed_wall_seconds` — number or `null`. `null` is the honest answer for a
  token-billed provider that does not meter wall clock; it neither zeroes nor inflates
  the charge, because the driver's own measurement is the floor. `accepted: false` with a
  nonzero wall is representable and settles at zero sessions over the metered wall.
* `adapter_identity` — REQUIRED nonempty string naming the answering adapter and its
  version (D-1). 200a's two deployed adapters — the retired constant-accept stub and the
  evidence adapter — were observationally identical on every ledger row, so no receipt
  could show WHICH reconciler answered. The identity rides inside the digested status
  receipt (`receipt_sha256` covers it), making the provenance verifiable prospectively.
  A legacy three-field adapter refuses typed at the contract check.

Anything else — a wrong key, a missing field, invalid JSON, a nonzero exit, a timeout —
is a typed `PaidSessionError`. Leg start retries it, so an adapter outage delays
admission rather than wedging the campaign. A nonzero exit specifically is the typed
subclass `ProviderReconcileRefusal`: "this adapter cannot answer this key", the one
behavior the negative control below expects.

### The standing negative control (canary nonce)

The constant-accept class is dead in code (D-1/DX-75/B2-X-V12), three layers deep:

1. `CommandProviderReconciler` REFUSES construction of any command naming the retired
   `provider_reconcile_accept.py` — a stale persisted campaign config or launchd export
   resurrecting it fails typed at wiring time, before any money row is written;
2. every admission-boundary reconcile (`_reconcile_paid_sessions_before_admission`, leg
   open) first runs `verify_reconciler_negative_control`: a well-formed canary key that
   no ledger can hold a reservation for is queried, and the wired adapter must REFUSE
   it. ANY answer — accepted or declined — proves the adapter speaks about keys outside
   its ledger (the constant-accept class under any filename) and refuses the leg typed
   (`ReconcilerNegativeControlError`). An outage is its own typed failure, never a pass;
3. the identity-bearing field contract above refuses every legacy adapter outright.

Pinned by `test_reconciler_negative_control.py`, including the exact retired stub body.

The key is minted inside `reserve()` and mixes the ledger's verified chain head, the
host, and the boot id, so two independent ledgers at the same extent cannot claim the
same provider session. It is exported to the paid child as
`MENAGERIE_PROVIDER_IDEMPOTENCY_KEY` for adapters that can pass it to the provider.

### The shipped receipt-backed adapter (`tools/provider_reconcile_evidence`)

The adapter this repo ships answers from the paid ledger's OWN chain-verified rows,
because the provider it fronts exposes no per-key status endpoint. It replaces a
constant-accept stub whose every answer was `accepted: true` — safe in the money
direction and useless in the truth direction, since a reservation the driver minted and
then crashed on BEFORE spawning anything was charged as a consummated session and no
operator could separate a real spend from a bookkeeping ghost.

Its `accepted: false` is a MACHINE PROOF, not an assumption, and the proof is the paid
dispatch chokepoint's own ordering (`driver_admission._run_operator_command`): the guard
is spawned behind a driver-owned release pipe and BLOCKS on it before executing the
operator command; `bind_custody()` durably appends `session_custody_bound` (`F_FULLFSYNC`
plus the anchor's commit/intent protocol); and only THEN does the driver write the release
byte. A guard that reads EOF instead of `b"1"` — exactly what a driver death before the
release produces — exits 70 without ever starting the operator command
(`author_executor.run_parent_deadline_guard`). So a reservation with no custody row
cannot have exposed a provider byte. That ordering is the adapter's load-bearing premise
and is pinned by `test_provider_reconcile_evidence.py`, which fails if the release gate,
its EOF refusal, or its position relative to the binding is moved.

The evidence rules, recomputed from verified rows on every invocation with zero
producer-supplied input:

* no `session_reserved` row names the key -> REFUSE, nonzero exit, empty stdout. This is
  the one money-NEGATIVE failure available to the adapter, and a misdirected
  `--records-root` is precisely how it would happen, so an unanswerable key is never
  answered. Leg start retries a refusal, so it delays admission instead of zeroing a
  charge this host cannot see;
* more than one such row -> REFUSE. The key mixes the verified chain head, the host, and
  the boot id, so it is unique by construction and a duplicate is tampering;
* otherwise `accepted` is true iff a marker in the closed set `PROVIDER_REACHED_EVENTS`
  (`session_custody_bound`, `session_custody_released`) names the reservation's
  `spend_id`, or a prior settlement already recorded a provider-accepted verdict for it.
  Widening that set is a new decision needing its own reviewed change: every addition can
  only move answers from `false` to `true`.

A custody row means the subtree was spawned and may have reached the provider, so it is
charged — the residual ambiguity is kept on the money-POSITIVE side, exactly where the
constant-accept reading put ALL of it.

`observed_wall_seconds` is always `null`. This host meters no provider wall, and the
contract's null answer neither zeroes nor inflates the charge because the writer's own
measurement — for a crash path, the durable reservation age
(`_reservation_age_seconds`) — is the floor. Claiming a wall here would be inventing
provider attestation.

The executor's attempt-bound publication receipt is deliberately NOT consulted. It is
strictly weaker evidence for this purpose: publication implies the operator command ran,
which implies the release byte, which implies the custody row, so it can never be a
positive this rule misses — and the durable attempt records carry no `spend_id`, so a
publication digest cannot be soundly bound to one reservation. Reading it would mean
attesting one session with another session's artifact.

## One reservation is ONE paid session

A reservation is one paid provider session: one provider idempotency key, one
custody epoch (at most one bind, at most one release), one settlement.  These are
fold invariants with zero producer input, and the writer refuses a duplicate or
post-settlement bind before any byte lands.  An operation that legitimately runs
more than one guard session (the author lane's source-request session followed by
its author session, and every bounded retry attempt) SETTLES the finished session
first and reserves anew — the settlement moves the verified chain head, so the
next `reserve()` necessarily mints a different spend id and provider key.  There
is no rebind lease.  The first step-200 leg (2026-08-20) rebound a released
reservation for its second guard session; the durable duplicate custody row then
made every later money read of the ledger refuse — the correct fold behavior,
which is why the refusal now happens at mint time and the durability substrate
folds every row before writing it.

## The ATTEMPT is the durable author budget unit

One author ATTEMPT buys up to two guard sessions; pricing the allowance in
session units (`quota_sessions = 2*A`) would let a work generation buy `2*A`
source-only failed attempts and desync `author_sessions_used/max`.  The attempt
is therefore its own durable, replay-derived unit:

* every author reservation carries `attempt_group`
  (`paid-session-reservation.v2`): `null` HEADS a new attempt; a spend id JOINS
  that attempt as a later guard session.  The fold enforces the closed join rule
  — the head must exist, be a head, share the lane and both accounting axes,
  every earlier member must already be settled, the join must stay within the
  lane's declared per-attempt session bound
  (`LANE_ATTEMPT_SESSION_BOUNDS`: author 2, checker 1; a lane without a
  declared bound refuses joins entirely), and a fully provider-declined
  (refunded) group is CLOSED — a revival join refuses, so a refund is final and
  can never let joins raise the chargeable projection past the allowance.  An
  attempt identity therefore cannot be forged, chained, moved across work
  generations, or extended into an unbounded session laundry;
* the allowance valve (`_admit_author_spend`) refuses a NEW attempt group once
  `AccountingState.quota_attempt_groups` reaches the allowance `A` for the exact
  `(campaign, stable_id, work_id, author)` coordinate — source-only failures
  included, across crashes and relaunches, from ledger rows and never a mutable
  counter.  A group every provider answer DECLINED weighs zero attempts (the
  same refund a declined session got under session pricing);
* canonical `budget.author_sessions_used` and `author_sessions_max` share the
  attempt unit: `used` is the attempt-group projection (plus the artifact-ledger
  staging floor), `max` is the allowance.  Raw guard sessions remain separately
  bounded (QUOTA `quota_sessions = AUTHOR_PAID_SESSIONS_PER_ATTEMPT * A`) and
  separately readable from the paid ledger's own session projection.

**Disclosed, bounded residual: the attempt-group shape is ROLE- and ORDINAL-unbound**
(N2, 2026-08-20). The closed join rule binds lane, both accounting axes, settlement
order, the per-attempt session bound, and refund finality — it deliberately binds NO
member role and NO member ordinal. Both guard sessions of one author attempt stamp the
same `dispatch_kind: "author_source"` at the single reserve site
(`_admit_author_spend`), and a member's position exists only as ledger append order
(`AccountingState.attempt_groups` keeps head-first append order; the bound refusal
computes "paid session N of the group" on the fly). The ledger therefore cannot PROVE
which member of a group was the source-request session and which the author session,
nor re-derive an intended order distinct from append order. Bounded because no money
consequence keys on either: members charge by their own reservation and settlement rows
on their own recorded axes, the allowance counts GROUPS, the refund decrement resolves
by head identity alone, and the rollover opening re-runs the SAME role/ordinal-free
join rule — a role swap or reorder inside a group cannot change any total, any attempt
count, or any carried charge, live or across rollover. The gap is forensic attribution,
not accounting; closing it means a fold-enforced role/ordinal binding in the
reservation vocabulary (a `v3` schema decision, never a patch), and until then the
shape is disclosed here rather than assumed bound.

Shutdown and the sealed wave window gate EVERY guard-session buy, not only
attempt starts (Sol R3 W1).  `_admit_author_spend` re-checks the shutdown
authority at the writer's own reserve site, so a mid-attempt join (the second
guard bought by `begin_guard_session`) refuses typed after a requested
shutdown — the finished guard's settlement stands and no new reservation
lands.  The author wave's worker-to-writer admission relay seals with the
window: once `cancel_pending()`/`close()` runs, newly requested,
already-queued, and racing-past-the-seal buys all refuse with the lane's
typed pause outcome — never an untyped hang, never a durable money row.

## Infrastructure deaths refund attempts, never paid-session fuses

The R4 provider-decline refund has one machine-only sibling.  The fold derives
an ATTEMPT member weight from each settlement: provider decline weighs zero;
provider acceptance normally weighs one; and provider acceptance weighs zero
attempts only when `local_outcome` is one of the closed raw infrastructure
outcomes in `ATTEMPT_REFUND_LOCAL_OUTCOMES`.  A group weighs zero only after
EVERY member has settled to zero.  This is not a persisted verdict: the row
stores the driver's raw machine disposition and
`settled_attempt_weight()` re-derives the weight on every fold.  Missing,
unknown, legacy, mixed, or contradictory evidence weighs one.

Two machine outcomes are minted by the driver's own kill/reap machinery, never
from any child evidence (legs 8 + 12, 2026-08-21). `failed-checker-session-wall`
is stamped only by the checker lane's external wall kill: the lane's bound is the
NAMED per-session wall (`CHECKER_SESSION_WALL_SECONDS`, which must contain the
operator wrapper's own granted attempt ladder) times
`CHECKER_WALL_EXTERNAL_KILL_FACTOR`, mirroring the author lane's `grant x 1.10` —
the driver killed the guarded group itself, so the session settles refundable and
the existing bounded transport ladder retries the batch. `failed-custody-reaped`
is stamped by the custody deadline reaper (below) and, since DECIDED_PLAN item 9
(recovery F1.2, 2026-09-01), by leg-start reconciliation for a BOUND custodian the
closed custody ladder proves dead (`already-dead`: flock FREE and process group
EMPTY; `stale-boot`: another boot id) REGARDLESS of deadline — the deadline plus
grace was only ever the mid-leg probe schedule, and a killed leg's settlement must
not depend on how fast the operator restarted it (restarting inside ~35 minutes
used to charge every in-flight attempt as `unknown`; waiting refunded them).

A third machine outcome, `failed-executor-typed-outage` (item 9 / recovery F1.1),
is stamped only by `classify_author_exit` from the closed proof rule
`author_dispatch.executor_typed_outage_outcome`: the executor exited with its
typed retryable/unavailable code (75/78), printed its own PRE-SESSION typed line
(`author executor error:` for 78, `author executor io error:` for 75 — the two
lines `author_executor.main` prints before any attempt exists), published no
payload at the envelope's required output path (a driver-observed non-empty
regular file), and emitted no harness session-failure notice and no
attempt-scoped `author executor <stage> failed:` line. Every leg is a
driver-observed fact; ambiguity (an unknown output path, a crossed code/line
pair, any evidence a session ran) charges the attempt as before. This closes
the contradiction between the classifier's own comment ("a session that died
without publishing is indistinguishable from broken infrastructure") and its
settlement: the pilot's 82 and the R6b clone's 742 sub-10 s executor deaths
(unset `MENAGERIE_AUTHOR_CLAUDE_BIN` -> exit 78 in ~2 s) were charged as model
attempts and terminalized 12 + 116 work generations `effort-exhausted`. Both
item-9 classes are cross-model storm-fuse classifications, so five distinct
models dying the same typed way inside one window HALT the lane. Money
invariants are untouched: provider-attested session weight, the machine wall
floor and replay-derived accounting stay; a refunded attempt terminalizes and
publishes nothing.

The author harness arm is deliberately narrower than its diagnostic failure
classifier.  It reads only `harness_is_error`, `harness_terminal_reason`, and
the observed subprocess return code from the executor-built envelope.  The
2026-08-20 storm shape is exactly `is_error=true`, terminal reason
`aborted_streaming`, and SIGTERM status `-15` or shell status `143`; structured
`api_error` is the other closed error terminal.  Generic timeout and
`model_error` remain chargeable because author behavior can influence them.
Harness subtype, `failure_class`, `failure_class_basis`, stdout/stderr,
quarantined error text, and model prose have zero authority.  The driver also
derives spawn `OSError` and its own author-queue watchdog as closed machine
outcomes.  A model printing any of those words without the exact trusted shape
remains chargeable.

This refund changes only effort discipline.  `settled_session_weight()` still
charges every provider-accepted session, and the machine wall floor still
charges its observed time, so cumulative QUOTA and epoch RUNAWAY_BREAKER see
the entire infrastructure storm.  For stream aborts and spawn failures, the
ladder settles each killed session, then replays the fold before buying a
fresh session-level retry.  Those retries use
deterministic per-model jittered backoff and have no attempt-count stop; the
next reservation is always subject to the ordinary shutdown, QUOTA, and
RUNAWAY admission fuses.  Provider weather and author-queue watchdog failures
retain their narrower existing outage ladders, but their paid settlements use
the same zero attempt weight.  Thus infrastructure cannot terminalize a model
by spending its attempt allowance, and zero attempt weight cannot make a storm
free or unbounded.

## Rollover of a fold-refused paid ledger (`tools/rollover_paid_ledger`)

A paid ledger whose durable rows the fold refuses (the 2026-08-20 residue:
`reserved -> bound -> released -> bound`) bricks its records root loudly at
admission-time reconciliation.  That refusal is permanent — no row edit, no
exemption, and a silently fresh ledger would reset cumulative QUOTA and forget
the residue's conservative charge.  The ONE sanctioned same-campaign
continuation is the rollover protocol:

1. the poisoned ledger is preserved immutably — renamed in place to a
   `.poisoned-<utc>` sibling (anchor alongside), byte digest recorded and
   re-verified, never edited;
2. a NEW accounting authority is minted whose genesis row
   (`accounting_rollover`, `paid-session-rollover.v1`) EMBEDS the poisoned
   ledger's chain-verified rows verbatim and names the preserved sibling, its
   whole-file digest, and the predecessor's anchor pin — lineage, not
   laundering;
3. the conservative totals carry forward as opening charges: every carried
   reservation (settled or not, nested rollovers included) is charged as
   consummated at its FULL reserved wall (constant-accept semantics), at its own
   recorded axes, disclosed as `rollover_opening_charges` — so cumulative
   QUOTA/RUNAWAY lose nothing.  The opening ATTEMPT count re-runs the SAME
   closed join rule as live folding: a carried reservation whose join topology
   is invalid charges as its OWN attempt head, so an invalid predecessor
   topology can only raise the opening attempt count, never lower it below
   what its sessions imply;
4. one `paid-ledger-rollover` operational event discloses all of it.

The row is a four-part-doctrine waiver, and the FOLD — not the tool — is its
authority, recomputed on every replay from the embedded digest-bound rows:
it grants nothing positive (the successor starts MORE charged, never less, and
carried spend ids and provider keys are permanently unmintable on the new
ledger); eligibility is the closed structural rule "an AUTHENTICATED
predecessor the accounting fold refuses", so a HEALTHY predecessor refuses
rollover at the tool AND at the fold-first append (the forged row lands
nowhere); and the census is segregated in `AccountingState.rollover`, never
counted as audited live spend.  The rollover row must be the successor
ledger's first row.

The predecessor's identity is ANCHOR-DERIVED, never producer-declared, on
every replay: (a) the named whole-file digest and byte count must re-derive
byte-for-byte from the embedded rows through the ledger's own canonical line
encoding (pure — a declared digest naming any other byte sequence refuses
before any disk read); (b) the row's `poisoned_anchor` pin must re-derive from
the embedded rows' own chain (identity = first chain head, extent, head); and
(c) the fold's injected lineage verifier (`preserved_lineage_verifier`, built
by the durability concern and seeded through the lifecycle seam — the
accounting concern stays pure) re-reads the preserved `.poisoned-*` sibling
AND its preserved commit/intent anchor from disk beside the ledger and
confirms the same digest, extent, and head.  A fold with no verifier (the bare
pure projections) refuses rollover rows fail-closed.  Consequences: a
synthetic chain-valid-but-fold-refused row sequence is unmintable through any
lifecycle writer (it has no anchored on-disk predecessor); forging one now
requires the rewrite-and-re-pin FILESYSTEM capability the ledger contract
already discloses as its global residual — never less; and the preserved
`.poisoned-*` siblings and anchors are PERMANENT artifacts of the records
root — deleting or altering one makes every later replay of the successor
refuse loudly, which is tampering evidence, not a bug.

**Threat-model scope: a coherently FABRICATED predecessor (file + anchor) is out of
contract scope, not an open residual** (N1, 2026-08-20). The verifier's identity is
anchor-derived from on-disk BYTES, and both the preserved sibling and its anchor are
keyless (`stable_hash`) and co-located, so an actor who already holds the
rewrite-and-re-pin FILESYSTEM capability disclosed below as the ledger's global residual
can materialize the ENTIRE evidence set coherently: a chain-valid, fold-refused
`.poisoned-*` sibling written from nothing, its matching commit/intent anchor, and a
successor whose `accounting_rollover` row embeds those fabricated rows. Every clause
(a)–(c) passes, because the result is byte-for-byte an HONEST rollover of that
fabricated history — the fold attests LINEAGE to the preserved bytes, never PROVENANCE
of the bytes (the same boundary the runnable-spec contract draws for coherent
reauthoring: separating the fabrication from its honest twin would require refusing
IDENTICAL inputs). The capability gains nothing new — the same actor can rewrite the
LIVE ledger and re-pin its anchor directly, strictly cheaper — so this widens nothing:
producer processes (the lifecycle writers, the tool, any code path without records-root
write capability) still cannot mint, replay, or launder a rollover, which is exactly
what `test_paid_ledger_rollover.py`'s synthetic-chain and false-identity refusals pin.
Closing it needs the keyed or off-host attestation the global-residual paragraph
already names as a separate decision — never a verifier tolerance.

The nested walk is hard-bounded (`ROLLOVER_MAX_DEPTH = 3`,
`ROLLOVER_MAX_CARRIED_ROWS = 50_000`, checked iteratively before any
recursion): a legitimate campaign rolls over only when durably poisoned (the
pilot needs depth 1), and one generation's rows are bounded by the RUNAWAY
session budget, so the constants are far above any legitimate history while
keeping replay work and recursion closed.

`test_paid_ledger_rollover.py` pins the pilot's exact residue shape, the
byte-identity of the preserved sibling, the conservative charge equality, the
fully-authenticated-healthy refusal, the synthetic-chain and false-identity
refusals, the invalid-topology conservative opening, the nesting bounds, the
live-writer lock, and the mint-failure byte-identical restoration.

## Custody: two kernel oracles, unioned

The paid subtree is the whole tree below the driver's guard, including the grandchild
that holds the credential and does the buying. Two independent oracles cover it:

1. the inherited **`flock`** on a lease-named file — survives the parent's death, needs no
   custodian, trusts no PID, and is answerable by anyone;
2. the paid **process group** — probed with signal 0, catching the descendant that closed
   the descriptor but stayed in the group.

The `flock` is AUTHORITATIVE and the group probe is strictly ADDITIVE. `verified-exit`
requires the flock to be `free`; the group probe can only ever add a refusal. That
asymmetry is deliberate: the group probe reuses the supervisor's `ps`-based membership
enumeration, which answers "unanswerable" on a host that cannot enumerate processes, and
treating that silence as blocking would refuse every clean paid session. The honest
consequence is that on such a host only the flock oracle is available and the residual
widens to any descendant that drops the descriptor.

Neither oracle signals anything. This module owns no raw process-group targeting: the
timeout arm kills the group through the hardened `signal_verified_process_group` while
the leader is still alive and verifiable, and the restart reaper does the same for a
lease whose identity still matches. After the guard is reaped its start token is gone,
so the group cannot be verified and must not be signalled — an unprovable subtree stays
disclosed instead.

There is deliberately no self-certification primitive: a token a wrapper writes about
itself is evidence about the writer, never about its descendants, and a positive custody
outcome derived from it is the producer-supplied field the four-part waiver doctrine
forbids. A third-party wrapper that says nothing therefore ends a clean session
normally, and a cooperating one cannot certify an escape.

Credential-bearing implies observable: `author_executor` hands the anonymous credential
descriptor and the custody lock to a descendant in ONE `pass_fds`, and refuses to expose
the credential at all if paid custody was declared but its descriptor did not survive.

Claude sessions lead their own process groups even beneath the paid deadline guard. The
guard remains the custody lease leader (`leader_pid == leader_pgid`), while the inherited
`flock` remains the whole-subtree oracle across the new group boundary. A campaign halt
first writes a parent-owned teardown record, then SIGTERMs each verified guard group; the
executor's machine handler tears down only its isolated Claude group. The attempt retains
bounded quarantined stderr plus one record per signal (`target_pid`, `target_pgid`,
`signal`, `reason`, `teardown_trigger`, and kernel outcome).

`campaign-teardown` is derived only from driver shutdown state plus that durable parent
record. It is control flow, not an author failure: it bypasses the retry-exhaustion and
consecutive no-publication paths, refunds the ATTEMPT projection, and leaves the model
eligible on resume. The raw provider session and machine wall remain charged to QUOTA and
RUNAWAY, so the mandatory loud teardown cannot hide spend or weaken either fuse.

**The custody deadline reaper** (`_reap_expired_dead_custody`, legs 8 + 12,
2026-08-21) runs at every writer-owned admission boundary and first at leg start.
A BOUND lease participates only past its durable reservation `deadline_at` plus
`CUSTODY_REAPER_GRACE_SECONDS`, and is force-settled only when the custodian is
machine-provably dead by TWO independent kernel signals
(`custodian_provably_dead`): the flock `FREE` (probed first, authoritative and
cheap) AND the process group enumerably EMPTY. `UNANSWERABLE` on either skips,
fail-closed; a LIVE custodian is never reaped on time alone. The settlement is
replay-derived with zero producer input — provider truth from the idempotency-key
query, the durable reservation age as the machine wall floor, and the closed
`failed-custody-reaped` disposition re-weighed by the fold — so sessions and wall
stay charged and only the ATTEMPT projection refunds. A typed refusal (adapter
outage, a settle raced by the owning thread) skips the lease, which stays
disclosed for the next boundary or leg start. Never-bound reservations remain the
startup ladder's job.

`subtree_liveness` opens the lock pathname WITHOUT `O_CREAT` and compares the inode
identity the binding pinned. A missing or replaced pathname is `unanswerable`, never
`free`, so one `rm` under the runtime root cannot retire a live subtree — and the
operator drain additionally requires the bound leader to be machine-provably gone.

**The one irreducible Darwin residual** (`CUSTODY_RESIDUAL`): with no `PR_SET_PDEATHSIG`,
no pidfd, and no cgroups, a descendant that both closes the inherited descriptor and
leaves the process group cannot be observed. Such an orphan is financially bounded by its
one reservation (it cannot reach the writer to buy again), holds no durable credential
pathname, and is reaped on restart. It is disclosed, never claimed to be impossible.

## Durability: the anchor's commit/intent protocol

A crash between the row's `F_FULLFSYNC` and the anchor update used to be byte-identical
to a deliberate out-of-band append, so any comparison made afterwards had to either
refuse both (bricking a crashed campaign) or adopt both (blessing a forgery). The
distinguishing evidence is therefore written BEFORE the row:

1. under the ledger's exclusive `flock`, the writer records the exact next row's chain
   head and byte extent as the anchor's `intent`, beside the current `committed` pin;
2. the row is appended and `F_FULLFSYNC`-ed;
3. the anchor is rewritten with `committed` advanced and `intent` cleared.

A reader accepts exactly two states — ledger matches `committed` (the row never landed)
or ledger matches `intent` (the anchor lagged) — and refuses everything else. `intent`
pins one intended row's exact `(rows, bytes, chain_head)`; a truncation cannot satisfy
`committed`, which is ahead of it. That pin forces a PREIMAGE on an out-of-band append
because the chain head is a function of the ROW BODY: `ledger_chain.row_content_digest`
hashes the row minus its `chain` envelope and minus the self-digest fields, so it reads
NOTHING the row declares about itself. Twice before, this paragraph was false. The chain
read the row's own `row_sha256` and nothing recomputed it, so a forger could satisfy an
open intent by copying the intended row's digest onto a different body (PSLRE4 D-2) and a
length-preserving in-place edit of a money row moved no pin at all (PSLRE4 D-1); the
re-derivation that closed those left the chain SELECTING its address from the first digest
field a row happened to carry, so adding one `close_sha256` re-opened both with strictly
less work (PSLRE5 P5-D-1). An address that is derived cannot be held by a field of any
name. Two independent authorities now cover the same body: the chain's derivation, and the
accounting concern's own re-derivation
(`paid_session_accounting.validate_row_digest`, which runs the SAME shared derivation, so
the two cannot model the exclusion set differently). The chain also RE-CERTIFIES every
digest field a row declares against that derivation, so the fields consumers key on — the
close-head index, the same-campaign predecessor spine, award revision references — are
evidence rather than producer strings; and a money row's top-level key vocabulary is closed
per schema, so a foreign digest field is unrepresentable as well as harmless.

The direction matrix (`_ANCHOR_DIRECTION_MATRIX`) is the permanent tripwire — including the
arms that ADD a field rather than substituting bytes
(`close-sha256-pinned-in-place-edit`, `close-sha256-substituted-into-open-intent`,
`forged-genesis-uniform-downgrade`) — together with the row-digest tripwires in
`test_checker_wall_breaker.py` (`test_in_place_money_row_edit_refuses_typed`,
`test_copied_row_digest_cannot_satisfy_an_open_anchor_intent`,
`test_a_digest_only_edit_leaves_the_address_and_still_refuses`,
`test_the_accounting_authority_refuses_a_copied_digest_without_the_chain`) and the substrate
tripwires in `test_ledger_chain_content_address.py`.

Before a relaunch, `tools/chain_address_sweep` recomputes BOTH the deleted ladder's head and
the derived head for every chain-bearing ledger under a records root and must report
`CHAIN_ADDRESS_SWEEP_CLEAN`: the derivation is transparent on every honest row of every
family, and that sweep is what turns "no on-disk row was stamped by an older rule" from an
assumption into a checked fact.

The sweep report is `menagerie.crawler.chain-address-sweep.v2`. Beyond the two heads it
names every row the verifier would now refuse, so a relaunch meets no refusal the sweep has
not already reported: a fused-digest divergence, a chain envelope outside its closed key
vocabulary OR carrying a non-machinery VALUE, and — on `paid.sessions` only — a
`legacy_settlements` row. The last one is not a forgery and never was: the
settlement-schema gate refuses a `paid-session-settlement.v1`/`v2`/`v3` row on this ledger
unconditionally and with zero producer input, which is exactly what makes forging the
cutover declaration worthless. Its cost is that such a row already on disk BRICKS the whole
money authority on first read, so the sweep names the row instead of letting an operator
meet a cutover message with no row attached.

Residual, disclosed rather than assumed absent: an adversary who rewrites the ledger AND
re-pins the unkeyed, co-located anchor consistently is still adopted. `stable_hash` is
keyless and the anchor sits beside the ledger, so the body-derived address raises that
attack's cost from "edit one row, touch nothing else" to "rewrite and re-pin" — a
work-factor change, not a closure. Closing it needs a keyed or off-host attestation and is a
separate decision.

## Scale envelope

Reserve, charge, quota, and runaway are pure over rows that were verified and indexed
ONCE, through the incremental `close_reads.derive` substrate. Measured cost is flat per
append — about 2.0 validations and 1.0 chain verifications per durable row, and 2 anchor
writes per append, at every ledger size. `_assert_pslc_scale_acceptance` compares those
ratios at two ledger sizes and fails if they GROW, so a future change that reads history
on the money path is caught even though the absolute numbers are small.
