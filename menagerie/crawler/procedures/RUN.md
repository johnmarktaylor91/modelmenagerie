# Campaign run

The scheduler is single-writer and environments are created, used, checkpointed, and removed one
at a time. Set `INTAKE` to the immutable snapshot directory printed by intake.

## Launch preflight (MANDATORY before any leg that promises to release rebuild parks)

The quarantine/park identity hashes the four COMMITTED setup artifacts
(`envs/<intent>/locks/<target>.lock` / `.resolved.json` / `.resolved.sha256` / `.probes.json`).
Editing `environment.yml` is NOT a rotation: rung 12 burned two multi-hour legs because the spec
changed while the artifacts did not, so all ~43 rebuild parks stayed provably live and every
re-drive was re-claimed before env/worker/reducer.

```bash
CRAWLER_RUNTIME="$(pwd)/.crawl-local"
python -m menagerie.crawler.tools.launch_preflight \
  --records-root menagerie/crawler/records --envs-root menagerie/crawler/envs \
  --env-root "$CRAWLER_RUNTIME/envs" \
  --state-root "$CRAWLER_RUNTIME/environment-operator" \
  --tmp-root "$CRAWLER_RUNTIME/tmp"
```

The preflight evaluates every current `deferred:environment-rebuild-required` row through the
production claimant and REFUSES (exit 2, `rotation-did-not-rotate`) when any row would still be
re-claimed at launch. To actually rotate a quarantined intent, run
`python -m menagerie.crawler.tools.rotate_setup_artifacts --intent <name> ...`, which re-solves
through the production environment operator, proves the probes against a really created prefix,
publishes the four artifacts plus a provenance binding, and prints the old/new identity pair.
Use `--report-only` for launches that promise no rotation.

Pass the same `--tmp-root` to `python -m menagerie.crawler run` (or `resume`) so the shared disk
authority measures the temporary filesystem the environment wrapper actually uses.

## Pip overlay (controlled, offline, identity-bearing)

pip-only libraries (speechbrain via the `speech` intent, pykeen via `graph`) ride in through a
per-intent overlay, never an ambient `pip install`: declare direct requirements in
`envs/<intent>/pip-overlay.in` (and `pip` among the intent's CONDA dependencies), then rotate the
intent. The rotation resolves the declaration ONCE
through the created prefix's own interpreter (so conda-provided distributions are never re-pinned
or shadowed), pins exact versions with sha256 wheel hashes into the committed
`envs/<intent>/locks/<target>.pip-overlay.txt`, and installs it strictly offline
(`--no-index --no-deps --require-hashes --only-binary=:all:` against a wheel CAS the operator
materializes and verifies). The pinned manifest is a FIFTH committed setup-artifact member: it
participates in the quarantine/park identity, the environment generation, the routed package
inventory (coverage, author disclosure, R1 digest resolution), and the same rotation/preflight
discipline as the conda lock family. At run time the lifecycle installs the overlay between
`create` and `probe`, so every probe attests the overlaid runtime; a declared overlay that is
unpinned, superseded, or unsupported by the backend refuses to run rather than degrading to a
conda-only environment. Wheels-only, by design: a dependency without an osx-arm64 wheel fails the
rotation loudly (source builds stay refused), and true platform walls (espnet's
ctc-segmentation) are not pip problems.

## Phase 1: PyTorch

```bash
python -m menagerie.crawler plan --intake "$INTAKE" --target osx-arm64 --phase pytorch
python -m menagerie.crawler run --intake "$INTAKE" --target osx-arm64 --phase pytorch --sequential-envs
```

Routing assigns PyTorch work from declared framework and dependency requirements to the thick
`core`, `graph`, `audio`, `speech`, `mmlab`, `tabular-ts`, `recsys`, `detect-misc`,
`legacy-torch`, and `oddballs` intents. R2/R3/R4 normally route to `core` unless exact source requirements prove
otherwise. The planner may split an incompatible thick intent, but it does not create per-model
convenience environments (`recsys` is such a proven split: recbole's only conda builds pin
python <=3.10 against the python >=3.11 stacks).

## Phase 2: native framework tail

Start this only after Phase 1 has no pending PyTorch workflow rows.

```bash
python -m menagerie.crawler plan --intake "$INTAKE" --target osx-arm64 --phase native-tail
python -m menagerie.crawler run --intake "$INTAKE" --target osx-arm64 --phase native-tail --sequential-envs
```

Native TensorFlow/Keras, JAX/Flax, and Paddle work routes to `tf-keras-arm64`,
`jax-flax-arm64`, and `paddle-arm64`. A different original and run framework is recorded only when
a transparent adapter delegates to the native implementation.

Static hints only prioritize probes. `deferred:needs-cuda` or `deferred:needs-x86` requires stored
source or focused probe evidence. The designated arm64 heavy-build pool gets at most two recorded
CPU source-build attempts before an x86 deferral; source that proves unavoidable CUDA can defer
earlier. Failed commands, logs, hashes, host/environment identity, and defer evidence stay in the
append-only attempt history.

Inspect progress without taking the writer role:

```bash
python -m menagerie.crawler status --intake "$INTAKE" --full --verify-partition
```

## Scheduler cutover and rollback (waves <-> pipelined)

`--scheduler pipelined` (or `MENAGERIE_SCHEDULER=pipelined`) runs the per-model execute-first
scheduler; the default remains `waves`. Records are scheduler-agnostic, so switching direction
mid-campaign is the same binary with the other flag against the unchanged ledgers -- the mode-switch
matrix in `tests/test_scheduler_under_fire.py` pins that a leg killed under either mode is finished
by the other to the wave-reference close. The ceremony around a deliberate switch:

1. Stop the running driver normally (SIGTERM drains; never SIGKILL a healthy leg).
2. Run the projection-only shadow audit; it must explain every intake ID and exits 2 otherwise:

   ```bash
   python -m menagerie.crawler.tools.scheduler_shadow_audit \
     --records-root menagerie/crawler/records --intake-root "$INTAKE" \
     --runtime-root <runtime-root> --author-model <model> --checker-model <model>
   ```

3. Start the next leg under the target mode. The driver itself appends the durable
   `scheduler-cutover` / `scheduler-rollback` operational event (old/new mode, ledger-derived
   occurrence, canonical tail counts/digests) on the first leg whose mode differs from the last
   recorded one -- provenance an auditor uses to bracket which legs ran under which mode.

Independently of any switch, EVERY leg appends exactly one `leg-config` operational event before its
first admission decision: the effective scheduler, both lane widths, the author/checker models, the
selection scope (`only_unsettled` / `only_status`), the named-lane stable IDs, and a `config_digest`
that content-addresses the whole set. It is appended before the doors, so a leg that refuses (a
declined award-holds census, a review pause) has still attested what it was launched as; the event is
per-occurrence, so an identical relaunch is a second row rather than a dedup into the first. "Which
config did that leg run under?" is a ledger read, not a shell-history reconstruction.

`--only-status` re-validation refuses under `pipelined` (typed); run it under `waves` -- same
binary, same records. Neither direction rewrites history and no reverse migration exists.

## Tiny-model acceptance dry-run

The opt-in acceptance harness exercises the real driver and isolated worker on four tiny local
PyTorch architectures across the ten rows required by the frozen metadata-batch minimum. It uses the
selected lock-built fixture prefix, shipped environment binder/compiler/supervisor, canonical attempt
sink, and reducer. Only the author, checker, and notifier remain deterministic. The explicit prefix
must have the exact fixture lock, resolved export, export digest, and probe receipt artifacts beside
it; there is no current-interpreter fallback. The commands write campaign state only beneath the
selected disposable root and print the canonical current funnel plus structured acceptance status as
JSON.

```bash
DRY_RUN_ENV_PREFIX=/path/to/round19-real-environment/prefix
python -m menagerie.crawler run --dry-run --dry-run-root /tmp/menagerie-crawler-dry-run \
  --dry-run-environment-prefix "$DRY_RUN_ENV_PREFIX" \
  --review-checkpoint-at 2 --progress-milestones 3
python -m menagerie.crawler resume --dry-run --dry-run-root /tmp/menagerie-crawler-dry-run \
  --dry-run-environment-prefix "$DRY_RUN_ENV_PREFIX" \
  --review-checkpoint-at 2 --progress-milestones 3 --after-review
```

The first command must exit with the documented paused code after two real `runs` revisions. The
resume command succeeds only after all ten expected rows have authenticated forward attempts and
current `runs` revisions. A terminal partition of source failures is an acceptance failure and exits
nonzero even though the deterministic driver itself reached terminality.

Environment freshness checks cover the complete sealed tree and exact external targets with full mode,
size, `mtime_ns`, `ctime_ns`, device, and inode triggers. A hardlink clone may change shared-inode ctime;
the first subsequent check rehashes once and, when content is byte-identical, refreshes only the local
authority's cheap baseline without changing its authority or generation. A changed content digest
invalidates the authority before spawn. Only a privileged/out-of-band actor preserving every cheap
field, a release-rejected coarse filesystem, or the final verify/read TOCTOU can defer detection.

On Linux, host ELF transport is a named exact capability derived from the selected interpreter's
`ldd` inventory. The same frozen member set and digest drive bubblewrap's read-only mounts and the
parent strace classifier after the exact worker-interpreter `execve`. `/lib*` and `/usr/lib*` remain
loader lookup locations, not read authorities; an unlisted host library is never blessed merely
because its pathname is below one of those directories. Bubblewrap's own pre-worker loader traffic
is launcher activity outside the model worker audit phase.

## Recorded release attestations (runbook for gated CI jobs)

The hosted `crawler-round21-macos-release` and `crawler-round21-linux-release`
jobs are gated on the `MENAGERIE_RELEASE_RUNNERS` repository variable because
GitHub provisions neither an Apple-Silicon macOS runner nor the locked Linux
prefix for this repo. Until real runners are provisioned, each gated job has
this RECORDED, artifact-producing runbook equivalent -- a hand-run is only
evidence when it mints the attestation JSON and records it into the gate
verdict ledger; an unrecorded local run proves nothing.

On an Apple-Silicon macOS host (the launch platform), from a clean checkout:

```bash
PYTHONPATH=. python -m menagerie.crawler.tools.release_lock \
  --lock menagerie/crawler/envs/locks/round19-osx-arm64.lock \
  --resolved-export menagerie/crawler/envs/locks/round19-osx-arm64.resolved.json \
  --prefix "$PREFIX" --conda conda   # one-time prefix materialization

MENAGERIE_RELEASE_GATE=1 \
MENAGERIE_PLATFORM_LOCK=menagerie/crawler/envs/locks/round19-osx-arm64.lock \
MENAGERIE_REAL_ENV_PREFIX="$PREFIX" \
MENAGERIE_RELEASE_ATTESTATION="$PWD/round21-macos-attestation.json" \
PYTHONPATH=. "$PREFIX/bin/python" -m pytest -q menagerie/crawler/tests \
  -m round21_macos_real -x --tb=short

# Record the attestation into the append-only, closure-keyed gate-verdict
# ledger the invocation preflight consumes:
PYTHONPATH=. python -m menagerie.crawler.tools.gate_runner record-attestation \
  --attestation "$PWD/round21-macos-attestation.json" \
  --gate round21-macos-attestation
```

The Linux equivalent (`crawler-round21-linux-release`) substitutes the
`round19-linux-64.lock` family and `--gate round21-linux-attestation`, and runs
on the Linux box that carries the 47-node registry.
