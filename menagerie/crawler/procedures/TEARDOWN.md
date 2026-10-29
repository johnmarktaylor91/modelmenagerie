# Safe teardown

Stop new dispatch and let the driver finish or terminate its one in-flight subprocess honestly.
Then checkpoint before removing disposable state:

```bash
python -m menagerie.crawler checkpoint --intake "$INTAKE" --verify-ledgers --verify-views
python -m menagerie.crawler teardown --target osx-arm64 --active-env --verify-disk
python -m menagerie.crawler status --intake "$INTAKE" --full --verify-partition
```

Teardown takes the same driver lock and removes only `.crawl-local/caches`,
`.crawl-local/scratch`, and, with `--active-env`, `.crawl-local/envs`. Environment lifecycle is
sequential: checkpoint an intent, remove that intent's environment and dedicated cache/scratch,
verify recovered disk space, and only then advance. Canonical records, intake snapshots, evidence,
source manifests, mirror manifests, locks, and derived code are not cleanup targets.

Rebuild disposable views after cleanup when required:

```bash
python -m menagerie.crawler.tools.rebuild_views --intake "$INTAKE/items.jsonl" --records-root menagerie/crawler/records --views-root menagerie/crawler/views --database .crawl-local/state.sqlite
```

Content-addressed-store garbage collection starts in report-only mark-and-sweep mode from committed
manifests. Do not delete any object without separate explicit authorization. Garbage collection never
deletes or rewrites canonical JSONL records.

The durable text corpus under `<records-root>/text-corpus/` is NEVER a cleanup target, and a work
dir under `.crawl-local/work/` may be reaped only once `text_corpus.work_dir_reap_clearance`
clears it. The full retention and GC contract is `procedures/TEXT_CORPUS.md`.

The implemented reaper is `python -m menagerie.crawler.tools.reap_disk`: report-only by default
(prints the tiered plan with per-tier byte totals and the protected-class summary), deleting only
under an explicit `--apply`, which takes the driver lock and recomputes the whole plan first. Its
`env-orphans` tier sweeps environment-operator crash residue (unregistered package caches and
wheelhouses, dangling prefix registrations, stale temporaries) through
`operator_environment.sweep_environment_orphans`.
