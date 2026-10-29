# Durable text corpus (fetch-once retention)

The crawl fetches every document once and keeps it: implementation sources, paper
HTML, model cards and READMEs, docs pages, supplements, probe bodies, and the
frozen discovery envelope all outlive the scratch tree, so prose authoring and
any future derived writing never require a re-crawl.

## Layout

```
<records-root>/text-corpus/
  corpus-index.jsonl          # append-only ledger, menagerie.crawler.text-corpus-index.v1
  sha256/<xx>/<64 hex>        # content-addressed document payloads
```

Payloads are the crawl's fetched third-party bytes. They are durable-local
material: never committed to the public repository (the payload tree is
gitignored; the checkpoint allowlist does not stage it), governed per document
by the `license_disposition` mirrored from the artifact-authority license
decisions. The index ledger is ordinary canonical record material -- hash-bound
metadata only -- and participates in the campaign-merge immutability
fingerprint.

## Population

* The driver promotes at the award commit (`_promote_award_corpus` in
  `driver_receipts.py`), idempotently and digest-checked; replayed awards
  converge on the identical corpus. Digest refusals are disclosed on the
  operational ledger.
* `python -m menagerie.crawler.tools.promote_corpus --work-root .crawl-local/work
  --records-root menagerie/crawler/records` backfills work dirs the hook never
  saw (pre-corpus campaigns, terminal models).

## GC contract (binding on every reaper -- `menagerie/crawler/reaper.py` implements it)

1. **The corpus is NEVER reapable.** Payloads and index alike, under every
   policy, in every mode. Route every candidate deletion through
   `text_corpus.assert_not_corpus_path`, which refuses corpus paths typed.
   The implemented reaper additionally refuses candidates that CONTAIN the
   corpus root (either containment direction) and re-guards every deletion on
   its resolved location, so a symlinked candidate can never reach corpus
   bytes.
2. **Author scratch is reapable only once promoted.** A work dir may be removed
   only when `text_corpus.work_dir_reap_clearance(work_dir, records_root)`
   clears it. Clearance is replay-derived and producer-blind: it re-enumerates
   the work dir's retention candidates from the machine-written envelopes and
   requires each document to be physically present in the corpus with matching
   bytes plus a matching index row. No persisted flag, prior report, or
   producer-supplied path steers the verdict.
3. **Garbage collection stays report-only until separately authorized** and
   never deletes or rewrites canonical JSONL records (`TEARDOWN.md`). The
   shipped tool is `python -m menagerie.crawler.tools.reap_disk`: report-only
   by default, deleting only under an explicit `--apply` that takes the driver
   lock and recomputes the plan from scratch -- a stale or forged plan can
   never force a deletion the fresh computation would refuse.
4. **Even a cleared work dir is never removed whole.** The referenced-path
   manifest keeps the load-bearing interior protected: `repair/` charge globs
   (the live repair budget), `driver-author-artifact.json` (deleting it
   re-buys an author session), attempt `record.json`/`events.jsonl`/
   `discovery.json`/broker evidence (re-read on every re-drive), and the whole
   author custody surface including `source-cas` (evidence replay). Reap tiers
   target only per-attempt scratch, superseded staged model copies, orphaned
   diagnostics sidecars, supervisor stream logs, derived caches, and
   environment-operator crash residue.

## Verification

```
PYTHONPATH=. python -m menagerie.crawler.tools.verify_corpus \
  --records-root menagerie/crawler/records --work-root .crawl-local/work
```

Fail-closed: every index row's payload present and re-hashing to its address;
every excerpt below the work roots resolving through its `source_id` to a
corpus document its locator verifies against. Payloads without index rows are
the benign payload-first state of an interrupted promotion; they are counted,
never failed, and the next promotion or union completes them.

## Multi-host

`tools/merge_host_shards.py` unions shard corpora into the merged output:
identical digests unify to one payload, differing bytes at one address refuse
loudly, index rows dedup on their deterministic `corpus_row_id`. The shard
roots stay read-only; the merge's post-write fingerprint re-proves it.

The corpus ships with the shard: `tools/handoff_bundle.py build` freezes the
whole records tree -- corpus index AND payload tree included -- into a digest
manifest on the source host, and `verify` (or the merge itself, given the
manifest) re-proves every byte on the collector before the shard is consumed.
