# Menagerie public metadata CSV -- schema brainstorm CONTEXT

## THE GOAL
Design the column schema for the **public per-model metadata CSV** that ships on the menagerie website
(modelmenagerie.ai). ONE ROW PER MODEL, ~11,670 models -- "every public neural-net architecture ever
described, in one CSV." It must be the **definitive one-stop metadata reference**: as COMPREHENSIVE and
USABLE as possible -- every column worth having, named clearly, ordered/grouped sensibly, sane types +
units, explicit nullability. A researcher, practitioner, or data-scientist should be able to open this CSV
and filter/sort/learn about every model.

We are at the BRAINSTORM stage: propose MORE columns than we'll keep (union now, cut in review). Be
exhaustive. The output is a column LIST + structure, not the data.

**COMPANION DELIVERABLE (JMT 2026-06-25):** the CSV ships with a human-readable **DATA DICTIONARY** doc,
also on the website -- one entry per column clearly explaining: name, type + unit, what it means, where it
comes from (source/derivation), how to interpret it, nullability, and example values. The per-column rows
of this schema ARE the seed for that dictionary; the schema must carry enough description per column to
generate a clear public-facing dictionary entry.

## WHAT WE ALREADY HAVE (explore the live data model before brainstorming)
- **Verification ledger** -- `menagerie/ledger.py` (`VerificationRun` dataclass + `verification_runs`
  table + `current_verification` view): stable_id, recipe_revision_sha256, name, zoo, variant, scope,
  status, forward_pass, backward_pass, metadata_ok, n_ops, graph_shape_hash, svg_sha256, torchlens_version,
  torch_version, python_version, device, runner_host, timings, peak_rss_mb, error_class/message, ...
- **Catalog + recipes** -- `menagerie/catalog.py` (`CatalogRow`, `load_rows`, the `models` table) +
  `menagerie/data/master_catalog.jsonl`: name, family, zoo, variant, constructor_call, tags, verified, ...
- **TorchLens Trace** (per-model STRUCTURE we uniquely captured) -- `torchlens/data_classes/`: op count +
  types, module tree, params/buffers (counts + shapes), depth/width, recurrence/loops, branching, FLOPs,
  activation memory, the various graph/architecture hashes (graph_shape_hash shape-blind, raw_event_shape
  shape-aware, the deterministic barcode, recipe_revision_sha256 constructor-identity).
- **Classics** -- `menagerie/classics/`: ~300+ hand-built historical architectures with provenance.
- Identity facts (already decided this sprint): recipe_revision_sha256 = constructor-only (input
  EXCLUDED); architecture-hash refinement = param-shape IN, input-induced activation-shape OUT.

## BRAINSTORM TASK (your lens is in your dispatch brief; also do a broad sweep)
Propose every column worth having. For EACH column give: **name** (snake_case, final-quality) | **type +
unit** | **one-line description** | **source/derivation** (ledger / catalog / trace-derived / external
lookup / computed) | **priority** (core / nice / stretch) | **nullability**. Group columns into logical
SECTIONS and propose a column ORDER. Call out naming/units conventions. Flag anything that needs an
external source (paper year, license, authors) vs already-in-hand. Note any column that is the menagerie's
UNIQUE value (only we have it because we traced every model). Disagreements across labs are signal -- be
opinionated.
