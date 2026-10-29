# Round 21 Conformance Registry

`conformance-round21.json` is the permanent VS11 totality registry for the
menagerie crawler release gate. It maps each Round-19 clause, Round-19
invariant, Round-20 finding, environment/shutdown/handoff matrix cell,
substitution-evasion class, preservation row, resolved decision, disagreement,
living-plan cross-reference, acceptance item, and deliberate-reversion cell to
real pytest proof nodes.

Each record is proof-bearing. There are no waiver, planned, aggregate-only, or
non-proof-bearing records. The fields mean:

- `clause_id`: stable normative ID from the Round-19/Round-21 plan.
- `source_locator`: public locator for the obligation.
- `invariant_ids` and `finding_ids`: exact cross-reference IDs covered by the
  record.
- `real_node_ids`: expanded pytest node IDs that execute the shipped real-prefix
  crawler path.
- `structural_node_ids`: structural inventories that supplement, but never
  replace, real nodes.
- `host`: `linux`, `macos`, or `both` for release attestation coverage.
- `expected_outcome`: always `passed`.
- `real_prefix` and `shipped_compiler`: always `true`.
- `deliberate_reversion_ids`: D01-D29 mutations that must make a named gate red.

`menagerie/crawler/tests/test_release_conformance_composition.py` is the
executable reader. `test_conformance_registry_is_total_and_executed`
asserts the checked-in JSON is exactly the extracted normative set, every record
has a real node, every node collects with its expanded parameters, every node is
inside the VS1 anti-substitution scan, and the required exact ID sets are
present. `test_ci_attestations_cover_registry_without_skip` consumes the
Linux and macOS release attestations plus the deliberate-reversion result and
requires every applicable node to have passed with no skipped, xfailed, failed,
or uncollected nodes.

The deliberate-reversion runner is
`menagerie/crawler/tools/round21_reversions.py`. It copies the repository to a
disposable checkout for each D-cell, applies one stable JSON/YAML/file
transformation before pytest starts, and succeeds only when the mapped proof
fails for the registered reason.

## Launch-platform scope disclosure (200b)

Of the 278 proof-bearing records, 46 carry launch-platform (macOS) release
attestation coverage (`host: both`) and 232 are `linux`-only. The linux-only
scope is AUTHORSHIP, not platform necessity: those proofs were authored on
the Linux gate host and have never been evaluated for macOS portability.
Closed reason vocabulary for a linux-only row: `authored-linux-only`
(default, covers all current rows) or `linux-platform-necessity` (reserved;
a row claiming it must name the Linux-only capability it exercises).
Widening the macOS registry across the authored-linux-only rows is scheduled
between 200b and step-500 (operator schedule call). The census
`test_release_conformance_composition.py` and the scope-disclosure test keep
these counts in lockstep with the registry: 46/278 macos-covered,
232/278 linux-only.
