# Frozen harvest-tier fixture

A data root for `test_harvest_tier_writes_records_and_rebuild_candidates`, frozen from main `b29bf49e`
(2026-10-08) so the live catalog pass cannot move it. It holds what `harvest_tiers.harvest_tier` reads for
three rows: the roster and June catalog rows of m9427 and m9413 (segmentation_models_pytorch 0.5.0) and m138
(diffusers), m9427's rung 0 model record and its harvester packet `f-fpn-segmentation`, and the header of
`catalog/standin-kwargs.tsv`. m9413 and m138 have no record or packet here: on main a 2a packet took each of
them after the test was written. The test links `stable_ids.jsonl` from the live data (append-only: an id
never moves). Refresh by copying those files again; never edit the copies by hand.
