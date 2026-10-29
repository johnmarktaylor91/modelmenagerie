#!/usr/bin/env bash
# Store trim worker step (decision 6, store_trim): run the offload and pack verbs against the live store.
# usage (inside a run-on job checkout): store_trim_job.sh MODE STEP...
#   MODE: report|apply; STEP: superseded|tiles|portable-rendered|portable-twin|pack|pack-views|ceiling[:LIMIT]
# Environment: TRIM_HOST (the catalog host name), TRIM_CATALOG (the proof catalog folder, default the r3-pack copy),
# TRIM_JOURNAL and TRIM_TAG (optional: every step appends its placement lines to TRIM_JOURNAL/TRIM_TAG.placement.jsonl,
# outside the job checkout, so they outlive a failed pull; store_box reconciles and deletes it), TRIM_CEILING_GIB and
# TRIM_CEILING_MIN_AGE_DAYS (optional: the ceiling step's target and working-set age; defaults in store_ceiling.py).
# Outputs: outputs/MODE-STEP.{json,out,err,free-before,free-after} and the placement lines (outputs/MODE-STEP.placement.jsonl,
# or a copy of the journal).
# store_box daily runs it after each verified batch; moved here from the r3-pack lane's trim-job.sh.
set -uo pipefail
mode=$1; shift
export MENAGERIE_STORE=$HOME/data/menagerie/store PYTHONDONTWRITEBYTECODE=1
catalog=${TRIM_CATALOG:-$HOME/scratch/menagerie/r3-pack/catalog}
export MENAGERIE_UNPACK_CACHE=$HOME/scratch/menagerie/store-trim/unpack-$$ MENAGERIE_UNPACK_CACHE_GB=2
# The unpack cache keeps leased trees past its cap (store._evict), so a trim run can leave GiBs; it is private to
# this job and regenerable (store._unpacked rebuilds and hash-checks), so clear it on every exit.
trap 'rm -rf "$MENAGERIE_UNPACK_CACHE"' EXIT
mkdir -p outputs
journal=
if [[ -n ${TRIM_JOURNAL:-} ]]; then
  mkdir -p "$TRIM_JOURNAL"; journal=$TRIM_JOURNAL/${TRIM_TAG:?TRIM_TAG names the journal}.placement.jsonl
  trap 'rm -rf "$MENAGERIE_UNPACK_CACHE"; [[ -f $journal ]] && cp "$journal" outputs/' EXIT
fi
flag=(); [[ $mode == apply ]] && flag=(--apply)
for step in "$@"; do
  policy=${step%%:*}; limit=(); [[ $step == *:* ]] && limit=(--limit "${step#*:}")
  verb=(store offload "$policy" --extra site/site-data); [[ $policy == pack || $policy == pack-views ]] && verb=(store "$policy" --extra site/site-data)
  # The ceiling keeps the working set: what queued and running runner and stage 3 requests name, and the traces
  # the axon render job is rendering (its staged attempts' result.json files name them).
  [[ $policy == ceiling ]] && verb=(store ceiling ${TRIM_CEILING_GIB:+--target-gib "$TRIM_CEILING_GIB"} \
    ${TRIM_CEILING_MIN_AGE_DAYS:+--min-age-days "$TRIM_CEILING_MIN_AGE_DAYS"} \
    --working-set "$HOME/scratch/menagerie/spool-*/new/*.json" --working-set "$HOME/scratch/menagerie/spool-*/claimed/*.json" \
    --working-set "$HOME/scratch/menagerie/spool-*/stage3/new/*.json" --working-set "$HOME/scratch/menagerie/spool-*/stage3/claimed/*.json" \
    --working-set "$HOME/scratch/menagerie/axon/*/attempts/*/result.json")
  df -B1 --output=avail "$MENAGERIE_STORE" | tail -1 > "outputs/$mode-$policy.free-before"
  nice -n 10 uv run --no-project --python 3.11 python -m menagerie.pipeline "${verb[@]}" \
    --catalog "$catalog" --host "$TRIM_HOST" \
    --links "$HOME/scratch/menagerie/store-box" --links "$HOME/scratch/menagerie/store-mirror-20261004" \
    "${limit[@]}" "${flag[@]}" \
    --lines "${journal:-outputs/$mode-$policy.placement.jsonl}" --report "outputs/$mode-$policy.json" \
    > "outputs/$mode-$policy.out" 2> "outputs/$mode-$policy.err"
  rc=$?
  df -B1 --output=avail "$MENAGERIE_STORE" | tail -1 > "outputs/$mode-$policy.free-after"
  echo "$step rc=$rc free_after=$(cat outputs/$mode-$policy.free-after)"
  (( rc == 0 )) || exit $rc
done
