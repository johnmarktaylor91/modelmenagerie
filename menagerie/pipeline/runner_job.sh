# A spool runner job (the tick's template; `runner ensure` submits it through `run-on --restartable`): one
# `runner serve` and `stage3 serve` pair on $RUNNER_SPOOL, on the real worker store. Each serve is restarted after any
# exit until $SPOOL/STOP exists. Moved from the catalog pass's runner B job (rung 2) unchanged in behaviour, with the
# spool, workers and work root as variables and the disk guard's pause line passed to `stage3 serve`. Every other
# committed environment is built in the background once the serves run (rung 2's runner A built tf, jax, paddle and
# torch-gpu; without it a lock change left their rows infra:env_missing, 2026-10-11).
# Environment: RUNNER_SPOOL (relative to $HOME), RUNNER_WORKERS, RUNNER_WORK_ROOT (relative to $HOME),
# MENAGERIE_DISK_PAUSE_GIB (optional), MENAGERIE_TICK_RUNNER (the runner's name, how the tick finds this job).
# Outputs: outputs/{envs,runner,stage3}.log.
set -u
: "${RUNNER_SPOOL:?set by runner ensure}" "${MENAGERIE_TICK_RUNNER:?set by runner ensure}"
export MENAGERIE_ENVS=$HOME/.cache/menagerie/envs MENAGERIE_STORE=$HOME/data/menagerie/store
# View JSON, SVG and HTML go into the worker store gzipped (store.gzips_on_write; the hub's staging store never does).
export MENAGERIE_STORE_PACK_FILES=1
export MENAGERIE_WORK_ROOT=$HOME/${RUNNER_WORK_ROOT:-scratch/menagerie/r2-run/work} MENAGERIE_HOST=worker
SPOOL=$HOME/$RUNNER_SPOOL
if [ "$MENAGERIE_TICK_RUNNER" = G ]; then
  export MENAGERIE_BIG_RUNNER=1
fi
mkdir -p outputs "$SPOOL" "$MENAGERIE_WORK_ROOT"
find "$SPOOL" -maxdepth 1 -name STOP -delete
# a new job is the resize a drain waited for (or a restart): it claims again
rm -f "$SPOOL/stage3/DRAIN"
uv run --no-project --python 3.11 python -m menagerie.pipeline envs build torch-main > outputs/envs.log 2>&1 || { tail -20 outputs/envs.log; exit 1; }
# Overlays (one per recipe's extras) grew about 8 GiB a day with nothing removing them (2026-10-18): every start
# prunes them and their own uv cache, keeping those a queued or claimed request names; a failed prune never stops
# the runner.
bash menagerie/pipeline/prune_overlays_job.sh >> outputs/envs.log 2>&1 \
  || echo "$(date -u +%FT%TZ) envs prune-overlays failed" >> outputs/envs.log
build_rest() {  # every committed lock not built yet, once the serves run, one builder at a time on this machine (the
  # lock also covers the disk reading); a failed or skipped build leaves its rows infra:env_missing, which the tick
  # reports, and never stops the runner. Pixi locks need the pixi that wrote them (~/toolchains/INSTALLED.md).
  local env free pause=${MENAGERIE_DISK_PAUSE_GIB:-0}
  exec 9> "$MENAGERIE_ENVS/.build.lock" && flock 9
  while IFS= read -r env; do
    [ -f "$SPOOL/STOP" ] && break
    free=$(df -BG --output=avail "$MENAGERIE_ENVS" | tail -1 | tr -dc 0-9)
    if [ "${free:-0}" -lt $(( ${pause%.*} + 6 )) ]; then  # a CUDA torch build writes about 5 GiB
      echo "$(date -u +%FT%TZ) envs build $env skipped: ${free:-unknown} GiB free" >> outputs/envs.log; continue
    fi
    PATH=$HOME/toolchains/pixi-0.81.0:$PATH uv run --no-project --python 3.11 python -m menagerie.pipeline envs build \
      "$env" >> outputs/envs.log 2>&1 || echo "$(date -u +%FT%TZ) envs build $env failed" >> outputs/envs.log
  done < <(uv run --no-project --python 3.11 python -m menagerie.pipeline envs missing 2>> outputs/envs.log)
}
# The serves run in the torch-main build above (same lock, canary passed), not a venv synced into this
# checkout: a private venv is 1.4 GiB that stops sharing uv's cache once the cache is cleaned (2026-10-12).
PY=$(uv run --no-project --python 3.11 python -c "import os; from pathlib import Path; from menagerie.pipeline \
  import envs; print(envs.python('torch-main', root=Path(os.environ['MENAGERIE_ENVS'])))" 2>> outputs/envs.log) \
  || { echo "torch-main is not built under $MENAGERIE_ENVS"; tail -5 outputs/envs.log; exit 1; }
loop() {  # name, command...
  local name=$1; shift
  while [ ! -f "$SPOOL/STOP" ]; do
    echo "$(date -u +%FT%TZ) $name start" >> "outputs/$name.log"
    "$@" >> "outputs/$name.log" 2>&1
    echo "$(date -u +%FT%TZ) $name exited rc=$?" >> "outputs/$name.log"
    sleep 5
  done
}
loop runner $PY -m menagerie.pipeline runner serve --spool "$SPOOL" --workers "${RUNNER_WORKERS:-2}" &
serves=$!
PYTHONPATH=$PWD loop stage3 $PY -m menagerie.pipeline stage3 serve --spool "$SPOOL" --workers "${RUNNER_WORKERS:-2}" &
serves="$serves $!"
build_rest &
wait $serves  # never on a build in flight: after STOP the job ends and the next start rebuilds an unmarked prefix
echo "$(date -u +%FT%TZ) STOP seen" >> outputs/stage3.log
