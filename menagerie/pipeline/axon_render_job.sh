# The axon feeder's render job (`axon_feed.Ops.render_submit` submits it through run-on on the worker): the
# collected cluster traces in $AXON_ATTEMPTS (relative to $HOME) are rendered again from the worker store
# (`flow.rerender`, no model runs); the ledger and store-index lines that wrote go to outputs/lines.jsonl, which the
# tick appends on main. The cluster has no Graphviz, so its batches never render.
# Environment: AXON_ATTEMPTS, AXON_IDS (space separated), MENAGERIE_MEMORY_GB.
set -u
: "${AXON_ATTEMPTS:?set by the feeder}" "${AXON_IDS:?set by the feeder}"
export MENAGERIE_ENVS=$HOME/.cache/menagerie/envs MENAGERIE_STORE=$HOME/data/menagerie/store
export MENAGERIE_STORE_PACK_FILES=1 MENAGERIE_HOST=worker MENAGERIE_WORK_ROOT=$PWD/outputs/work
mkdir -p outputs "$MENAGERIE_WORK_ROOT"
PY=$(uv run --no-project --python 3.11 python -c "import os; from pathlib import Path; from menagerie.pipeline \
  import envs; print(envs.python('torch-main', root=Path(os.environ['MENAGERIE_ENVS'])))") \
  || { echo "torch-main is not built under $MENAGERIE_ENVS"; exit 1; }
# shellcheck disable=SC2086  # the ids are one word each
PYTHONPATH=$PWD $PY -m menagerie.pipeline.axon_feed render "$HOME/$AXON_ATTEMPTS" outputs/lines.jsonl $AXON_IDS \
  > outputs/render.log 2>&1
rc=$?
grep '^{' outputs/render.log | tail -20
exit $rc
