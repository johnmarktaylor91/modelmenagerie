# Worker job: prune overlay environments and their own uv cache, keeping every overlay a queued or claimed runner or
# stage 3 request names. Run at every runner start (runner_job.sh) and by the tick's reclaim step (disk_reclaim.py)
# when the worker's free disk falls under its line. Prints the prune's JSON report.
set -u
export MENAGERIE_ENVS=${MENAGERIE_ENVS:-$HOME/.cache/menagerie/envs}
S=$HOME/scratch/menagerie
nice -n 10 uv run --no-project --python 3.11 python -m menagerie.pipeline envs prune-overlays \
  --keep-for "$S/spool-*/new/*.json" --keep-for "$S/spool-*/claimed/*.json" \
  --keep-for "$S/spool-*/stage3/new/*.json" --keep-for "$S/spool-*/stage3/claimed/*.json"
