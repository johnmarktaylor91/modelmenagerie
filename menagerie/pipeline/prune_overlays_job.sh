# Worker job: prune overlay environments and their own uv cache, keeping every overlay a queued or claimed runner or
# stage 3 request names, then remove the store's unpack cache entries unused for its lease (a bulk reader such as a site
# export leaves GiBs there). Run at every runner start (runner_job.sh) and by the tick's reclaim step (disk_reclaim.py)
# when the worker's free disk falls under its line. Prints both JSON reports; exits with the overlay prune's status.
set -u
export MENAGERIE_ENVS=${MENAGERIE_ENVS:-$HOME/.cache/menagerie/envs}
S=$HOME/scratch/menagerie
nice -n 10 uv run --no-project --python 3.11 python -m menagerie.pipeline envs prune-overlays \
  --keep-for "$S/spool-*/new/*.json" --keep-for "$S/spool-*/claimed/*.json" \
  --keep-for "$S/spool-*/stage3/new/*.json" --keep-for "$S/spool-*/stage3/claimed/*.json"
rc=$?
nice -n 10 uv run --no-project --python 3.11 python -m menagerie.pipeline store trim-unpack \
  || echo "store trim-unpack failed (rc=$?); the overlay prune's status stands" >&2
exit $rc
