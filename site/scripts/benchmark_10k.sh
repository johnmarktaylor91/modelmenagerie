#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Generating 10k synthetic fixture..."
python scripts/gen_fixture.py --n 10000

rm -rf dist .astro
TIME_LOG="$(mktemp)"
START="$(date +%s)"

if command -v /usr/bin/time >/dev/null 2>&1; then
  /usr/bin/time -v -o "$TIME_LOG" env TORCHLENS_SKIP_PUBLIC_FIXTURE=1 npm run build
else
  TORCHLENS_SKIP_PUBLIC_FIXTURE=1 npm run build
fi

END="$(date +%s)"
WALL_SECONDS="$((END - START))"
FILE_COUNT="$(find dist -type f | wc -l | tr -d ' ')"
DIST_SIZE="$(du -sh dist | awk '{print $1}')"
PEAK_MEM="unavailable"

if [[ -s "$TIME_LOG" ]]; then
  PEAK_MEM="$(awk -F: '/Maximum resident set size/ {gsub(/^[ \t]+/, "", $2); print $2 " KB"}' "$TIME_LOG")"
fi

rm -f "$TIME_LOG"

cat <<REPORT
10k benchmark results
wall_clock_seconds: $WALL_SECONDS
peak_memory: $PEAK_MEM
dist_file_count: $FILE_COUNT
dist_size: $DIST_SIZE
REPORT
