#!/usr/bin/env bash
# Site QA harness: one command, a JSON verdict against the launch thresholds (qa/thresholds.json).
#
#   bash qa/run.sh                     # from site/: install, astro check, build, serve dist, check it
#   bash qa/run.sh --url https://...   # check a deployed site instead (no build, no file checks)
#   options: --out DIR (default qa/out), --skip-lighthouse, --skip-install, --pages N (crawl size, 400)
#
# Run it on a worker, never on the hub; it needs Node 22.18 or later and a Chrome or Chromium
# (CHROME=/path, default: chromium or google-chrome on PATH). docs/site.md has the run-on line.
# Writes OUT/verdict.json and prints the verdict; exits 1 when a gate fails, 2 when the run broke.
set -uo pipefail

cd "$(dirname "$0")/.."
site_dir=$(pwd)
url=""
out="qa/out"
skip_lh=0
skip_install=0
max_pages=400
while [ $# -gt 0 ]; do
  case "$1" in
    --url) url="${2%/}"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --skip-lighthouse) skip_lh=1; shift ;;
    --skip-install) skip_install=1; shift ;;
    --pages) max_pages="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$out"
out=$(cd "$out" && pwd)
log="$out/run.log"
: > "$log"
step() { echo "== $*" | tee -a "$log"; }

if [ -z "${CHROME:-}" ]; then
  for candidate in chromium chromium-browser google-chrome google-chrome-stable; do
    if command -v "$candidate" > /dev/null; then CHROME=$(command -v "$candidate"); break; fi
  done
fi
export CHROME="${CHROME:-}"
[ -n "$CHROME" ] || { echo "no Chrome found; set CHROME" >&2; exit 2; }
export CHROME_PATH="$CHROME"

if [ "$skip_install" -eq 0 ]; then
  step "install"
  { [ -n "$url" ] || npm ci --no-audit --no-fund; } >> "$log" 2>&1 || { echo "npm ci failed (see $log)" >&2; exit 2; }
  (cd qa && npm ci --no-audit --no-fund) >> "$log" 2>&1 || { echo "qa npm ci failed (see $log)" >&2; exit 2; }
fi

server_pid=""
cleanup() { [ -n "$server_pid" ] && kill "$server_pid" 2> /dev/null; }
trap cleanup EXIT

if [ -z "$url" ]; then
  step "astro check"
  npx astro check > "$out/check.log" 2>&1
  echo "{\"exit\": $?, \"errors\": $(grep -oE '[0-9]+ errors?' "$out/check.log" | tail -1 | grep -oE '[0-9]+' || echo -1)}" > "$out/check.json"
  step "build"
  start=$(date +%s)
  npm run build > "$out/build.log" 2>&1
  build_exit=$?
  echo "{\"exit\": $build_exit, \"seconds\": $(( $(date +%s) - start ))}" > "$out/build.json"
  [ "$build_exit" -eq 0 ] || { echo "build failed (see $out/build.log)" >&2; exit 2; }
  step "files, links and copy in dist"
  node qa/dist-checks.mjs dist "$out/dist.json" >> "$log" 2>&1 || { echo "dist checks broke (see $log)" >&2; exit 2; }
  port=$(( 4400 + RANDOM % 400 ))
  npx astro preview --port "$port" --host 127.0.0.1 > "$out/preview.log" 2>&1 &
  server_pid=$!
  url="http://127.0.0.1:$port"
  for _ in $(seq 1 60); do curl -sf "$url/" > /dev/null && break; sleep 1; done
  curl -sf "$url/" > /dev/null || { echo "preview server did not start (see $out/preview.log)" >&2; exit 2; }
fi

step "crawl $url"
node qa/crawl.mjs "$url" "$out/crawl.json" "$max_pages" >> "$log" 2>&1 || { echo "crawl broke (see $log)" >&2; exit 2; }
step "pages: axe, overflow, console, screenshots"
node qa/pages.mjs "$url" "$out" >> "$log" 2>&1 || { echo "page checks broke (see $log)" >&2; exit 2; }
step "share previews and structured data"
node qa/meta.mjs "$url" "$out/meta.json" >> "$log" 2>&1 || { echo "meta checks broke (see $log)" >&2; exit 2; }
if [ "$skip_lh" -eq 0 ]; then
  step "lighthouse"
  node qa/lighthouse.mjs "$url" "$out/lighthouse" >> "$log" 2>&1 || { echo "lighthouse broke (see $log)" >&2; exit 2; }
fi
step "verdict"
node qa/verdict.mjs "$out" "$site_dir/qa/thresholds.json" | tee -a "$log"
exit "${PIPESTATUS[0]}"
