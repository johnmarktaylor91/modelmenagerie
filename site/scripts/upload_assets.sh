#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Upload Model Menagerie content-addressed assets to Cloudflare R2.

Usage:
  scripts/upload_assets.sh [EXPORT_ASSETS_DIR]

Default:
  EXPORT_ASSETS_DIR=export/assets

Configuration, option A:
  Create an rclone S3 remote named "r2" that points at the target R2 bucket.

Configuration, option B:
  Set these environment variables:
    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET

The upload is idempotent for content-addressed assets. Existing object keys are
left untouched via rclone copy --ignore-existing.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

assets_dir="${1:-export/assets}"

usage
echo

if ! command -v rclone >/dev/null 2>&1; then
  echo "error: rclone is required but was not found on PATH" >&2
  exit 1
fi

if [[ ! -d "$assets_dir" ]]; then
  echo "error: asset directory not found: $assets_dir" >&2
  exit 1
fi

if rclone listremotes | grep -qx "r2:"; then
  destination="r2:"
elif [[ -n "${R2_ACCOUNT_ID:-}" && -n "${R2_ACCESS_KEY_ID:-}" && -n "${R2_SECRET_ACCESS_KEY:-}" && -n "${R2_BUCKET:-}" ]]; then
  destination="s3:${R2_BUCKET}"
  export RCLONE_CONFIG_S3_TYPE="s3"
  export RCLONE_CONFIG_S3_PROVIDER="Cloudflare"
  export RCLONE_CONFIG_S3_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
  export RCLONE_CONFIG_S3_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
  export RCLONE_CONFIG_S3_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
  export RCLONE_CONFIG_S3_ACL="private"
else
  echo "error: configure rclone remote 'r2' or set R2_ACCOUNT_ID/R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/R2_BUCKET" >&2
  exit 1
fi

echo "Copying $assets_dir -> $destination"
rclone copy "$assets_dir" "$destination" --ignore-existing --progress
