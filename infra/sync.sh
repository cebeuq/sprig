#!/usr/bin/env bash
# Rsync-push the repo from the Mac to klaus-1:code/sprig-c/, recording git
# HEAD + working-tree diff into .sync_meta (copied into each run dir by
# train.py). Usage: infra/sync.sh   (override host with SPRIG_HOST=...)
set -euo pipefail

HOST="${SPRIG_HOST:-klaus-1}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="$HOST:code/sprig-c/"
META="$SRC/.sync_meta"

{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  echo "src: $SRC"
  if git -C "$SRC" rev-parse HEAD >/dev/null 2>&1; then
    echo "git_head: $(git -C "$SRC" rev-parse HEAD)"
    echo "git_status: |"
    git -C "$SRC" status --porcelain | sed 's/^/  /' || true
    echo "git_diff: |"
    git -C "$SRC" diff | sed 's/^/  /' || true
  else
    echo "git_head: none"
  fi
} > "$META"

ssh "$HOST" mkdir -p code/sprig-c

rsync -az --delete \
  --exclude '/.git/' \
  --exclude '/.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '/runs/' \
  --exclude '/local_runs/' \
  --exclude '/local_data/' \
  "$SRC/" "$DST"

echo "synced $SRC -> $DST"
