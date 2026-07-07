#!/usr/bin/env bash
# Crash-resilient training wrapper, run ON klaus-1 (inside tmux session
# 'sprig'):
#   bash infra/run_forever.sh configs/main64.yaml ~/runs/sprig/main64
# Restarts train.py (--resume auto) with a 60 s backoff after any non-zero
# exit; a STOP file in the run dir ends the loop (train.py also honors it
# per-step). All restarts are appended to <run_dir>/restarts.log.
set -euo pipefail

CONFIG="${1:?usage: run_forever.sh <config.yaml> <run_dir>}"
RUN_DIR="${2:?usage: run_forever.sh <config.yaml> <run_dir>}"
VENV="${VENV:-$HOME/venvs/sprig}"
BACKOFF="${BACKOFF:-60}"

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/restarts.log"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "$(ts) run_forever started (config=$CONFIG run_dir=$RUN_DIR)" >> "$LOG"

until [[ -f "$RUN_DIR/STOP" ]]; do
  echo "$(ts) launching train.py" >> "$LOG"
  set +e
  "$VENV/bin/python" train.py --config "$CONFIG" --run-dir "$RUN_DIR" --resume auto
  code=$?
  set -e
  echo "$(ts) train.py exited code=$code" >> "$LOG"
  if [[ "$code" -eq 0 ]]; then
    echo "$(ts) clean exit; stopping" >> "$LOG"
    break
  fi
  if [[ -f "$RUN_DIR/STOP" ]]; then
    break
  fi
  echo "$(ts) restarting in ${BACKOFF}s" >> "$LOG"
  sleep "$BACKOFF"
done

echo "$(ts) run_forever finished" >> "$LOG"
