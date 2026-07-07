#!/usr/bin/env bash
# One-shot remote status check, run FROM the Mac:
#   infra/status.sh [run_dir_on_klaus1]
# Shows: last scalars.jsonl line of the (given or newest) run, tmux session
# liveness, GPU utilization, and disk usage on klaus-1.
set -euo pipefail

HOST="${SPRIG_HOST:-klaus-1}"
RUN_ARG="${1:-}"

ssh "$HOST" bash -s -- "$RUN_ARG" <<'EOF'
set -uo pipefail
RUN="${1:-}"
if [[ -z "$RUN" ]]; then
  RUN="$(ls -td "$HOME"/runs/sprig/*/ 2>/dev/null | head -1 || true)"
fi
echo "== run: ${RUN:-<none>}"
if [[ -n "${RUN:-}" && -f "$RUN/scalars.jsonl" ]]; then
  echo "-- last scalars:"
  tail -n 1 "$RUN/scalars.jsonl"
else
  echo "-- no scalars.jsonl"
fi
if [[ -n "${RUN:-}" && -f "$RUN/restarts.log" ]]; then
  echo "-- last restart-log line:"
  tail -n 1 "$RUN/restarts.log"
fi
echo "== tmux:"
if tmux has-session -t sprig 2>/dev/null; then
  tmux list-windows -t sprig
else
  echo "no tmux session 'sprig'"
fi
echo "== gpu:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
echo "== disk:"
df -h "$HOME"
EOF
