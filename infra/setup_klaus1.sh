#!/usr/bin/env bash
# One-time environment setup, run ON klaus-1:
#   bash infra/setup_klaus1.sh
# Creates ~/venvs/sprig inheriting the known-good system torch
# (2.9.1+cu128 on system Python 3.12), installs the remaining deps,
# creates the data/run directory layout, and prints the environment.
set -euo pipefail

VENV="${VENV:-$HOME/venvs/sprig}"

mkdir -p "$(dirname "$VENV")"
python3 -m venv --system-site-packages "$VENV"

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install \
  numpy pillow transformers sentencepiece protobuf tensorboard \
  matplotlib pyyaml tqdm einops pytest

mkdir -p "$HOME"/data/sprig/{proc2d,clevr,probe,t5} "$HOME/runs/sprig"

echo "== torch =="
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo "== pip list =="
"$VENV/bin/pip" list

echo "setup complete: venv at $VENV"
