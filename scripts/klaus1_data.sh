#!/usr/bin/env bash
# Full dataset pipeline on klaus-1. Run from ~/code/sprig-c inside tmux:
#   bash scripts/klaus1_data.sh 2>&1 | tee -a ~/data/sprig/datagen.log
set -euo pipefail

PY="$HOME/venvs/sprig/bin/python"
DATA="$HOME/data/sprig"
cd "$HOME/code/sprig-c"
mkdir -p "$DATA/t5"

echo "[1/6] $(date -u +%H:%M:%S) generating proc2d splits + probe (30 workers)"
$PY scripts/gen_data.py --out-root "$DATA" --workers 30

echo "[2/6] $(date -u +%H:%M:%S) T5 embeddings for proc2d splits (GPU)"
for s in train val test parse_eval val_fast; do
  echo "  embedding $s"
  $PY -m sprig.data.embed_t5 --data-dir "$DATA/proc2d/$s" --device cuda --batch-size 512
done

echo "[3/6] $(date -u +%H:%M:%S) null + prompt bank + minimal pairs"
$PY -m sprig.data.embed_t5 --null-out "$DATA/t5/null.f16" --device cuda
$PY -m sprig.data.embed_t5 --prompts-out "$DATA/t5/promptbank.npz" --device cuda
$PY - <<'PYEOF'
import json, sys
sys.path.insert(0, ".")
from sprig.eval.prompts import MINIMAL_PAIRS
pl = []
for a, b, _attr in MINIMAL_PAIRS:
    pl += [a, b]
json.dump(pl, open("/tmp/minimal_pairs.json", "w"))
print("minimal pairs:", len(pl))
PYEOF
$PY -m sprig.data.embed_t5 --prompts-out "$DATA/t5/minimal_pairs.npz" \
    --prompts-file /tmp/minimal_pairs.json --device cuda

echo "[3b] converting promptbank.npz -> prompt_bank.pt"
$PY - <<'PYEOF'
import numpy as np, torch, os
d = np.load(os.path.expanduser("~/data/sprig/t5/promptbank.npz"), allow_pickle=True)
torch.save({"emb": torch.from_numpy(d["emb"]),
            "emb_len": torch.from_numpy(d["len"]).to(torch.int32)},
           os.path.expanduser("~/data/sprig/t5/prompt_bank.pt"))
print("prompt_bank.pt written:", d["emb"].shape)
PYEOF

echo "[4/6] $(date -u +%H:%M:%S) CLEVR preprocessing"
for s in train val; do
  $PY -m sprig.data.clevr.prep --clevr-root "$DATA/clevr/CLEVR_v1.0" \
      --split "$s" --out "$DATA/clevr/$s"
done

echo "[5/6] $(date -u +%H:%M:%S) T5 embeddings for CLEVR"
for s in train val; do
  $PY -m sprig.data.embed_t5 --data-dir "$DATA/clevr/$s" --device cuda --batch-size 512
done

echo "[6/6] $(date -u +%H:%M:%S) verification: holdout scan + disk usage"
$PY - <<'PYEOF'
import json, os, re
root = os.path.expanduser("~/data/sprig/proc2d")
bad = 0
HOLD = [("blue", "triangle"), ("red", "ring"), ("green", "star"), ("yellow", "cross")]
# Word-boundary regexes: "checkered ring" must NOT match "red ring".
PATS = [re.compile(r"\b%s\s+(\w+\s+){0,2}%s\b" % (c, s)) for c, s in HOLD]
for split in ("train", "val", "test", "parse_eval", "val_fast"):
    p = os.path.join(root, split, "meta.jsonl")
    with open(p) as f:
        for line in f:
            m = json.loads(line)
            for o in m.get("objects", []):
                if (o.get("color"), o.get("shape")) in HOLD:
                    bad += 1
            cap = " ".join(m.get("captions", [m.get("caption", "")]))
            for pat in PATS:
                if pat.search(cap):
                    bad += 1
print("holdout violations:", bad)
assert bad == 0, "HOLDOUT LEAKED"
PYEOF
du -sh "$DATA"/proc2d/* "$DATA"/probe "$DATA"/t5 "$DATA"/clevr/train "$DATA"/clevr/val
echo "DATAGEN_ALL_DONE $(date -u +%H:%M:%S)"
