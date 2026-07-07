"""SPRIG dataset over precomputed memmap directories (contract C1).

On-disk layout of a dataset directory (one directory per split), written by
the procgen writer (`sprig/data/procgen/writer.py`) or by CLEVR prep
(`sprig/data/clevr/prep.py`) + `sprig/data/embed_t5.py`:

    images.u8            uint8   raw memmap, [N, 64, 64, 3]
    emb.f16              float16 packed-ragged token embeddings, [total_tokens, 768]
    emb_offsets.i64      int64   [N+1]; caption i occupies rows offsets[i]:offsets[i+1]
    meta.jsonl           one JSON object per sample (caption(s), tier, objects, GT tree, ...)
    meta_offsets.i64     int64   [N+1] byte offsets of line starts into meta.jsonl
                                 (a trailing entry equal to the file size; an [N]-shaped
                                 file of line starts is also accepted)
    tier_idx/tier{t}.i64 int64   sample indices belonging to tier t (t = 0..3)

Multi-caption variant (CLEVR: 3 synthesized captions per image): instead of a
single `emb.f16`/`emb_offsets.i64` pair, the directory holds one ragged pair
per caption variant:

    emb0.f16 / emb0_offsets.i64
    emb1.f16 / emb1_offsets.i64
    emb2.f16 / emb2_offsets.i64

and each meta line stores `"captions": [c0, c1, c2]`. The dataset picks a
variant uniformly at random per visit when `train=True`, and deterministically
as `(idx + epoch) % n_variants` when `train=False` (use `set_epoch` to rotate).

Batch contract C1 (produced by `collate`):
    {image: u8 [B,64,64,3], emb: f16 [B,Lmax,768] zero-padded,
     emb_len: i32 [B], tier: i8 [B], idx: i64 [B]}

Null-caption substitution: with probability `p_null` (train only) the caption
embedding is replaced by the precomputed empty-string embedding loaded from
`null_emb_path` (packed f16, shape [L0, 768], written by
`embed_t5.py --null-out`).
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

EMB_DIM = 768
IMG_SIZE = 64
MAX_TIERS = 4


def load_null_emb(path: str) -> np.ndarray:
    """Load the packed-f16 null (empty caption) embedding -> [L0, 768]."""
    arr = np.fromfile(path, dtype=np.float16)
    if arr.size == 0 or arr.size % EMB_DIM != 0:
        raise ValueError("null embedding file %r has invalid size %d" % (path, arr.size))
    return arr.reshape(-1, EMB_DIM)


def load_tier_indices(root: str, n: Optional[int] = None) -> List[np.ndarray]:
    """Load per-tier index arrays from root/tier_idx/tier{t}.i64.

    If the directory is absent, returns a single tier containing all indices
    (requires `n`).
    """
    tier_dir = os.path.join(root, "tier_idx")
    if not os.path.isdir(tier_dir):
        if n is None:
            raise FileNotFoundError("no tier_idx/ in %r and n not given" % root)
        return [np.arange(n, dtype=np.int64)]
    tiers: List[np.ndarray] = []
    for t in range(MAX_TIERS):
        p = os.path.join(tier_dir, "tier%d.i64" % t)
        if os.path.exists(p):
            tiers.append(np.fromfile(p, dtype=np.int64))
        elif t == 0:
            tiers.append(np.zeros(0, dtype=np.int64))
        else:
            break
    return tiers


def read_meta(root: str, idx: int) -> Dict:
    """Random-access read of one meta.jsonl record using meta_offsets.i64."""
    offsets = np.fromfile(os.path.join(root, "meta_offsets.i64"), dtype=np.int64)
    with open(os.path.join(root, "meta.jsonl"), "rb") as f:
        f.seek(int(offsets[idx]))
        line = f.readline()
    return json.loads(line)


class SprigDataset(Dataset):
    """Dataset over a memmap directory; see module docstring for the layout."""

    def __init__(
        self,
        root: str,
        p_null: float = 0.0,
        null_emb_path: Optional[str] = None,
        train: bool = True,
        seed: int = 0,
        emit_obj_mask: bool = False,
    ) -> None:
        self.root = root
        self.train = train
        self.p_null = float(p_null)
        self.seed = seed
        # Object-pixel masks from GT bboxes in meta.jsonl (for the weighted
        # emission loss / object-crop resurrection). Offsets loaded eagerly;
        # the meta file handle is opened lazily per worker process.
        self.emit_obj_mask = bool(emit_obj_mask)
        self._meta_offsets: Optional[np.ndarray] = None
        self._meta_file = None
        if self.emit_obj_mask:
            self._meta_offsets = np.fromfile(
                os.path.join(root, "meta_offsets.i64"), dtype=np.int64)

        flat = np.memmap(os.path.join(root, "images.u8"), dtype=np.uint8, mode="r")
        if flat.size % (IMG_SIZE * IMG_SIZE * 3) != 0:
            raise ValueError("images.u8 in %r has size not divisible by 64*64*3" % root)
        self.images = flat.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        self.n = self.images.shape[0]

        # Embedding variants: single (emb.f16) or multi (emb0.f16, emb1.f16, ...).
        pairs: List[Tuple[str, str]] = []
        if os.path.exists(os.path.join(root, "emb.f16")):
            pairs.append(("emb.f16", "emb_offsets.i64"))
        else:
            v = 0
            while os.path.exists(os.path.join(root, "emb%d.f16" % v)):
                pairs.append(("emb%d.f16" % v, "emb%d_offsets.i64" % v))
                v += 1
        if not pairs:
            raise FileNotFoundError("no emb.f16 or emb0.f16 found in %r" % root)
        self.emb: List[np.ndarray] = []
        self.emb_offsets: List[np.ndarray] = []
        for emb_name, off_name in pairs:
            off = np.fromfile(os.path.join(root, off_name), dtype=np.int64)
            if off.shape[0] != self.n + 1:
                raise ValueError(
                    "%s has %d entries, expected N+1=%d" % (off_name, off.shape[0], self.n + 1)
                )
            emb = np.memmap(os.path.join(root, emb_name), dtype=np.float16, mode="r")
            emb = emb.reshape(-1, EMB_DIM)
            if emb.shape[0] != int(off[-1]):
                raise ValueError("%s row count %d != offsets[-1]=%d" % (emb_name, emb.shape[0], int(off[-1])))
            self.emb.append(emb)
            self.emb_offsets.append(off)
        self.n_variants = len(self.emb)

        # Per-sample tier from the tier index arrays (default tier 0).
        self.tier = np.zeros(self.n, dtype=np.int8)
        tier_dir = os.path.join(root, "tier_idx")
        if os.path.isdir(tier_dir):
            for t in range(MAX_TIERS):
                p = os.path.join(tier_dir, "tier%d.i64" % t)
                if os.path.exists(p):
                    idxs = np.fromfile(p, dtype=np.int64)
                    self.tier[idxs] = t

        self.null_emb: Optional[np.ndarray] = None
        if null_emb_path is not None:
            self.null_emb = load_null_emb(null_emb_path)
        if self.p_null > 0.0 and self.null_emb is None:
            raise ValueError("p_null > 0 requires null_emb_path")

        self._epoch = 0
        self._rng: Optional[np.random.Generator] = None

    def set_epoch(self, epoch: int) -> None:
        """Rotate the deterministic caption-variant choice (eval mode)."""
        self._epoch = int(epoch)

    def _get_rng(self) -> np.random.Generator:
        # Created lazily per worker process so forked workers do not share a
        # RNG state. Null substitution / variant choice are i.i.d. per visit
        # and are not part of the checkpointable state (the sampler is).
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = 0 if info is None else info.id
            ss = np.random.SeedSequence([self.seed, wid, os.getpid()])
            self._rng = np.random.Generator(np.random.PCG64(ss))
        return self._rng

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx = int(idx)
        image = torch.from_numpy(np.array(self.images[idx]))  # copy off the memmap

        use_null = False
        if self.train and self.p_null > 0.0:
            use_null = bool(self._get_rng().random() < self.p_null)

        if use_null:
            emb_np = self.null_emb
        else:
            if self.n_variants == 1:
                v = 0
            elif self.train:
                v = int(self._get_rng().integers(self.n_variants))
            else:
                v = (idx + self._epoch) % self.n_variants
            off = self.emb_offsets[v]
            lo, hi = int(off[idx]), int(off[idx + 1])
            emb_np = self.emb[v][lo:hi]
        emb = torch.from_numpy(np.array(emb_np, dtype=np.float16))  # copy off the memmap

        out = {
            "image": image,
            "emb": emb,
            "emb_len": torch.tensor(emb.shape[0], dtype=torch.int32),
            "tier": torch.tensor(int(self.tier[idx]), dtype=torch.int8),
            "idx": torch.tensor(idx, dtype=torch.int64),
        }
        if self.emit_obj_mask:
            out["objmask"] = self._obj_mask(idx)
        return out

    def _obj_mask(self, idx: int) -> torch.Tensor:
        """u8 [IMG_SIZE, IMG_SIZE]: 1 inside any GT object bbox, else 0."""
        if self._meta_file is None:
            self._meta_file = open(os.path.join(self.root, "meta.jsonl"), "rb")
        self._meta_file.seek(int(self._meta_offsets[idx]))
        m = json.loads(self._meta_file.readline())
        mask = torch.zeros(IMG_SIZE, IMG_SIZE, dtype=torch.uint8)
        for o in m.get("objects") or []:
            bb = o.get("bbox") or o.get("cell")
            if not bb or len(bb) != 4:
                continue
            x0 = max(0, int(math.floor(float(bb[0]))))
            y0 = max(0, int(math.floor(float(bb[1]))))
            x1 = min(IMG_SIZE, int(math.ceil(float(bb[2]))))
            y1 = min(IMG_SIZE, int(math.ceil(float(bb[3]))))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 1
        return mask


def collate(items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate to contract C1; ragged embeddings zero-padded to the batch max."""
    b = len(items)
    lens = [int(it["emb_len"]) for it in items]
    lmax = max(1, max(lens))
    emb = torch.zeros(b, lmax, EMB_DIM, dtype=torch.float16)
    for i, it in enumerate(items):
        li = lens[i]
        if li > 0:
            emb[i, :li] = it["emb"]
    out = {
        "image": torch.stack([it["image"] for it in items]),
        "emb": emb,
        "emb_len": torch.tensor(lens, dtype=torch.int32),
        "tier": torch.stack([it["tier"] for it in items]),
        "idx": torch.stack([it["idx"] for it in items]),
    }
    if "objmask" in items[0]:
        out["objmask"] = torch.stack([it["objmask"] for it in items])
    return out


class TierCurriculumSampler(Sampler):
    """Infinite sampler drawing tiers according to a stepwise weight schedule.

    schedule: list of (step_start, [w0..w_{T-1}]) pairs; the entry with the
    largest step_start <= current training step is active. `batch_size`
    converts sample draws into training steps (step = draws // batch_size), so
    schedules can be written in optimizer steps as in DESIGN.md section 7.

    Checkpointable via state_dict/load_state_dict (RNG state + draw counter).
    Iterate it from the main process (the default for torch DataLoader
    samplers) so the state advances where checkpoints are taken; with
    num_workers > 0 the prefetch queue makes resumes approximate by up to
    (num_workers * prefetch_factor) batches, which is acceptable.
    """

    def __init__(
        self,
        tier_indices: Sequence[np.ndarray],
        schedule: Sequence[Tuple[int, Sequence[float]]],
        batch_size: int = 1,
        seed: int = 0,
    ) -> None:
        self.tier_indices = [np.asarray(a, dtype=np.int64) for a in tier_indices]
        if not self.tier_indices:
            raise ValueError("tier_indices is empty")
        sched = sorted(((int(s), list(map(float, w))) for s, w in schedule), key=lambda x: x[0])
        if not sched or sched[0][0] != 0:
            raise ValueError("schedule must start at step 0")
        for _, w in sched:
            if len(w) != len(self.tier_indices):
                raise ValueError("schedule weight vectors must have one entry per tier")
        self.schedule = sched
        self.batch_size = int(batch_size)
        self.seed = seed
        self._draws = 0
        self._rng = np.random.Generator(np.random.PCG64(seed))

    def _weights_at(self, step: int) -> np.ndarray:
        w = self.schedule[0][1]
        for s0, wi in self.schedule:
            if s0 <= step:
                w = wi
            else:
                break
        w = np.asarray(w, dtype=np.float64)
        # Zero out empty tiers so we never draw from them.
        for t, arr in enumerate(self.tier_indices):
            if arr.size == 0:
                w[t] = 0.0
        total = w.sum()
        if total <= 0.0:
            raise ValueError("all active tiers at step %d are empty or zero-weight" % step)
        return w / total

    def __iter__(self) -> Iterator[int]:
        n_tiers = len(self.tier_indices)
        while True:
            step = self._draws // self.batch_size
            w = self._weights_at(step)
            t = int(self._rng.choice(n_tiers, p=w))
            arr = self.tier_indices[t]
            i = int(arr[int(self._rng.integers(arr.size))])
            self._draws += 1
            yield i

    def state_dict(self) -> Dict:
        return {"draws": self._draws, "rng_state": self._rng.bit_generator.state}

    def load_state_dict(self, state: Dict) -> None:
        self._draws = int(state["draws"])
        self._rng = np.random.Generator(np.random.PCG64(self.seed))
        self._rng.bit_generator.state = state["rng_state"]
