"""Precompute frozen T5-base caption embeddings for SPRIG datasets.

CLI (run once per dataset directory; the encoder never runs during training):

    python -m sprig.data.embed_t5 --data-dir ~/data/sprig/proc2d/train
    python -m sprig.data.embed_t5 --null-out ~/data/sprig/t5/null.f16
    python -m sprig.data.embed_t5 --prompts-out ~/data/sprig/t5/promptbank.npz \
        [--prompts-file prompts.json]

--data-dir reads captions from <dir>/meta.jsonl in order and writes
packed-ragged fp16 token embeddings (valid tokens only, per the tokenizer
attention mask) plus int64 offsets [N+1]:

  * single-caption meta ({"caption": ...})      -> emb.f16 / emb_offsets.i64
  * multi-caption meta ({"captions": [c0..ck]}) -> emb{v}.f16 / emb{v}_offsets.i64
    for every variant v (or just one with --variant V).

Captions are tokenized with truncation to --max-len (64) tokens,
length-bucketed within --chunk-size (100k) caption chunks, encoded in batches
of --batch-size (512), and written back in the original caption order.

--null-out embeds the empty string (a single </s> token embedding) to a
packed f16 file — the null-caption substitute used by the dataloader (C1).

--prompts-out embeds a JSON list of prompts (from --prompts-file, else from
`sprig.eval.prompts.PROMPTS`, imported lazily) into an npz with keys
{emb: f16 [P, Lmax, 768] zero-padded, len: i32 [P], prompts: str array}.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

EMB_DIM = 768
DEFAULT_MODEL = "google-t5/t5-base"


def read_meta_captions(meta_path: str) -> Tuple[List[str], int]:
    """Read captions from meta.jsonl.

    Returns (flat_captions, n_variants): for single-caption meta the flat list
    is the N captions and n_variants == 1; for multi-caption meta the list is
    variant-major, i.e. [all c0, then all c1, ...], length N * n_variants.
    """
    singles: List[str] = []
    multis: List[List[str]] = []
    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "captions" in d:
                multis.append(list(d["captions"]))
            else:
                singles.append(d["caption"])
    if singles and multis:
        raise ValueError("meta.jsonl mixes 'caption' and 'captions' records")
    if singles:
        return singles, 1
    if not multis:
        return [], 1
    n_var = len(multis[0])
    for caps in multis:
        if len(caps) != n_var:
            raise ValueError("inconsistent number of caption variants in meta.jsonl")
    flat: List[str] = []
    for v in range(n_var):
        flat.extend(caps[v] for caps in multis)
    return flat, n_var


def embed_corpus(
    captions: Sequence[str],
    encode_batch: Callable[[List[str]], List[np.ndarray]],
    token_len: Callable[[str], int],
    batch_size: int = 512,
    chunk_size: int = 100000,
) -> Iterator[np.ndarray]:
    """Yield one [L_i, 768] f16 array per caption, in the original order.

    Within each chunk of `chunk_size` captions, indices are sorted by token
    length so batches are near-uniform in length (minimal padding waste), then
    the results are restored to the original order before yielding.
    """
    n = len(captions)
    for c0 in range(0, n, chunk_size):
        chunk = list(captions[c0 : c0 + chunk_size])
        lens = np.asarray([token_len(c) for c in chunk], dtype=np.int64)
        order = np.argsort(lens, kind="stable")
        results: List[Optional[np.ndarray]] = [None] * len(chunk)
        for b0 in range(0, len(order), batch_size):
            batch_idx = order[b0 : b0 + batch_size]
            embs = encode_batch([chunk[int(j)] for j in batch_idx])
            if len(embs) != len(batch_idx):
                raise RuntimeError("encode_batch returned wrong number of embeddings")
            for j, e in zip(batch_idx, embs):
                results[int(j)] = np.asarray(e, dtype=np.float16)
        for r in results:
            assert r is not None
            if r.ndim != 2 or r.shape[1] != EMB_DIM:
                raise RuntimeError("embedding has shape %r, expected [L, %d]" % (r.shape, EMB_DIM))
            yield r


def write_packed(
    arrays: Iterator[np.ndarray], n: int, emb_path: str, offsets_path: str
) -> np.ndarray:
    """Stream n ragged [L,768] f16 arrays to a packed file + [N+1] offsets."""
    offsets = np.zeros(n + 1, dtype=np.int64)
    tmp = emb_path + ".tmp"
    count = 0
    with open(tmp, "wb") as f:
        for i, a in enumerate(arrays):
            if i >= n:
                raise RuntimeError("more arrays than expected (n=%d)" % n)
            a.astype(np.float16).tofile(f)
            offsets[i + 1] = offsets[i] + a.shape[0]
            count += 1
    if count != n:
        os.remove(tmp)
        raise RuntimeError("expected %d arrays, got %d" % (n, count))
    os.replace(tmp, emb_path)
    offsets.tofile(offsets_path)
    return offsets


def _load_t5_encoder_nommap(model_name: str, dtype):
    """Load T5EncoderModel with weights read via plain file reads (no mmap).

    transformers/safetensors normally mmap the checkpoint and materialize
    weight pages lazily at forward time; on memory-constrained hosts (e.g. an
    8 GB Mac under pressure) those page-ins can SIGBUS the process. Reading
    the bytes up front and deserializing in memory avoids the mmap entirely.
    Only handles single-file safetensors checkpoints; callers fall back to
    from_pretrained otherwise.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load as st_load
    from transformers import AutoConfig, T5EncoderModel

    cfg = AutoConfig.from_pretrained(model_name)
    path = hf_hub_download(model_name, "model.safetensors")
    with open(path, "rb") as f:
        data = f.read()
    sd = st_load(data)
    del data
    model = T5EncoderModel(cfg)
    # The checkpoint carries the full T5; keep encoder weights only
    # (embed_tokens is tied to `shared` at construction).
    model.load_state_dict(sd, strict=False)
    model.tie_weights()
    return model.to(dtype)


def make_t5_encoder(
    model_name: str = DEFAULT_MODEL, device: str = "cpu", max_len: int = 64
) -> Tuple[Callable[[str], int], Callable[[List[str]], List[np.ndarray]]]:
    """Build (token_len, encode_batch) over a frozen T5 encoder.

    bf16 on cuda, fp32 on cpu; outputs are cast to fp16 numpy arrays holding
    valid (attention-masked) token embeddings only.
    """
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    try:
        model = _load_t5_encoder_nommap(model_name, dtype)
    except Exception:
        model = T5EncoderModel.from_pretrained(model_name, dtype=dtype)
    model = model.to(device).eval()

    def token_len(text: str) -> int:
        return len(tok(text, truncation=True, max_length=max_len).input_ids)

    def encode_batch(texts: List[str]) -> List[np.ndarray]:
        enc = tok(
            texts,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc).last_hidden_state  # [B, L, 768]
        mask = enc.attention_mask.bool()
        return [
            out[i, mask[i]].to(torch.float16).cpu().numpy() for i in range(len(texts))
        ]

    return token_len, encode_batch


def _load_prompts(prompts_file: Optional[str]) -> List[str]:
    if prompts_file is not None:
        with open(prompts_file, "r") as f:
            prompts = json.load(f)
        if not isinstance(prompts, list):
            raise ValueError("--prompts-file must contain a JSON list of strings")
        return [str(p) for p in prompts]
    from sprig.eval.prompts import PROMPTS  # lazy: eval module may not exist yet

    return list(PROMPTS)


def _embed_data_dir(
    data_dir: str,
    variant: Optional[int],
    token_len: Callable[[str], int],
    encode_batch: Callable[[List[str]], List[np.ndarray]],
    batch_size: int,
    chunk_size: int,
) -> None:
    captions, n_var = read_meta_captions(os.path.join(data_dir, "meta.jsonl"))
    n = len(captions) // n_var
    variants = range(n_var) if variant is None else [variant]
    for v in variants:
        if not (0 <= v < n_var):
            raise ValueError("--variant %d out of range (meta has %d variants)" % (v, n_var))
        caps_v = captions[v * n : (v + 1) * n]
        if n_var == 1:
            emb_path = os.path.join(data_dir, "emb.f16")
            off_path = os.path.join(data_dir, "emb_offsets.i64")
        else:
            emb_path = os.path.join(data_dir, "emb%d.f16" % v)
            off_path = os.path.join(data_dir, "emb%d_offsets.i64" % v)
        arrays = embed_corpus(caps_v, encode_batch, token_len, batch_size, chunk_size)
        offsets = write_packed(arrays, n, emb_path, off_path)
        print(
            "wrote %s: %d captions, %d tokens" % (emb_path, n, int(offsets[-1])),
            flush=True,
        )


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None, help="dataset dir with meta.jsonl")
    ap.add_argument(
        "--variant",
        type=int,
        default=None,
        help="embed only this caption variant (default: all variants in meta)",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None, help="cpu/cuda (default: auto)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=100000)
    ap.add_argument("--null-out", default=None, help="write empty-caption embedding here")
    ap.add_argument("--prompts-out", default=None, help="write promptbank.npz here")
    ap.add_argument("--prompts-file", default=None, help="JSON list of prompts")
    args = ap.parse_args(argv)

    if args.data_dir is None and args.null_out is None and args.prompts_out is None:
        ap.error("nothing to do: give --data-dir, --null-out, and/or --prompts-out")

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    token_len, encode_batch = make_t5_encoder(args.model, device, args.max_len)

    if args.data_dir is not None:
        _embed_data_dir(
            args.data_dir,
            args.variant,
            token_len,
            encode_batch,
            args.batch_size,
            args.chunk_size,
        )

    if args.null_out is not None:
        null = encode_batch([""])[0]
        os.makedirs(os.path.dirname(os.path.abspath(args.null_out)), exist_ok=True)
        null.astype(np.float16).tofile(args.null_out)
        print("wrote %s: [%d, %d]" % (args.null_out, null.shape[0], null.shape[1]))

    if args.prompts_out is not None:
        prompts = _load_prompts(args.prompts_file)
        embs = encode_batch(prompts) if prompts else []
        lens = np.asarray([e.shape[0] for e in embs], dtype=np.int32)
        lmax = int(lens.max()) if len(embs) else 1
        emb = np.zeros((len(prompts), lmax, EMB_DIM), dtype=np.float16)
        for i, e in enumerate(embs):
            emb[i, : e.shape[0]] = e
        os.makedirs(os.path.dirname(os.path.abspath(args.prompts_out)), exist_ok=True)
        np.savez(args.prompts_out, emb=emb, len=lens, prompts=np.array(prompts))
        print("wrote %s: %d prompts, Lmax=%d" % (args.prompts_out, len(prompts), lmax))


if __name__ == "__main__":
    main()
