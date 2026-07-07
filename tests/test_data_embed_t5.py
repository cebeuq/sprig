"""Tests for sprig/data/embed_t5.py: order restoration under length bucketing,
packed writing, meta reading, promptbank plumbing. Tokenizer tests hit the
network (tokenizer-only, small); the real encoder runs behind the slow gate
(SPRIG_RUN_SLOW=1)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from sprig.data.embed_t5 import (
    EMB_DIM,
    embed_corpus,
    read_meta_captions,
    write_packed,
)

RUN_SLOW = os.environ.get("SPRIG_RUN_SLOW") == "1"


def fake_encoder(captions):
    """Deterministic fake: caption 'cap{i}' -> [len_i, 768] filled with i.

    Lengths vary so bucketing actually reorders. Returns (token_len,
    encode_batch, expected_lens).
    """
    index = {c: i for i, c in enumerate(captions)}
    lens = {c: (7 - (i % 7)) + 1 for i, c in enumerate(captions)}

    def token_len(text):
        return lens[text]

    def encode_batch(texts):
        return [
            np.full((lens[t], EMB_DIM), float(index[t]), dtype=np.float16)
            for t in texts
        ]

    return token_len, encode_batch, [lens[c] for c in captions]


def test_embed_corpus_restores_order():
    captions = ["cap%d" % i for i in range(37)]
    token_len, encode_batch, exp_lens = fake_encoder(captions)
    out = list(
        embed_corpus(captions, encode_batch, token_len, batch_size=4, chunk_size=10)
    )
    assert len(out) == 37
    for i, a in enumerate(out):
        assert a.dtype == np.float16
        assert a.shape == (exp_lens[i], EMB_DIM)
        assert np.all(a == np.float16(i)), "caption %d misplaced after bucketing" % i


def test_write_packed_offsets(tmp_path):
    captions = ["cap%d" % i for i in range(11)]
    token_len, encode_batch, exp_lens = fake_encoder(captions)
    emb_path = str(tmp_path / "emb.f16")
    off_path = str(tmp_path / "emb_offsets.i64")
    arrays = embed_corpus(captions, encode_batch, token_len, batch_size=3, chunk_size=5)
    offsets = write_packed(arrays, len(captions), emb_path, off_path)

    off = np.fromfile(off_path, dtype=np.int64)
    assert off.shape == (12,)
    assert np.array_equal(off, offsets)
    assert off[0] == 0
    assert np.array_equal(np.diff(off), np.asarray(exp_lens, dtype=np.int64))
    emb = np.fromfile(emb_path, dtype=np.float16).reshape(-1, EMB_DIM)
    assert emb.shape[0] == off[-1]
    for i in range(11):
        assert np.all(emb[off[i] : off[i + 1]] == np.float16(i))


def test_write_packed_count_mismatch(tmp_path):
    arrays = iter([np.zeros((2, EMB_DIM), dtype=np.float16)])
    with pytest.raises(RuntimeError):
        write_packed(arrays, 2, str(tmp_path / "e.f16"), str(tmp_path / "o.i64"))
    assert not os.path.exists(str(tmp_path / "e.f16"))


def test_read_meta_captions_single(tmp_path):
    p = tmp_path / "meta.jsonl"
    with open(p, "w") as f:
        for i in range(4):
            f.write(json.dumps({"idx": i, "caption": "c%d" % i}) + "\n")
    caps, n_var = read_meta_captions(str(p))
    assert n_var == 1
    assert caps == ["c0", "c1", "c2", "c3"]


def test_read_meta_captions_multi_variant_major(tmp_path):
    p = tmp_path / "meta.jsonl"
    with open(p, "w") as f:
        for i in range(3):
            f.write(
                json.dumps({"idx": i, "captions": ["a%d" % i, "b%d" % i, "c%d" % i]})
                + "\n"
            )
    caps, n_var = read_meta_captions(str(p))
    assert n_var == 3
    assert caps == ["a0", "a1", "a2", "b0", "b1", "b2", "c0", "c1", "c2"]


def _get_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("google-t5/t5-base")
    except Exception as e:  # download/network failure -> skip, not fail
        pytest.skip("t5-base tokenizer unavailable: %r" % e)


def test_tokenizer_truncation_and_null():
    tok = _get_tokenizer()
    long_caption = " ".join(["a red square next to a blue circle"] * 30)
    ids = tok(long_caption, truncation=True, max_length=64).input_ids
    assert len(ids) == 64
    null_ids = tok("", truncation=True, max_length=64).input_ids
    assert len(null_ids) >= 1  # at least </s>
    short = tok("a red square", truncation=True, max_length=64).input_ids
    assert 1 < len(short) <= 64


@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="set SPRIG_RUN_SLOW=1 to run the T5 encoder")
def test_t5_encoder_end_to_end(tmp_path):
    from sprig.data.embed_t5 import make_t5_encoder

    token_len, encode_batch = make_t5_encoder(device="cpu", max_len=64)
    caps = ["a red square", "", "two blue circles above a green triangle"]
    embs = encode_batch(caps)
    for c, e in zip(caps, embs):
        assert e.dtype == np.float16
        assert e.ndim == 2 and e.shape[1] == EMB_DIM
        assert e.shape[0] == token_len(c)
        assert np.isfinite(e.astype(np.float32)).all()
    assert embs[1].shape[0] >= 1  # null caption -> at least </s>
