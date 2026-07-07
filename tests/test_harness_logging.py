"""scalars.jsonl appender: one JSON object per log call, append-safe."""
from __future__ import annotations

import json

import train


def test_one_json_line_per_log(tmp_path):
    lg = train.ScalarsLogger(tmp_path, tensorboard=False)
    lg.log(1, {"a": 1.0, "b": 2})
    lg.log(2, {"c": 0.5, "note": "hello"})
    lg.close()

    lines = (tmp_path / "scalars.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    recs = [json.loads(ln) for ln in lines]
    assert recs[0]["step"] == 1 and recs[0]["a"] == 1.0 and recs[0]["b"] == 2.0
    assert recs[1]["step"] == 2 and recs[1]["note"] == "hello"
    assert all("time" in r for r in recs)


def test_reopen_appends(tmp_path):
    lg = train.ScalarsLogger(tmp_path, tensorboard=False)
    lg.log(1, {"a": 1.0})
    lg.close()

    lg2 = train.ScalarsLogger(tmp_path, tensorboard=False)
    lg2.log(2, {"a": 2.0})
    lg2.close()

    lines = (tmp_path / "scalars.jsonl").read_text().strip().splitlines()
    assert [json.loads(ln)["step"] for ln in lines] == [1, 2]


def test_tensor_scalars_are_serialized(tmp_path):
    import torch
    lg = train.ScalarsLogger(tmp_path, tensorboard=False)
    lg.log(3, {"t": torch.tensor(4.5)})
    lg.close()
    rec = json.loads((tmp_path / "scalars.jsonl").read_text().strip())
    assert rec["t"] == 4.5
