"""Tests for scripts/gen_reference_vectors.py."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "gen_reference_vectors.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gen_reference_vectors", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_truncate_ids():
    module = _load_module()
    assert module.truncate_ids([1, 2, 3, 4, 5], 3) == [1, 2, 3]
    assert module.truncate_ids([1, 2], 5) == [1, 2]


def test_build_feeds_includes_token_type_ids_when_present():
    module = _load_module()
    feeds = module.build_feeds(["input_ids", "attention_mask", "token_type_ids"], [5, 6, 7])
    assert set(feeds) == {"input_ids", "attention_mask", "token_type_ids"}
    assert feeds["input_ids"].tolist() == [[5, 6, 7]]
    assert feeds["attention_mask"].tolist() == [[1, 1, 1]]
    assert feeds["token_type_ids"].tolist() == [[0, 0, 0]]
    assert feeds["input_ids"].dtype == np.int64


def test_build_feeds_omits_token_type_ids_when_absent():
    module = _load_module()
    feeds = module.build_feeds(["input_ids", "attention_mask"], [5, 6])
    assert set(feeds) == {"input_ids", "attention_mask"}


def test_build_feeds_raises_on_unknown_input():
    module = _load_module()
    with pytest.raises(ValueError):
        module.build_feeds(["input_ids", "attention_mask", "position_ids"], [5, 6, 7])


def test_write_read_vectors_roundtrip(tmp_path):
    module = _load_module()
    samples = [{
        "inputs": {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
        },
        "reference": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    }]
    p = tmp_path / "v.tvbin"
    module.write_vectors(str(p), samples)
    r = module.read_vectors(str(p))
    assert len(r) == 1
    assert list(r[0]["inputs"].keys()) == ["input_ids", "attention_mask"]
    assert r[0]["inputs"]["input_ids"].tolist() == [[1, 2, 3]]
    assert r[0]["inputs"]["input_ids"].dtype == np.int64
    assert np.allclose(r[0]["reference"], [0.1, 0.2, 0.3, 0.4], atol=1e-6)


def test_main_writes_vectors_file(monkeypatch, tmp_path):
    module = _load_module()

    class FakeEnc:
        def __init__(self, ids):
            self.ids = ids

    class FakeTokenizer:
        @classmethod
        def from_file(cls, path):
            return cls()

        def encode(self, text):
            return FakeEnc([1, 2, 3, 4])

    fake_tok_mod = types.ModuleType("tokenizers")
    fake_tok_mod.Tokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tok_mod)

    class FakeIO:
        def __init__(self, name):
            self.name = name

    class FakeSession:
        def get_inputs(self):
            return [FakeIO("input_ids"), FakeIO("attention_mask")]

        def get_outputs(self):
            return [FakeIO("last_hidden_state")]

        def run(self, output_names, feeds):
            seq = feeds["input_ids"].shape[1]
            return [np.ones((1, seq, 2), dtype=np.float32)]

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = lambda *a, **k: FakeSession()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    fixture = tmp_path / "f.jsonl"
    fixture.write_text(
        '{"text":"hello world"}\n{"text":"second sentence"}\n{"text":"third one"}\n',
        encoding="utf-8",
    )
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    model = tmp_path / "ref.onnx"
    model.write_bytes(b"x")
    out = tmp_path / "v.tvbin"

    module.main([
        "--model", str(model),
        "--tokenizer", str(tokenizer),
        "--fixture", str(fixture),
        "--output", str(out),
        "--num-samples", "2",
        "--max-tokens", "3",
    ])

    r = module.read_vectors(str(out))
    assert len(r) == 2
    # ids [1,2,3,4] truncated to 3 -> output (1,3,2) -> 6 floats
    assert r[0]["reference"].size == 6
    assert r[0]["inputs"]["input_ids"].tolist() == [[1, 2, 3]]
