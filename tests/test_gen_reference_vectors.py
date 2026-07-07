"""Tests for scripts/gen_reference_vectors.py."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

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
