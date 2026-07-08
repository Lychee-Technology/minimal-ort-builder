"""Tests for scripts/optimize_model.py — CLI argument validation."""

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "optimize_model.py"


def _run(*extra_args):
    cmd = [sys.executable, str(SCRIPT)] + list(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True)


def _load_module():
    spec = importlib.util.spec_from_file_location("optimize_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_missing_required_args_exits_nonzero():
    """Running with no args must exit non-zero (argparse error)."""
    result = _run()
    assert result.returncode != 0


def test_missing_input_file_exits_nonzero(tmp_path):
    """--input pointing at a nonexistent file must exit non-zero with a clear error."""
    result = _run(
        "--input",
        str(tmp_path / "nonexistent.onnx"),
        "--output",
        str(tmp_path / "out.onnx"),
        "--model_type",
        "bert",
        "--target_platform",
        "arm",
        "--work_dir",
        str(tmp_path / "work"),
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


def test_invalid_target_platform_exits_nonzero(tmp_path):
    """--target_platform with an invalid value must exit non-zero.
    argparse rejects the value before any file I/O, so no real input file is needed.
    """
    result = _run(
        "--input",
        str(tmp_path / "model.onnx"),
        "--output",
        str(tmp_path / "out.onnx"),
        "--model_type",
        "bert",
        "--target_platform",
        "sparc",
        "--work_dir",
        str(tmp_path / "work"),
    )
    assert result.returncode != 0


def test_help_exits_zero():
    """--help must exit 0 and mention key flags."""
    result = _run("--help")
    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--output" in result.stdout
    assert "--model_type" in result.stdout
    assert "--target_platform" in result.stdout


def test_help_mentions_optional_max_length_for_shape_specialization():
    """The CLI should still expose optional shape specialization for opt-in use."""
    result = _run("--help")
    assert result.returncode == 0
    assert "--max_length" in result.stdout
    assert "shape specialization" in result.stdout.lower()


def test_help_mentions_optional_skip_int8_quantize_flag():
    """The CLI should expose a way to skip step 4 for pre-quantized ONNX inputs."""
    result = _run("--help")
    assert result.returncode == 0
    assert "--skip-int8-quantize" in result.stdout


def test_help_mentions_reference_output_flag():
    result = _run("--help")
    assert result.returncode == 0
    assert "--reference-output" in result.stdout


def test_main_writes_reference_output_when_requested(monkeypatch, tmp_path):
    """--reference-output must persist the pre-int8 (step 3) graph."""
    module = _load_module()

    monkeypatch.setattr(module, "_step0_inline_external_data",
                        lambda i, o: Path(o).write_bytes(b"s0"))
    monkeypatch.setattr(module, "_step1_transformer_opt",
                        lambda i, o, mt: Path(o).write_bytes(b"s1"))
    monkeypatch.setattr(module, "_step2b_shape_inference",
                        lambda i, o: Path(o).write_bytes(b"s2b"))
    monkeypatch.setattr(module, "_step3_ort_graph_opt",
                        lambda i, o: Path(o).write_bytes(b"REF"))
    monkeypatch.setattr(module, "_step4_int8_quantize",
                        lambda i, o: Path(o).write_bytes(b"q"))

    input_path = tmp_path / "model.onnx"
    input_path.write_bytes(b"x")
    ref_path = tmp_path / "reference.onnx"
    out_path = tmp_path / "out.onnx"

    old_argv = sys.argv
    sys.argv = [
        str(SCRIPT),
        "--input", str(input_path),
        "--output", str(out_path),
        "--model_type", "bert",
        "--target_platform", "arm",
        "--work_dir", str(tmp_path / "work"),
        "--reference-output", str(ref_path),
    ]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    assert ref_path.read_bytes() == b"REF"




def test_main_skips_shape_specialization_when_max_length_not_provided(
    monkeypatch, tmp_path
):
    """The default pipeline should keep dynamic shapes by skipping step 2a."""
    module = _load_module()

    calls = []

    monkeypatch.setattr(
        module,
        "_step0_inline_external_data",
        lambda inp, out: calls.append(("step0", inp, out)),
    )
    monkeypatch.setattr(
        module,
        "_step1_transformer_opt",
        lambda inp, out, model_type: calls.append(("step1", inp, out, model_type)),
    )
    monkeypatch.setattr(
        module,
        "_step2a_shape_specialization",
        lambda inp, out, max_length: calls.append(("step2a", inp, out, max_length)),
    )
    monkeypatch.setattr(
        module,
        "_step2b_shape_inference",
        lambda inp, out: calls.append(("step2b", inp, out)),
    )
    monkeypatch.setattr(
        module,
        "_step3_ort_graph_opt",
        lambda inp, out: (calls.append(("step3", inp, out)), Path(out).write_bytes(b"ort"))[-1],
    )
    monkeypatch.setattr(
        module,
        "_step4_int8_quantize",
        lambda inp, out: calls.append(("step4", inp, out)),
    )

    input_path = tmp_path / "model.onnx"
    output_path = tmp_path / "optimized.onnx"
    input_path.write_bytes(b"onnx")

    old_argv = sys.argv
    sys.argv = [
        str(SCRIPT),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model_type",
        "bert",
        "--target_platform",
        "arm",
        "--work_dir",
        str(tmp_path / "work"),
    ]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    call_names = [call[0] for call in calls]
    assert call_names == ["step0", "step1", "step2b", "step3", "step4"]



def test_main_skips_step4_when_skip_int8_quantize_is_requested(monkeypatch, tmp_path):
    """Pre-quantized ONNX inputs should still run steps 0/1/2b/3 and stop before step 4."""
    module = _load_module()

    calls = []

    monkeypatch.setattr(
        module,
        "_step0_inline_external_data",
        lambda inp, out: calls.append(("step0", inp, out)),
    )
    monkeypatch.setattr(
        module,
        "_step1_transformer_opt",
        lambda inp, out, model_type: calls.append(("step1", inp, out, model_type)),
    )
    monkeypatch.setattr(
        module,
        "_step2a_shape_specialization",
        lambda inp, out, max_length: calls.append(("step2a", inp, out, max_length)),
    )
    monkeypatch.setattr(
        module,
        "_step2b_shape_inference",
        lambda inp, out: calls.append(("step2b", inp, out)),
    )
    monkeypatch.setattr(
        module,
        "_step3_ort_graph_opt",
        lambda inp, out: (calls.append(("step3", inp, out)), Path(out).write_bytes(b"ort"))[-1],
    )
    monkeypatch.setattr(
        module,
        "_step4_int8_quantize",
        lambda inp, out: calls.append(("step4", inp, out)),
    )

    input_path = tmp_path / "model.onnx"
    output_path = tmp_path / "optimized.onnx"
    input_path.write_bytes(b"onnx")

    old_argv = sys.argv
    sys.argv = [
        str(SCRIPT),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model_type",
        "bert",
        "--target_platform",
        "arm",
        "--work_dir",
        str(tmp_path / "work"),
        "--skip-int8-quantize",
    ]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    call_names = [call[0] for call in calls]
    assert call_names == ["step0", "step1", "step2b", "step3"]


def test_help_mentions_gptq4_calibration_flags():
    """The CLI should expose the calibrated 4-bit scheme and its calibration inputs."""
    result = _run("--help")
    assert result.returncode == 0
    assert "--quant-scheme" in result.stdout
    assert "gptq4" in result.stdout
    assert "--calibration-tokenizer" in result.stdout
    assert "--calibration-fixture" in result.stdout


def _base_argv(input_path, output_path, tmp_path):
    return [
        str(SCRIPT),
        "--input", str(input_path),
        "--output", str(output_path),
        "--model_type", "bert",
        "--target_platform", "arm",
        "--work_dir", str(tmp_path / "work"),
    ]


def _wire_pipeline(monkeypatch, module, calls):
    """Monkeypatch steps 0/1/2b/3 and both step-4 variants to record call names."""
    monkeypatch.setattr(module, "_step0_inline_external_data",
                        lambda inp, out: calls.append("step0"))
    monkeypatch.setattr(module, "_step1_transformer_opt",
                        lambda inp, out, mt: calls.append("step1"))
    monkeypatch.setattr(module, "_step2b_shape_inference",
                        lambda inp, out: calls.append("step2b"))
    monkeypatch.setattr(
        module, "_step3_ort_graph_opt",
        lambda inp, out: (calls.append("step3"), Path(out).write_bytes(b"ort"))[-1],
    )
    monkeypatch.setattr(module, "_step4_int8_quantize",
                        lambda inp, out: calls.append("int8"))


def test_quant_scheme_gptq4_dispatches_gptq_step(monkeypatch, tmp_path):
    """--quant-scheme gptq4 runs the GPTQ step (not int8) with calibration paths threaded."""
    module = _load_module()
    calls = []
    _wire_pipeline(monkeypatch, module, calls)

    recorded = {}

    def fake_gptq(inp, out, tok, fixture, num, max_tokens, block_size=32):
        calls.append("gptq4")
        recorded.update(tok=tok, fixture=fixture, num=num, max_tokens=max_tokens)

    monkeypatch.setattr(module, "_step4_gptq_4bit", fake_gptq)

    input_path = tmp_path / "model.onnx"
    input_path.write_bytes(b"onnx")
    tok = tmp_path / "tokenizer.json"
    tok.write_text("{}")
    fixture = tmp_path / "cal.jsonl"
    fixture.write_text('{"text": "hi"}\n')

    old_argv = sys.argv
    sys.argv = _base_argv(input_path, tmp_path / "out.onnx", tmp_path) + [
        "--quant-scheme", "gptq4",
        "--calibration-tokenizer", str(tok),
        "--calibration-fixture", str(fixture),
        "--calibration-num-samples", "2",
        "--calibration-max-tokens", "64",
    ]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    assert calls == ["step0", "step1", "step2b", "step3", "gptq4"]
    assert "int8" not in calls
    assert recorded["tok"] == tok
    assert recorded["fixture"] == fixture
    assert recorded["num"] == 2
    assert recorded["max_tokens"] == 64


def test_quant_scheme_defaults_to_int8(monkeypatch, tmp_path):
    """Omitting --quant-scheme keeps the existing dynamic int8 behavior."""
    module = _load_module()
    calls = []
    _wire_pipeline(monkeypatch, module, calls)
    monkeypatch.setattr(module, "_step4_gptq_4bit",
                        lambda *a, **k: calls.append("gptq4"))

    input_path = tmp_path / "model.onnx"
    input_path.write_bytes(b"onnx")

    old_argv = sys.argv
    sys.argv = _base_argv(input_path, tmp_path / "out.onnx", tmp_path)
    try:
        module.main()
    finally:
        sys.argv = old_argv

    assert calls == ["step0", "step1", "step2b", "step3", "int8"]


def test_skip_int8_quantize_wins_over_gptq4(monkeypatch, tmp_path):
    """--skip-int8-quantize takes precedence: no step 4 runs even with --quant-scheme gptq4."""
    module = _load_module()
    calls = []
    _wire_pipeline(monkeypatch, module, calls)
    monkeypatch.setattr(module, "_step4_gptq_4bit",
                        lambda *a, **k: calls.append("gptq4"))

    input_path = tmp_path / "model.onnx"
    input_path.write_bytes(b"onnx")

    old_argv = sys.argv
    sys.argv = _base_argv(input_path, tmp_path / "out.onnx", tmp_path) + [
        "--quant-scheme", "gptq4",
        "--skip-int8-quantize",
    ]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    assert calls == ["step0", "step1", "step2b", "step3"]


def test_gptq4_without_calibration_paths_exits_nonzero(tmp_path):
    """--quant-scheme gptq4 without existing calibration inputs must fail fast."""
    input_path = tmp_path / "model.onnx"
    input_path.write_bytes(b"onnx")
    result = _run(
        "--input", str(input_path),
        "--output", str(tmp_path / "out.onnx"),
        "--model_type", "bert",
        "--target_platform", "arm",
        "--work_dir", str(tmp_path / "work"),
        "--quant-scheme", "gptq4",
    )
    assert result.returncode != 0
    assert "gptq4" in result.stderr.lower()


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class _FakeTokenizer:
    def encode(self, text):
        # Deterministic ids from text length so truncation is observable.
        return _FakeEncoding(list(range(1, len(text) + 1)))


def test_fixture_calibration_reader_iterates_then_none_and_rewinds(monkeypatch, tmp_path):
    """The reader yields one feed dict per fixture sample, then None, and rewind() restarts."""
    module = _load_module()
    # Running optimize_model.py as a script puts scripts/ on sys.path[0], so the
    # sibling gen_reference_vectors import resolves; mirror that for the unit test.
    monkeypatch.syspath_prepend(str(SCRIPT.parent))

    fixture = tmp_path / "cal.jsonl"
    fixture.write_text('{"text": "abcd"}\n{"text": "ef"}\n{"text": "ghi"}\n')

    reader = module._FixtureCalibrationReader(
        _FakeTokenizer(),
        fixture,
        input_names=["input_ids", "attention_mask"],
        num_samples=2,
        max_tokens=3,
    )

    first = reader.get_next()
    assert set(first) == {"input_ids", "attention_mask"}
    assert first["input_ids"].tolist() == [[1, 2, 3]]  # "abcd" → 4 ids, truncated to 3
    second = reader.get_next()
    assert second["input_ids"].tolist() == [[1, 2]]  # "ef" → 2 ids
    assert reader.get_next() is None  # only 2 samples requested

    reader.rewind()
    assert reader.get_next()["input_ids"].tolist() == [[1, 2, 3]]


def test_step1_transformer_opt_disables_attention_fusion(monkeypatch, tmp_path):
    """Transformer optimization must explicitly keep attention fusion disabled."""
    module = _load_module()

    recorded = {}

    class FakeFusionOptions:
        def __init__(self, model_type):
            recorded["model_type"] = model_type
            self.enable_attention = True

    class FakeOptModel:
        def save_model_to_file(self, path):
            recorded["saved_to"] = path

    class FakeOptimizerModule:
        @staticmethod
        def optimize_model(path, model_type, optimization_options):
            recorded["input_path"] = path
            recorded["attention_enabled"] = optimization_options.enable_attention
            return FakeOptModel()

    import sys
    import types

    fake_transformers = types.ModuleType("onnxruntime.transformers")
    fake_transformers.optimizer = FakeOptimizerModule()

    fake_fusion = types.ModuleType("onnxruntime.transformers.fusion_options")
    fake_fusion.FusionOptions = FakeFusionOptions

    monkeypatch.setitem(sys.modules, "onnxruntime.transformers", fake_transformers)
    monkeypatch.setitem(
        sys.modules, "onnxruntime.transformers.fusion_options", fake_fusion
    )

    input_path = tmp_path / "in.onnx"
    output_path = tmp_path / "out.onnx"
    input_path.write_bytes(b"onnx")

    module._step1_transformer_opt(input_path, output_path, "bert")

    assert recorded["model_type"] == "bert"
    assert recorded["input_path"] == str(input_path)
    assert recorded["attention_enabled"] is False
    assert recorded["saved_to"] == str(output_path)


def test_step1_transformer_opt_copies_input_on_failure(monkeypatch, tmp_path, capsys):
    """Transformer optimization must copy the input forward if optimization fails."""
    module = _load_module()

    import sys
    import types

    class FakeFusionOptions:
        def __init__(self, model_type):
            self.enable_attention = True

    class FakeOptimizerModule:
        @staticmethod
        def optimize_model(path, model_type, optimization_options):
            raise RuntimeError("boom")

    fake_transformers = types.ModuleType("onnxruntime.transformers")
    fake_transformers.optimizer = FakeOptimizerModule()

    fake_fusion = types.ModuleType("onnxruntime.transformers.fusion_options")
    fake_fusion.FusionOptions = FakeFusionOptions

    monkeypatch.setitem(sys.modules, "onnxruntime.transformers", fake_transformers)
    monkeypatch.setitem(
        sys.modules, "onnxruntime.transformers.fusion_options", fake_fusion
    )

    input_path = tmp_path / "in.onnx"
    output_path = tmp_path / "out.onnx"
    input_path.write_bytes(b"test-model")

    module._step1_transformer_opt(input_path, output_path, "bert")

    assert output_path.read_bytes() == b"test-model"
    stderr = capsys.readouterr().err
    assert "transformer opt failed" in stderr


def test_step3_ort_graph_opt_uses_enable_all(monkeypatch, tmp_path):
    """Offline ORT graph optimization should run at ORT_ENABLE_ALL."""
    module = _load_module()

    recorded = {}

    class FakeSessionOptions:
        def __init__(self):
            self.graph_optimization_level = None
            self.optimized_model_filepath = None

    class FakeGraphOptimizationLevel:
        ORT_ENABLE_ALL = "enable_all"

    class FakeOrtModule:
        SessionOptions = FakeSessionOptions
        GraphOptimizationLevel = FakeGraphOptimizationLevel

        @staticmethod
        def InferenceSession(path, sess_options, providers):
            recorded["path"] = path
            recorded["graph_optimization_level"] = (
                sess_options.graph_optimization_level
            )
            recorded["optimized_model_filepath"] = sess_options.optimized_model_filepath
            recorded["providers"] = providers

    import sys

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrtModule)

    input_path = tmp_path / "in.onnx"
    output_path = tmp_path / "out.onnx"
    input_path.write_bytes(b"onnx")

    module._step3_ort_graph_opt(input_path, output_path)

    assert recorded["path"] == str(input_path)
    assert recorded["graph_optimization_level"] == "enable_all"
    assert recorded["optimized_model_filepath"] == str(output_path)
    assert recorded["providers"] == ["CPUExecutionProvider"]
