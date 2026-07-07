#!/usr/bin/env python3
"""gen_reference_vectors.py — build correctness test vectors for the smoke test.

Runs inside the lambda-build container. Tokenizes fixture text with the model's
tokenizer.json, runs the pre-int8 reference ONNX through the pip onnxruntime, and
writes a compact binary file (magic 'TVB1', little-endian) of token inputs plus the
reference output for each sample. The C smoke test replays these inputs against the
shipped minimal .so/.ort and compares outputs by cosine similarity.
"""

import argparse
import json
import struct

import numpy as np

MAGIC = b"TVB1"
KNOWN_INPUTS = ("input_ids", "attention_mask", "token_type_ids")

_DTYPE_TO_CODE = {
    np.dtype("float32"): 0,
    np.dtype("float16"): 1,
    np.dtype("int64"): 2,
    np.dtype("int32"): 3,
}
_CODE_TO_DTYPE = {code: dt for dt, code in _DTYPE_TO_CODE.items()}


def truncate_ids(ids, max_tokens):
    return list(ids)[:max_tokens]


def build_feeds(input_names, ids):
    """Build model feeds for whichever known inputs the model declares."""
    length = len(ids)
    feeds = {}
    for name in input_names:
        if name == "input_ids":
            feeds[name] = np.array([list(ids)], dtype=np.int64)
        elif name == "attention_mask":
            feeds[name] = np.ones((1, length), dtype=np.int64)
        elif name == "token_type_ids":
            feeds[name] = np.zeros((1, length), dtype=np.int64)
    unknown = [n for n in input_names if n not in KNOWN_INPUTS]
    if unknown:
        raise ValueError(
            f"gen_reference_vectors: unhandled model input(s) {unknown}; "
            f"build_feeds only knows {list(KNOWN_INPUTS)}"
        )
    return feeds


def write_vectors(path, samples):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(samples)))
        for sample in samples:
            feeds = sample["inputs"]
            f.write(struct.pack("<I", len(feeds)))
            for name, arr in feeds.items():
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<I", _DTYPE_TO_CODE[arr.dtype]))
                f.write(struct.pack("<I", arr.ndim))
                for dim in arr.shape:
                    f.write(struct.pack("<q", int(dim)))
                data = np.ascontiguousarray(arr).tobytes()
                f.write(struct.pack("<Q", len(data)))
                f.write(data)
            ref = np.ascontiguousarray(
                np.asarray(sample["reference"], dtype=np.float32).reshape(-1)
            )
            f.write(struct.pack("<Q", ref.size))
            f.write(ref.tobytes())


def read_vectors(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        (num_samples,) = struct.unpack("<I", f.read(4))
        samples = []
        for _ in range(num_samples):
            (num_inputs,) = struct.unpack("<I", f.read(4))
            inputs = {}
            for _ in range(num_inputs):
                (name_len,) = struct.unpack("<I", f.read(4))
                name = f.read(name_len).decode("utf-8")
                (code,) = struct.unpack("<I", f.read(4))
                (ndim,) = struct.unpack("<I", f.read(4))
                dims = [struct.unpack("<q", f.read(8))[0] for _ in range(ndim)]
                (nbytes,) = struct.unpack("<Q", f.read(8))
                raw = f.read(nbytes)
                arr = np.frombuffer(raw, dtype=_CODE_TO_DTYPE[code]).reshape(dims)
                inputs[name] = arr
            (ref_count,) = struct.unpack("<Q", f.read(8))
            # The reference payload is always float32 (write_vectors casts to it);
            # itemsize keeps the byte count in sync with that dtype.
            ref_dtype = np.dtype(np.float32)
            ref = np.frombuffer(
                f.read(ref_count * ref_dtype.itemsize), dtype=ref_dtype
            ).copy()
            samples.append({"inputs": inputs, "reference": ref})
        return samples


def read_texts(fixture_path, num_samples):
    texts = []
    with open(fixture_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text")
            if not text:
                continue
            texts.append(text)
            if len(texts) >= num_samples:
                break
    if not texts:
        raise ValueError(f"no usable 'text' records in {fixture_path}")
    return texts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate correctness test vectors")
    parser.add_argument("--model", required=True, help="Reference (pre-int8) ONNX path")
    parser.add_argument("--tokenizer", required=True, help="tokenizer.json path")
    parser.add_argument("--fixture", required=True, help="JSONL fixture with 'text' field")
    parser.add_argument("--output", required=True, help="Output .tvbin path")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--inputs-only",
        action="store_true",
        help="Emit token feeds with an empty reference payload (ref_count=0); "
        "skips reference output computation. Used by the benchmark harness, "
        "which times Run() and ignores the reference.",
    )
    args = parser.parse_args(argv)

    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer)
    # The session is still needed to learn which inputs the model declares so
    # build_feeds produces exactly those; only the reference Run() is skipped
    # under --inputs-only. The model may be a .ort artifact, which onnxruntime
    # loads the same as .onnx.
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    texts = read_texts(args.fixture, args.num_samples)
    samples = []
    for text in texts:
        ids = truncate_ids(tokenizer.encode(text).ids, args.max_tokens)
        feeds = build_feeds(input_names, ids)
        if args.inputs_only:
            reference = np.empty(0, dtype=np.float32)
        else:
            output = session.run([output_name], feeds)[0]
            reference = np.asarray(output, dtype=np.float32).reshape(-1)
        samples.append({"inputs": feeds, "reference": reference})

    write_vectors(args.output, samples)
    kind = "input-only" if args.inputs_only else "reference"
    print(f"    wrote {len(samples)} {kind} sample(s) to {args.output}")


if __name__ == "__main__":
    main()
