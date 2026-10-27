"""Generate the SafeTensors schema consumed by src/metadata.zig, without Zig."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np
from safetensors import safe_open
import torch

from main import CHUNK_ROWS, decode_chunk, truncate_bf16_bits, validate_layout


GATE_PREFIX = "layers.0.mlp.gate_proj"
DEFAULT_SHAPE = (6144, 2048)
# Default matches src/metadata.zig, which parses exactly layers 0..15.
DEFAULT_LAYERS = 16
DTYPE_ITEMSIZE = {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4}


def projection_shapes(shape):
    """Default shapes match the model header; scale its hidden/MLP dimensions."""
    intermediate, hidden = shape
    if intermediate <= 0 or intermediate % 16 or hidden <= 0 or hidden % 32:
        raise ValueError("shape requires positive rows divisible by 16 and columns divisible by 32")
    return {
        "mlp.down_proj": (hidden, intermediate),
        "mlp.gate_proj": (intermediate, hidden),
        "mlp.up_proj": (intermediate, hidden),
        "self_attn.k_proj": (hidden // 2, hidden),
        "self_attn.o_proj": (hidden, hidden * 3 // 2),
        "self_attn.q_proj": (hidden * 3 // 2, hidden),
        "self_attn.v_proj": (hidden // 2, hidden),
    }


def schema_entries(shape, layers=DEFAULT_LAYERS):
    """Yield (name, (dtype, dims)) in the exact order tensors are generated."""
    if layers < 1:
        raise ValueError("layers must be a positive integer")
    shapes = projection_shapes(shape)
    for layer in range(layers):
        for norm in ("input_layernorm", "post_attention_layernorm"):
            yield f"layers.{layer}.{norm}.weight", ("BF16", [shape[1]])
        for projection, (rows, cols) in shapes.items():
            prefix = f"layers.{layer}.{projection}"
            yield prefix + ".weight", ("U8", [rows, cols // 2])
            yield prefix + ".weight_scale", ("F8_E4M3", [rows, cols // 16])
            yield prefix + ".input_scale", ("F32", [])
            yield prefix + ".weight_scale_2", ("F32", [])


def expected_schema(shape, layers=DEFAULT_LAYERS):
    return dict(schema_entries(shape, layers))


def iter_tensors(shape=DEFAULT_SHAPE, seed=42, layers=DEFAULT_LAYERS):
    """Yield (name, tensor) in schema order, one tensor materialized at a time.

    The NumPy PCG64 draw order matches the historical buffered generator, so a
    given (shape, seed, layers) reproduces identical tensor payloads.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    for name, (dtype, dims) in schema_entries(shape, layers):
        if dtype == "BF16":  # Layer-norm weights: uniform FP32 truncated to BF16 bits.
            fp32 = torch.from_numpy(rng.uniform(0.5, 1.5, dims[0]).astype(np.float32))
            yield name, truncate_bf16_bits(fp32).view(torch.bfloat16)
        elif dtype == "U8":  # Packed E2M1 nibble pairs.
            # Uniform random bytes are two independent uniform random nibbles.
            packed = rng.integers(0, 256, size=dims, dtype=np.uint8)
            # Guarantee every E2M1 code, including signed zeros, at both ends.
            codes = np.arange(16, dtype=np.uint8)
            packed.flat[:8] = codes[::2] | (codes[1::2] << 4)
            reversed_codes = codes[::-1]
            packed.flat[-8:] = reversed_codes[::2] | (reversed_codes[1::2] << 4)
            yield name, torch.from_numpy(packed)
        elif dtype == "F8_E4M3":  # Block scales.
            scale_bits = rng.integers(1, 127, size=dims, dtype=np.uint8)
            # All positive finite E4M3FN scales: subnormals through 448, no NaNs.
            coverage = min(126, scale_bits.size)
            scale_bits.flat[:coverage] = np.arange(1, coverage + 1, dtype=np.uint8)
            yield name, torch.from_numpy(scale_bits).view(torch.float8_e4m3fn)
        else:  # F32 scalar input/global scales.
            yield name, torch.tensor(rng.uniform(0.2, 1.2), dtype=torch.float32)


def generate_tensors(shape=DEFAULT_SHAPE, seed=42, layers=DEFAULT_LAYERS):
    """Materialize every tensor at once; prefer iter_tensors or write_dataset."""
    return dict(iter_tensors(shape, seed, layers))


def tensor_payload(tensor, dtype):
    """Return the tensor storage as a little-endian NumPy array ready to write."""
    if dtype == "F8_E4M3":
        return tensor.view(torch.uint8).numpy()
    if dtype == "BF16":
        return tensor.view(torch.uint16).numpy().astype("<u2", copy=False)
    if dtype == "F32":
        return tensor.numpy().astype("<f4", copy=False)
    return tensor.numpy()  # U8: byte order is irrelevant.


def write_dataset(path, shape=DEFAULT_SHAPE, seed=42, layers=DEFAULT_LAYERS):
    """Stream the dataset to a SafeTensors file one tensor at a time.

    Only the JSON header and one payload (plus a single transient copy) are
    held in memory instead of the whole dataset. Payloads follow iter_tensors
    order, so (shape, seed, layers) reproduces identical tensor contents; the
    on-disk arrangement is free to differ from safetensors.torch.save_file.
    """
    entries = list(schema_entries(shape, layers))
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, dims) in entries:
        size = DTYPE_ITEMSIZE[dtype] * int(np.prod(dims))
        header[name] = {"dtype": dtype, "shape": dims, "data_offsets": [offset, offset + size]}
        offset += size
    header_blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    try:
        with temporary.open("wb") as stream:
            stream.write(struct.pack("<Q", len(header_blob)))
            stream.write(header_blob)
            for (_, (dtype, _)), (_, tensor) in zip(entries, iter_tensors(shape, seed, layers)):
                stream.write(tensor_payload(tensor, dtype).tobytes())
            if stream.tell() != 8 + len(header_blob) + offset:
                raise RuntimeError("streamed payload does not match the header size")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)


def read_header(path):
    with Path(path).open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(header_size)), header_size + 8


def validate_dataset(path, shape=DEFAULT_SHAPE, layers=DEFAULT_LAYERS, reference_output=None):
    """Read back every tensor and decode every projection using the reference.

    Optionally write only layer 0 gate_proj's raw little-endian BF16 bits, as
    main.py does. No extra reference tensors are needed in the SafeTensors file.
    """
    schema = expected_schema(shape, layers)
    header, data_start = read_header(path)
    if header.get("__metadata__") != {"format": "pt"}:
        raise ValueError("Expected metadata format=pt")
    # safe_open validates the actual SafeTensors container and data offsets.
    with safe_open(path, framework="pt", device="cpu") as dataset:
        if set(dataset.keys()) != set(schema):
            raise ValueError(f"Tensor names do not match the expected {layers}-layer schema")
        next_offset = 0
        for name in sorted(schema, key=lambda key: header[key]["data_offsets"][0]):
            dtype, dims = schema[name]
            entry = header[name]
            if (entry["dtype"], entry["shape"]) != (dtype, dims):
                raise ValueError(f"Incorrect dtype/shape for {name}: {entry}")
            size = int(np.prod(dims)) * DTYPE_ITEMSIZE[dtype]
            if entry["data_offsets"] != [next_offset, next_offset + size]:
                raise ValueError(f"Incorrect payload size/offsets for {name}")
            next_offset += size
        if data_start + next_offset != Path(path).stat().st_size:
            raise ValueError("Unexpected file size")

        output_context = Path(reference_output).open("wb") if reference_output else nullcontext(None)
        values_checked = 0
        with torch.inference_mode(), output_context as output:
            for layer in range(layers):
                for projection in projection_shapes(shape):
                    prefix = f"layers.{layer}.{projection}"
                    packed = dataset.get_tensor(prefix + ".weight")
                    scales = dataset.get_tensor(prefix + ".weight_scale")
                    global_scale = dataset.get_tensor(prefix + ".weight_scale_2")
                    validate_layout(packed, scales, global_scale)
                    if not torch.isfinite(global_scale).all() or not 0 < global_scale.item() < 2:
                        raise ValueError(f"Invalid global scale for {prefix}")
                    scale_bits = scales.view(torch.uint8).numpy()
                    if not np.all((scale_bits >= 1) & (scale_bits <= 126)):
                        raise ValueError(f"Invalid positive finite E4M3 scales for {prefix}")
                    coverage = min(126, scale_bits.size)
                    if not np.array_equal(scale_bits.flat[:coverage], np.arange(1, coverage + 1)):
                        raise ValueError(f"Missing representative E4M3 scales for {prefix}")
                    seen = np.zeros(16, dtype=bool)
                    for start in range(0, packed.shape[0], CHUNK_ROWS):
                        part = packed[start:start + CHUNK_ROWS]
                        raw = part.numpy()
                        seen[np.unique(raw & 15)] = True
                        seen[np.unique(raw >> 4)] = True
                        _, _, fp32 = decode_chunk(part, scales[start:start + CHUNK_ROWS], global_scale)
                        if not torch.isfinite(fp32).all():
                            raise ValueError(f"Non-finite reference output for {prefix}")
                        bits = truncate_bf16_bits(fp32).numpy()
                        # Check exact upper FP32 bits independently of Torch's integer conversion.
                        expected = (fp32.numpy().view(np.uint32) >> 16).astype(np.uint16)
                        if not np.array_equal(bits, expected):
                            raise ValueError(f"BF16 truncation mismatch for {prefix}")
                        # BF16 interpreted back as FP32 must retain those exact bits (including -0).
                        restored = torch.from_numpy(bits).view(torch.bfloat16).float().numpy().view(np.uint32)
                        if not np.array_equal(restored, expected.astype(np.uint32) << 16):
                            raise ValueError(f"BF16 bit-pattern roundtrip failed for {prefix}")
                        if output is not None and prefix == GATE_PREFIX:
                            bits.astype("<u2", copy=False).tofile(output)
                        values_checked += fp32.numel()
                    if not seen.all():
                        raise ValueError(f"Missing E2M1 encodings for {prefix}")
        return values_checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("model.safetensors"))
    parser.add_argument("--shape", nargs=2, type=int, default=DEFAULT_SHAPE, metavar=("ROWS", "COLS"))
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS,
                        help="number of transformer layers (default: %(default)s; the Zig benchmark needs 16)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true", help="read back and reference-decode every projection")
    parser.add_argument("--validate-only", action="store_true", help="validate --output without generating it")
    parser.add_argument("--reference-output", type=Path, help="also write raw truncated BF16 for layer 0 gate_proj; implies validation")
    args = parser.parse_args()
    try:
        projection_shapes(args.shape)
        if args.layers < 1:
            raise ValueError("layers must be positive")
        if args.seed < 0:
            raise ValueError("seed must be nonnegative")
        if args.reference_output and args.output.resolve() == args.reference_output.resolve():
            raise ValueError("reference output must differ from the SafeTensors output")
    except ValueError as exc:
        parser.error(str(exc))
    if args.layers < DEFAULT_LAYERS:
        print(f"warning: the Zig benchmark loader requires layers 0..{DEFAULT_LAYERS - 1}; "
              f"a {args.layers}-layer file is only usable by the Python tools", file=sys.stderr)
    torch.set_num_threads(1)
    if not args.validate_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_dataset(args.output, args.shape, args.seed, args.layers)
        print(f"Generated {args.output}: {args.output.stat().st_size:,} bytes, "
              f"shape={tuple(args.shape)}, layers={args.layers}, seed={args.seed}")
    if args.validate or args.validate_only or args.reference_output:
        if args.reference_output:
            args.reference_output.parent.mkdir(parents=True, exist_ok=True)
        count = validate_dataset(args.output, args.shape, args.layers, args.reference_output)
        print(f"Validated {30 * args.layers} tensors and {count:,} decoded values with the Python reference (BF16 truncation).")
        if args.reference_output:
            print(f"Reference BF16: {args.reference_output}")


if __name__ == "__main__":
    main()
