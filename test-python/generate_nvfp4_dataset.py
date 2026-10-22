"""Generate the SafeTensors schema consumed by src/metadata.zig, without Zig."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import struct

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from main import CHUNK_ROWS, decode_chunk, truncate_bf16_bits, validate_layout


GATE_PREFIX = "layers.0.mlp.gate_proj"
DEFAULT_SHAPE = (6144, 2048)


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


def expected_schema(shape):
    schema = {}
    for layer in range(16):
        for norm in ("input_layernorm", "post_attention_layernorm"):
            schema[f"layers.{layer}.{norm}.weight"] = ("BF16", [shape[1]])
        for projection, (rows, cols) in projection_shapes(shape).items():
            prefix = f"layers.{layer}.{projection}"
            schema[prefix + ".input_scale"] = ("F32", [])
            schema[prefix + ".weight_scale_2"] = ("F32", [])
            schema[prefix + ".weight"] = ("U8", [rows, cols // 2])
            schema[prefix + ".weight_scale"] = ("F8_E4M3", [rows, cols // 16])
    return schema


def generate_tensors(shape=DEFAULT_SHAPE, seed=42):
    shapes = projection_shapes(shape)
    rng = np.random.Generator(np.random.PCG64(seed))
    tensors = {}
    for layer in range(16):
        for norm in ("input_layernorm", "post_attention_layernorm"):
            fp32 = torch.from_numpy(rng.uniform(0.5, 1.5, shape[1]).astype(np.float32))
            tensors[f"layers.{layer}.{norm}.weight"] = truncate_bf16_bits(fp32).view(torch.bfloat16)
        for projection, (rows, cols) in shapes.items():
            prefix = f"layers.{layer}.{projection}"
            # Uniform random bytes are two independent uniform random nibbles.
            packed = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
            # Guarantee every E2M1 code, including signed zeros, at both ends.
            codes = np.arange(16, dtype=np.uint8)
            packed.flat[:8] = codes[::2] | (codes[1::2] << 4)
            reversed_codes = codes[::-1]
            packed.flat[-8:] = reversed_codes[::2] | (reversed_codes[1::2] << 4)
            scale_bits = rng.integers(1, 127, size=(rows, cols // 16), dtype=np.uint8)
            # All positive finite E4M3FN scales: subnormals through 448, no NaNs.
            coverage = min(126, scale_bits.size)
            scale_bits.flat[:coverage] = np.arange(1, coverage + 1, dtype=np.uint8)
            tensors[prefix + ".weight"] = torch.from_numpy(packed)
            tensors[prefix + ".weight_scale"] = torch.from_numpy(scale_bits).view(torch.float8_e4m3fn)
            tensors[prefix + ".input_scale"] = torch.tensor(rng.uniform(0.2, 1.2), dtype=torch.float32)
            tensors[prefix + ".weight_scale_2"] = torch.tensor(rng.uniform(0.2, 1.2), dtype=torch.float32)
    return tensors


def read_header(path):
    with Path(path).open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(header_size)), header_size + 8


def validate_dataset(path, shape=DEFAULT_SHAPE, reference_output=None):
    """Read back every tensor and decode every projection using the reference.

    Optionally write only layer 0 gate_proj's raw little-endian BF16 bits, as
    main.py does. No extra reference tensors are needed in the SafeTensors file.
    """
    schema = expected_schema(shape)
    header, data_start = read_header(path)
    if header.get("__metadata__") != {"format": "pt"}:
        raise ValueError("Expected metadata format=pt")
    # safe_open validates the actual SafeTensors container and data offsets.
    with safe_open(path, framework="pt", device="cpu") as dataset:
        if set(dataset.keys()) != set(schema):
            raise ValueError("Tensor names do not match the 16-layer Zig loader schema")
        next_offset = 0
        for name in sorted(schema, key=lambda key: header[key]["data_offsets"][0]):
            dtype, dims = schema[name]
            entry = header[name]
            if (entry["dtype"], entry["shape"]) != (dtype, dims):
                raise ValueError(f"Incorrect dtype/shape for {name}: {entry}")
            size = int(np.prod(dims)) * {"U8": 1, "F8_E4M3": 1, "F32": 4, "BF16": 2}[dtype]
            if entry["data_offsets"] != [next_offset, next_offset + size]:
                raise ValueError(f"Incorrect payload size/offsets for {name}")
            next_offset += size
        if data_start + next_offset != Path(path).stat().st_size:
            raise ValueError("Unexpected file size")

        output_context = Path(reference_output).open("wb") if reference_output else nullcontext(None)
        values_checked = 0
        with torch.inference_mode(), output_context as output:
            for layer in range(16):
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true", help="read back and reference-decode every projection")
    parser.add_argument("--validate-only", action="store_true", help="validate --output without generating it")
    parser.add_argument("--reference-output", type=Path, help="also write raw truncated BF16 for layer 0 gate_proj; implies validation")
    args = parser.parse_args()
    try:
        projection_shapes(args.shape)
        if args.seed < 0:
            raise ValueError("seed must be nonnegative")
        if args.reference_output and args.output.resolve() == args.reference_output.resolve():
            raise ValueError("reference output must differ from the SafeTensors output")
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(1)
    if not args.validate_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tensors = generate_tensors(args.shape, args.seed)
        save_file(tensors, args.output, metadata={"format": "pt"})
        del tensors
        print(f"Generated {args.output}: {args.output.stat().st_size:,} bytes, shape={tuple(args.shape)}, seed={args.seed}")
    if args.validate or args.validate_only or args.reference_output:
        if args.reference_output:
            args.reference_output.parent.mkdir(parents=True, exist_ok=True)
        count = validate_dataset(args.output, args.shape, args.reference_output)
        print(f"Validated 480 tensors and {count:,} decoded values with the Python reference (BF16 truncation).")
        if args.reference_output:
            print(f"Reference BF16: {args.reference_output}")


if __name__ == "__main__":
    main()
