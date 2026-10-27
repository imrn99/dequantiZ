# Synthetic NVFP4 datasets

Run these commands from the repository root. The generator is
`test-python/generate_nvfp4_dataset.py`. It uses the existing `pyproject.toml`
and `uv.lock`: Python 3.14+, NumPy, PyTorch, and SafeTensors. There are no new
dependencies or Python projects. PyTorch is already required by the reference
and supplies native FP8/BF16 tensor types to the SafeTensors writer.

```bash
uv run --locked --project test-python python test-python/generate_nvfp4_dataset.py \
  --output model.safetensors --shape 6144 2048 --seed 42 --validate
```

`uv` selects the Python version and resolves the locked dependencies automatically.
Initial environment setup may download Python/packages; generation never downloads
model data or accesses the network. After setup, add `--offline` to `uv run` to
enforce offline execution. No manual environment activation or `pip install` is
needed. The output file and any raw `.bf16` references are ignored by Git.

`--output` defaults to `model.safetensors` in the current directory and replaces
an existing file. `--seed` defaults to 42 and selects a deterministic NumPy PCG64
stream. Identical arguments and locked dependencies produce identical file bytes.

`--layers` defaults to 16 and selects how many `layers.{0..N-1}` blocks are
generated, 30 tensors each. The Zig benchmark loader parses exactly layers
0..15: generating fewer layers prints a warning and produces a file usable only
by the Python tools. Generation streams one tensor at a time into the
SafeTensors container (JSON header first, then each payload), so peak memory
stays near the largest single tensor instead of the whole dataset.

## Existing Zig data flow

`src/main.zig` opens the hardcoded relative path `model.safetensors`, reads its
8-byte little-endian header length and JSON header, and calls
`metadata.buildLayerMetaData`. The loader parses all 16 layers before selecting
`layers.0.mlp.gate_proj` with `N_THREADS = 1`. Its required tensor names come
from `src/metadata.zig`, and default dimensions were checked against the existing
model header. The generator needs no existing model file.

For every `layers.{0..15}.{projection}`, the file contains:

| Suffix | SafeTensors dtype | Shape for a logical R × C projection |
| --- | --- | --- |
| `.weight` | `U8` | `[R, C/2]` |
| `.weight_scale` | `F8_E4M3` (E4M3FN) | `[R, C/16]` |
| `.weight_scale_2` | `F32` | `[]` (global weight scale) |
| `.input_scale` | `F32` | `[]` (required metadata; unused in weight decoding) |

The seven projections have these default logical dimensions:

| Projection | R × C |
| --- | --- |
| `mlp.down_proj` | 2048 × 6144 |
| `mlp.gate_proj` | 6144 × 2048 |
| `mlp.up_proj` | 6144 × 2048 |
| `self_attn.k_proj` | 1024 × 2048 |
| `self_attn.o_proj` | 2048 × 3072 |
| `self_attn.q_proj` | 3072 × 2048 |
| `self_attn.v_proj` | 1024 × 2048 |

Every layer also contains `input_layernorm.weight` and
`post_attention_layernorm.weight` as `BF16 [2048]`. All `30 * layers` required
tensors (480 by default) have real payloads, including projections the
benchmark does not select. The
unused top-level `embed_tokens.weight` and `norm.weight` from the original model
are omitted. The file retains metadata `{"format":"pt"}`. SafeTensors writes
the compact JSON descriptors and contiguous data ranges; offsets are relative
to the payload, and Zig adds the header length plus 8. Absolute offsets need not
match the downloaded file. This layout targets the existing little-endian loader.

The low nibble stores the earlier E2M1 value. Codes 0..15 decode to
`[0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6]`.
Each flat row-major block of 16 values uses one scale byte; no swizzling or
padding is applied. Random bytes vary across blocks and tensors. The first and
last blocks explicitly cover all 16 codes, including both zero signs. Scale
bytes are positive finite codes 1..126; the first 126 blocks cover subnormals
through 448 explicitly (or as many codes as fit for small tensors). Other
blocks use pseudorandom scales. Scalar global and input scales vary between
projections and are nontrivial positive FP32 values.

The selected default projection has 12,582,912 values, 6 MiB of packed weights,
0.75 MiB of block scales, and 24 MiB of decoded BF16. The complete file is about
468 MiB because the unchanged loader requires every layer. No dequantized
reference tensors are stored in it. Generation writes the JSON header first,
then streams each tensor payload as it is produced; reference validation
decodes in 64-row chunks.

## Validation and raw reference output

`--validate` reads the saved file using SafeTensors, verifies all names, dtypes,
shapes, byte counts and offsets, checks scale encodings and E2M1 coverage, and
decodes every projection through the existing Python reference. It checks BF16
bits exactly against the upper 16 bits of FP32, including signed zero.

To validate an existing generated file without rewriting it:

```bash
uv run --locked --project test-python python test-python/generate_nvfp4_dataset.py \
  --output model.safetensors --validate-only
```

Use `--reference-output test-python/synthetic_reference.bf16` with either command
to also save the selected layer 0 gate projection as raw little-endian BF16.
This option implies validation. Other projections are checked without writing
reference outputs. The existing reference CLI can independently produce the same
bytes:

```bash
uv run --locked --project test-python python test-python/main.py \
  --input model.safetensors --output test-python/nvfp4_reference.bf16 \
  --bf16-mode truncate
```

The reference's original no-argument defaults are preserved (run from
`test-python`, it reads `../model.safetensors`, writes `nvfp4_reference.bf16`,
and uses PyTorch BF16 rounding). Use **`--bf16-mode truncate`** for Zig comparisons.
The reusable `decode_chunk` retains two separate FP32 operations:
`(E2M1_value * block_scale) * global_scale`. Combining the scales can change
the final BF16 bits. `truncate_bf16_bits` discards the low 16 bits rather than
rounding to nearest even.

```bash
uv run --locked --project test-python python -m unittest discover \
  -s test-python -p 'test_*.py' -v
zig build test
```

The Python tests cover byte determinism, schema, reference output, signed zeros,
packing, FP8 extremes, block/row/chunk boundaries, invalid inputs, multiplication
order, and truncation versus rounding. Existing Zig tests check E2M1/E4M3 lookup
tables and metadata parsing; the benchmark currently does not compare or write
its complete output. Python validation stays outside all Zig timing.

## Benchmark and dimensions

```bash
zig build run -Doptimize=ReleaseFast
```

If using the repository's Nix environment, prefix Zig commands with
`nix develop --command`, for example
`nix develop --command zig build run -Doptimize=ReleaseFast`.
Use a Zig version compatible with the checkout's development APIs.

The benchmark opens `model.safetensors` relative to its working directory and
has no input-path CLI. Either generate that file at the repository root or run
the built executable from a directory containing it. No Zig source, SIMD,
compilation flags, worker count, allocations, file operations, or timing logic
have been changed. Compare the existing `decode+write` timing at the same shape,
optimization mode, and CPU; the outer initialization timing includes reads.
For the default shape, throughput in Gvalues/s is `12.582912 / milliseconds`.

For a quick dataset or different seed:

```bash
uv run --locked --project test-python python test-python/generate_nvfp4_dataset.py \
  --output /tmp/synthetic/model.safetensors --shape 80 32 --seed 123 --validate
```

`--shape ROWS COLS` sets the gate/up intermediate dimension and hidden dimension
throughout every layer. Rows must be positive multiples of 16; columns must be
positive multiples of 32 so the derived half-width K/V and 1.5-width Q/O
projections also contain complete 16-value blocks. Down projections transpose
the logical dimensions; layer norms use the hidden dimension. Supply the same
`--shape` when using `--validate-only` for a nondefault file. Smaller workloads
change the performance conditions and should not be compared directly with the
6144 × 2048 baseline.

`--layers N` (default 16) scales the dataset linearly: each layer adds 30
tensors and about 1/16 of the default file size. Values below 16 print a
warning because the Zig benchmark loader requires layers 0..15; use them for
fast Python-side iteration on small synthetic datasets. Supply the same
`--layers` when using `--validate-only` for a nondefault file.
