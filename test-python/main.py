# LLM-generated

"""CPU reference for a ModelOpt NVFP4 layer in a local safetensors file.

Run: uv run nvfp4_reference.py

The output .bf16 file is raw, row-major, little-endian BF16 (no header).
"""

from pathlib import Path

import torch
from safetensors import safe_open

# Edit these two constants.
SAFETENSORS_FILE = "../model.safetensors"
LAYER_KEY = "layers.0.self_attn.q_proj.weight"


def main() -> None:
    weight_key = LAYER_KEY if LAYER_KEY.endswith(".weight") else LAYER_KEY + ".weight"
    prefix = weight_key[: -len(".weight")]
    scale_key = prefix + ".weight_scale"
    global_key = prefix + ".weight_scale_2"

    with safe_open(SAFETENSORS_FILE, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        missing = {weight_key, scale_key, global_key} - keys
        if missing:
            examples = sorted(
                k
                for k in keys
                if k.endswith(".weight")
                and k[: -len(".weight")] + ".weight_scale" in keys
            )
            raise KeyError(
                f"Missing: {sorted(missing)}\n"
                f"Available NVFP4 layers (first 15):\n" + "\n".join(examples[:15])
            )

        packed = f.get_tensor(weight_key)
        scales = f.get_tensor(scale_key)
        tensor_scale = f.get_tensor(global_key)

    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError(
            f"Expected packed weight [out, in/2] uint8, got {packed.dtype} {tuple(packed.shape)}"
        )

    rows, packed_cols = packed.shape
    if packed_cols % 8:
        raise ValueError(
            "Packed last dimension must be divisible by 8 (16 FP4 weights per scale)"
        )
    expected_scales_shape = (rows, packed_cols // 8)
    if tuple(scales.shape) != expected_scales_shape:
        raise ValueError(
            f"Expected flat, row-major group scales {expected_scales_shape}, got {tuple(scales.shape)}. "
            "This may be a swizzled/padded layout requiring deswizzling."
        )
    if tensor_scale.numel() != 1:
        raise ValueError("Expected a single FP32 global scale")

    # Native PyTorch FP8 E4M3FN conversion (not a hand-written decoder).
    if scales.dtype == torch.uint8:
        scales_fp8 = scales.contiguous().view(torch.float8_e4m3fn)
    elif scales.dtype == torch.float8_e4m3fn:
        scales_fp8 = scales
    else:
        raise TypeError(f"Unsupported group-scale dtype: {scales.dtype}")
    scales_fp32 = scales_fp8.to(torch.float32)
    global_scale = tensor_scale.to(torch.float32).reshape(())

    # NVIDIA ModelOpt's published E2M1 code points, in nibble index order.
    # Low nibble = even feature index, high nibble = odd feature index.
    fp4_values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
    )
    codes = torch.empty((rows, packed_cols * 2), dtype=torch.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    unscaled = fp4_values[codes.long()]

    # Each consecutive set of 16 FP4 weights shares one E4M3FN scale.
    decoded_fp32 = (
        unscaled.reshape(rows, -1, 16) * scales_fp32.unsqueeze(-1)
    ) * global_scale
    decoded_fp32 = decoded_fp32.reshape(rows, packed_cols * 2)
    decoded_bf16 = decoded_fp32.to(torch.bfloat16)

    # Direct byte-for-byte comparison against your Zig BFloat16 output.
    output_path = Path("nvfp4_reference.bf16")
    raw_bf16 = decoded_bf16.contiguous().view(torch.int16).numpy()
    raw_bf16.astype("<u2", copy=False).tofile(output_path)

    print(f"weight key:    {weight_key}")
    print(f"packed shape:  {tuple(packed.shape)}")
    print(f"group shape:   {tuple(scales.shape)} ({scales.dtype})")
    print(f"global scale:  {global_scale.item():.9g}")
    print(f"decoded shape: {tuple(decoded_fp32.shape)}")
    print(f"BF16 bytes:    {output_path} ({raw_bf16.size * 2} bytes)")
    print("\nFirst elements, flattened row-major:")
    print(" index   fp4   group_scale       float32        bf16     bf16_bits")
    flat_fp4 = unscaled.reshape(-1)
    flat_scales = scales_fp32.repeat_interleave(16, dim=1).reshape(-1)
    flat_fp32 = decoded_fp32.reshape(-1)
    flat_bf16 = decoded_bf16.reshape(-1).to(torch.float32)
    flat_bits = decoded_bf16.contiguous().view(torch.int16).reshape(-1)
    for i in range(min(64, flat_fp32.numel())):
        print(
            f"{i:6} {flat_fp4[i].item():6g} "
            f"{flat_scales[i].item():13.7g} "
            f"{flat_fp32[i].item():13.8g} "
            f"{flat_bf16[i].item():13.8g} "
            f"    0x{int(flat_bits[i]) & 0xFFFF:04x}"
        )


if __name__ == "__main__":
    main()
