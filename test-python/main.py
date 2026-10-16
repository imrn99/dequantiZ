# LLM-generated

from pathlib import Path

import torch
from safetensors import safe_open

# Model and layer settings.
SAFETENSORS_FILE = "../model.safetensors"
LAYER_KEY = "layers.0.self_attn.q_proj.weight"

# Execution settings.
DEVICE = "cpu"  # "cpu", "cuda", "mps", or "auto" (prefers CUDA, then MPS)
CPU_THREADS = 1  # PyTorch CPU intra-op threads; use None for the default
CHUNK_ROWS = 64  # Tune for memory use: fewer rows means less peak GPU/RAM usage


def select_device() -> torch.device:
    if DEVICE == "auto":
        kind = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        kind = DEVICE
    if kind not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unknown DEVICE={DEVICE!r}; expected cpu, cuda, mps, or auto")
    if kind == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE='cuda' requested but PyTorch cannot access CUDA")
    if kind == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("DEVICE='mps' requested but PyTorch cannot access MPS")
    return torch.device(kind)


def main() -> None:
    if CPU_THREADS is not None:
        if CPU_THREADS < 1:
            raise ValueError("CPU_THREADS must be positive or None")
        # Must be set before launching tensor calculations.
        torch.set_num_threads(CPU_THREADS)
    if CHUNK_ROWS < 1:
        raise ValueError("CHUNK_ROWS must be positive")

    device = select_device()
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
            f"Expected packed [out, in/2] uint8, got {packed.dtype} {tuple(packed.shape)}"
        )

    rows, packed_cols = packed.shape
    if packed_cols % 8:
        raise ValueError(
            "Packed last dimension must be divisible by 8 (16 FP4 weights/scale)"
        )
    expected_scales_shape = (rows, packed_cols // 8)
    if tuple(scales.shape) != expected_scales_shape:
        raise ValueError(
            f"Expected flat row-major scales {expected_scales_shape}, got {tuple(scales.shape)}. "
            "This may be a swizzled/padded layout requiring deswizzling."
        )
    if tensor_scale.numel() != 1:
        raise ValueError("Expected a single global tensor scale")
    if scales.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise TypeError(f"Unsupported group-scale dtype: {scales.dtype}")

    # E2M1 codepoints from NVIDIA ModelOpt (independent of the Zig decoder).
    # Low nibble corresponds to the earlier weight in each byte.
    fp4_values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=device,
    )
    global_scale_cpu = tensor_scale.to(torch.float32).reshape(())
    global_scale = global_scale_cpu.to(device)

    output_path = Path("nvfp4_reference.bf16")
    print(f"device:         {device}")
    print(f"CPU threads:    {torch.get_num_threads()}")
    print(f"chunk rows:     {CHUNK_ROWS}")
    print(f"weight key:     {weight_key}")
    print(f"packed shape:   {tuple(packed.shape)}")
    print(f"group shape:    {tuple(scales.shape)} ({scales.dtype})")
    print(f"global scale:   {global_scale_cpu.item():.9g}")
    print(f"decoded shape:  {(rows, packed_cols * 2)}")
    print("\nFirst elements, flattened row-major:")
    print(" index   fp4   group_scale       float32        bf16     bf16_bits")

    shown = 0
    with torch.inference_mode(), output_path.open("wb") as out_file:
        for start in range(0, rows, CHUNK_ROWS):
            end = min(start + CHUNK_ROWS, rows)
            packed_part = packed[start:end].to(device)

            # Convert E4M3FN to f32 on CPU: works even where GPU lacks FP8 support.
            scales_part = scales[start:end].contiguous()
            if scales_part.dtype == torch.uint8:
                scales_part = scales_part.view(torch.float8_e4m3fn)
            scales_fp32 = scales_part.to(torch.float32).to(device)

            codes = torch.empty(
                (end - start, packed_cols * 2), dtype=torch.uint8, device=device
            )
            codes[:, 0::2] = packed_part & 0x0F
            codes[:, 1::2] = packed_part >> 4
            unscaled = fp4_values[codes.long()]

            decoded_fp32 = (
                unscaled.reshape(end - start, -1, 16) * scales_fp32.unsqueeze(-1)
            ) * global_scale
            decoded_fp32 = decoded_fp32.reshape(end - start, packed_cols * 2)
            decoded_bf16 = decoded_fp32.to(torch.bfloat16)

            # Stage each chunk to CPU to write exactly 2 bytes per BF16 value.
            bits = decoded_bf16.cpu().contiguous().view(torch.int16)
            bits.numpy().astype("<u2", copy=False).tofile(out_file)

            count = min(64 - shown, decoded_fp32.numel())
            if count > 0:
                fp4_sample = unscaled.reshape(-1)[:count].cpu().tolist()
                scales_sample = (
                    scales_fp32.repeat_interleave(16, dim=1)
                    .reshape(-1)[:count]
                    .cpu()
                    .tolist()
                )
                f32_sample = decoded_fp32.reshape(-1)[:count].cpu().tolist()
                bf16_sample = decoded_bf16.reshape(-1)[:count].float().cpu().tolist()
                bits_sample = bits.reshape(-1)[:count].tolist()
                for j in range(count):
                    print(
                        f"{shown + j:6} {fp4_sample[j]:6g} "
                        f"{scales_sample[j]:13.7g} "
                        f"{f32_sample[j]:13.8g} "
                        f"{bf16_sample[j]:13.8g} "
                        f"    0x{bits_sample[j] & 0xFFFF:04x}"
                    )
                shown += count

    print(f"\nBF16 bytes:     {output_path} ({rows * packed_cols * 4} bytes)")


if __name__ == "__main__":
    main()
