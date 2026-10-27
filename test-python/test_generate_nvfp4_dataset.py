"""Regression tests for the interchange format and reference's exact semantics."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from generate_nvfp4_dataset import (
    GATE_PREFIX,
    expected_schema,
    generate_tensors,
    projection_shapes,
    validate_dataset,
    write_dataset,
)
from main import decode_chunk, truncate_bf16_bits


def raw_bytes(tensor):
    """Comparable raw bytes for every SafeTensors dtype used by the schema."""
    if tensor.dtype == torch.float8_e4m3fn:
        tensor = tensor.view(torch.uint8)
    elif tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)
    return tensor.reshape(-1).numpy().tobytes()


class ReferenceTests(unittest.TestCase):
    def test_all_codes_low_nibble_first_and_signed_zero(self):
        packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8)
        scales = torch.tensor([[0x38]], dtype=torch.uint8)  # E4M3FN 1.0
        _, _, decoded = decode_chunk(packed, scales, torch.tensor(1.0))
        expected = np.array(
            [0.0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(decoded.numpy().ravel().view(np.uint32), expected.view(np.uint32))

    def test_block_and_row_boundaries_and_fp8_extremes(self):
        packed = torch.full((2, 16), 0x22, dtype=torch.uint8)  # 32 x 1.0 per row
        scales = torch.tensor([[1, 7], [0x38, 0x7E]], dtype=torch.uint8)
        _, _, decoded = decode_chunk(packed, scales, torch.tensor(0.5))
        expected = np.repeat(np.array([[1 / 1024, 7 / 1024], [0.5, 224]], dtype=np.float32), 16, axis=1)
        np.testing.assert_array_equal(decoded.numpy(), expected)

    def test_global_scale_applied_after_block_scale(self):
        # Combining the scales changes BF16 from 0x3ed0 to 0x3ecf here.
        packed = torch.full((1, 8), 0x55, dtype=torch.uint8)  # FP4 3.0
        scales = torch.tensor([[0x24]], dtype=torch.uint8)  # E4M3FN 0.1875
        global_scale = torch.tensor(0x3F38E38E, dtype=torch.int32).view(torch.float32)
        _, _, decoded = decode_chunk(packed, scales, global_scale)
        self.assertEqual(decoded.view(torch.int32)[0, 0].item(), 0x3ED00000)
        self.assertEqual(truncate_bf16_bits(decoded)[0, 0].item(), 0x3ED0)
        combined = torch.tensor(3.0) * (torch.tensor(0.1875) * global_scale)
        self.assertEqual(truncate_bf16_bits(combined).item(), 0x3ECF)

    def test_truncation_differs_from_rounding_and_preserves_zero_signs(self):
        fp32 = np.array([0x00000000, 0x80000000, 0x3F80FFFF, 0xBF80FFFF], dtype=np.uint32).view(np.float32)
        tensor = torch.from_numpy(fp32)
        np.testing.assert_array_equal(truncate_bf16_bits(tensor).numpy(), [0, 0x8000, 0x3F80, 0xBF80])
        self.assertEqual(tensor.to(torch.bfloat16).view(torch.uint16)[2].item(), 0x3F81)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_schema_matches_loader_and_model_shapes(self):
        schema = expected_schema((6144, 2048))
        self.assertEqual(len(schema), 480)
        self.assertEqual(schema[GATE_PREFIX + ".weight"], ("U8", [6144, 1024]))
        self.assertEqual(schema[GATE_PREFIX + ".weight_scale"], ("F8_E4M3", [6144, 128]))
        self.assertEqual(schema[GATE_PREFIX + ".weight_scale_2"], ("F32", []))
        self.assertEqual(schema["layers.15.input_layernorm.weight"], ("BF16", [2048]))
        self.assertEqual(schema["layers.15.self_attn.o_proj.weight"], ("U8", [2048, 1536]))

    def test_layer_count_parametrizes_the_schema(self):
        schema = expected_schema((80, 32), layers=2)
        self.assertEqual(len(schema), 60)
        self.assertIn("layers.1.post_attention_layernorm.weight", schema)
        self.assertNotIn("layers.2.input_layernorm.weight", schema)
        for layers in (0, -3):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                expected_schema((80, 32), layers=layers)

    def test_streamed_writer_matches_save_file_payloads(self):
        # The streaming writer arranges the file differently than
        # safetensors.torch.save_file, but every payload must stay identical.
        shape = (48, 32)
        with tempfile.TemporaryDirectory() as directory:
            names = ("streamed.safetensors", "buffered.safetensors")
            streamed, buffered = [Path(directory) / name for name in names]
            write_dataset(streamed, shape, seed=7, layers=2)
            save_file(generate_tensors(shape, 7, layers=2), buffered, metadata={"format": "pt"})
            self.assertEqual(streamed.stat().st_size, buffered.stat().st_size)
            with safe_open(streamed, framework="pt") as left, safe_open(buffered, framework="pt") as right:
                self.assertEqual(set(left.keys()), set(right.keys()))
                for name in left.keys():
                    one, other = left.get_tensor(name), right.get_tensor(name)
                    self.assertEqual(one.dtype, other.dtype)
                    self.assertEqual(tuple(one.shape), tuple(other.shape))
                    self.assertEqual(raw_bytes(one), raw_bytes(other))

    def test_custom_layer_count_generates_and_validates(self):
        shape = (32, 32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two_layers.safetensors"
            write_dataset(path, shape, seed=5, layers=2)
            self.assertGreater(validate_dataset(path, shape, layers=2), 0)
            with safe_open(path, framework="pt") as dataset:
                self.assertEqual(len(dataset.keys()), 60)

    def test_file_determinism_roundtrip_and_reference_bytes(self):
        # More than CHUNK_ROWS exercises both full and final partial chunks.
        shape = (80, 32)
        with tempfile.TemporaryDirectory() as directory:
            a, b, changed, reference = [Path(directory) / name for name in ("a.safetensors", "b.safetensors", "c.safetensors", "reference.bf16")]
            for path, seed in ((a, 42), (b, 42), (changed, 43)):
                write_dataset(path, shape, seed)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertNotEqual(a.read_bytes(), changed.read_bytes())
            self.assertGreater(validate_dataset(a, shape, reference_output=reference), 0)
            with safe_open(a, framework="pt") as dataset:
                _, _, fp32 = decode_chunk(
                    dataset.get_tensor(GATE_PREFIX + ".weight"),
                    dataset.get_tensor(GATE_PREFIX + ".weight_scale"),
                    dataset.get_tensor(GATE_PREFIX + ".weight_scale_2"),
                )
            expected = (fp32.numpy().view(np.uint32) >> 16).astype("<u2").tobytes()
            self.assertEqual(reference.read_bytes(), expected)
            self.assertEqual(len(expected), shape[0] * shape[1] * 2)

    def test_rejects_bad_scale_and_missing_tensor(self):
        shape = (16, 32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            tensors = generate_tensors(shape, 1)
            tensors[GATE_PREFIX + ".weight_scale"].view(torch.uint8)[0, 0] = 127
            save_file(tensors, path, metadata={"format": "pt"})
            with self.assertRaisesRegex(ValueError, "E4M3"):
                validate_dataset(path, shape)
            del tensors[GATE_PREFIX + ".input_scale"]
            save_file(tensors, path, metadata={"format": "pt"})
            with self.assertRaisesRegex(ValueError, "names"):
                validate_dataset(path, shape)

    def test_rejects_shapes_without_complete_blocks(self):
        for shape in ((0, 32), (16, 0), (17, 32), (16, 48), (-16, 32)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                projection_shapes(shape)


if __name__ == "__main__":
    unittest.main()
