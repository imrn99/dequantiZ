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
)
from main import decode_chunk, truncate_bf16_bits


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

    def test_file_determinism_roundtrip_and_reference_bytes(self):
        # More than CHUNK_ROWS exercises both full and final partial chunks.
        shape = (80, 32)
        with tempfile.TemporaryDirectory() as directory:
            a, b, changed, reference = [Path(directory) / name for name in ("a.safetensors", "b.safetensors", "c.safetensors", "reference.bf16")]
            for path, seed in ((a, 42), (b, 42), (changed, 43)):
                save_file(generate_tensors(shape, seed), path, metadata={"format": "pt"})
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertNotEqual(a.read_bytes(), changed.read_bytes())
            self.assertGreater(validate_dataset(a, shape, reference), 0)
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
