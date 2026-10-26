## TODO

- skip thread pinning on MacOS
- test single output buffer vs sharded buffer per thread

## Tensor file

From the repository root, either download the original model:

```bash
curl -L -o model.safetensors https://huggingface.co/nvidia/Nemotron-3-Embed-1B-NVFP4/resolve/main/model.safetensors
```

Or generate synthetic inputs without downloading a model:

```bash
uv run --locked --project test-python python test-python/generate_nvfp4_dataset.py \
  --output model.safetensors --shape 6144 2048 --seed 42 --validate
```

Both alternatives write/replace `model.safetensors`, the benchmark's hardcoded input path.
Run the benchmark with either file:

```bash
zig build run --release=fast -- --n-threads 1
```
See [the Python documentation](test-python/README.md)
for the exact tensor format, reference output, offline use, and tests.
