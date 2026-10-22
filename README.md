## TODO

- test buffer alloc outside of threads + operating on slices
  - likely better for small scale, worse for large
- test threading larger data volumes
  - generate dummy data
- impl and test vectorized decoding functions to replce LUTs

## Notes

- tensor binary blob is organized "homogeneously" rather than accordig to layer hierarchy
  - something to consider for truly efficient deser, though it may change depending on models?
- 

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
zig build run -Doptimize=ReleaseFast
```
See [the Python documentation](test-python/README.md)
for the exact tensor format, reference output, offline use, and tests.
