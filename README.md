## Usage

Run with:

```
zig build run --release=fast -- <OPTIONS>
```

The following options are available:

- `--n-threads <N_THREADS>`: specify the number of thread used to decode tensor data. Defaults to `1`.
- `--layer-ID <LAYER_ID>`: specify which layer to decode. For the base model, this must be between 0 and 15. Defaults to `0`.
- `--synthetic-data`: attempts to read `synth.safetensors` instead of `model.safetensors`. See section below for more detail.

The program parses the `safetensors` file header and uses the metadata to read, decode, and initialize weights of the `mlp.gate_proj` tensor. The routine handling this takes tensor metadata as argument, I chose this specific one because it is the largest one in the chosen model.

## Tensor file

From the repository root, either download the original model:

```bash
curl -L -o model.safetensors https://huggingface.co/nvidia/Nemotron-3-Embed-1B-NVFP4/resolve/main/model.safetensors
```

Or generate synthetic inputs without downloading a model:

```bash
uv run --locked --project test-python python test-python/generate_nvfp4_dataset.py --output synth.safetensors --layer 1 --shape 24576 8192 --seed 42
```

The `--layer <N_LAYER>` option can be used to generate a specific number of layer. This reduces file size when testing on a single larger tensor. 
