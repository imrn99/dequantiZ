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

```
curl -L -o model.safetensors https://huggingface.co/nvidia/Nemotron-3-Embed-1B-NVFP4/resolve/main/model.safetensors
```
