## todo

- read + decode
  - exact method tbd, probably to change once all the steps are implemented correctly
- write to target array

## Notes

- tensor binary blob is organized "homogeneously" rather than accordig to layer hierarchy
  - something to consider for truly efficient deser, though it may change depending on models?
- 

## Links
```
curl -L -o model.safetensors https://huggingface.co/nvidia/Nemotron-3-Embed-1B-NVFP4/resolve/main/model.safetensors
```

  
- https://fleuret.org/public/lbdl.pdf
  - p153 onwards, on quantization
- https://huggingface.co/docs/optimum/en/concept_guides/quantization
- https://ziglang.org/documentation/master/#Vectors
  - syntax of all time I guess
  - https://www.openmymind.net/SIMD-With-Zig/
- https://andrewkelley.me/post/zig-new-async-io-text-version.html
  - can all multithreading be done through this API?
  - are threads pinned to physical cores?
