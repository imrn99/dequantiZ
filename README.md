## todo

- I/O bs
- read header to fetch relevant metadata (layer data loc + size)
- read + decode
  - exact method tbd, probably to change once all the steps are implemented correctly
- write to target array

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
