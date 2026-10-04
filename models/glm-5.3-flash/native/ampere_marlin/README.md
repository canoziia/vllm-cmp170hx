# `_ampere_marlin_C` (decode only): compiled sm_80 W4A16 MoE decode

CUDA library used by `patches/dflash2/0006-glm5-compiled-marlin-moe-decode.patch`
(`VLLM_GLM5_MARLIN_DECODE_CUDA=1`). `scripts/build-glm53-dflash2-image.sh`
compiles it during the image build and installs it as
`<site-packages>/vllm/_ampere_marlin_C.abi3.so`; the patch then imports it as
`vllm._ampere_marlin_C`. `VLLM_GLM5_MARLIN_DECODE_LIB` is only needed for an
out-of-image build.

## Provenance and licence

| File | Origin | Change |
|---|---|---|
| `decode.cu` | Morrowmake/vllm-cmp170hx @ `3a2bf16dae8b97f5ff2c7e9bc5809d24545e6340`, `csrc/libtorch_stable/moe/ampere_marlin/decode.cu` | none (sha256 `befd0e51…ab44cc8`) |
| `decode_orig.cu` | same directory | none (sha256 `cac8a5ef…5e1f8`) |
| `module.cpp` | same directory | none (sha256 `30a2b57a…6b43eb7`) |
| `build.py` | adapted from that directory's `build_standalone.py` | decode sources only; no `VLLM_BUILD_AMPERE_MARLIN` guard; tool preflight; post-build schema check in a fresh interpreter |
| `LICENSE` | Morrowmake/vllm-cmp170hx root | Apache-2.0, unchanged |
| `install-into-vllm.sh` | this repository | image-build wrapper (header shim, install into the vllm package) |

All sources carry `SPDX-License-Identifier: Apache-2.0` and
`Copyright contributors to the vLLM project`. The prefill GEMM (`ops.cu`,
`kernels_sm80.cu`, …) is not included; it needs Morrowmake's modified
`marlin_moe_wna16` templates.

## Build

Inside the image build (what the build script does):

```bash
bash install-into-vllm.sh /path/to/ampere_marlin
```

Notes, measured on the `lazymio/vllm-backport:latest-sm80` base:

- `nvcc`, `ninja` and a host `g++` are required; no GPU.
- The CUDA toolkit in the image lacks `cusparse.h`, `cusolverDn.h` and a few
  other library headers; pip ships them in
  `dist-packages/nvidia/cu13/include`. Adding that whole directory to `CPATH`
  breaks the build (`__cudaLaunch` macro errors from a second CUDA runtime
  header set), so `install-into-vllm.sh` symlinks only the **missing** headers
  into a temporary directory and puts that on `CPATH`.
- `VLLM_BUILD_AMPERE_MARLIN=1` is exported for parity with Morrowmake's builder.
- The binary uses the ATen C++ API and is tied to the exact torch build;
  `build_info()` records torch/CUDA/C++11-ABI and the runtime loader refuses a
  mismatch. Rebuild whenever the base image changes (the image build does).

Out-of-image build (testing only):

```bash
podman run --rm --entrypoint python3 -v "$PWD":/src:ro,Z -v /tmp/am-out:/work/out:Z \
  <image> /src/build.py --out /work/out
export VLLM_GLM5_MARLIN_DECODE_LIB=/work/out/_ampere_marlin_C.abi3.so
```

## Ops

| op | variant | what |
|---|---|---|
| `decode_gemm` / `decode_act` | `exact` | W13 with one fp32 plane in Marlin's block-8 two-chain reduction order |
| `decode_gemm_orig` / `decode_act_orig` | `orig` (default) | W13 split four ways along K into fp32 partials; faster, different summation order |

Instantiated shapes: W13 (K, 2N) = (4096, 4096) and (4096, 1024); W2 (K, N) =
(2048, 4096) and (512, 4096), i.e. GLM-5.3-Flash with whole experts (PP,
N = 2048) or TP4 shards (N = 512). Any other shape raises `unsupported rows=…`.
