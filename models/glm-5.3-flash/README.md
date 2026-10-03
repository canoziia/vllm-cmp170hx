# GLM-5.3-Flash NVFP4 PD (prefill/decode disaggregated) on eight CMP 170HX GPUs

Model: [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
at revision `da920bb0b9f4a06727223a349e55468e38352348`.

- `Glm5NextForConditionalGeneration` (`model_type: glm5_next`), natively
  multimodal (24-layer vision tower);
- 45 decoder layers: 34 KDA linear-attention layers and 11 DeepSeek sparse
  attention layers (3, 7, ..., 43; `index_topk 2048`), mHC (`hc_mult 4`);
- MoE: 288 routed experts, top-8, one shared expert; layers 0-2 are dense;
- one MTP layer (`num_nextn_predict_layers: 1`, layer 45, BF16);
- `max_position_embeddings: 1048576`;
- ModelOpt NVFP4 for the routed experts and dense MLPs; attention, shared
  experts, gates, embeddings, `lm_head` and the vision tower stay BF16.
  190.4 GiB on disk.

## Image

The deployment reuses `localhost/vllm-backport:deepseek-v4.1-flash`
(`scripts/build-deepseek-v41-image.sh`): the pinned upstream source
(`344303947/dsv41-flash-pp5-170hx@d63af5a4`) already registers `glm5next`, and
the image carries the shared vLLM series including
`patches/vllm/0007-nvfp4-marlin-scale-factor-amax.patch`, which this model
**requires**. Without it every PP4 rank OOMs in
`process_weights_after_loading` (the NVFP4 -> Marlin conversion computed its
scale factor with a boolean gather that asks for 6.75 GiB per MoE layer).

## Roles

| role | GPUs (NUMA) | port | PP4 partition | weights per rank |
|---|---|---|---|---|
| prefill | 1,2,3,4 (0) | 8301 | `13,12,12,8` | 44.3 / 50.3 / 50.3 / 50.1 GiB |
| decode | 5,6,7,8 (1) | 8302 | `13,12,12,8` | same |
| lmcache | sees 1..8 | 5559 (HTTP 18559) | - | - |
| router | - | **9103** | - | - |

Layer sizes from the safetensors headers: layers 0-2 0.34 GiB, layers 3-44
~4.09 GiB, the MTP layer 13.84 GiB, embeddings / `lm_head` / vision 1.18 /
1.18 / 1.05 GiB. The last rank carries the MTP layer and `lm_head`, so it
gets the fewest layers. The partition must be identical on both roles: the
shared LMCache server holds one layout.

## Why these numbers

**The unified KV block is 4480 tokens under PP.** vLLM sizes the attention
block so that one attention page is at least one recurrent-state page
(`Setting attention block size to 4480 tokens ...`). With TP4 the KDA state is
split four ways and the block is 1152 (the value in the upstream README); with
PP every rank holds the full state (64 heads x 128 x 128 fp32 per layer), so
the block is 4480. The LMCache fork requires a state checkpoint at every chunk
boundary and at least one full block per prefill step, hence on both roles:

- LMCache `--chunk-size=4480 --separate-object-groups`;
- `--prefix-cache-retention-interval=4480`;
- `--max-num-batched-tokens=4480` (the fork rejects anything smaller:
  `Mamba-hybrid models with LMCache require max_num_batched_tokens >= block_size`).

`--kv-cache-dtype=bfloat16` keeps the block at 4480; an fp8 attention page
would double it.

**KV is 4 GiB per rank.** A 4480-token warmup step needs ~5.8 GiB of
activations on the 50 GiB ranks, and the LMCache server allocates ~210 MiB per
GPU when it registers the KV cache. At 6 GiB of KV the registration OOM'd on
ranks 1 and 2 (92 MiB free) and the engine waited forever. At 4 GiB:

- KV pool **1,236,992 tokens** (1.18x of a 1M-token request) on both roles;
- CUDA graph capture 0.31-0.51 GiB per rank;
- free after startup: rank 0 ~12 GiB, ranks 1-3 2.1-2.3 GiB.

Rank 0 has room for one more layer, but moving a 4.1 GiB layer from any other
rank does not raise the minimum free memory, so the partition stays as is.

## Verified (through the router, `:9103`)

| check | result |
|---|---|
| chat | correct answer; `reasoning` and `content` split by `glm45` |
| PD handoff, fresh 31,903-token prompt | decode hop read **31,360** tokens (7 x 4480) from LMCache, computed 543 |

Decode with MTP-3 (`scripts/benchmark-vllm.mjs`; counting/prose 500 tokens,
code stops naturally):

| shape | c=1 per request | accepted / step | c=4 full batch | c=8 full batch |
|---|---|---|---|---|
| counting | 67.0 tok/s | 2.86 | 215.4 tok/s | 312.3 tok/s |
| prose | 62.4 tok/s | 2.33 | 184.4 tok/s | 276.3 tok/s |
| code | 82.2 tok/s | 3.20 | 225.3 tok/s | 345.2 tok/s |

Not yet measured: long-context prefill (262k / 1M) memory and speed, image
inputs, tool calls.

## Run

```bash
# from this directory; .env (ignored) holds the machine paths and the key:
#   GLM_MODEL_CACHE=/root/app/vllm/cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4
#   VLLM_CACHE_PREFILL=/root/app/vllm/cache/glm53f-pd-prefill
#   VLLM_CACHE_DECODE=/root/app/vllm/cache/glm53f-pd-decode
#   LMCACHE_L2_PATH=/root/app/lmcache/glm-5.3-flash
#   VLLM_API_KEY=...
podman compose -f compose.pd.yml --podman-run-args=--ipc=host up -d
podman compose -f compose.pd.yml ps
```

`--podman-run-args=--ipc=host` is required (the connector shares the KV cache
over CUDA IPC and podman-compose ignores the compose `ipc` key).

Restart the LMCache server together with the engines: stop both engines,
restart the server, then start the engines. The server keeps an engine's KV
cache mapped over CUDA IPC after that engine is gone. When only the decode role
was replaced, the server still held 4.6 GiB on two of its GPUs, and the new
engine failed vLLM's startup check (`Free memory on device ... is less than
desired GPU memory utilization`). The model is
addressed by repository ID and revision; `GLM_MODEL_CACHE` is the HF hub
directory mounted read-only at the same path inside the containers.
