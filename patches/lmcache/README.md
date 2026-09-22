# Official LMCache patch series

This directory patches the **official LMCache v0.5.5 CUDA 13.0 release**. It
must not be applied to the LMCache wheel bundled by the third-party DeepSeek
base image.

Pinned inputs are in `manifests/lmcache.env`:

- source commit `05a013b29da78cf2321b9b46ec5039dde2fb0bb0` (`v0.5.5`);
- official server image digest `sha256:59b350...`;
- official payload image digest `sha256:0dd644...`.

`scripts/build-deepseek-v41-lmcache-images.sh` extracts the complete official package (Python,
compiled extensions, and dist-info) from the official payload image, applies
`series`, and injects that identical tree into both images:

1. the standalone LMCache server image;
2. the DeepSeek/vLLM client image.

The payload is selected with `PYTHONPATH=/opt/lmcache-patched`, so it replaces
rather than mixes with the base image's bundled LMCache package. Verify at
runtime that `lmcache.__file__` begins with `/opt/lmcache-patched/`.

## Patches

1. `0001-native-fs-key-and-size-batching.patch`
   - folds `cache_salt` into the 256-bit chunk identity because the v0.5.5
     native FS grammar accepts at most four key fields;
   - includes `object_group_id` in store shape grouping;
   - splits native store calls by exported `memoryview.nbytes` and aggregates
     sub-batch completion into one logical L2 task;
   - logs failed native store sub-batches with future ID, key count, and error.

2. `0002-rank-aware-uneven-pp-layout.patch`
   - registers layouts by encoded `(kv_rank, object_group_id)`;
   - keeps rank-specific reference counts and removes stale layouts on worker
     unregister;
   - reserves L2 prefetch buffers by rank and object group;
   - retains integer-key fallback layouts for homogeneous/legacy consumers.

   This is required by the formal PP6 partition `7,7,7,7,7,5`; using one
   `(model_name, world_size)` layout caused half-size reads such as
   `expected 733184, got 366592`.

3. `0003-event-capability-cache.patch`
   - caches the validated backend Event class after the initial fail-closed
     capability check;
   - avoids repeated `inspect.signature` in each hot store Future.

4. `0004-vllm-unified-kv-compatibility.patch`
   - carries the vLLM unified-layout compatibility required by the pinned
     engines: excludes non-prefix-cacheable scratch rings (including zero-span
     block slicing), resolves logical vs physical KV views, and enforces
     recurrent-state checkpoints;
   - this is an explicit compatibility patch, not an implicit dependency on
     the third-party LMCache wheel.

5. `0005-registration-heartbeat-lifecycle.patch`
   - starts worker heartbeat immediately after KV registration, so an idle
     registered worker cannot exceed server registration grace;
   - passes the logical MP worker rank in layout hints, avoiding unsafe rank
     inference from local CUDA device numbering.

6. `0006-adopt-existing-native-fs.patch`
   - scans completed native-FS object files after restart;
   - seeds byte accounting and eviction listeners oldest-first;
   - makes the configured `adopt_existing=true` meaningful instead of leaving
     old files unaccounted.

7. `0007-mtp-mamba-relocation.patch`
   - mirrors vLLM align-mode Mamba speculative-block relocation in LMCache's
     request tracker and nulls the old slot;
   - permits multi-block prefill with MTP after fixing its store metadata,
     instead of requiring `max_num_batched_tokens == block_size`.

The server-image build runs `scripts/test-lmcache-patches.py` once. It checks
native extension ABI, salt isolation, mixed-size native-FS round trips,
six-rank layout lifetime, and existing-file adoption. The client-image build
only checks its ABI and vLLM connector import against the DeepSeek runtime.
These are build gates, not a replacement for real GPU eviction/restore tests.

The official v0.5.5 package already contains the native event/completion
callback design from the merged deadlock fixes. Those fixes are not duplicated
here.

## Build

First build the normal DeepSeek image on the same branch, then run:

```bash
DEEPSEEK_BASE_IMAGE=localhost/deepseek-v41-cmp170hx:73d0be8-debug-eventfix \
  scripts/build-deepseek-v41-lmcache-images.sh
```

Outputs default to:

```text
localhost/deepseek-v41-lmcache:official-v0.5.5-patched
localhost/deepseek-v41-cmp170hx:official-lmcache-v0.5.5-patched
```

Override `SERVER_OUTPUT_IMAGE` and `CLIENT_OUTPUT_IMAGE` when publishing.
Never replace a running deployment solely because the images build: first run
the registration, complete-store, eviction, restore, restart-adoption, and
corrupt-file fallback acceptance tests.
