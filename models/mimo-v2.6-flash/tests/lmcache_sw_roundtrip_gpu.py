"""Regression test for LMCache's default sliding-window window path on MiMo.
Run inside the LMCache server image next to `lmcache server --chunk-size=256
--separate-object-groups` (both with --pid=host --ipc=host); URL, NCH, NB env.

MiMo-geometry LMCache MP round-trip test (one PP rank, 13 layers).

Groups (as registered by MiMo PP rank 2 in production): full attention
layers 0-1 (bs 32, 4 heads, hs 320, tokens/block 32), SWA-128 groups
2-3, 4-6, 7-8, 9-10, 11-12 (bs 16, 8 heads, hs 320). Tensors are views of
shared per-position buffers (vLLM hybrid layout), block ids come from one
shared pool. Store happens in two incremental ops; retrieves go into fresh
blocks with vLLM-style null blocks (id 0) for SWA positions outside the
window, and are compared block by block.
"""
import os, sys, time, random
import torch, zmq
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter, LMCacheMPSchedulerAdapter, LoadStoreOp, ParallelStrategy)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo

URL = os.environ.get("URL", "tcp://127.0.0.1:15761")
CHUNK, PAGE = 256, 40960
NCH = int(os.environ.get('NCH', '20')); NB = int(os.environ.get('NB', '20000'))
GROUPS = [  # (layer indices, tokens_per_block, heads, sw)
    ((0, 1), 32, 4, -1), ((2, 3), 16, 8, 128), ((4, 5, 6), 16, 8, 128),
    ((7, 8), 16, 8, 128), ((9, 10), 16, 8, 128), ((11, 12), 16, 8, 128)]
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
bufs = [torch.zeros(NB, PAGE, dtype=torch.bfloat16, device=dev) for _ in range(3)]
kv = {}
for layers, bs, nh, _ in GROUPS:
    for pos, li in enumerate(layers):
        kv[li] = bufs[pos].view(NB, bs, nh, 320)
kv_caches = {f"model.layers.{i}.self_attn.attn": kv[i] for i in range(13)}
infos = [EngineGroupInfo(engine_group_id=g, layer_indices=layers, tokens_per_block=bs, sw_size_tokens=sw)
         for g, (layers, bs, _, sw) in enumerate(GROUPS)]

ps = ParallelStrategy(mla_only=False, vllm_world_size=1, vllm_worker_id=0, tp_size=1, pp_size=1, n_servers=1)
ctx = zmq.Context()
MODEL = "/models/MiMo-V2.6-Flash-MOPD-swtest"
w = LMCacheMPWorkerAdapter(URL, ctx, MODEL, 16, ps)
s = LMCacheMPSchedulerAdapter([URL], ctx, MODEL, 16, ps)
w.register_kv_caches(kv_caches, infos, {"kv_layout": "NHD"})

pool = list(range(1, NB)); random.seed(7); random.shuffle(pool)
def take(n):
    out = pool[:n]; del pool[:n]; return out

N = NCH * CHUNK
tokens = [random.randrange(1000, 150000) for _ in range(N)]
# Store-side allocation: every group owns real blocks for every position.
src = [take(N // GROUPS[g][1]) for g in range(len(GROUPS))]
for b in bufs:
    b.normal_()
torch.cuda.synchronize()
ref = [b.clone() for b in bufs]  # snapshot of the stored content

def wait(fut, what):
    r = fut.result(timeout=120)
    torch.cuda.synchronize()
    return r

def store(rid, start, end):
    ops = [ids[start // GROUPS[g][1]: end // GROUPS[g][1]] for g, ids in enumerate(src)]
    ev = w.create_recorded_event()
    w.submit_store_request(rid, LoadStoreOp(token_ids=tokens[:end], block_ids=ops, start=start, end=end), ev)
    return wait(w.store_futures[rid], "store")

if NCH == 20:
    print("store A", store("st-a", 0, 12 * CHUNK), "store B", store("st-b", 12 * CHUNK, N), flush=True)
else:  # 4096-token prefill steps, like the engine
    t0 = time.time(); rs = [store(f"st-{i}", i, min(i + 16 * CHUNK, N)) for i in range(0, N, 16 * CHUNK)]
    print(f"stored {N} tokens in {len(rs)} ops, results {set(map(str, rs))}, {time.time() - t0:.1f}s", flush=True)
    time.sleep(int(os.environ.get("SETTLE", "15")))
time.sleep(1)

def lookup(rid, ntok):
    s.maybe_submit_lookup_request(rid, tokens[:ntok])
    for _ in range(2400):
        r = s.check_lookup_result(rid)
        if r is not None:
            return r
        time.sleep(0.05)
    raise TimeoutError

def check(name, ntok, vllm_hit=0):
    rid = f"rt-{name}"
    hit = lookup(rid, ntok)
    start = vllm_hit // CHUNK * CHUNK; end = hit; skip = vllm_hit - start
    # retrieve-side allocation: fresh blocks; SWA positions outside the window
    # of the prefix end get the null block 0 (vLLM SWA manager)
    dst = []
    for g, (layers, bs, nh, sw) in enumerate(GROUPS):
        nblk = end // bs
        ids = take(nblk)
        if sw > 0:
            keep_from = max(0, (end - sw)) // bs
            ids = [0 if i < keep_from else b for i, b in enumerate(ids)]
        dst.append(ids)
    for b in bufs:  # poison the destination blocks
        idx = torch.tensor([i for ids in dst for i in ids if i], device=dev)
        b[idx] = float("nan")
    ops = [ids[start // GROUPS[g][1]: end // GROUPS[g][1]] for g, ids in enumerate(dst)]
    ev = w.create_recorded_event()
    w.submit_retrieve_request(rid, LoadStoreOp(token_ids=tokens[:end], block_ids=ops, start=start, end=end,
                                               skip_first_n_tokens=skip), ev)
    t0 = time.time(); res = wait(w.retrieve_futures[rid][0], "retrieve"); el = time.time() - t0
    bad = []; checked = 0
    for g, (layers, bs, nh, sw) in enumerate(GROUPS):
        lo = vllm_hit // bs if sw < 0 else max(vllm_hit, end - sw) // bs
        for blk in range(lo, end // bs):
            d, o = dst[g][blk], src[g][blk]
            for pos in range(len(layers)):
                checked += 1
                if not torch.equal(bufs[pos][d], ref[pos][o]):
                    bad.append((g, blk, pos))
    print(f"{name}: tokens {ntok} lookup-hit {hit} retrieve [{start},{end}) skip {skip} result {res} {el:.2f}s "
          f"blocks checked {checked} mismatched {len(bad)} {bad[:6]}", flush=True)
    return not bad

ok = True
if NCH != 20:
    ok &= check("long-prefix", N + 11)
    print("ALL-OK" if ok else "FAILED", flush=True); os._exit(0 if ok else 1)
ok &= check("full-prefix", N + 37)
ok &= check("short-prefix-12ch", 12 * CHUNK + 5)
ok &= check("short-prefix-7ch", 7 * CHUNK + 200)
ok &= check("partial-gpu-hit", N + 1, vllm_hit=3 * CHUNK + 64)
ok &= check("partial-gpu-hit-late", N + 1, vllm_hit=19 * CHUNK + 128)
print("ALL-OK" if ok else "FAILED", flush=True)
os._exit(0 if ok else 1)
