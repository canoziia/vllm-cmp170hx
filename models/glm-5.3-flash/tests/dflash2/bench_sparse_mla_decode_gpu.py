"""Explicit GPU opt-in. Loads source modules under distinct names, never monkeypatches vllm."""
import argparse
import importlib.util
import json
import sys
import os
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-gpu', action='store_true')
    p.add_argument('--ours', default='/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_mla_sparse_kernel.py')
    p.add_argument('--mm', default=os.environ.get('MM_TRITON_MLA_SPARSE', 'vllm-cmp170hx@3a2bf16dae/vllm/v1/attention/ops/triton_mla_sparse.py'))
    p.add_argument('--candidate', default=str(Path(__file__).parents[1] / 'work-sparse/triton_mla_sparse_mm_experimental.py'))
    args = p.parse_args()
    if not args.run_gpu:
        print('GPU NOT RUN: pass --run-gpu explicitly')
        return
    import torch
    torch.manual_seed(123)
    modules = {n: load('sparse_bench_' + n, path) for n, path in
               [('ours', args.ours), ('mm', args.mm), ('candidate', args.candidate)]}
    print(json.dumps({'sources': {n: m.__file__ for n, m in modules.items()},
                      'device': torch.cuda.get_device_name(), 'torch': torch.__version__}))

    def measure(fn, graph=False):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        if graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                held = fn()
            fn = g.replay
        samples = []
        for _ in range(10):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(50):
                fn()
            b.record()
            b.synchronize()
            samples.append(a.elapsed_time(b) * 1000 / 50)
        return sorted(samples)[len(samples)//2]

    for ctx in (32, 512, 2048, 8192):
        for rows in (4, 8):
            q = torch.randn(rows, 64, 512, device='cuda', dtype=torch.bfloat16)
            kv = torch.randn(ctx, 1, 512, device='cuda', dtype=torch.bfloat16)
            # Narrow vs padded inputs are separate cases; never truncate production IDs.
            for width in sorted({min(ctx, 2048), 2048, 2176}):
                for mode in ('valid', 'holes', 'empty'):
                    ids = torch.full((rows, 1, width), -1, device='cuda', dtype=torch.int32)
                    n = min(ctx, 2048, width)
                    for r in range(rows):
                        ids[r, 0, :n] = torch.randperm(ctx, device='cuda')[:n].int()
                    if mode == 'holes':
                        ids[:, :, ::3] = -1
                        ids[:, :, 1::7] = ctx  # out-of-range invalid
                        ids[0] = -1  # entirely empty query amid nonempty queries
                    if mode == 'empty':
                        ids.fill_(-1)
                    valid = (ids[:, 0] >= 0) & (ids[:, 0] < ctx)
                    gathered = kv[ids[:, 0].clamp(0, ctx-1).long(), 0].float()
                    logits = torch.einsum('mhd,mkd->mhk', q.float(), gathered) / (512 ** .5)
                    logits.masked_fill_(~valid[:, None], -float('inf'))
                    probs = torch.softmax(logits, -1).nan_to_num()
                    ref = torch.einsum('mhk,mkd->mhd', probs, gathered)
                    outputs = {}
                    for name, mod in modules.items():
                        if name == 'ours':
                            fn = lambda mod=mod: mod.triton_mla_sparse_attention(q, kv, ids, 512 ** -.5)
                        else:
                            fn = lambda mod=mod: mod.triton_mla_sparse_fwd(q, kv, ids, 512 ** -.5, block_dpe=0)[0]
                        out = fn()
                        torch.cuda.synchronize()
                        outputs[name] = out
                        err = (out.float() - ref).abs()
                        record = dict(ctx=ctx, M=rows, H=64, D=512, width=width, mode=mode, kernel=name,
                                      finite=bool(out.isfinite().all()), max_abs=float(err.max()),
                                      rms=float(err.square().mean().sqrt()),
                                      event_us=measure(fn), graph_us=measure(fn, True))
                        if name != 'ours':
                            record['config'] = mod._pick_config(rows, width, 64, 512, q.device.index)
                        print(json.dumps(record), flush=True)
                        if name == 'candidate':
                            torch.testing.assert_close(out.float(), ref, atol=.035, rtol=.035)
                    print(json.dumps({'pair_max_abs': float((outputs['ours'].float()-outputs['candidate'].float()).abs().max())}))


if __name__ == '__main__':
    main()
