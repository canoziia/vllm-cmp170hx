"""Run only in isolated patched vLLM GPU environment; no remote access."""
import argparse
import json

p = argparse.ArgumentParser()
p.add_argument('--run-gpu', action='store_true')
a = p.parse_args()
if not a.run_gpu:
    print('GPU NOT RUN (use --run-gpu)')
    raise SystemExit()
import torch
from vllm.models.glm5next.nvidia.ops import idx_glue as g
from vllm.models.glm5next.nvidia.ops import kpool_compress as k
assert torch.cuda.get_device_capability() == (8, 0)
print(json.dumps({'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'module': g.__file__}))
torch.manual_seed(123)

def timing(fn):
    for _ in range(10): fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): fn()
    samples = []
    for _ in range(10):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100): graph.replay()
        end.record(); end.synchronize()
        samples.append(start.elapsed_time(end) * 10)
    return sorted(samples)[5]

for m in [1, 4, 8, 32, 64, 128]:
    q = torch.randn(m * 32, 128, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(m, 32, device='cuda', dtype=torch.float32)
    def old():
        fp8, scale = k.fwht128_quant_fp8(q)
        return fp8, (w * scale.view(m, 32)) * (2.0 ** -6)
    def new(): return g.fwht128_quant_fp8_wscale(q, w, 2.0 ** -6)
    ref, got = old(), new()
    assert torch.equal(ref[0].view(torch.uint8), got[0].view(torch.uint8))
    assert torch.equal(ref[1], got[1])
    # Graph outputs persist, replay with changing input (not only warm fixed data).
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): out = new()
    for _ in range(20):
        q.normal_(); w.normal_(); graph.replay()
        expected = old()
        assert torch.equal(expected[0].view(torch.uint8), out[0].view(torch.uint8))
        assert torch.equal(expected[1], out[1])
    print(json.dumps({'part': 'fwht', 'M': m, 'bitwise': True, 'old_graph_us': timing(old), 'new_graph_us': timing(new)}))

for m in [1, 4, 8, 32, 64]:
    pools = torch.randint(-1, 512, (m, 512), device='cuda', dtype=torch.int32)
    pools[:, ::7] = -1
    pos = torch.arange(m, device='cuda', dtype=torch.int64) + 2047
    buf = torch.empty(m + 3, 2176, device='cuda', dtype=torch.int32)
    def old():
        buf.fill_(-1)
        expanded = k.expand_pools_and_append_tail(pools.to(torch.int64), pos.to(torch.int32) + 1, 4)
        buf[:m, :expanded.shape[1]] = expanded
    def new(): g.expand_pools_into_buffer(pools, pos, buf, m + 3, 4)
    for base in [0, 15, 2047, 4607, 8191]:
        pos.copy_(torch.arange(m, device='cuda') + base)
        old(); expected = buf.clone(); buf.fill_(12345); new()
        assert torch.equal(buf, expected)
    print(json.dumps({'part': 'expand', 'M': m, 'width': 2176, 'bitwise': True, 'old_graph_us': timing(old), 'new_graph_us': timing(new)}))

# Same shapes on separate streams, separate caller-owned output/cache.
streams = [torch.cuda.Stream(), torch.cuda.Stream()]
records = []
for stream in streams:
    with torch.cuda.stream(stream):
        q = torch.randn(256, 128, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(8, 32, device='cuda')
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream): out = g.fwht128_quant_fp8_wscale(q, w, 2.0 ** -6)
        records.append((stream, graph, q, w, out))
for _ in range(100):
    for stream, graph, q, w, out in records:
        with torch.cuda.stream(stream): q.normal_(); w.normal_(); graph.replay()
torch.cuda.synchronize()
for stream, graph, q, w, out in records:
    fp8, scale = k.fwht128_quant_fp8(q)
    assert torch.equal(fp8.view(torch.uint8), out[0].view(torch.uint8))
    assert torch.equal((w * scale.view(8, 32)) * (2.0 ** -6), out[1])
print('PASS local kernels, graph replay, two-stream FWHT; model rollback NOT TESTED')
