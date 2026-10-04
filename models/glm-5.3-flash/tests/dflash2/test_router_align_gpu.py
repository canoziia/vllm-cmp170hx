"""Run in installed patched GPU image; never auto-runs from CPU checks."""
import json
import os
os.environ['VLLM_GLM5_ROUTER_ALIGN_DECODE'] = '1'
import torch
from vllm.models.glm5next.nvidia.ops.router_align_decode import maybe_align


def verify(ids, out):
    flat = ids.cpu().flatten().tolist()
    sorted_ids, experts, ntpp = [t.cpu() for t in out]
    expected, owners = [], []
    for e in range(288):
        rows = [i for i, v in enumerate(flat) if v == e]
        if rows:
            padded = rows + [len(flat)] * ((-len(rows)) % 8)
            expected += padded
            owners += [e] * (len(padded) // 8)
    n = ntpp.item()
    assert n == len(expected)
    assert sorted_ids[:n].tolist() == expected
    assert experts[:n // 8].tolist() == owners
    assert (sorted_ids[n:] == len(flat)).all()
    assert (experts[n // 8:] == -1).all()
    return n


def main():
    torch.manual_seed(103)
    graphs = []
    for m in (4, 8):
        ids = torch.randint(0, 288, (m, 8), device='cuda', dtype=torch.int32)
        for mode in ('random', 'same', 'invalid', 'all_invalid'):
            if mode == 'same': ids.fill_(17)
            if mode == 'invalid': ids.flatten()[::3] = -1; ids.flatten()[1::5] = 288
            if mode == 'all_invalid': ids.fill_(-1)
            out = maybe_align(ids, 288)
            n = verify(ids, out)
            print(json.dumps(dict(M=m, mode=mode, ntpp=n, route_set='PASS', invalid_lanes='PASS')))
        ids.random_(0, 288)
        # Warm compile, then fresh graph-owned buffers for each shape.
        maybe_align(ids, 288)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = maybe_align(ids, 288)
        graphs.append((graph, ids, out))
    for step in range(200):
        for graph, ids, out in graphs:
            ids.random_(0, 288)
            ids.flatten()[step % ids.numel()] = -1
            graph.replay()
            verify(ids, out)
    print('alternating multi-shape graphs: PASS (200 rounds)')


if __name__ == '__main__':
    main()
