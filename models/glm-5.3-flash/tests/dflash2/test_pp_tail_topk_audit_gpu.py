"""Deployed FlashInfer scratch isolation / graph-vs-side-stream hazard test.

No GPU result claimed locally. Negative control need NOT corrupt results:
shared scratch + overlapping top-k kernel intervals establish the hazard.
Profiler kernel overlap is evidence of concurrent kernels, not proof that
both kernels accessed the scratch at precisely the same instruction.
Run with pytest -s; optional PP_TOPK_TRACE_DIR retains Chrome traces.
"""
import contextlib
from contextvars import ContextVar
import json
import os
from pathlib import Path
import tempfile

import pytest
import torch

from vllm.v1.worker.gpu.pp_draft_tail import (
    validate_flashinfer_topk_contract, private_flashinfer_topk_workspace,
)
from vllm.model_executor.layers.logits_processor import _flashinfer_topk


def _kernel_intervals(trace):
    """Kineto Chrome timestamps/durations share a single GPU timeline."""
    by_stream = {}
    for event in trace['traceEvents']:
        if event.get('cat') != 'kernel' or event.get('ph') != 'X':
            continue
        stream = event.get('args', {}).get('stream')
        assert stream is not None, 'profiler omitted CUDA stream identity'
        by_stream.setdefault(stream, []).append(
            (event['ts'], event['ts'] + event['dur'], event['name']))
    return by_stream


def _overlap_us(a, b):
    # Kernels on each individual stream are ordered and non-overlapping.
    a, b = sorted(a), sorted(b)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _profile(path, launch):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                          torch.profiler.ProfilerActivity.CUDA]) as p:
        launch()
        torch.cuda.synchronize()
    p.export_chrome_trace(str(path))
    return _kernel_intervals(json.loads(path.read_text()))


def _one_stream(intervals):
    # Calibration must contain only the top-k workload, no gate/sleep kernels.
    active = [s for s, events in intervals.items() if events]
    assert len(active) == 1, f'ambiguous top-k calibration streams: {active}'
    return active[0], {event[2] for event in intervals[active[0]]}


class _ScratchObserver:
    """Observe only real impl calls, excluding workspace save/restore reads."""
    def __init__(self, key):
        self.key = key
        self.calls = []
        self._active = ContextVar('topk_scratch_observation', default=None)

    def observe(self, lookup, result):
        active = self._active.get()
        if lookup == self.key and active is not None:
            active.append(None if result is None else result.data_ptr())

    def call(self, impl, *args, **kwargs):
        pointers = []
        token = self._active.set(pointers)
        try:
            result = impl(*args, **kwargs)
            assert pointers, 'real top_k call did not read observed scratch cache'
            self.calls.append(pointers)
            return result
        finally:
            self._active.reset(token)

    def assert_pointers(self, expected, count):
        assert len(self.calls) == count, (len(self.calls), count)
        for index, pointers in enumerate(self.calls):
            assert pointers and set(pointers) == {expected}, (index, pointers, expected)


def test_real_topk_two_streams(record_property):
    assert torch.cuda.is_available(), 'requires deployed CUDA GPU'
    device = torch.device('cuda', torch.cuda.current_device())
    impl = _flashinfer_topk()
    assert impl is not None, 'requires deployed FlashInfer'
    assert hasattr(torch.cuda, '_sleep'), 'requires CUDA launch gate support'
    holder = {}
    validate_flashinfer_topk_contract(device, holder)
    import flashinfer.utils as utils
    original_cache = utils._cache_buf
    key = holder['key']
    observer = _ScratchObserver(key)

    class ObservedCache(dict):
        def get(self, lookup, default=None):
            result = super().get(lookup, default)
            observer.observe(lookup, result)
            return result

    cache = ObservedCache(original_cache)
    utils._cache_buf = cache
    main, side, gate_stream = [torch.cuda.Stream() for _ in range(3)]
    rows = int(os.environ.get('PP_TOPK_ROWS', '256'))
    vocab = int(os.environ.get('PP_TOPK_VOCAB', '65536'))
    repeats = int(os.environ.get('PP_TOPK_REPEATS', '32'))
    rounds = int(os.environ.get('PP_TOPK_ROUNDS', '8'))
    assert rows > 0 and vocab >= 32 and repeats > 0 and rounds > 0
    inputs = [torch.randn(rows, vocab, device=device) for _ in range(2)]
    references = [torch.topk(x, 32, dim=-1) for x in inputs]
    torch.cuda.synchronize()
    trace_dir = os.environ.get('PP_TOPK_TRACE_DIR')
    temporary = tempfile.TemporaryDirectory() if not trace_dir else None
    directory = Path(trace_dir or temporary.name)
    directory.mkdir(parents=True, exist_ok=True)

    def call(i):
        # Workspace save/restore happens outside this observation scope.
        return observer.call(impl, inputs[i], 32, sorted=True, deterministic=True)

    try:
        # Allocate/compile the main buffer before observing graph capture.
        with torch.cuda.stream(main):
            call(0)
        with torch.cuda.stream(side), private_flashinfer_topk_workspace(device, holder):
            call(1)
        torch.cuda.synchronize()
        shared_ptr = cache[key].data_ptr()
        private_ptr = holder['buf'].data_ptr()
        assert shared_ptr != private_ptr

        graph = torch.cuda.CUDAGraph()
        observer.calls.clear()
        with torch.cuda.graph(graph, stream=main):
            for _ in range(repeats):
                graph_values, graph_indices = call(0)
        # These are pointers actually returned to FlashInfer at capture, not
        # merely pointers inferred from holder. Replay has no Python lookup.
        observer.assert_pointers(shared_ptr, repeats)
        torch.cuda.synchronize()

        def main_only():
            with torch.cuda.stream(main):
                graph.replay()

        def side_only():
            with torch.cuda.stream(side), private_flashinfer_topk_workspace(device, holder):
                call(1)

        main_id, main_names = _one_stream(_profile(directory / 'main-calibration.json', main_only))
        side_id, side_names = _one_stream(_profile(directory / 'side-calibration.json', side_only))
        assert main_id != side_id
        # Calibration identifies actual top-k kernel names. Gate/sleep and
        # unrelated kernels cannot count toward the overlap evidence.
        for isolated in (True, False):
            label = 'isolated' if isolated else 'shared-negative'
            outputs = []
            observer.calls.clear()
            gate = torch.cuda.Event()

            def launch():
                # Delay both queues behind a common future event, so Python
                # can enqueue both workloads before either starts. Long graph
                # batches + eager side calls provide sustained overlap pressure.
                with torch.cuda.stream(gate_stream):
                    torch.cuda._sleep(int(os.environ.get('PP_TOPK_GATE_CYCLES', '50000000')))
                    gate.record()
                main.wait_event(gate)
                side.wait_event(gate)
                for _ in range(rounds):
                    with torch.cuda.stream(main):
                        graph.replay()
                    with torch.cuda.stream(side):
                        ctx = (private_flashinfer_topk_workspace(device, holder)
                               if isolated else contextlib.nullcontext())
                        with ctx:
                            for _ in range(repeats):
                                outputs.append(call(1))

            intervals = _profile(directory / f'{label}.json', launch)
            expected_ptr = private_ptr if isolated else shared_ptr
            observer.assert_pointers(expected_ptr, rounds * repeats)
            assert cache[key].data_ptr() == shared_ptr, 'main cache was not restored'
            assert holder['buf'].data_ptr() == private_ptr, 'private buffer grew after capture'
            a = [e for e in intervals.get(main_id, []) if e[2] in main_names]
            b = [e for e in intervals.get(side_id, []) if e[2] in side_names]
            assert a and b, 'profiler failed to observe calibrated top-k streams'
            overlap = _overlap_us(a, b)
            record_property(f'{label}_main_scratch_ptr', shared_ptr)
            record_property(f'{label}_side_scratch_ptr', expected_ptr)
            record_property(f'{label}_kernel_overlap_us', overlap)
            if overlap <= 0:
                pytest.fail(f'{label}: overlap inconclusive; adjust rows/repeats/gate '
                            'or inspect trace, do not claim the concurrency test passed')
            mismatches = sum(
                not (torch.equal(values, references[1].values)
                     and torch.equal(indices, references[1].indices))
                for values, indices in outputs)
            main_mismatch = not (torch.equal(graph_values, references[0].values)
                                 and torch.equal(graph_indices, references[0].indices))
            record_property(f'{label}_side_mismatches', mismatches)
            record_property(f'{label}_main_final_mismatch', int(main_mismatch))
            print(f'{label}: main_ptr={shared_ptr:#x}, side_ptr={expected_ptr:#x}, '
                  f'top-k overlap={overlap:.3f} us, side mismatches={mismatches}, '
                  f'main final mismatch={main_mismatch}')
            if isolated:
                assert mismatches == 0 and not main_mismatch
            else:
                # No required corruption: same captured scratch pointer plus
                # measured kernel overlap is the negative control's hazard.
                assert expected_ptr == shared_ptr
    finally:
        # Synchronize before restoring globals, including on assertion failure.
        torch.cuda.synchronize()
        utils._cache_buf = original_cache
        if temporary:
            temporary.cleanup()
