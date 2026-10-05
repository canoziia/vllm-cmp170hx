"""Patch 0030 (dev 0025, VLLM_PP_METADATA_CACHE_0025) CPU wire tests, including real patched coordinator method execution."""
import ast
import importlib.util
import os
import pickle
from collections import deque, namedtuple
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

ROOT = Path(os.environ.get("GLM_DFLASH2_TREE", "/usr/local/lib/python3.12/dist-packages")) / 'vllm' / 'distributed'
spec = importlib.util.spec_from_file_location('cache0025', ROOT/'pp_metadata_cache.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

class Handle:
    def __init__(self, works=(), retained=()): self.retained = retained; self.waited = False
    def wait(self): self.waited = True
    def is_completed(self): return True

class Wire:
    def __init__(self): self.queue = deque(); self.trace = []
    def isend(self, tensor, dst, group):
        self.queue.append(tensor.clone()); self.trace.append(('send', group, tuple(tensor.shape), tensor.dtype)); return Handle()
    def recv(self, tensor, src, group):
        tensor.copy_(self.queue.popleft()); self.trace.append(('recv', group, tuple(tensor.shape), tensor.dtype))
    def irecv(self, tensor, src, group): self.recv(tensor, src, group); return Handle()
    def is_initialized(self): return True

TensorMetadata = namedtuple('TensorMetadata', ['device', 'dtype', 'size'])
def split(tensors):
    return ([(key, TensorMetadata(value.device.type, value.dtype, value.size()) if isinstance(value, torch.Tensor) else value)
             for key, value in tensors.items()], [v for v in tensors.values() if isinstance(v, torch.Tensor)])

# Execute the real modified methods without importing GPU/vLLM dependencies.
source = ast.parse((ROOT/'parallel_state.py').read_text())
cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'GroupCoordinator')
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
           ('isend_tensor_dict', 'irecv_tensor_dict', '_reap_completed_isends')]
wire = Wire()
namespace = dict(torch=SimpleNamespace(Tensor=torch.Tensor, distributed=wire, empty=torch.empty, cuda=torch.cuda),
                 TensorMetadata=TensorMetadata, Handle=Handle, deque=deque, _split_tensor_dict=split)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             ast.ClassDef(name='Coordinator', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])),
             '<real0025-methods>', 'exec'), namespace)
Coordinator = namespace['Coordinator']

def coordinator(rank, enabled):
    obj = Coordinator(); obj.world_size=2; obj.rank_in_group=rank; obj.ranks=[0,1]
    obj.cpu_group='gloo'; obj.device_group='nccl'; obj.use_cpu_custom_send_recv=False
    obj._pending_isends=deque(); obj._should_use_all_gather=lambda *args: False
    obj._pp_metadata_cache=module.PPMetadataCache(obj, Handle, wire) if enabled else None
    def send_object(value, dst):
        data=pickle.dumps(value)
        wire.isend(torch.tensor([len(data)]), dst, 'gloo')
        wire.isend(torch.frombuffer(bytearray(data), dtype=torch.uint8), dst, 'gloo')
        return Handle()
    def recv_object(src):
        size=torch.empty(1,dtype=torch.int64); wire.recv(size,src,'gloo')
        payload=torch.empty(size.item(),dtype=torch.uint8); wire.recv(payload,src,'gloo')
        return pickle.loads(payload.numpy().tobytes())
    obj.isend_object=send_object; obj.recv_object=recv_object
    return obj

class Tests(unittest.TestCase):
    def setUp(self): wire.queue.clear(); wire.trace.clear()
    def test_mixed_cancel_shape_budget_sequence(self):
        for enabled in (False, True):
            sender=coordinator(0,enabled); receiver=coordinator(1,enabled)
            previous=None
            # Empty/cancelled cohort, prefill/decode bucket changes, budget and
            # lengths changes, key/order/dtype changes with identical byte sizes.
            sequence=[{'hidden_states':torch.ones(8,4), 'aux_fc_partial':torch.ones(8,2)},
                      {'hidden_states':torch.ones(8,4), 'aux_fc_partial':torch.zeros(8,2)},
                      {}, {}, {'hidden_states':torch.ones(64,4)},
                      {'hidden_states':torch.ones(8,4), '__verification_budget':3},
                      {'hidden_states':torch.ones(8,4), '__verification_budget':4},
                      {'hidden_states':torch.ones(8,4), '__verification_budget':4,
                       '__verification_lengths':torch.tensor([1,2],dtype=torch.int32)},
                      {'hidden_states':torch.ones(4,8)},
                      {'renamed':torch.ones(4,8)},
                      {'renamed':torch.ones(4,4,dtype=torch.float64)},
                      {'b':torch.ones(1), 'a':torch.ones(1)},
                      {'a':torch.ones(1), 'b':torch.ones(1)}]
            for repeat in range(12):
                for tensors in sequence:
                    metadata=pickle.dumps(split(tensors)[0]); start=len(wire.trace)
                    handles=sender.isend_tensor_dict(tensors)
                    # CPU tensors share the CPU channel: header/payload MUST precede
                    # all tensors, in original insertion order.
                    send_trace=wire.trace[start:]
                    count=sum(v.numel()>0 for v in tensors.values() if isinstance(v,torch.Tensor))
                    meta_ops=1 if enabled and metadata==previous else 2
                    self.assertEqual(len(send_trace),meta_ops+count)
                    if enabled: self.assertEqual(send_trace[0][2],(4,))
                    out, received, post=receiver.irecv_tensor_dict()
                    self.assertFalse(post); self.assertEqual(list(out),list(tensors))
                    for key,value in tensors.items():
                        if isinstance(value,torch.Tensor):
                            self.assertEqual(out[key].dtype,value.dtype)
                            self.assertEqual(out[key].shape,value.shape)
                            self.assertTrue(torch.equal(out[key],value))
                        else: self.assertEqual(out[key],value)
                    for handle in handles+received: handle.wait()
                    previous=metadata
            self.assertFalse(wire.queue)
    def test_bad_headers(self):
        for header in ([module._MAGIC,1,0,0], [module._MAGIC,1,1,0],
                       [0,1,0,0], [module._MAGIC,2,0,0], [module._MAGIC,1,0,-1]):
            receiver=coordinator(1,True)
            wire.queue.append(torch.tensor(header,dtype=torch.int64))
            with self.assertRaises(RuntimeError): receiver._pp_metadata_cache.recv(0)
    def test_returned_object_mutation_cannot_poison_cache(self):
        sender=coordinator(0,True); receiver=coordinator(1,True)
        metadata=[('budget',{'nested':[1,2]})]
        sender._pp_metadata_cache.send(metadata,1)
        out=receiver._pp_metadata_cache.recv(0); out[0][1]['nested'][0]=99
        sender._pp_metadata_cache.send(metadata,1)
        self.assertEqual(receiver._pp_metadata_cache.recv(0),metadata)
    def test_empty_metadata_handle_drained(self):
        sender=coordinator(0,True); receiver=coordinator(1,True)
        handles=sender.isend_tensor_dict({}); receiver.irecv_tensor_dict()
        sender._reap_completed_isends()
        self.assertTrue(handles[0].waited); self.assertFalse(sender._pending_isends)
    def test_default_and_pp_only_gate(self):
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        text=ast.unparse(init)
        self.assertIn("group_name == 'pp'",text)
        self.assertIn("os.environ.get('VLLM_PP_METADATA_CACHE_0025', '0') == '1'",text)

if __name__=='__main__': unittest.main()
