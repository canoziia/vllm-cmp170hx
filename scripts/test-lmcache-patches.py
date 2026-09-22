#!/usr/bin/env python3
"""Offline regression gates; run inside BOTH built images, no GPU/model needed.

Uses real official native FS extensions and a temporary directory. This is not
an end-to-end DeepSeek GPU/L1/L2 acceptance test.
"""
import tempfile
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
import lmcache.cuda_ops  # Explicitly check native ABI, not just import lmcache.
import lmcache.lmcache_fs
import lmcache.lmcache_native
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector  # noqa: F401
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter, _object_key_to_string,
)
from lmcache.v1.distributed.l2_adapters.fs_key_codec import object_key_to_filename
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import _make_adapter_class
from lmcache.v1.distributed.storage_controllers.store_controller import _group_keys_by_shape
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry


def key(n, rank=0, group=0, salt="fixture"):
    return ObjectKey(model_name="fixture", chunk_hash=n.to_bytes(32, "big"),
                     kv_rank=rank, object_group_id=group, cache_salt=salt)


def wait_for(fn):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = fn()
        if result is not None:
            return result
        time.sleep(0.02)
    raise TimeoutError("native fixture completion exceeded 10 seconds")


class RegressionTests(unittest.TestCase):
    def test_salt_and_shape_boundaries(self):
        a = key(1)
        self.assertEqual(len(_object_key_to_string(a).split("@")), 4)
        self.assertNotEqual(_object_key_to_string(a),
                            _object_key_to_string(replace(a, cache_salt="other")))
        self.assertEqual(len(_group_keys_by_shape([
            a, key(2), key(3, group=1), key(4, rank=1)])), 3)

    def test_six_rank_layout_lifecycle(self):
        registry = LayoutDescRegistry()
        ranks = [ObjectKey.ComputeKVRank(6, r, 6, r) for r in range(6)]
        descs = []
        for i, rank in enumerate(ranks):
            groups = {g: MemoryLayoutDesc(
                shapes=[torch.Size([7 if i < 5 else 5, 1024, g + 1])],
                dtypes=[torch.uint8]) for g in range(2)}
            descs.append(groups)
            registry.register("fixture", 6, groups[0],
                              AttnWindowDesc(num_chunks_in_sw=[-1, -1]),
                              groups, kv_rank=rank)
        layouts = registry.find_group_layout_descs("fixture", 6)
        for rank, groups in zip(ranks, descs):
            for gid, desc in groups.items():
                self.assertEqual(layouts[(rank, gid)], desc)
        registry.register("fixture", 6, descs[0][0],
                          AttnWindowDesc(num_chunks_in_sw=[-1, -1]),
                          descs[0], kv_rank=ranks[0])
        registry.unregister("fixture", 6, ranks[0])
        self.assertIn((ranks[0], 0), registry.find_group_layout_descs("fixture", 6))
        registry.unregister("fixture", 6, ranks[0])
        self.assertNotIn((ranks[0], 0), registry.find_group_layout_descs("fixture", 6))
        for rank in ranks[1:]:
            registry.unregister("fixture", 6, rank)
        self.assertIsNone(registry.find_group_layout_descs("fixture", 6))

    def test_real_native_fs_mixed_sizes(self):
        with tempfile.TemporaryDirectory(prefix="lmcache-native-fixture-") as directory:
            client = lmcache.lmcache_fs.LMCacheFSClient(directory, 2, "", False, 0)
            adapter = NativeConnectorL2Adapter(client)
            try:
                keys = [key(1), key(2, group=1), key(3, rank=1)]
                data = [bytearray([i + 1]) * n for i, n in enumerate([4096, 8192, 6144])]
                objs = [SimpleNamespace(byte_array=memoryview(b), get_size=lambda b=b: len(b))
                        for b in data]
                tid = adapter.submit_store_task(keys, objs)
                result = wait_for(lambda: adapter.pop_completed_store_tasks().get(tid))
                self.assertGreaterEqual(result, 0, result)
                out = [bytearray(len(b)) for b in data]
                tid = adapter.submit_load_task(keys, [
                    SimpleNamespace(byte_array=memoryview(b)) for b in out])
                result = wait_for(lambda: adapter.query_load_result(tid))
                self.assertEqual(out, data)
            finally:
                adapter.close()

    def test_native_fs_adopts_completed_files(self):
        with tempfile.TemporaryDirectory(prefix="lmcache-adopt-fixture-") as directory:
            keys = [key(11, salt=""), key(12, rank=1, group=1, salt="")]
            for i, obj_key in enumerate(keys):
                with open(f"{directory}/{object_key_to_filename(obj_key)}", "wb") as f:
                    f.write(bytes([i + 1]) * (4096 + i * 1024))
            with open(f"{directory}/unfinished.tmp", "wb") as f:
                f.write(b"not an object")
            client = lmcache.lmcache_fs.LMCacheFSClient(directory, 1, "", False, 0)
            cls = _make_adapter_class(NativeConnectorL2Adapter)
            adapter = cls(client, base_path=directory, adopt_existing=True)
            try:
                self.assertEqual(adapter.adopt_existing_keys(), 2)
            finally:
                adapter.close()


if __name__ == "__main__":
    unittest.main()
