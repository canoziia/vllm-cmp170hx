#!/usr/bin/env python3
"""GPU regression for uniform MTP verification with boundary-capped acceptance."""

import torch

from vllm.v1.worker.gpu.model_runner import _cap_decode_checkpoint_acceptance

cases = 0
for depth in (1, 3, 5, 6):
    positions: list[int] = []
    counts: list[int] = []
    prompts: list[int] = []
    for position in range(120, 140):
        for count in range(1, depth + 2):
            positions.append(position)
            counts.append(count)
            prompts.append(100)
    # Prefill rows must remain untouched.
    positions += [127, 128]
    counts += [depth + 1, depth + 1]
    prompts += [200, 200]
    size = len(positions)
    idx = torch.arange(size, device="cuda", dtype=torch.int32)
    computed = torch.tensor(positions, device="cuda", dtype=torch.int32)
    prompt = torch.tensor(prompts, device="cuda", dtype=torch.int32)
    sampled = torch.tensor(counts, device="cuda", dtype=torch.int32)
    rejected = depth + 1 - sampled.clone()
    _cap_decode_checkpoint_acceptance[(size,)](
        idx, computed, prompt, sampled, rejected, UNIT=128
    )
    torch.cuda.synchronize()
    for row, (position, count, prompt_len) in enumerate(zip(positions, counts, prompts)):
        expected = min(count, 128 - position % 128) if position >= prompt_len else count
        assert sampled[row].item() == expected
        assert (sampled[row] + rejected[row]).item() == depth + 1
        cases += 1
print(f"DECODE_ACCEPTANCE_CAP cases={cases} PASS")
