#!/usr/bin/env python3
"""GLM PD tail acceptance (GPU state/logits + one E2E round).

No CPU mock contract tests. Run only against a disposable GLM deployment.

  python test_pd_tail_gpu.py instrument /path/to/patched/vllm-tree
    Adds diagnostic hooks ONLY to that disposable tree. Mount a writable
    /tailcheck into both engines; set TAIL_TEST_ROLE=prefill/decode. Create
    /tailcheck/capture AFTER warmup, then send one max_tokens=1 pair with
    cache_salt=tail-bits-<unique>. Remove capture before E2E. Never ship the
    instrumented tree in an image or a source patch.
  python test_pd_tail_gpu.py compare /tailcheck --salt tail-bits-<unique>
    Requires torch+CUDA. Compares source full-prefill vs destination restore,
    target logits, draft candidate IDs, unary logits and selector scores.
  VLLM_API_KEY=... python test_pd_tail_gpu.py e2e --prefill http://...:8301 \
      --decode http://...:8302 --model canada-quant/GLM-5.3-Flash-W4A16-MTP

The short deterministic token prompts deliberately separate transport/state
correctness from tokenizer/template changes. No repeated performance loops.
Use --baseline on the original deployment for one same-workload hop comparison.
Use --smoke for a single pair after the formal image/compose restart.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid


def instrument(root):
    def replace(relative, old, new):
        p = root / relative
        text = p.read_text()
        assert text.count(old) == 1, (relative, old)
        p.write_text(text.replace(old, new))

    connector = 'vllm/distributed/kv_transfer/kv_connector/v1/glm_pd_tail_connector.py'
    replace(connector, '            self.tail_pages.pack(ref, tensors)', '''            if ref.salt.startswith("tail-bits-"):
                torch.save({"names": [x.name for x in selections], "tensors": [x.cpu() for x in tensors]},
                           f"/tailcheck/{ref.salt}-source-{self.worker_adapter.worker_id}.pt")
            self.tail_pages.pack(ref, tensors)''')
    replace(connector, '                self.tail_restored[req] = (ref, hidden)', '''                if ref.salt.startswith("tail-bits-"):
                    torch.save({"names": [x.name for x in selections], "tensors": [x.gather().cpu() for x in selections] + [x.cpu() for x in hidden]},
                               f"/tailcheck/{ref.salt}-restored-{self.worker_adapter.worker_id}.pt")
                self.tail_restored[req] = (ref, hidden)''')
    replace('vllm/v1/worker/gpu/model_runner.py', '        invalid_drafts = None\n', '''        from pathlib import Path
        if Path("/tailcheck/capture").exists() and input_batch.num_draft_tokens == 0:
            import os
            torch.save(logits.cpu(), "/tailcheck/logits-" + os.environ["TAIL_TEST_ROLE"] + ".pt")
        invalid_drafts = None
''')
    replace('vllm/v1/worker/gpu/pp_draft_tail.py', '    walk(candidate_ids, scores, views, rows)\n', '''    from pathlib import Path
    if Path("/tailcheck/capture").exists():
        import os
        torch.save({"ids": candidate_ids.cpu(), "unary": unary_logits.cpu(), "scores": scores.cpu()},
                   "/tailcheck/candidates-" + os.environ["TAIL_TEST_ROLE"] + ".pt")
    walk(candidate_ids, scores, views, rows)
''')


def compare(root, salt):
    import torch
    assert torch.cuda.is_available()
    assert salt and salt.startswith('tail-bits-')

    def equal(label, a, b):
        assert a.shape == b.shape and a.dtype == b.dtype, label
        x = a.cuda().contiguous().view(torch.uint8)
        y = b.cuda().contiguous().view(torch.uint8)
        assert torch.equal(x, y), (label, int((x != y).sum()))

    for rank in range(4):
        a = torch.load(root / f'{salt}-source-{rank}.pt', weights_only=True)
        b = torch.load(root / f'{salt}-restored-{rank}.pt', weights_only=True)
        assert a['names'] == b['names']
        for i, (x, y) in enumerate(zip(a['tensors'], b['tensors'], strict=True)):
            equal(f'rank{rank}/tensor{i}', x, y)
        print('STATE_BITS_PASS', rank, len(a['tensors']), flush=True)
    for name in ('logits', 'candidates'):
        a = torch.load(root / f'{name}-prefill.pt', weights_only=True)
        b = torch.load(root / f'{name}-decode.pt', weights_only=True)
        if isinstance(a, torch.Tensor):
            a, b = {'tensor': a}, {'tensor': b}
        assert a.keys() == b.keys()
        for key in a:
            equal(name + '/' + key, a[key], b[key])
        print('FIRST_STEP_BITS_PASS', name, flush=True)


def e2e(args):
    key = os.environ['VLLM_API_KEY']

    def request(endpoint, tokens, salt):
        body = dict(model=args.model, prompt=tokens, cache_salt=salt,
                    max_tokens=16, temperature=0, return_token_ids=True)
        req = urllib.request.Request(endpoint.rstrip('/') + '/v1/completions',
            data=json.dumps(body).encode(), headers={
                'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=240) as response:
            result = json.load(response)
        return result, time.monotonic() - start

    def pair(length, label, tokens=None, salt=None):
        tokens = tokens if tokens is not None else [1234] * length
        salt = salt or 'tail-e2e-' + uuid.uuid4().hex
        a, prefill = request(args.prefill, tokens, salt)
        b, decode = request(args.decode, tokens, salt)
        ids = a['choices'][0]['token_ids']
        assert ids is not None and ids == b['choices'][0]['token_ids'], label
        cached = b['usage']['prompt_tokens_details']['cached_tokens']
        if not args.baseline:
            assert cached == len(tokens), (label, cached)
        print(json.dumps(dict(case=label, length=len(tokens), prefill_s=prefill,
                              decode_s=decode, cached=cached, equal_tokens=True)), flush=True)
        return ids, salt

    if args.smoke:
        pair(5624, 'formal-image-smoke')
        return
    if not args.baseline:
        for remainder in (1, 16, 504, 4600, 5119):
            pair(5120 + remainder, f'remainder-{remainder}')
    pair(107000, '107k')
    if not args.baseline:
        ids, salt = pair(5624, 'multi-turn-1')
        pair(0, 'multi-turn-2', [1234] * 5624 + ids + [5678] * 33, salt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: pair(5136 + i * 17, f'c8-{i}'), range(8)))
    print('E2E_PASS', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    ins = sub.add_parser('instrument'); ins.add_argument('root', type=Path)
    comp = sub.add_parser('compare'); comp.add_argument('root', type=Path)
    comp.add_argument('--salt', required=True)
    end = sub.add_parser('e2e')
    end.add_argument('--prefill', default='http://127.0.0.1:8301')
    end.add_argument('--decode', default='http://127.0.0.1:8302')
    end.add_argument('--model', default='canada-quant/GLM-5.3-Flash-W4A16-MTP')
    end.add_argument('--baseline', action='store_true')
    end.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.command == 'instrument': instrument(args.root)
    elif args.command == 'compare': compare(args.root, args.salt)
    else: e2e(args)


if __name__ == '__main__':
    main()
