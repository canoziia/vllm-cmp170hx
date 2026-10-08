#!/usr/bin/env python3
"""Small real-API local prefix/PD regression. API key comes from the environment."""
import argparse
import concurrent.futures
import json
import os
import time
import urllib.request
import uuid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefill', default='http://127.0.0.1:8301')
    p.add_argument('--decode', default='http://127.0.0.1:8302')
    p.add_argument('--model', default='canada-quant/GLM-5.3-Flash-W4A16-MTP')
    p.add_argument('--baseline', action='store_true')
    args = p.parse_args()
    key = os.environ['VLLM_API_KEY']

    def request(length, salt, endpoint=args.prefill):
        body = dict(model=args.model, prompt=[1234] * length, cache_salt=salt,
                    max_tokens=8, temperature=0, return_token_ids=True,
                    logprobs=1)
        req = urllib.request.Request(endpoint + '/v1/completions',
            data=json.dumps(body).encode(), headers={
                'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=240) as response:
            result = json.load(response)
        choice = result['choices'][0]
        cached = result['usage']['prompt_tokens_details']['cached_tokens']
        return dict(length=length, cached=cached, seconds=time.monotonic()-start,
                    ids=choice['token_ids'], scores=choice['logprobs']['token_logprobs'])

    def show(case, result):
        print(json.dumps(dict(case=case, **result)), flush=True)

    salt = 'prefix-retention-' + uuid.uuid4().hex
    results = []
    for length in (4487, 4561, 4757, 5045, 5084):
        result = request(length, salt)
        show('multi', result)
        results.append(result)
    assert results[0]['cached'] == 0
    if not args.baseline:
        assert all(r['cached'] == 3072 for r in results[1:]), results
    # Same final prompt with a fresh salt: compare fixed greedy outputs and
    # chosen-token scores, not only HTTP status or usage accounting.
    fresh = request(5084, 'prefix-fresh-' + uuid.uuid4().hex)
    show('fresh-correctness', fresh)
    assert fresh['cached'] == 0
    assert fresh['ids'] == results[-1]['ids']
    delta = max(abs(a-b) for a, b in zip(fresh['scores'], results[-1]['scores']))
    assert delta < 0.1, delta
    print(json.dumps(dict(case='score-check', max_abs_logprob_delta=delta)), flush=True)
    # Check the two sides of a hash boundary; don't claim the last incomplete
    # drafter page or an unmaterialized KDA state as a hit.
    for length in (4096, 4097):
        salt = 'prefix-remainder-' + uuid.uuid4().hex
        show('remainder-first', request(length, salt))
        hit = request(length + 17, salt)
        show('remainder-second', hit)
        if not args.baseline:
            assert hit['cached'] == 3072, hit
    if not args.baseline:
        def concurrent_case(i):
            salt = 'prefix-c8-' + uuid.uuid4().hex
            first = request(4487 + i, salt)
            second = request(4561 + i, salt)
            assert first['cached'] == 0 and second['cached'] == 3072
            assert len(second['ids']) == 8
            return second
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for result in pool.map(concurrent_case, range(8)):
                show('c8-second', result)
        salt = 'prefix-tail-' + uuid.uuid4().hex
        a = request(5624, salt)
        b = request(5624, salt, args.decode)
        show('tail-prefill', a)
        show('tail-decode', b)
        assert b['cached'] == 5624 and a['ids'] == b['ids']
    print('PREFIX_RETENTION_PASS', flush=True)


if __name__ == '__main__':
    main()
