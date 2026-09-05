import json,time,urllib.request,os
url=os.environ.get('VLLM_BASE_URL','http://127.0.0.1:8000')+'/v1/completions';headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['VLLM_API_KEY']}
results=[]
for j,n in enumerate([70,140,250]):
 text=''.join(f'Record {i}: The service stores immutable records on disk and checks sequence numbers before updating state. Readers may reuse verified prefixes, but incomplete writes must never be exposed.\n' for i in range(n))+'\nQuestion: Explain the rules for safely reusing cached state, in a few sentences. Answer:'
 runs=[]
 for r in range(3):
  body={'model':'RadixArk/Qwen3.8-Flash-Next-NVFP4','prompt':text,'max_tokens':48,'temperature':0,'seed':42,'top_p':1,'logprobs':5,'return_token_ids':True,'cache_salt':f'cache-regression-{os.environ.get("RUN_TAG","stage1")}-{j}'}
  t=time.perf_counter()
  with urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(body).encode(),headers=headers),timeout=900) as resp:data=json.load(resp)
  data['wall_s']=time.perf_counter()-t;runs.append(data)
 first=runs[0]['choices'][0]
 summary={'case':j,'prompt_tokens':runs[0]['usage']['prompt_tokens'],'cached':[x['usage']['prompt_tokens_details']['cached_tokens'] for x in runs],'times':[x['wall_s'] for x in runs],'text_equal':[x['choices'][0]['text']==first['text'] for x in runs],'token_ids_equal':[x['choices'][0].get('token_ids')==first.get('token_ids') for x in runs],'first_top_logprobs':[x['choices'][0]['logprobs']['top_logprobs'][0] for x in runs]}
 print(json.dumps(summary),flush=True)
 assert summary['cached'][0] == 0 and all(c > 0 for c in summary['cached'][1:]), summary
 assert isinstance(first.get('token_ids'),list) and len(first['token_ids']) > 0
 assert all(summary['text_equal']) and all(summary['token_ids_equal']), summary
 assert all(lp==summary['first_top_logprobs'][0] for lp in summary['first_top_logprobs']), summary
 results.append({'summary':summary,'responses':runs})
with open('/tmp/cache-regression-results.json','w') as f:json.dump(results,f)
