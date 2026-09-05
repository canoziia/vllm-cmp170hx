import assert from 'node:assert/strict';import {randomUUID} from 'node:crypto';
const base=process.env.VLLM_BASE_URL||'http://192.168.3.108:8000',model='RadixArk/Qwen3.8-Flash-Next-NVFP4';
async function post(path,b){const r=await fetch(base+path,{method:'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer '+process.env.VLLM_API_KEY},body:JSON.stringify({model,...b}),signal:AbortSignal.timeout(240000)});const d=await r.json();if(!r.ok)throw Error(JSON.stringify(d));return d;}
for(const max of [64,256,1024,1500]){
 let input=[{role:'user',content:Array.from({length:100},(_,i)=>`Record ${i}: validate checksums, commit immutable transactions and recover safely.`).join('\n')+'\nDescribe transaction recovery in detail.'}],salt=randomUUID();
 let first=await post('/v1/responses',{input,max_output_tokens:max,temperature:0,seed:42,cache_salt:salt,store:false});input.push(...first.output,{role:'user',content:'Continue.'});
 let second=await post('/v1/responses',{input,max_output_tokens:32,temperature:0,seed:42,cache_salt:salt,store:false});let oldTotal=first.usage.input_tokens+first.usage.output_tokens,hit=second.usage.input_tokens_details.cached_tokens;
 console.log(JSON.stringify({test:'responses',max,first:first.usage,second:second.usage,shared_recompute:oldTotal-hit}));assert(oldTotal-hit<=128,'too much shared recomputation');
}
const batch=await Promise.all(Array.from({length:32},async(_,i)=>{
 const salt=randomUUID(),messages=[{role:'user',content:`Session ${i}. `+'Immutable data checksums are verified before recovery. '.repeat(120)+'\nExplain WAL recovery in numbered steps.'}];
 const a=await post('/v1/chat/completions',{messages,max_tokens:256,temperature:0,seed:42,cache_salt:salt,return_token_ids:true});messages.push(a.choices[0].message,{role:'user',content:'Summarize.'});
 const b=await post('/v1/chat/completions',{messages,max_tokens:32,temperature:0,seed:42,cache_salt:salt,return_token_ids:true});const rem=a.usage.prompt_tokens+a.usage.completion_tokens-b.usage.prompt_tokens_details.cached_tokens;
 assert(rem<=128,`concurrent cache miss ${rem}`);return {i,rem,usage:b.usage};}));console.log(JSON.stringify({test:'concurrent',batch}));
console.log('MULTITURN_HIT_TESTS_PASS');
