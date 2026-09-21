# PP6 DeepSeek V4.1 diagnostic ledger (2026-09-21 UTC)

## Contract retained verbatim

请你先通过剪裁的模型验证并修复刚刚退出的bug后，再执行之前的老的三个goal。以后如果遇到压缩，记得原样保留记住老goal的内容文本

定位单流热OFF的性能缺口 为什么同为PP6，真正plain约41 tok/s，而当前debug镜像热关DSpark只有31.8 tok/s？量化不生成draft时仍保留的计算、通信、调度及诊断框架开销，用受控A/B验证归因。
定位DSpark ON的单流step延迟增长 为什么开启后只有约24–25 step/s？拆分target六token验证、采样、draft生成和PP反馈等待，解释相对plain及热OFF增加的每轮耗时，验证可行优化。
定位c32下PP2的异常与流水线阻塞 为什么c1–16基本正常，c32时PP2的CPU execute跨度升到约30ms？按相同内部cohort对齐各rank，区分Engram、L14/L20 source工作、主机同步及资源竞争，验证瓶颈与优化收益。
共同要求：先修复debug的PP形状安全及trace归属问题；结论必须有实际时间线和A/B证据，不用猜测代替归因。
原启动约束：按照之前的配置重新运行起容器，没出问题绝对不要关闭它，因为启动要一个小时。

## Current deployed instance

VM `162.105.151.207`; vLLM container ID `22890c49c8d402280a113144023352e669e71391bd7299c2d69f3dddbc5a45e2`, started 2026-09-20 23:13:27 UTC. Image ID `2d2625fa99a730292b646131655cb21376c38b5037f85a3d5ad06846756186c9`, tag `localhost/deepseek-v41-cmp170hx:73d0be8-debug-eventfix`. PP6 7/7/7/7/7/5, seq32, KV 6GiB/rank, CPU Engram, DSpark, LMCache 16GB, host IPC. Keep running; do not restart to collect plain baseline without separate authorization. DSpark control is `1`, tracer disabled. Production LMCache unchanged, healthy. Deployment `/root/app/deepseek-v41/deployment-73d0be8-debug-kv6-seq32-lmcache/`; bench `/root/app/deepseek-v41/bench/full-fixed-73d0be8/`; trace files `/root/app/deepseek-v41/cache/vllm-perf-debug/steps-full-*.jsonl`.

## Prior exit: evidence and mitigation scope

Original full instance PP2 repeatedly stuck inside `lmcache ... check_event_support -> inspect.signature` after first 1024-token stores; `SigPnd=0x400`; EngineCore `sample_tokens` RPC timeout. A small 8-expert dummy DeepSeek with Zero Engram, 40 layers/PP6, DSpark and independent LMCache reproduced PP2 stack+SIGSEGV when hot CPU profiler was started/stopped. Also reproduced worker hangs in unrelated Triton and SamplingParams paths, including without LMCache, so Event check alone is **not** a root-cause fix. Disabled hot CPU profiler (`torch_profile_steps` rejected in worker and `profile` command rejected), retained Graph-neutral Event timer. Cached LMCache Event capability check after initial fail-closed validation as a separate image-layer change. Four consecutive c32×1050-token fixture runs with LMCache, async tracer on all six ranks and an intentionally requested but ignored `torch_profile_steps=32` completed; all six rank traces and 768 LMCache 1024-token stores verified. This is evidence of mitigation, not proof all native races are eliminated. Fixture artifacts `/root/app/diag/fixture-*` and `/root/app/deepseek-v41/cache/vllm-perf-debug/steps-fixture-*`.

Repo branch `deepseek-v41-pp6-cpu-engram`, commit `10cbe00`: optional debug patch, separate LMCache image-layer patch with base-source SHA check, control script and build instructions. Actual full image was built as a layer on 101438e after GitHub upstream fetch failed; it contains precisely the disabled profiler file and Event cache file, and image ID above is verified. The default build remains primer-only.

## Full-model measurements (prefill/warmup excluded)

Counting prompt API usage 53 prompt tokens, 1000 completion tokens per request; `cache_salt` unique. Table order is Full-batch decode then Output, both tok/s.

| Mode | c | Full-batch decode | Output | Notes |
|---|---:|---:|---:|---|
| ON | 1 | 144.72–146.92 | 135.87–139.65 | repeated after warmup, 24.02–24.54 steps/s; 5.95 tokens/event |
| OFF | 1 | 25.41–37.02 | 25.17–36.52 | three repeats after warmup; middle run had p90 event gap 86ms versus ~30ms in fast runs |
| ON | 32 | 685.3–942.4 | 654.9–894.4 | substantial cohort/run variability, not a stable regression estimate |
| OFF | 32 | 451.3–587.2 | 439.7–570.1 | substantial variability |

Historical `41.06` Full / `39.45` Output from `/root/app/deepseek-v41/bench/pp6-shadow-matrix/s3-776776-plain-exact16/orchestrator.log` used *Zero-Clean* model, PP6 partition **7/7/6/7/7/6**, 500 completion tokens, seq8 and different Graph configuration. It is **not** a same-condition CPU Engram/LMCache plain A/B. Another s4 partition 7/7/7/7/7/5 plain run achieved 40.87 Full but also Zero-Clean, seq8. Never subtract that from the present hot-OFF to claim component attribution.

ON single stream trace `steps-full-on-c1-rank*.jsonl` aligns 168 real 1/6/1 decode steps across ranks; wall ~41ms/step. PP5: target GPU 4.42ms; sampling GPU 0.89ms; context GPU 0.20ms; draft query+Markov GPU 2.60ms; CPU draft context 1.99ms and draft metadata 2.78ms; full PP5 sampling span CPU 8.20ms. PP0/2 execute CPU 6.59/7.88ms. These are partly overlapping GPU/CPU events, **not addends** to 41ms. OFF trace `steps-full-off-c1-v2-rank*.jsonl` sampled many idle steps and induced pronounced probe effects; only filter 1/1/0 real decode. Its PP5 context GPU ~0.26ms and CPU ~1.9ms; no draft query. Hot-OFF retains feedback/context/aux. The uninstrumented OFF c1 median per-event gap ~26ms in fast runs, while the slow run had long-tail gaps; sampled run 24 tok/s is not representative. ON traces before/after sampling gave ~24.2–24.5 steps/s, so ON sampling effect was small.

c32 `steps-full-on-c32-rank*.jsonl` actual cohorts 1/7/24, repeat `steps-full-on-c32-repeat-rank*.jsonl` cohorts 7/25. At cohort 24 PP2 CPU execute 7.86ms, metadata+model inputs 5.88ms, Graph dispatch 0.08ms; PP3/4 GPU target Event 38.42/40.34ms. Repeat cohort 25 PP2 execute 8.84ms, metadata 6.86ms; PP3/4 GPU 40.54/42.90ms. Old PP2 30ms phenomenon was with 6/11/15 cohorts in a different run; it did not reproduce under current aligned cohorts. PP2 L14/L20 source and L14 Engram contribution remains unseparated. `py-spy` external raw sample `/root/app/deepseek-v41/bench/full-fixed-73d0be8/pp2-c32-external.raw`; most main-thread samples in PP communication/wait rather than a persistent Engram frame, but external sampling cannot measure device UVA lookup inside Graph.

## Unfinished gates

1. Plain A under **same** CPU Engram/LMCache/PP6/Graph and prompt needs a separate full model load or a verified in-place mechanism; cannot claim 41→31.8 attribution from Zero-Clean historical baseline.
2. ON incremental 9–10ms step accounting must use aligned end-to-end critical-path timestamps, not sum overlapping Event spans. Need controlled single-factor optimization A/B.
3. c32 old PP2 30ms requires reproduction on same cohort and decomposition of model-state metadata, Engram UVA, L14/L20 source and host sync; no causal optimization tested yet.
4. LMCache store was observed; evict/restore correctness remains unverified.
