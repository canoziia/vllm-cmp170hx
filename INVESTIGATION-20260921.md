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

## Later controlled A/B and a second hot-OFF performance regime

Same live full instance, after warmup and with tracer off: alternating ON→OFF→ON→OFF→ON single-stream measured Full/Output tok/s: ON 145.77/139.04, OFF 24.67/24.45, ON 146.71/139.24, OFF 36.77/36.25, ON 146.08/138.94. Mode restored ON. The OFF slow run has 215 of 999 inter-event intervals >60ms, predominantly every second event; fast OFF run has one >60ms. Repeat OFF runs 25.41/25.17 and 37.02/36.52 show the same split. Thus a fixed ~9ms hot-OFF tax versus historical 41 is **falsified** by a second, intermittent slow regime; the fast same-instance OFF baseline is ~36–37 Full tok/s. ON remains ~24.2–24.5 steps/s across alternation. Original historical 41 was Zero-Clean, not a matching CPU Engram A/B.

Async OFF trace `/root/app/deepseek-v41/cache/vllm-perf-debug/steps-full-off-phase-rank*.jsonl` captured a transition in a 1000-token c1 request: client event gaps >60ms begin ~output event 540 and predominantly alternate. At PP5, early (steps 80–249) vs late (590–799), target GPU Event 3.37→3.54ms, but `draft_context_total` GPU 0.35→3.83ms and PP5 sample CPU 4.83→11.60ms, while PP1/3/4 CPU execute and metadata also increase. OFF still runs context-only draft and sample/draft feedback. This correlated spike is **not yet an isolated cause**: many host spans rise concurrently, and CUDA Event timings can include queued work. It is a candidate critical-path source, not proven Engram, source layer, or LMCache. External py-spy during another slow OFF run mostly saw PP0 waiting for scheduler broadcast and cannot resolve the GPU side. A bounded nvidia-smi sample during an intermediate 33.86 Full OFF run showed GPU0–4 clocks around 1470–1485MHz; not a clock-collapse explanation for that run.

A short subsequent generation verified file restoration and scheduler logged K=5 at 01:35:07 UTC; vLLM container ID unchanged, healthy, debug disabled.

## PP2 spike reproduced with actual aligned cohort, and tracer perturbation

The original c32 trace (`/tmp/decode-traces/steps-decode-trace-seq32-20260920-143918-c32-rank*.jsonl`) at actual 6/11/15 cohorts showed PP2 execute ~30.34/30.51/28.04ms while adjacent ranks were ~7–17ms, and PP2 GPU target 13.10/20.77/20.84ms. The new c6 request settles into 1/5 cohorts, not exactly six: at cohort 5 PP2 execute 30.89ms (median 29.52) versus PP0/1/3/4/5 24.43/17.15/16.70/19.14/18.46ms; PP2 `metadata_and_model_inputs` 12.01ms, `forward_dispatch` 14.85ms, GPU target 12.26ms. This is not just Engram host lookup: other ranks' forward dispatch also rose to ~9–11ms, though PP2 was highest. c11 trace with all 11 concurrent shows PP2 execute 47.50ms, metadata 37.53ms, forward only 0.59ms, GPU target 18.54ms vs PP3/4 20.53/21.32ms; PP3/4 receive enqueue 84.9/94.0ms. A repeat c11 settled into 5/6 cohorts: PP2 cohort 5 execute 21.04ms, cohort 6 10.39ms; no blanket PP2 30ms. Full per-rank records remain on VM under `steps-full-cohort6-*`, `steps-full-cohort11-*` and `steps-full-cohort11-repeat-*`.

Important probe effect: c6 uninstrumented warmup 445.0/423.2 Full/Output, trace 370.5/352.2, post-trace uninstrumented 423.7/404.1 and 424.2/405.9. For c11 uninstrumented runs varied 558–751 Full, trace 507 Full. Dense `sample_every=1` Event probes on all six ranks materially perturb c6/c11, so the 30–47ms spans cannot be uncritically assigned to production without lower-rate A/B. External 120Hz nonblocking py-spy `pp2-c11-gil.raw` during an uninstrumented c11 750.75 Full run had few active main-thread samples and showed metadata builders and PP waits but no persistent Engram frame. No single-factor Engram/L14/L20/LMCache change has yet been made.

## Aligned PP2 A/B at lower probe rates

Further same-instance c11 probes `steps-full-c11-every3-rank*.jsonl` and `steps-full-c11-every5-rank*.jsonl` (VM cache directory) aligned all six ranks by actual cohort. With every-third sampling and actual cohort 7, PP2 CPU execute mean 27.10ms (median 27.69), metadata+model inputs 20.23ms, forward dispatch 0.35ms; PP0/1/3/4/5 execute 13.91/7.52/8.82/9.46/8.62ms. The measured Full-batch/Output were 699.5/657.9 versus a later uninstrumented 560.2/526.2, so a raw 27ms is not a measured slowdown attribution. Every-fifth A/B with actual 1/3/7 cohorts measured 635.3/601.9, post 653.0/614.5; PP2 execute ~18.7–19.4ms, metadata ~13.8–14.6ms, other ranks 5.6–9.4ms. Repeat every-fifth with actual 1/4/6 cohorts measured 733.5/695.5 and PP2 execute 7.1–8.1ms. Thus PP2 host metadata tail is recurrent but state/cohort dependent and does not have a single constant 30ms cost. Every-10 sampled mostly idle callbacks, proving the admission bug. No controlled Engram/source/LMCache toggle was possible without changing deployed code or restarting the expensive live model; attribution remains open.

## Sampling admission fix for the next debug image

`PerfDebugTracer.begin_step` in the committed optional patch is being updated so callbacks with zero scheduled tokens do not advance the sampling interval nor consume bounded samples. A synthetic 4000-idle/20-real-step test on the fixed image, `sample_every=10`, samples exactly real steps 10 and 20. Fresh pinned-source patch application and syntax checks pass. This is **not live-validated on the full model**: a second fixture attempted while the full model occupied GPUs 0–5 failed in NCCL initialization with CUDA memory exhaustion; the production container stayed healthy. Never restart it just to test the fixture. Dense-trace conclusions above remain marked as probe-perturbed.

## Representative six-rank single-stream ON timeline (old deployed image)

From `steps-full-on-c1-rank*.jsonl`, matched a steady 1/6/1 cohort by nearest host wall start, with stage5 worker counter offset -6; do **not** join stage5 by equal `step` alone. Example rank0 local step 5576 (PP5 5570): rank0 begins t=0 and spends 7.87ms CPU; rank1 begins +0.02ms, `pp_irecv_enqueue` 7.47ms then CPU execute 6.67ms; rank2 +0.09ms, recv 15.50ms then execute 7.24ms; rank3 t≈0, recv 24.01ms then execute 4.59ms; rank4 +0.54ms, recv 29.45ms then execute 6.42ms; rank5 +14.62ms, recv 23.24ms then execute 6.53ms, sample_tokens CPU 8.52ms. Rank5 CPU returns ~53.09ms after rank0 starts, but the *next rank0 decode starts at +41.74ms*, so this is an overlapped pipeline, not a 53ms step. Rank5 Event spans: target 4.61ms, target sampling 0.91ms, context 0.20ms, draft query+Markov 2.61ms (inside draft_propose 2.96ms), sample broadcast 0.11ms and draft broadcast 0.03ms. Feedback GPU waits on earlier broadcast-stream work and spans across queued work; they cannot be added to CPU durations or target Events. A new host-wall alignment script, `scripts/analyze-pp-steps.py`, reports the dominant step offsets and rejects ambiguous cross-rank matches; use SSE output for rate when sampling is sparse. This timeline locates ON stage costs and PP overlap, but **does not establish a causal optimization**.

## Reversible PP2 host-affinity A/B (2026-09-21 03:03–03:08 UTC)

Same full-model container, debug off, DSpark ON, c32, 53 API prompt tokens and 1000 completion tokens/request. Only PP2 main thread (PID 710438) was changed from unrestricted CPU 0–39 (A) to NUMA-node-1 CPU 12–19 (B) and back to 0–39 (A2), with separate per-phase warmups. The script trap restored the original affinity and the container remained healthy. Measured Full-batch/Output tok/s: A 1112.10/1037.11 and 906.62/861.04; B 827.99/784.27 and 1129.64/1061.77; A2 908.94/859.01 and 940.04/858.36. The B range overlaps A and A2; **no reproducible affinity optimization** is demonstrated. This intervention did not move pinned Engram allocations and cannot isolate UVA Engram versus L14/L20 source or GPU synchronization. Raw runs are in `/root/app/deepseek-v41/bench/full-fixed-73d0be8/pp2-affinity/`.

## Unfinished gates

1. Plain A under **same** CPU Engram/LMCache/PP6/Graph and prompt needs a separate full model load or a verified in-place mechanism; cannot claim 41→31.8 attribution from Zero-Clean historical baseline.
2. ON incremental 9–10ms step accounting must use aligned end-to-end critical-path timestamps, not sum overlapping Event spans. Need controlled single-factor optimization A/B.
3. c32 old PP2 30ms requires reproduction on same cohort and decomposition of model-state metadata, Engram UVA, L14/L20 source and host sync; no causal optimization tested yet.
4. LMCache store was observed; evict/restore correctness remains unverified.
