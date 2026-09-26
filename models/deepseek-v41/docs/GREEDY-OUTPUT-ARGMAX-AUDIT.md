# Greedy output: what actually changes it, and how to measure it

2026-09-26, six-GPU SM80 DeepSeek-V4.1-Flash deployment (PP6, DSpark K=5,
max_num_seqs=32, async scheduling, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`).

## Why the obvious test does not work

The natural correctness A/B for a scheduler or speculative-decoding change is
"same prompt, `temperature=0`, are the token IDs identical?". On this stack that
question has no answer, because the answer changes between two runs of the *same*
configuration:

| comparison (identical config, same prompts, greedy) | byte-identical outputs |
|---|---|
| concurrency 1, run vs run (batch shape fixed) | **4/4 identical** |
| concurrency 4, run vs run | 1/4 |
| concurrency 32, run vs run | **3/32**, diverging from token ~27 |
| concurrency 1 vs 4 (shape changes) | 3/4 and 1/4 |

And the deeper case: **one request, single-row shape, sent 8 times** produced
**three different argmax tokens** at one position (61463 x6, 1350 x1, 11309 x1)
and at another the top-2 gap measured exactly `0.00000`. So the emitted string is
not a well-defined function of the input here once a decision is close.

This matches upstream: vllm-project/vllm#41758 (ngram spec decode changes greedy
output), #54506 (RFC: batch invariance for spec decode must cover the forward
pass, `M=1` vs `M=k+1`; measured acceptance 100% -> 72% purely from this), and
#39096 (on SM<90 `VLLM_BATCH_INVARIANT=1` does not hold with torch.compile and/or
CUDA graphs). The model-specific DeepSeek-V4 indexer variant was already fixed in
#52492, and the guard is present in our pinned source
(`deepseek_v4/attention.py`, `deepseek_v4_1/attention.py`,
`deepseek_v4_1/ampere/ampere_sparse.py`), so it does not explain our numbers.

## The test that does work

Under greedy decoding, every token an engine emits must be an argmax of the target
model *at its own prefix*, whatever the acceptance policy did. So audit every
position rather than comparing two strings, and allow exact ties:

    # 1. capture full token-ID sequences (greedy, cold cache_salt each request)
    node tests/token-id-probe.mjs <label> --requests 32 --concurrency 1 --max-tokens 128
    #    do this twice: once with DSpark on (control file 1) and once with it off (0)
    # 2. audit every position against a fixed single-row teacher-forced oracle
    ./set-dspark 0            # canonical oracle shape: concurrency 1, K=0
    python3 tests/argmax-audit.py gtprobe/<label>-ar-c1/run0.json gtprobe/<label>-spec-c1/run0.json

`tests/greedy-margin-audit.py` is the cheaper variant that only examines the first
divergence point of each pair and reports the logit margin; it is useful for a
quick look but, as the results below show, only the per-position audit is a
statement about the whole sequence.

## Result: DSpark adds no violations over plain decode

4096 positions per arm (32 sequences x 128 tokens), audited token by token:

| arm | at/near argmax | exact float tie | **below target argmax** | rate |
|---|---|---|---|---|
| AR (`DSpark` off, 1 row per step) | 4032 | 30 | **64** | **1.56%** |
| DSpark K=5 (6 rows per step) | 4032 | 27 | **64** | **1.56%** |

Every flagged position was 0.25-0.5 nats below the top score - i.e. the runner-up
still carries 61-78% of the probability - and none was a large gap. Two things
follow:

1. **`1.56%` is the floor of the method, not a defect rate.** The oracle scores a
   teacher-forced prefill while the sequence was produced by decode steps over
   paged FP8 KV, and those two execution paths already disagree on close calls in
   the *absence* of speculative decoding.
2. **Speculative decoding is indistinguishable from plain decode on that floor**
   (64 vs 64, with `argmax`/`exact_tie` differing by 3 and 27 out of 4096).
   Enabling DSpark does change the emitted text relative to disabling it - 0/32
   sequences matched - but at the same violation rate as plain decode against
   itself, so the change is tie-breaking, not wrong tokens.

## What this does NOT establish

- It does not prove two configurations emit the *same* text; they demonstrably do
  not, and cannot on this stack.
- Position-level overlap between the AR and DSpark flag sets is **not** measured:
  the audit log keeps only the first few flagged positions per sequence, so
  "are these the same 64 positions" is open. If that matters, have
  `argmax-audit.py` persist every position.
- The c32 arm was never audited: that run was stopped part-way, and the JSON left
  behind was a stale copy of the c1 result (identical counters gave it away), so
  it was deleted rather than reported.
- Quality/safety of a *specific* acceptance policy still needs its own gate;
  what this doc kills is the idea that string equality can serve as that gate.

## Consequence for the adaptive-verification branch

`docs/ADAPTIVE-PP6-RUNTIME-VALIDATION.md` lists six greedy token-ID pairs (code
diverged at token 23, prose at 5, JSON at 9) as an unresolved quality gate. Given
the table above, that observation cannot distinguish a trimming bug from this
deployment's normal tie behaviour, and the report already says it proves neither
incorrectness nor strict greedy equivalence. The live gate is the per-position
argmax audit: an adaptive change has to show a below-argmax rate that is not
worse than the 1.56% plain-decode floor.
