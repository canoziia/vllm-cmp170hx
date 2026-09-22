#!/usr/bin/env node
/**
 * Benchmark an OpenAI-compatible vLLM server. Node.js 18+; no dependencies.
 *
 * API key is read from VLLM_API_KEY. It is never written to result files.
 * Every request gets a unique cache_salt; cached_tokens must be zero.
 *
 * Standard examples
 * -----------------
 *
 * 1. Ordinary decode: a short (~16-token user input before chat templating),
 *    500 output tokens/request, ignore EOS, concurrency 1/2/4/8/16/32:
 *
 *    VLLM_API_KEY=... node scripts/benchmark-vllm.mjs \
 *      --mode decode --base-url http://127.0.0.1:8000 \
 *      --model deepseek-ai/DeepSeek-V4.1-Flash \
 *      --concurrency 1,2,4,8,16,32 --output results/decode.jsonl
 *
 * 2. Prefill: approximately 65,536 input tokens and exactly one output token.
 *    The authoritative input count is the server's usage.prompt_tokens:
 *
 *    VLLM_API_KEY=... node scripts/benchmark-vllm.mjs \
 *      --mode prefill --input-tokens 65536 --concurrency 1 \
 *      --base-url http://127.0.0.1:8000 \
 *      --model deepseek-ai/DeepSeek-V4.1-Flash \
 *      --output results/prefill-65536.jsonl
 *
 * 3. Near-full speculative acceptance: deterministic counting prompt,
 *    1,000 output tokens/request, ignore EOS. For DSpark K=5 the maximum is
 *    six emitted tokens/step; accepted_tokens_per_step close to 6 indicates
 *    near-full acceptance:
 *
 *    VLLM_API_KEY=... node scripts/benchmark-vllm.mjs \
 *      --mode counting --concurrency 1,2,4,8,16,32 \
 *      --base-url http://127.0.0.1:8000 \
 *      --model deepseek-ai/DeepSeek-V4.1-Flash \
 *      --output results/counting.jsonl
 *
 * Decode definitions
 * ------------------
 * - One streamed token_ids event is treated as one speculative decode step.
 * - per_request_step_s = (events - 1) / (last token - first token).
 * - full_batch_step_s = events in the all-requests-active intersection /
 *   concurrency / intersection duration.
 * - accepted_tokens_per_step = completion tokens / streamed events.
 * - output_tok_s includes TTFT and tail drain; full_batch_tok_s does not.
 */

import { mkdirSync, appendFileSync } from "node:fs";
import { dirname } from "node:path";
import { randomUUID } from "node:crypto";

const defaults = {
  mode: "decode",
  baseUrl: process.env.VLLM_BASE_URL || "http://127.0.0.1:8000",
  model: process.env.VLLM_MODEL || "",
  concurrency: "1",
  inputTokens: null,
  maxTokens: null,
  timeoutMs: 30 * 60 * 1000,
  seed: 20260922,
  output: "benchmark-results.jsonl",
};

function help() {
  console.log(`Usage: node scripts/benchmark-vllm.mjs [options]

  --mode decode|prefill|counting
  --base-url URL
  --model MODEL                 Discover from /v1/models if omitted
  --concurrency LIST            e.g. 1,2,4,8,16,32
  --input-tokens N              Required override for prefill approximation
  --max-tokens N                Override mode default (decode=500, prefill=1, counting=1000)
  --timeout-ms N
  --seed N
  --output PATH                 Append one JSON object per concurrency
  --help

Environment: VLLM_API_KEY (required), VLLM_BASE_URL, VLLM_MODEL`);
}

function parseArgs(argv) {
  const c = { ...defaults };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const take = () => {
      if (++i >= argv.length) throw new Error(`Missing value after ${arg}`);
      return argv[i];
    };
    if (arg === "--mode") c.mode = take();
    else if (arg === "--base-url") c.baseUrl = take();
    else if (arg === "--model") c.model = take();
    else if (arg === "--concurrency") c.concurrency = take();
    else if (arg === "--input-tokens") c.inputTokens = Number(take());
    else if (arg === "--max-tokens") c.maxTokens = Number(take());
    else if (arg === "--timeout-ms") c.timeoutMs = Number(take());
    else if (arg === "--seed") c.seed = Number(take());
    else if (arg === "--output") c.output = take();
    else if (arg === "--help" || arg === "-h") { help(); process.exit(0); }
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!["decode", "prefill", "counting"].includes(c.mode)) {
    throw new Error("--mode must be decode, prefill, or counting");
  }
  c.concurrencies = String(c.concurrency).split(",").map(Number);
  if (c.concurrencies.some(n => !Number.isInteger(n) || n <= 0)) {
    throw new Error("--concurrency must contain positive integers");
  }
  const modeMax = { decode: 500, prefill: 1, counting: 1000 };
  c.maxTokens ??= modeMax[c.mode];
  if (c.inputTokens === null) c.inputTokens = c.mode === "prefill" ? 65536 : 16;
  for (const [name, n] of [["input-tokens", c.inputTokens], ["max-tokens", c.maxTokens], ["timeout-ms", c.timeoutMs], ["seed", c.seed]]) {
    if (!Number.isInteger(n) || n <= 0) throw new Error(`${name} must be a positive integer`);
  }
  c.baseUrl = c.baseUrl.replace(/\/+$/, "");
  c.apiKey = process.env.VLLM_API_KEY;
  if (!c.apiKey) throw new Error("VLLM_API_KEY is required");
  return c;
}

const nowMs = () => Number(process.hrtime.bigint()) / 1e6;
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const percentile = (xs, p) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.ceil(p * s.length) - 1))];
};
const round = (n, d = 3) => Number.isFinite(n) ? +n.toFixed(d) : null;

const WORDS = (
  "system kernel memory buffer thread process socket packet register cache " +
  "pointer allocate schedule interrupt virtual physical address translate " +
  "compile execute branch predict pipeline vector matrix tensor gradient " +
  "cluster network storage device driver module segment offset boundary " +
  "harbor lantern meadow copper violin orchard glacier saffron thimble walnut"
).split(" ");

function prefillPrompt(approxTokens, seed) {
  let state = seed >>> 0;
  const words = [];
  const count = Math.max(8, Math.floor(approxTokens / 1.3));
  for (let i = 0; i < count; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    words.push(WORDS[state % WORDS.length]);
  }
  return `[run ${seed}] Notes: ${words.join(" ")}\nWrite a short summary of the notes above.`;
}

const COUNTING_PROMPT =
  "Output the integers from 1 through 5000 in ascending order, separated by single spaces. " +
  "Output numbers only, without any explanation or punctuation. Begin exactly as follows:\n" +
  "1 2 3 4 5 6 7 8 9 10";
const DECODE_PROMPT = "Write a 500-word article.";

async function discoverModel(c) {
  if (c.model) return c.model;
  const r = await fetch(`${c.baseUrl}/v1/models`, {
    headers: { Authorization: `Bearer ${c.apiKey}` },
    signal: AbortSignal.timeout(c.timeoutMs),
  });
  if (!r.ok) throw new Error(`/v1/models returned ${r.status}: ${await r.text()}`);
  const model = (await r.json())?.data?.[0]?.id;
  if (!model) throw new Error("No model returned by /v1/models");
  return model;
}

async function oneRequest(c, model, concurrency, index) {
  const prompt = c.mode === "prefill"
    ? prefillPrompt(c.inputTokens, c.seed + concurrency * 1000 + index)
    : c.mode === "counting" ? COUNTING_PROMPT : DECODE_PROMPT;
  const endpoint = c.mode === "decode" ? "/v1/chat/completions" : "/v1/completions";
  const body = {
    model,
    max_tokens: c.maxTokens,
    temperature: 0,
    top_p: 1,
    ignore_eos: true,
    cache_salt: randomUUID(),
    return_token_ids: true,
    stream: true,
    stream_options: { include_usage: true },
    ...(endpoint.includes("chat")
      ? { messages: [{ role: "user", content: prompt }] }
      : { prompt }),
  };
  const startMs = nowMs();
  const response = await fetch(`${c.baseUrl}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${c.apiKey}` },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(c.timeoutMs),
  });
  if (!response.ok) throw new Error(`request ${index} returned ${response.status}: ${await response.text()}`);

  let buffer = "", firstMs = null, lastMs = null, usage = null;
  const events = [];
  const decoder = new TextDecoder();
  const consume = raw => {
    for (const line of raw.split(/\r?\n/)) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      const chunk = JSON.parse(data);
      if (chunk.usage) usage = chunk.usage;
      for (const choice of chunk.choices || []) {
        const d = choice.delta || choice;
        const visible = [d.content, d.reasoning_content, d.reasoning, choice.text]
          .some(v => typeof v === "string" && v.length > 0);
        const ids = Array.isArray(choice.token_ids) ? choice.token_ids :
          Array.isArray(d.token_ids) ? d.token_ids : null;
        if (visible || (ids && ids.length)) {
          const t = nowMs(); firstMs ??= t; lastMs = t;
          events.push({ timestampMs: t, tokenCount: ids?.length ?? 1, exact: ids !== null });
        }
      }
    }
  };
  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk, { stream: true });
    const records = buffer.split(/\r?\n\r?\n/); buffer = records.pop() || "";
    for (const record of records) consume(record);
  }
  buffer += decoder.decode(); if (buffer.trim()) consume(buffer);
  const endMs = nowMs();
  if (!usage) throw new Error(`request ${index}: missing usage`);
  if (usage.completion_tokens !== c.maxTokens) {
    throw new Error(`request ${index}: completion_tokens=${usage.completion_tokens}, expected ${c.maxTokens}`);
  }
  const cached = usage.prompt_tokens_details?.cached_tokens ?? null;
  if (cached !== 0) throw new Error(`request ${index}: cached_tokens=${cached}, expected 0`);
  firstMs ??= endMs; lastMs ??= firstMs;
  const decodeSeconds = Math.max(0, (lastMs - firstMs) / 1000);
  return {
    id: index,
    prompt_tokens: usage.prompt_tokens,
    completion_tokens: usage.completion_tokens,
    cached_tokens: cached,
    start_ms: startMs,
    first_ms: firstMs,
    last_ms: lastMs,
    end_ms: endMs,
    ttft_s: (firstMs - startMs) / 1000,
    e2e_s: (endMs - startMs) / 1000,
    decode_s: decodeSeconds,
    decode_tok_s: decodeSeconds > 0 ? (usage.completion_tokens - 1) / decodeSeconds : null,
    step_s: decodeSeconds > 0 && events.length > 1 ? (events.length - 1) / decodeSeconds : null,
    accepted_tokens_per_step: events.length ? usage.completion_tokens / events.length : null,
    exact_token_ids: events.every(e => e.exact),
    events,
  };
}

function summarize(c, concurrency, results, wallSeconds) {
  const firstStart = Math.min(...results.map(r => r.start_ms));
  const lastEnd = Math.max(...results.map(r => r.end_ms));
  const activeStart = Math.max(...results.map(r => r.first_ms));
  const activeEnd = Math.min(...results.map(r => r.last_ms));
  const activeSeconds = Math.max(0, (activeEnd - activeStart) / 1000);
  let activeTokens = 0, activeEvents = 0;
  for (const r of results) for (const e of r.events) {
    if (e.timestampMs > activeStart && e.timestampMs <= activeEnd) {
      activeTokens += e.tokenCount; activeEvents++;
    }
  }
  const promptTotal = results.reduce((s, r) => s + r.prompt_tokens, 0);
  const completionTotal = results.reduce((s, r) => s + r.completion_tokens, 0);
  const ttfts = results.map(r => r.ttft_s);
  const rates = results.map(r => r.decode_tok_s).filter(Number.isFinite);
  const steps = results.map(r => r.step_s).filter(Number.isFinite);
  const accepted = results.map(r => r.accepted_tokens_per_step).filter(Number.isFinite);
  return {
    timestamp: new Date().toISOString(), mode: c.mode, concurrency,
    input_tokens_setting: c.inputTokens, max_tokens_per_request: c.maxTokens,
    requests: results.length,
    prompt_tokens_each: results.map(r => r.prompt_tokens),
    cached_tokens_each: results.map(r => r.cached_tokens),
    completion_tokens_total: completionTotal,
    ttft_avg_s: mean(ttfts), ttft_p50_s: percentile(ttfts, .5), ttft_max_s: Math.max(...ttfts),
    per_request_tok_s: rates.length ? mean(rates) : null,
    per_request_step_s: steps.length ? mean(steps) : null,
    accepted_tokens_per_step: accepted.length ? mean(accepted) : null,
    full_batch_tok_s: activeSeconds > 0 ? activeTokens / activeSeconds : null,
    full_batch_step_s: activeSeconds > 0 ? (activeEvents / concurrency) / activeSeconds : null,
    full_batch_seconds: activeSeconds,
    output_tok_s: completionTotal / ((lastEnd - firstStart) / 1000),
    aggregate_prefill_tok_s: promptTotal / ((Math.max(...results.map(r => r.first_ms)) - firstStart) / 1000),
    wall_s: wallSeconds,
    exact_token_ids: results.every(r => r.exact_token_ids),
    results,
  };
}

async function main() {
  const c = parseArgs(process.argv.slice(2));
  const model = await discoverModel(c);
  mkdirSync(dirname(c.output), { recursive: true });
  for (const concurrency of c.concurrencies) {
    const start = nowMs();
    const results = await Promise.all(Array.from({ length: concurrency }, (_, i) =>
      oneRequest(c, model, concurrency, i + 1)));
    const summary = summarize(c, concurrency, results, (nowMs() - start) / 1000);
    appendFileSync(c.output, JSON.stringify({ ...summary, api_key: undefined }) + "\n");
    const publicLine = {
      mode: c.mode, concurrency,
      prompt_tokens: [...new Set(summary.prompt_tokens_each)],
      cached_tokens: [...new Set(summary.cached_tokens_each)],
      ttft_avg_s: round(summary.ttft_avg_s), ttft_max_s: round(summary.ttft_max_s),
      prefill_tok_s: round(summary.aggregate_prefill_tok_s, 1),
      per_request_tok_s: round(summary.per_request_tok_s, 2),
      per_request_step_s: round(summary.per_request_step_s, 2),
      accepted_tokens_per_step: round(summary.accepted_tokens_per_step, 2),
      full_batch_tok_s: round(summary.full_batch_tok_s, 2),
      full_batch_step_s: round(summary.full_batch_step_s, 2),
      output_tok_s: round(summary.output_tok_s, 2),
      wall_s: round(summary.wall_s), exact_token_ids: summary.exact_token_ids,
    };
    console.log(JSON.stringify(publicLine));
  }
}

main().catch(error => { console.error(error.stack || error.message); process.exit(1); });
