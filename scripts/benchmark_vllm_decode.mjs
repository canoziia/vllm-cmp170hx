#!/usr/bin/env node

/**
 * Benchmark an OpenAI-compatible vLLM server with concurrent streaming requests.
 * Requires Node.js 18+; no third-party packages are needed.
 *
 * Examples:
 *   node benchmark_vllm_decode.mjs
 *   node benchmark_vllm_decode.mjs --concurrency 8 --max-tokens 1024
 *   VLLM_API_KEY=xxx node benchmark_vllm_decode.mjs --base-url http://host:8000
 */

import { readFileSync, writeFileSync } from "node:fs";
import { randomUUID } from "node:crypto";

const defaults = {
  baseUrl: process.env.VLLM_BASE_URL || "http://192.168.2.16:8000",
  apiKey: process.env.VLLM_API_KEY || "canoziia",
  model: process.env.VLLM_MODEL || "",
  concurrency: 8,
  maxTokens: 1024,
  prompt: "Write a 500-word article.",
  temperature: 0,
  topP: 1,
  websiteMode: false,
  inputTokens: 16,
  warmupTokens: 8,
  ignoreEos: false,
  timeoutMs: 10 * 60 * 1000,
};

function usage() {
  console.log(`Usage: node benchmark_vllm_decode.mjs [options]

Options:
  --base-url URL       vLLM base URL (default: ${defaults.baseUrl})
  --api-key TOKEN      API token (default: VLLM_API_KEY or configured token)
  --model MODEL        Model ID (default: discover from /v1/models)
  -c, --concurrency N  Concurrent requests (default: ${defaults.concurrency})
  --max-tokens N       Maximum completion tokens/request (default: ${defaults.maxTokens})
  --prompt TEXT        User prompt (default: ${JSON.stringify(defaults.prompt)})
  --prompt-file PATH   Read a fixed UTF-8 user prompt (not website mode)
  --uncached           Unique cache salt per request; report cached-token count
  --seed N             Reproducible prompt generation and server sampling seed
  --json-output PATH   Save measurements and request timings as JSON
  --temperature N      Sampling temperature (default: ${defaults.temperature})
  --top-p N            Top-p sampling value (default: ${defaults.topP})
  --website-mode       Match gengchaogit/llm_speedtest request style
  --input-tokens N     Website-mode input setting (default: ${defaults.inputTokens})
  --no-warmup          Skip the short warm-up request
  --ignore-eos         Ignore EOS and always generate max-tokens (saturation test)
  --timeout-ms N       Timeout per request (default: ${defaults.timeoutMs})
  -h, --help           Show this help

Environment: VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL`);
}

function parseArgs(argv) {
  const config = { ...defaults };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const take = () => {
      if (++i >= argv.length) throw new Error(`Missing value after ${arg}`);
      return argv[i];
    };
    if (arg === "--base-url") config.baseUrl = take();
    else if (arg === "--api-key") config.apiKey = take();
    else if (arg === "--model") config.model = take();
    else if (arg === "-c" || arg === "--concurrency") config.concurrency = Number(take());
    else if (arg === "--max-tokens") config.maxTokens = Number(take());
    else if (arg === "--prompt") config.prompt = take();
    else if (arg === "--prompt-file") config.promptFile = take();
    else if (arg === "--uncached") config.uncached = true;
    else if (arg === "--seed") config.seed = Number(take());
    else if (arg === "--json-output") config.jsonOutput = take();
    else if (arg === "--temperature") config.temperature = Number(take());
    else if (arg === "--top-p") config.topP = Number(take());
    else if (arg === "--website-mode") config.websiteMode = true;
    else if (arg === "--input-tokens") config.inputTokens = Number(take());
    else if (arg === "--timeout-ms") config.timeoutMs = Number(take());
    else if (arg === "--no-warmup") config.warmupTokens = 0;
    else if (arg === "--ignore-eos") config.ignoreEos = true;
    else if (arg === "-h" || arg === "--help") {
      usage();
      process.exit(0);
    } else throw new Error(`Unknown option: ${arg}`);
  }

  for (const [name, value] of [
    ["concurrency", config.concurrency],
    ["max-tokens", config.maxTokens],
    ["input-tokens", config.inputTokens],
    ["timeout-ms", config.timeoutMs],
  ]) {
    if (!Number.isFinite(value) || value <= 0 || !Number.isInteger(value)) {
      throw new Error(`${name} must be a positive integer`);
    }
  }
  if (!Number.isFinite(config.temperature) || config.temperature < 0) {
    throw new Error("temperature must be a non-negative number");
  }
  if (!Number.isFinite(config.topP) || config.topP < 0 || config.topP > 1) {
    throw new Error("top-p must be between 0 and 1");
  }
  if (config.seed !== undefined && (!Number.isInteger(config.seed) || config.seed < 0)) {
    throw new Error("seed must be a non-negative integer");
  }
  // Defaults used by the website at the time of inspection.
  if (config.websiteMode && !argv.includes("--temperature")) config.temperature = 1;
  if (config.websiteMode && !argv.includes("--top-p")) config.topP = 0.1;
  if (config.promptFile) {
    if (config.websiteMode) throw new Error("--prompt-file cannot be used with --website-mode");
    config.prompt = readFileSync(config.promptFile, "utf8");
  }
  config.baseUrl = config.baseUrl.replace(/\/+$/, "");
  return config;
}

const nowMs = () => Number(process.hrtime.bigint()) / 1e6;
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const percentile = (xs, p) => {
  const sorted = [...xs].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[index];
};
const fmt = (n, digits = 2) => Number.isFinite(n) ? n.toFixed(digits) : "n/a";

function headers(config) {
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${config.apiKey}`,
  };
}

async function discoverModel(config) {
  if (config.model) return config.model;
  const response = await fetch(`${config.baseUrl}/v1/models`, {
    headers: headers(config),
    signal: AbortSignal.timeout(config.timeoutMs),
  });
  if (!response.ok) throw new Error(`/v1/models returned ${response.status}: ${await response.text()}`);
  const body = await response.json();
  const model = body?.data?.[0]?.id;
  if (!model) throw new Error("No model was returned by /v1/models");
  return model;
}

const WEBSITE_SYSTEM_PROMPT = "You are a helpful assistant.";
const WEBSITE_TOKEN_WORDS = [
  "a", "the", "and", "of", "to", "in", "is", "it", "that", "for", "with", "as",
  "on", "by", "from", "this", "be", "are", "or", "not", "we", "you", "they", "can",
  "will", "if", "all", "one", "time", "world", "life", "work", "data", "model", "token",
  "text", "idea", "mind", "story", "light", "space", "future", "human", "system", "simple",
  "clear", "reason", "change", "value", "truth",
];

function makeWebsitePrompt(tokenCount, seed, baseSeed = Date.now()) {
  // The website treats each selected short English word as one input token and
  // varies the prompt per concurrent request to avoid prefix-cache hits.
  let state = ((baseSeed ^ ((seed + 1) * 1000003)) >>> 0);
  const words = [];
  for (let i = 0; i < tokenCount; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    words.push(WEBSITE_TOKEN_WORDS[Math.floor((state / 4294967296) * WEBSITE_TOKEN_WORDS.length)]);
  }
  return words.join(" ");
}

async function streamCompletion(config, model, id, maxTokens = config.maxTokens, prompt = config.prompt) {
  const messages = config.websiteMode
    ? [
        { role: "system", content: WEBSITE_SYSTEM_PROMPT },
        { role: "user", content: prompt },
      ]
    : [{ role: "user", content: prompt }];
  const startMs = nowMs();
  const response = await fetch(`${config.baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: headers(config),
    body: JSON.stringify({
      model,
      messages,
      max_tokens: maxTokens,
      temperature: config.temperature,
      ...(config.seed !== undefined ? { seed: config.seed } : {}),
      top_p: config.topP,
      presence_penalty: 0,
      frequency_penalty: 0,
      ...(config.ignoreEos ? { ignore_eos: true } : {}),
      ...(config.uncached ? { cache_salt: randomUUID() } : {}),
      return_token_ids: true,
      stream: true,
      stream_options: { include_usage: true },
    }),
    signal: AbortSignal.timeout(config.timeoutMs),
  });

  if (!response.ok) throw new Error(`request ${id} returned ${response.status}: ${await response.text()}`);
  if (!response.body) throw new Error(`request ${id} returned no response body`);

  let firstTokenMs = null;
  let lastTokenMs = null;
  const tokenEvents = [];
  let hasExactTokenIds = false;
  let usageData = null;
  let finishReason = null;
  let buffer = "";
  const decoder = new TextDecoder();

  const handleEvent = (event) => {
    const dataLines = event
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim());
    for (const data of dataLines) {
      if (!data || data === "[DONE]") continue;
      const chunk = JSON.parse(data);
      if (chunk.usage) usageData = chunk.usage;
      for (const choice of chunk.choices || []) {
        if (choice.finish_reason) finishReason = choice.finish_reason;
        const delta = choice.delta || {};
        // Count timing for visible output and reasoning output. Token counts come from
        // the server's final usage object, not from text length or chunk count.
        const hasTokenData = [delta.content, delta.reasoning_content, delta.reasoning]
          .some((value) => typeof value === "string" && value.length > 0);
        if (hasTokenData) {
          const t = nowMs();
          if (firstTokenMs === null) firstTokenMs = t;
          lastTokenMs = t;
          // With MTP, one delta can contain multiple accepted tokens. vLLM's
          // return_token_ids extension exposes the exact number for each chunk.
          const exactTokenCount = Array.isArray(choice.token_ids) ? choice.token_ids.length : null;
          if (exactTokenCount !== null) hasExactTokenIds = true;
          tokenEvents.push({ timestampMs: t, tokenCount: exactTokenCount ?? 1 });
        }
      }
    }
  };

  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk, { stream: true });
    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop() || "";
    for (const event of events) handleEvent(event);
  }
  buffer += decoder.decode();
  if (buffer.trim()) handleEvent(buffer);
  const endMs = nowMs();

  if (!usageData || !Number.isFinite(usageData.completion_tokens)) {
    throw new Error(`request ${id} did not return token usage; ensure vLLM supports stream_options.include_usage`);
  }
  if (firstTokenMs === null) {
    // This should only happen for an empty completion. Keep E2E metrics valid.
    firstTokenMs = endMs;
    lastTokenMs = endMs;
  }

  const completionTokens = usageData.completion_tokens;
  const decodeSeconds = Math.max(0, (lastTokenMs - firstTokenMs) / 1000);
  return {
    id,
    promptTokens: usageData.prompt_tokens,
    cachedTokens: usageData.prompt_tokens_details?.cached_tokens ?? null,
    completionTokens,
    totalTokens: usageData.total_tokens,
    finishReason,
    startMs,
    firstTokenMs,
    lastTokenMs,
    endMs,
    tokenEvents,
    hasExactTokenIds,
    ttftSeconds: (firstTokenMs - startMs) / 1000,
    e2eSeconds: (endMs - startMs) / 1000,
    // The first streamed token is produced at TTFT, so exclude it from decode rate.
    decodeSeconds,
    decodeTokensPerSecond: decodeSeconds > 0 ? Math.max(0, completionTokens - 1) / decodeSeconds : NaN,
  };
}

async function main() {
  const config = parseArgs(process.argv.slice(2));
  const model = await discoverModel(config);
  console.log(`Server:      ${config.baseUrl}`);
  console.log(`Model:       ${model}`);
  console.log(`Concurrency: ${config.concurrency}`);
  console.log(`Mode:        ${config.websiteMode ? `llm_speedtest-compatible (${config.inputTokens} input setting)` : "standard"}`);
  if (!config.websiteMode) console.log(`Prompt:      ${config.promptFile || JSON.stringify(config.prompt)}`);
  console.log(`Sampling:    temperature=${config.temperature}, top_p=${config.topP}`);
  console.log(`Max tokens:  ${config.maxTokens}/request`);
  console.log(`Ignore EOS:  ${config.ignoreEos ? "yes (saturation test)" : "no"}`);

  if (config.warmupTokens > 0) {
    process.stdout.write(`Warm-up:     ${config.warmupTokens} tokens ... `);
    await streamCompletion(
      config,
      model,
      "warmup",
      config.warmupTokens,
      config.websiteMode ? makeWebsitePrompt(96, 999999, config.seed) : "Hi",
    );
    console.log("done");
  }

  process.stdout.write("Benchmark:   running ...\n");
  const batchStartMs = nowMs();
  const results = await Promise.all(
    Array.from({ length: config.concurrency }, (_, i) => streamCompletion(
      config,
      model,
      i + 1,
      config.maxTokens,
      config.websiteMode ? makeWebsitePrompt(config.inputTokens, i + 1, config.seed) : config.prompt,
    )),
  );
  const batchEndMs = nowMs();

  const wallSeconds = (batchEndMs - batchStartMs) / 1000;
  const totalPromptTokens = results.reduce((sum, r) => sum + r.promptTokens, 0);
  const totalCompletionTokens = results.reduce((sum, r) => sum + r.completionTokens, 0);
  const earliestRequestMs = Math.min(...results.map((r) => r.startMs));
  const latestRequestEndMs = Math.max(...results.map((r) => r.endMs));
  const makespanSeconds = (latestRequestEndMs - earliestRequestMs) / 1000;
  const outputTokenThroughput = makespanSeconds > 0
    ? totalCompletionTokens / makespanSeconds
    : NaN;
  const latestFirstTokenMs = Math.max(...results.map((r) => r.firstTokenMs));
  const aggregatePrefillSeconds = (latestFirstTokenMs - earliestRequestMs) / 1000;
  const aggregatePrefillRate = aggregatePrefillSeconds > 0
    ? totalPromptTokens / aggregatePrefillSeconds
    : NaN;
  const firstDecodeMs = Math.min(...results.map((r) => r.firstTokenMs));
  const lastDecodeMs = Math.max(...results.map((r) => r.lastTokenMs));
  const aggregateDecodeSeconds = (lastDecodeMs - firstDecodeMs) / 1000;
  // Exclude one TTFT token per request, matching the per-request decode-rate convention.
  const aggregateDecodeTokens = Math.max(0, totalCompletionTokens - results.length);
  const aggregateDecodeRate = aggregateDecodeSeconds > 0
    ? aggregateDecodeTokens / aggregateDecodeSeconds
    : NaN;
  // The website counts all completion tokens and measures through the final SSE
  // event, rather than subtracting one TTFT token per request.
  const websiteDecodeSeconds = (Math.max(...results.map((r) => r.endMs)) - firstDecodeMs) / 1000;
  const websiteDecodeRate = websiteDecodeSeconds > 0
    ? totalCompletionTokens / websiteDecodeSeconds
    : NaN;

  // Ideal/all-active decode throughput: use only the intersection in which every
  // request has started producing tokens and no request has finished yet.
  const allActiveStartMs = Math.max(...results.map((r) => r.firstTokenMs));
  const allActiveEndMs = Math.min(...results.map((r) => r.lastTokenMs));
  const allActiveSeconds = Math.max(0, (allActiveEndMs - allActiveStartMs) / 1000);
  const allActiveTokensByRequest = results.map((r) => {
    const streamedTokens = r.tokenEvents.reduce((sum, event) => sum + event.tokenCount, 0);
    if (streamedTokens === 0) return 0;
    // Exclude the chunk exactly at the opening boundary (it defines TTFT), and
    // include chunks through the closing boundary. If token_ids are unavailable,
    // scale one-event counts to the authoritative final usage count.
    const tokensInWindow = r.tokenEvents
      .filter((event) => event.timestampMs > allActiveStartMs && event.timestampMs <= allActiveEndMs)
      .reduce((sum, event) => sum + event.tokenCount, 0);
    return r.hasExactTokenIds
      ? tokensInWindow
      : tokensInWindow * (r.completionTokens / streamedTokens);
  });
  const allActiveTokens = allActiveTokensByRequest.reduce((sum, n) => sum + n, 0);
  const idealDecodeRate = allActiveSeconds > 0 ? allActiveTokens / allActiveSeconds : NaN;
  const totalStreamedTokens = results.reduce(
    (sum, r) => sum + r.tokenEvents.reduce((eventSum, event) => eventSum + event.tokenCount, 0),
    0,
  );
  const tokenEventCoverage = totalCompletionTokens > 0 ? totalStreamedTokens / totalCompletionTokens : NaN;
  const exactTokenTiming = results.every((r) => r.hasExactTokenIds);
  const perRequestRates = results.map((r) => r.decodeTokensPerSecond).filter(Number.isFinite);
  const ttfts = results.map((r) => r.ttftSeconds);

  console.log("\nPer-request results:");
  console.log(" ID  prompt  output  TTFT(s)  decode(s)  decode tok/s  E2E(s)  finish");
  for (const r of results) {
    console.log(
      `${String(r.id).padStart(3)}  ${String(r.promptTokens).padStart(6)}  ` +
      `${String(r.completionTokens).padStart(6)}  ${fmt(r.ttftSeconds).padStart(7)}  ` +
      `${fmt(r.decodeSeconds).padStart(9)}  ${fmt(r.decodeTokensPerSecond).padStart(12)}  ` +
      `${fmt(r.e2eSeconds).padStart(6)}  ${r.finishReason || "unknown"}`,
    );
  }

  console.log("\nCache counts per request:", results.map(r => r.cachedTokens));
  if (config.uncached && results.some(r => r.cachedTokens !== 0)) {
    throw new Error("Uncached test requires confirmed cached_tokens=0 for every request");
  }
  console.log("\nSummary:");
  console.log(`  Wall time:                         ${fmt(wallSeconds)} s`);
  console.log(`  Prompt tokens (total):             ${totalPromptTokens}`);
  console.log(`  Aggregate Prefill throughput:       ${fmt(aggregatePrefillRate)} tok/s`);
  console.log(`  Prefill window (to latest TTFT):    ${fmt(aggregatePrefillSeconds)} s`);
  console.log(`  Completion tokens (total):         ${totalCompletionTokens}`);
  console.log(`  Output Token Throughput:            ${fmt(outputTokenThroughput)} tok/s`);
  console.log(`  Full-Batch Decode Throughput:       ${fmt(idealDecodeRate)} tok/s`);
  console.log(`  Makespan:                           ${fmt(makespanSeconds)} s`);
  console.log(`  Full-batch window/tokens:           ${fmt(allActiveSeconds)} s / ${fmt(allActiveTokens, 1)}`);
  console.log(`  Timed token/usage coverage:         ${fmt(tokenEventCoverage * 100)}% (${exactTokenTiming ? "exact token_ids" : "estimated"})`);
  console.log(`  Per-request decode rate avg/p50:    ${fmt(mean(perRequestRates))} / ${fmt(percentile(perRequestRates, 0.5))} tok/s`);
  console.log(`  TTFT avg/p50/p95:                   ${fmt(mean(ttfts))} / ${fmt(percentile(ttfts, 0.5))} / ${fmt(percentile(ttfts, 0.95))} s`);
  if (config.jsonOutput) {
    const { apiKey, prompt, ...publicConfig } = config;
    writeFileSync(config.jsonOutput, JSON.stringify({
      config: publicConfig, model, results, outputTokenThroughput,
      aggregatePrefillRate, idealDecodeRate, exactTokenTiming, tokenEventCoverage,
      wallSeconds,
    }, null, 2));
  }
  console.log("\nDefinitions:");
  console.log("  Output Token Throughput = all output tokens / (latest request end - earliest request start).");
  console.log("                            Includes TTFT and tail-drain time (makespan-based).");
  console.log("  Full-Batch Decode Throughput = output tokens emitted while every request is decoding /");
  console.log("                                 (earliest last-token - latest first-token).");
}

main().catch((error) => {
  console.error(`Error: ${error.stack || error.message}`);
  process.exit(1);
});
