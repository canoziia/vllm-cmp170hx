#!/usr/bin/env node
/** Verify that generated output becomes reusable history on the next turn. */
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";

const baseUrl = (process.env.VLLM_BASE_URL || "http://127.0.0.1:8001").replace(/\/+$/, "");
const model = process.env.VLLM_MODEL || "nvidia/Qwen3.8-Flash-Next-NVFP4";
const apiKey = process.env.VLLM_API_KEY;
assert(apiKey, "VLLM_API_KEY is required");

async function post(body) {
  const response = await fetch(`${baseUrl}/v1/responses`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({ model, ...body }),
    signal: AbortSignal.timeout(10 * 60 * 1000),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(JSON.stringify(result));
  return result;
}

const input = [{
  role: "user",
  content: Array.from(
    { length: 100 },
    (_, i) => `Record ${i}: validate checksums, commit immutable transactions, and recover safely.`,
  ).join("\n") + "\nDescribe transaction recovery in exhaustive detail until the output limit.",
}];
const salt = randomUUID();
const first = await post({
  input,
  max_output_tokens: 1024,
  temperature: 0,
  cache_salt: salt,
  store: false,
});
assert(first.output.length > 0, "first turn returned no reusable output items");
input.push(...first.output, { role: "user", content: "Summarize the preceding answer." });
const second = await post({
  input,
  max_output_tokens: 32,
  temperature: 0,
  cache_salt: salt,
  store: false,
});
const sharedHistory = first.usage.input_tokens + first.usage.output_tokens;
const cachedTokens = second.usage.input_tokens_details?.cached_tokens ?? 0;
const sharedRecompute = sharedHistory - cachedTokens;
const summary = {
  first: first.usage,
  second: second.usage,
  shared_history_tokens: sharedHistory,
  cached_tokens: cachedTokens,
  shared_recompute_tokens: sharedRecompute,
};
console.log(JSON.stringify(summary));
assert(cachedTokens > 0, "generated history produced no cache hit");
assert(
  sharedRecompute <= 160,
  `shared recomputation ${sharedRecompute} exceeds 128-token checkpoint + 32-token MTP margin`,
);
console.log("MULTITURN_GENERATED_HISTORY_CACHE PASS");
