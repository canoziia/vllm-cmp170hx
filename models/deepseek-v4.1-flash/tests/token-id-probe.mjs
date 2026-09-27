#!/usr/bin/env node
/**
 * Strict output-equivalence probe for the PP decode-cohort balance A/B.
 *
 * Same request shape as scripts/benchmark-vllm.mjs (greedy, ignore_eos,
 * return_token_ids, random cache_salt so every run is a cold prefill), but this
 * one KEEPS the full token-ID sequence per request so two arms can be compared
 * token by token.
 *
 * usage: idprobe.mjs <label> [--concurrency N] [--repeats N] [--max-tokens N]
 */
import { randomUUID } from "node:crypto";
import { writeFileSync, mkdirSync } from "node:fs";

const arg = (k, d) => {
  const i = process.argv.indexOf(k);
  return i === -1 ? d : process.argv[i + 1];
};
const label = process.argv[2];
if (!label) { console.error("usage: idprobe.mjs <label> [--concurrency N] ..."); process.exit(2); }

const baseUrl = arg("--base-url", "http://127.0.0.1:18000");
const concurrency = Number(arg("--concurrency", 32));
const repeats = Number(arg("--repeats", 2));
const requests = Number(arg("--requests", concurrency));
const maxTokens = Number(arg("--max-tokens", 128));
const apiKey = process.env.VLLM_API_KEY || "";
const outDir = process.env.OUT_DIR || `./idprobe/${label}`;
mkdirSync(outDir, { recursive: true });

// 32 distinct prompts, so each request has its own expected token sequence.
const GENRES = [
  "Explain why the sky is blue", "Write a haiku about hard drives", "List the planets in order",
  "Count from 1 to 60 separated by commas", "Translate good morning into French and German",
  "Summarize the plot of a heist film", "Write a Python function to parse ISO 8601 dates",
  "What is 17 times 23", "Name three advantages of pipeline parallelism", "Recite the alphabet backwards",
  "Describe the taste of salt", "Write a JSON object describing a book", "Give a recipe for shortbread",
  "Explain gradient descent to a beginner", "List five prime numbers above 100", "What does DNS do",
  "Write a SQL query joining orders and customers", "Compare cats and owls", "Explain what a ring buffer is",
  "Write a short dialogue between two routers", "How do I center a div", "Enumerate the TCP handshake",
  "Describe a winter morning in a fishing village", "Write a bash one-liner to find large files",
  "Explain FP8 quantization briefly", "List the steps to brew pour-over coffee", "What is the capital of Bhutan",
  "Write a regex for a semver string", "Explain KV cache paging", "Give three metaphors for latency",
  "Write a haiku about pipeline bubbles", "Convert 1024 bytes to KiB and MiB",
];

const runOne = async (model, prompt, index) => {
  const body = {
    model, max_tokens: maxTokens, temperature: 0, top_p: 1, ignore_eos: true,
    cache_salt: randomUUID(), return_token_ids: true, stream: true,
    stream_options: { include_usage: true },
    messages: [{ role: "user", content: prompt }],
  };
  const res = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify(body), signal: AbortSignal.timeout(600000),
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  const ids = [];
  let exact = true, usage = null, buf = "";
  const dec = new TextDecoder();
  for await (const chunk of res.body) {
    buf += dec.decode(chunk, { stream: true });
    const lines = buf.split(/\r?\n/); buf = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const d = line.slice(5).trim();
      if (!d || d === "[DONE]") continue;
      const c = JSON.parse(d);
      if (c.usage) usage = c.usage;
      for (const ch of c.choices || []) {
        const src = ch.delta || ch;
        const got = Array.isArray(ch.token_ids) ? ch.token_ids : Array.isArray(src.token_ids) ? src.token_ids : null;
        if (got) ids.push(...got); else if ((src.content ?? "").length) exact = false;
      }
    }
  }
  return { index, prompt, token_ids: ids, exact, completion_tokens: usage?.completion_tokens ?? ids.length };
};

const runPool = async (model, n, workers) => {
  const prompts = Array.from({ length: n }, (_, i) =>
    `${GENRES[i % GENRES.length]} (variant ${i}, answer directly and continue at length)`);
  const results = [];
  let cursor = 0;
  const worker = async () => { while (cursor < prompts.length) { const i = cursor++; results.push(await runOne(model, prompts[i], i)); } };
  await Promise.all(Array.from({ length: Math.min(workers ?? n, prompts.length) }, worker));
  return results.sort((a, b) => a.index - b.index);
};

const models = await (await fetch(`${baseUrl}/v1/models`, { headers: { Authorization: `Bearer ${apiKey}` } })).json();
const model = models.data[0].id;
console.error(`probe ${label}: model=${model} requests=${requests} concurrency=${concurrency} repeats=${repeats} max_tokens=${maxTokens}`);

for (let r = 0; r < repeats; r++) {
  const t0 = Date.now();
  const results = await runPool(model, requests, concurrency);
  const allExact = results.every(x => x.exact && x.token_ids.length > 0);
  writeFileSync(`${outDir}/run${r}.json`, JSON.stringify({
    label, repeat: r, model, requests, concurrency, maxTokens, all_token_ids_exact: allExact,
    wall_s: (Date.now() - t0) / 1000, results,
  }));
  console.error(`  repeat ${r}: requests=${results.length} exact_ids=${allExact ? "yes" : "NO"} tok/req=${(results.reduce((a, x) => a + x.token_ids.length, 0) / results.length).toFixed(1)}`);
}
