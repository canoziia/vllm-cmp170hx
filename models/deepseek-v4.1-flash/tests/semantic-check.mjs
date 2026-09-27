#!/usr/bin/env node
/**
 * Semantic correctness fixture under a binding cohort cap.
 *
 * Exact token equality is meaningless on this deployment (greedy output already
 * differs between two identical c32 runs), so check content correctness on
 * prompts with an objectively verifiable answer, repeated to fill the
 * concurrency so the per-step decode cap actually binds.
 *
 * usage: semantic-check.mjs <label> [--concurrency N] [--each N]
 */
import { randomUUID } from "node:crypto";

const arg = (k, d) => { const i = process.argv.indexOf(k); return i === -1 ? d : process.argv[i + 1]; };
const label = process.argv[2] || "run";
const baseUrl = arg("--base-url", "http://127.0.0.1:18000");
const concurrency = Number(arg("--concurrency", 32));
const each = Number(arg("--each", 5));
const maxTokens = Number(arg("--max-tokens", 400));
const apiKey = process.env.VLLM_API_KEY || "";

// prompt -> predicate over the decoded answer
const CASES = [
  ["Count from 1 to 60 separated by commas, nothing else.", t => /(^|\D)58\D+59\D+60\b/.test(t.replace(/\s/g, " ")) && /(^|\D)1[,. ]/.test(t)],
  ["What is 17 times 23? Answer with just the number.", t => /391/.test(t) && !/39[02-9]|38\d/.test(t)],
  ["Recite the lowercase English alphabet backwards with no spaces.", t => /zyxwvutsrqponmlkjihgfedcba/.test(t.toLowerCase())],
  ["List the eight planets in order from the Sun, comma separated.", t => /mercury/i.test(t) && /venus/i.test(t) && /neptune/i.test(t) && /mars/i.test(t)],
  ["Convert 2048 bytes to KiB. Answer with just the number and unit.", t => /\b2\s*KiB/i.test(t)],
  ["Write the next four terms of the sequence: 2, 6, 18, 54,", t => /162/.test(t) && /486/.test(t) && /1458/.test(t)],
];

const models = await (await fetch(`${baseUrl}/v1/models`, { headers: { Authorization: `Bearer ${apiKey}` } })).json();
const model = models.data[0].id;

const one = async ([prompt]) => {
  const res = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({
      model, messages: [{ role: "user", content: prompt }], max_tokens: maxTokens,
      temperature: 0, top_p: 1, cache_salt: randomUUID(), stream: true,
    }),
    signal: AbortSignal.timeout(120000),
  });
  if (!res.ok) throw new Error(`${res.status}`);
  let text = "", buf = ""; const dec = new TextDecoder();
  for await (const ch of res.body) {
    buf += dec.decode(ch, { stream: true });
    const lines = buf.split(/\r?\n/); buf = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const d = line.slice(5).trim(); if (!d || d === "[DONE]") continue;
      const c = JSON.parse(d);
      for (const choice of c.choices || []) {
        const d = choice.delta ?? {};
        // the model answers inside the reasoning block first; the harness
        // counts reasoning_content as output too, so do the same here
        text += d.content ?? d.reasoning_content ?? d.reasoning ?? "";
      }
    }
  }
  return { prompt, text };
};

const jobs = CASES.flatMap(c => Array.from({ length: each }, () => c));
const results = [];
let cursor = 0;
const worker = async () => { while (cursor < jobs.length) { const i = cursor++; results.push({ i, ...(await one(jobs[i])) }); } };
await Promise.all(Array.from({ length: Math.min(concurrency, jobs.length) }, worker));
results.sort((a, b) => a.i - b.i);

console.log(`semantic check ${label}: model=${model} concurrency=${concurrency} max_tokens=${maxTokens} requests=${results.length}`);
let pass = 0, total = 0;
for (const [prompt, pred] of CASES) {
  const rows = results.filter(r => r.prompt === prompt);
  const ok = rows.filter(r => pred(r.text)).length;
  pass += ok; total += rows.length;
  const bad = rows.find(r => !pred(r.text));
  console.log(`  ${ok === rows.length ? "PASS" : "FAIL"}  ${ok}/${rows.length}  ${prompt.slice(0, 52)}` +
    (bad ? `\n        unexpected: ${JSON.stringify(bad.text.slice(0, 90))}` : ""));
}
console.log(`SEMANTIC_${pass === total ? "PASS" : "FAIL"} ${pass}/${total} under concurrency ${concurrency}`);
process.exit(pass === total ? 0 : 1);
