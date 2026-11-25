#!/usr/bin/env bash
# Reproducible 23-point standard benchmark matching the README tables:
#   - Cold prefill (c=1, cached_tokens=0): 64K, 128K, 256K, 512K, 1M
#   - Decode (full_batch_tok_s / step_ms, max_tokens=500):
#       prose, code, counting across c = 1, 2, 4, 8, 16, 32
#
# Usage:
#   VLLM_API_KEY=... bash scripts/run-standard-benchmark.sh
#   VLLM_API_KEY=... VLLM_BASE_URL=http://127.0.0.1:8000 bash scripts/run-standard-benchmark.sh --quick
#
# Optional overrides:
#   VLLM_BASE_URL         (default: http://127.0.0.1:8000)
#   VLLM_MODEL            (default: auto-discovered from /v1/models)
#   PREFILL_LENGTHS       (default: "65536 131072 262144 524288 1048576")
#   DECODE_CONCURRENCIES  (default: "1,2,4,8,16,32")
#   DECODE_MAX_TOKENS     (default: 500)
#   OUTPUT_FILE           (default: benchmark-results.jsonl)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_JS="${SCRIPT_DIR}/benchmark-vllm.mjs"

BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
MODEL="${VLLM_MODEL:-}"
PREFILL_LENGTHS="${PREFILL_LENGTHS:-65536 131072 262144 524288 1048576}"
DECODE_CONCURRENCIES="${DECODE_CONCURRENCIES:-1,2,4,8,16,32}"
DECODE_MAX_TOKENS="${DECODE_MAX_TOKENS:-500}"
OUTPUT_FILE="${OUTPUT_FILE:-benchmark-results.jsonl}"
SKIP_PREFILL=0
SKIP_DECODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      PREFILL_LENGTHS="65536 131072"
      DECODE_CONCURRENCIES="1,8,32"
      shift
      ;;
    --prefill-only)
      SKIP_DECODE=1
      shift
      ;;
    --decode-only)
      SKIP_PREFILL=1
      shift
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,18p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${VLLM_API_KEY:-}" ]]; then
  echo "Error: VLLM_API_KEY must be set in the environment." >&2
  exit 1
fi

COMMON_ARGS=(--base-url "$BASE_URL" --output "$OUTPUT_FILE")
if [[ -n "$MODEL" ]]; then
  COMMON_ARGS+=(--model "$MODEL")
fi

TMP_RUN="$(mktemp)"
trap 'rm -f "$TMP_RUN"' EXIT

echo "=== Standard vLLM SM80 Benchmark ==="
echo "  base_url:     $BASE_URL"
echo "  model:        ${MODEL:-<auto-discover>}"
echo "  output_file:  $OUTPUT_FILE"
echo "  prefill:      $([[ $SKIP_PREFILL -eq 1 ]] && echo 'skipped' || echo "$PREFILL_LENGTHS")"
echo "  decode:       $([[ $SKIP_DECODE -eq 1 ]] && echo 'skipped' || echo "prose,code,counting c=${DECODE_CONCURRENCIES} max_tokens=${DECODE_MAX_TOKENS}")"
echo

if [[ $SKIP_PREFILL -eq 0 && $SKIP_DECODE -eq 0 ]]; then
  echo ">>> Warmup (decode + subwidth + 4K prefill JIT)..."
  WARM_OUT="$(mktemp)"
  node "$BENCH_JS" --mode counting --max-tokens 64 --concurrency 1,32 "${COMMON_ARGS[@]/--output "$OUTPUT_FILE"/--output "$WARM_OUT"}" >/dev/null
  node "$BENCH_JS" --mode prose --max-tokens 128 --concurrency 1,2,4,8 "${COMMON_ARGS[@]/--output "$OUTPUT_FILE"/--output "$WARM_OUT"}" >/dev/null
  node "$BENCH_JS" --mode prefill --input-tokens 4096 --concurrency 1 "${COMMON_ARGS[@]/--output "$OUTPUT_FILE"/--output "$WARM_OUT"}" >/dev/null
  rm -f "$WARM_OUT"
fi

if [[ $SKIP_PREFILL -eq 0 ]]; then
  for LEN in $PREFILL_LENGTHS; do
    echo ">>> Running prefill (${LEN} approx tokens)..."
    node "$BENCH_JS" --mode prefill --input-tokens "$LEN" --concurrency 1 "${COMMON_ARGS[@]}" | tee -a "$TMP_RUN"
  done
fi

if [[ $SKIP_DECODE -eq 0 ]]; then
  for MODE in counting code prose; do
    echo ">>> Running ${MODE} (c=${DECODE_CONCURRENCIES}, max_tokens=${DECODE_MAX_TOKENS})..."
    node "$BENCH_JS" --mode "$MODE" --max-tokens "$DECODE_MAX_TOKENS" --concurrency "$DECODE_CONCURRENCIES" "${COMMON_ARGS[@]}" | tee -a "$TMP_RUN"
  done
fi

echo
echo "=== Markdown Summary ==="
node -e '
const fs = require("fs");
const lines = fs.readFileSync(process.argv[1], "utf8").split("\n").map(s => s.trim()).filter(s => s.startsWith("{"));
const rows = lines.map(s => JSON.parse(s));
const prefills = rows.filter(r => r.mode === "prefill");
if (prefills.length) {
  console.log("\n**Cold Prefill (`tok/s`, `c=1`, `cached_tokens=0`)**\n");
  console.log("| Prompt Tokens | TTFT (s) | Prefill (`tok/s`) |");
  console.log("|---:|---:|---:|");
  for (const r of prefills) {
    console.log(`| ${r.prompt_tokens[0].toLocaleString("en-US")} | ${r.ttft_avg_s.toFixed(3)} | **${r.prefill_tok_s.toFixed(1)}** |`);
  }
}
const modes = ["prose", "code", "counting"];
const concs = [...new Set(rows.filter(r => modes.includes(r.mode)).map(r => r.concurrency))].sort((a, b) => a - b);
if (concs.length) {
  console.log("\n**Decode Throughput & Step Latency (`full_batch_tok_s` / `step_ms`)**\n");
  console.log("| Concurrency `c` | `prose` | `code` | `counting` |");
  console.log("|---:|---:|---:|---:|");
  const fmt = (r) => {
    if (!r) return "—";
    const stepMs = r.full_batch_step_s ? (1000 / r.full_batch_step_s).toFixed(2) : "—";
    return `**${r.full_batch_tok_s.toFixed(1)}** (\`${stepMs} ms\`, acc ${r.accepted_tokens_per_step.toFixed(2)})`;
  };
  for (const c of concs) {
    const cells = modes.map(m => fmt(rows.find(r => r.mode === m && r.concurrency === c)));
    console.log(`| **${c}** | ${cells.join(" | ")} |`);
  }
}
' "$TMP_RUN"
