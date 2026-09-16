#!/usr/bin/env bash
set -euo pipefail

URL="http://127.0.0.1:8000"
MODEL="/workspace/Llama-3.1-8B-Instruct/"

metric() {
  local name="$1"

  curl -fsS "${URL}/metrics" |
    awk -v name="$name" '
      $1 ~ ("^" name "\\{") {
        print $2
        found = 1
        exit
      }
      END {
        if (!found) print 0
      }
    '
}

# 输出必须是一行，才能被 `read -r ... < <(snapshot)` 正确解析。
snapshot() {
  local local_queries local_hits external_queries external_hits cached_tokens

  local_queries="$(metric "vllm:prefix_cache_queries_total")"
  local_hits="$(metric "vllm:prefix_cache_hits_total")"
  external_queries="$(metric "vllm:external_prefix_cache_queries_total")"
  external_hits="$(metric "vllm:external_prefix_cache_hits_total")"
  cached_tokens="$(metric "vllm:prompt_tokens_cached_total")"

  printf '%s %s %s %s %s\n' \
    "${local_queries}" \
    "${local_hits}" \
    "${external_queries}" \
    "${external_hits}" \
    "${cached_tokens}"
}

run_bench() {
  vllm bench serve \
    --backend openai \
    --base-url "${URL}" \
    --model "${MODEL}" \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 512 \
    --random-range-ratio 0 \
    --num-prompts 200 \
    --max-concurrency 32 \
    --request-rate inf \
    --ignore-eos \
    --seed 42
}

echo "=== Round 1: cold cache; expected to store KV into CPU/disk offload cache ==="
run_bench

echo
echo "Waiting 5 seconds for asynchronous KV stores to finish..."
sleep 5

read -r q1 h1 eq1 eh1 cached1 < <(snapshot)

echo "After round 1:"
echo "  local queries=${q1}, local hits=${h1}"
echo "  external queries=${eq1}, external hits=${eh1}"
echo "  cached prompt tokens=${cached1}"

echo
echo "=== Round 2: same seed and identical prompts ==="
run_bench

echo
echo "Waiting 2 seconds for metrics to settle..."
sleep 2

read -r q2 h2 eq2 eh2 cached2 < <(snapshot)

awk \
  -v q1="$q1" -v h1="$h1" \
  -v eq1="$eq1" -v eh1="$eh1" \
  -v c1="$cached1" \
  -v q2="$q2" -v h2="$h2" \
  -v eq2="$eq2" -v eh2="$eh2" \
  -v c2="$cached2" '
BEGIN {
  local_queries = q2 - q1
  local_hits = h2 - h1
  external_queries = eq2 - eq1
  external_hits = eh2 - eh1
  cached_tokens = c2 - c1

  total_hits = local_hits + external_hits
  expected = 200 * 4096

  printf "\n=== Round 2 delta ===\n"

  printf "Local GPU prefix:       queries=%.0f, hits=%.0f", \
    local_queries, local_hits
  if (local_queries > 0) {
    printf ", rate=%.2f%%\n", 100 * local_hits / local_queries
  } else {
    printf "\n"
  }

  printf "External CPU/disk:      queries=%.0f, hits=%.0f", \
    external_queries, external_hits
  if (external_queries > 0) {
    printf ", rate=%.2f%%\n", 100 * external_hits / external_queries
  } else {
    printf " (no external lookup)\n"
  }

  printf "Cached prompt tokens:   %.0f\n", cached_tokens

  if (local_queries > 0) {
    printf "Total cache hit rate:   %.2f%%\n", \
      100 * total_hits / local_queries
  }

  printf "\nExpected prompt tokens: %d\n", expected

  if (cached_tokens >= expected * 0.99) {
    print "Result: Round 2 is approximately fully cache-hit."
  } else if (cached_tokens > 0) {
    print "Result: Round 2 is partially cache-hit."
  } else {
    print "Result: Round 2 did not reuse prefix KV. Check seed/config/cache state."
  }
}'
