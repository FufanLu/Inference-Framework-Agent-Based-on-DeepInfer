#!/usr/bin/env bash
# The burst experiment of docs/illustrated/09: steady interactive traffic plus
# a burst of 2000 requests, sent directly or as jobs under several runner
# limits. Expects a running stack (scripts/dev.sh unified 4).
#
#   scripts/burst_sweep.sh http://127.0.0.1:8080 bench/results/burst/memory
set -euo pipefail
GATEWAY="${1:-http://127.0.0.1:8080}"
OUT="${2:-bench/results/burst/latest}"
BURST="${BURST:-2000}"
BIN=target/release/lev-burst
mkdir -p "$OUT"

run() {
    local name="$1"
    shift
    echo "== $name"
    "$BIN" --gateway "$GATEWAY" --burst-requests "$BURST" --burst-at 8 --out "$OUT/$name" "$@" >/dev/null
    python3 - "$OUT/$name/summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
b, i = d["burst"], d["interactive"]["during_burst"]
print(f"   burst ok {b['ok']}/{b['requests']} makespan {b['makespan_s']} s | "
      f"interactive TTFT p50 {i['ttft_p50_ms']} p99 {i['ttft_p99_ms']} ms (n={i['n']})")
PY
}

run none --mode none --duration 30
run direct --mode direct --duration 60
for cfg in "32 50" "64 100" "128 200" "256 400" "512 2000"; do
    set -- $cfg
    run "jobs-$1-$2" --mode jobs --duration 60 --max-in-flight "$1" --max-queue-delay-ms "$2" --job-priority 1
done
# Ablation: the same limits without engine priority.
run "jobs-512-2000-p0" --mode jobs --duration 60 --max-in-flight 512 --max-queue-delay-ms 2000 --job-priority 0
run "jobs-128-200-p0" --mode jobs --duration 60 --max-in-flight 128 --max-queue-delay-ms 200 --job-priority 0
# Back to the defaults.
curl -fsS -X PUT "$GATEWAY/admin/jobs" -H 'content-type: application/json' \
    -d '{"max_in_flight": 32, "max_queue_delay_ms": 50, "priority": 1}' >/dev/null
