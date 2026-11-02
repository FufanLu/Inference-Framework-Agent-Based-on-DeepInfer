#!/usr/bin/env bash
# Run a local Leviathan stack without Docker: sim workers + the gateway.
#
#   scripts/dev.sh                 # 2 prefill + 2 decode sim workers (PD)
#   scripts/dev.sh unified 4       # 4 unified sim workers
#   PROFILE=fast-dev scripts/dev.sh unified 2
#   WORKER_ARGS="--total-blocks 1000" scripts/dev.sh unified 8   # small KV per replica
#
# Gateway: http://127.0.0.1:8080. Logs: .dev/logs/. Ctrl-C stops everything.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-pd}"
COUNT="${2:-2}"
PROFILE="${PROFILE:-h200-vllm}"
PY="${PYTHON:-.venv/bin/python}"
LOGS=.dev/logs
mkdir -p "$LOGS"

make -s proto-py PYTHON="$PY"
cargo build -q --release -p lev-gateway -p lev-bench

pids=()
cleanup() {
    echo "stopping ${#pids[@]} processes"
    kill "${pids[@]}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_worker() {
    local id="$1" role="$2" port="$3"
    (cd worker && exec "../$PY" -m leviathan_worker.server --backend sim --host 127.0.0.1 --port "$port" \
        --worker-id "$id" --role "$role" --profile "$PROFILE" \
        --tokenizer ../assets/tokenizers/v41/tokenizer.json --log-format text ${WORKER_ARGS:-}) >"$LOGS/$id.log" 2>&1 &
    pids+=($!)
    workers+=(--worker "http://127.0.0.1:$port")
}

workers=()
port=50051
case "$MODE" in
pd)
    for i in $(seq 0 $((COUNT - 1))); do start_worker "prefill-$i" prefill $((port++)); done
    for i in $(seq 0 $((COUNT - 1))); do start_worker "decode-$i" decode $((port++)); done
    ;;
unified)
    for i in $(seq 0 $((COUNT - 1))); do start_worker "sim-$i" unified $((port++)); done
    ;;
*)
    echo "usage: $0 [pd|unified] [count]" >&2
    exit 2
    ;;
esac

target/release/lev-gateway --listen 127.0.0.1:8080 "${workers[@]}" >"$LOGS/gateway.log" 2>&1 &
pids+=($!)

for _ in $(seq 1 100); do
    if curl -fsS http://127.0.0.1:8080/readyz 2>/dev/null | grep -q "ready"; then
        break
    fi
    sleep 0.2
done
echo "gateway: $(curl -fsS http://127.0.0.1:8080/readyz 2>/dev/null || echo 'not ready, see .dev/logs/gateway.log')"
echo "try:     .venv/bin/python scripts/smoke.py"
echo "bench:   target/release/lev-bench --requests 200 --concurrency 16"
wait
