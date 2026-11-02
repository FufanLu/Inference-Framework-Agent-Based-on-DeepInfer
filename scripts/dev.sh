#!/usr/bin/env bash
# Run a local Leviathan stack without Docker: sim workers + the gateway.
#
#   scripts/dev.sh                 # 2 prefill + 2 decode sim workers (PD)
#   scripts/dev.sh unified 4       # 4 unified sim workers
#   scripts/dev.sh stop            # stop the stack started from this checkout
#
# Environment:
#   PROFILE=fast-dev               sim profile (default h200-vllm)
#   WORKER_ARGS="--total-blocks 800"  extra worker flags (e.g. small KV per replica)
#   GATEWAY_PORT=8080 BASE_PORT=50051  ports, so several stacks can coexist
#
# Logs and PIDs live in .dev/. Only processes recorded in .dev/pids are ever
# stopped; other stacks on the machine are left alone.
set -euo pipefail
cd "$(dirname "$0")/.."

DEV=.dev
LOGS=$DEV/logs
PIDS=$DEV/pids
mkdir -p "$LOGS"

stop_stack() {
    [[ -f "$PIDS" ]] || return 0
    while read -r pid; do
        kill "$pid" 2>/dev/null || true
    done <"$PIDS"
    rm -f "$PIDS"
}

MODE="${1:-pd}"
COUNT="${2:-2}"
if [[ "$MODE" == "stop" ]]; then
    stop_stack
    echo "stopped"
    exit 0
fi
case "$MODE" in
pd | unified) ;;
*)
    echo "usage: $0 [pd|unified] [count] | stop" >&2
    exit 2
    ;;
esac

PROFILE="${PROFILE:-h200-vllm}"
PY="${PYTHON:-.venv/bin/python}"
GATEWAY_PORT="${GATEWAY_PORT:-8080}"
BASE_PORT="${BASE_PORT:-50051}"

stop_stack # a previous stack from this checkout
make -s proto-py PYTHON="$PY"
cargo build -q --release -p lev-gateway -p lev-bench

trap 'stop_stack; exit 0' INT TERM
trap 'stop_stack' EXIT

workers=()
start_worker() {
    local id="$1" role="$2" port="$3"
    # shellcheck disable=SC2086 # WORKER_ARGS is a list of flags
    (cd worker && exec "../$PY" -m leviathan_worker.server --backend sim --host 127.0.0.1 --port "$port" \
        --worker-id "$id" --role "$role" --profile "$PROFILE" \
        --tokenizer ../assets/tokenizers/v41/tokenizer.json --log-format text ${WORKER_ARGS:-}) \
        >"$LOGS/$id.log" 2>&1 &
    echo $! >>"$PIDS"
    workers+=(--worker "http://127.0.0.1:$port")
}

port=$BASE_PORT
if [[ "$MODE" == "pd" ]]; then
    for i in $(seq 0 $((COUNT - 1))); do start_worker "prefill-$i" prefill $((port++)); done
    for i in $(seq 0 $((COUNT - 1))); do start_worker "decode-$i" decode $((port++)); done
else
    for i in $(seq 0 $((COUNT - 1))); do start_worker "sim-$i" unified $((port++)); done
fi
expected=$((port - BASE_PORT))

target/release/lev-gateway --listen "127.0.0.1:$GATEWAY_PORT" "${workers[@]}" >"$LOGS/gateway.log" 2>&1 &
echo $! >>"$PIDS"

for _ in $(seq 1 150); do
    if curl -fsS "http://127.0.0.1:$GATEWAY_PORT/readyz" 2>/dev/null | grep -q "ready: $expected healthy"; then
        break
    fi
    sleep 0.2
done
echo "gateway: $(curl -fsS "http://127.0.0.1:$GATEWAY_PORT/readyz" 2>/dev/null || echo "not ready, see $LOGS/gateway.log")"
echo "try:     .venv/bin/python scripts/smoke.py --base-url http://127.0.0.1:$GATEWAY_PORT"
echo "bench:   target/release/lev-bench --gateway http://127.0.0.1:$GATEWAY_PORT --requests 200"
wait
