# Leviathan console

The operating console for the Leviathan gateway. It is a static React app that talks only to the gateway's HTTP API. Every number on screen comes from the API, and a page that has no data says why and how to produce some. The UI is in English by default; a 中文 toggle sits in the sidebar and in Settings.

| Page | What it shows | Gateway API |
| --- | --- | --- |
| **Overview** | Six service characteristics over a 30 s / 60 s / 5 min window: requests/s, TTFT p50/p99, TPOT p50/p90, prefix-cache hit rate, output tokens/s and DSpark acceptance. Each one has a 5-minute curve, the gateway identity, recent requests and a cluster summary. | `/admin/stats`, `/admin/overview`, `/admin/requests`, `/admin/router`, `/admin/live` |
| **Cluster topology** | A functional block diagram: the gateway with prefill, decode and unified pools.<br>Each worker block shows health, backend, running/waiting, KV in use (worker report) and cached prefix blocks (router index), plus inflight, backlog, dispatched and index source.<br>Every live `route` event sends a pulse along the drawn net. Disaggregated plans start the KV leg when the gateway reports `prefill_done`.<br>The side panel shows the decision's candidates on one est. TTFT axis.<br>Controls: policy, index mode and PD. | `/admin/router` (1 s), `/admin/workers`, `PUT /admin/router`, `POST /admin/workers`, `/admin/live` |
| **Playground** | Compose a request (thinking and effort, the demo `get_weather` tool, JSON schema, optional API key) and stream it through Chat Completions, Responses or Anthropic Messages.<br>Results are split into reasoning, content and tool calls, with a raw SSE frame log.<br>The Prompt Inspector shows one row per KV block, labelled with its chained hash.<br>A protocol-parity check proves the three protocols produce identical token IDs. | `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/leviathan/render` |
| **Request audit** | The audit log with outcome, worker, protocol and text filters; each row's timing bar shares one time axis.<br>The detail drawer has a timing diagram, a stage waterfall (`timeline`), routing and usage, output text or tokens, and the plan JSON.<br>Deterministic replay shows a token-level verdict. | `/admin/requests`, `/admin/requests/{id}`, `/admin/requests/{id}/replay`, `/v1/leviathan/detokenize` |
| **Agent Studio** | A browser-side agent loop over `/v1/chat/completions` with three demo tools that run in the page (weather with labelled demo data, calculator, unit converter). A timing diagram shows every model call and tool execution. | `/v1/chat/completions` |
| **Tenants & usage** | Tenants with RPM, TPM and monthly quota, plus month-to-date usage against the quota.<br>Create and edit tenants; create keys (the secret is shown once) and revoke them.<br>Daily usage is stacked as uncached prompt, cached prompt and completion. | `/admin/tenants`, `/admin/tenants/{id}/keys`, `/admin/keys`, `/admin/usage` |

## Run it

Requirements: Node 24 and pnpm 11.

```sh
cd apps/console
pnpm install
pnpm dev                 # http://localhost:5173, gateway at http://localhost:8080
VITE_GATEWAY_URL=http://127.0.0.1:18080 pnpm dev   # another gateway
pnpm build               # type-check (tsc -b) and build into dist/
pnpm preview             # serve dist/ on :4173
pnpm lint                # eslint
```

A local backend with real sim workers, from the repository root:

```sh
scripts/dev.sh                          # 2 prefill + 2 decode sim workers, gateway on :8080
cargo run -p lev-bench -- --requests 200 --concurrency 8 --policies cache_aware   # traffic
```

### Docker

```sh
docker build -t leviathan-console --build-arg VITE_GATEWAY_URL=http://localhost:8080 apps/console
docker run --rm -p 3000:80 leviathan-console      # http://localhost:3000
```

The image is a multi-stage build: `pnpm build` in Node 24, then `nginx:1.27-alpine` serving `dist/`. `nginx.conf` falls back to `index.html` for client routes, caches hashed assets for a year and answers `/healthz`. You can override the base images with `--build-arg NODE_IMAGE=… NGINX_IMAGE=…`.

## Configuration

- `VITE_GATEWAY_URL` sets the build-time default gateway (`http://localhost:8080`).
- **Settings** (sidebar) overrides, per browser:
  - the gateway URL;
  - the admin bearer token, used for `/admin/*` and the live stream;
  - the API key, used for `/v1/*`, with Test connection.

  The overrides live in `localStorage` under `leviathan.console.settings.v1`. Theme and language live in `leviathan.console.theme.v1` and `leviathan.console.locale.v1`.
- API key secrets are never stored, unless you press **Use in Playground** on a newly created key.

## How it works

- **Live events.** One shared connection to `/admin/live` lives in `src/api/live.ts`.
  - It reads the stream with `fetch`, because EventSource cannot send `Authorization`.
  - It reconnects with exponential backoff and jitter, and treats 45 s without a byte as dead (the gateway sends a heartbeat every 15 s).
  - On 401 it stops retrying until the settings change.
  - React sees throttled snapshots (every 250 ms); the packet animation subscribes to raw events.
- **Polling.** TanStack Query polls each resource at its own interval: router 1 s, audit 2 s, stats 2 s. Polling pauses in background tabs and stops after a 401.
- **Exact seeds.** Seeds are u64 and can exceed 2^53, so `parseGatewayJson` quotes every `seed` before `JSON.parse`. The UI shows them exactly.
- **Packets.** Each packet rides the SVG path that draws its net (`getPointAtLength`, one rAF loop, capped at 48 in flight; `src/pages/cluster/packets.ts`). With `prefers-reduced-motion`, nets highlight briefly instead.
- **Curves.** The Overview curves bucket finished requests from `request_done` events, merged with the audit log. Buckets from before the console could observe anything are hatched rather than drawn as zeros.
- **i18n.** Typed dictionaries (`src/i18n/en.ts` is the type source; `zh.ts` must match it). Components read them through `useT()`.

## Design

The console is styled as the live datasheet of the running cluster.
- **Characteristics.** Every figure is printed with its conditions, sample size and source.
- **Diagrams.** Each view uses the datasheet diagram that fits it:
  - the cluster is a functional block diagram;
  - requests and agent runs are timing diagrams;
  - the prompt is a memory map of KV blocks;
  - replay is a revision compare.
- **Tables** are three-line tables (三线表).
- **Colour.** One spot ink, route magenta, is reserved for live routing. State is carried by line form as well as hue: solid is live, dashed is stale, struck is down. The tokens are in `src/index.css` and `DESIGN.md`.
