# Product

<!-- impeccable:product-schema 1 -->

> Source: the team lead's written brief (2026-11-01) and follow-ups (2026-11-02), plus the repository (README, docs/illustrated, gateway and router source).
> Confirmed by the lead: visual direction "The Live Datasheet"; English UI by default with a 中文 toggle. Facts marked *(inferred)* come from the brief and repo, not from a confirmed answer.

## Platform

web

## Stack

Pinned by the brief: Vite + React + TypeScript (strict) + Tailwind, static build served by nginx (no Node server at runtime). Node 24, pnpm 11. Allowed: shadcn/ui-style components, lucide icons, Recharts or visx, framer-motion.

## Users

- **Primary:** platform / inference-infra engineers (English or Chinese speaking) operating a Leviathan cluster serving DeepSeek-V4.1: checking cluster health, switching routing policy, debugging a slow or wrong request, verifying determinism by replay. *(inferred from the brief's page list)*
- **Secondary, decisive for this repo:** interviewers and reviewers watching a live demo (screen share or a laptop on a meeting-room display) of a resume/showcase project. They must understand within seconds what is special about the system. *(stated in the brief)*

## Product Purpose

Leviathan is the online serving layer that deepseek-recipe (protocol conversion, prompt rendering, tokenizer) and DeepInfer (inference engine) leave out: multi-protocol API, cache-aware routing, prefill/decode disaggregation, tenancy, observability and audit/replay. The console is its operating surface: overview, cluster topology, playground with prompt inspector, request audit with token-level replay, an agent studio, and tenants & usage (limits, quotas, API keys, daily usage). Success: an operator can see and change what the router is doing, and a viewer can watch the mechanism work on real traffic.

## Positioning

The gateway tokenizes, so the router knows at token-block granularity which worker already holds a prompt prefix (xxh3 chained block hashes, exact from KV events or approximate from a shadow index) and routes by minimising an estimated TTFT that unifies cache hit and load. Three protocols render to identical token IDs, so they share KV cache. Every request is replayable token by token from its recorded inputs and seed. A generic LLM dashboard cannot show any of this; the console's job is to make it visible.

## Operating Context

- Talks only to the Leviathan gateway HTTP API (default `http://localhost:8080`, configurable; optional admin bearer token and API key). Admin endpoints: overview, stats, router snapshot and runtime policy switch, workers, audit log with stage timelines, replay, tenants, keys, usage, `/admin/live` SSE (route, request_done, stage, worker events).
- Backends behind the gateway: `sim` (calibrated simulator for local/CI/demo), `vllm` on H200, `deepinfer` adapter (not run on real hardware). Demo topology is 2 prefill + 2 decode sim workers, or 2–4 unified.
- Traffic in demos comes from `lev-bench` or from the console's own Playground / Agent Studio.

## Capabilities and Constraints

- **Real data only.** Every number in the UI comes from the gateway API; no mock, placeholder or decorative figures. Absence of data is shown as a teaching empty state.
- UI copy is English by default with a 中文 toggle (typed en/zh dictionaries, choice persisted in localStorage); technical identifiers stay as-is in both (TTFT, TPOT, KV cache, prefill/decode, policy names `round_robin | least_load | p2c | cache_aware`, index modes `exact | approximate | off`). Docs under apps/console are in English.
- Light and dark themes are both required.
- Responsive down to about 1024px wide. WCAG AA contrast. `prefers-reduced-motion` disables packet animation.
- Must handle: SSE reconnect with backoff, 401 (admin token; API key when anonymous access is disabled), 429 with `Retry-After` (`rate_limit_exceeded`, `insufficient_quota`), 503 (no workers).
- API key secrets are shown once at creation (copy-once UI); the console never stores them except when the user chooses to use one in the Playground.
- Agent Studio tools (calculator, weather, unit converter) run in the browser and must be labelled as demo tools; the weather tool's data is explicitly a demo stub, not a real forecast.

## Brand Commitments

- Name: Leviathan. No logo, colour or typeface exists in the repo. *(inferred: none to preserve)*
- Voice: the repo's docs are precise, engineering-literate, and always cite real measured data (no made-up illustrative values). The console inherits that voice in both languages.

## Evidence on Hand

- `docs/illustrated/01–04`: lifecycle, recipe protocol (cross-protocol identical token IDs), block hashing and prefix index, cache-aware routing (the ~51-sequence crossover).
- Live system data only at runtime. No customers, benchmarks, or throughput claims may be invented in the UI.

## Product Principles

1. Show the mechanism, not a summary of it: routing decisions, block hashes and token IDs are first-class objects on screen.
2. Every figure is traceable to an API field; when there is no data, say why and how to produce some.
3. Operators first, demo-legible always: dense where work happens, instantly readable from across a room.
4. Determinism is a feature to prove, not a claim to state (replay, identical token IDs across protocols).

## Accessibility & Inclusion

WCAG AA contrast in both themes; full keyboard reach; reduced-motion respected for all non-essential animation.
