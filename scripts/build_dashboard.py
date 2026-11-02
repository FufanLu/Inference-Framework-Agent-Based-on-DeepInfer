#!/usr/bin/env python3
"""Generate dashboards/grafana/leviathan.json from the metric names the gateway exports."""

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}


def panel(pid, title, exprs, x, y, w=8, h=8, unit="short", kind="timeseries", legend=None):
    targets = [
        {"datasource": DS, "expr": expr, "legendFormat": fmt, "refId": chr(65 + i)}
        for i, (expr, fmt) in enumerate(exprs)
    ]
    return {
        "id": pid,
        "type": kind,
        "title": title,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": targets,
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}} if kind == "timeseries" else {},
    }


def q(metric, quantile, by="model"):
    return f"histogram_quantile({quantile}, sum by (le, {by}) (rate({metric}_bucket[1m])))"


panels = [
    panel(1, "Requests / s by outcome", [("sum by (outcome) (rate(lev_requests_total[1m]))", "{{outcome}}")], 0, 0, unit="reqps"),
    panel(2, "TTFT (gateway)", [(q("lev_ttft_seconds", 0.5), "p50"), (q("lev_ttft_seconds", 0.99), "p99")], 8, 0, unit="s"),
    panel(3, "TPOT", [(q("lev_tpot_seconds", 0.5), "p50"), (q("lev_tpot_seconds", 0.99), "p99")], 16, 0, unit="s"),
    panel(4, "Prefix cache hit rate", [(
        "sum(rate(lev_cached_tokens_total[1m])) / clamp_min(sum(rate(lev_prompt_tokens_total[1m])), 1)", "hit rate")],
        0, 8, unit="percentunit"),
    panel(5, "Tokens / s", [
        ("sum(rate(lev_prompt_tokens_total[1m]))", "prompt"),
        ("sum(rate(lev_cached_tokens_total[1m]))", "prompt (cached)"),
        ("sum(rate(lev_completion_tokens_total[1m]))", "completion")], 8, 8),
    panel(6, "DSpark draft acceptance", [(
        "sum(rate(lev_spec_accepted_tokens_total[1m])) / clamp_min(sum(rate(lev_spec_draft_tokens_total[1m])), 1)",
        "accepted / drafted")], 16, 8, unit="percentunit"),
    panel(7, "Routing decisions / s", [("sum by (policy, plan) (rate(lev_route_decisions_total[1m]))", "{{policy}} · {{plan}}")], 0, 16, unit="ops"),
    panel(8, "Worker KV usage", [("lev_worker_kv_usage", "{{worker}}")], 8, 16, unit="percentunit"),
    panel(9, "Worker running / waiting", [("lev_worker_running", "{{worker}} running"), ("lev_worker_waiting", "{{worker}} waiting")], 16, 16),
    panel(10, "Dispatch retries / s", [("sum(rate(lev_dispatch_retries_total[1m]))", "retries")], 0, 24, unit="ops"),
    panel(11, "Worker decode tokens / s", [("lev_worker_decode_tokens_per_second", "{{worker}}")], 8, 24),
    panel(12, "Tokens by tenant / s", [("sum by (tenant) (rate(lev_completion_tokens_total[1m]))", "{{tenant}}")], 16, 24),
]

dashboard = {
    "uid": "leviathan",
    "title": "Leviathan",
    "tags": ["leviathan", "deepseek"],
    "timezone": "browser",
    "refresh": "5s",
    "time": {"from": "now-15m", "to": "now"},
    "schemaVersion": 39,
    "panels": panels,
}

out = Path(__file__).resolve().parent.parent / "dashboards" / "grafana" / "leviathan.json"
out.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {out}")
