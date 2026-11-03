"""Plot the burst sweep (scripts/burst_sweep.sh) as an SVG scatter.

x: how long the burst took to finish; y: interactive TTFT p99 during the
burst (log scale). One point per run, labelled with its runner limits.

    python3 bench/plot_burst.py bench/results/burst/memory docs/images/burst-pareto.svg
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

W, H = 760, 440
LEFT, RIGHT, TOP, BOTTOM = 76, 24, 86, 58

# Reference palette (dataviz skill): categorical slots 1-3, light and dark.
STYLE = """
.viz { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --grid:#e6e5e1;
       --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; }
@media (prefers-color-scheme: dark) {
  .viz { --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8f8e88; --grid:#33332f;
         --s1:#3987e5; --s2:#d95926; --s3:#199e70; }
}
.viz text { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; fill: var(--ink2); font-size: 12px; }
.viz .title { fill: var(--ink); font-size: 15px; font-weight: 600; }
.viz .axis { stroke: var(--muted); stroke-width: 1; }
.viz .grid { stroke: var(--grid); stroke-width: 1; }
.viz .ref { stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 4 4; }
.viz .label { fill: var(--ink); font-size: 11.5px; }
"""


def load(directory: Path) -> dict[str, dict]:
    runs = {}
    for summary in sorted(directory.glob("*/summary.json")):
        runs[summary.parent.name] = json.loads(summary.read_text())
    return runs


def main() -> None:
    directory, out = Path(sys.argv[1]), Path(sys.argv[2])
    runs = load(directory)
    baseline = runs["none"]["interactive"]["during_burst"]["ttft_p99_ms"]
    points = []  # (series, x, y, label)
    for name, run in runs.items():
        if name == "none":
            continue
        x = run["burst"]["makespan_s"]
        y = run["interactive"]["during_burst"]["ttft_p99_ms"]
        if name == "direct":
            points.append(("direct", x, y, "direct"))
            continue
        parts = name.split("-")  # jobs-<in_flight>-<delay>[-p0]
        series = "jobs, priority 0" if name.endswith("-p0") else "jobs, priority 1"
        points.append((series, x, y, f"{parts[1]} / {parts[2]} ms"))

    xs = [p[1] for p in points]
    x_min, x_max = 0.0, math.ceil(max(xs) / 20) * 20
    y_lo, y_hi = 10.0, 10 ** math.ceil(math.log10(max(p[2] for p in points)))
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM

    def sx(x: float) -> float:
        return LEFT + (x - x_min) / (x_max - x_min) * pw

    def sy(y: float) -> float:
        return TOP + ph - (math.log10(y) - math.log10(y_lo)) / (math.log10(y_hi) - math.log10(y_lo)) * ph

    colors = {"jobs, priority 1": "var(--s1)", "jobs, priority 0": "var(--s2)", "direct": "var(--s3)"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" class="viz" role="img" '
        'aria-label="Burst makespan against interactive TTFT p99">',
        f"<style>{STYLE}</style>",
        f'<rect width="{W}" height="{H}" fill="var(--surface)"/>',
        f'<text x="{LEFT}" y="24" class="title">A 2000-request burst: finish time vs. interactive TTFT p99</text>',
        f'<text x="{LEFT}" y="42">4 sim replicas (h200-vllm profile), 8 interactive clients; labels: max_in_flight / max_queue_delay</text>',
    ]
    for exp in range(int(math.log10(y_lo)), int(math.log10(y_hi)) + 1):
        y = 10**exp
        label = f"{y / 1000:g} s" if y >= 1000 else f"{y:g} ms"
        svg.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" class="grid"/>')
        svg.append(f'<text x="{LEFT - 8}" y="{sy(y) + 4:.1f}" text-anchor="end">{label}</text>')
    for x in range(int(x_min), int(x_max) + 1, 20):
        svg.append(f'<text x="{sx(x):.1f}" y="{TOP + ph + 18}" text-anchor="middle">{x} s</text>')
    svg.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{TOP + ph}" y2="{TOP + ph}" class="axis"/>')
    svg.append(
        f'<text x="{LEFT + pw / 2}" y="{H - 14}" text-anchor="middle">time for the whole burst to finish (lower is better)</text>'
    )
    svg.append(
        f'<text transform="translate(18 {TOP + ph / 2}) rotate(-90)" text-anchor="middle">interactive TTFT p99 (log)</text>'
    )
    svg.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{sy(baseline):.1f}" y2="{sy(baseline):.1f}" class="ref"/>')
    svg.append(f'<text x="{W - RIGHT - 4}" y="{sy(baseline) - 6:.1f}" text-anchor="end">no burst: {baseline:g} ms</text>')
    placed = [(sx(x), sy(y)) for _, x, y, _ in points]
    # Points of both priorities that coincide share one label.
    shared: dict[str, int] = {}
    for _, x, y, label in points:
        for _, ox, oy, other in points:
            if other == label and (ox, oy) != (x, y) and abs(sx(ox) - sx(x)) < 12 and abs(sy(oy) - sy(y)) < 16:
                shared[label] = shared.get(label, 0) + 1
    labelled: set[str] = set()
    for series, x, y, label in points:
        cx, cy = sx(x), sy(y)
        # Label to the right unless that runs into a neighbour or the edge.
        crowded = any(0 < ox - cx < 120 and abs(oy - cy) < 16 for ox, oy in placed)
        tip = f"{series} {label}: burst {x:g} s, TTFT p99 {y:g} ms"
        svg.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{colors[series]}" stroke="var(--surface)" stroke-width="2">'
            f"<title>{tip}</title></circle>"
        )
        if label in shared:
            if label in labelled:
                continue
            labelled.add(label)
            label, crowded = f"{label} (both priorities)", False
        anchor, dx = ("end", -9) if crowded or cx > W - RIGHT - 120 else ("start", 9)
        svg.append(f'<text x="{cx + dx:.1f}" y="{cy + 4:.1f}" text-anchor="{anchor}" class="label">{label}</text>')
    # Legend: one row above the plot.
    for i, series in enumerate(colors):
        x = LEFT + i * 150
        svg.append(f'<circle cx="{x + 5}" cy="62" r="5" fill="{colors[series]}"/>')
        svg.append(f'<text x="{x + 15}" y="66">{series}</text>')
    svg.append("</svg>")
    out.write_text("\n".join(svg) + "\n")
    print(f"wrote {out} ({len(points)} points)")


if __name__ == "__main__":
    main()
