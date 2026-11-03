---
version: 1
slug: "src"
primary_target: "src"
related_targets: []
---

# Leviathan console

Scope: the whole console app (`src/`), six pages: Overview, Cluster topology (signature), Playground + Prompt Inspector, Request audit + Replay, Agent Studio, Tenants & usage.
Visitor mode: Operate. Audience: infra engineers running the cluster; interviewers watching a live demo.
Task: see and change what the router does; debug and replay a request; prove cross-protocol token identity.
Constraints: real gateway data only, teaching empty states; English copy by default with a 中文 toggle, identifiers as-is; light + dark; >=1024px; WCAG AA; reduced motion stops packets.

## Direction contract

THESIS: The console is the live datasheet of the running cluster. Every figure is a measured characteristic printed with its conditions, sample size and source; every diagram is a datasheet diagram drawn from live API data. It refuses the category default of dark glowing KPI cards, big-number hero tiles and decorative sparklines.

OWN-WORLD: Cool white paper sheet with near-black ink; dark theme is a graphite sheet with bone ink, same rules. 三线表 tables everywhere (heavy top and bottom rule, hairline under the header, no verticals, no zebra). Condensed Archivo headers with units in parentheses; Archivo body; system CJK; Chivo Mono only for IDs, hashes, token IDs, JSON. One spot ink, route magenta, reserved for live routing: in-flight nets, packets, the chosen candidate. State is line form plus hue: live solid, stale dashed, unhealthy struck. Square corners, 1px rules, no shadows except overlays.

STORY: The visitor sees a cluster drawn as a functional block diagram, watches each real routing decision travel from gateway to worker (or prefill to decode), reads why that worker won in the candidate table, flips the policy and watches the decisions change, then opens a request, replays it, and sees the token-level verdict.

FIRST VIEWPORT: 集群拓扑 at 1440x900. Left: 232px contents sidebar with live counts and gateway/SSE status. Top of main: page title, and on the same line the router controls (policy segmented control as the primary action, index mode, PD toggle). Main left ~68%: the block diagram canvas, gateway block at left-center, pool enclosures (prefill, decode, unified; empty pools omitted) to the right, each worker an IC-style block with KV block cells; nets drawn orthogonally; magenta packets run along them. Right ~32%: latest routing decision, its candidates as a table with est. TTFT bars on one shared axis, chosen row inked magenta, then the recent route log.

FORM: Engineering datasheet / technical-standard document (三线表), position 5 of 7 on the ordered list; seed key 160b67f7. Raises: reserved route ink (orienteering course purple); state as line form (emission-line rail); discrete KV block cells and one-row-per-block inspector (Crouwel grid); one shared time scale per view (botanical folio); packets ride the drawn SVG net (zoo map single geometry); focus dims unrelated nets (streaming wall).

Signature interaction: live packets on the block diagram, one per real `route` event; disaggregated plans run gateway to prefill, then a dashed KV hand-off net to decode. Motion grammar: 150-250 ms state transitions, exponential ease-out, packets ~700 ms per leg, no page-load choreography; reduced motion replaces packets with a 1 s static highlight of the chosen net.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
