---
name: Leviathan Console
description: The live datasheet of a running Leviathan inference cluster.
colors:
  route: "oklch(0.54 0.235 348)"
  route-wash: "oklch(0.955 0.028 348)"
  paper: "oklch(0.975 0.003 250)"
  sheet: "oklch(1 0 0)"
  wash: "oklch(0.948 0.004 255)"
  rule: "oklch(0.875 0.005 255)"
  rule-2: "oklch(0.8 0.006 255)"
  ctrl: "oklch(0.6 0.01 262)"
  ink: "oklch(0.22 0.012 262)"
  ink-2: "oklch(0.43 0.012 262)"
  ink-3: "oklch(0.53 0.012 262)"
  ok: "oklch(0.5 0.115 155)"
  warn: "oklch(0.535 0.12 70)"
  bad: "oklch(0.53 0.19 27)"
  bad-wash: "oklch(0.955 0.03 27)"
typography:
  display:
    fontFamily: "Archivo Variable, PingFang SC, Hiragino Sans GB, Microsoft YaHei UI, Microsoft YaHei, Noto Sans CJK SC, Source Han Sans SC, system-ui, sans-serif"
    fontSize: "1.75rem"
    fontWeight: 700
    lineHeight: "2.125rem"
    letterSpacing: "-0.005em"
    fontVariation: "'wdth' 75"
  headline:
    fontFamily: "Archivo Variable, PingFang SC, Hiragino Sans GB, Microsoft YaHei UI, Microsoft YaHei, Noto Sans CJK SC, Source Han Sans SC, system-ui, sans-serif"
    fontSize: "1.375rem"
    fontWeight: 700
    lineHeight: "1.875rem"
    fontVariation: "'wdth' 75"
  title:
    fontFamily: "Archivo Variable, PingFang SC, Hiragino Sans GB, Microsoft YaHei UI, Microsoft YaHei, Noto Sans CJK SC, Source Han Sans SC, system-ui, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 600
    lineHeight: "1.5rem"
    fontVariation: "'wdth' 80"
  reading:
    fontFamily: "Archivo Variable, PingFang SC, Hiragino Sans GB, Microsoft YaHei UI, Microsoft YaHei, Noto Sans CJK SC, Source Han Sans SC, system-ui, sans-serif"
    fontSize: "1.75rem"
    fontWeight: 600
    lineHeight: "2.125rem"
    letterSpacing: "-0.02em"
    fontFeature: "'tnum'"
  body:
    fontFamily: "Archivo Variable, PingFang SC, Hiragino Sans GB, Microsoft YaHei UI, Microsoft YaHei, Noto Sans CJK SC, Source Han Sans SC, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: "1.375rem"
  body-sm:
    fontFamily: "Archivo Variable, PingFang SC, Hiragino Sans GB, Microsoft YaHei UI, Microsoft YaHei, Noto Sans CJK SC, Source Han Sans SC, system-ui, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: "1.25rem"
    fontFeature: "'tnum'"
  label:
    fontFamily: "Archivo Variable, PingFang SC, Hiragino Sans GB, Microsoft YaHei UI, Microsoft YaHei, Noto Sans CJK SC, Source Han Sans SC, system-ui, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 600
    lineHeight: "1rem"
    fontVariation: "'wdth' 80"
  caption:
    fontFamily: "Archivo Variable, PingFang SC, Hiragino Sans GB, Microsoft YaHei UI, Microsoft YaHei, Noto Sans CJK SC, Source Han Sans SC, system-ui, sans-serif"
    fontSize: "0.6875rem"
    fontWeight: 400
    lineHeight: "1rem"
  mono:
    fontFamily: "Chivo Mono Variable, ui-monospace, SF Mono, Menlo, Consolas, PingFang SC, monospace"
    fontSize: "0.75rem"
    fontWeight: 400
    lineHeight: "1.125rem"
rounded:
  xs: "1px"
  sm: "2px"
spacing:
  cell-y: "6px"
  cell-x: "10px"
  control-x: "12px"
  block: "12px"
  section: "24px"
  page-x: "24px"
  page-x-wide: "32px"
  column: "32px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.sm}"
    padding: "0 12px"
    height: "32px"
  button-primary-hover:
    backgroundColor: "{colors.ink-2}"
  button-secondary:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "0 12px"
    height: "32px"
  button-secondary-hover:
    backgroundColor: "{colors.wash}"
  button-ghost:
    textColor: "{colors.ink-2}"
    rounded: "{rounded.sm}"
    padding: "0 12px"
    height: "32px"
  button-danger:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.bad}"
    rounded: "{rounded.sm}"
    padding: "0 12px"
    height: "32px"
  button-sm:
    typography: "{typography.label}"
    padding: "0 10px"
    height: "28px"
  input:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.sm}"
    padding: "0 10px"
    height: "32px"
  segmented-cell:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink-2}"
    padding: "0 12px"
    height: "32px"
  segmented-cell-selected:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  tag:
    typography: "{typography.caption}"
    rounded: "{rounded.xs}"
    padding: "0 6px"
    height: "18px"
  nav-item:
    textColor: "{colors.ink-2}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.sm}"
    padding: "0 10px"
    height: "36px"
  nav-item-active:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  table-header-cell:
    textColor: "{colors.ink-2}"
    typography: "{typography.label}"
    padding: "7px 10px 6px"
  table-cell:
    typography: "{typography.body-sm}"
    padding: "6px 10px"
  table-row-chosen:
    backgroundColor: "{colors.route-wash}"
    textColor: "{colors.route}"
---

# Design System: Leviathan Console

## Overview

**Creative North Star: "The Live Datasheet"**

The console is printed like an engineering datasheet for a cluster that is running right now. Every figure is a measured characteristic, given with its unit, its conditions, its sample window and its source. Every diagram is a datasheet figure drawn from live API data: a functional block diagram with orthogonal nets, IC-style worker blocks, a figure key set beside it, and characteristic curves on framed, dotted-grid plots. The page reads as cool white paper with near-black ink. The dark theme is a graphite sheet with bone ink and follows the same rules; only the token values change.

Density follows a technical standard rather than a dashboard. Tables are three-line tables (三线表). Headers are condensed Archivo with the unit in parentheses. Hierarchy comes from rule weight (1.5px heavy, 1px, 0.75px hairline), from condensed width, and from ink depth, not from cards, tints or glow. There is one spot ink, route magenta. It appears only where a routing decision is happening, so the eye can follow a request across the screen.

State never depends on hue alone. The form of a line is the state: solid is live, dashed is stale, dotted is idle or unused, struck is down. The same vocabulary covers status marks, diagram nets, KV cells and pool frames. Motion exists only to show real events: a packet for each real `route` event, a brief hit outline on the worker that received it, and a row flash for new audit entries. Nothing animates on page load.

The system rejects the category default: dark glowing KPI cards, big-number hero tiles, decorative sparklines.

**Key Characteristics:**
- Paper and ink, with one spot ink (route magenta) reserved for live routing.
- State is carried by line form as well as hue: solid live, dashed stale, dotted idle, struck down.
- Three-line tables: heavy top and bottom rules, a hairline under the header, no row rules, no verticals, no zebra.
- Condensed Archivo (wdth 75 to 80) for titles, headers and parameter names. Archivo at normal width for body text. Chivo Mono only for machine identifiers.
- Square corners (1 to 2px) and thin rules. Shadows appear only on overlays.
- Both themes pass WCAG AA: every text token is at least 4.5:1 on paper, sheet and wash, and control borders are at least 3:1.

## Colors

A restrained palette: neutral paper and ink with a faint cool cast, three signal hues for health, and one saturated spot ink kept for live routing.

### Primary
- **Route Magenta** (`route`): the spot ink, used only for live routing. That covers packets on the block diagram, the reduced-motion net highlight, the hit outline on a worker block that just received a request, the chosen candidate in the routing-decision table (its marker, worker ID, est. TTFT bar and value), the count of recent routes on the gateway block, and the figure-key swatches for these. Dark value is `oklch(0.73 0.18 350)`.
- **Route Wash** (`route-wash`): the background of the chosen candidate row, and nothing else. Dark value is `oklch(0.26 0.05 350)`.

### Neutral
- **Cool Paper** (`paper`): the page ground and the background of characteristic cells. It is also the text colour on inked fills (primary button, selected segment, active nav item, selection).
- **White Sheet** (`sheet`): raised working surfaces. These are controls, worker and gateway blocks, drawers and dialogs.
- **Wash** (`wash`): the sidebar ground, row hover and selection, the code-block ground, skeletons, and the disabled-field ground.
- **Hairline Rule** (`rule`): dividers inside blocks, the tab baseline, segment separators, chart grid lines and hatch strokes.
- **Strong Rule** (`rule-2`): chart frames, inactive pool frames, unused and unhealthy nets, empty KV cells, and the scrollbar thumb.
- **Control Edge** (`ctrl`): the border of every interactive control (button, field, segmented control, switch). It is at least 3:1 on every surface.
- **Ink** (`ink`): primary text, heavy table rules, healthy worker block borders, the focus outline and inked fills.
- **Ink 2** (`ink-2`): secondary text, table header text, field labels and descriptions.
- **Ink 3** (`ink-3`): tertiary text such as units, conditions, captions and the placeholder. Also used for live net strokes and cached KV cells.

### Signal
- **Healthy Green** (`ok`): online, connected and ok outcomes.
- **Amber** (`warn`): retrying, rejected and exhausted.
- **Fault Red** (`bad`): offline, error and danger actions, plus KV cells above the 95% penalty threshold. **Fault Wash** (`bad-wash`) is used only as the danger-button hover ground.

### Named Rules
**The Spot Ink Rule.** Route magenta means "a routing decision is happening here, now". It never decorates, never marks a link or a primary action, and never shows health. When nothing is being routed, no magenta is on screen.

**The Line Form Rule.** Every state that has a hue also has a line form: solid live, dashed stale, dotted idle or unused, struck down. A status must still read correctly in greyscale.

**The Same Rules, Graphite Sheet Rule.** Dark mode swaps only token values. Bone ink goes on graphite. No component changes structure, weight or accent use between themes.

## Typography

**Display Font:** Archivo Variable, condensed through the `wdth` axis (with PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans CJK SC, Source Han Sans SC, system-ui)
**Body Font:** Archivo Variable at normal width (same CJK fallback)
**Label/Mono Font:** Chivo Mono Variable (with ui-monospace, SF Mono, Menlo, Consolas)

**Character:** One grotesque family set at three widths does the work of a datasheet's type system. It is tight and condensed for titles and column heads, and open for prose and values. The mono face is a separate voice that appears only where the machine is speaking.

### Hierarchy
- **Display** (700, 1.75rem/2.125rem, wdth 75, -0.005em): page titles only, set on one line with the page's controls above a 1.5px ink rule.
- **Headline** (700, 1.375rem/1.875rem, wdth 75): drawer and dialog titles.
- **Title** (600, 1.125rem/1.5rem, wdth 80): section and panel headings. Characteristic names step down to 1rem.
- **Reading** (600, 1.75rem, -0.02em, tabular): a measured value inside a characteristic cell. It always has its unit in ink-2 and its conditions line under it. This is not a hero number.
- **Body** (400, 0.875rem/1.375rem): base prose. Descriptions run to a maximum of 110ch, and empty-state copy to 68ch.
- **Body small** (400, 0.8125rem/1.25rem, tabular): table cells, controls and inputs.
- **Label** (600, 0.75rem/1rem, wdth 80): table headers, field labels, the figure-key title and parameter names. Sentence case, with units in parentheses in a lighter weight.
- **Caption** (400, 0.6875rem/1rem): conditions, footnotes, block footers and tags (tags use semicondensed wdth 90 at 600).
- **Mono** (Chivo Mono, about 0.75rem): worker IDs, request IDs, hashes, token IDs, JSON, endpoints, policy and index-mode values, and shell commands.

### Named Rules
**The Machine Voice Rule.** Chivo Mono is only for strings a machine emits or accepts: IDs, hashes, token IDs, JSON, endpoints, enum values and commands. Prose, labels and numbers set in Archivo.

**The Condensed Header Rule.** Titles, column heads and parameter names are condensed (wdth 75 for titles, 80 for headers and labels). Body text and values are never condensed.

**The Tabular Figures Rule.** Any column or row of numbers uses tabular figures, right-aligned in tables.

## Layout

The app shell has a fixed contents sidebar (220px, or 232px from 1280px) on the wash ground, with a right hairline. The main column uses 24px side padding (32px from 1280px), 20px top and 48px bottom. The layout is built for widths of 1024px and up.

Each page starts with a header row: the title on the left and the page's controls right-aligned on the same line, then one line of description, closed by a 1.5px ink rule. Sections stack with 24 to 28px between them. A section heading sits 10px above its content, and its caption runs inline on the same baseline.

From 1280px, working pages split into a fluid main column and a 340 to 400px side column with 32px between them. Cluster puts the diagram in the main column and the decision panel plus route log in the side column. Overview splits 1.6fr to 1fr. Overview characteristics form a grid of cells separated by 1px rule gaps, bounded above and below by a 1.5px ink rule (one column, two from 768px, three from 1536px). Rhythm inside blocks is 8 to 12px. Table cells are 6px by 10px.

### Named Rules
**The One Time Scale Rule.** Charts in one view share one clock-time axis and window, and comparable bars (est. TTFT across candidates) share one axis. That axis is stated in the header.

## Elevation & Depth

The page is flat. Depth comes from the surface step (paper, then sheet, then wash) and from rule weight: 1.5px ink for heavy rules and the gateway block, 1px for blocks and controls, 0.75px hairlines for pool frames, header underlines and chart frames. Only overlays cast a shadow. Focus dims unrelated diagram nets to 18% opacity and unrelated worker blocks to 45%. It does not lift anything.

### Shadow Vocabulary
- **Overlay** (`--shadow`, light: `0 12px 32px -12px oklch(0.22 0.02 262 / 0.28), 0 2px 6px -2px oklch(0.22 0.02 262 / 0.12)`; dark: `0 16px 40px -12px oklch(0 0 0 / 0.6), 0 2px 8px -2px oklch(0 0 0 / 0.4)`): drawers and dialogs, over a dimmed `--overlay` scrim.
- **Hit outline** (`inset 0 0 0 1px` route): not elevation. It is a 700ms inner stroke that doubles a worker block's border in route magenta when a request reaches it.

### Named Rules
**The Overlays Only Rule.** A shadow means "this sheet is lying on top of the page". Blocks, cells, cards and controls on the page carry rules, not shadows.

## Shapes

Corners are square or close to it: 2px (`rounded.sm`) for buttons, fields, segmented controls, the switch and nav items, and 1px (`rounded.xs`) for tags, code blocks and skeletons. Diagram blocks, pool frames, characteristic cells and tables have no rounding at all. The switch is a rectangle with a square thumb. The disclosure marker is a small ink triangle. Packets are small squares (6px, or 7px hollow for KV). Diagram nets are orthogonal, run horizontally and vertically, and end in solid arrowheads. Pool enclosures carry their name in a boxed tab that breaks the top frame line, as on a datasheet drawing.

## Components

### Buttons
Quiet, ruled and exact.
- **Shape:** near-square (2px). Heights are 32px (md, 0.8125rem) and 28px (sm, 0.75rem), weight 500, with a 1px border on every variant.
- **Primary:** an ink fill with paper text. Hover steps to ink-2. Disabled uses ink-3. Each page has at most one primary action.
- **Secondary (default):** a sheet ground, control-edge border and ink text. Hover goes to wash.
- **Ghost:** a transparent border with ink-2 text. Hover adds wash and ink text.
- **Danger:** a sheet ground with a fault-red border and text. Hover goes to fault wash.
- **Busy:** a spinning loader replaces the icon, and `aria-busy` is set. Colour transitions take 150ms ease-out.

### Segmented Control
A row of ruled cells with the selected cell inked. It has a control-edge outer border and rule separators. Selected cells are ink on paper. Hover goes to wash. While a change is in flight, the pending cell shows wash with a pulsing 1px underline. A mono variant (Chivo Mono, 0.75rem) is used for enum values such as policy and index mode. It is the primary control on Cluster.

### Switch
A 34 by 18px rectangle with a 2px radius and a 12px square thumb. Off is sheet with a control edge and an ink-3 thumb. On is an ink fill with a paper thumb. The thumb moves in 200ms expo-out. The label sits to the right, followed by state text in ink-3.

### Inputs / Fields
- **Style:** a sheet ground, a 1px control-edge border, 2px radius, 32px height and 0.8125rem text. The placeholder is ink-3 in Archivo, even for mono fields.
- **Focus:** the border goes to ink, with a 1px ink ring.
- **Error / Disabled:** errors use a fault-red border and ring with a red message (`role="alert"`). Disabled fields use wash with ink-3 text. The field label is a condensed 0.75rem/600 ink-2 label above the control, and the hint below is in ink-3.

### Tags
An outlined rectangle 18px tall with a 1px radius, a `currentColor` border and semicondensed 0.6875rem/600 text. The colour comes from the meaning: role tags (PREFILL, DECODE, UNIFIED) use ink-2, and outcome tags use ok, bad, warn or ink-3.

### Three-Line Table
The datasheet table. It has a 1.5px ink rule at the top and bottom and a 0.75px ink hairline under the header. There are no row rules, no vertical rules and no zebra striping. Headers are condensed 0.75rem/600 ink-2 labels, bottom-aligned, with units in parentheses at weight 400 in ink-3. Cells are 0.8125rem with tabular figures, and numbers are right-aligned. Rows that open something get a wash on hover and on selection. New rows flash wash for 1.2s. The chosen routing candidate is the only row that may use route wash.

### Status Mark
An 18px line glyph plus a label, coloured by tone. Solid means live, dashed (3 / 2.5) means stale, round-dotted (1 / 2.5) means idle, and a solid line with a diagonal strike means down. Health tones are ok, warn, bad, ink and muted.

### KV Cells
KV usage is a row of discrete block cells (24 by default, 2px gaps), not a smooth bar. Ink cells are blocks in use. Ink-3 cells with a border are prefix blocks the router's index holds. Empty cells have a strong-rule outline. When usage is not reported, the outlines go dotted. Above 95% usage the filled cells turn fault red. The cells form a `role="meter"` with a spoken label.

### Navigation
Sidebar items are 36px tall with a 2px radius and a 16px lucide icon at stroke 1.75. Inactive items are ink-2 and go to a rule-tinted ground on hover. The active item is an ink fill with paper text. The cluster item carries a mono healthy-worker count. The sidebar foot holds gateway and live-stream status marks, the endpoint in mono, and the settings, theme and 中文 toggles. Tabs use a rule baseline. The active tab gets a 2px ink underline and weight 600, with an optional mono count.

### Drawer
A right-hand sheet (up to 760px or 92vw) with a 1px ink left edge and the overlay shadow, over the scrim. The header has a condensed headline and an ink-3 description, closed by a 1.5px ink rule. It slides in over 220ms with expo-out (24px) and out over 160ms ease-in.

### Block Diagram (signature)
A functional block diagram of the cluster. The gateway block has a 1.5px ink frame and a parameter list. Pool enclosures use 0.75px frames with boxed name tabs (ink-3 when the pool is active, strong rule when it is not). Worker blocks are 1px ink frames on sheet, with a mono ID, role tag, status mark, KV cells and a metric grid. An unhealthy worker gets a control-edge frame, 80% opacity and a struck ID. Nets are orthogonal SVG paths. In-use nets are ink-3 at 1.25px. Unused or unhealthy nets are strong rule, dotted (1 / 3). The KV hand-off net from prefill to decode is dashed (5 / 3). Each real `route` event launches a magenta square packet that rides the same SVG path (650ms request leg, 520ms KV leg, ease-in-out, with a four-step fading trail, capped at 48). The net also flashes magenta at 2px and fades over 900ms. Under `prefers-reduced-motion`, no packets run. The chosen net is instead highlighted in magenta for 1s and then removed. A figure key beside the diagram explains every line form and swatch. Hovering a worker dims everything unrelated to it.

### Characteristic Curve
A framed plot with a strong-rule 0.75px frame, a dotted major grid (1 / 3), a solid zero line and clock-time ticks in 10.5px ink-3. Gaps mark buckets with no value. A hatched band marks the time the console was not observing, and is labelled as such. A 3px ink band on the time axis marks the window the headline figure covers. A secondary curve (p99 beside p50) is lighter, never dashed.

## Do's and Don'ts

### Do:
- **Do** state every figure with its unit, conditions and window. Put units in parentheses in condensed headers.
- **Do** use three-line tables for tabular data: a 1.5px ink rule at top and bottom, a 0.75px hairline under the header, and no row rules.
- **Do** pair every state hue with its line form: solid live, dashed stale, dotted idle or unused, struck down.
- **Do** keep route magenta for live routing (packets, the chosen candidate, highlighted nets, the hit outline), so that seeing it means a decision is happening.
- **Do** set IDs, hashes, token IDs, JSON, endpoints and enum values in Chivo Mono, and everything else in Archivo.
- **Do** use the condensed widths (wdth 75 for titles, 80 for headers and labels) for titles, headers and parameter names.
- **Do** check new text tokens at 4.5:1 or better on paper, sheet and wash in both themes, and new control borders at 3:1 or better.
- **Do** replace packets with a brief static net highlight under `prefers-reduced-motion`, and keep state transitions at 150 to 250ms with expo-out.
- **Do** show a teaching empty state (why there is no data, plus the command that produces some) instead of placeholder figures.

### Don't:
- **Don't** use dark glowing KPI cards, big-number hero tiles or decorative sparklines.
- **Don't** use route magenta for links, primary buttons, brand accents or health status.
- **Don't** dash a data curve or a status line to mean anything except stale. The KV hand-off net is the one dashed net, and dotted chart grid lines are the plot's own furniture.
- **Don't** add row rules, vertical rules or zebra striping to tables.
- **Don't** put shadows on page-level blocks, cells or controls. Shadows belong to drawers and dialogs.
- **Don't** round corners beyond 2px, or put rounding on diagram blocks, frames, tables or cells.
- **Don't** set prose, labels or numbers in the mono face, or condense body text.
- **Don't** animate on page load. Motion only reports real events.
