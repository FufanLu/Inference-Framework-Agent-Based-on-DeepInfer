/**
 * Packets on the block diagram. Each packet rides the very SVG path that draws
 * its net (getPointAtLength), so drawing and motion cannot disagree. One rAF
 * loop moves every packet; DOM nodes are created per packet and removed on
 * arrival. A cap keeps bursts of route events from flooding the layer.
 */

const SVG_NS = "http://www.w3.org/2000/svg";
const MAX_PACKETS = 48;
const TRAIL = [0, 0.035, 0.07, 0.105];

export type PacketKind = "request" | "kv";

interface Packet {
  path: SVGPathElement;
  length: number;
  start: number;
  duration: number;
  nodes: SVGRectElement[];
  onArrive?: () => void;
}

function easeInOut(t: number) {
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
}

export class PacketEngine {
  private readonly layer: SVGGElement;
  private packets: Packet[] = [];
  private frame: number | null = null;

  constructor(layer: SVGGElement) {
    this.layer = layer;
  }

  /** Send a packet along `path`. Returns false when the layer is saturated. */
  launch(path: SVGPathElement, duration: number, kind: PacketKind, onArrive?: () => void): boolean {
    if (this.packets.length >= MAX_PACKETS) {
      onArrive?.();
      return false;
    }
    const length = path.getTotalLength();
    if (!Number.isFinite(length) || length <= 0) {
      onArrive?.();
      return false;
    }
    const size = kind === "kv" ? 7 : 6;
    const nodes = TRAIL.map((_, index) => {
      const rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("width", String(size));
      rect.setAttribute("height", String(size));
      rect.setAttribute("x", String(-size / 2));
      rect.setAttribute("y", String(-size / 2));
      if (kind === "kv") {
        rect.setAttribute("fill", index === 0 ? "var(--sheet)" : "none");
        rect.setAttribute("stroke", "var(--route)");
        rect.setAttribute("stroke-width", "1.5");
      } else {
        rect.setAttribute("fill", "var(--route)");
      }
      rect.setAttribute("opacity", String(index === 0 ? 1 : 0.5 - index * 0.13));
      this.layer.appendChild(rect);
      return rect;
    });
    this.packets.push({ path, length, start: performance.now(), duration, nodes, onArrive });
    if (this.frame === null) this.frame = requestAnimationFrame(this.tick);
    return true;
  }

  private tick = (now: number) => {
    const still: Packet[] = [];
    for (const packet of this.packets) {
      const t = (now - packet.start) / packet.duration;
      if (t >= 1 || !packet.path.isConnected) {
        for (const node of packet.nodes) node.remove();
        packet.onArrive?.();
        continue;
      }
      packet.nodes.forEach((node, index) => {
        const local = Math.max(0, easeInOut(t) - (TRAIL[index] ?? 0));
        const point = packet.path.getPointAtLength(local * packet.length);
        node.setAttribute("transform", `translate(${point.x},${point.y})`);
        node.style.visibility = local <= 0 && index > 0 ? "hidden" : "visible";
      });
      still.push(packet);
    }
    this.packets = still;
    this.frame = this.packets.length > 0 ? requestAnimationFrame(this.tick) : null;
  };

  get active() {
    return this.packets.length;
  }

  dispose() {
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    for (const packet of this.packets) for (const node of packet.nodes) node.remove();
    this.packets = [];
    this.frame = null;
  }
}
