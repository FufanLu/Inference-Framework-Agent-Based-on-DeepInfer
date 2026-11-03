import type { Plan } from "@/api/types";

/** "worker" or "prefill → decode". */
export function planSummary(plan: Plan | null | undefined): string {
  if (!plan) return "—";
  return plan.kind === "unified" ? plan.worker.worker_id : `${plan.prefill.worker_id} → ${plan.decode.worker_id}`;
}
