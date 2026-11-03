import type { ReactNode } from "react";
import type { Outcome, Role } from "@/api/types";
import { useT } from "@/i18n";
import { cn } from "@/lib/cn";

export function Tag({ children, className, title }: { children: ReactNode; className?: string; title?: string }) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex h-[18px] items-center rounded-xs border border-current px-1.5 text-2xs font-semibold leading-none whitespace-nowrap semicond",
        className,
      )}
    >
      {children}
    </span>
  );
}

const roleLabel: Record<Role, string> = { prefill: "PREFILL", decode: "DECODE", unified: "UNIFIED" };

export function RoleTag({ role }: { role: Role }) {
  return <Tag className="text-ink-2">{roleLabel[role]}</Tag>;
}

const outcomeLabel: Record<Outcome, string> = { ok: "ok", error: "error", rejected: "rejected", cancelled: "cancelled" };
const outcomeTone: Record<Outcome, string> = {
  ok: "text-ok",
  error: "text-bad",
  rejected: "text-warn",
  cancelled: "text-ink-3",
};
export function OutcomeTag({ outcome }: { outcome: Outcome }) {
  const t = useT();
  return (
    <Tag className={outcomeTone[outcome]} title={t.outcome[outcome]}>
      {outcomeLabel[outcome]}
    </Tag>
  );
}
