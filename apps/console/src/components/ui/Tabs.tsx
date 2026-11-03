import { Tabs as RadixTabs } from "radix-ui";
import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

export const TabsRoot = RadixTabs.Root;

export function TabsList({ children, className, label }: { children: ReactNode; className?: string; label: string }) {
  return (
    <RadixTabs.List aria-label={label} className={cn("flex items-end gap-5 border-b border-rule", className)}>
      {children}
    </RadixTabs.List>
  );
}

export function TabsTrigger({ value, children, count }: { value: string; children: ReactNode; count?: ReactNode }) {
  return (
    <RadixTabs.Trigger
      value={value}
      className={cn(
        "relative -mb-px inline-flex h-9 items-center gap-1.5 border-b-2 border-transparent text-sm text-ink-2",
        "transition-colors duration-150 hover:text-ink",
        "data-[state=active]:border-ink data-[state=active]:font-semibold data-[state=active]:text-ink",
      )}
    >
      {children}
      {count !== undefined ? <span className="font-mono text-2xs text-ink-3 tabular-nums">{count}</span> : null}
    </RadixTabs.Trigger>
  );
}

export function TabsContent({ value, children, className }: { value: string; children: ReactNode; className?: string }) {
  return (
    <RadixTabs.Content value={value} className={cn("focus-visible:outline-none", className)}>
      {children}
    </RadixTabs.Content>
  );
}
