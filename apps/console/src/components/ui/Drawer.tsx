import { Dialog } from "radix-ui";
import { X } from "lucide-react";
import type { ReactNode } from "react";
import { useT } from "@/i18n";
import { cn } from "@/lib/cn";

interface DrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  width?: string;
}

/** A sheet that slides in from the right, over a dimmed page. */
export function Drawer({ open, onOpenChange, title, description, children, width = "min(760px, 92vw)" }: DrawerProps) {
  const t = useT();
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="anim-overlay fixed inset-0 z-40 bg-[var(--overlay)]" />
        <Dialog.Content
          style={{ width }}
          className={cn(
            "anim-sheet fixed inset-y-0 right-0 z-50 flex flex-col border-l border-ink bg-sheet shadow-[var(--shadow)]",
            "focus:outline-none",
          )}
        >
          <header className="flex items-start justify-between gap-4 border-b-[1.5px] border-ink px-6 pt-5 pb-4">
            <div className="min-w-0">
              <Dialog.Title className="cond-tight text-xl font-bold text-ink">{title}</Dialog.Title>
              {description ? (
                <Dialog.Description className="mt-1 text-xs text-ink-3">{description}</Dialog.Description>
              ) : (
                <Dialog.Description className="sr-only">{t.common.details}</Dialog.Description>
              )}
            </div>
            <Dialog.Close
              className="-mr-2 inline-flex size-8 shrink-0 items-center justify-center rounded-sm text-ink-2 hover:bg-wash hover:text-ink"
              aria-label={t.common.close}
            >
              <X className="size-4" />
            </Dialog.Close>
          </header>
          <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">{children}</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
