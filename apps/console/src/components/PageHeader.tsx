import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

interface PageHeaderProps {
  title: string;
  description?: ReactNode;
  /** Controls set on the title line, right-aligned. */
  actions?: ReactNode;
  className?: string;
}

/** Title and the page's controls on one line; one line of description under them. */
export function PageHeader({ title, description, actions, className }: PageHeaderProps) {
  return (
    <header className={cn("flex flex-col gap-1 border-b-[1.5px] border-ink pb-3", className)}>
      <div className="flex flex-wrap items-center justify-between gap-x-8 gap-y-2">
        <h1 className="cond-tight text-2xl font-bold tracking-[-0.005em] text-ink">{title}</h1>
        {actions ? <div className="flex flex-wrap items-center gap-x-5 gap-y-2">{actions}</div> : null}
      </div>
      {description ? <p className="max-w-[110ch] text-sm text-ink-2">{description}</p> : null}
    </header>
  );
}

/** A titled region inside a page: heading, optional caption line, content. */
export function Section({
  title,
  caption,
  aside,
  children,
  className,
}: {
  title: string;
  caption?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("flex min-w-0 flex-col", className)}>
      <div className="mb-2.5 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-0.5">
          <SectionHeading>{title}</SectionHeading>
          {caption ? <p className="text-xs text-ink-3">{caption}</p> : null}
        </div>
        {aside ? <div className="flex flex-wrap items-center gap-3">{aside}</div> : null}
      </div>
      {children}
    </section>
  );
}

/** Section and panel titles: condensed Archivo, one weight step under the page title. */
export function SectionHeading({ children, className }: { children: ReactNode; className?: string }) {
  return <h2 className={cn("cond text-lg leading-6 font-semibold text-ink", className)}>{children}</h2>;
}
