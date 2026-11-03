import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { LoaderCircle } from "lucide-react";
import { cn } from "@/lib/cn";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  busy?: boolean;
  icon?: ReactNode;
}

const variants: Record<Variant, string> = {
  primary:
    "border-ink bg-ink text-paper hover:bg-ink-2 hover:border-ink-2 disabled:bg-ink-3 disabled:border-ink-3 disabled:hover:bg-ink-3",
  secondary: "border-ctrl bg-sheet text-ink hover:bg-wash disabled:text-ink-3 disabled:hover:bg-sheet",
  ghost: "border-transparent text-ink-2 hover:bg-wash hover:text-ink disabled:text-ink-3 disabled:hover:bg-transparent",
  danger: "border-bad bg-sheet text-bad hover:bg-bad-wash disabled:opacity-60",
};

const sizes: Record<Size, string> = {
  sm: "h-7 gap-1.5 px-2.5 text-xs",
  md: "h-8 gap-2 px-3 text-sm",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", busy = false, icon, className, children, disabled, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type="button"
      className={cn(
        "inline-flex shrink-0 items-center justify-center rounded-sm border font-medium whitespace-nowrap",
        "transition-colors duration-150 ease-out select-none",
        variants[variant],
        sizes[size],
        className,
      )}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
      {...props}
    >
      {busy ? <LoaderCircle className="size-3.5 animate-spin" aria-hidden /> : icon}
      {children}
    </button>
  );
});
