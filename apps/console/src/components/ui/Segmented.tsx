import { ToggleGroup } from "radix-ui";
import { cn } from "@/lib/cn";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  /** Longer explanation, shown as the native tooltip. */
  title?: string;
  disabled?: boolean;
}

interface SegmentedProps<T extends string> {
  value: T | null;
  options: readonly SegmentedOption<T>[];
  onChange: (value: T) => void;
  label: string;
  disabled?: boolean;
  /** The value being applied, while a change is in flight. */
  pending?: T | null;
  size?: "sm" | "md";
  mono?: boolean;
  className?: string;
}

/** A row of ruled cells; the selected cell is inked. */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  disabled,
  pending,
  size = "md",
  mono = false,
  className,
}: SegmentedProps<T>) {
  return (
    <ToggleGroup.Root
      type="single"
      value={value ?? ""}
      onValueChange={(next) => {
        if (next) onChange(next as T);
      }}
      aria-label={label}
      disabled={disabled}
      className={cn("inline-flex rounded-sm border border-ctrl bg-sheet", className)}
    >
      {options.map((option, index) => {
        const isPending = pending === option.value;
        return (
          <ToggleGroup.Item
            key={option.value}
            value={option.value}
            title={option.title}
            disabled={option.disabled}
            className={cn(
              "relative whitespace-nowrap text-ink-2 transition-colors duration-150 ease-out",
              size === "sm" ? "h-6 px-2 text-xs" : "h-8 px-3 text-sm",
              mono && "font-mono text-[0.75rem] tracking-tight",
              index > 0 && "border-l border-rule",
              "hover:bg-wash hover:text-ink",
              "data-[state=on]:bg-ink data-[state=on]:text-paper data-[state=on]:hover:bg-ink",
              "disabled:cursor-not-allowed disabled:text-ink-3 disabled:hover:bg-transparent",
              "focus-visible:z-10",
              isPending && "bg-wash text-ink",
            )}
          >
            {option.label}
            {isPending ? (
              <span aria-hidden className="absolute inset-x-1 bottom-0.5 h-px animate-pulse bg-current" />
            ) : null}
          </ToggleGroup.Item>
        );
      })}
    </ToggleGroup.Root>
  );
}
