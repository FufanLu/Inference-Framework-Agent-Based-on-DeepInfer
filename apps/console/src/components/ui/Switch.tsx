import { Switch as RadixSwitch } from "radix-ui";
import { useId } from "react";
import { cn } from "@/lib/cn";

interface SwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  /** Text next to the label describing the current state. */
  stateText?: string;
  disabled?: boolean;
  className?: string;
}

/** A rectangular two-state switch with a square thumb. */
export function Switch({ checked, onChange, label, stateText, disabled, className }: SwitchProps) {
  const id = useId();
  return (
    <div className={cn("inline-flex items-center gap-2", className)}>
      <RadixSwitch.Root
        id={id}
        checked={checked}
        onCheckedChange={onChange}
        disabled={disabled}
        className={cn(
          "relative h-[18px] w-[34px] shrink-0 rounded-sm border border-ctrl bg-sheet transition-colors duration-150",
          "data-[state=checked]:border-ink data-[state=checked]:bg-ink",
          "disabled:opacity-50",
        )}
      >
        <RadixSwitch.Thumb
          className={cn(
            "block size-3 translate-x-[2px] rounded-xs bg-ink-3 transition-transform duration-200 ease-[var(--ease-out-expo)]",
            "data-[state=checked]:translate-x-[18px] data-[state=checked]:bg-paper",
          )}
        />
      </RadixSwitch.Root>
      <label htmlFor={id} className="cursor-pointer text-sm text-ink select-none">
        {label}
        {stateText ? <span className="ml-1.5 text-ink-3">{stateText}</span> : null}
      </label>
    </div>
  );
}
