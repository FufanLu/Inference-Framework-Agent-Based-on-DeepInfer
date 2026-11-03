import { forwardRef, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes } from "react";
import { cn } from "@/lib/cn";

const control =
  "w-full rounded-sm border border-ctrl bg-sheet text-ink placeholder:font-sans placeholder:text-ink-3 " +
  "transition-colors duration-150 hover:border-ink-3 focus:border-ink focus:outline-none focus-visible:outline-none " +
  "focus:ring-1 focus:ring-ink disabled:cursor-not-allowed disabled:bg-wash disabled:text-ink-3 " +
  "aria-[invalid=true]:border-bad aria-[invalid=true]:ring-bad";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input(
  { className, ...props },
  ref,
) {
  return <input ref={ref} className={cn(control, "h-8 px-2.5 text-sm", className)} {...props} />;
});

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function Textarea(
  { className, ...props },
  ref,
) {
  return <textarea ref={ref} className={cn(control, "block resize-y px-2.5 py-2 text-sm leading-relaxed", className)} {...props} />;
});

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(function Select(
  { className, children, ...props },
  ref,
) {
  return (
    <select ref={ref} className={cn(control, "h-8 cursor-pointer appearance-auto px-2 text-sm", className)} {...props}>
      {children}
    </select>
  );
});

interface FieldProps {
  label: string;
  htmlFor?: string;
  hint?: ReactNode;
  error?: string | null;
  children: ReactNode;
  className?: string;
  aside?: ReactNode;
}

export function Field({ label, htmlFor, hint, error, children, className, aside }: FieldProps) {
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={htmlFor} className="cond text-xs font-semibold text-ink-2">
          {label}
        </label>
        {aside}
      </div>
      {children}
      {error ? (
        <p className="text-xs text-bad" role="alert">
          {error}
        </p>
      ) : hint ? (
        <p className="text-xs text-ink-3">{hint}</p>
      ) : null}
    </div>
  );
}
