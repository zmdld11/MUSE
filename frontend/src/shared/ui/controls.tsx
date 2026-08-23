import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "@/shared/utils/cn";

/** 图标按钮（播放器控制区主力控件；title 即 tooltip + aria-label） */
export function IconButton({
  title,
  disabled,
  variant = "ghost",
  size = "md",
  children,
  className,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  title: string;
  variant?: "ghost" | "accent";
  size?: "md" | "sm" | "lg";
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      disabled={disabled}
      className={cn(
        "inline-flex shrink-0 items-center justify-center rounded-md text-content-2 transition-all duration-150",
        "hover:bg-surface-2 hover:text-content-1 active:scale-95",
        "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
        "disabled:pointer-events-none disabled:opacity-30",
        size === "sm" && "h-7 w-7",
        size === "md" && "h-9 w-9",
        size === "lg" && "h-11 w-11",
        variant === "accent" &&
          "bg-accent text-[#05121c] hover:bg-accent-press hover:text-[#05121c] active:bg-accent-press",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

/** 带文字的按钮 */
export function Button({
  variant = "subtle",
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "subtle" | "accent";
}) {
  return (
    <button
      type="button"
      className={cn(
        "inline-flex items-center gap-2 rounded-md px-3.5 py-2 text-sm transition-all duration-150 active:scale-[0.98]",
        "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
        variant === "subtle" &&
          "bg-surface-1 text-content-1 stroke-[1.5] hover:bg-surface-2",
        variant === "accent" &&
          "bg-accent font-medium text-[#05121c] hover:bg-accent-press",
        "disabled:pointer-events-none disabled:opacity-40",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

/** 分段控件（视图切换等） */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div
      role="tablist"
      className="inline-flex items-center gap-0.5 rounded-lg bg-surface-1 p-0.5"
    >
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={value === o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            "rounded-md px-3 py-1.5 text-xs transition-all duration-150",
            value === o.value
              ? "bg-surface-2 text-content-1 shadow-sm"
              : "text-content-3 hover:text-content-2",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
