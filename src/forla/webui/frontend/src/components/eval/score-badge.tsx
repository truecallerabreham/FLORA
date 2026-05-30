/**
 * ScoreBadge — colored score indicator for eval results (0–10 scale).
 */

import { cn } from "@/lib/utils";

interface ScoreBadgeProps {
  score: number;
  max?: number;
  className?: string;
}

function scoreColor(score: number, max: number): string {
  const pct = score / max;
  if (pct >= 0.8) return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
  if (pct >= 0.6) return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
  if (pct >= 0.4) return "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200";
  return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
}

export function ScoreBadge({ score, max = 10, className }: ScoreBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-semibold tabular-nums",
        scoreColor(score, max),
        className
      )}
    >
      {score.toFixed(1)}/{max}
    </span>
  );
}

export function StatusBadge({
  status,
  className,
}: {
  status: string;
  className?: string;
}) {
  const colors: Record<string, string> = {
    completed: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
    running: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
    pending: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200",
    error: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
    cancelled: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-semibold capitalize",
        colors[status] || colors.pending,
        className
      )}
    >
      {status}
    </span>
  );
}
