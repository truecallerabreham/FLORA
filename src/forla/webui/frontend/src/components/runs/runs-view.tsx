/**
 * RunsView — displays persisted agent/orchestrator run history with filters.
 */

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { StatusBadge } from "@/components/eval/score-badge";
import { RunDetail } from "@/components/runs/run-detail";
import { evalApiClient } from "@/services/eval-api";
import {
  Clock,
  Cpu,
  Trash2,
  ArrowLeft,
  Search,
  RefreshCw,
  Zap,
} from "lucide-react";
import type { Run } from "@/types/eval";

type RunTypeFilter = "all" | "agent" | "orchestrator" | "eval_task";

export function RunsView() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<Run | null>(null);
  const [filter, setFilter] = useState<RunTypeFilter>("all");
  const [searchQuery, setSearchQuery] = useState("");

  const loadRuns = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params: Record<string, any> = { limit: 100 };
      if (filter !== "all") params.run_type = filter;
      const data = await evalApiClient.listRuns(params);
      setRuns(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load runs");
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  const handleDelete = async (run: Run) => {
    try {
      await evalApiClient.deleteRun(run.id);
      setRuns((prev) => prev.filter((r) => r.id !== run.id));
      if (selectedRun?.id === run.id) setSelectedRun(null);
    } catch (e) {
      console.error("Delete failed:", e);
    }
  };

  const filteredRuns = runs.filter((r) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      r.agent_name.toLowerCase().includes(q) ||
      r.task_input?.toLowerCase().includes(q) ||
      r.model?.toLowerCase().includes(q)
    );
  });

  // Detail view
  if (selectedRun) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex items-center gap-2 p-3 border-b">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSelectedRun(null)}
          >
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back
          </Button>
          <span className="text-sm font-medium">
            {selectedRun.agent_name}
          </span>
          <StatusBadge status={selectedRun.status} />
        </div>
        <div className="flex-1 overflow-auto">
          <RunDetail run={selectedRun} />
        </div>
      </div>
    );
  }

  // List view
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 p-3 border-b">
        <h2 className="text-sm font-semibold">Runs</h2>

        {/* Type filter */}
        <div className="flex items-center gap-1 ml-2">
          {(["all", "agent", "orchestrator", "eval_task"] as const).map(
            (t) => (
              <Button
                key={t}
                variant={filter === t ? "default" : "ghost"}
                size="sm"
                className="h-7 text-xs"
                onClick={() => setFilter(t)}
              >
                {t === "all" ? "All" : t === "eval_task" ? "Eval" : t}
              </Button>
            )
          )}
        </div>

        {/* Search */}
        <div className="relative ml-auto max-w-xs flex-1">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            placeholder="Search runs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="h-7 pl-7 text-xs"
          />
        </div>

        <Button variant="ghost" size="sm" className="h-7" onClick={loadRuns}>
          <RefreshCw className="h-3.5 w-3.5" />
        </Button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {loading ? (
          <div className="flex items-center justify-center h-32 text-sm text-muted-foreground">
            Loading runs...
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-32 gap-2">
            <p className="text-sm text-destructive">{error}</p>
            <Button variant="outline" size="sm" onClick={loadRuns}>
              Retry
            </Button>
          </div>
        ) : filteredRuns.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-sm text-muted-foreground">
            {runs.length === 0
              ? "No persisted runs yet. Use persist=True when calling agent.run() or orchestrator.run()."
              : "No runs match your search."}
          </div>
        ) : (
          <div className="divide-y">
            {filteredRuns.map((run) => (
              <RunRow
                key={run.id}
                run={run}
                onSelect={() => setSelectedRun(run)}
                onDelete={() => handleDelete(run)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function RunRow({
  run,
  onSelect,
  onDelete,
}: {
  run: Run;
  onSelect: () => void;
  onDelete: () => void;
}) {
  const dur =
    run.duration_ms >= 1000
      ? `${(run.duration_ms / 1000).toFixed(1)}s`
      : `${run.duration_ms}ms`;

  const totalTokens = run.tokens_input + run.tokens_output;
  const timeAgo = formatTimeAgo(run.created_at);

  return (
    <div
      className="flex items-center gap-3 px-4 py-2.5 hover:bg-muted/50 cursor-pointer group"
      onClick={onSelect}
    >
      {/* Left: type badge + name */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-[10px] px-1.5 py-0 shrink-0">
            {run.run_type}
          </Badge>
          <span className="text-sm font-medium truncate">
            {run.agent_name}
          </span>
          <StatusBadge status={run.status} />
        </div>
        {run.task_input && (
          <p className="text-xs text-muted-foreground truncate mt-0.5">
            {run.task_input}
          </p>
        )}
      </div>

      {/* Right: stats */}
      <div className="flex items-center gap-3 text-xs text-muted-foreground shrink-0">
        {run.model && (
          <span className="hidden sm:inline truncate max-w-[120px]">
            {run.model}
          </span>
        )}
        <span className="flex items-center gap-1">
          <Clock className="h-3 w-3" />
          {dur}
        </span>
        <span className="flex items-center gap-1">
          <Zap className="h-3 w-3" />
          {totalTokens.toLocaleString()}
        </span>
        <span className="flex items-center gap-1">
          <Cpu className="h-3 w-3" />
          {run.llm_calls}
        </span>
        <span className="text-muted-foreground/60">{timeAgo}</span>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100"
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
        >
          <Trash2 className="h-3 w-3" />
        </Button>
      </div>
    </div>
  );
}

function formatTimeAgo(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diffMs = now - then;
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}
