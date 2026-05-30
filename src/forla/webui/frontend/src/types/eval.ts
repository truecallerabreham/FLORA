/**
 * TypeScript types for Forla persistence, runs, and evaluation.
 * Aligned with store/_models.py DB tables.
 */

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

export interface Run {
  id: string;
  run_type: "agent" | "orchestrator" | "eval_task";
  agent_name: string;
  model?: string;
  status: "completed" | "error" | "cancelled";
  finish_reason?: string;
  task_input?: string;
  duration_ms: number;
  tokens_input: number;
  tokens_output: number;
  llm_calls: number;
  tool_calls: number;
  cost_estimate?: number;
  trace_id?: string;
  tags?: string[];
  parent_run_id?: string;
  file_path?: string;
  created_at: string;
}

export interface RunData {
  run_id: string;
  run_type: string;
  agent_name: string;
  model?: string;
  response: Record<string, any>;
}

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

export interface Dataset {
  id: string;
  name: string;
  version: string;
  description: string;
  source: "user" | "builtin" | "generated";
  categories?: string[];
  default_eval_criteria?: string[];
  task_count: number;
  metadata?: Record<string, any>;
  tasks?: EvalTask[];
  created_at: string;
  updated_at: string;
}

export interface EvalTask {
  id: string;
  dataset_id: string;
  name: string;
  input: string;
  expected_output?: string;
  category: string;
  eval_criteria?: string[];
  rubric?: Record<string, any>;
  metadata?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface BuiltinDataset {
  name: string;
  description: string;
  task_count: number;
  categories?: string[];
}

// ---------------------------------------------------------------------------
// Target Configs
// ---------------------------------------------------------------------------

export interface TargetConfig {
  id: string;
  name: string;
  target_type: "forla_agent" | "claude_code" | "discovered_agent";
  config?: Record<string, any>;
  entity_id?: string;
  description: string;
  created_at: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// Eval Runs
// ---------------------------------------------------------------------------

export interface EvalRun {
  id: string;
  dataset_id: string;
  dataset_name: string;
  status: "pending" | "running" | "completed" | "error" | "cancelled";
  target_ids?: string[];
  target_names?: string[];
  judge_type?: string;
  judge_config?: Record<string, any>;
  total_tasks: number;
  completed_tasks: number;
  current_target?: string;
  current_task?: string;
  error_message?: string;
  file_path?: string;
  started_at?: string;
  completed_at?: string;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Eval Results
// ---------------------------------------------------------------------------

export interface EvalResult {
  id: string;
  eval_run_id: string;
  run_id?: string;
  task_id: string;
  target_name: string;
  overall_score: number;
  dimensions?: Record<string, number>;
  reasoning?: Record<string, string>;
  success: boolean;
  error?: string;
  duration_ms: number;
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  iterations: number;
  tool_calls: number;
  created_at: string;
}
