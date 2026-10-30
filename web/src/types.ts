// Hand-written mirrors of backend schemas (docs/plan/03-data-contracts.md). Names are identical.

export type Lang = "id" | "en";

export interface Health {
  ok: boolean;
  version: string;
  demo: boolean;
  ffmpeg: boolean;
  api_key_configured: boolean;
  models: Record<string, string>;
}

export type Stage =
  | "created" | "ingesting" | "transcribing" | "planning" | "segmenting" | "measuring"
  | "analyzing" | "auditing" | "designing" | "rendering" | "done" | "failed";

export type VerdictStatus = "PASS" | "PASS_WITH_WARNINGS" | "NEEDS_RERECORD" | "REJECT";
export type ExperimentType = "free_fall" | "pendulum" | "melting" | "unknown";

export interface Headline {
  experiment?: ExperimentType;
  quantity: string; // "g" | "period"
  value: number;
  sigma: number | null;
  ci95: [number, number];
  unit: string;
  evidence: string;
  formatted_en: string;
  formatted_id: string;
  ref?: number | null;
  rel_diff_pct?: number | null;
  period?: number;
  period0?: number;
  length_m?: number | null;
  length_source?: string | null;
  n_cycles?: number;
  scale_method?: string | null;
  [key: string]: unknown;
}

export interface RunState {
  run_id: string;
  created_at: string;
  updated_at: string;
  stage: Stage;
  stage_history: [Stage, string][];
  original_filename: string;
  lang: Lang;
  capture_fps: number | null;
  note: string | null;
  series_id: string | null;
  experiment: ExperimentType | null;
  verdict: VerdictStatus | null;
  headline: Headline | null;
  cost_usd: number;
  revise_rounds: number;
  warnings: string[];
  error: string | null;
  artifacts: Record<string, string>;
  demo: boolean;
}

export interface Quantity {
  value: number;
  unit: string;
  source_turn: number | null;
}

export interface ReferenceObject {
  concept: string;
  length_m: number | null;
  length_source: string;
  frame_hint_s: number | null;
}

export interface ExperimentSpec {
  experiment: ExperimentType;
  target_concept: string;
  target_alt_concepts: string[];
  reference: ReferenceObject | null;
  object_diameter_m: number | null;
  spoken_params: Record<string, Quantity>;
  analysis_window_s: [number, number];
  motion_axis: string;
  hypothesis: string | null;
  hypothesis_prediction: Quantity | null;
  language: Lang;
  setup_description: string;
  confidence: number;
  warnings: string[];
}

export interface GateResult {
  gate: "G1" | "G2" | "G3" | "G4" | "G5";
  name: string;
  status: "pass" | "fail" | "warn" | "skipped";
  hard: boolean;
  metrics: Record<string, number | string>;
  thresholds: Record<string, number | string>;
  message_en: string;
  message_id: string;
  remedy: "reanalyze" | "rerecord" | "none";
  diagnosis: string | null;
  evidence_id?: string | null;
}

export interface Claim {
  id: string;
  claim_type: string;
  quantity: string | null;
  value: number | null;
  ci95: [number, number] | null;
  unit: string | null;
  evidence: string[];
  statement_en: string;
  statement_id: string;
  text_en?: string | null; // rendered by the API (M10)
  text_id?: string | null;
}

export interface Evidence {
  id: string;
  tool: string;
  created_at: string;
  inputs: string[];
  data: Record<string, unknown>;
  summary: string;
}

export interface Verdict {
  status: VerdictStatus;
  unsupported_claims: string[];
  overreaching_statements: string[];
  gate_failures: string[];
  fix_instructions: string[];
  rerecord_tips_en: string[];
  rerecord_tips_id: string[];
  auditor_model: string;
  notes: string;
  overrides: string[];
}

export interface TranscriptTurn {
  index: number;
  text: string;
  start_ms: number | null;
  end_ms: number | null;
}

export interface Transcript {
  text: string;
  turns: TranscriptTurn[];
}

export interface DesignerProposal {
  title: string;
  changed_variable: string;
  controlled_variables: string[];
  new_values: Quantity[];
  prediction_en: string;
  prediction_id: string;
  prediction_en_rendered?: string | null;
  prediction_id_rendered?: string | null;
  protocol_steps_en: string[];
  protocol_steps_id: string[];
  safety_notes_en: string[];
  safety_notes_id: string[];
  predicted_values: { id: string; input: number; predicted: number; unit_out: string }[];
}

/** GET /api/runs/{id}: RunState plus what the run view needs (07 §1). */
export interface RunDetail extends RunState {
  spec: ExperimentSpec | null;
  calibration: Record<string, unknown> | null;
  gates: GateResult[];
  claims: Claim[];
  transcript: Transcript | null;
  verdict_detail: Verdict | null;
  next_experiment: DesignerProposal | null;
}

export interface Ledger {
  evidence: Evidence[];
  claims: Claim[];
}

export interface TrackPointView {
  frame: number;
  t: number;
  x_px: number;
  y_px: number;
  valid: boolean;
  flag: string;
}

export interface TrackView {
  fps: number;
  points: TrackPointView[];
  fit_curve: Record<string, number>[];
}

export interface SeriesState {
  series_id: string;
  created_at: string;
  experiment: "pendulum";
  run_ids: string[];
  status: "collecting" | "analyzing" | "done" | "failed";
  headline: Record<string, unknown> | null;
  results?: SeriesResult | null;
}

export interface SeriesPoint {
  run_id?: string;
  L: number;
  T0: number;
  sigma_T: number;
  sigma_L: number;
}

export interface SeriesResult {
  n_runs: number;
  points: SeriesPoint[];
  linear: { k: number; b: number; sigma_k: number; sigma_b: number; chi2_red: number };
  g: number;
  sigma_g: number;
  ci_g: [number, number];
  n: number;
  sigma_n: number;
  sqrt_law_supported: boolean;
  flags: string[];
  runs?: SeriesPoint[];
}

export type EventType =
  | "stage_started" | "stage_completed" | "progress" | "model_call" | "model_result"
  | "reasoning_summary" | "tool_call" | "tool_result" | "claim_added" | "gate_result"
  | "verdict" | "artifact" | "warning" | "error" | "done";

export interface FrameLabEvent {
  ts: string;
  run_id: string;
  seq: number;
  type: EventType;
  stage: string;
  [key: string]: unknown;
}
