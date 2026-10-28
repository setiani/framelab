// Hand-written mirrors of backend schemas (docs/plan/03-data-contracts.md). Keep names identical.
// M0 contains only what the skeleton uses; extend in M10.

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
  | "analyzing" | "auditing" | "rendering" | "designing" | "done" | "failed";

export type VerdictStatus = "PASS" | "PASS_WITH_WARNINGS" | "NEEDS_RERECORD" | "REJECT";

export interface RunState {
  run_id: string;
  created_at: string;
  updated_at: string;
  stage: Stage;
  original_filename: string;
  lang: "id" | "en";
  capture_fps: number | null;
  experiment: "free_fall" | "pendulum" | "melting" | "unknown" | null;
  verdict: VerdictStatus | null;
  headline: Record<string, unknown> | null;
  cost_usd: number;
  warnings: string[];
  artifacts: Record<string, string>;
  demo: boolean;
}

export interface FrameLabEvent {
  ts: string;
  run_id: string;
  seq: number;
  type: string;
  stage: string;
  [key: string]: unknown;
}
