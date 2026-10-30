// Typed API client (docs/plan/07 §1–§2). Errors carry the backend's {error: {code, message}}.
import type {
  EventType,
  FrameLabEvent,
  Health,
  Ledger,
  RunDetail,
  RunState,
  SeriesState,
  TrackView,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let code = "http_error";
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      code = body?.error?.code ?? code;
      message = body?.error?.message ?? message;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, code, message);
  }
  return (await res.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const getHealth = () => request<Health>("/api/health");
export const listRuns = (limit = 20) => request<RunState[]>(`/api/runs?limit=${limit}`);
export const getRun = (id: string) => request<RunDetail>(`/api/runs/${id}`);
export const getLedger = (id: string) => request<Ledger>(`/api/runs/${id}/ledger`);
export const getTrack = (id: string) => request<TrackView>(`/api/runs/${id}/track`);
export const rerun = (id: string, body: { capture_fps?: number; lang?: string }) =>
  request<{ run_id: string }>(`/api/runs/${id}/rerun`, json(body));

export const listSeries = () => request<SeriesState[]>("/api/series");
export const getSeries = (id: string) => request<SeriesState>(`/api/series/${id}`);
export const createSeries = (runIds: string[]) =>
  request<SeriesState>("/api/series", json({ experiment: "pendulum", run_ids: runIds }));
export const addSeriesRun = (id: string, runId: string) =>
  request<SeriesState>(`/api/series/${id}/runs`, json({ run_id: runId }));
export const analyzeSeries = (id: string) =>
  request<{ status: string }>(`/api/series/${id}/analyze`, { method: "POST" });

export function artifactUrl(runId: string, name: string): string {
  return `/api/runs/${runId}/artifacts/${name}`;
}

/** The published file name of an artifact, from RunState.artifacts (name → relative path). */
export function artifactFile(state: RunState, name: string): string | null {
  const rel = state.artifacts[name];
  return rel ? rel.split("/").pop() ?? null : null;
}

/** Upload with progress (fetch has no upload progress). Resolves with the new run id. */
export function createRun(form: FormData, onProgress: (fraction: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/runs");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body: { run_id?: string; error?: { code: string; message: string } } = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* not JSON */
      }
      if (xhr.status === 201 && body.run_id) resolve(body.run_id);
      else reject(new ApiError(xhr.status, body.error?.code ?? "upload_failed",
                               body.error?.message ?? `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new ApiError(0, "network", "The upload could not reach the backend."));
    xhr.send(form);
  });
}

const EVENT_TYPES: EventType[] = [
  "stage_started", "stage_completed", "progress", "model_call", "model_result",
  "reasoning_summary", "tool_call", "tool_result", "claim_added", "gate_result", "verdict",
  "artifact", "warning", "error", "done",
];

/**
 * SSE (07 §2): past events are replayed first, then live ones. The server closes the stream
 * after `done` or `error`; the source is closed here too, so the browser does not reconnect.
 */
export function subscribe(runId: string, onEvent: (e: FrameLabEvent) => void): () => void {
  const source = new EventSource(`/api/runs/${runId}/events`);
  const handle = (msg: MessageEvent) => {
    const event = JSON.parse(msg.data) as FrameLabEvent;
    onEvent(event);
    if (event.type === "done" || event.type === "error") source.close();
  };
  for (const type of EVENT_TYPES) source.addEventListener(type, handle as EventListener);
  return () => source.close();
}
