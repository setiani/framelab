import type { FrameLabEvent, Health, RunState } from "./types";

// Typed API helpers (docs/plan/07 §1). M0: health only; the rest lands in M10.

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export const getHealth = () => getJson<Health>("/api/health");

export const getRun = (runId: string) => getJson<RunState>(`/api/runs/${runId}`);

export function subscribe(runId: string, onEvent: (e: FrameLabEvent) => void): EventSource {
  const source = new EventSource(`/api/runs/${runId}/events`);
  source.onmessage = (msg) => onEvent(JSON.parse(msg.data) as FrameLabEvent);
  return source;
}
