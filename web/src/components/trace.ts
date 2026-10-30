// Agent trace (07 §4.3): model calls, tool calls with short results, claims, gates and the
// verdict, live from SSE. Following can be paused; it pauses itself when you scroll up.
import { h } from "../dom";
import { fmt, plain } from "../format";
import { getLang, t, type Key } from "../i18n";
import type { FrameLabEvent } from "../types";

const MODEL_NAME: Record<string, string> = {
  "muse-spark-1.3": "Muse Spark", "muse-wonder-1.0": "Muse Wonder", "sam-3.1": "SAM 3.1",
};

function brief(value: unknown, max = 140): string {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text && text.length > max ? `${text.slice(0, max - 1)}…` : text ?? "";
}

export function trace() {
  const list = h("div.trace", { role: "log", "aria-live": "off" });
  const follow = h("button.btn.small", { type: "button", "aria-pressed": "true" }, t("follow"));
  const count = h("span.tiny.muted.num");
  const el = h("section.panel", { "aria-labelledby": "trace-title" },
    h("div.panel-head", null, h("h2#trace-title", null, t("trace")),
      h("span", { style: "display:flex;gap:10px;align-items:center" }, count, follow)),
    list);
  let following = true;
  let t0: number | null = null;
  let n = 0;
  const calls = new Map<string, HTMLElement>();
  const openTools: HTMLElement[] = [];
  const setFollow = (on: boolean) => {
    following = on;
    follow.setAttribute("aria-pressed", String(on));
    follow.textContent = on ? t("follow") : t("paused");
    if (on) list.scrollTop = list.scrollHeight;
  };
  follow.addEventListener("click", () => setFollow(!following));
  list.addEventListener("wheel", (e) => { if (e.deltaY < 0 && following) setFollow(false); }, { passive: true });
  list.addEventListener("touchmove", () => following && setFollow(false), { passive: true });

  const row = (kind: string, icon: string, ts: string, ...body: (Node | string | null)[]) => {
    const time = new Date(ts).getTime();
    t0 ??= time;
    const s = Math.max(0, (time - t0) / 1000);
    const stamp = `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
    return h(`div.ev.${kind}`, null, h("span.ts", null, stamp), h("span.ic", { "aria-hidden": "true" }, icon),
      h("div.body", null, ...body));
  };

  function push(e: FrameLabEvent): void {
    const lang = getLang();
    let node: HTMLElement | null = null;
    switch (e.type) {
      case "stage_started":
        node = row("stage", "▸", e.ts, h("strong", null, t(`stage_${e.stage}` as Key)));
        break;
      case "progress":
        node = row("info", "·", e.ts, String(e.message ?? ""));
        break;
      case "model_call": {
        const model = String(e.model ?? "");
        node = row("model", "M", e.ts, h("span.tag", null, MODEL_NAME[model] ?? model), " ",
          String(e.purpose ?? ""), h("div.meta", null, "…"));
        calls.set(String(e.call_id), node);
        break;
      }
      case "model_result": {
        const target = calls.get(String(e.call_id));
        const usage = (e.usage ?? {}) as Record<string, number>;
        const meta = [
          fmt(Number(e.latency_s ?? 0), null, "s", lang),
          `$${plain(Number(e.cost_usd ?? 0), 4, lang)}`,
          usage.input ? `${usage.input}→${usage.output ?? 0} tok` : "",
          e.ok === false ? "error" : "",
        ].filter(Boolean).join("  ");
        const metaEl = target?.querySelector(".meta");
        if (metaEl) metaEl.textContent = meta;
        break;
      }
      case "tool_call":
        node = row("tool", "ƒ", e.ts, h("code", null, String(e.tool)), " ",
          h("span.tiny.muted", null, brief(e.args, 120)), h("div.meta", null, "…"));
        node.dataset.tool = String(e.tool);
        openTools.push(node);
        break;
      case "tool_result": {
        const idx = openTools.findIndex((r) => r.dataset.tool === e.tool);
        const target = idx >= 0 ? openTools.splice(idx, 1)[0] : null;
        const ids = (e.evidence_ids ?? []) as string[];
        const meta = `${e.ok ? "ok" : "rejected"}: ${brief(e.summary, 160)}${ids.length ? `  [${ids.join(", ")}]` : ""}`;
        if (target) {
          target.querySelector(".meta")!.textContent = meta;
          target.dataset.evidence = ids.join(" ");
          if (!e.ok) target.classList.add("fail");
        }
        break;
      }
      case "claim_added":
        node = row("added", "C", e.ts, h("code", null, String(e.claim_id)), " ", String(e.claim_type ?? ""),
          h("div.meta", null, ((e.evidence ?? []) as string[]).join(", ")));
        node.dataset.evidence = [String(e.claim_id), ...((e.evidence ?? []) as string[])].join(" ");
        break;
      case "gate_result":
        node = row("gate", "G", e.ts, h("code", null, String(e.gate)), " ", String(e.status),
          e.diagnosis ? h("span.tiny.muted", null, ` ${e.diagnosis}`) : null);
        if (e.evidence_id) node.dataset.evidence = String(e.evidence_id);
        break;
      case "verdict":
        node = row("added", "V", e.ts, h("strong", null, `${e.final ? "Final verdict" : "Auditor"}: ${e.status}`),
          e.model ? h("span.tiny.muted", null, ` ${MODEL_NAME[String(e.model)] ?? e.model}`) : null);
        break;
      case "warning":
        node = row("warn", "!", e.ts, String(e.message ?? ""));
        break;
      case "error":
        node = row("err", "×", e.ts, String(e.message ?? ""));
        break;
      case "done":
        node = row("stage", "■", e.ts, h("strong", null, t("stage_done")),
          h("span.tiny.muted.num", null, `  $${plain(Number(e.cost_usd ?? 0), 3, lang)}`));
        break;
      default:
        return;
    }
    if (node) {
      list.append(node);
      n += 1;
      count.textContent = `${n} ${t("events")}`;
      if (following) list.scrollTop = list.scrollHeight;
    }
  }

  /** Mark the rows that produced any of these ledger ids (claim → evidence chain). */
  function highlight(ids: Set<string>): void {
    let first: HTMLElement | null = null;
    for (const r of list.querySelectorAll<HTMLElement>(".ev")) {
      const hit = (r.dataset.evidence ?? "").split(" ").some((id) => id && ids.has(id));
      r.classList.toggle("hl", hit);
      if (hit && !first) first = r;
    }
    if (first && ids.size) {
      setFollow(false);
      list.scrollTop = first.offsetTop - list.offsetTop - 8;
    }
  }
  return { el, push, highlight };
}
