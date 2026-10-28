import "./styles.css";
import { getHealth } from "./api";
import { runView } from "./views/run";
import { seriesView } from "./views/series";
import { uploadView } from "./views/upload";

// Hash routing: #/ upload · #/run/<id> · #/series/<id> (docs/plan/07 §4)
const app = document.querySelector<HTMLDivElement>("#app")!;

function shell(content: HTMLElement): HTMLElement {
  const root = document.createElement("div");
  root.className = "shell";
  root.innerHTML = `
    <header class="top"><h1>FrameLab</h1><span class="tag">Every phone is a physics lab.</span></header>
    <div class="card mono" id="health">Checking backend…</div>`;
  root.appendChild(document.createElement("br"));
  root.appendChild(content);
  const footer = document.createElement("footer");
  footer.className = "models";
  footer.textContent = "Built with Meta SAM 3.1 · Muse Spark 1.3 · Muse Wonder · Muse Voice Transcribe";
  root.appendChild(footer);
  return root;
}

function route(): void {
  const hash = location.hash || "#/";
  const view = hash.startsWith("#/run/") ? runView() : hash.startsWith("#/series/") ? seriesView() : uploadView();
  app.replaceChildren(shell(view));
  const box = document.querySelector<HTMLDivElement>("#health")!;
  getHealth()
    .then((h) => {
      box.innerHTML = `<span class="status-dot ok"></span>backend v${h.version} · ffmpeg ${h.ffmpeg ? "ok" : "missing"} · API key ${h.api_key_configured ? "configured" : "missing"}${h.demo ? " · demo mode" : ""}`;
    })
    .catch(() => {
      box.innerHTML = `<span class="status-dot bad"></span>backend not reachable — run: uv run uvicorn framelab.api.main:app --reload`;
    });
}

window.addEventListener("hashchange", route);
route();
