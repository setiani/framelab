import "./styles.css";
import { getHealth } from "./api";
import { h } from "./dom";
import { getLang, setLang, t } from "./i18n";
import type { Lang } from "./types";
import { runView } from "./views/run";
import { seriesView } from "./views/series";
import { uploadView } from "./views/upload";

// Hash routing: #/ upload · #/run/<id> · #/series[/<id>] (docs/plan/07 §4)
export interface View {
  el: HTMLElement;
  destroy?: () => void;
}

const app = document.querySelector<HTMLDivElement>("#app")!;
let current: View | null = null;

/** The mark: a viewfinder bracket around the tracked object (the overlay's cyan). */
function mark(): SVGSVGElement {
  const ns = "http://www.w3.org/2000/svg";
  const s = document.createElementNS(ns, "svg");
  s.setAttribute("width", "26");
  s.setAttribute("height", "26");
  s.setAttribute("viewBox", "0 0 26 26");
  s.setAttribute("aria-hidden", "true");
  s.innerHTML = `
    <path d="M2 8V2h6M18 2h6v6M24 18v6h-6M8 24H2v-6" fill="none" stroke="#142027" stroke-width="2"/>
    <circle cx="13" cy="15" r="4.5" fill="#00b8d9"/>
    <path d="M13 4.5v3.2M13 6.1" stroke="#00b8d9" stroke-width="1.6" stroke-linecap="round" opacity=".55"/>`;
  return s;
}

function topbar(path: string): HTMLElement {
  const lang = getLang();
  const status = h("span.status", { role: "status" }, h("span.dot"), "…");
  getHealth()
    .then((health) => {
      status.replaceChildren(h("span.dot.ok"), h("span.label", null, health.demo ? t("demo") : t("backend_ok")));
      status.title = `FrameLab ${health.version}`;
    })
    .catch(() => status.replaceChildren(h("span.dot.bad"), h("span.label", null, t("backend_down"))));
  const toggle = h("div.lang-toggle", { role: "group", "aria-label": "Language" });
  for (const code of ["id", "en"] as Lang[]) {
    toggle.append(
      h("button", {
        type: "button",
        "aria-pressed": String(code === lang),
        onclick: () => {
          setLang(code);
          route();
        },
      }, code.toUpperCase()),
    );
  }
  const link = (href: string, label: string, active: boolean) =>
    h("a", { href, "aria-current": active ? "page" : null }, label);
  return h(
    "header.topbar",
    null,
    h("a.brand", { href: "#/" }, mark(), "FrameLab"),
    h("nav.nav", { "aria-label": "Main" },
      link("#/", t("nav_new"), path === "upload"),
      link("#/series", t("nav_series"), path === "series")),
    h("span.spacer"),
    status,
    toggle,
  );
}

function route(): void {
  current?.destroy?.();
  const hash = location.hash || "#/";
  let kind = "upload";
  if (hash.startsWith("#/run/")) {
    kind = "run";
    current = runView(decodeURIComponent(hash.slice(6)));
  } else if (hash.startsWith("#/series")) {
    kind = "series";
    const id = hash.slice(9);
    current = seriesView(id ? decodeURIComponent(id) : null);
  } else {
    current = uploadView();
  }
  document.documentElement.lang = getLang();
  const main = h("main", { id: "content" }, current.el);
  const footer = h("footer.credits", null,
    "Built with Meta SAM 3.1, Muse Spark 1.3 and Muse Voice Transcribe. "
    + "Every number in a report is copied from evidence computed by code.");
  app.replaceChildren(topbar(kind), main, footer);
  window.scrollTo({ top: 0 });
}

window.addEventListener("hashchange", route);
route();
