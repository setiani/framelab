// Upload view (#/): hero, upload form, samples, recent runs, recording guide (docs/plan/07 §4.2).
import { ApiError, artifactUrl, createRun, getHealth, listRuns, listSamples, replaySample } from "../api";
import { errorText, experimentLabel, guideDialog, headlineText, verdictChip } from "../components/common";
import { clear, h, relativeTime } from "../dom";
import { getLang, t } from "../i18n";
import type { View } from "../main";
import type { Health, RunState, Sample } from "../types";

function hero(): HTMLElement {
  const video = h("video", {
    src: "/hero-freefall.webm", poster: "/hero-freefall.jpg", muted: true, loop: true,
    playsinline: true, "aria-label": t("hero_caption"),
  });
  video.muted = true;
  // A looping clip is decorative motion: respect the reduced-motion setting.
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    video.autoplay = true;
    void video.play().catch(() => undefined);
  } else {
    video.controls = true;
  }
  return h("section.hero-copy", null,
    h("h1", null, t("hero_title")),
    h("p", null, t("hero_body")),
    h("figure", { style: "margin:0" },
      h("div.viewfinder", null, video, h("span.vf-b")),
      h("figcaption.caption", null, t("hero_caption"))));
}

function uploadForm(): HTMLElement {
  const lang = getLang();
  let file: File | null = null;
  const fileInput = h("input", { type: "file", accept: "video/mp4,video/quicktime,.mp4,.mov",
                                 class: "visually-hidden", id: "video-file" });
  const fileLabel = h("span.file");
  const drop = h("label.dropzone", { for: "video-file" },
    h("strong", null, t("drop_here")), h("span.small.muted", null, t("drop_hint")), fileLabel);
  const setFile = (f: File | null) => {
    file = f;
    fileLabel.textContent = f ? `${f.name} · ${(f.size / 1e6).toFixed(1)} MB` : "";
    submit.disabled = !f;
  };
  fileInput.addEventListener("change", () => setFile(fileInput.files?.[0] ?? null));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const f = e.dataTransfer?.files?.[0];
    if (f) setFile(f);
  });

  const langSeg = h("div.seg", { role: "radiogroup", "aria-label": t("lang_label") });
  for (const code of ["id", "en"]) {
    langSeg.append(h("label", null,
      h("input", { type: "radio", name: "lang", value: code, checked: code === lang }),
      h("span", null, code === "id" ? "Bahasa Indonesia" : "English")));
  }
  const fps = h("select.input", { name: "capture_fps", id: "fps" },
    h("option", { value: "" }, t("fps_auto")),
    h("option", { value: "120" }, "120 fps"), h("option", { value: "240" }, "240 fps"));
  const note = h("textarea.input", { name: "note", maxlength: 300, id: "note",
                                     placeholder: t("note_placeholder") });
  const bar = h("div");
  const progress = h("div.progress", { hidden: true, role: "progressbar", "aria-label": t("uploading") }, bar);
  const message = h("div", { role: "alert" });
  const submit = h("button.btn.primary", { type: "submit", disabled: true }, t("start"));
  const dlg = guideDialog();
  const paused = h("div.notice", { hidden: true, role: "status" });
  const fields = h("fieldset.plain");

  const form = h("form.panel.stack", { "aria-labelledby": "upload-title" },
    h("div.panel-head", null, h("h2#upload-title", null, t("upload_title")),
      h("button.btn.small", { type: "button", onclick: () => dlg.showModal() }, t("how_to_record"))),
    paused, fields);
  fields.append(drop, fileInput,
    h("div.field", null, h("span", null, t("lang_label")), langSeg),
    h("label.field", { for: "fps" }, h("span", null, t("fps_label")), fps),
    h("label.field", { for: "note" }, h("span", null, t("note_label")), note),
    progress, message, submit);
  form.append(dlg);

  // Live uploads can be off (demo, no API key, daily budget): say so instead of failing later.
  getHealth().then((health: Health) => {
    const up = health.uploads;
    if (!up || up.enabled) return;
    fields.disabled = true;
    drop.classList.add("off");
    paused.hidden = false;
    const toSamples = h("button.btn.small", { type: "button", onclick: () =>
      document.getElementById("samples-title")?.scrollIntoView({ behavior: "smooth", block: "start" }) },
      t("see_samples"));
    clear(paused, h("strong", null, t("uploads_off")),
      h("span", null, errorText(up.reason ?? "", up.message ?? "")),
      health.samples ? h("div", null, toSamples) : null);
  }).catch(() => undefined);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!file) return;
    const data = new FormData();
    data.append("video", file);
    data.append("lang", (form.querySelector("input[name=lang]:checked") as HTMLInputElement).value);
    if (fps.value) data.append("capture_fps", fps.value);
    if (note.value.trim()) data.append("note", note.value.trim());
    submit.disabled = true;
    progress.hidden = false;
    clear(message);
    try {
      const runId = await createRun(data, (f) => { bar.style.width = `${Math.round(f * 100)}%`; });
      location.hash = `#/run/${runId}`;
    } catch (err) {
      progress.hidden = true;
      submit.disabled = false;
      const e = err as ApiError;
      clear(message, h("div.error-box", null, errorText(e.code ?? "", e.message)));
    }
  });
  return form;
}

function recentRuns(): HTMLElement {
  const body = h("div", null, h("p.empty", null, "…"));
  const panel = h("section.panel", { "aria-labelledby": "recent-title" },
    h("div.panel-head", null, h("h2#recent-title", null, t("recent")),
      h("a.small", { href: "#/series" }, t("series_open"))),
    body);
  listRuns(20)
    .then((runs) => {
      if (!runs.length) return clear(body, h("p.empty", null, t("no_runs")));
      clear(body, runsTable(runs));
    })
    .catch(() => clear(body, h("p.empty", null, t("backend_down"))));
  return panel;
}

function runsTable(runs: RunState[]): HTMLElement {
  const lang = getLang();
  const rows = runs.map((r) => h("tr", null,
    h("td", null, h("a.file", { href: `#/run/${r.run_id}` }, r.original_filename),
      r.demo ? h("span.tiny.muted", null, t("sample_note")) : null),
    h("td.hide-sm.small", null, experimentLabel(r.experiment)),
    h("td", null, verdictChip(r)),
    h("td.num.small.res", null, headlineText(r)),
    h("td.hide-sm.small.muted", null, relativeTime(r.created_at, lang))));
  return h("table.runs", null,
    h("thead", null, h("tr", null, h("th", null, t("video")), h("th.hide-sm", null, ""),
      h("th", null, t("status")), h("th", null, t("result")), h("th.hide-sm", null, ""))),
    h("tbody", null, ...rows));
}

function sampleCard(sample: Sample): HTMLElement {
  const lang = getLang();
  const message = h("div.small", { role: "alert" });
  const watch = h("button.btn.primary.small", { type: "button" }, t("watch_replay"));
  watch.addEventListener("click", async () => {
    watch.disabled = true;
    try {
      const res = await replaySample(sample.key);
      location.hash = `#/run/${res.run_id}`;
    } catch (err) {
      watch.disabled = false;
      const e = err as ApiError;
      clear(message, h("span.error-text", null, e.code === "rate_limited" ? t("replay_limited") : e.message));
    }
  });
  const result = sample.verdict === "NEEDS_RERECORD" ? null : headlineText(sample);
  const kind = experimentLabel(sample.experiment);
  return h("article.sample", null,
    h("div.thumb", null, h("img", { src: artifactUrl(sample.run_id, "keyframe.png"), alt: "",
                                    loading: "lazy", width: 320, height: 180 })),
    h("div.sample-body", null,
      h("div.sample-meta", null,
        h("span.small.muted", null, kind === sample.title[lang] ? "" : kind),
        verdictChip({ verdict: sample.verdict, stage: "done" })),
      h("h3", null, sample.title[lang] ?? sample.key),
      h("p.small", null, sample.blurb[lang] ?? ""),
      result ? h("p.num.sample-result", null, result) : null,
      h("div.sample-actions", null, watch,
        h("a.small", { href: `#/run/${sample.run_id}` }, t("open_result"))),
      message));
}

function samplesSection(): HTMLElement {
  const grid = h("div.samples");
  const section = h("section", { "aria-labelledby": "samples-title", hidden: true },
    h("div.section-head", null, h("h2#samples-title", null, t("samples")),
      h("p.small.muted", null, t("samples_intro"))),
    grid);
  listSamples().then((samples) => {
    if (!samples.length) return;
    clear(grid, ...samples.map(sampleCard));
    section.hidden = false;
  }).catch(() => undefined);
  return section;
}

export function uploadView(): View {
  const el = h("div.stack.gap-lg", null,
    h("div.hero", null, hero(), uploadForm()),
    samplesSection(),
    recentRuns());
  return { el };
}
