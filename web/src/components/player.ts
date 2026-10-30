// Player (07 §4.3): the annotated overlay when it exists, else the processed input video,
// with a frame scrubber and a frame/time readout.
import { artifactUrl } from "../api";
import { h } from "../dom";
import { fmt } from "../format";
import { getLang, t } from "../i18n";
import type { RunState } from "../types";

export function player(runId: string) {
  const video = h("video", { controls: true, playsinline: true, preload: "metadata", muted: true });
  const range = h("input", { type: "range", min: 0, max: 1000, value: 0, step: 1, "aria-label": t("frame") });
  const readout = h("span.readout.num");
  const hint = h("p.tiny.muted", { style: "margin-top:8px" });
  const el = h("section.panel.player", { "aria-label": t("video") },
    h("div.panel-head", null, h("h2", null, t("video"))), video,
    h("div.scrub", null, range, readout), hint);
  let fps = 30;
  let src = "";
  const show = () => {
    const tt = video.currentTime || 0;
    readout.textContent = `${t("frame")} ${Math.round(tt * fps)} · ${fmt(tt, null, "s", getLang())}`;
    if (video.duration) range.value = String(Math.round((tt / video.duration) * 1000));
  };
  video.addEventListener("timeupdate", show);
  video.addEventListener("loadedmetadata", show);
  range.addEventListener("input", () => {
    if (!video.duration) return;
    video.pause();
    video.currentTime = (Number(range.value) / 1000) * video.duration;
  });
  // Arrow keys on the slider step one frame at a time.
  range.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    video.pause();
    video.currentTime = Math.max(0, video.currentTime + (e.key === "ArrowRight" ? 1 : -1) / fps);
  });
  function update(state: RunState, trackFps?: number) {
    if (trackFps) fps = trackFps;
    const name = state.artifacts.overlay ? "overlay.mp4" : state.artifacts.video ? "video.mp4" : null;
    hint.textContent = name === "overlay.mp4" ? t("overlay_hint") : name ? t("original_hint") : "";
    if (name) {
      const next = artifactUrl(runId, name);
      if (next !== src) {
        src = next;
        video.src = next;
      }
    }
    el.hidden = !name;
  }
  return { el, update };
}
