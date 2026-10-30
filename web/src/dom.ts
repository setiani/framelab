// Tiny DOM helpers. Components are functions returning { el, update } (docs/plan/07 §4.5).

type Child = Node | string | number | null | undefined | false;
type Attrs = Record<string, string | number | boolean | null | undefined | EventListener>;

/** h("div.panel#id", { onclick: fn, title: "x" }, children…) */
export function h<K extends keyof HTMLElementTagNameMap>(
  spec: K | `${K}.${string}` | `${K}#${string}`,
  attrs: Attrs | null = null,
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const [tagAndId, ...classes] = spec.split(".");
  const [tag, id] = tagAndId.split("#");
  const el = document.createElement(tag as K);
  if (id) el.id = id;
  if (classes.length) el.className = classes.join(" ");
  for (const [key, value] of Object.entries(attrs ?? {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2), value as EventListener);
    } else if (key === "class") {
      el.className = [el.className, String(value)].filter(Boolean).join(" ");
    } else if (value === true) {
      el.setAttribute(key, "");
    } else {
      el.setAttribute(key, String(value));
    }
  }
  append(el, children);
  return el;
}

export function append(el: Element, children: Child[]): void {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function clear(el: Element, ...children: Child[]): void {
  el.replaceChildren();
  append(el, children);
}

export function svg(tag: string, attrs: Record<string, string | number> = {}): SVGElement {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

export function relativeTime(iso: string, lang: "id" | "en"): string {
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  const rtf = new Intl.RelativeTimeFormat(lang, { numeric: "auto" });
  if (s < 60) return rtf.format(-Math.round(s), "second");
  if (s < 3600) return rtf.format(-Math.round(s / 60), "minute");
  if (s < 86400) return rtf.format(-Math.round(s / 3600), "hour");
  return rtf.format(-Math.round(s / 86400), "day");
}
