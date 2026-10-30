// Claim Ledger (07 §4.3): select a claim to see the evidence chain it rests on. The chain is
// the claim's evidence and, recursively, their inputs, as recorded by code in ledger.jsonl.
import { getLedger } from "../api";
import { h } from "../dom";
import { getLang, t } from "../i18n";
import type { Claim, Evidence, Ledger } from "../types";

export function chainOf(claim: Claim, ledger: Ledger): Evidence[] {
  const byId = new Map(ledger.evidence.map((e) => [e.id, e]));
  const seen = new Set<string>();
  const out: Evidence[] = [];
  const queue = [...claim.evidence];
  while (queue.length) {
    const id = queue.shift()!;
    if (seen.has(id)) continue;
    seen.add(id);
    const ev = byId.get(id);
    if (!ev) continue;
    out.push(ev);
    queue.push(...ev.inputs);
  }
  return out;
}

export function ledgerPanel(runId: string, onSelect: (ids: Set<string>) => void) {
  const list = h("div.claims", { role: "list" });
  const chain = h("div.chain", { "aria-live": "polite" });
  const el = h("section.panel", { "aria-labelledby": "ledger-title" },
    h("div.panel-head", null, h("h2#ledger-title", null, t("ledger")),
      h("span.tiny.muted", null, t("ledger_hint"))),
    list, chain);
  let claims: Claim[] = [];
  let ledger: Ledger | null = null;
  let selected: string | null = null;
  let shown = false;

  async function select(claim: Claim) {
    selected = claim.id;
    for (const b of list.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.id === claim.id));
    if (!ledger || !ledger.claims.some((c) => c.id === claim.id)) ledger = await getLedger(runId);
    const nodes = chainOf(claim, ledger);
    onSelect(new Set([claim.id, ...nodes.map((n) => n.id)]));
    chain.replaceChildren(h("h3", null, `${t("chain")}: ${claim.id}`),
      ...nodes.map((ev) => h("div.node", null,
        h("div", null, h("span.nid", null, ev.id), " ", h("span.ntool", null, ev.tool)),
        h("div.nsum", null, ev.summary),
        h("details", null, h("summary.tiny.muted", null, t("evidence")),
          h("pre", null, JSON.stringify(ev.data, null, 1).slice(0, 4000))))));
  }

  function update(next: Claim[]) {
    if (shown && next.length === claims.length && claims.every((c, i) => c.id === next[i].id)) return;
    shown = true;
    claims = next;
    ledger = null;
    const lang = getLang();
    if (!claims.length) {
      list.replaceChildren(h("p.empty", null, t("no_claims")));
      return;
    }
    list.replaceChildren(...claims.map((c) => {
      const btn = h("button.claim", { type: "button", "aria-pressed": String(c.id === selected), role: "listitem" },
        h("span.id", null, c.id),
        h("span", null, h("span", null, (lang === "id" ? c.text_id : c.text_en) ?? c.statement_en),
          h("div.type", null, `${c.claim_type} · ${c.evidence.join(", ")}`)));
      btn.dataset.id = c.id;
      btn.addEventListener("click", () => void select(c));
      return btn;
    }));
  }
  return { el, update };
}
