#!/usr/bin/env python3
"""
Section 1 — Build ticket records & feature blocks.

Join Tickets + Time-entries on (instance, ticketnumber) and produce one clean,
feature-ready record per ticket, computed once so the combo sweep never redoes
the expensive parts.

Per ticket we keep only feature-relevant fields:
    ticket_id, title, description, notes, issue_type, sub_issue_type,
    hours (coverage weight), touches
Identifiers, dates and person/instance IDs are deliberately NOT carried into
features (§1.3) — `instance` is kept for analysis/logging only and never enters
clustering.

Text-poor tickets (no title/description AND no notes) are dropped (§1.2); the
kept-vs-dropped counts are logged.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

import config as C


@dataclass
class TicketRecord:
    ticket_id: str          # stable key: f"{instance}::{ticketnumber}" (analysis-only)
    instance: str           # analysis-only — MUST NOT enter clustering
    title: str
    description: str
    notes: str              # all time-entry notes, concatenated chronologically
    issue_type: str
    sub_issue_type: str
    hours: float            # coverage weight
    touches: int            # number of time entries


def _clean_text(v) -> str:
    if v is None:
        return ""
    s = str(v)
    if s.strip().lower() in ("nan", "none", "null"):
        return ""
    return re.sub(r"\s+", " ", s).strip()


def _concat_notes(sub_df: pd.DataFrame) -> str:
    """Concatenate summarynotes + internalnotes across a ticket's time entries,
    ordered chronologically by startdatetime."""
    sub = sub_df.sort_values(TE := C.TimeEntryCols.startdatetime, kind="stable") \
        if C.TimeEntryCols.startdatetime in sub_df.columns else sub_df
    parts: list[str] = []
    for _, r in sub.iterrows():
        for col in (C.TimeEntryCols.summarynotes, C.TimeEntryCols.internalnotes):
            t = _clean_text(r.get(col))
            if t:
                parts.append(t)
    # De-dup consecutive identical fragments (common with copy-pasted updates)
    out: list[str] = []
    for p in parts:
        if not out or out[-1] != p:
            out.append(p)
    return " — ".join(out)


def build_records(
    tickets_xlsx: str = C.TICKETS_XLSX,
    time_entries_xlsx: str = C.TIME_ENTRIES_XLSX,
    *,
    limit: int | None = None,
    log=print,
) -> tuple[list[TicketRecord], dict]:
    """Join the two exports.

    Returns (records, meta). `meta` carries corpus hour totals — including the
    text-poor tickets dropped in §1.2 — because §2.6 coverage is a fraction of
    ALL engineer-hours (dropped tickets' hours count toward "Other")."""
    log(f"Loading tickets  … {tickets_xlsx}")
    tk = pd.read_excel(tickets_xlsx, sheet_name="Export", engine="openpyxl")
    log(f"  {len(tk):,} ticket rows")

    log(f"Loading time entries … {time_entries_xlsx}")
    te = pd.read_excel(time_entries_xlsx, sheet_name="Export", engine="openpyxl")
    log(f"  {len(te):,} time-entry rows")

    TC, EC = C.TicketCols, C.TimeEntryCols

    # ── normalise join keys to string on both sides ─────────────────────────
    for df, cols in ((tk, (TC.instance, TC.ticketnumber)), (te, (EC.instance, EC.ticketnumber))):
        for c in cols:
            df[c] = df[c].astype(str).str.strip()

    # ── notes + touches + hours from time entries, grouped per ticket ───────
    log("Aggregating time-entry notes per ticket …")
    notes_by_key: dict[tuple[str, str], str] = {}
    touches_by_key: dict[tuple[str, str], int] = {}
    hours_by_key: dict[tuple[str, str], float] = {}
    for (inst, tn), grp in te.groupby([EC.instance, EC.ticketnumber], sort=False):
        key = (inst, tn)
        notes_by_key[key] = _concat_notes(grp)
        touches_by_key[key] = len(grp)
        hours_by_key[key] = float(pd.to_numeric(grp[EC.hours], errors="coerce").fillna(0).sum())

    # ── one record per ticket row ────────────────────────────────────────────
    # Access columns as arrays: the Power-BI headers contain '[' / ']', which
    # itertuples/namedtuple would mangle into positional fields.
    def col(name):
        return tk[name].tolist() if name in tk.columns else [None] * len(tk)

    inst_c = [str(x).strip() for x in col(TC.instance)]
    tn_c = [str(x).strip() for x in col(TC.ticketnumber)]
    title_c = col(TC.title)
    desc_c = col(TC.description)
    it_c = col(TC.issue_type)
    st_c = col(TC.sub_issue_type)
    hours_c = pd.to_numeric(pd.Series(col(TC.hours)), errors="coerce").fillna(0.0).tolist()

    records: list[TicketRecord] = []
    kept = dropped = 0
    kept_hours = dropped_hours = 0.0
    seen: set[tuple[str, str]] = set()
    for i in range(len(tk)):
        if limit and kept >= limit:
            break
        inst, tn = inst_c[i], tn_c[i]
        key = (inst, tn)
        if key in seen:          # collapse duplicate ticket rows defensively
            continue
        seen.add(key)

        title = _clean_text(title_c[i])
        description = _clean_text(desc_c[i])
        notes = notes_by_key.get(key, "")

        # Hours: prefer the ticket's own SumTotalHoursWorked; fall back to the
        # summed time-entry hours when the ticket total is missing/zero.
        hours = float(hours_c[i]) or hours_by_key.get(key, 0.0)

        # §1.2 — drop text-poor tickets (their hours still count toward "Other")
        if not title and not description and not notes:
            dropped += 1
            dropped_hours += hours
            continue

        records.append(
            TicketRecord(
                ticket_id=f"{inst}::{tn}",
                instance=inst,
                title=title,
                description=description,
                notes=notes,
                issue_type=_clean_text(it_c[i]),
                sub_issue_type=_clean_text(st_c[i]),
                hours=float(hours),
                touches=touches_by_key.get(key, 1),
            )
        )
        kept += 1
        kept_hours += float(hours)

    total = kept + dropped
    log(f"  kept {kept:,} / dropped {dropped:,} text-poor "
        f"({dropped / total * 100:.1f}% dropped)" if total else "  no tickets")
    meta = {
        "kept": kept,
        "dropped": dropped,
        "kept_hours": round(kept_hours, 3),
        "dropped_hours": round(dropped_hours, 3),
        "total_hours": round(kept_hours + dropped_hours, 3),
    }
    return records, meta


# ── persistence ──────────────────────────────────────────────────────────────

def save_records(records: list[TicketRecord], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def load_records(path: str) -> list[TicketRecord]:
    out: list[TicketRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(TicketRecord(**json.loads(line)))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build ticket records (Section 1).")
    ap.add_argument("--tickets", default=C.TICKETS_XLSX)
    ap.add_argument("--time-entries", default=C.TIME_ENTRIES_XLSX)
    ap.add_argument("--out", default=f"{C.CACHE_DIR}/records.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap kept tickets (smoke test)")
    args = ap.parse_args()

    recs, meta = build_records(args.tickets, args.time_entries, limit=args.limit)
    save_records(recs, args.out)
    Path(args.out).with_name("records_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"→ wrote {len(recs):,} records to {args.out}  |  meta: {meta}")
