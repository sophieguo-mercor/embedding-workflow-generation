#!/usr/bin/env python3
"""
End-to-end orchestrator for the embedding-based workflow-generation pipeline
(ENT-2261).

Stages
------
1. records   — join Tickets + Time-entries → one feature-ready record/ticket (§1)
2. sweep     — run the field-combo sweep: embed, cluster, name, categorize,
               score coverage + LLM coherence per combo (§2). Writes side-by-side
               summary for human review (§3).
   << human reviews results/sweep_summary.json + results/combos/*.json, picks combo* >>
3. stability — reseed combo* 2–3× and check ARI (§4)
4. finalize  — PII-gate + freeze combo* as versioned taxonomy.md/clusters.jsonl/
               metrics.json in Name|Category|Description shape (§5)

Usage
-----
    export OPENAI_API_KEY=...   ANTHROPIC_API_KEY=...
    pip install -r requirements.txt

    # Smoke test end-to-end with NO API spend (stub embeddings/LLM, tiny sample):
    python run.py --dry-run --limit 500

    # Real sweep:
    python run.py --stage sweep

    # After a human picks a combo, finalize it (runs stability first):
    python run.py --stage finalize --select notes_weighted
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import config as C


def log(msg: str, *, section: bool = False) -> None:
    if section:
        print(f"\n{'─'*64}\n  {msg}\n{'─'*64}", flush=True)
    else:
        print(f"  {msg}", flush=True)


def _records_paths():
    return f"{C.CACHE_DIR}/records.jsonl", f"{C.CACHE_DIR}/records_meta.json"


def stage_records(args) -> tuple[list, dict]:
    from records import build_records, save_records
    log("Stage 1 — Build ticket records", section=True)
    recs, meta = build_records(args.tickets, args.time_entries, limit=args.limit, log=log)
    rpath, mpath = _records_paths()
    save_records(recs, rpath)
    Path(mpath).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"→ {len(recs):,} records; meta={meta}")
    return recs, meta


def load_or_build_records(args):
    from records import load_records
    rpath, mpath = _records_paths()
    if not args.rebuild_records and Path(rpath).exists() and Path(mpath).exists():
        log(f"Using cached records ({rpath})")
        return load_records(rpath), json.loads(Path(mpath).read_text())
    return stage_records(args)


def stage_sweep(args):
    from sweep import run_sweep
    recs, meta = load_or_build_records(args)
    log("Stage 2 — Field-combo sweep", section=True)
    combos = {args.combo: C.WEIGHT_VECTORS[args.combo]} if args.combo else None
    run_sweep(recs, meta["total_hours"], combos=combos, dry_run=args.dry_run, log=log)
    log(f"\nHuman review next: {C.RESULTS_DIR}/sweep_summary.json + "
        f"{C.RESULTS_DIR}/combos/*.json → pick combo* → "
        f"`python run.py --stage finalize --select <combo>`")


def _load_combo_artifact(name: str) -> dict:
    p = Path(f"{C.RESULTS_DIR}/combos/{name}.json")
    if not p.exists():
        raise SystemExit(f"No combo artifact at {p}. Run the sweep first.")
    return json.loads(p.read_text(encoding="utf-8"))


def stage_stability(args):
    from stability import stability_check
    if not args.select:
        raise SystemExit("--select <combo> is required for the stability stage.")
    recs, _ = load_or_build_records(args)
    art = _load_combo_artifact(args.select)
    log("Stage 3 — Stability check", section=True)
    return stability_check(recs, art, dry_run=args.dry_run, log=log)


def stage_finalize(args):
    from stability import stability_check
    from finalize import finalize
    if not args.select:
        raise SystemExit("--select <combo> is required for the finalize stage.")
    recs, meta = load_or_build_records(args)
    art = _load_combo_artifact(args.select)

    log("Stage 3 — Stability check", section=True)
    stab = None if args.skip_stability else stability_check(recs, art, dry_run=args.dry_run, log=log)

    log("Stage 4 — Finalize (PII gate + freeze)", section=True)
    finalize(art, meta, stability=stab, log=log)


STAGES = {
    "records": lambda a: stage_records(a),
    "sweep": stage_sweep,
    "stability": stage_stability,
    "finalize": stage_finalize,
    "all": None,  # handled below
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGES), default="sweep")
    ap.add_argument("--tickets", default=C.TICKETS_XLSX)
    ap.add_argument("--time-entries", default=C.TIME_ENTRIES_XLSX)
    ap.add_argument("--limit", type=int, default=None, help="cap kept tickets (smoke test)")
    ap.add_argument("--combo", default=None, help="run only this one combo in the sweep")
    ap.add_argument("--select", default=None, help="combo* for stability/finalize")
    ap.add_argument("--dry-run", action="store_true",
                    help="stub embeddings + LLM (zero API spend)")
    ap.add_argument("--rebuild-records", action="store_true")
    ap.add_argument("--skip-stability", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if args.stage == "all":
        stage_sweep(args)
        if args.select:
            stage_finalize(args)
    else:
        STAGES[args.stage](args)
    log(f"\nDone in {time.time()-t0:.0f}s", section=True)


if __name__ == "__main__":
    main()
