#!/usr/bin/env python3
"""
End-to-end orchestrator for the workflow-generation pipeline. Two clustering
methods share one spine and are selected with --method:
  * kmeans  (ENT-2261, default) — field-combo sweep, k chosen by silhouette.
  * hdbscan (ENT-2289)          — UMAP + HDBSCAN density sweep, + §1.5 normalization.

Stages
------
1.  records   — join Tickets + Time-entries → one feature-ready record/ticket (§1)
1.5 normalize — (hdbscan) LLM intent normalization → normalized_issue/_resolution,
                batched + cached in cache/normalize.jsonl (§1.5)
2.  sweep      — embed, cluster, name, categorize, score coverage + LLM coherence
                per combo/config (§2). --method picks kmeans or hdbscan. Writes a
                side-by-side summary for human review (§3).
    << human reviews the summary + per-combo/config artifacts, picks the winner >>
3.  stability  — kmeans: reseed 2–3× and check ARI (§4). hdbscan: bootstrap/Jaccard
                (not yet wired — raises with guidance).
4.  finalize   — PII-gate + freeze the winner as versioned taxonomy.md/clusters.jsonl/
                metrics.json in Name|Category|Description shape (§5). kmeans only so far.

Usage
-----
    export OPENAI_API_KEY=...   ANTHROPIC_API_KEY=...
    pip install -r requirements.txt

    # Smoke test end-to-end with NO API spend (stub embeddings/LLM, tiny sample):
    python run.py --dry-run --limit 500

    # k-means (ENT-2261): sweep, then finalize a human-picked combo (runs stability):
    python run.py --stage sweep
    python run.py --stage finalize --select notes_weighted

    # HDBSCAN (ENT-2289): normalize once (cached), then sweep:
    python run.py --stage normalize
    python run.py --stage sweep --method hdbscan
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
    return (f"{C.CACHE_DIR}/records.jsonl",
            f"{C.CACHE_DIR}/records_meta.json",
            f"{C.CACHE_DIR}/records_source.json")


def _source_sig(args) -> dict:
    """Identity of the inputs a records cache was built from. The cache is only
    reused when this matches, so switching cleaned↔raw, re-cleaning the twins, or
    changing --limit all force a rebuild instead of silently reusing stale records."""
    def mtime(p):
        try:
            return int(Path(p).stat().st_mtime)
        except OSError:
            return None
    return {
        "tickets": args.tickets,
        "time_entries": args.time_entries,
        "limit": args.limit,
        "tickets_mtime": mtime(args.tickets),
        "time_entries_mtime": mtime(args.time_entries),
    }


def stage_clean(args) -> None:
    """Notes twin (deterministic, clean.py) + description twin (LLM batch,
    llm_clean.py). Honors --dry-run / --collect / --submit-only / --force-new."""
    from llm_clean import run_clean_stage
    log("Stage 0 — Clean raw exports (§1.1): notes=regex, description=LLM batch",
        section=True)
    run_clean_stage(args, log=log)


def stage_records(args) -> tuple[list, dict]:
    from records import build_records, save_records
    log("Stage 1 — Build ticket records", section=True)
    recs, meta = build_records(args.tickets, args.time_entries, limit=args.limit, log=log)
    rpath, mpath, spath = _records_paths()
    save_records(recs, rpath)
    Path(mpath).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    Path(spath).write_text(json.dumps(_source_sig(args), indent=2), encoding="utf-8")
    log(f"→ {len(recs):,} records; meta={meta}")
    return recs, meta


def load_or_build_records(args):
    from records import load_records
    rpath, mpath, spath = _records_paths()
    if not args.rebuild_records and all(Path(p).exists() for p in (rpath, mpath, spath)):
        cached_sig = json.loads(Path(spath).read_text())
        if cached_sig == _source_sig(args):
            log(f"Using cached records ({rpath})")
            return load_records(rpath), json.loads(Path(mpath).read_text())
        log(f"Records cache was built from a different source "
            f"({cached_sig.get('tickets')}, limit={cached_sig.get('limit')}) — rebuilding.")
    return stage_records(args)


def stage_normalize(args):
    """§1.5 (ENT-2289) — LLM intent normalization. Batched + cached in
    cache/normalize.jsonl; the hdbscan sweep consumes it automatically."""
    from normalize import run_normalize_stage
    load_or_build_records(args)          # guarantee cache/records.jsonl exists
    log("Stage 1.5 — LLM intent normalization (§1.5)", section=True)
    run_normalize_stage(args, log=log)


def stage_sweep(args):
    recs, meta = load_or_build_records(args)
    if args.method == "hdbscan":
        from sweep_hdbscan import run_sweep_hdbscan
        log("Stage 2 — UMAP + HDBSCAN sweep (ENT-2289)", section=True)
        wcfg = {args.config: C.HDBSCAN_WEIGHTS[args.config]} if args.config else None
        mcs = [args.min_cluster_size] if args.min_cluster_size else None
        run_sweep_hdbscan(recs, meta["total_hours"], weight_configs=wcfg,
                          min_cluster_sizes=mcs, dry_run=args.dry_run,
                          allow_reducer_fallback=args.allow_reducer_fallback, log=log)
        log(f"\nHuman review next: {C.RESULTS_DIR}/hdbscan_summary.json + "
            f"{C.RESULTS_DIR}/hdbscan/configs/*.json → pick config*")
        return
    from sweep import run_sweep
    log("Stage 2 — Field-combo sweep (k-means, ENT-2261)", section=True)
    combos = {args.combo: C.WEIGHT_VECTORS[args.combo]} if args.combo else None
    run_sweep(recs, meta["total_hours"], combos=combos, dry_run=args.dry_run,
              fixed_k=args.k, log=log)
    log(f"\nHuman review next: {C.RESULTS_DIR}/sweep_summary.json + "
        f"{C.RESULTS_DIR}/combos/*.json → pick combo* → "
        f"`python run.py --stage finalize --select <combo>`")


def _load_combo_artifact(name: str) -> dict:
    p = Path(f"{C.RESULTS_DIR}/combos/{name}.json")
    if not p.exists():
        raise SystemExit(f"No combo artifact at {p}. Run the sweep first.")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_hdbscan_artifact(name: str) -> dict:
    p = Path(f"{C.RESULTS_DIR}/hdbscan/configs/{name}.json")
    if not p.exists():
        raise SystemExit(f"No hdbscan config artifact at {p}. Run "
                         f"`--stage sweep --method hdbscan` first.")
    return json.loads(p.read_text(encoding="utf-8"))


def stage_stability(args):
    if not args.select:
        raise SystemExit("--select <config> is required for the stability stage.")
    recs, _ = load_or_build_records(args)
    if args.method == "hdbscan":
        from stability_hdbscan import stability_check_hdbscan
        art = _load_hdbscan_artifact(args.select)
        log("Stage 4 — Bootstrap stability (ENT-2289)", section=True)
        return stability_check_hdbscan(
            recs, art, dry_run=args.dry_run,
            allow_reducer_fallback=args.allow_reducer_fallback, log=log)
    from stability import stability_check
    art = _load_combo_artifact(args.select)
    log("Stage 3 — Stability check", section=True)
    return stability_check(recs, art, dry_run=args.dry_run, log=log)


def stage_finalize(args):
    if not args.select:
        raise SystemExit("--select <combo/config> is required for the finalize stage.")
    recs, meta = load_or_build_records(args)

    if args.method == "hdbscan":
        from stability_hdbscan import stability_check_hdbscan
        from finalize import finalize_hdbscan
        art = _load_hdbscan_artifact(args.select)
        log("Stage 4 — Bootstrap stability (ENT-2289)", section=True)
        stab = None if args.skip_stability else stability_check_hdbscan(
            recs, art, dry_run=args.dry_run,
            allow_reducer_fallback=args.allow_reducer_fallback, log=log)
        log("Stage 5 — Finalize (PII gate + freeze)", section=True)
        finalize_hdbscan(art, meta, stability=stab, log=log)
        return

    from stability import stability_check
    from finalize import finalize
    art = _load_combo_artifact(args.select)
    log("Stage 3 — Stability check", section=True)
    stab = None if args.skip_stability else stability_check(recs, art, dry_run=args.dry_run, log=log)
    log("Stage 4 — Finalize (PII gate + freeze)", section=True)
    finalize(art, meta, stability=stab, log=log)


STAGES = {
    "clean": stage_clean,
    "records": lambda a: stage_records(a),
    "normalize": stage_normalize,
    "sweep": stage_sweep,
    "stability": stage_stability,
    "finalize": stage_finalize,
    "all": None,  # handled below
}


def main():
    # Load .env for EVERY stage. The clean/normalize batch stages already do this
    # internally, but the sweep/stability/finalize stages build the OpenAI/Anthropic
    # clients (which read keys from the environment only), so without this a real
    # `run.py --stage sweep` fails even with the keys in .env. setdefault semantics:
    # anything already exported wins.
    from llm_clean import load_dotenv
    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGES), default="sweep")
    ap.add_argument("--method", choices=["kmeans", "hdbscan"], default="kmeans",
                    help="clustering method for the sweep/stability/finalize stages: "
                         "kmeans (ENT-2261, default) or hdbscan (ENT-2289)")
    ap.add_argument("--tickets", default=C.TICKETS_XLSX)
    ap.add_argument("--time-entries", default=C.TIME_ENTRIES_XLSX)
    ap.add_argument("--limit", type=int, default=None, help="cap kept tickets (smoke test)")
    ap.add_argument("--combo", default=None, help="kmeans: run only this one combo")
    ap.add_argument("--k", type=int, default=None,
                    help="kmeans: fixed k; skips the silhouette k-sweep")
    ap.add_argument("--config", default=None,
                    help="hdbscan: run only this one weight config (see C.HDBSCAN_WEIGHTS)")
    ap.add_argument("--min-cluster-size", type=int, default=None,
                    help="hdbscan: run only this min_cluster_size (else sweeps the list)")
    ap.add_argument("--allow-reducer-fallback", action="store_true",
                    help="hdbscan: use TruncatedSVD if umap-learn is missing (smoke tests only)")
    ap.add_argument("--select", default=None, help="combo* for stability/finalize")
    ap.add_argument("--dry-run", action="store_true",
                    help="stub embeddings + LLM (zero API spend)")
    ap.add_argument("--rebuild-records", action="store_true")
    ap.add_argument("--skip-stability", action="store_true")
    ap.add_argument("--use-raw", action="store_true",
                    help="read the RAW exports instead of the cleaned twins "
                         "(cleaned is the default)")
    ap.add_argument("--use-cleaned", action="store_true",
                    help=argparse.SUPPRESS)   # deprecated: cleaned is now the default
    # `clean` stage (LLM description batch) options — see llm_clean.py
    ap.add_argument("--sample", type=int, default=None,
                    help="clean stage: clean only the first N tickets")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="clean stage: descriptions per batch request")
    ap.add_argument("--char-limit", type=int, default=None,
                    help="clean stage: per-description char cap sent to the LLM")
    ap.add_argument("--poll-interval", type=int, default=None,
                    help="clean stage: seconds between batch status checks")
    ap.add_argument("--collect", action="store_true",
                    help="clean stage: collect an already-submitted batch")
    ap.add_argument("--submit-only", action="store_true",
                    help="clean stage: create the batch and exit")
    ap.add_argument("--force-new", action="store_true",
                    help="clean stage: discard an in-flight batch manifest")
    args = ap.parse_args()

    # Record-consuming stages read the cleaned twins by DEFAULT; --use-raw opts
    # out. Only default paths are swapped (explicit --tickets/--time-entries win),
    # and only when the twin exists — otherwise warn and fall back to raw so a
    # fresh checkout without cleaned twins still runs. `--stage clean` produces
    # the twins, so it always reads the raw exports it was pointed at.
    if not args.use_raw and args.stage != "clean":
        for attr, raw, cleaned in (
            ("tickets", C.TICKETS_XLSX, C.CLEAN_TICKETS_XLSX),
            ("time_entries", C.TIME_ENTRIES_XLSX, C.CLEAN_TIME_ENTRIES_XLSX),
        ):
            if getattr(args, attr) != raw:
                continue                        # user passed an explicit path
            if Path(cleaned).exists():
                setattr(args, attr, cleaned)
            else:
                log(f"note: {cleaned} not found — using raw {attr} "
                    f"(run `--stage clean`, or pass --use-raw to silence this)")

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
