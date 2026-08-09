#!/usr/bin/env python3
"""
Backfill the §2.8 MECE consolidation onto EXISTING HDBSCAN config artifacts.

The sweep now runs consolidation inline (sweep_hdbscan.py §2.8) and stores it under
each config JSON's `consolidation` key. This CLI is for the other direction: adding
(or refreshing) consolidation on config JSONs that were produced BEFORE §2.8 existed,
without re-running the whole clustering sweep. It reuses the exact same shared code
path (consolidate.build_consolidation), then rewrites the config JSON in place and
regenerates its .xlsx (with the `workflows` sheet).

    python merge_hdbscan_taxonomy.py --config normalized_categorical__mcs100
    python merge_hdbscan_taxonomy.py --all          # every config with >= CONSOLIDATE_MIN_CLUSTERS
    python merge_hdbscan_taxonomy.py --config X --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config as C
import consolidate
from llm import LLM
from to_excel_hdbscan import config_to_excel


def backfill_one(json_path: Path, llm: LLM, *, refresh_xlsx: bool, log=print) -> None:
    artifact = json.loads(json_path.read_text(encoding="utf-8"))
    clusters = artifact.get("clusters", [])
    if len(clusters) < C.CONSOLIDATE_MIN_CLUSTERS:
        log(f"[skip] {json_path.stem}: {len(clusters)} clusters "
            f"< CONSOLIDATE_MIN_CLUSTERS={C.CONSOLIDATE_MIN_CLUSTERS}")
        return
    log(f"[merge] {json_path.stem}: {len(clusters)} clusters | model={llm.model}"
        f"{' (dry-run)' if llm.dry_run else ''}")
    artifact["consolidation"] = consolidate.build_consolidation(llm, clusters, log=log)
    json_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[merge] patched {json_path}")
    if refresh_xlsx:
        out_dir = Path(C.RESULTS_DIR) / "hdbscan" / "configs_xlsx"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = config_to_excel(json_path, out_dir)
        log(f"[merge] refreshed {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="config artifact name under results/hdbscan/configs/")
    ap.add_argument("--all", action="store_true",
                    help="backfill every config artifact (skips those with too few clusters)")
    ap.add_argument("--model", default=C.CONSOLIDATE_MODEL,
                    help=f"Anthropic model id (default {C.CONSOLIDATE_MODEL})")
    ap.add_argument("--no-xlsx", action="store_true", help="patch JSON only; don't refresh .xlsx")
    ap.add_argument("--dry-run", action="store_true",
                    help="stub consolidation (no API call) — smoke-test wiring")
    args = ap.parse_args()

    if not (args.config or args.all):
        raise SystemExit("Pass --config <name> or --all.")

    configs_dir = Path(C.RESULTS_DIR) / "hdbscan" / "configs"
    if args.all:
        paths = sorted(configs_dir.glob("*.json"))
    else:
        paths = [configs_dir / f"{args.config}.json"]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit(f"No config artifact(s) found in {configs_dir}.")

    if not args.dry_run:
        from llm_clean import load_dotenv   # match run.py: load ANTHROPIC_API_KEY from .env
        load_dotenv()
    llm = LLM(model=args.model, dry_run=args.dry_run)

    for p in paths:
        backfill_one(p, llm, refresh_xlsx=not args.no_xlsx)
    if not args.dry_run:
        print(f"[merge] usage: {llm.usage}")


if __name__ == "__main__":
    main()
