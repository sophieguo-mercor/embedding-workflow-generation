#!/usr/bin/env python3
"""
Section 5 — finalize the winning combo as frozen, auditable ground truth.

1. PII gate (pii.py): hard-fail on PII in structured fields, blank PII in prose.
2. Freeze + version: taxonomy.md / clusters.jsonl / metrics.json, recording the
   combo, weights, k, model versions, and dataset window.
3. Output in the same `Name | Category | Description` shape the dashboard
   classifier already consumes.

Only clusters that QUALIFY (named, coherent, big-enough — §2.6) become rubric
rows. Non-qualifying clusters are recorded in clusters.jsonl but folded into an
explicit "Other" workflow so the rubric stays auditable.
"""
from __future__ import annotations

import json
from pathlib import Path

import config as C
import pii


def finalize(
    combo_artifact: dict,
    records_meta: dict,
    *,
    stability: dict | None = None,
    dataset_window: str = "Mercor v3",
    out_dir: str = C.RESULTS_DIR,
    log=print,
) -> dict:
    clusters = combo_artifact["clusters"]
    qualifying = [c for c in clusters if c["qualifies"]]

    # Build rubric rows (Name | Category | Description)
    rows = [{
        "name": c["title"],
        "category": c["category"],
        "description": c["description"],
    } for c in qualifying]
    # Explicit catch-all so the shape is complete and auditable
    rows.append({
        "name": "Long-tail / Other",
        "category": "Other",
        "description": "Tickets whose cluster was text-poor, too small to be a "
                       "real workflow, or flagged as covering several processes.",
    })

    # §5.1 — PII gate (mutates descriptions in place, hard-fails structured PII)
    rows = pii.gate_rubric(rows, log=log)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # taxonomy.md — the Name | Category | Description artifact the dashboard reads
    md_lines = [f"{r['name']} | {r['category']} | {r['description']}" for r in rows]
    (out / "taxonomy.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # clusters.jsonl — full per-cluster detail (incl. non-qualifying), for audit
    with (out / "clusters.jsonl").open("w", encoding="utf-8") as f:
        for c in clusters:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # metrics.json — everything needed to reproduce / trace the frozen rubric
    metrics = {
        "combo": combo_artifact["combo"],
        "weights": combo_artifact["weights"],
        "k": combo_artifact["k"],
        "best_silhouette": combo_artifact.get("best_silhouette"),
        "coverage": combo_artifact["coverage"],
        "llm_score": combo_artifact["llm_score"],
        "n_workflows": len(rows),          # incl. the Other row
        "n_qualifying_clusters": len(qualifying),
        "stability": stability,
        "models": {"embed": C.EMBED_MODEL, "llm": C.LLM_MODEL},
        "dataset_window": dataset_window,
        "records_meta": records_meta,
        "thresholds": {
            "min_cluster_tickets": C.MIN_CLUSTER_TICKETS,
            "min_cluster_mass_frac": C.MIN_CLUSTER_MASS_FRAC,
            "low_ari_threshold": C.LOW_ARI_THRESHOLD,
            "k_range": [C.GLOBAL_K_MIN, C.GLOBAL_K_MAX],
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                      encoding="utf-8")

    log(f"[finalize] froze {len(rows)} workflows → {out}/taxonomy.md")
    log(f"[finalize]   clusters.jsonl ({len(clusters)} clusters), metrics.json written")
    return metrics
