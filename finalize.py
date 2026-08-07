#!/usr/bin/env python3
"""
Section 5 — finalize the winning cluster set as frozen, auditable ground truth.

Deliberately identical OUTPUT contract for both methods (ENT-2261 k-means and
ENT-2289 UMAP+HDBSCAN), so the dashboard classifier — and the eventual bake-off —
consume the same shape regardless of how the clusters were produced:

1. PII gate (pii.py): hard-fail on PII in structured fields, blank PII in prose.
2. Freeze + version: taxonomy.md / clusters.jsonl / metrics.json, recording the
   method-specific provenance (weights, k or min_cluster_size, seeds/params, model
   versions, dataset window) needed to reproduce and trace the rubric.
3. Output in the same `Name | Category | Description` shape the dashboard consumes.

Only clusters that QUALIFY (named, coherent, big-enough — §2.6) become rubric
rows. Non-qualifying clusters are recorded in clusters.jsonl but folded into an
explicit "Other" workflow so the rubric stays auditable. Low-stability clusters
(§4) are FLAGGED in metrics/clusters.jsonl for the reviewer, never dropped.

`finalize()` is k-means; `finalize_hdbscan()` is HDBSCAN. Both assemble their own
provenance and then call the shared _build_rows / _gate_and_write, so the artifact
shape can't drift between methods.
"""
from __future__ import annotations

import json
from pathlib import Path

import config as C
import pii

OTHER_ROW = {
    "name": "Long-tail / Other",
    "category": "Other",
    "description": "Tickets whose cluster was text-poor, too small to be a "
                   "real workflow, or flagged as covering several processes.",
}


# ── shared, method-agnostic core ──────────────────────────────────────────────

def _build_rows(qualifying: list[dict]) -> list[dict]:
    """Qualifying clusters → Name|Category|Description rows + the explicit Other row."""
    rows = [{
        "name": c["title"],
        "category": c["category"],
        "description": c["description"],
    } for c in qualifying]
    rows.append(dict(OTHER_ROW))     # complete + auditable
    return rows


def _gate_and_write(rows: list[dict], clusters: list[dict], metrics: dict,
                    out_dir: str, *, log=print) -> dict:
    """PII-gate the rows, then freeze taxonomy.md / clusters.jsonl / metrics.json."""
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
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                      encoding="utf-8")

    log(f"[finalize] froze {len(rows)} workflows → {out}/taxonomy.md")
    log(f"[finalize]   clusters.jsonl ({len(clusters)} clusters), metrics.json written")
    return metrics


# ── k-means (ENT-2261) ────────────────────────────────────────────────────────

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
    rows = _build_rows(qualifying)

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
    return _gate_and_write(rows, clusters, metrics, out_dir, log=log)


# ── UMAP + HDBSCAN (ENT-2289) ─────────────────────────────────────────────────

def finalize_hdbscan(
    config_artifact: dict,
    records_meta: dict,
    *,
    stability: dict | None = None,
    dataset_window: str = "Mercor v3",
    out_dir: str | None = None,
    log=print,
) -> dict:
    out_dir = out_dir or f"{C.RESULTS_DIR}/hdbscan"
    clusters = [dict(c) for c in config_artifact["clusters"]]     # copy to annotate

    # Merge the §4 bootstrap verdict onto each cluster — FLAG low-stability ones
    # for the reviewer, never drop them (they still become rubric rows).
    verdict_by_id = {c["cluster_id"]: c for c in (stability or {}).get("clusters", [])}
    for c in clusters:
        sv = verdict_by_id.get(c["cluster_id"])
        if sv:
            c["stability"] = {"mean_jaccard": sv.get("mean_jaccard"),
                              "verdict": sv.get("verdict")}

    qualifying = [c for c in clusters if c["qualifies"]]
    rows = _build_rows(qualifying)
    flagged = [c["cluster_id"] for c in qualifying
               if c.get("stability", {}).get("verdict") not in (None, "stable")]

    metrics = {
        "config": config_artifact["config"],
        "weights": config_artifact["weights"],
        "min_cluster_size": config_artifact["min_cluster_size"],
        "coverage": config_artifact["coverage"],
        "noise_mass": config_artifact.get("noise_mass"),
        "llm_score": config_artifact["llm_score"],
        "n_workflows": len(rows),          # incl. the Other row
        "n_qualifying_clusters": len(qualifying),
        "low_stability_qualifying": flagged,   # flagged, NOT dropped (§4.3)
        "stability": stability,
        "backends": config_artifact.get("backends"),
        "umap": config_artifact.get("umap"),
        "models": {"embed": C.EMBED_MODEL, "llm": C.LLM_MODEL, "normalize": C.NORMALIZE_MODEL},
        "dataset_window": dataset_window,
        "records_meta": records_meta,
        "thresholds": {
            # HDBSCAN's ticket-floor is ALIGNED to this config's min_cluster_size
            # (not the k-means-only MIN_CLUSTER_TICKETS), so record the aligned value.
            "min_cluster_tickets": config_artifact["min_cluster_size"],
            "min_cluster_mass_frac": C.MIN_CLUSTER_MASS_FRAC,
            "min_cluster_size_sweep": list(C.HDBSCAN_MIN_CLUSTER_SIZES),
            "jaccard_stable": C.JACCARD_STABLE,
            "jaccard_doubtful": C.JACCARD_DOUBTFUL,
        },
    }
    if flagged:
        log(f"[finalize] {len(flagged)} qualifying cluster(s) flagged low-stability "
            f"(kept, not dropped): {flagged}")
    return _gate_and_write(rows, clusters, metrics, out_dir, log=log)
