#!/usr/bin/env python3
"""
Section 4 — stability check.

Re-run k-means on combo* 2–3 times with different seeds and compute pairwise
Adjusted Rand Index (ARI) across runs (plus the original sweep run). Low mean
agreement means the clusters are partly a seeding artifact and the combo choice
should be revisited.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import config as C
from sweep import build_feature_blocks, combine


def stability_check(records, combo_artifact: dict, *, reseeds: int = C.STABILITY_RESEEDS,
                    dry_run: bool = False, log=print) -> dict:
    weights = tuple(combo_artifact["weights"][f] for f in C.WEIGHT_ORDER)
    k = combo_artifact["k"]
    log(f"Stability check on combo '{combo_artifact['combo']}' (k={k}, {reseeds} reseeds) …")

    sem_l2, cat_block = build_feature_blocks(records, dry_run=dry_run, log=log)
    X = combine(weights, sem_l2, cat_block)

    # Original sweep labels + `reseeds` fresh seeds
    label_sets = [np.array(combo_artifact["labels"])]
    for i in range(reseeds):
        seed = C.KMEANS_SEED + 1 + i
        km = KMeans(n_clusters=k, n_init=C.KMEANS_N_INIT, random_state=seed)
        label_sets.append(km.fit_predict(X))
        log(f"  reseed {i+1}/{reseeds} (seed={seed}) done")

    aris = [adjusted_rand_score(a, b) for a, b in combinations(label_sets, 2)]
    mean_ari = float(np.mean(aris)) if aris else 1.0
    result = {
        "combo": combo_artifact["combo"],
        "k": k,
        "n_runs": len(label_sets),
        "pairwise_ari": [round(a, 4) for a in aris],
        "mean_ari": round(mean_ari, 4),
        "threshold": C.LOW_ARI_THRESHOLD,
        "stable": mean_ari >= C.LOW_ARI_THRESHOLD,
    }
    verdict = "STABLE" if result["stable"] else "UNSTABLE (flagged)"
    log(f"  mean pairwise ARI = {mean_ari:.4f}  →  {verdict}")
    Path(C.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{C.RESULTS_DIR}/stability.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
