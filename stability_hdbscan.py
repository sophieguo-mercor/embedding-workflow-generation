#!/usr/bin/env python3
"""
Section 4 (ENT-2289) — bootstrap stability check for the HDBSCAN sweep.

ENT-2261 reseeds k-means and checks ARI; that test doesn't apply here — HDBSCAN
is deterministic and UMAP's seed is fixed, so reseeding would test nothing. The
right question for an artifact meant to classify FUTURE tickets is: would each
workflow reappear on a DIFFERENT slice of tickets? That's Hennig's clusterboot:

    for b in 1..B:
        resample the tickets WITH replacement
        re-run UMAP + HDBSCAN on the resample
        for each ORIGINAL cluster: max Jaccard against any bootstrap cluster
            (comparing only the original points present in this resample)
    per-cluster stability = mean of those max-Jaccards over the B resamples

Conventional reading (Hennig): mean Jaccard > 0.75 = stable; 0.60–0.75 = pattern
present but membership doubtful; < 0.60 = do not trust the cluster. Low-stability
clusters are FLAGGED for the reviewer, never silently dropped.

Reuses sweep_hdbscan's feature/reduce/cluster functions verbatim, so the pipeline
under test is exactly the one that produced config*.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config as C
from sweep_hdbscan import build_feature_blocks, cluster_hdbscan, combine, reduce_umap

_SILENT = lambda *a, **k: None      # inner reduce/cluster calls, per-bootstrap, stay quiet


def _verdict(mean_jaccard: float) -> str:
    if mean_jaccard > C.JACCARD_STABLE:
        return "stable"
    if mean_jaccard >= C.JACCARD_DOUBTFUL:
        return "doubtful"
    return "unstable"


def stability_check_hdbscan(records, config_artifact: dict, *,
                            n_bootstrap: int = C.HDBSCAN_N_BOOTSTRAP,
                            dry_run: bool = False,
                            allow_reducer_fallback: bool = False,
                            log=print) -> dict:
    weights = config_artifact["weights"]
    mcs = config_artifact["min_cluster_size"]
    orig_labels = np.asarray(config_artifact["labels"])
    n = len(records)
    if len(orig_labels) != n:
        raise SystemExit(
            f"artifact has {len(orig_labels)} labels but {n} records were loaded — "
            f"rebuild records or re-run the sweep so they line up.")

    name = config_artifact.get("config", "config*")
    title_of = {c["cluster_id"]: c.get("title", f"cluster {c['cluster_id']}")
                for c in config_artifact.get("clusters", [])}
    orig_ids = sorted(c for c in set(orig_labels.tolist()) if c != -1)
    log(f"Bootstrap stability on '{name}' "
        f"({len(orig_ids)} clusters, {n_bootstrap} resamples, mcs={mcs}) …")

    if not orig_ids:
        result = {"config": name, "min_cluster_size": mcs, "n_bootstrap": n_bootstrap,
                  "n_clusters": 0, "mean_jaccard": None, "clusters": [],
                  "low_stability_clusters": [], "note": "no real clusters to test (all noise)."}
        _write(result, log=log)
        return result

    # Combined feature matrix, built exactly as the sweep did (normalized_* merged
    # from cache if present so this reproduces config*'s feature space).
    if Path(C.NORMALIZE_CACHE).exists():
        from normalize import attach_normalized
        attach_normalized(records, log=log)
    needed = sorted({b for b, w in weights.items() if w > 0})
    blocks = build_feature_blocks(records, needed, dry_run=dry_run, log=log)
    X = combine(weights, blocks)
    if X is None:
        raise SystemExit(f"config '{name}' needs a block that is unavailable "
                         f"(normalized_* not populated?). Run normalize.py first.")

    orig_members = {cid: set(np.where(orig_labels == cid)[0].tolist()) for cid in orig_ids}
    per_cluster: dict[int, list[float]] = {cid: [] for cid in orig_ids}
    rng = np.random.default_rng(C.UMAP_SEED)
    backend = None

    for b in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)          # WITH replacement
        uniq = np.unique(sample)                      # Jaccard is over unique original points
        sampled_set = set(uniq.tolist())
        Xr, bk = reduce_umap(X[uniq], allow_fallback=allow_reducer_fallback, log=_SILENT)
        backend = backend or bk
        labels_b, _, _ = cluster_hdbscan(Xr, mcs, log=_SILENT)

        boot_clusters: dict[int, set] = {}
        for j, lab in enumerate(labels_b.tolist()):
            if lab == -1:
                continue
            boot_clusters.setdefault(lab, set()).add(int(uniq[j]))

        for cid in orig_ids:
            present = orig_members[cid] & sampled_set
            if not present:
                per_cluster[cid].append(0.0)
                continue
            best = 0.0
            for bset in boot_clusters.values():
                inter = len(present & bset)
                if inter:
                    best = max(best, inter / len(present | bset))
            per_cluster[cid].append(best)
        log(f"  bootstrap {b + 1}/{n_bootstrap}: {len(boot_clusters)} clusters")

    clusters_out = []
    for cid in orig_ids:
        mean_j = float(np.mean(per_cluster[cid]))
        clusters_out.append({
            "cluster_id": cid,
            "title": title_of.get(cid, f"cluster {cid}"),
            "mean_jaccard": round(mean_j, 4),
            "min_jaccard": round(float(np.min(per_cluster[cid])), 4),
            "verdict": _verdict(mean_j),
        })
    clusters_out.sort(key=lambda c: c["mean_jaccard"])   # least stable first, for review
    low = [c["cluster_id"] for c in clusters_out if c["verdict"] != "stable"]
    mean_all = round(float(np.mean([c["mean_jaccard"] for c in clusters_out])), 4)

    result = {
        "config": name,
        "min_cluster_size": mcs,
        "n_bootstrap": n_bootstrap,
        "n_clusters": len(orig_ids),
        "mean_jaccard": mean_all,
        "reducer_backend": backend,
        "thresholds": {"stable": C.JACCARD_STABLE, "doubtful": C.JACCARD_DOUBTFUL},
        "n_stable": sum(1 for c in clusters_out if c["verdict"] == "stable"),
        "n_doubtful": sum(1 for c in clusters_out if c["verdict"] == "doubtful"),
        "n_unstable": sum(1 for c in clusters_out if c["verdict"] == "unstable"),
        "low_stability_clusters": low,      # flagged for the reviewer, NOT dropped
        "clusters": clusters_out,
    }
    log(f"  mean per-cluster Jaccard = {mean_all}  →  "
        f"{result['n_stable']} stable / {result['n_doubtful']} doubtful / "
        f"{result['n_unstable']} unstable")
    if low:
        log(f"  ⚠️  flagged for review (not dropped): {low}")
    _write(result, log=log)
    return result


def _write(result: dict, *, log=print) -> None:
    out = Path(f"{C.RESULTS_DIR}/hdbscan")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stability.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    log(f"[stability] → {out}/stability.json")
