#!/usr/bin/env python3
"""
Section 2 — the field-combo sweep (the heart of the pipeline).

For each hand-picked weight combo (w_title, w_description, w_notes, w_cat):

    X = L2( concat( w_f · L2(emb_f) for f in {title,desc,notes} if w_f>0 )
            ++ w_cat · L2(tfidf_cat) )
    best_k = argmax_k silhouette(KMeans(X, k))        # k chosen WITHIN this space only
    clusters = KMeans(X, best_k, n_init=10, seed)
    named = name_clusters(clusters)                   # 1 LLM call/cluster → title+desc+coherence flag
    categorized = assign_categories(named)            # 1 batched call → shared category labels
    coverage = mass_weighted_coverage(named)          # % engineer-hours in named, coherent, big-enough clusters
    llm_score = coherence_judge(named)                # combo-comparable coherence + distinctness

Each combo writes a self-contained artifact to results/combos/<name>.json.
"""
from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import numpy as np

# Benign float32 accumulation warnings from KMeans/silhouette on the many
# all-zero rows an empty-field combo produces. Results are unaffected.
warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

import config as C
from categorical import build_categorical_block
from embeddings import embed_all_semantic
from llm import LLM


# ── feature blocks (computed once, §1) ───────────────────────────────────────

def build_feature_blocks(records, *, dry_run: bool = False, log=print):
    """Return (sem_l2: {field: (n,dim) L2-normalised}, cat_block: (n,vocab) L2)."""
    log("Building semantic blocks (embeddings) …")
    sem = embed_all_semantic(records, dry_run=dry_run, log=log)
    sem_l2 = {f: normalize(m, norm="l2", axis=1) for f, m in sem.items()}
    log("Building categorical block (TF-IDF) …")
    cat_block, vec, _ = build_categorical_block(records)
    log(f"  categorical vocab size: {len(vec.vocabulary_)}")
    return sem_l2, cat_block


def combine(combo: tuple[float, float, float, float], sem_l2, cat_block) -> np.ndarray:
    """Concatenate the weighted blocks for a combo and L2-normalise the result.
    A zero-weight field is DROPPED (not concatenated)."""
    w_title, w_desc, w_notes, w_cat = combo
    weights = {"title": w_title, "description": w_desc, "notes": w_notes}
    blocks = [w * sem_l2[f] for f, w in weights.items() if w > 0]
    if w_cat > 0:
        blocks.append(w_cat * cat_block)
    if not blocks:
        raise ValueError("combo has all-zero weights")
    X = np.hstack(blocks).astype(np.float32)
    return normalize(X, norm="l2", axis=1)


# ── k selection via silhouette (§2 step 2) ───────────────────────────────────

def choose_k(X: np.ndarray, *, log=print) -> tuple[int, dict[int, float]]:
    """Sweep k over [K_MIN, K_MAX] (capped by ticket volume) and return the k
    with the highest silhouette. Silhouette is sampled for speed on large N."""
    n = X.shape[0]
    k_cap = max(C.GLOBAL_K_MIN, n // C.MIN_TICKETS_PER_CLUSTER)
    k_max = min(C.GLOBAL_K_MAX, k_cap, n - 1)
    if k_max < C.GLOBAL_K_MIN:
        # tiny dataset (smoke test): fall back to a small valid range
        k_min = max(2, min(C.GLOBAL_K_MIN, n // 2))
        k_max = max(k_min + 1, min(k_max if k_max > k_min else n - 1, n - 1))
    else:
        k_min = C.GLOBAL_K_MIN
    step = max(1, (k_max - k_min) // 12)  # ~12 probes, not every integer
    ks = list(range(k_min, k_max + 1, step))
    sample_size = min(C.SILHOUETTE_SAMPLE, n)

    scores: dict[int, float] = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=C.KMEANS_N_INIT, random_state=C.KMEANS_SEED)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels, sample_size=sample_size, random_state=C.KMEANS_SEED) \
            if len(set(labels)) > 1 else -1.0
        scores[k] = float(s)
        log(f"    k={k:>3}  silhouette={s:.4f}")
    best_k = max(scores, key=scores.get)
    log(f"  → best_k={best_k} (silhouette={scores[best_k]:.4f}); "
        f"range [{k_min},{k_max}] step {step}")
    return best_k, scores


# ── cluster sampling for the naming call (§2 step 4) ─────────────────────────

def sample_cluster_members(X, labels, cluster_id, centroid, records, rng) -> list[dict]:
    """Centroid-nearest members + a few diverse ones — not uniform-random."""
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    d = np.linalg.norm(X[idx] - centroid, axis=1)
    order = idx[np.argsort(d)]
    nearest = list(order[:C.NAMING_NEAREST])
    rest = list(order[C.NAMING_NEAREST:])
    diverse = list(rng.sample(rest, min(C.NAMING_DIVERSE, len(rest)))) if rest else []
    chosen = nearest + diverse
    return [{
        "title": records[i].title,
        "issue_type": records[i].issue_type,
        "sub_issue_type": records[i].sub_issue_type,
        "notes": records[i].notes,
    } for i in chosen]


# ── coverage (§2 step 6) ──────────────────────────────────────────────────────

def mass_weighted_coverage(clusters: list[dict], total_hours: float) -> dict:
    """% of engineer-hours in named, coherent, big-enough clusters vs "Other".

    total_hours is the WHOLE corpus (incl. text-poor drops), so coverage can't
    trivially sit near 100%."""
    covered = 0.0
    n_named = 0
    for c in clusters:
        if c["qualifies"]:
            covered += c["hours"]
            n_named += 1
    frac = covered / total_hours if total_hours else 0.0
    return {
        "coverage": round(frac, 4),
        "covered_hours": round(covered, 2),
        "total_hours": round(total_hours, 2),
        "n_named_clusters": n_named,
        "n_clusters": len(clusters),
    }


# ── one combo end-to-end ──────────────────────────────────────────────────────

def run_combo(
    name: str,
    combo: tuple[float, float, float, float],
    sem_l2,
    cat_block,
    records,
    total_hours: float,
    llm: LLM,
    *,
    out_dir: str,
    log=print,
) -> dict:
    log(f"\n══ combo '{name}' weights={combo} ══", )
    X = combine(combo, sem_l2, cat_block)
    log(f"  combined feature dim: {X.shape[1]}")

    best_k, sil_scores = choose_k(X, log=log)
    km = KMeans(n_clusters=best_k, n_init=C.KMEANS_N_INIT, random_state=C.KMEANS_SEED)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    hours = np.array([r.hours for r in records], dtype=np.float64)
    rng = random.Random(C.KMEANS_SEED)

    # §2.4 — name each cluster (one LLM call/cluster)
    log(f"  naming {best_k} clusters …")
    clusters: list[dict] = []
    named_for_cat: list[dict] = []
    for cid in range(best_k):
        members = np.where(labels == cid)[0]
        n_tickets = int(len(members))
        cl_hours = float(hours[members].sum())
        hours_frac = cl_hours / total_hours if total_hours else 0.0
        samples = sample_cluster_members(X, labels, cid, centroids[cid], records, rng)
        named = llm.name_cluster(samples, cluster_id=cid)

        big_enough = n_tickets >= C.MIN_CLUSTER_TICKETS and hours_frac >= C.MIN_CLUSTER_MASS_FRAC
        qualifies = bool(named["coherent"]) and big_enough  # §2.6 "Other" gate

        cluster = {
            "cluster_id": cid,
            "title": named["title"],
            "description": named["description"],
            "coherent": named["coherent"],
            "coherence_note": named["coherence_note"],
            "n_tickets": n_tickets,
            "hours": round(cl_hours, 2),
            "hours_frac": round(hours_frac, 4),
            "big_enough": big_enough,
            "qualifies": qualifies,
            "examples": [s["title"] or s["notes"][:120] for s in samples[:3]],
        }
        clusters.append(cluster)
        named_for_cat.append({"id": cid, "title": named["title"], "description": named["description"]})

    # §2.5 — batched category assignment over ALL named clusters
    log("  assigning shared categories (1 batched call) …")
    cats = llm.assign_categories(named_for_cat)
    for c in clusters:
        c["category"] = cats.get(c["cluster_id"], "Other")

    # §2.6 — combo-level coherence/distinctness judge (blind, combo-comparable)
    log("  scoring combo coherence/distinctness (LLM judge) …")
    judge_input = [{"title": c["title"], "description": c["description"], "examples": c["examples"]}
                   for c in clusters if c["qualifies"]] or \
                  [{"title": c["title"], "description": c["description"], "examples": c["examples"]}
                   for c in clusters]
    llm_score = llm.coherence_judge(judge_input)

    cov = mass_weighted_coverage(clusters, total_hours)
    log(f"  coverage={cov['coverage']:.1%}  "
        f"llm_overall={llm_score['overall']:.1f}  k={best_k}  "
        f"named={cov['n_named_clusters']}/{cov['n_clusters']}")

    artifact = {
        "combo": name,
        "weights": dict(zip(C.WEIGHT_ORDER, combo)),
        "k": best_k,
        "silhouette_by_k": sil_scores,
        "best_silhouette": sil_scores[best_k],
        "coverage": cov,
        "llm_score": llm_score,
        "clusters": clusters,
        "labels": labels.tolist(),  # ticket-order cluster assignment (for stability §4)
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"{name}.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


def run_sweep(records, total_hours, *, combos=None, dry_run=False,
              out_dir=f"{C.RESULTS_DIR}/combos", log=print) -> dict:
    """Run every combo and write a side-by-side summary (Section 3 input)."""
    combos = combos or C.WEIGHT_VECTORS
    sem_l2, cat_block = build_feature_blocks(records, dry_run=dry_run, log=log)
    llm = LLM(dry_run=dry_run)

    summary = []
    for name, combo in combos.items():
        art = run_combo(name, combo, sem_l2, cat_block, records, total_hours,
                        llm, out_dir=out_dir, log=log)
        summary.append({
            "combo": name,
            "weights": art["weights"],
            "k": art["k"],
            "coverage": art["coverage"]["coverage"],
            "n_named": art["coverage"]["n_named_clusters"],
            "coherence": art["llm_score"]["coherence"],
            "distinctness": art["llm_score"]["distinctness"],
            "llm_overall": art["llm_score"]["overall"],
            "best_silhouette": art["best_silhouette"],
        })

    summary.sort(key=lambda s: (-s["coverage"], -s["llm_overall"]))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{C.RESULTS_DIR}/sweep_summary.json").write_text(
        json.dumps({"combos": summary, "llm_usage": llm.usage}, indent=2), encoding="utf-8")
    log("\n── sweep summary (Section 3 — human picks combo*) ──")
    log(f"{'combo':<18}{'k':>4}{'cover':>8}{'named':>7}{'coh':>6}{'dist':>6}{'overall':>9}")
    for s in summary:
        log(f"{s['combo']:<18}{s['k']:>4}{s['coverage']:>8.1%}{s['n_named']:>7}"
            f"{s['coherence']:>6.1f}{s['distinctness']:>6.1f}{s['llm_overall']:>9.1f}")
    return {"combos": summary, "llm_usage": llm.usage}
