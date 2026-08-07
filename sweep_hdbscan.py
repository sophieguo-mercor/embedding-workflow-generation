#!/usr/bin/env python3
"""
Section 2 (ENT-2289) — the BERTopic-style config sweep: UMAP + HDBSCAN.

The density-based analog of sweep.py (k-means). It shares the SAME feature
blocks, the SAME LLM naming/category/judge passes, and — the point of the
bake-off — the SAME scoring in metrics.py, so this rubric is directly comparable
to ENT-2261's. Only the clustering core differs:

    X    = L2( concat( w_f · L2(block_f) for f in BLOCKS if w_f>0 ) )
    Xr   = UMAP(X, n_components=5, metric='cosine', min_dist=0.0, seed fixed)
    labels, probs = HDBSCAN(Xr, min_cluster_size=mcs)      # label -1 == noise
    keywords = c_tf_idf(clusters)                          # deterministic, auditable
    named    = name each cluster (1 LLM call, c-TF-IDF keywords passed alongside)
    coverage   = metrics.mass_weighted_coverage(named, total_hours)       # SHARED
    llm_score  = llm.coherence_judge(metrics.select_judge_clusters(named))# SHARED
    noise_mass = metrics.noise_mass(hours_in_noise, total_hours)          # SHARED

A "config" is (weight vector, min_cluster_size); the sweep is their cross-product.
Each writes a self-contained artifact to results/hdbscan/configs/<name>.json.

Why noise is a first-class output: k-means force-assigns every ticket, so ENT-2261
needs an artificial mass floor or coverage sits near 100%. HDBSCAN drops bad-fit
tickets per-point as noise, so coverage discriminates by default and noise_mass is
tracked SEPARATELY (low coverage from too-aggressive noise vs. from failed
coherence flags call for opposite fixes — ENT-2289 §2.7).

Dependencies
------------
Production uses umap-learn + hdbscan (requirements.txt). For a zero-install smoke
test the clusterer auto-falls back to sklearn.cluster.HDBSCAN (shipped with
scikit-learn); the UMAP reducer falls back to a LINEAR TruncatedSVD ONLY under
--allow-reducer-fallback — a loud, artifact-recorded downgrade, never silent,
because SVD does not preserve the local structure UMAP does.
"""
from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
# Benign float32 accumulation warnings from the reducer on the many all-zero rows
# an empty-field / stub-embedding combo produces. Results are unaffected (matches
# sweep.py). Real runs on real embeddings don't hit this.
warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

import config as C
import metrics
from categorical import build_categorical_block
from embeddings import embed_field
from llm import LLM


# ── feature blocks (computed once, §1) ───────────────────────────────────────

def _field_has_text(records, field: str) -> bool:
    """True iff any record carries non-empty text for `field`. getattr default
    keeps this safe for the normalized_* fields, which aren't on TicketRecord
    until normalize.py adds them."""
    return any((getattr(r, field, "") or "").strip() for r in records)


def build_feature_blocks(records, needed, *, dry_run: bool = False, log=print):
    """Return {block: (n,dim) L2-normalised matrix, or None if unavailable}.

    Semantic blocks are embedded (reusing embed_field's per-(id,field) cache);
    `cat` is the reused ENT-2261 categorical TF-IDF block. A semantic block whose
    text is entirely empty (e.g. normalized_* before normalize.py has run) is
    returned as None so callers skip configs that need it rather than clustering
    on all-zero columns."""
    blocks: dict[str, np.ndarray | None] = {}
    for b in needed:
        if b == "cat":
            log("Building categorical block (TF-IDF) …")
            cat_block, vec, _ = build_categorical_block(records)
            log(f"  categorical vocab size: {len(vec.vocabulary_)}")
            blocks["cat"] = cat_block           # already L2-normalised
        elif not _field_has_text(records, b):
            log(f"note: block '{b}' has no text on these records — unavailable "
                f"(run normalize.py to populate the normalized_* fields)")
            blocks[b] = None
        else:
            m = embed_field(records, b, dry_run=dry_run, log=log)
            blocks[b] = normalize(m, norm="l2", axis=1)
    return blocks


def combine(weights: dict[str, float], blocks) -> np.ndarray | None:
    """Concatenate the weighted available blocks and L2-normalise. A zero-weight
    block is dropped. Returns None if any positively-weighted block is
    unavailable, so the caller can skip the config."""
    parts = []
    for b, w in weights.items():
        if w <= 0:
            continue
        blk = blocks.get(b)
        if blk is None:
            return None
        parts.append(w * blk)
    if not parts:
        raise ValueError("config has no positively-weighted, available blocks")
    X = np.hstack(parts).astype(np.float32)
    return normalize(X, norm="l2", axis=1)


# ── UMAP reduction (§2 step 2) ────────────────────────────────────────────────

def reduce_umap(X: np.ndarray, *, allow_fallback: bool = False, log=print):
    """Reduce X to ~5 components. Returns (Xr, backend_name). Prefers umap-learn;
    with --allow-reducer-fallback and umap missing, uses TruncatedSVD (LINEAR —
    a wiring smoke-test stand-in, NOT comparable to a real UMAP run)."""
    n = X.shape[0]
    n_comp = min(C.UMAP_N_COMPONENTS, max(2, min(n - 1, X.shape[1] - 1)))
    n_neighbors = min(C.UMAP_N_NEIGHBORS, max(2, n - 1))
    try:
        import umap  # umap-learn
    except ImportError:
        if not allow_fallback:
            raise SystemExit(
                "umap-learn is not installed. `pip install umap-learn` for the real "
                "reducer, or pass --allow-reducer-fallback to substitute a LINEAR "
                "TruncatedSVD (smoke tests only — not comparable to a real UMAP run).")
        from sklearn.decomposition import TruncatedSVD
        log("⚠️  UMAP unavailable — FALLING BACK to TruncatedSVD (linear; NOT a "
            "substitute for UMAP). For wiring smoke tests only.")
        Xr = TruncatedSVD(n_components=n_comp, random_state=C.UMAP_SEED).fit_transform(X)
        return Xr.astype(np.float32), "truncated_svd_fallback"
    log(f"Reducing with UMAP → {n_comp}d (metric={C.UMAP_METRIC}, "
        f"min_dist={C.UMAP_MIN_DIST}, n_neighbors={n_neighbors}) …")
    reducer = umap.UMAP(n_components=n_comp, metric=C.UMAP_METRIC,
                        min_dist=C.UMAP_MIN_DIST, n_neighbors=n_neighbors,
                        random_state=C.UMAP_SEED)
    return reducer.fit_transform(X).astype(np.float32), "umap"


# ── HDBSCAN clustering (§2 step 3) ────────────────────────────────────────────

def cluster_hdbscan(Xr: np.ndarray, min_cluster_size: int, *, log=print):
    """Cluster the reduced space. Returns (labels, probabilities, backend). Label
    -1 is noise. Prefers the `hdbscan` package; auto-falls back to
    sklearn.cluster.HDBSCAN (shipped with scikit-learn) — a legitimate
    implementation, so this fallback is silent-but-recorded, not gated."""
    mcs = max(2, min(min_cluster_size, Xr.shape[0] - 1))
    try:
        import hdbscan as _h
        clusterer = _h.HDBSCAN(min_cluster_size=mcs, metric="euclidean")
        backend = "hdbscan"
    except ImportError:
        from sklearn.cluster import HDBSCAN as _SK
        clusterer = _SK(min_cluster_size=mcs, metric="euclidean")
        backend = "sklearn.cluster.HDBSCAN"
        log(f"note: `hdbscan` package not installed — using {backend}")
    labels = np.asarray(clusterer.fit_predict(Xr))
    probs = np.asarray(getattr(clusterer, "probabilities_", np.ones(len(labels))),
                       dtype=np.float64)
    return labels, probs, backend


# ── c-TF-IDF keywords (§2 step 4) ─────────────────────────────────────────────

# Dutch + English function words. This corpus is bilingual, and with few clusters
# the c-TF-IDF idf term can't suppress stopwords on its own (they get nonzero
# tf·idf and rank high) — so the vectorizer strips them, exactly as BERTopic's
# default does. Not exhaustive; the highest-frequency function words are enough to
# keep the keyword lists meaningful as a naming input and audit backstop.
_STOPWORDS = frozenset("""
de het een en van te dat die in op voor met als maar om aan er nog toe uit
naar bij ook tot je zijn was heb hebben wij jij zij deze dit niet wel geen
ik we ze me mijn zijn haar hun ons onze door over onder tegen zonder tussen
worden wordt werd zou zal kan kunnen moet mag graag even svp aub dank
the a an and or of to in on for with as but at by from is are was were be been
this that these those it its we you they he she i me my our your their not no
have has had do does did will would can could should may might please thanks
""".split())


def c_tf_idf(records, labels: np.ndarray, *, top_n: int | None = None):
    """Class-based TF-IDF (BERTopic style): treat each cluster as one document,
    score terms by (term-freq within class) · log(1 + avg_words / term_total).
    Deterministic, no LLM. Returns {cluster_id: [top terms]}; noise (-1) skipped."""
    top_n = top_n or C.CTFIDF_TOP_N
    cluster_ids = sorted(c for c in set(labels.tolist()) if c != -1)
    if not cluster_ids:
        return {}
    docs = []
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        docs.append(" ".join(((records[i].title or "") + " " + (records[i].notes or ""))
                             for i in idx))
    cv = CountVectorizer(min_df=1, lowercase=True, token_pattern=r"(?u)\b\w\w\w+\b",
                         stop_words=list(_STOPWORDS))
    try:
        counts = cv.fit_transform(docs).toarray().astype(np.float64)
    except ValueError:                       # empty vocabulary (all-stopword docs)
        return {cid: [] for cid in cluster_ids}
    words_per_class = counts.sum(axis=1)
    words_per_class[words_per_class == 0] = 1.0
    tf = counts / words_per_class[:, None]
    term_total = counts.sum(axis=0)
    term_total[term_total == 0] = 1.0
    idf = np.log(1.0 + counts.sum() / counts.shape[0] / term_total)  # 1 + avg_words/f_t
    ctfidf = tf * idf[None, :]
    vocab = np.array(cv.get_feature_names_out())
    out = {}
    for row, cid in enumerate(cluster_ids):
        order = np.argsort(ctfidf[row])[::-1][:top_n]
        out[cid] = [str(vocab[j]) for j in order if ctfidf[row, j] > 0]
    return out


# ── cluster sampling for the naming call (§2 step 5) ─────────────────────────

def sample_cluster_members(labels, probs, cluster_id, records, rng) -> list[dict]:
    """Highest-membership-probability members + a few diverse ones. The
    density-based analog of k-means' centroid-nearest: density clusters have no
    meaningful centroid, but probability ranks how core a point is."""
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    order = idx[np.argsort(probs[idx])[::-1]]          # most-core first
    top = list(order[:C.HDBSCAN_NAMING_EXEMPLARS])
    rest = list(order[C.HDBSCAN_NAMING_EXEMPLARS:])
    diverse = list(rng.sample(rest, min(C.HDBSCAN_NAMING_DIVERSE, len(rest)))) if rest else []
    return [{
        "title": records[i].title,
        "issue_type": records[i].issue_type,
        "sub_issue_type": records[i].sub_issue_type,
        "notes": records[i].notes,
    } for i in (top + diverse)]


# ── one config end-to-end ─────────────────────────────────────────────────────

def run_config(name, weights, mcs, blocks, records, total_hours, llm, *,
               allow_reducer_fallback=False, out_dir, log=print) -> dict | None:
    log(f"\n══ config '{name}'  mcs={mcs}  weights={dict(weights)} ══")
    X = combine(weights, blocks)
    if X is None:
        log(f"  SKIP '{name}': a positively-weighted block is unavailable "
            f"(normalized_* needs normalize.py). Not run.")
        return None
    log(f"  combined feature dim: {X.shape[1]}")

    Xr, reducer = reduce_umap(X, allow_fallback=allow_reducer_fallback, log=log)
    labels, probs, clusterer = cluster_hdbscan(Xr, mcs, log=log)

    real_ids = sorted(c for c in set(labels.tolist()) if c != -1)
    hours = np.array([r.hours for r in records], dtype=np.float64)
    noise_hours = float(hours[labels == -1].sum())
    log(f"  {len(real_ids)} clusters + noise "
        f"({int((labels == -1).sum())} tickets, {noise_hours:.1f}h)")

    keywords = c_tf_idf(records, labels)
    rng = random.Random(C.UMAP_SEED)

    # §2.5 — name each real cluster (one LLM call, c-TF-IDF keywords alongside)
    clusters: list[dict] = []
    named_for_cat: list[dict] = []
    for cid in real_ids:
        members = np.where(labels == cid)[0]
        n_tickets = int(len(members))
        cl_hours = float(hours[members].sum())
        hours_frac = cl_hours / total_hours if total_hours else 0.0
        kw = keywords.get(cid, [])
        samples = sample_cluster_members(labels, probs, cid, records, rng)
        named = llm.name_cluster(samples, cluster_id=cid, keywords=kw)

        cluster = {
            "cluster_id": cid,
            "title": named["title"],
            "description": named["description"],
            "coherent": named["coherent"],
            "coherence_note": named["coherence_note"],
            "keywords": kw,
            "n_tickets": n_tickets,
            "hours": round(cl_hours, 2),
            "hours_frac": hours_frac,       # full precision for the gate; rounded below
            "big_enough": False,            # both set by metrics.mark_qualification
            "qualifies": False,
            "examples": [s["title"] or s["notes"][:120] for s in samples[:3]],
        }
        metrics.mark_qualification(cluster)     # §2.6 "Other" gate (shared)
        cluster["hours_frac"] = round(hours_frac, 4)
        clusters.append(cluster)
        named_for_cat.append({"id": cid, "title": named["title"], "description": named["description"]})

    # §2.6 — batched category assignment over ALL named clusters
    log("  assigning shared categories (1 batched call) …")
    cats = llm.assign_categories(named_for_cat)
    for c in clusters:
        c["category"] = cats.get(c["cluster_id"], "Other")

    # §2.7 — scoring, all via the shared metrics module
    log("  scoring config coherence/distinctness (LLM judge) …")
    llm_score = llm.coherence_judge(metrics.select_judge_clusters(clusters))
    cov = metrics.mass_weighted_coverage(clusters, total_hours)
    nmass = metrics.noise_mass(noise_hours, total_hours)
    log(f"  coverage={cov['coverage']:.1%}  noise_mass={nmass:.1%}  "
        f"llm_overall={llm_score['overall']:.1f}  clusters={len(clusters)}  "
        f"named={cov['n_named_clusters']}")

    artifact = {
        "config": name,
        "weights": dict(weights),
        "min_cluster_size": mcs,
        "n_clusters": len(clusters),
        "coverage": cov,
        "noise_mass": nmass,
        "llm_score": llm_score,
        "backends": {"reducer": reducer, "clusterer": clusterer},
        "umap": {"n_components": C.UMAP_N_COMPONENTS, "metric": C.UMAP_METRIC,
                 "min_dist": C.UMAP_MIN_DIST, "n_neighbors": C.UMAP_N_NEIGHBORS,
                 "seed": C.UMAP_SEED},
        "clusters": clusters,
        "labels": labels.tolist(),      # ticket-order assignment (-1 == noise), for stability §4
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"{name}.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


def run_sweep_hdbscan(records, total_hours, *, weight_configs=None,
                      min_cluster_sizes=None, dry_run=False,
                      allow_reducer_fallback=False, out_dir=None, log=print) -> dict:
    """Run every (weight config × min_cluster_size) and write a side-by-side
    summary (Section 3 input)."""
    weight_configs = weight_configs or C.HDBSCAN_WEIGHTS
    min_cluster_sizes = min_cluster_sizes or C.HDBSCAN_MIN_CLUSTER_SIZES
    out_dir = out_dir or f"{C.RESULTS_DIR}/hdbscan/configs"

    needed = sorted({b for w in weight_configs.values() for b, v in w.items() if v > 0})
    blocks = build_feature_blocks(records, needed, dry_run=dry_run, log=log)
    llm = LLM(dry_run=dry_run)

    summary = []
    for wname, weights in weight_configs.items():
        for mcs in min_cluster_sizes:
            cname = f"{wname}__mcs{mcs}"
            art = run_config(cname, weights, mcs, blocks, records, total_hours, llm,
                             allow_reducer_fallback=allow_reducer_fallback,
                             out_dir=out_dir, log=log)
            if art is None:
                continue
            summary.append({
                "config": cname,
                "weights": art["weights"],
                "min_cluster_size": mcs,
                "n_clusters": art["n_clusters"],
                "coverage": art["coverage"]["coverage"],
                "noise_mass": art["noise_mass"],
                "n_named": art["coverage"]["n_named_clusters"],
                "coherence": art["llm_score"]["coherence"],
                "distinctness": art["llm_score"]["distinctness"],
                "llm_overall": art["llm_score"]["overall"],
            })

    summary.sort(key=lambda s: (-s["coverage"], -s["llm_overall"]))
    Path(f"{C.RESULTS_DIR}/hdbscan").mkdir(parents=True, exist_ok=True)
    Path(f"{C.RESULTS_DIR}/hdbscan_summary.json").write_text(
        json.dumps({"configs": summary, "llm_usage": llm.usage}, indent=2), encoding="utf-8")
    log("\n── HDBSCAN sweep summary (Section 3 — human picks config*) ──")
    log(f"{'config':<26}{'clust':>6}{'cover':>8}{'noise':>8}{'named':>7}{'overall':>9}")
    for s in summary:
        log(f"{s['config']:<26}{s['n_clusters']:>6}{s['coverage']:>8.1%}"
            f"{s['noise_mass']:>8.1%}{s['n_named']:>7}{s['llm_overall']:>9.1f}")
    return {"configs": summary, "llm_usage": llm.usage}


if __name__ == "__main__":
    from records import load_records

    ap = argparse.ArgumentParser(
        description="ENT-2289 UMAP+HDBSCAN sweep (Section 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=f"{C.CACHE_DIR}/records.jsonl",
                    help="records.jsonl from `python run.py --stage records`")
    ap.add_argument("--config", default=None, help="run only this named weight config")
    ap.add_argument("--min-cluster-size", type=int, default=None,
                    help="run only this min_cluster_size (else sweeps the config list)")
    ap.add_argument("--dry-run", action="store_true",
                    help="stub embeddings + LLM (zero API spend)")
    ap.add_argument("--allow-reducer-fallback", action="store_true",
                    help="substitute TruncatedSVD if umap-learn is missing (smoke tests only)")
    args = ap.parse_args()

    recs = load_records(args.records)
    # Coverage denominator is the WHOLE corpus (incl. text-poor drops), same as
    # k-means — read it from records_meta.json when present, else sum kept hours.
    meta_path = Path(args.records).with_name("records_meta.json")
    total_hours = (json.loads(meta_path.read_text())["total_hours"]
                   if meta_path.exists() else sum(r.hours for r in recs))

    wcfg = {args.config: C.HDBSCAN_WEIGHTS[args.config]} if args.config else None
    mcs_list = [args.min_cluster_size] if args.min_cluster_size else None
    run_sweep_hdbscan(recs, total_hours, weight_configs=wcfg, min_cluster_sizes=mcs_list,
                      dry_run=args.dry_run, allow_reducer_fallback=args.allow_reducer_fallback)
