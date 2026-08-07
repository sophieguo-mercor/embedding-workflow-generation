#!/usr/bin/env python3
"""
Central configuration for the embedding-based workflow-generation pipeline
(ENT-2261).

Everything the sweep needs to be reproducible lives here: the weight-vector
shortlist, the k-search range, the mass/coherence thresholds, and the model
versions. `finalize.py` records the resolved values into `metrics.json` so a
frozen rubric can always be traced back to the settings that produced it.
"""
from __future__ import annotations

# ── Column names ─────────────────────────────────────────────────────────────
# Power-BI export headers. Kept here so a schema change is a one-file edit.
class TicketCols:
    instance = "all_tickets_tasks[instance]"
    ticketnumber = "all_tickets_tasks[ticketnumber]"
    ticket_id = "all_tickets_tasks[id_unique_id]"
    title = "all_tickets_tasks[title]"
    description = "all_tickets_tasks[description]"
    issue_type = "ticket_issue_type[label]"
    sub_issue_type = "ticket_sub_issue_type[label]"
    hours = "[SumTotalHoursWorked]"


class TimeEntryCols:
    instance = "all_time_entries[instance]"
    ticketnumber = "all_time_entries[ticketnumber]"
    startdatetime = "all_time_entries[startdatetime]"
    summarynotes = "all_time_entries[summarynotes]"
    internalnotes = "all_time_entries[internalnotes]"
    hours = "[Sumhoursworked]"


# ── Feature fields ───────────────────────────────────────────────────────────
SEMANTIC_FIELDS = ("title", "description", "notes")  # embedded, in weight order
# Full per-field weight vector order used everywhere: (title, description, notes, cat)
WEIGHT_ORDER = ("title", "description", "notes", "cat")

# ── The field-combo sweep shortlist (§2) ─────────────────────────────────────
# Each combo is (w_title, w_description, w_notes, w_cat). A zero weight DROPS the
# field (it is not concatenated at all), not merely down-weights it. Hand-picked,
# 5–10 combos, per the ticket — no adaptive expansion in v1.
WEIGHT_VECTORS: dict[str, tuple[float, float, float, float]] = {
    "labels_only":      (0.0, 0.0, 0.0, 1.0),
    "notes_only":       (0.0, 0.0, 1.0, 0.0),
    "description_only": (0.0, 1.0, 0.0, 0.0),
    "title_notes":      (1.0, 0.0, 1.0, 0.0),
    "all_equal":        (1.0, 1.0, 1.0, 1.0),
    "notes_weighted":   (0.3, 0.3, 1.0, 0.5),
    "semantic_no_cat":  (1.0, 1.0, 1.0, 0.0),
    "title_desc_cat":   (1.0, 1.0, 0.0, 1.0),
}

# ── k search range (§2 step 2) ───────────────────────────────────────────────
# Retuned UP from the source pipeline's ~20–40: TechOne's rubric needs ~72
# fine-grained workflows. Silhouette is expected to undershoot 72 (it favours
# coarser splits); finer granularity comes from naming/review, not k alone.
GLOBAL_K_MIN = 30
GLOBAL_K_MAX = 90
TARGET_WORKFLOWS = 72

# Hard cap on k so clusters can't be too small to represent a real workflow:
#   k <= total_tickets / MIN_TICKETS_PER_CLUSTER
MIN_TICKETS_PER_CLUSTER = 25

# ── KMeans ───────────────────────────────────────────────────────────────────
KMEANS_SEED = 42
KMEANS_N_INIT = 10
SILHOUETTE_SAMPLE = 5000  # subsample size for silhouette on large N (speed)

# ── Coverage / "Other" bucket thresholds (§2 step 6) ─────────────────────────
# A cluster's hours go to "Other" if it fails ANY of: text-poor drop (§1.2),
# below this mass floor, or its coherence flag fired at naming time (§2.4).
MIN_CLUSTER_MASS_FRAC = 0.003   # cluster must hold >=0.3% of total engineer-hours
MIN_CLUSTER_TICKETS = MIN_TICKETS_PER_CLUSTER

# ── Stability check (§4) ─────────────────────────────────────────────────────
STABILITY_RESEEDS = 3           # extra KMeans runs on combo* with different seeds
LOW_ARI_THRESHOLD = 0.60        # mean pairwise ARI below this is flagged "unstable"

# ── Cluster naming sampling (§2 step 4) ──────────────────────────────────────
NAMING_NEAREST = 8              # centroid-nearest members shown to the namer
NAMING_DIVERSE = 4              # plus a few diverse (far / random) members


# ══ ENT-2289 — BERTopic-style sweep (UMAP + HDBSCAN) ══════════════════════════
# A second clustering method, sharing this repo's feature blocks, LLM passes, and
# the metrics.py scoring. Only the clustering core differs (see sweep_hdbscan.py).
# Everything above (k-means, silhouette) is untouched and still ENT-2261's.

# Feature blocks in weight order. `normalized_issue`/`normalized_resolution` are
# the §1.5 intent-extraction fields — they light up once normalize.py populates
# them on the records; until then, configs that need them are skipped (a config
# is not silently clustered on empty text).
BLOCKS_HDBSCAN = ("title", "description", "notes",
                  "normalized_issue", "normalized_resolution", "cat")

# The hand-picked weight configs (§2 step 1) — chosen to ABLATE what's in
# question (normalization, categorical), not to grid-search. A block absent from
# a dict is weight 0 (dropped, not concatenated). Each is swept over the
# min-cluster-size list below → the full config set is their cross-product.
HDBSCAN_WEIGHTS: dict[str, dict[str, float]] = {
    "raw_only":               {"title": 1.0, "description": 1.0, "notes": 1.0},
    "normalized_only":        {"normalized_issue": 1.0, "normalized_resolution": 1.0},
    "normalized_categorical": {"normalized_issue": 1.0, "normalized_resolution": 1.0, "cat": 1.0},
    "raw_categorical":        {"title": 1.0, "description": 1.0, "notes": 1.0, "cat": 1.0},
    "all_equal":              {b: 1.0 for b in BLOCKS_HDBSCAN},
}

# min_cluster_size sweep (§2 step 3): "how many tickets before this is a real
# workflow, not noise." Picked from business judgment, NOT a target count — this
# method has no target k. Tune for the real ~154k-row corpus.
HDBSCAN_MIN_CLUSTER_SIZES = (25, 50, 100)

# ── UMAP reduction (§2 step 2) ────────────────────────────────────────────────
# Reduce before clustering: compute at 154k rows + distance concentration in high
# dims, and the standard pairing for HDBSCAN. Fixed seed ⇒ reproducible pipeline
# (so Section 4 is a bootstrap check, not a reseed check).
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
UMAP_N_NEIGHBORS = 15
UMAP_SEED = 42

# ── c-TF-IDF keywords (§2 step 4) ─────────────────────────────────────────────
CTFIDF_TOP_N = 10               # distinctive terms kept per cluster (auditable)

# ── HDBSCAN naming sampling (§2 step 5) ───────────────────────────────────────
# Density clusters have no centroid; sample by membership probability instead —
# the principled analog of k-means' centroid-nearest heuristic.
HDBSCAN_NAMING_EXEMPLARS = 8    # highest-probability members shown to the namer
HDBSCAN_NAMING_DIVERSE = 4      # plus a few diverse members

# ── Bootstrap stability (§4) ──────────────────────────────────────────────────
# HDBSCAN is deterministic (UMAP seed fixed), so ENT-2261's reseed-ARI check tests
# nothing here. Instead resample tickets with replacement, re-cluster, and measure
# whether each workflow reappears (Hennig's clusterboot, per-cluster Jaccard) —
# the property that matters for an artifact meant to classify FUTURE tickets.
HDBSCAN_N_BOOTSTRAP = 15        # bootstrap resamples of config* (compute vs. confidence)
# Conventional reading of mean per-cluster Jaccard (Hennig): flag low ones for the
# reviewer, never silently drop them.
JACCARD_STABLE = 0.75           # > this: stable, a genuine reappearing pattern
JACCARD_DOUBTFUL = 0.60         # [DOUBTFUL, STABLE): pattern present, membership doubtful
#                                 < JACCARD_DOUBTFUL: do not trust the cluster

# ── Models ───────────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-large"  # OpenAI; handles Dutch/English — no translation
# text-embedding-3-large is natively 3072-d; we request a reduced 1024-d via the
# OpenAI `dimensions` param for memory parity with clustering (raise to 3072 for
# max fidelity — expect ~3x the embedding cache + per-combo feature-matrix RAM).
EMBED_DIM = 1024
LLM_MODEL = "claude-sonnet-4-6"         # naming / category / coherence judge

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
CACHE_DIR = "cache"
RESULTS_DIR = "results"
TICKETS_XLSX = "data/Tickets Mercor v3.xlsx"
TIME_ENTRIES_XLSX = "data/Time entries Mercor v3.xlsx"
SYNONYMS_JSON = "synonyms.json"

# Cleaned twins written by clean.py (§1.1): same schema, noise stripped from the
# description / notes columns. Reusable downstream; the pipeline reads these when
# run with `--use-cleaned`.
CLEANED_DIR = "data/cleaned"
CLEAN_TICKETS_XLSX = "data/cleaned/Tickets Mercor v3.cleaned.xlsx"
CLEAN_TIME_ENTRIES_XLSX = "data/cleaned/Time entries Mercor v3.cleaned.xlsx"

# ── LLM description cleaner (llm_clean.py, §1.1) ──────────────────────────────
# Descriptions are cleaned by an Anthropic Message Batch (signatures / quoted
# threads / footers removed by the model); notes stay deterministic (clean.py).
BATCH_MODEL = LLM_MODEL                     # claude-sonnet-4-6
BATCH_SIZE = 12                             # descriptions packed per batch request
BATCH_CHAR_LIMIT = 1500                     # per-description char cap sent to the LLM
BATCH_MAX_TOKENS = 4096                     # response cap per request
DESC_CLEAN_CACHE = "cache/desc_clean.jsonl"        # {id, text} per cleaned ticket
DESC_BATCH_MANIFEST = "cache/desc_batch_manifest.json"  # in-flight batch (resume)

# ── LLM intent normalization (normalize.py, ENT-2289 §1.5) ────────────────────
# One Message-Batch call per ticket over title + cleaned description + cleaned
# notes → normalized_issue / normalized_resolution (English). This is the biggest
# NEW cost line (~154k tickets), so it is batched + cached; the cache survives
# across sweep runs so the cost is paid once, not per config.
NORMALIZE_MODEL = "claude-haiku-4-5-20251001"   # cheap, high-throughput (ticket §1.5)
NORMALIZE_BATCH_SIZE = 10             # tickets/request (3 fields each → smaller than §1.4)
NORMALIZE_CHAR_LIMIT = 1200           # per-field char cap sent to the model
NORMALIZE_MAX_TOKENS = 4096           # response cap per request
NORMALIZE_CACHE = "cache/normalize.jsonl"              # {id, issue, resolution} per ticket
NORMALIZE_BATCH_MANIFEST = "cache/normalize_batch_manifest.json"  # in-flight batch (resume)
