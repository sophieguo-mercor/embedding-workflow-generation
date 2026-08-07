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
