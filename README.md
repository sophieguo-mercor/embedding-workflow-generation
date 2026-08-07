# Workflow Generation from TechOne Tickets (ENT-2261 · ENT-2289)

Induce a workflow rubric from TechOne IT-support tickets by clustering them in
**semantic vector space**, so paraphrases and Dutch↔English wordings of the same
process collapse into one workflow — closing the gap where keyword clustering
([ENT-2260](https://linear.app/mercor/issue/ENT-2260)) under-counts recurring
workflows worded differently.

The repo houses **two candidate methods on one shared spine**, selected with
`--method`:

- **k-means** ([ENT-2261](https://linear.app/mercor/issue/ENT-2261), default) —
  weighted per-field embeddings + a categorical TF-IDF block, clustered with
  KMeans, `k` chosen by silhouette against a ~72-workflow target.
- **UMAP + HDBSCAN** ([ENT-2289](https://linear.app/mercor/issue/ENT-2289)) — a
  BERTopic-style pipeline that adds an **LLM normalization** pass, reduces with
  UMAP, and clusters with HDBSCAN (variable-density clusters + per-point noise, no
  target `k`).

Both share the join, cleaning, embeddings, categorical block, the **`metrics.py`**
scoring (coverage / "Other" gate / LLM-coherence judge / noise mass), the PII
gate, and the finalize output contract — so both produce a rubric in the same
`Name | Category | Description` shape the dashboard consumes, scored the **same
way**, ready for an eventual head-to-head bake-off (the bake-off run itself is out
of scope here). Keeping the scorer shared is what makes the two comparable; see
`metrics.py`.

## Method A — k-means sweep (ENT-2261)

The shared spine (stages 0–1.5) plus the k-means clustering core. Scoring and the
finalize output live in the shared `metrics.py` / `finalize.py`.

| Stage | Module | What it does |
|------|--------|--------------|
| **0. Clean** (§1.1) | `llm_clean.py`, `clean.py` | *(optional, shared)* Strip email boilerplate and write cleaned twins with identical schema (`data/cleaned/*.cleaned.xlsx`). **Descriptions** are cleaned by an **Anthropic Message Batch** (an LLM handles signatures / quoted threads / the fuzzy cases); **notes** stay deterministic regex. Reusable downstream; **read by default** by later stages (pass `--use-raw` to opt out). See [Text cleaning](#text-cleaning-11) below. |
| **1. Records** (§1) | `records.py` | Join Tickets + Time-entries on `(instance, ticketnumber)`; one feature-ready record per ticket (`title`, `description`, `notes`, `issue_type`, `sub_issue_type`, `hours`, `touches`). Drops text-poor tickets; identifiers/dates/person-IDs never enter features. |
| **1.4 Semantic block** | `embeddings.py` | Embed `title`/`description`/`notes` separately with OpenAI `text-embedding-3-large` (handles Dutch/English — no translation; reduced to 1024-d via the `dimensions` param). Cached per `(ticket_id, field)` so re-runs never re-embed. Token-aware batching keeps every request under OpenAI's per-request limit (budgets on `max(tiktoken, chars/4)` to match OpenAI's own upper-bound guard), with a split-and-retry fallback. |
| **1.5 Categorical block** | `categorical.py`, `synonyms.json` | Normalize `issue_type`/`sub_issue_type` (rule-based canonicalization + curated synonym map), pool into namespaced tokens (`issue=… sub=…`), `TfidfVectorizer(min_df=2, smooth_idf=True)`, L2-normalize. |
| **2. Sweep** (§2) | `sweep.py`, `prompts.py`, `llm.py` | For each hand-picked weight combo `(w_title, w_desc, w_notes, w_cat)`: build the combined vector, pick `k` by silhouette (within-combo only), KMeans, name each cluster (1 LLM call → title/description/coherence flag), batched category pass, hours-weighted coverage + a blind combo-comparable LLM coherence/distinctness score. |
| **3. Human review** (§3) | `results/sweep_summary.json` | All combos laid out side by side — coverage, LLM score, `k`, cluster count, named clusters. A human picks `combo*`. |
| **4. Stability** (§4) | `stability.py` | Reseed KMeans on `combo*` 2–3×; report pairwise ARI; flag if agreement is low. |
| **5. Finalize** (§5) | `finalize.py`, `pii.py` | PII gate (hard-fail on structured fields, blank prose), then freeze a versioned `taxonomy.md` / `clusters.jsonl` / `metrics.json` recording combo/weights/`k`/model versions/dataset window. |

### The combined feature vector

```
X = L2( concat( w_f · L2(emb_f) for f in {title,description,notes} if w_f>0 )
        ++ w_cat · L2(tfidf_cat) )
```
A zero-weighted field is **dropped**, not just zeroed. The weight shortlist lives
in `config.py` (`WEIGHT_VECTORS`), along with the `k` range, mass/coherence
thresholds, and model versions.

### Coverage & the "Other" bucket

Coverage is **hours-weighted** — the % of engineer-hours in named, coherent,
big-enough clusters. A ticket's hours land in "Other" if it was dropped as
text-poor (§1.2), its cluster is below the mass floor, or its cluster failed the
coherence flag (§2.4). The denominator is the whole corpus (including dropped
tickets), so coverage can't trivially sit near 100% for every combo.

### Text cleaning (§1.1)

The raw `description` column is an HTML-to-text **email dump**, not clean prose:
measured on a 20k-row sample, ~42% carry double-encoded `_x000D_` escapes, ~29%
an *"Incoming Email Processor"* footer, ~26% a signature block, ~17% inline-image
refs, plus tracking-link wrappers and `????` mojibake. That boilerplate is
**60–75% of the bytes**; the actual signal — the requester's ask / symptom /
affected system — is a short block at the top (median ~274 chars). Left in, the
signatures and quoted threads pull clustering toward *sender/company* instead of
*issue*. The time-entry `summarynotes` / `internalnotes` are already clean
engineer prose (the same patterns are <0.5% there).

Cleaning splits by column, because the two need different tools:

**Descriptions → an LLM (`llm_clean.py`, Anthropic Message Batch).** Signatures
vary endlessly and ~8% of descriptions carry genuine quoted reply threads whose
boundaries a regex gets wrong. The model is told to return only the requester's
own wording — the ask, symptom, affected system / person / device — with
signatures, quoted threads, footers, greetings, image refs and tracking URLs
removed; keep the original Dutch/English (no translation, no summarising); return
`""` if nothing substantive remains. Before sending, a cheap **deterministic
prepass** (`clean.prepass_description`) strips encoded escapes, `[cid:]` refs and
giant tracking URLs to cut tokens — but leaves signatures/threads for the model.

The batch harness mirrors the sibling `resolution-extraction` pipeline and is
resume-safe on two levels: `cache/desc_clean.jsonl` (ids already cleaned are
skipped) and `cache/desc_batch_manifest.json` (an in-flight batch is reconnected,
never re-submitted — no double-spend). It packs `BATCH_SIZE` (12) descriptions
per request with a cache_control'd system prefix, polls to completion, and parses
each response as `[{"id": pos, "text": "..."}]`. Empty/short descriptions are
never sent (freebie path); a group whose JSON fails to parse falls back to the
deterministic clean and can be re-submitted on a re-run. Illegal control chars in
model output are stripped before writing (openpyxl rejects them).

*Scale:* ~138k descriptions → ~11.5k requests in one batch; minutes-to-hours of
(50%-priced) batch latency. Run `--dry-run` first for a free request/token
estimate; `--sample N` for a cheap quality check.

**Notes → deterministic regex (`clean.py`, no LLM, no network).** The time-entry
notes are already clean, so they get a fast structural scrub: decode `_x000D_`
escapes, strip `[cid:]`, unwrap `text<mailto:/tel:/http>` links, strip known HTML
tags, fix `????` mojibake, collapse whitespace — **keeping** technical URLs
(signal in an engineer's note). The same regex path is also the fallback for any
description the LLM didn't cover.

**Shared choices:** **PII is intentionally kept** for reuse fidelity — `pii.py`
still redacts at publish time (§5.1). Nullish cells (`nan`/`none`/`null`) → `""`.
The cleaned twins have the **same sheet (`Export`) and headers** as the raw
exports, so the pipeline (and anything else) reads them unchanged; they live in
`data/cleaned/` (git-ignored — still may contain PII). `ANTHROPIC_API_KEY` is
read from the environment / `.env` only.

## Method B — UMAP + HDBSCAN sweep (ENT-2289)

A **BERTopic-style** third candidate that swaps the clustering core: clean → **LLM
normalization** → embed → **UMAP** → **HDBSCAN** → c-TF-IDF + LLM naming. It reuses
the join, cleaning, embeddings, categorical block, `metrics.py` scoring, PII gate,
and finalize output contract verbatim — only the clustering core and the new
normalization pass differ, so its rubric stays directly comparable to k-means'.

Why HDBSCAN: the ~72-workflow target was dropped, which removes the main reason to
use k-means (direct control of `k`) and exposes its bias toward even, spherical
clusters on our long-tailed data (`touches` median 1, tail past 200). HDBSCAN
finds variable-density, variable-size clusters in one pass and labels bad-fit
tickets as **noise per-point**, instead of forcing every ticket into its nearest
centroid. `min_cluster_size` is also a natural expression of the mass-floor rule.

| Stage | Module | What it does |
|------|--------|--------------|
| **1.5 Normalize** *(new)* | `normalize.py` | One Anthropic Message-Batch call per ticket over `title` + cleaned `description`/`notes` → `normalized_issue` / `normalized_resolution` (English, consistent phrasing). The biggest new cost line (~154k tickets), so it's **batched + cached** in `cache/normalize.jsonl` and resume-safe (manifest reconnect) — paid once, reused by every sweep. See [Normalization](#normalization-15). |
| **1.6 Semantic blocks** | `embeddings.py` | Same encoder; also embeds `normalized_issue` / `normalized_resolution`, cached per `(ticket_id, field)`. |
| **2. Sweep** (§2) | `sweep_hdbscan.py` | Per config `(weight vector over 6 blocks, min_cluster_size)`: build the combined vector → **UMAP** → **HDBSCAN** → **c-TF-IDF** keywords per cluster → 1 LLM naming call (keywords passed alongside; members sampled by membership **probability**, not centroid) → batched category pass → shared coverage + LLM coherence + a separate **noise-mass** metric. |
| **3. Human review** (§3) | `results/hdbscan_summary.json` | Configs side by side — coverage, LLM score, **noise mass**, cluster count, `min_cluster_size`, named clusters. A human picks `config*`. |
| **4. Bootstrap stability** (§4) | `stability_hdbscan.py` | HDBSCAN is deterministic (UMAP seed fixed), so reseeding tests nothing. Instead resample tickets with replacement, re-cluster, and measure per-cluster **Jaccard** recovery (Hennig's `clusterboot`): `>0.75` stable, `0.60–0.75` doubtful, `<0.60` unstable. Low-stability clusters are **flagged, not dropped**. |
| **5. Finalize** (§5) | `finalize.py` (`finalize_hdbscan`), `pii.py` | Same PII gate + `Name \| Category \| Description` freeze as k-means (shared writers), into `results/hdbscan/`. `clusters.jsonl` keeps the c-TF-IDF keywords + per-cluster stability verdict; `metrics.json` records config / weights / `min_cluster_size` / UMAP params + seed / model versions. |

### The combined feature vector (6 blocks)

```
X  = L2( concat( w_f · L2(block_f)
         for f in {title, description, notes,
                   normalized_issue, normalized_resolution, cat} if w_f>0 ) )
Xr = UMAP(X, n_components=5, metric='cosine', min_dist=0.0, random_state=fixed)
labels, noise = HDBSCAN(Xr, min_cluster_size=mcs)     # label -1 == noise
```

A zero-weighted block is **dropped**, not zeroed. The hand-picked configs live in
`config.py` (`HDBSCAN_WEIGHTS`) and **ablate** what's in question rather than
grid-search: `raw_only`, `normalized_only`, `normalized_categorical`,
`raw_categorical`, `all_equal` — each swept over `HDBSCAN_MIN_CLUSTER_SIZES`
(25/50/100). A config whose `normalized_*` blocks aren't populated yet (before
`normalize.py` runs) is **skipped with a warning**, never silently clustered on
empty columns. **Noise mass** (% of engineer-hours HDBSCAN labeled noise) is
tracked separately from coverage, so a reviewer can tell low coverage from
too-aggressive noise apart from low coverage from failed coherence flags.

### Normalization (§1.5)

Cleaning (§1.1) removes boilerplate; normalization goes further and extracts
*intent* — a 2026 FLAIRS study found LLM semantic normalization before embedding
was the single largest contributor to support-ticket cluster quality. For each
ticket, one call to a **cheap high-throughput model** (`NORMALIZE_MODEL`, Haiku by
default) restates the request as two consistent English fields: `normalized_issue`
(the root problem) and `normalized_resolution` (what was done, from the notes; `""`
if the notes don't say). The **raw fields are kept too**, so the sweep can *ablate*
normalized vs. raw rather than assume normalization helps.

The batch harness mirrors §1.1's `llm_clean.py` and is resume-safe on two levels:
`cache/normalize.jsonl` (`{id, issue, resolution}` per ticket — skipped on re-runs;
the durable store, survives records rebuilds) and
`cache/normalize_batch_manifest.json` (an in-flight batch is reconnected, never
re-submitted). `normalize.attach_normalized()` merges the cache back onto records;
the sweep calls it automatically when the cache exists. Run `--dry-run` for a free
estimate, `--sample N` for a cheap quality check.

**Requires** `umap-learn` and `hdbscan` (in `requirements.txt`). If `umap-learn`
isn't installed the sweep errors, unless you pass `--allow-reducer-fallback` — a
**linear TruncatedSVD** stand-in for wiring smoke tests only, recorded in the
artifact's `backends` (never a silent downgrade). The `hdbscan` package
auto-falls-back to scikit-learn's `HDBSCAN` if absent.

## Setup

Use a virtual environment (the `.venv/` directory is git-ignored, so it never gets committed):

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# API keys are read from the environment only — never hardcoded or committed.
cp .env.example .env             # then fill in your real keys
set -a; source .env; set +a      # load them into the current shell
```

Re-activate with `source .venv/bin/activate` in any new shell before running the pipeline.

`.env` is git-ignored; `.env.example` is the committed template. It holds two keys:
`OPENAI_API_KEY` (embeddings, §1.4) and `ANTHROPIC_API_KEY` (naming/category/judge, §2).
If you prefer, `export OPENAI_API_KEY=... ANTHROPIC_API_KEY=...` by hand instead of sourcing `.env`.

Place the two Power BI exports in `data/` (git-ignored — large and may contain PII):
`Tickets Mercor v3.xlsx`, `Time entries Mercor v3.xlsx`.

## Usage

`--method {kmeans,hdbscan}` selects the clustering method for the
`sweep`/`stability`/`finalize` stages (default `kmeans`). Stages 0–1 (clean,
records) are shared; `normalize` (§1.5) feeds only the hdbscan method. `.env` is
loaded automatically for every stage.

```bash
# End-to-end wiring smoke test on 500 tickets, ZERO API spend (stub embeddings/LLM):
python run.py --dry-run --limit 500

# ── Shared prep ────────────────────────────────────────────────────────────
# (Optional) Clean the raw exports → data/cleaned/*.cleaned.xlsx (§1.1).
python run.py --stage clean --dry-run     # free: prompt + request/token estimate
python run.py --stage clean --sample 24   # cheap quality check on 24 tickets
python run.py --stage clean               # full run (~11.5k requests, batch-priced)
python run.py --stage clean --collect     # reconnect to an in-flight batch and finish

# Build records once (cached to cache/records.jsonl; source-aware auto-rebuild).
# Reads the cleaned twins by default; add --use-raw for the raw exports:
python run.py --stage records [--use-raw]

# ── Method A: k-means (ENT-2261, default) ──────────────────────────────────
python run.py --stage sweep                          #   → pick combo* from the summary
python run.py --stage finalize --select notes_weighted   # stability (ARI) + PII-gate + freeze

# ── Method B: UMAP + HDBSCAN (ENT-2289) ────────────────────────────────────
python run.py --stage normalize --dry-run            # free request/token estimate
python run.py --stage normalize --sample 24          # cheap quality check
python run.py --stage normalize                      # full run (batch-priced, cached once)
python run.py --stage sweep     --method hdbscan     #   → pick config* from the summary
python run.py --stage finalize  --method hdbscan --select raw_categorical__mcs50
#   hdbscan sweep/finalize also accept --config / --min-cluster-size / --allow-reducer-fallback

# Tests (data-free, no API):
python tests/test_pipeline.py
python tests/test_metrics.py
python tests/test_clean.py
```

`make preprocess` / `make smoke` / `make records [USE_RAW=1]` / `make sweep` /
`make finalize COMBO=<name>` wrap the k-means track.

## Outputs (`results/`, git-ignored)

**k-means** (`results/`):
- `combos/<combo>.json` — per-combo: named clusters, coverage, LLM scores, `k`, silhouette-by-k, cluster labels.
- `sweep_summary.json` — all combos side by side for human selection (§3).
- `stability.json` — pairwise ARI across reseeds and the stable/unstable verdict (§4).
- `taxonomy.md` / `clusters.jsonl` / `metrics.json` — the frozen rubric, per-cluster detail, and provenance.

**HDBSCAN** (`results/hdbscan/`, so it never clobbers the k-means rubric):
- `configs/<config>__mcs<n>.json` — per-config: named clusters + c-TF-IDF keywords, coverage, LLM scores, **noise mass**, `min_cluster_size`, UMAP params, backends, cluster labels (`-1` = noise).
- `hdbscan_summary.json` — all configs side by side (adds noise mass) for human selection (§3).
- `stability.json` — per-cluster bootstrap **Jaccard** + verdict, and the flagged low-stability clusters (§4).
- `taxonomy.md` / `clusters.jsonl` / `metrics.json` — same shape as k-means; `clusters.jsonl` also carries the keywords + per-cluster stability verdict, `metrics.json` the hdbscan provenance.

## Open questions & scope (from the tickets, unresolved by design in v1)

- **Shared bake-off code — done:** coverage, the "Other" rule, and the coherence judge now live in one `metrics.py` both methods call, so their scores are directly comparable. Running the actual three-way bake-off (vs. ENT-2260 keywords) is still out of scope.
- **Coherence-flag routing (§2.4/§6):** a flagged cluster's hours go to "Other" but nothing re-splits it (HDBSCAN's hierarchy makes selective re-splitting more feasible — not implemented in v1).
- **Thresholds:** the mass floor, ARI/Jaccard bars, `min_cluster_size` range, and the noise-mass level that disqualifies a config are `config.py` defaults tuned for the full ~154k corpus — review them, and expect low coverage/stability on small subsamples where clusters can't clear the ticket-count gate.
- **Normalization cost:** the §1.5 pass is ~154k cheap-model calls; budget it, and rely on `cache/normalize.jsonl` so it's paid once across sweep runs.
- **Deferred (ENT-2289 "next steps"):** a sequence / process-trace feature block from the notes (real chance of a null result), and instruction-conditioned embeddings — both left out to validate the core pipeline first. Records preserve the chronological note structure the sequence work needs.
- **Fixed shortlist:** the hand-picked combos/configs don't auto-expand if results tie or all underperform (accepted v1 scope).

## Notes on scope vs. the source pipeline

Naming is simplified to **one** LLM call per cluster (+ one batched category
pass), not the source pipeline's four passes, because this output is a one-shot,
human-reviewed artifact — the reviewer absorbs the MECE-merge/naming-polish work.
Category assignment stays a separate batched pass because it's the one step that
needs cross-cluster visibility to reuse labels across related clusters.
