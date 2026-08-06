# Embedding-based Workflow Generation (ENT-2261)

An **embedding-based** alternative to the keyword-clustering rubric generator in
[ENT-2260](https://linear.app/mercor/issue/ENT-2260). It clusters TechOne
IT-support tickets in **semantic vector space** so paraphrases and Dutch↔English
translations of the same process collapse into one workflow — closing the gap
where keyword clustering under-counts recurring workflows worded differently.

It reuses the clustering + LLM-naming methodology of the internal
`workflow_taxonomy` pipeline, adapted to TechOne tickets, and produces a second
candidate rubric in the `Name | Category | Description` shape the dashboard
classifier already consumes — scored on the **same** coverage + LLM-coherence
metrics as ENT-2260, so the two methods can be compared head-to-head later (the
bake-off itself is out of scope here).

## Pipeline

| Stage | Module | What it does |
|------|--------|--------------|
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

```bash
# End-to-end wiring smoke test on 500 tickets, ZERO API spend (stub embeddings/LLM):
python run.py --dry-run --limit 500

# Build records once (cached to cache/records.jsonl):
python run.py --stage records

# Run the full field-combo sweep (needs both API keys):
python run.py --stage sweep

#   → review results/sweep_summary.json + results/combos/*.json, pick combo*

# Stability check + PII-gate + freeze the chosen combo:
python run.py --stage finalize --select notes_weighted

# Tests (data-free, no API):
python tests/test_pipeline.py
```

`make smoke` / `make records` / `make sweep` / `make finalize COMBO=<name>` wrap these.

## Outputs (`results/`, git-ignored)

- `combos/<combo>.json` — per-combo: named clusters, coverage, LLM scores, `k`, silhouette-by-k, cluster labels.
- `sweep_summary.json` — all combos side by side for human selection (§3).
- `stability.json` — pairwise ARI across reseeds and the stable/unstable verdict (§4).
- `taxonomy.md` — the frozen rubric in `Name | Category | Description` shape.
- `clusters.jsonl` — full per-cluster detail (including non-qualifying clusters).
- `metrics.json` — combo, weights, `k`, coverage, LLM scores, model versions, dataset window, thresholds.

## Open questions (from the ticket, unresolved by design in v1)

- **Coherence-flag routing (§2.4/§6):** a flagged cluster's hours go to "Other" but nothing re-splits it; reaching ~72 workflows may still need manual re-clustering.
- **Thresholds:** the mass floor, "low" ARI, and any coverage/LLM bar for combo selection are set to defaults in `config.py` and are meant to be reviewed.
- **Shared bake-off code:** for a truly comparable comparison with ENT-2260, the coverage definition, "Other" rule, and coherence rubric should become one shared function/prompt.
- **Fixed shortlist:** the ~8 combos don't auto-expand if results tie or all underperform (accepted v1 scope).

## Notes on scope vs. the source pipeline

Naming is simplified to **one** LLM call per cluster (+ one batched category
pass), not the source pipeline's four passes, because this output is a one-shot,
human-reviewed artifact — the reviewer absorbs the MECE-merge/naming-polish work.
Category assignment stays a separate batched pass because it's the one step that
needs cross-cluster visibility to reuse labels across related clusters.
