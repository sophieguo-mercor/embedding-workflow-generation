#!/usr/bin/env python3
"""
Prompt builders for the three LLM passes:
  * name_cluster        — §2.4  one call per cluster: title + description + coherence flag
  * assign_categories   — §2.5  one batched call over all named clusters: shared category labels
  * coherence_judge     — §2.6  combo-level coherence + distinctness score, blind to combo

All prompts demand English output regardless of source-note language
("English output regardless of source-note language" — Done-when).
"""
from __future__ import annotations

import json

# ── §2.4 per-cluster naming ──────────────────────────────────────────────────

NAME_SYSTEM = """You are labelling clusters of IT-support tickets to induce a workflow rubric.
A "workflow" is one recurring, repeatable support process (e.g. "Password Reset",
"New User Onboarding", "Backup Failure Remediation").

You will see a sample of tickets from ONE cluster (title, category labels, and
resolution notes). The notes may be in Dutch or English; ALWAYS write your output
in English.

Return ONLY a JSON object:
{
  "title": "<short workflow name, Title Case, <=8 words>",
  "description": "<1-2 sentences: what the recurring process is and how it's resolved>",
  "coherent": <true|false>,
  "coherence_note": "<if false: name the 2+ distinct processes you see; else "">"
}

Set "coherent" to false when the sample clearly describes TWO OR MORE distinct
processes fused together (a coarse split), not one workflow. Judge only from this
sample. Output the JSON and nothing else."""


def build_name_user(samples: list[dict], keywords: list[str] | None = None) -> str:
    """samples: list of {title, issue_type, sub_issue_type, notes} for the cluster.
    keywords: optional distinctive c-TF-IDF terms for the cluster (ENT-2289 §2.5).
    When omitted the prompt is identical to the ENT-2261 k-means naming call."""
    lines = ["Tickets in this cluster:\n"]
    for i, s in enumerate(samples, 1):
        lines.append(f"--- ticket {i} ---")
        if s.get("title"):
            lines.append(f"title: {s['title']}")
        cat = " / ".join(x for x in (s.get("issue_type"), s.get("sub_issue_type")) if x)
        if cat:
            lines.append(f"labels: {cat}")
        if s.get("notes"):
            lines.append(f"notes: {s['notes'][:600]}")
        lines.append("")
    if keywords:
        lines.append("Distinctive keywords for this cluster (c-TF-IDF, most "
                     "characteristic terms): " + ", ".join(keywords))
        lines.append("")
    return "\n".join(lines)


# ── §2.5 batched category assignment ─────────────────────────────────────────

CATEGORY_SYSTEM = """You are grouping named IT-support workflows into a small set of
CATEGORIES (higher-level themes). You will receive the FULL list of this run's
named workflows (title + description) at once.

Assign each workflow exactly one category. REUSE a category label across related
workflows rather than minting a new one per workflow — aim for roughly 6-12
categories total. Categories are generated bottom-up from THIS list only; do not
invent labels for workflows that aren't present. Output English.

Return ONLY a JSON array, one object per input workflow, in the same order:
[{"id": <int, the workflow's id>, "category": "<category label>"}]"""


def build_category_user(named: list[dict]) -> str:
    """named: list of {id, title, description}."""
    payload = [{"id": n["id"], "title": n["title"], "description": n["description"]} for n in named]
    return "Named workflows:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


# ── §2.6 combo-level coherence / distinctness judge ──────────────────────────

JUDGE_SYSTEM = """You are scoring the QUALITY of a candidate workflow taxonomy induced
by clustering IT-support tickets. You are blind to how it was produced, so your
score is comparable across candidates.

You will receive a list of named clusters (title + description + a few example
ticket snippets). Judge two things against this fixed rubric:

COHERENCE (0-10): does each cluster read as ONE coherent recurring process, with
its examples consistent with its title/description? (10 = every cluster is a
clean single process; 0 = clusters are grab-bags.)

DISTINCTNESS (0-10): are the clusters distinct from EACH OTHER, with minimal
overlap/near-duplication? (10 = no two clusters describe the same process; 0 =
heavy duplication.)

Return ONLY JSON:
{"coherence": <0-10 float>, "distinctness": <0-10 float>,
 "overall": <0-10 float>, "notes": "<one sentence>"}"""


def build_judge_user(clusters: list[dict]) -> str:
    """clusters: list of {title, description, examples: [str]}."""
    lines = ["Candidate taxonomy clusters:\n"]
    for i, c in enumerate(clusters, 1):
        lines.append(f"[{i}] {c['title']} — {c['description']}")
        for ex in c.get("examples", [])[:3]:
            lines.append(f"      e.g. {ex[:200]}")
    return "\n".join(lines)


# ── §2.8 MECE consolidation ──────────────────────────────────────────────────
# The naming pass (§2.4) over-fragments: the same process is split across many
# clusters. This pass sees ALL named clusters at once and merges them into a MECE
# CATEGORY -> WORKFLOW taxonomy. The per-cluster `category` from §2.5 is withheld
# so the merge is unsupervised. The `coh` flag is passed as a *prior* (Option C):
# an incoherent cluster is placed only if its description names one process, else
# it routes to the residual regardless of its volume.

CONSOLIDATE_SYSTEM = """You are a taxonomy architect consolidating a large set of micro-clusters of IT
support tickets into a clean, actionable operational taxonomy.

Each input item is a micro-cluster produced by density clustering. It is described
by a title, a description, representative keywords, a few example ticket subjects,
its volume (ticket count and hours), and a `coh` (coherent) flag. These micro-clusters
are over-fragmented: the SAME underlying process is often split across many clusters
(e.g. new-user onboarding, remote-desktop login), and some clusters are text-poor,
incoherent, or span several unrelated processes.

Your job: build a two-level semantic hierarchy over these clusters —
CATEGORY -> WORKFLOW -> (member clusters) — that is MECE:

- MUTUALLY EXCLUSIVE: no two workflows may describe the same underlying process.
  Each input cluster is assigned to exactly ONE workflow.
- COLLECTIVELY EXHAUSTIVE: every input cluster id must appear somewhere. Clusters
  that are incoherent, text-poor, or genuinely span multiple unrelated processes go
  into a single top-level `unclassified` bucket rather than being forced into a
  workflow they only partially fit.

A WORKFLOW is a single distinct recurring operational process an agent performs
(its typical trigger + typical resolution). A CATEGORY is a broad theme grouping
related workflows.

Rules:
- Merge by PROCESS, not by surface features. Do NOT merge two clusters merely
  because they share a vendor, product, or keyword. DO merge clusters that represent
  the same recurring task even if their wording differs. Keep genuinely distinct
  sub-processes separate even when they share vocabulary.
- Prioritize DISTINCTNESS. Aim for workflows that are each individually actionable —
  neither hundreds of near-duplicates nor a handful of mega-buckets. As a soft guide,
  target roughly 50-90 workflows, but let the semantics decide; never split or merge
  solely to hit a count.
- Base decisions primarily on the description; keywords and examples are supporting
  evidence. Volume (hours/tickets) may inform whether a fine distinction among
  coherent clusters is worth preserving (large clusters merit their own workflow),
  but volume is NEVER the sole reason to merge distinct processes.
- The `coh` flag marks clusters an earlier pass judged incoherent (text-poor, or
  spanning multiple processes). Treat coh:false as a prior that the cluster is
  UNRELIABLE, not as a verdict. Place a coh:false cluster in a workflow ONLY if its
  description clearly and unambiguously identifies a single recurring process. If the
  description is itself generic or mixed (e.g. "miscellaneous", "general IT support",
  "administration and support tasks"), route it to `unclassified` — regardless of how
  many hours or tickets it carries. Do NOT keep a vague cluster out of `unclassified`
  merely because it is large.
- Name each workflow for the operational process it represents; write a one-to-two
  sentence description covering the recurring task, its trigger, and typical resolution.
- Only use cluster ids that appear in the input. Assign each id exactly once across
  all workflows plus unclassified. Do not invent ids or members.

Output STRICT JSON only, no prose outside the JSON, matching this schema:

{
  "taxonomy": [
    {
      "category": "string",
      "category_description": "string",
      "workflows": [
        {
          "name": "string",
          "description": "string",
          "member_cluster_ids": [int, ...],
          "merge_note": "string - why these clusters are one process; \\"\\" if a singleton"
        }
      ]
    }
  ],
  "unclassified": {
    "member_cluster_ids": [int, ...],
    "note": "string - brief reason these were not placeable"
  }
}"""


def build_consolidate_user(clusters: list[dict]) -> str:
    """One compact JSON line per cluster. The prior `category` field is intentionally
    omitted so the merge is unsupervised (not anchored to §2.5's rollup)."""
    lines = [f"Here are the {len(clusters)} micro-clusters to consolidate. One JSON "
             "object per line: fields - id, t(itle), d(escription), kw(keywords), "
             "ex(examples), nt(n_tickets), h(hours), coh(erent).\n"]
    for c in clusters:
        lines.append(json.dumps({
            "id": c["cluster_id"],
            "t": c.get("title", ""),
            "d": c.get("description", ""),
            "kw": c.get("keywords", [])[:8],
            "ex": c.get("examples", [])[:3],
            "nt": c.get("n_tickets", 0),
            "h": round(c.get("hours", 0.0), 1),
            "coh": bool(c.get("coherent", True)),
        }, ensure_ascii=False))
    lines.append(f"\nBuild the MECE CATEGORY -> WORKFLOW taxonomy per your "
                 f"instructions. Assign every one of the {len(clusters)} ids exactly "
                 f"once. Return only the JSON object.")
    return "\n".join(lines)


# Targeted repair — place only the clusters the model omitted, into EXISTING
# workflows (or unclassified). Far more reliable than regenerating the full partition.
PLACE_SYSTEM = """You are placing a few IT-support ticket micro-clusters into an EXISTING workflow
taxonomy. You will get the list of existing workflows (name + category + description)
and a small set of unplaced clusters. For each unplaced cluster, choose the single
existing workflow whose recurring process it best matches. If none is a clear,
unambiguous match — or the cluster is text-poor / generic / spans multiple processes
— assign it to "unclassified". Match by PROCESS, not shared keywords/vendors. Do not
invent workflow names; use an existing name verbatim, or exactly "unclassified".

Output STRICT JSON only: {"assignments": [{"id": int, "workflow": "exact name or unclassified"}]}"""


def build_place_user(missing: list[int], clusters: list[dict], wf_index: list[dict]) -> str:
    """missing: cluster ids to place. clusters: full cluster list (for lookup).
    wf_index: [{workflow, category, description}] of the existing workflows."""
    by_id = {c["cluster_id"]: c for c in clusters}
    miss_lines = []
    for i in missing:
        c = by_id.get(i, {})
        miss_lines.append(json.dumps({
            "id": i, "t": c.get("title", ""), "d": c.get("description", ""),
            "kw": c.get("keywords", [])[:8], "coh": bool(c.get("coherent", True)),
        }, ensure_ascii=False))
    return ("EXISTING WORKFLOWS:\n" + json.dumps(wf_index, ensure_ascii=False)
            + "\n\nUNPLACED CLUSTERS (one JSON object per line):\n" + "\n".join(miss_lines)
            + "\n\nAssign each unplaced id to an existing workflow name or "
              '"unclassified". Return only the JSON object.')
