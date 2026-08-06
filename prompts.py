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


def build_name_user(samples: list[dict]) -> str:
    """samples: list of {title, issue_type, sub_issue_type, notes} for the cluster."""
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
