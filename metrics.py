#!/usr/bin/env python3
"""
Shared scoring for every workflow-generation method (ENT-2260 keywords,
ENT-2261 k-means, ENT-2289 UMAP+HDBSCAN).

These functions define coverage, the "Other"-bucket rule, and LLM-judge selection
ONCE, so the three methods' rubrics are comparable in the eventual bake-off. They
are pure (no I/O, no network, no LLM call) and method-neutral: they operate on the
cluster contract below, which any clustering method fills in. The LLM coherence
judge itself stays in llm.py — this module only decides WHICH clusters it scores.

Cluster contract
----------------
Every method must produce, per cluster, at least these keys:
    cluster_id:int  title:str  description:str  category:str
    coherent:bool   coherence_note:str
    n_tickets:int   hours:float  hours_frac:float
    examples:list[str]
Method-specific extras (k-means silhouette, HDBSCAN persistence, …) may live
alongside and are ignored here. `mark_qualification` derives `big_enough` and
`qualifies`.
"""
from __future__ import annotations

import config as C


def mark_qualification(
    cluster: dict,
    *,
    min_tickets: int = C.MIN_CLUSTER_TICKETS,
    min_mass_frac: float = C.MIN_CLUSTER_MASS_FRAC,
) -> dict:
    """The §2.6 "Other" gate. A cluster's hours count toward coverage only if it
    is coherent AND big enough (by both ticket count and hours mass). Mutates and
    returns the cluster dict. Identical rule for every method."""
    big_enough = cluster["n_tickets"] >= min_tickets and cluster["hours_frac"] >= min_mass_frac
    cluster["big_enough"] = big_enough
    cluster["qualifies"] = bool(cluster["coherent"]) and big_enough
    return cluster


def mass_weighted_coverage(clusters: list[dict], total_hours: float) -> dict:
    """% of engineer-hours in named, coherent, big-enough clusters vs "Other".

    total_hours is the WHOLE corpus (incl. text-poor drops AND any HDBSCAN noise,
    both of which are simply absent from `clusters`), so coverage can't trivially
    sit near 100%."""
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


def noise_mass(noise_hours: float, total_hours: float) -> float:
    """% of engineer-hours the clustering labeled noise. 0.0 for methods with no
    noise concept (k-means force-assigns every ticket). Reported SEPARATELY from
    coverage so a reviewer can distinguish low coverage from too-aggressive noise
    from low coverage from failed coherence flags — those call for opposite fixes
    (ENT-2289 §2.7)."""
    return round(noise_hours / total_hours, 4) if total_hours else 0.0


def select_judge_clusters(clusters: list[dict]) -> list[dict]:
    """Which clusters the blind, combo-comparable LLM judge scores: the qualifying
    ones, or all of them if none qualify (so a bad config still gets a score rather
    than crashing the sweep)."""
    def payload(cs):
        return [{"title": c["title"], "description": c["description"], "examples": c["examples"]}
                for c in cs]
    qualifying = [c for c in clusters if c["qualifies"]]
    return payload(qualifying) if qualifying else payload(clusters)
