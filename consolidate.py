#!/usr/bin/env python3
"""
§2.8 — MECE consolidation of the named HDBSCAN micro-clusters.

The naming pass (§2.4) over-fragments: the same recurring process is split across
many clusters (a dozen onboarding clusters, several RDS/Citrix clusters, …). This
module folds them into a two-level CATEGORY -> WORKFLOW taxonomy via ONE strong-model
call (Opus, see config.CONSOLIDATE_MODEL), then guarantees a clean partition of the
cluster ids without ever regenerating the whole thing:

  1. one streaming consolidation call                 (llm.consolidate_taxonomy)
  2. deterministic repair — drop invented ids, de-dup (mechanical_clean)
  3. targeted placement of any ids the model omitted   (llm.place_missing_clusters)
  4. fold leftovers into `unclassified` (honest CE), then assert a full partition
  5. join member ids back to source clusters for hours/ticket rollups (add_rollups)

`build_consolidation(llm, clusters)` returns the dict the sweep stores under the
config artifact's `consolidation` key. Deterministic given a fixed LLM response;
the only LLM-generated content is validated against the known cluster_id set before
anything is persisted.
"""
from __future__ import annotations


def collect_assigned(result: dict) -> list[int]:
    """Flatten every member_cluster_id across workflows + unclassified (with repeats,
    so duplicates are detectable)."""
    ids: list[int] = []
    for cat in result.get("taxonomy", []):
        for wf in cat.get("workflows", []):
            ids.extend(int(i) for i in wf.get("member_cluster_ids", []))
    ids.extend(int(i) for i in result.get("unclassified", {}).get("member_cluster_ids", []))
    return ids


def validate_partition(result: dict, expected: set[int]) -> tuple[list[int], list[int]]:
    """Return (missing, invalid_or_dup)."""
    assigned = collect_assigned(result)
    seen, dups = set(), []
    for i in assigned:
        if i in seen:
            dups.append(i)
        seen.add(i)
    missing = sorted(expected - seen)
    invalid = sorted({i for i in assigned if i not in expected} | set(dups))
    return missing, invalid


def mechanical_clean(result: dict, expected: set[int]) -> list[int]:
    """Make the assignment a clean partial partition WITHOUT another full LLM pass:
    drop invented ids, and de-dup with first-occurrence-wins (workflows in listed
    order, then unclassified). Returns the still-missing ids. The model's semantic
    groupings are preserved — only its bookkeeping is fixed."""
    seen: set[int] = set()

    def keep(ids):
        out = []
        for i in ids:
            try:
                i = int(i)
            except (TypeError, ValueError):
                continue
            if i in expected and i not in seen:
                out.append(i)
                seen.add(i)
        return out

    for cat in result.get("taxonomy", []):
        for wf in cat.get("workflows", []):
            wf["member_cluster_ids"] = keep(wf.get("member_cluster_ids", []))
    unc = result.setdefault("unclassified", {"member_cluster_ids": [], "note": ""})
    unc["member_cluster_ids"] = keep(unc.get("member_cluster_ids", []))
    return sorted(expected - seen)


def _workflow_index(result: dict) -> list[dict]:
    return [{"workflow": wf["name"], "category": cat["category"],
             "description": wf["description"]}
            for cat in result.get("taxonomy", []) for wf in cat.get("workflows", [])]


def _apply_assignments(result: dict, assignments: list[dict]) -> None:
    """Insert targeted-placement results: known workflow name → that workflow,
    else → unclassified."""
    name_to_wf = {wf["name"]: wf for cat in result.get("taxonomy", [])
                  for wf in cat.get("workflows", [])}
    for item in assignments:
        try:
            i, w = int(item["id"]), str(item["workflow"])
        except (KeyError, ValueError, TypeError):
            continue
        if w in name_to_wf:
            name_to_wf[w]["member_cluster_ids"].append(i)
        else:
            result["unclassified"]["member_cluster_ids"].append(i)


def add_rollups(result: dict, clusters: list[dict]) -> dict:
    """Join assigned ids back to source clusters for hours/ticket totals, and add a
    top-level `summary`."""
    by_id = {c["cluster_id"]: c for c in clusters}

    def agg(ids):
        ids = [i for i in ids if i in by_id]
        return {
            "n_clusters": len(ids),
            "n_tickets": sum(by_id[i].get("n_tickets", 0) for i in ids),
            "hours": round(sum(by_id[i].get("hours", 0.0) for i in ids), 1),
        }

    tot_h = sum(c.get("hours", 0.0) for c in clusters)
    tot_t = sum(c.get("n_tickets", 0) for c in clusters)
    for cat in result.get("taxonomy", []):
        cat_ids = []
        for wf in cat.get("workflows", []):
            wf["rollup"] = agg(wf.get("member_cluster_ids", []))
            cat_ids.extend(wf.get("member_cluster_ids", []))
        cat["rollup"] = agg(cat_ids)
    unc = result.setdefault("unclassified", {"member_cluster_ids": [], "note": ""})
    unc["rollup"] = agg(unc.get("member_cluster_ids", []))

    n_wf = sum(len(c.get("workflows", [])) for c in result.get("taxonomy", []))
    named_h = tot_h - unc["rollup"]["hours"]
    result["summary"] = {
        "n_categories": len(result.get("taxonomy", [])),
        "n_workflows": n_wf,
        "n_source_clusters": len(clusters),
        "clustered_hours": round(tot_h, 1),
        "clustered_tickets": tot_t,
        "classified_hours_frac": round(named_h / tot_h, 4) if tot_h else 0.0,
        "unclassified_hours_frac": round(unc["rollup"]["hours"] / tot_h, 4) if tot_h else 0.0,
    }
    return result


def build_consolidation(llm, clusters: list[dict], *, log=print) -> dict:
    """Full §2.8 pipeline over one config's named clusters. Returns the consolidation
    dict (taxonomy + unclassified + rollups + model). Raises if a full partition of
    the cluster ids cannot be reached even after the safety-net."""
    expected = {c["cluster_id"] for c in clusters}
    result = llm.consolidate_taxonomy(clusters)

    raw_missing, raw_invalid = validate_partition(result, expected)
    missing = mechanical_clean(result, expected)
    log(f"    partition: raw missing={len(raw_missing)} invalid/dup={len(raw_invalid)} "
        f"→ after clean missing={len(missing)}")

    if missing:
        log(f"    placing {len(missing)} omitted clusters via targeted call")
        assignments = llm.place_missing_clusters(missing, clusters, _workflow_index(result))
        _apply_assignments(result, assignments)

    leftover = mechanical_clean(result, expected)
    if leftover:
        log(f"    {len(leftover)} still unplaced → unclassified (honest CE)")
        result["unclassified"]["member_cluster_ids"].extend(leftover)
        note = result["unclassified"].get("note", "")
        result["unclassified"]["note"] = (note + " ").lstrip() + \
            f"[{len(leftover)} clusters auto-routed here: unplaced by the model.]"

    missing, invalid = validate_partition(result, expected)
    if missing or invalid:
        raise RuntimeError(f"consolidation partition invalid: "
                           f"missing={missing[:20]} invalid={invalid[:20]}")

    add_rollups(result, clusters)
    result["model"] = getattr(llm, "model", None)
    s = result["summary"]
    log(f"    → {s['n_workflows']} workflows / {s['n_categories']} categories | "
        f"classified {s['classified_hours_frac']:.1%} of clustered hours | "
        f"unclassified {s['unclassified_hours_frac']:.1%}")
    return result
