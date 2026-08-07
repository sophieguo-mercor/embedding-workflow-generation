#!/usr/bin/env python3
"""
Fast, data-free unit tests for the shared scoring module (metrics.py) — the
coverage / "Other"-gate / judge-selection logic every clustering method
(ENT-2260, ENT-2261, ENT-2289) must call identically for the bake-off to be
comparable.

Run: `python -m pytest tests/` or `python tests/test_metrics.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics


def _cluster(cid=0, *, coherent=True, n_tickets=100, hours=500.0, hours_frac=0.10,
             title="T", description="D", examples=("a", "b")):
    """A minimal cluster dict matching the metrics.py contract."""
    return {
        "cluster_id": cid,
        "title": title,
        "description": description,
        "coherent": coherent,
        "coherence_note": "",
        "n_tickets": n_tickets,
        "hours": hours,
        "hours_frac": hours_frac,
        "examples": list(examples),
    }


# ── mark_qualification — the §2.6 "Other" gate ────────────────────────────────
# Explicit thresholds so the tests don't depend on config values.
GATE = dict(min_tickets=25, min_mass_frac=0.003)


def test_qualifies_when_coherent_and_big():
    c = metrics.mark_qualification(_cluster(coherent=True, n_tickets=100, hours_frac=0.10), **GATE)
    assert c["big_enough"] is True
    assert c["qualifies"] is True


def test_incoherent_big_cluster_does_not_qualify():
    c = metrics.mark_qualification(_cluster(coherent=False, n_tickets=100, hours_frac=0.10), **GATE)
    assert c["big_enough"] is True          # size gate is independent of coherence
    assert c["qualifies"] is False


def test_too_few_tickets_does_not_qualify():
    c = metrics.mark_qualification(_cluster(coherent=True, n_tickets=24, hours_frac=0.10), **GATE)
    assert c["big_enough"] is False
    assert c["qualifies"] is False


def test_below_mass_floor_does_not_qualify():
    c = metrics.mark_qualification(_cluster(coherent=True, n_tickets=100, hours_frac=0.0029), **GATE)
    assert c["big_enough"] is False
    assert c["qualifies"] is False


def test_boundary_is_inclusive():
    # exactly at both thresholds must PASS (>= gate)
    c = metrics.mark_qualification(_cluster(coherent=True, n_tickets=25, hours_frac=0.003), **GATE)
    assert c["big_enough"] is True
    assert c["qualifies"] is True


def test_mark_qualification_mutates_and_returns_same_dict():
    c = _cluster()
    out = metrics.mark_qualification(c, **GATE)
    assert out is c                          # mutates in place, returns the same object
    assert "big_enough" in c and "qualifies" in c


# ── mass_weighted_coverage ────────────────────────────────────────────────────

def test_coverage_counts_only_qualifying_hours():
    clusters = [
        metrics.mark_qualification(_cluster(0, coherent=True, n_tickets=100, hours=600.0, hours_frac=0.30), **GATE),
        metrics.mark_qualification(_cluster(1, coherent=False, n_tickets=100, hours=300.0, hours_frac=0.15), **GATE),
        metrics.mark_qualification(_cluster(2, coherent=True, n_tickets=10, hours=100.0, hours_frac=0.05), **GATE),
    ]
    # total_hours is the WHOLE corpus (2000h) — larger than the clustered hours
    # (1000h) to model text-poor drops / HDBSCAN noise that never enter `clusters`.
    cov = metrics.mass_weighted_coverage(clusters, total_hours=2000.0)
    assert cov["covered_hours"] == 600.0     # only cluster 0 qualifies
    assert cov["coverage"] == 0.30           # 600 / 2000
    assert cov["n_named_clusters"] == 1
    assert cov["n_clusters"] == 3
    assert cov["total_hours"] == 2000.0


def test_coverage_zero_total_hours_no_div_by_zero():
    clusters = [metrics.mark_qualification(_cluster(hours=0.0, hours_frac=0.0), **GATE)]
    cov = metrics.mass_weighted_coverage(clusters, total_hours=0.0)
    assert cov["coverage"] == 0.0


def test_coverage_rounds_to_four_places():
    c = metrics.mark_qualification(_cluster(coherent=True, n_tickets=100, hours=1.0, hours_frac=0.10), **GATE)
    cov = metrics.mass_weighted_coverage([c], total_hours=3.0)
    assert cov["coverage"] == 0.3333         # round(1/3, 4)


# ── noise_mass (HDBSCAN-specific diagnostic, 0 for k-means) ────────────────────

def test_noise_mass_fraction_rounded():
    assert metrics.noise_mass(250.0, 1000.0) == 0.25
    assert metrics.noise_mass(1.0, 3.0) == 0.3333


def test_noise_mass_zero_cases():
    assert metrics.noise_mass(0.0, 1000.0) == 0.0     # no noise (k-means)
    assert metrics.noise_mass(50.0, 0.0) == 0.0       # empty corpus → no div-by-zero


# ── select_judge_clusters ─────────────────────────────────────────────────────

def test_judge_selects_only_qualifying_when_some_qualify():
    clusters = [
        metrics.mark_qualification(_cluster(0, coherent=True, n_tickets=100, hours_frac=0.10, title="keep"), **GATE),
        metrics.mark_qualification(_cluster(1, coherent=False, n_tickets=100, hours_frac=0.10, title="drop"), **GATE),
    ]
    picked = metrics.select_judge_clusters(clusters)
    assert [p["title"] for p in picked] == ["keep"]
    # payload is exactly the judge-visible fields — nothing more leaks in
    assert set(picked[0]) == {"title", "description", "examples"}


def test_judge_falls_back_to_all_when_none_qualify():
    clusters = [
        metrics.mark_qualification(_cluster(0, coherent=False, n_tickets=100, hours_frac=0.10), **GATE),
        metrics.mark_qualification(_cluster(1, coherent=True, n_tickets=1, hours_frac=0.0001), **GATE),
    ]
    assert not any(c["qualifies"] for c in clusters)
    picked = metrics.select_judge_clusters(clusters)
    assert len(picked) == 2                  # bad config still gets a score, not a crash


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
