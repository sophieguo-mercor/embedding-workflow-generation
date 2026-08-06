#!/usr/bin/env python3
"""
Fast, data-free unit tests for the deterministic pieces of the pipeline:
the PII gate (§5.1) and categorical normalization (§1.5).

Run: `python -m pytest tests/` or `python tests/test_pipeline.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pii
import categorical as cat


# ── PII gate (§5.1) ───────────────────────────────────────────────────────────

def test_blank_prose_redacts_common_pii():
    text = "Reset done, email jan@example.com from 192.168.0.1, call +31 6 1234 5678."
    clean, n = pii.blank_prose(text)
    assert "jan@example.com" not in clean
    assert "192.168.0.1" not in clean
    assert "[EMAIL]" in clean and "[IP]" in clean and "[PHONE]" in clean
    assert n >= 3


def test_scan_structured_hard_fails_on_pii():
    pii.scan_structured("Password Reset", field="name")  # clean → no raise
    raised = False
    try:
        pii.scan_structured("Reset for jan@example.com", field="name")
    except pii.PIIError:
        raised = True
    assert raised, "structured field with an email must hard-fail"


def test_gate_rubric_blanks_description_keeps_clean_names():
    rows = [{"name": "Password Reset", "category": "Identity & Access",
             "description": "User jan@example.com locked out."}]
    out = pii.gate_rubric(rows, log=lambda *a, **k: None)
    assert out[0]["name"] == "Password Reset"
    assert "jan@example.com" not in out[0]["description"]


# ── Categorical normalization (§1.5) ──────────────────────────────────────────

def test_canonicalize_rules():
    assert cat.canonicalize("  Back-Up  ") == "back up"
    assert cat.canonicalize("QNP - Password") == "password"
    assert cat.canonicalize("Office/365") == "office 365"


def test_synonym_map_translates_and_drops_placeholders():
    norm = cat.CategoricalNormalizer({"wachtwoord": "password", "overig": cat.PLACEHOLDER})
    assert norm.normalize_label("Wachtwoord") == "password"
    assert norm.normalize_label("Overig") == ""      # placeholder → dropped


def test_tfidf_block_pools_fields_and_l2_normalizes():
    import numpy as np
    from dataclasses import dataclass

    @dataclass
    class R:
        issue_type: str
        sub_issue_type: str

    recs = [
        R("Change", "Password"), R("Change", "Password"),
        R("Incident", "Network"), R("Incident", "Network"),
        R("Change", "Network"), R("Request", "Mailbox"),
    ]
    X, vec, _ = cat.build_categorical_block(recs, min_df=2, synonyms={})
    # rows L2-normalised (non-empty rows have unit norm)
    norms = np.linalg.norm(X, axis=1)
    assert np.all((np.isclose(norms, 1.0)) | np.isclose(norms, 0.0))
    # namespaced tokens present in the pooled vocabulary
    assert any(t.startswith("issue=") for t in vec.vocabulary_)
    assert any(t.startswith("sub=") for t in vec.vocabulary_)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
