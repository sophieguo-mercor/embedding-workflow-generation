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
import embeddings as emb


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


# ── Token-aware embedding batching (§1.4) ─────────────────────────────────────

def _count(text, encoder):
    return emb._prepare_text(text, encoder)[1]


def test_prepare_text_caps_per_input_tokens():
    enc = emb._get_encoder()
    huge = "word " * 40000                         # ~40k tokens of input
    text, n = emb._prepare_text(huge, enc)
    assert n <= emb.MAX_INPUT_TOKENS
    assert len(text) <= emb.MAX_INPUT_CHARS


def test_batches_respect_token_budget_and_item_cap():
    enc = emb._get_encoder()
    # 500 fat inputs (~6k tokens each) → many would blow a 128-item request
    items = [(i, "lorem ipsum " * 2000) for i in range(500)]
    max_items = 128
    seen = []
    for batch in emb._token_aware_batches(iter(items), enc, max_items):
        assert 1 <= len(batch) <= max_items
        tok = sum(_count(t, enc) for _, t in batch)
        assert tok <= emb.MAX_REQUEST_TOKENS, f"batch of {tok} tokens exceeds budget"
        seen.extend(idx for idx, _ in batch)
    assert seen == list(range(500))               # every item emitted exactly once, in order


def test_batches_handle_single_oversized_input():
    enc = emb._get_encoder()
    items = [(0, "x " * 100000)]                   # one input far over every limit
    batches = list(emb._token_aware_batches(iter(items), enc, 128))
    assert len(batches) == 1 and len(batches[0]) == 1
    assert _count(batches[0][0][1], enc) <= emb.MAX_INPUT_TOKENS


def test_heuristic_fallback_when_no_encoder():
    # encoder=None forces the chars/4 path; batches must still respect the budget
    items = [(i, "a" * 40000) for i in range(50)]
    for batch in emb._token_aware_batches(iter(items), None, 128):
        tok = sum(emb._prepare_text(t, None)[1] for _, t in batch)
        assert tok <= emb.MAX_REQUEST_TOKENS


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
