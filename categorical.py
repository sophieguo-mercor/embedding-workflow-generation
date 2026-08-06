#!/usr/bin/env python3
"""
Section 1.5 — Categorical feature block.

Build a TF-IDF vector over issue_type + sub_issue_type after normalising
spelling/language variants and stripping placeholder values.

Encoding (per the ticket):
  * Two-pass normalisation: rule-based canonicalisation, then a curated
    synonym/translation map (synonyms.json).
  * Pool both fields into ONE shared vocabulary as namespaced tokens
    ("issue=<value>", "sub=<value>"). Pooling is what lets IDF do anything:
    with two tokens per ticket, IDF isn't cancelled by the per-row L2 norm the
    way it is for a lone one-hot field.
  * TfidfVectorizer(min_df=2, smooth_idf=True), then L2-normalise the block.

Effect: (a) up-weights the finer, rarer sub_issue_type over the coarser
issue_type, and (b) dampens high-frequency placeholder values automatically.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import config as C

PLACEHOLDER = "__placeholder__"


def _load_synonyms(path: str = C.SYNONYMS_JSON) -> dict[str, str]:
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8")).get("map", {})


# ── Pass 1: rule-based canonicalisation ──────────────────────────────────────

_PREFIXES = re.compile(r"^(qnp\s*-\s*|td\s+|ict\s*-\s*|klant\s*-\s*)", re.I)


def canonicalize(raw: str) -> str:
    """Lowercase, trim/collapse whitespace, standardise hyphens/punctuation,
    strip company/tool prefixes. Merges most trivial variants."""
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    if not s:
        return ""
    s = _PREFIXES.sub("", s)
    s = s.replace("_", " ")
    s = re.sub(r"[-/]+", " ", s)            # "back-up" -> "back up" (map fixes -> backup)
    s = re.sub(r"[^\w\s.]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class CategoricalNormalizer:
    """Two-pass raw -> canonical label mapper, auditable."""

    def __init__(self, synonyms: dict[str, str] | None = None):
        self.synonyms = synonyms if synonyms is not None else _load_synonyms()

    def normalize_label(self, raw: str) -> str:
        c = canonicalize(raw)
        if not c:
            return ""
        # Pass 2: curated synonym / translation map (exact match on canonical form)
        mapped = self.synonyms.get(c, c)
        if mapped == PLACEHOLDER:
            return ""      # placeholder -> dropped entirely
        return mapped


def _tokens(norm: CategoricalNormalizer, issue_type: str, sub_issue_type: str) -> str:
    """Namespaced token document for one ticket, e.g. 'issue=change sub=password'."""
    toks: list[str] = []
    it = norm.normalize_label(issue_type)
    st = norm.normalize_label(sub_issue_type)
    if it:
        toks.append("issue=" + it.replace(" ", "_"))
    if st:
        toks.append("sub=" + st.replace(" ", "_"))
    return " ".join(toks)


def build_categorical_block(
    records,
    *,
    min_df: int = 2,
    synonyms: dict[str, str] | None = None,
) -> tuple[np.ndarray, TfidfVectorizer, CategoricalNormalizer]:
    """Return an (n_tickets, vocab) L2-normalised dense TF-IDF matrix for the
    categorical block, plus the fitted vectorizer and normaliser (for audit)."""
    norm = CategoricalNormalizer(synonyms)
    docs = [_tokens(norm, r.issue_type, r.sub_issue_type) for r in records]

    vec = TfidfVectorizer(
        min_df=min_df,
        smooth_idf=True,
        token_pattern=r"[^\s]+",   # keep "issue=..."/"sub=..." tokens intact
        lowercase=False,           # already canonicalised
    )
    X = vec.fit_transform(docs)          # already TF-IDF; rows may be all-zero
    X = normalize(X, norm="l2", axis=1)  # L2-normalise the block (§1.5)
    return X.toarray().astype(np.float32), vec, norm


def audit_table(records, synonyms: dict[str, str] | None = None) -> dict:
    """raw_label -> canonical_label table, for the auditable artifact (§1.5)."""
    norm = CategoricalNormalizer(synonyms)
    table: dict[str, dict] = {}
    for r in records:
        for field, val in (("issue_type", r.issue_type), ("sub_issue_type", r.sub_issue_type)):
            raw = (val or "").strip()
            if not raw:
                continue
            canon = norm.normalize_label(raw)
            table.setdefault(raw, {"canonical": canon, "field": field, "n": 0})["n"] += 1
    return dict(sorted(table.items(), key=lambda kv: -kv[1]["n"]))
