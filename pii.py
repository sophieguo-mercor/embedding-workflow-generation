#!/usr/bin/env python3
"""
Section 5.1 — PII gate.

Two behaviours, per the ticket:
  * Structured fields (workflow name, category)  -> HARD FAIL on any PII match.
    These should be clean, generated labels; PII here means the pipeline leaked
    raw ticket content into a place that will be published, so we refuse.
  * Prose fields (descriptions)                  -> BLANK OUT PII in place.

Detection is regex-based and deliberately conservative: emails, phone numbers,
IPv4, and common Dutch/EU identifiers (IBAN, BSN-like 9-digit runs). It is a
safety net, not a guarantee — the frozen artifact is human-reviewed (§3).
"""
from __future__ import annotations

import re

# ── detectors ────────────────────────────────────────────────────────────────
_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    "bsn": re.compile(r"(?<!\d)\d{9}(?!\d)"),  # Dutch social-security-number shape
    "url_creds": re.compile(r"\b\w+://[^\s:@]+:[^\s:@]+@\S+"),
}

_REDACT = {
    "email": "[EMAIL]", "ipv4": "[IP]", "phone": "[PHONE]",
    "iban": "[IBAN]", "bsn": "[ID]", "url_creds": "[URL]",
}


class PIIError(RuntimeError):
    """Raised when PII is found in a structured field (hard fail)."""


def find_pii(text: str) -> list[tuple[str, str]]:
    """Return [(kind, matched_text), ...] found in `text`."""
    if not text:
        return []
    hits: list[tuple[str, str]] = []
    for kind, pat in _PATTERNS.items():
        for m in pat.finditer(text):
            hits.append((kind, m.group(0)))
    return hits


def scan_structured(value: str, *, field: str) -> None:
    """Hard-fail if a structured field carries PII."""
    hits = find_pii(value or "")
    if hits:
        kinds = ", ".join(sorted({k for k, _ in hits}))
        raise PIIError(f"PII ({kinds}) found in structured field '{field}': {value!r}")


def blank_prose(text: str) -> tuple[str, int]:
    """Redact PII in a prose field. Returns (clean_text, n_redactions)."""
    if not text:
        return text, 0
    n = 0
    out = text
    for kind, pat in _PATTERNS.items():
        out, k = pat.subn(_REDACT[kind], out)
        n += k
    return out, n


def gate_rubric(rows: list[dict], *, log=print) -> list[dict]:
    """Apply the gate to finalized rubric rows.

    Each row: {"name": ..., "category": ..., "description": ...}
    - name, category  -> hard fail on PII
    - description      -> blanked in place
    """
    total_redactions = 0
    for r in rows:
        scan_structured(r.get("name", ""), field="name")
        scan_structured(r.get("category", ""), field="category")
        r["description"], k = blank_prose(r.get("description", ""))
        total_redactions += k
    log(f"[pii] structured fields clean; redacted {total_redactions} PII spans from descriptions")
    return rows
