#!/usr/bin/env python3
"""
Offline tests for the LLM description cleaner (§1.1, llm_clean.py). No network:
the Anthropic client is only constructed on the submit/collect paths, never here.
Covers the deterministic prepass, user-message framing, and result parsing
(clean JSON, ```json fences, malformed input, out-of-range ids).

Run: `python tests/test_llm_clean.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clean
import llm_clean as L


# ── prepass: token-cutting scrub WITHOUT the signature/thread cut ─────────────

def test_prepass_strips_structural_noise_but_keeps_signature():
    raw = ("Graag een muis._x000D_[cid:img1]\n"
           "Zie https://eu.content.exclaimer.net?url=x&signature=AAAA\n"
           "Met vriendelijke groet, Jan")
    pp = clean.prepass_description(raw)
    assert "_x000D_" not in pp and "cid:" not in pp and "exclaimer.net" not in pp
    # boundary NOT cut — the LLM decides on the signature
    assert "Met vriendelijke groet" in pp
    assert "Graag een muis" in pp


def test_prepass_vs_full_clean_differ_on_signature():
    raw = "De VPN werkt niet.\nMet vriendelijke groet,\nJan de Vries\nACME BV"
    assert "vriendelijke groet" in clean.prepass_description(raw)      # kept for LLM
    assert "vriendelijke groet" not in clean.clean_description(raw)    # regex cuts it


def test_prepass_nullish_empty():
    for v in (None, "nan", "NULL", "   ", ""):
        assert clean.prepass_description(v) == ""


# ── user-message framing ──────────────────────────────────────────────────────

def test_user_message_frames_and_truncates():
    msg = L.build_user_message(["short one", "x" * 5000], char_limit=1500)
    assert "--- TICKET 0 ---" in msg and "--- TICKET 1 ---" in msg
    assert "…[truncated]" in msg
    # ticket 0 untouched, ticket 1 capped near the limit
    assert "short one" in msg


# ── result parsing ────────────────────────────────────────────────────────────

def test_parse_plain_json():
    out = L.parse_result_text('[{"id":0,"text":"A"},{"id":1,"text":""}]', 2)
    assert out == {0: "A", 1: ""}


def test_parse_strips_json_fences():
    fenced = "```json\n[{\"id\":0,\"text\":\"hello\"}]\n```"
    assert L.parse_result_text(fenced, 1) == {0: "hello"}


def test_parse_ignores_out_of_range_ids():
    out = L.parse_result_text('[{"id":0,"text":"ok"},{"id":9,"text":"nope"}]', 1)
    assert out == {0: "ok"}


def test_parse_raises_on_bad_json():
    raised = False
    try:
        L.parse_result_text("not json at all", 3)
    except Exception:
        raised = True
    assert raised, "malformed JSON must raise so the group is re-submitted"


# ── group building + resume-skip ──────────────────────────────────────────────

def test_build_groups_chunks_and_skips_done():
    items = [(f"i::{n}", f"desc {n}") for n in range(5)]
    done = {"i::1", "i::3"}
    groups = L.build_groups(items, done, batch_size=2)
    packed = [gid for g in groups.values() for gid in g["ids"]]
    assert set(packed) == {"i::0", "i::2", "i::4"}       # done ones skipped
    assert all(len(g["ids"]) <= 2 for g in groups.values())


def test_cache_round_trip(tmp_path=None):
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "desc_clean.jsonl")
    done = set()
    w = L.CacheWriter(path, done)
    assert w.emit("a::1", "clean text") is True
    assert w.emit("a::1", "dup") is False                # idempotent
    w.emit("a::2", "")
    w.close()
    loaded = L.load_cache(path)
    assert loaded == {"a::1": "clean text", "a::2": ""}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
