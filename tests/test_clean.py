#!/usr/bin/env python3
"""
Unit tests for the text cleaner (§1.1). Cases are distilled from real strings in
the raw exports: `_x000D_` escapes, cid refs, exclaimer tracking links, the
Incoming-Email-Processor footer, Dutch/English quoted threads, and signatures.

Run: `python -m pytest tests/` or `python tests/test_clean.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clean


# ── structural scrubs ─────────────────────────────────────────────────────────

def test_decodes_x000d_to_newlines():
    raw = "Goedemiddag,_x000D_Graag een Visio licentie._x000D_Bedankt!"
    out = clean.clean_description(raw)
    assert "_x000D_" not in out
    assert "Visio licentie" in out


def test_strips_cid_image_refs():
    out = clean.clean_description("Zie scherm [cid:07f5b333-2091-4aaf] hierboven")
    assert "cid:" not in out
    assert "Zie scherm" in out and "hierboven" in out


def test_unwraps_link_keeps_text_drops_target():
    raw = "mail c.kalisvaart@buroboot.nl<mailto:c.kalisvaart@buroboot.nl> graag"
    out = clean.clean_description(raw)
    assert "mailto:" not in out
    assert "c.kalisvaart@buroboot.nl" in out          # visible text kept


def test_drops_exclaimer_tracking_url_in_description():
    raw = ("Bekijk website "
           "https://eu.content.exclaimer.net?url=xxx&signature=AAAA1234 nu")
    out = clean.clean_description(raw)
    assert "exclaimer.net" not in out
    assert "Bekijk website" in out and "nu" in out


def test_fixes_mojibake_runs():
    out = clean.clean_description("office manager ???? afdeling")
    assert "????" not in out
    assert "office manager" in out and "afdeling" in out


# ── boundary stripping (signature / thread / footer) ──────────────────────────

def test_strips_incoming_email_processor_footer():
    raw = ("Graag een Visio licentie aanschaffen.\n\n"
           "**Created via Incoming Email Processor**\n"
           "From: Martijn Smit <m.smit@doove.nl>\nTo: Support <support@hupra.nl>")
    out = clean.clean_description(raw)
    assert "Incoming Email Processor" not in out
    assert "m.smit@doove.nl" not in out
    assert out.strip().startswith("Graag een Visio licentie")


def test_strips_dutch_quoted_reply_thread():
    raw = ("Willen jullie de licentie opzeggen?\n"
           "Van: Leo Verriet <leo@verriet.eu>\n"
           "Verzonden: dinsdag 27 januari 2026 11:03\n"
           "Onderwerp: RE: Factuur 110956\n\nDag Jannita, ...")
    out = clean.clean_description(raw)
    assert "Van:" not in out and "Verzonden:" not in out
    assert "leo@verriet.eu" not in out
    assert out == "Willen jullie de licentie opzeggen?"


def test_strips_signature_block():
    raw = ("Graag een muis aansluiten.\n\n"
           "Met vriendelijke groet,\nRené Klaassen\nPurchaser\n"
           "r.klaassen@halmasolutions.com")
    out = clean.clean_description(raw)
    assert "vriendelijke groet" not in out
    assert "r.klaassen@halmasolutions.com" not in out
    assert out == "Graag een muis aansluiten."


def test_cut_uses_earliest_boundary():
    # signature appears before the footer — content ends at the signature.
    raw = ("De VPN werkt niet meer.\n"
           "Met hartelijke groet, Christian\n"
           "**Created via Incoming Email Processor**\nFrom: x")
    out = clean.clean_description(raw)
    assert out == "De VPN werkt niet meer."


def test_midsentence_from_is_not_a_boundary():
    raw = "De mail from jan komt niet aan bij de klant."
    out = clean.clean_description(raw)
    assert "komt niet aan" in out                      # not truncated


# ── notes mode ────────────────────────────────────────────────────────────────

def test_notes_keep_urls():
    raw = ("Config uitgevoerd: "
           "https://learn.microsoft.com/business-central/admin-smtp aangemaakt")
    out = clean.clean_note(raw)
    assert "learn.microsoft.com" in out                # technical url is signal


def test_notes_still_decode_escapes():
    out = clean.clean_note("Gekeken naar config._x000D_Lijkt goed.")
    assert "_x000D_" not in out
    assert "Gekeken naar config" in out


# ── invariants ────────────────────────────────────────────────────────────────

def test_nullish_and_empty_become_empty():
    for v in (None, "nan", "None", "NULL", "   ", ""):
        assert clean.clean_description(v) == ""
        assert clean.clean_note(v) == ""


def test_clean_text_is_idempotent():
    raw = ("Graag een licentie._x000D_[cid:abc]\n\nMet vriendelijke groet,\nJan")
    once = clean.clean_description(raw)
    twice = clean.clean_description(once)
    assert once == twice


def test_plain_content_survives_unchanged_semantically():
    raw = "Printer is momenteel offline."
    assert clean.clean_description(raw) == raw


def test_strips_worksheet_illegal_control_chars():
    # openpyxl rejects these; both the scrub and safe_cell must remove them,
    # while keeping tab/newline/CR.
    raw = "Ticket\x07 aangemaakt\x0b voor\x1f RC4.\tKlaar.\n"
    assert clean.clean_description(raw) == "Ticket aangemaakt voor RC4. Klaar."
    llm_out = "Alert\x03 geregistreerd.\x00"
    assert clean.safe_cell(llm_out) == "Alert geregistreerd."
    assert clean.safe_cell(None) is None          # non-str passes through


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
