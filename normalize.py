#!/usr/bin/env python3
"""
Section 1.5 (ENT-2289) — LLM intent normalization.

For each ticket, one Anthropic Message-Batch call over its title + cleaned
description + cleaned notes produces two short ENGLISH fields:

  * normalized_issue      — the root problem in plain language (what was needed /
                            what was broken), phrased consistently so paraphrases
                            and Dutch↔English wordings of the same process collapse.
  * normalized_resolution — what was actually done to resolve it, from the notes.

This is the pass ENT-2289 adds on TOP of the §1.4 cleaning (which is reused
verbatim and already cached). A 2026 FLAIRS study found LLM semantic normalization
before embedding was the single largest contributor to support-ticket cluster
quality — hence it is ablated in the sweep, not assumed.

Cost & persistence
------------------
This is the biggest NEW cost line in the pipeline (~154k tickets), so it mirrors
the sibling §1.4 batch harness (llm_clean.py): the Message Batches API (separate
higher-throughput lane, ~50% token price), a cache_control'd system prefix billed
once, and resume-safety on two levels so the expensive run is paid for exactly
once:

  1. cache/normalize.jsonl — {id, issue, resolution} per ticket. Append-only and
     idempotent: ids already present are skipped on re-runs. This is the DURABLE
     store — it survives records rebuilds and every sweep re-run, so no config
     ever triggers re-normalization. gitignored (still may echo ticket text).
  2. cache/normalize_batch_manifest.json — an in-flight batch is reconnected, never
     re-submitted (double-spend guard). Removed on a clean collect.

Consume with normalize.attach_normalized(records) (the sweep does this
automatically when the cache exists), which merges the two fields back onto the
in-memory records; missing ids get "".

CLI
---
    python normalize.py --dry-run          # prompt + request/token estimate, no API calls
    python normalize.py --sample 24        # small real batch to eyeball quality
    python normalize.py                     # full run; re-run (or --collect) to finish
    python normalize.py --submit-only       # create batch + exit
    python normalize.py --collect           # collect an already-submitted batch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config as C
from llm_clean import (MAX_REQUESTS_PER_BATCH, _strip_fences, load_dotenv)
from records import load_records

MIN_CONTENT_CHARS = 20      # combined text shorter than this → freebie, not sent


# ── prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You normalize IT-support tickets into a canonical form for clustering. For each ticket you receive its title, the cleaned description (the customer's request), and the cleaned notes (the engineer's resolution log). The text may be Dutch or English.

For each ticket produce two SHORT fields, both in ENGLISH regardless of the source language:

- "issue": the root problem in plain language — what the customer actually needed, or what was broken. One or two sentences. Name the affected system / application / device, but GENERALIZE away specifics that don't define the process (personal names, ticket numbers, hostnames, dates). Phrase similar problems the SAME way so paraphrases collapse (e.g. always "Password reset for <application>", not sometimes "can't log in" and sometimes "forgot password").

- "resolution": what was actually done to resolve it, drawn from the notes. One or two sentences, same generalizing style. If the notes do not say what was done, return "".

RULES:
- English only. Do NOT copy the source-language wording — restate the intent in consistent English.
- Be concise and consistent; this text is for clustering, not for a human reader.
- Do NOT invent a resolution the notes don't support — return "" for resolution when unknown.
- If there is no substantive request at all, return "" for issue.

## Output
Return ONLY a JSON array — one object per input ticket, same order, no markdown fences, no prose:
[{"id": 0, "issue": "...", "resolution": "..."}, {"id": 1, "issue": "...", "resolution": ""}]

## Examples
Input:
--- TICKET 0 ---
title: Wachtwoord reset
description: Goedemiddag, ik kan niet meer inloggen op mijn Office 365 account, wachtwoord werkt niet meer.
notes: Wachtwoord gereset via admin portal, gebruiker kan weer inloggen. — Bevestigd met gebruiker.
--- TICKET 1 ---
title: Nieuwe medewerker
description: Graag een account aanmaken voor Lisanne, start maandag.
notes:
Output:
[{"id": 0, "issue": "Password reset for Office 365 account after the user could no longer log in.", "resolution": "Reset the account password via the admin portal and confirmed the user could log in again."}, {"id": 1, "issue": "New-employee account provisioning request.", "resolution": ""}]
"""


def build_user_message(tickets: list[dict], char_limit: int) -> str:
    """tickets: list of {title, description, notes}."""
    parts = []
    for i, t in enumerate(tickets):
        def cap(v):
            v = (v or "").strip()
            return v[:char_limit] + " …[truncated]" if len(v) > char_limit else v
        block = [f"--- TICKET {i} ---",
                 f"title: {cap(t.get('title'))}",
                 f"description: {cap(t.get('description'))}",
                 f"notes: {cap(t.get('notes'))}"]
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def parse_result_text(text: str, n: int) -> dict[int, dict]:
    """Parse one succeeded response → {position: {issue, resolution}}. Raises on bad
    JSON so the caller can skip the group and re-submit it on a later run."""
    parsed = json.loads(_strip_fences(text))
    out: dict[int, dict] = {}
    for item in parsed:
        pos = item.get("id")
        if isinstance(pos, int) and 0 <= pos < n:
            out[pos] = {
                "issue": str(item.get("issue", "")).strip(),
                "resolution": str(item.get("resolution", "")).strip(),
            }
    return out


# ── batch client ──────────────────────────────────────────────────────────────

class Normalizer:
    """Builds batch Requests. No network beyond client init, so one object serves
    the dry-run, submit, and collect paths (mirrors llm_clean.DescriptionCleaner)."""

    def __init__(self, model: str = C.NORMALIZE_MODEL, max_tokens: int = C.NORMALIZE_MAX_TOKENS):
        self.model = model
        self.max_tokens = max_tokens
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise SystemExit("pip install anthropic")
        from anthropic import Anthropic
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        import os

        self._MCP = MessageCreateParamsNonStreaming
        self._Request = Request
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Set ANTHROPIC_API_KEY in your environment (or .env).")
        self.client = Anthropic()

    def make_request(self, custom_id: str, tickets: list[dict], char_limit: int):
        return self._Request(
            custom_id=custom_id,
            params=self._MCP(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user",
                           "content": build_user_message(tickets, char_limit)}],
            ),
        )


# ── record reading ──────────────────────────────────────────────────────────

def read_tickets(records_path: str, *, log=print) -> list[tuple[str, dict]]:
    """Load records → [(ticket_id, {title, description, notes})]. Trivially short
    tickets (combined text < MIN_CONTENT_CHARS) are freebie-skipped and never sent
    — attach_normalized returns "" for anything absent from the cache."""
    recs = load_records(records_path)
    items: list[tuple[str, dict]] = []
    n_empty = 0
    for r in recs:
        payload = {"title": r.title, "description": r.description, "notes": r.notes}
        combined = (r.title or "") + (r.description or "") + (r.notes or "")
        if len(combined.strip()) < MIN_CONTENT_CHARS:
            n_empty += 1
            continue
        items.append((r.ticket_id, payload))
    log(f"  {len(recs):,} records | {len(items):,} to normalize | "
        f"{n_empty:,} empty/short (freebie)")
    return items


# ── cache + manifest ──────────────────────────────────────────────────────────

def load_normalize_cache(path: str = C.NORMALIZE_CACHE) -> dict[str, dict]:
    """{ticket_id: {issue, resolution}} from the durable jsonl cache."""
    out: dict[str, dict] = {}
    p = Path(path)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    out[rec["id"]] = {"issue": rec.get("issue", ""),
                                      "resolution": rec.get("resolution", "")}
                except Exception:
                    continue
    return out


class CacheWriter:
    """Append-mode, idempotent {id, issue, resolution} sink."""

    def __init__(self, path: str, done: set[str]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self._done = done

    def emit(self, ticket_id: str, issue: str, resolution: str) -> bool:
        if ticket_id in self._done:
            return False
        self._fh.write(json.dumps(
            {"id": ticket_id, "issue": issue, "resolution": resolution},
            ensure_ascii=False) + "\n")
        self._fh.flush()
        self._done.add(ticket_id)
        return True

    def close(self):
        self._fh.close()


def build_groups(items, done: set[str], batch_size: int) -> dict[str, dict]:
    """Chunk not-yet-normalized items into {custom_id: {ids, tickets}}."""
    todo = [(k, t) for k, t in items if k not in done]
    groups: dict[str, dict] = {}
    for c, i in enumerate(range(0, len(todo), batch_size)):
        chunk = todo[i:i + batch_size]
        groups[f"g{c:06d}"] = {"ids": [k for k, _ in chunk],
                               "tickets": [t for _, t in chunk]}
    return groups


def save_manifest(path: str, batch_id: str, model: str, groups: dict) -> None:
    slim = {cid: {"ids": g["ids"]} for cid, g in groups.items()}   # drop ticket text
    Path(path).write_text(
        json.dumps({"batch_id": batch_id, "model": model, "groups": slim},
                   ensure_ascii=False), encoding="utf-8")


# ── consume: merge the cache back onto records ────────────────────────────────

def attach_normalized(records, cache_path: str = C.NORMALIZE_CACHE, *, log=print) -> int:
    """Set normalized_issue / normalized_resolution on each record from the cache.
    Missing ids get "". Returns the count of records that got a non-empty issue.
    No-op-ish (all "") when the cache is absent — callers then see the normalized
    blocks as unavailable and skip those configs."""
    cache = load_normalize_cache(cache_path)
    attached = 0
    for r in records:
        entry = cache.get(r.ticket_id)
        if entry:
            r.normalized_issue = entry.get("issue", "")
            r.normalized_resolution = entry.get("resolution", "")
            if r.normalized_issue:
                attached += 1
        else:
            r.normalized_issue = ""
            r.normalized_resolution = ""
    if cache:
        log(f"[normalize] attached {attached:,}/{len(records):,} normalized records "
            f"from {cache_path}")
    return attached


# ── submit / collect ──────────────────────────────────────────────────────────

def collect(normalizer: Normalizer, manifest_path: str, cache_path: str,
            poll_interval: int, *, log=print) -> None:
    import time
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    batch_id = manifest["batch_id"]
    groups = manifest["groups"]
    log(f"Reconnecting to batch {batch_id} ({len(groups):,} requests) …")

    while True:
        b = normalizer.client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        log(f"  status={b.processing_status}  processing={c.processing} "
            f"succeeded={c.succeeded} errored={c.errored} "
            f"canceled={c.canceled} expired={c.expired}")
        if b.processing_status == "ended":
            break
        time.sleep(poll_interval)

    done = set(load_normalize_cache(cache_path))
    writer = CacheWriter(cache_path, done)
    written = errored = bad_json = 0
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_create": 0}

    for r in normalizer.client.messages.batches.results(batch_id):
        grp = groups.get(r.custom_id)
        if grp is None:
            continue
        ids = grp["ids"]
        if r.result.type != "succeeded":
            err = getattr(getattr(r.result, "error", None), "type", r.result.type)
            errored += len(ids)
            log(f"    ! {r.custom_id} {err} ({len(ids)}) — re-submit next run")
            continue
        msg = r.result.message
        u = msg.usage
        usage["input"] += getattr(u, "input_tokens", 0) or 0
        usage["output"] += getattr(u, "output_tokens", 0) or 0
        usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        usage["cache_create"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        text = "".join(blk.text for blk in msg.content if blk.type == "text")
        try:
            by_pos = parse_result_text(text, len(ids))
        except Exception as e:
            bad_json += len(ids)
            log(f"    ! {r.custom_id} bad JSON ({len(ids)}): {e} — re-submit next run")
            continue
        for pos, tid in enumerate(ids):
            entry = by_pos.get(pos, {"issue": "", "resolution": ""})
            if writer.emit(tid, entry["issue"], entry["resolution"]):
                written += 1
    writer.close()
    Path(manifest_path).unlink(missing_ok=True)

    log(f"\nCollected {batch_id}: {written:,} normalized, {errored:,} errored, "
        f"{bad_json:,} unparseable.")
    log(f"Tokens — input {usage['input']:,} (cache read {usage['cache_read']:,}, "
        f"create {usage['cache_create']:,}) / output {usage['output']:,}")
    if errored or bad_json:
        log("Some tickets weren't normalized; re-run to submit a fresh batch for them.")


def submit(normalizer: Normalizer, groups: dict, manifest_path: str, *,
           char_limit: int, log=print) -> None:
    if len(groups) > MAX_REQUESTS_PER_BATCH:
        raise SystemExit(f"{len(groups):,} requests exceeds the "
                         f"{MAX_REQUESTS_PER_BATCH:,}/batch limit. Use --sample.")
    requests = [normalizer.make_request(cid, g["tickets"], char_limit)
                for cid, g in groups.items()]
    log(f"Submitting batch of {len(requests):,} requests …")
    batch = normalizer.client.messages.batches.create(requests=requests)
    save_manifest(manifest_path, batch.id, normalizer.model, groups)
    log(f"  batch {batch.id}  status={batch.processing_status}  (manifest → {manifest_path})")


# ── stage entry point ─────────────────────────────────────────────────────────

def run_normalize_stage(args, *, log=print) -> None:
    load_dotenv()
    records_path = getattr(args, "records", None) or f"{C.CACHE_DIR}/records.jsonl"
    manifest = C.NORMALIZE_BATCH_MANIFEST
    cache = C.NORMALIZE_CACHE
    batch_size = getattr(args, "batch_size", None) or C.NORMALIZE_BATCH_SIZE
    char_limit = getattr(args, "char_limit", None) or C.NORMALIZE_CHAR_LIMIT
    sample = getattr(args, "sample", None) or getattr(args, "limit", None)
    poll = getattr(args, "poll_interval", None) or 30

    # ── collect-only / reconnect an in-flight batch (double-spend guard) ──────
    if getattr(args, "collect", False):
        if not Path(manifest).exists():
            raise SystemExit(f"No {manifest} to collect. Submit a batch first.")
        collect(Normalizer(), manifest, cache, poll, log=log)
        return
    if Path(manifest).exists() and not getattr(args, "force_new", False):
        log(f"Found in-flight batch in {manifest} — resuming it "
            f"(use --force-new to discard).")
        collect(Normalizer(), manifest, cache, poll, log=log)
        return

    # ── build requests locally ────────────────────────────────────────────────
    log(f"Reading records … {records_path}")
    if not Path(records_path).exists():
        raise SystemExit(f"No {records_path}. Build it first: "
                         f"python run.py --stage records")
    items = read_tickets(records_path, log=log)
    if sample:
        items = items[:sample]
        log(f"  --sample/--limit → first {len(items):,} tickets")
    done = set(load_normalize_cache(cache))
    groups = build_groups(items, done, batch_size)
    log(f"  {len(groups):,} requests to submit "
        f"({len(done):,} already normalized, skipped)")

    if getattr(args, "dry_run", False):
        log("\n===== SYSTEM PROMPT (cached prefix) =====\n" + SYSTEM_PROMPT[:1800])
        if groups:
            first = next(iter(groups.values()))
            log("\n===== USER MESSAGE (first request) =====\n"
                + build_user_message(first["tickets"], char_limit)[:2000])
        approx_in = sum(len((t.get("title") or "") + (t.get("description") or "")
                            + (t.get("notes") or "")) for _, t in items) // 4
        log(f"\n===== PLAN =====\nrequests: {len(groups):,}  "
            f"~input tokens (uncached user): {approx_in:,}  "
            f"(model {C.NORMALIZE_MODEL}, batch_size {batch_size}) — NO API CALLS")
        return

    if getattr(args, "force_new", False):
        Path(manifest).unlink(missing_ok=True)
    if not groups:
        log("Nothing to submit — all tickets already normalized or empty.")
        return
    normalizer = Normalizer()
    submit(normalizer, groups, manifest, char_limit=char_limit, log=log)
    if getattr(args, "submit_only", False):
        log("Submitted. Collect later with:  python normalize.py --collect")
        return
    collect(normalizer, manifest, cache, poll, log=log)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=f"{C.CACHE_DIR}/records.jsonl")
    ap.add_argument("--sample", type=int, default=None, help="normalize only the first N tickets")
    ap.add_argument("--batch-size", type=int, default=C.NORMALIZE_BATCH_SIZE)
    ap.add_argument("--char-limit", type=int, default=C.NORMALIZE_CHAR_LIMIT)
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--submit-only", action="store_true", help="create the batch and exit")
    ap.add_argument("--collect", action="store_true", help="collect an already-submitted batch")
    ap.add_argument("--force-new", action="store_true", help="discard an in-flight manifest")
    ap.add_argument("--dry-run", action="store_true", help="print prompt + estimate, no API calls")
    args = ap.parse_args()
    run_normalize_stage(args)


if __name__ == "__main__":
    main()
