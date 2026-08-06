#!/usr/bin/env python3
"""
Section 1.4 — Semantic feature block.

Embed `title`, `description`, `notes` separately with OpenAI's
text-embedding-3-large (handles Dutch/English — no translation step needed).
Embeddings are cached per (ticket_id, field) on disk so re-running the sweep
never re-embeds.

Cache layout (one .npz per field):
    cache/emb_title.npz  cache/emb_description.npz  cache/emb_notes.npz
Each stores parallel arrays: `ids` (str) and `vecs` (float32, n×dim). A field's
empty-text tickets get a zero vector (and are still cached, so we never re-ask).

The OPENAI_API_KEY is read from the environment only — never hardcoded.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import config as C


def _cache_path(field: str, cache_dir: str) -> Path:
    return Path(cache_dir) / f"emb_{field}.npz"


def _load_cache(field: str, cache_dir: str) -> dict[str, np.ndarray]:
    p = _cache_path(field, cache_dir)
    if not p.exists():
        return {}
    d = np.load(p, allow_pickle=True)
    return {str(i): v for i, v in zip(d["ids"], d["vecs"])}


def _save_cache(field: str, cache_dir: str, cache: dict[str, np.ndarray]) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    ids = np.array(list(cache.keys()), dtype=object)
    vecs = np.vstack(list(cache.values())).astype(np.float32) if cache \
        else np.zeros((0, C.EMBED_DIM), dtype=np.float32)
    np.savez(_cache_path(field, cache_dir), ids=ids, vecs=vecs)


# OpenAI embeddings limits: <=8191 tokens/input, <=300k tokens/request, <=2048
# inputs/request. We batch under all three with margin.
MAX_INPUT_CHARS = 24000        # cheap pre-truncation before token counting
MAX_INPUT_TOKENS = 8000        # per-input hard cap (< 8191)
MAX_REQUEST_TOKENS = 250_000   # per-request token budget (< 300k, safety margin)


def _openai_client():
    try:
        import openai  # noqa: F401
    except ImportError:
        raise SystemExit("pip install openai")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment.")
    from openai import OpenAI
    return OpenAI()  # reads OPENAI_API_KEY from env


def _get_encoder():
    """tiktoken encoder for accurate token counts; None → char heuristic fallback."""
    try:
        import tiktoken
    except Exception:
        return None
    try:
        return tiktoken.encoding_for_model(C.EMBED_MODEL)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _prepare_text(text: str, encoder) -> tuple[str, int]:
    """Truncate one input to the per-input cap and return (text, size_estimate).

    IMPORTANT: OpenAI enforces its per-request token limit with a chars/4-style
    UPPER BOUND, not the real BPE count — verified empirically (a 24k-char input
    bills 3,857 tokens but counts as 6,000 against the request limit). So we
    budget on max(real_tokens, ceil(chars/4)) to match OpenAI's guard; using the
    true tiktoken count alone lets compressible text slip a request over 300k."""
    text = text[:MAX_INPUT_CHARS]
    n_real = 0
    if encoder is not None:
        toks = encoder.encode(text)
        if len(toks) > MAX_INPUT_TOKENS:
            toks = toks[:MAX_INPUT_TOKENS]
            text = encoder.decode(toks)
        n_real = len(toks)
    elif len(text) > MAX_INPUT_TOKENS * 4:          # no encoder: keep chars in lock-step
        text = text[:MAX_INPUT_TOKENS * 4]
    char_est = -(-len(text) // 4)                   # ceil(chars/4) — OpenAI's bound
    return text, max(n_real, char_est)


def _token_aware_batches(items, encoder, max_items: int):
    """Yield batches of (idx, prepared_text) that respect BOTH the per-request
    token budget and the max-items cap. `items` is an iterable of (idx, text)."""
    batch, batch_tokens = [], 0
    for idx, text in items:
        ptext, ntok = _prepare_text(text, encoder)
        if batch and (batch_tokens + ntok > MAX_REQUEST_TOKENS or len(batch) >= max_items):
            yield batch
            batch, batch_tokens = [], 0
        batch.append((idx, ptext))
        batch_tokens += ntok
    if batch:
        yield batch


def _embed_with_split(client, model, batch, log):
    """Send one batch; if OpenAI still rejects it for the per-request token limit
    (a residual estimate mismatch), split in half and retry recursively. Returns
    [(idx, embedding), ...]. Defense-in-depth on top of token-aware batching."""
    inputs = [t for _, t in batch]
    try:
        resp = client.embeddings.create(model=model, input=inputs, dimensions=C.EMBED_DIM)
        return [(batch[j][0], resp.data[j].embedding) for j in range(len(batch))]
    except Exception as e:
        if "max_tokens_per_request" in str(e) and len(batch) > 1:
            mid = len(batch) // 2
            log(f"[embed] request over token limit — splitting {len(batch)} → {mid}+{len(batch) - mid}")
            return (_embed_with_split(client, model, batch[:mid], log)
                    + _embed_with_split(client, model, batch[mid:], log))
        raise


def embed_field(
    records,
    field: str,
    *,
    cache_dir: str = C.CACHE_DIR,
    model: str = C.EMBED_MODEL,
    batch_size: int = 128,
    dry_run: bool = False,
    log=print,
) -> np.ndarray:
    """Return an (n_records, dim) matrix of embeddings for `field`, in record
    order. Uses/updates the on-disk (ticket_id, field) cache."""
    cache = _load_cache(field, cache_dir)
    texts = [(getattr(r, field) or "").strip() for r in records]
    ids = [r.ticket_id for r in records]

    # Which non-empty, uncached texts actually need an API call?
    todo_idx = [i for i, (tid, t) in enumerate(zip(ids, texts)) if t and tid not in cache]
    log(f"[embed:{field}] {len(records):,} records | "
        f"{len(cache):,} cached | {len(todo_idx):,} to embed")

    if dry_run:
        # Deterministic pseudo-random unit vectors (seeded by ticket_id+field) so
        # the sweep wiring can be exercised end-to-end with ZERO API spend. Never
        # written to the real cache. NOT a substitute for real embeddings.
        log(f"[embed:{field}] DRY RUN — stub vectors for {len(todo_idx):,} texts (no {model} calls)")
        dim = C.EMBED_DIM
        for i in todo_idx:
            seed = abs(hash((ids[i], field))) % (2**32)
            v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
            cache[ids[i]] = v / (np.linalg.norm(v) or 1.0)
    elif todo_idx:
        client = _openai_client()
        encoder = _get_encoder()
        if encoder is None:
            log(f"[embed:{field}] tiktoken not installed — using chars/4 token estimate")
        items = ((i, texts[i]) for i in todo_idx)
        done = 0
        for bnum, batch in enumerate(_token_aware_batches(items, encoder, batch_size)):
            for i, vec in _embed_with_split(client, model, batch, log):
                cache[ids[i]] = np.asarray(vec, dtype=np.float32)
            done += len(batch)
            if bnum % 10 == 0:
                log(f"[embed:{field}]   {done:,}/{len(todo_idx):,}")
        _save_cache(field, cache_dir, cache)
        log(f"[embed:{field}] cache now {len(cache):,} vectors → {_cache_path(field, cache_dir)}")

    # Assemble the ordered matrix; empty-text tickets -> zero vector.
    dim = C.EMBED_DIM
    for v in cache.values():
        dim = len(v)
        break
    out = np.zeros((len(records), dim), dtype=np.float32)
    for i, tid in enumerate(ids):
        if tid in cache:
            out[i] = cache[tid]
    return out


def embed_all_semantic(records, *, cache_dir: str = C.CACHE_DIR, dry_run: bool = False, log=print):
    """Embed every semantic field (title/description/notes) → {field: matrix}."""
    return {
        field: embed_field(records, field, cache_dir=cache_dir, dry_run=dry_run, log=log)
        for field in C.SEMANTIC_FIELDS
    }


if __name__ == "__main__":
    import argparse
    from records import load_records

    ap = argparse.ArgumentParser(description="Embed semantic fields (Section 1.4).")
    ap.add_argument("--records", default=f"{C.CACHE_DIR}/records.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    recs = load_records(args.records)
    embed_all_semantic(recs, dry_run=args.dry_run)
