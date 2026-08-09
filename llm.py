#!/usr/bin/env python3
"""
Anthropic API wrapper for the three LLM passes (naming, category, judge).

Thin, retry-safe client with JSON parsing. The ANTHROPIC_API_KEY is read from
the environment only — never hardcoded. Set `dry_run=True` to return stub
outputs and make zero API calls (for smoke-testing the sweep wiring).
"""
from __future__ import annotations

import json
import os
import random
import re
import time

import config as C
import prompts


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


class LLM:
    def __init__(self, model: str = C.LLM_MODEL, *, dry_run: bool = False, max_retries: int = 5):
        self.model = model
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self._client = None
        if not dry_run:
            self._client = self._make_client()

    def _make_client(self):
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise SystemExit("pip install anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Set ANTHROPIC_API_KEY in your environment.")
        from anthropic import Anthropic
        return Anthropic()

    def _call(self, system: str, user: str, *, max_tokens: int = 1500):
        delay, last_err = 2.0, None
        for _ in range(self.max_retries):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                self.usage["calls"] += 1
                self.usage["input_tokens"] += resp.usage.input_tokens
                self.usage["output_tokens"] += resp.usage.output_tokens
                text = "".join(b.text for b in resp.content if b.type == "text")
                return json.loads(_strip_fences(text))
            except json.JSONDecodeError as e:
                last_err = f"bad JSON: {e}"
            except Exception as e:
                last_err = str(e)
                if any(k in last_err.lower() for k in ("rate", "429", "overload")):
                    delay = min(delay * 2, 60)
            time.sleep(delay + random.uniform(0, 1))
            delay = min(delay * 1.6, 60)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    def _call_streaming(self, system: str, user: str, *, max_tokens: int):
        """Streaming variant of _call. Required by the SDK when max_tokens is large
        (a non-streaming request that may exceed 10 min is rejected). Same parse
        contract (strip fences -> json.loads) and same retry/usage accounting."""
        delay, last_err = 2.0, None
        for _ in range(self.max_retries):
            try:
                with self._client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                ) as stream:
                    text = "".join(stream.text_stream)
                    final = stream.get_final_message()
                self.usage["calls"] += 1
                self.usage["input_tokens"] += final.usage.input_tokens
                self.usage["output_tokens"] += final.usage.output_tokens
                return json.loads(_strip_fences(text))
            except json.JSONDecodeError as e:
                last_err = f"bad JSON: {e}"
            except Exception as e:
                last_err = str(e)
                if any(k in last_err.lower() for k in ("rate", "429", "overload")):
                    delay = min(delay * 2, 60)
            time.sleep(delay + random.uniform(0, 1))
            delay = min(delay * 1.6, 60)
        raise RuntimeError(f"streaming call failed after {self.max_retries} attempts: {last_err}")

    # ── §2.4 name one cluster ───────────────────────────────────────────────
    def name_cluster(self, samples: list[dict], *, cluster_id: int = 0,
                     keywords: list[str] | None = None) -> dict:
        if self.dry_run:
            return {
                "title": f"Cluster {cluster_id}",
                "description": "Stub description (dry run).",
                "coherent": True,
                "coherence_note": "",
            }
        out = self._call(prompts.NAME_SYSTEM, prompts.build_name_user(samples, keywords))
        return {
            "title": str(out.get("title", f"Cluster {cluster_id}")).strip(),
            "description": str(out.get("description", "")).strip(),
            "coherent": bool(out.get("coherent", True)),
            "coherence_note": str(out.get("coherence_note", "")).strip(),
        }

    # ── §2.5 batched category assignment ────────────────────────────────────
    def assign_categories(self, named: list[dict]) -> dict[int, str]:
        if not named:
            return {}
        if self.dry_run:
            return {n["id"]: "Uncategorized" for n in named}
        # Output is one {id, category} per cluster, so max_tokens must scale with
        # the cluster count — a fixed cap truncates the JSON for large sweeps (HDBSCAN
        # can produce hundreds of clusters). k-means sizes stay near the old 3000.
        max_tokens = min(16000, 1000 + 32 * len(named))
        try:
            out = self._call(prompts.CATEGORY_SYSTEM, prompts.build_category_user(named),
                             max_tokens=max_tokens)
        except Exception as e:
            # Don't discard a completed run (naming/coverage/keywords) if the single
            # category pass fails — fall back to "Other" for all; can be re-run later.
            print(f"[llm] category assignment failed ({e}); defaulting all to 'Other'.")
            return {}
        result: dict[int, str] = {}
        for item in out:
            try:
                result[int(item["id"])] = str(item["category"]).strip()
            except (KeyError, ValueError, TypeError):
                continue
        # Fill any the model skipped
        for n in named:
            result.setdefault(n["id"], "Other")
        return result

    # ── §2.6 combo-level coherence / distinctness judge ─────────────────────
    def coherence_judge(self, clusters: list[dict]) -> dict:
        if self.dry_run:
            return {"coherence": 5.0, "distinctness": 5.0, "overall": 5.0, "notes": "dry run"}
        out = self._call(prompts.JUDGE_SYSTEM, prompts.build_judge_user(clusters), max_tokens=800)
        return {
            "coherence": float(out.get("coherence", 0)),
            "distinctness": float(out.get("distinctness", 0)),
            "overall": float(out.get("overall", 0)),
            "notes": str(out.get("notes", "")).strip(),
        }

    # ── §2.8 MECE consolidation (single big streaming call) ─────────────────
    def consolidate_taxonomy(self, clusters: list[dict]) -> dict:
        """Merge all named clusters into a CATEGORY -> WORKFLOW taxonomy. Returns the
        raw {taxonomy, unclassified} dict (consolidate.py validates + repairs it).
        dry_run → a stub that dumps every id into one workflow."""
        if self.dry_run:
            return {
                "taxonomy": [{
                    "category": "All (dry run)",
                    "category_description": "Stub — no LLM call.",
                    "workflows": [{
                        "name": "All clusters (dry run)",
                        "description": "Stub workflow holding every cluster.",
                        "member_cluster_ids": [c["cluster_id"] for c in clusters],
                        "merge_note": "dry run",
                    }],
                }],
                "unclassified": {"member_cluster_ids": [], "note": "dry run"},
            }
        return self._call_streaming(prompts.CONSOLIDATE_SYSTEM,
                                    prompts.build_consolidate_user(clusters),
                                    max_tokens=C.CONSOLIDATE_MAX_TOKENS)

    def place_missing_clusters(self, missing: list[int], clusters: list[dict],
                               wf_index: list[dict]) -> list[dict]:
        """Targeted follow-up: assign the ids the consolidation pass omitted to an
        existing workflow name (or 'unclassified'). Returns [{id, workflow}]."""
        if self.dry_run or not missing:
            return []
        out = self._call_streaming(prompts.PLACE_SYSTEM,
                                   prompts.build_place_user(missing, clusters, wf_index),
                                   max_tokens=4000)
        return out.get("assignments", []) if isinstance(out, dict) else []
