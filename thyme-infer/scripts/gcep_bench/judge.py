# Modified for anonymous review: release paths, configuration, and documentation.
"""Qwen3.5-397B text-only judge wrapper for gcep benchmarks.

Hard rules:
- text-only: ``images == []`` is asserted by construction (no image args exist);
- temperature=0, top_p=1, n=1, enable_thinking=False (thinking judges invalidate);
- strict parser ``Score:\\s*[01]`` — deliberate fix vs upstream ``"1" in response``;
- retries: transport errors exp-backoff x5; parse errors x2 same prompt/temp;
  a normal ``Score: 0`` is NEVER a retry condition;
- fail-closed: exhausted retries -> UNSCORED (never default 0); any UNSCORED in a
  headline metric marks the run INCOMPLETE;
- cache: JSONL, only successful schema-valid temperature=0 responses are cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# m_rlsd lives at <repo>/src; phase3_eval_opd.py inserts it before importing us,
# but keep this module standalone-importable for rescore/tests.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from m_rlsd.inference.remote_vlm_client import RemoteVLMClient, RemoteVLMConfig  # noqa: E402

SCORE_RE = re.compile(r"Score:\s*([01])\s*(?:\n|$)", re.IGNORECASE)


def _default_score_parser(raw_text: str) -> Optional[str]:
    m = SCORE_RE.search(raw_text)
    return m.group(1) if m else None

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "qwen3.5-397b-judge"

NO_THINKING_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass
class JudgeOutcome:
    score: Optional[int]  # 0/1, or None when UNSCORED (or non-binary verdict)
    status: str  # "SCORED" | "UNSCORED"
    raw_text: str = ""
    attempts: int = 0
    cache_hit: bool = False
    error: Optional[str] = None
    returned_model: str = ""
    verdict: Optional[str] = None  # parsed verdict string ("0"/"1"/"A"/"unclear"...)


@dataclass
class GcepJudgeConfig:
    base_url: str = field(default_factory=lambda: os.environ.get("JUDGE_BASE_URL", DEFAULT_BASE_URL))
    model: str = field(default_factory=lambda: os.environ.get("JUDGE_MODEL", DEFAULT_MODEL))
    api_key: str = field(default_factory=lambda: os.environ.get("JUDGE_API_KEY", ""))
    timeout_sec: float = 120.0
    max_transport_retries: int = 5
    max_parse_retries: int = 2
    # Leave room for an optional analysis prefix before the required score text.
    max_tokens: int = 512


class GcepTextJudge:
    """Fail-closed, cached, text-only judge client."""

    def __init__(self, config: Optional[GcepJudgeConfig] = None, *, cache_path: Optional[str] = None):
        self.config = config or GcepJudgeConfig()
        cfg = RemoteVLMConfig(
            base_url=self.config.base_url,
            model=self.config.model,
            api_key=self.config.api_key or "EMPTY",
            timeout_sec=self.config.timeout_sec,
            max_retries=0,  # transport retry policy implemented here
        )
        self.client = RemoteVLMClient(cfg)
        self.cache_path = cache_path
        self._cache: dict[str, dict] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("status") == "SCORED":
                        self._cache[row["cache_key"]] = row

    # ------------------------------------------------------------------ cache
    def cache_key(self, *, benchmark_id: str, dataset_revision: str, sample_id: str,
                  question: str, gold: str, prediction: str, prompt_hash: str,
                  verdict_schema: str = "score01", max_tokens: Optional[int] = None) -> str:
        h = hashlib.sha256()
        payload = {
            "schema_version": "1",
            "benchmark_id": benchmark_id,
            "dataset_revision": dataset_revision,
            "sample_id": str(sample_id),
            "content_sha256": hashlib.sha256(
                json.dumps([question, gold, prediction], ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "prompt_hash": prompt_hash,
            "judge_model_requested": self.config.model,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if verdict_schema != "score01":
            # Preserve cache compatibility for score01 payloads.
            payload["verdict_schema"] = verdict_schema
        h.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        return h.hexdigest()

    # ------------------------------------------------------------------ call
    def _one_call(self, prompt: str, max_tokens: Optional[int] = None) -> tuple[str, dict[str, Any]]:
        """Single transport attempt with exp-backoff retries (x5, jitter)."""
        effective_mt = max_tokens or self.config.max_tokens
        messages = [RemoteVLMClient.build_text_user_message(prompt)]
        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_transport_retries + 1):
            try:
                text, raw = self.client.generate_text(
                    messages,
                    temperature=0.0,
                    max_tokens=effective_mt,
                    extra_body=NO_THINKING_EXTRA_BODY,
                )
                return text, raw
            except Exception as exc:  # transport-level (RuntimeError from client)
                last_exc = exc
                if attempt < self.config.max_transport_retries:
                    sleep_s = (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(sleep_s)
        raise RuntimeError(f"judge transport failed after retries: {last_exc}")

    def judge(self, prompt: str, *, benchmark_id: str, dataset_revision: str,
              sample_id: str, question: str, gold: str, prediction: str,
              prompt_hash: str, verdict_parser=None, verdict_schema: str = "score01",
              max_tokens: Optional[int] = None) -> JudgeOutcome:
        """Fail-closed judged call.

        verdict_parser: optional callable raw_text -> Optional[str]. Returns the
        parsed verdict string, or None on parse failure (triggers bounded parse
        retry, then UNSCORED). Default: strict ``Score:\\s*[01]`` (score01 schema).
        A verdict of "0"/"1" also fills ``score``; other verdicts (e.g. "A",
        "unclear") leave ``score=None`` with status SCORED — the scorer decides.

        max_tokens: per-call decoding budget override (default config.max_tokens
        =512). At temperature=0 a completed response is budget-invariant (prefix
        property): raising the budget only lets truncated responses finish, it
        never changes a completed verdict. The budget is part of the cache key.
        """
        parser = verdict_parser or _default_score_parser
        key = self.cache_key(
            benchmark_id=benchmark_id, dataset_revision=dataset_revision,
            sample_id=sample_id, question=question, gold=gold,
            prediction=prediction, prompt_hash=prompt_hash,
            verdict_schema=verdict_schema, max_tokens=max_tokens,
        )
        if key in self._cache:
            row = self._cache[key]
            cached_verdict = row.get("verdict")
            if cached_verdict is None and row.get("score") is not None:
                cached_verdict = str(row["score"])
            return JudgeOutcome(score=row["score"], status="SCORED",
                                raw_text=row.get("raw_text", ""), cache_hit=True,
                                returned_model=row.get("returned_model", ""),
                                verdict=cached_verdict)

        if max_tokens is not None and max_tokens != self.config.max_tokens:
            # Budget-raise reuse (safe direction only): a verdict that COMPLETED
            # at a lower-or-equal budget is prefix-identical at temperature=0,
            # so it is valid for this higher-budget request. Never reuse in the
            # reverse direction (higher-budget verdict for a lower-budget call).
            alt_key = self.cache_key(
                benchmark_id=benchmark_id, dataset_revision=dataset_revision,
                sample_id=sample_id, question=question, gold=gold,
                prediction=prediction, prompt_hash=prompt_hash,
                verdict_schema=verdict_schema, max_tokens=None)
            row = self._cache.get(alt_key)
            if row is not None:
                row_budget = row.get("max_tokens") or self.config.max_tokens
                if row_budget <= max_tokens:
                    cached_verdict = row.get("verdict")
                    if cached_verdict is None and row.get("score") is not None:
                        cached_verdict = str(row["score"])
                    return JudgeOutcome(score=row["score"], status="SCORED",
                                        raw_text=row.get("raw_text", ""), cache_hit=True,
                                        returned_model=row.get("returned_model", ""),
                                        verdict=cached_verdict)

        attempts = 0
        raw_text = ""
        returned_model = ""
        last_error: Optional[str] = None
        base_budget = max_tokens if max_tokens is not None else self.config.max_tokens
        for parse_attempt in range(self.config.max_parse_retries + 1):
            attempts += 1
            # Increase the response budget by a factor of four on each parse retry.
            # The budget is part of the cache key; successful results use the actual budget.
            budget = base_budget * (4 ** parse_attempt)
            try:
                raw_text, raw = self._one_call(prompt, max_tokens=budget)
                choices = raw.get("choices") or []
                if choices:
                    returned_model = str(raw.get("model", ""))
            except Exception as exc:
                last_error = str(exc)
                break  # transport exhausted; no point retrying parse
            verdict = parser(raw_text.strip())
            if verdict is not None:
                score = int(verdict) if verdict in ("0", "1") else None
                self._store_cache(key, score, raw_text, returned_model, verdict,
                                  max_tokens=budget)
                return JudgeOutcome(score=score, status="SCORED", raw_text=raw_text,
                                    attempts=attempts, returned_model=returned_model,
                                    verdict=verdict)
            last_error = f"parse error: schema={verdict_schema} no verdict in output: {raw_text[:200]!r}"
            if parse_attempt < self.config.max_parse_retries:
                continue
        return JudgeOutcome(score=None, status="UNSCORED", raw_text=raw_text,
                            attempts=attempts, error=last_error,
                            returned_model=returned_model)

    def _store_cache(self, key: str, score: Optional[int], raw_text: str,
                     returned_model: str, verdict: Optional[str] = None,
                     max_tokens: Optional[int] = None) -> None:
        row = {"cache_key": key, "score": score, "status": "SCORED",
               "raw_text": raw_text, "returned_model": returned_model,
               "verdict": verdict,
               "max_tokens": max_tokens or self.config.max_tokens}
        self._cache[key] = row
        if self.cache_path:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def healthcheck(config: Optional[GcepJudgeConfig] = None) -> dict:
    cfg = config or GcepJudgeConfig()
    client = RemoteVLMClient(RemoteVLMConfig(
        base_url=cfg.base_url, model=cfg.model, api_key=cfg.api_key or "EMPTY"))
    return client.healthcheck()
