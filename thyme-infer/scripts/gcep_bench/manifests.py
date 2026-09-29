# Modified for anonymous review: release paths, configuration, and documentation.
"""Manifest / audit artifact helpers.

Artifacts per run dir:
  protocol_manifest.json / dataset_manifest.jsonl / judge_audit.jsonl /
  coverage_report.json / parity_report.json

All writes are JSON(L) with sorted keys for stable diffs.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable

SCHEMA_VERSION = "1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    os.replace(tmp, path)


def append_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_protocol_manifest(
    *,
    run_id: str,
    model_id: str,
    checkpoint_path: str,
    benchmark_id: str,
    dataset_revision: str,
    protocol_name: str,
    scoring_layer: str,
    generation_config: dict,
    prompt_hash: str,
    adapter_commit: str = "",
    judge: dict | None = None,
    deviations: list[str] | None = None,
) -> dict:
    """Protocol manifest. ``scoring_layer`` must be one of the
    five scoring labels (OFFICIAL_DETERMINISTIC / OFFICIAL_LLM_EXTRACTOR /
    OFFICIAL_LLM_JUDGE / ADAPTED_LLM_JUDGE / CUSTOM_DIAGNOSTIC)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "model_id": model_id,
        "checkpoint_path": checkpoint_path,
        "benchmark_id": benchmark_id,
        "dataset_revision": dataset_revision,
        "protocol_name": protocol_name,
        "scoring_layer": scoring_layer,
        "generation_config": generation_config,
        "prompt_hash": prompt_hash,
        "adapter_commit": adapter_commit,
        "judge": judge or {},
        "deviations_from_upstream": deviations or [],
    }


def check_coverage(
    predictions: list[dict],
    expected_ids: list[str],
    *,
    allow_unscored: bool = False,
) -> dict:
    """Prediction coverage check. Returns a report; ``ok`` False means STOP."""
    pred_ids = [str(p.get("sample_id", p.get("index", ""))) for p in predictions]
    dup = sorted({i for i in pred_ids if pred_ids.count(i) > 1})
    pred_set = set(pred_ids)
    expected_set = {str(i) for i in expected_ids}
    missing = sorted(expected_set - pred_set)
    extra = sorted(pred_set - expected_set)
    unscored = []
    if not allow_unscored:
        unscored = sorted(
            str(p.get("sample_id", p.get("index", "")))
            for p in predictions
            if str(p.get("status", "SUCCESS")) not in {"SUCCESS", "SCORED"}
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "n_expected": len(expected_set),
        "n_predictions": len(pred_ids),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "n_duplicate": len(dup),
        "n_unscored": len(unscored),
        "missing_ids": missing[:100],
        "extra_ids": extra[:100],
        "duplicate_ids": dup[:100],
        "unscored_ids": unscored[:100],
    }
    report["ok"] = not (missing or extra or dup or unscored)
    return report
