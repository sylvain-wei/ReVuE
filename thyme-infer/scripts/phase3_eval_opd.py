# Modified for anonymous review: release paths, configuration, and documentation.
"""Evaluate released checkpoints using the configured benchmark protocols.

Usage from the package root, after activating the environment:
    torchrun --standalone --nproc_per_node=8 \
        thyme-infer/scripts/phase3_eval_opd.py \
        --model-path /path/to/checkpoint --output-dir /path/to/eval_output \
        --benchmarks HRBench4K --no-ats

DP design: each rank loads the model independently and processes
`dataset[rank::world_size]`. Each rank writes its partial predictions to
`predictions_{benchmark}_rank{r}.jsonl`. Rank 0 polls for all benchmark-specific
rank-done files (or complete per-rank shards), then aggregates into
`eval_{benchmark}.json` and `predictions_{benchmark}.jsonl`.
This avoids `torch.distributed.init_process_group` so the lightweight VLMEvalKit
Thyme model class works unchanged.
"""

import argparse
import contextlib
import hashlib
import json
import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
import random
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from m_rlsd.inference.remote_vlm_client import RemoteVLMConfig
from m_rlsd.judges import JudgeClient, score_vqa_outcome

DEFAULT_MAX_PIXELS = 0
OFFICIAL_THYME_PROMPT_RECIPE = "official_thyme_vlmevalkit"
DATASET_PROMPT_RECIPE = "dataset_build_prompt"
_MME_REALWORLD_FINAL_PATTERNS = (
    (
        "answer_prefix",
        re.compile(r"answer\s*[:：]\s*([A-E])\b", re.I),
    ),
    (
        "answer_tag",
        re.compile(r"<answer>\s*([A-E])\b.*?</answer>", re.I | re.S),
    ),
    (
        "final_answer_phrase",
        re.compile(
            r"(?:final answer|the correct answer|correct answer|the correct option|"
            r"correct option|the answer|answer)"
            r"(?:\s+based on the options provided)?"
            r"\s+is\s*[:：]?\s*\**\s*([A-E])\b",
            re.I,
        ),
    ),
)

# DP setup: under torchrun, restrict each rank to its own GPU BEFORE any
# `import torch` or `import vlmeval` happens, so that downstream calls like
# `device_map='auto'` (Thyme/model.py:94) shard the model onto the rank's
# single visible GPU instead of slicing it across all GPUs in every rank.
#
# We also have to STRIP `RANK / LOCAL_RANK / WORLD_SIZE` from the environment
# afterwards: transformers >=4.52 sees `WORLD_SIZE>0 + device_map="auto"` and
# silently rewrites it to `tp_plan="auto"` (transformers/modeling_utils.py:4175),
# which then requires `torch.distributed` to be initialised — and for our
# lightweight DP we deliberately don't init dist. Caching rank/world to module
# globals first preserves them for our own use; `_dp_env()` reads from these
# globals rather than from os.environ.
_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "-1"))
_RANK = int(os.environ.get("RANK", "0"))
_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
if _LOCAL_RANK >= 0:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _WORLD_SIZE > 1 and ("," in visible or visible == ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_LOCAL_RANK)
    # Hide torchrun env from transformers / accelerate so they don't auto-TP.
    # This is needed even for `torchrun --nproc_per_node=1`: transformers only
    # needs to see LOCAL_RANK/WORLD_SIZE to enable tensor-parallel planning, which
    # breaks Thyme's vision tower with "Sharding propagation failed".
    for _k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
               "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_RUN_ID",
               "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
               "TORCHELASTIC_ERROR_FILE", "GROUP_RANK", "ROLE_RANK",
               "ROLE_NAME", "LOCAL_WORLD_SIZE", "ROLE_WORLD_SIZE",
               "OMP_NUM_THREADS"):
        os.environ.pop(_k, None)

THYME_INFER_ROOT = os.environ.get("THYME_INFER_ROOT", str(REPO_ROOT / "thyme-infer"))

PHASE3_FOCUS_BENCHMARKS = [
    "VStarBench",
    "HRBench8K",
    "HRBench4K",
    "MME-RealWorld-Lite",
    "MathVista_MINI",
]
BENCHMARK_LIST = list(PHASE3_FOCUS_BENCHMARKS)
TOOL_USE_REPORT_META_KEY = "tool_use_report"
TOOL_USE_REPORT_SCRIPT = REPO_ROOT / "thyme-infer/scripts/build_overall_tool_use_report.py"
TOOL_USE_REPORT_CHECKPOINT_LABELS = {
    "SFT_CKPT": "Thyme-SFT 7B (SFT model baseline)",
    "gt_only_OPD": "GT-Only 7B OPD",
}


def _json_default(o):
    """Coerce numpy scalars (int64/float32/...) to native Python types for json.dumps."""
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _truncate_for_artifact(text: str, max_chars: int) -> tuple[str, bool]:
    """Keep prediction artifacts bounded without changing scoring inputs."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    marker = f"\n...[truncated {len(text) - max_chars} chars]...\n"
    keep_head = max(0, max_chars // 2)
    keep_tail = max(0, max_chars - keep_head)
    return text[:keep_head] + marker + text[-keep_tail:], True


def _generate_model_prediction(model, msg, benchmark: str, *, capture_output: bool) -> str:
    """Run Thyme generation while suppressing its extremely verbose debug prints."""
    if not capture_output:
        return model.generate(message=msg, dataset=benchmark)
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return model.generate(message=msg, dataset=benchmark)


# Regexes for tool-use extraction from conversation history
_CODE_BLOCK_RE = re.compile(
    r'<code>\s*(?:```\s*)?(?:python\s*)?([\s\S]*?)\s*(?:```\s*)?</code>', re.IGNORECASE
)
_SANDBOX_OUTPUT_RE = re.compile(
    r'<sandbox_output>\s*(.*?)\s*</sandbox_output>', re.DOTALL
)


def _extract_tool_use_features(model) -> dict:
    """Extract structured tool-use features from the model's last conversation history.

    After _generate_model_prediction, the Thyme model stores its full
    conversation_history (including all <code> blocks and <sandbox_output>
    segments) as self.last_conversation_history. This function parses that
    history to produce structured tool-use statistics for the predictions JSONL,
    so tool-use analysis no longer depends on --verbose run.log parsing.
    """
    conv = getattr(model, "last_conversation_history", None)
    empty = {
        "code_block_count": 0,
        "has_tool_use": False,
        "sandbox_output_count": 0,
        "tool_types": [],
        "full_response_chars": 0,
    }
    if conv is None:
        return empty

    # Concatenate all assistant text to get the full response
    full_response = ""
    for msg in conv:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if isinstance(content, str):
            full_response += content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    full_response += item.get("text", "")

    if not full_response:
        return empty

    # Count <code> blocks
    code_blocks = _CODE_BLOCK_RE.findall(full_response)

    # Count <sandbox_output> segments
    sandbox_outputs = _SANDBOX_OUTPUT_RE.findall(full_response)

    # Classify tool types
    tool_types = []
    for code in code_blocks:
        code_lower = code.lower()
        if any(kw in code_lower for kw in ['crop', 'resize', 'zoom', 'paste', 'slice', 'region', '.crop(']):
            tool_types.append("crop/zoom")
        elif any(kw in code_lower for kw in ['enhance', 'filter', 'blur', 'sharpen', 'contrast', 'brightness']):
            tool_types.append("image-enhance")
        elif any(kw in code_lower for kw in ['count', 'sum', 'len(', 'max(', 'min(', 'np.', 'calculate', 'math.']):
            tool_types.append("compute")
        else:
            tool_types.append("other")

    return {
        "code_block_count": len(code_blocks),
        "has_tool_use": len(code_blocks) > 0,
        "sandbox_output_count": len(sandbox_outputs),
        "tool_types": tool_types,
        "full_response_chars": len(full_response),
    }


def _dp_env():
    """Return (rank, world_size). Reads from cached module globals (set at top
    of file before we strip the launcher env vars). Falls back to (0, 1) if not
    under torchrun."""
    return (max(_RANK, 0), max(_WORLD_SIZE, 1))


def _rank_done_flag_path(flag_dir: Path, benchmark: str, rank: int) -> Path:
    return flag_dir / f"{benchmark}_rank{rank}.done"


def _rank_progress_path(progress_dir: Path, benchmark: str, rank: int) -> Path:
    return progress_dir / f"{benchmark}_rank{rank}.json"


def _partial_rank_predictions_path(output_dir: Path, benchmark: str, rank: int) -> Path:
    return output_dir / f"predictions_{benchmark}_rank{rank}.partial.jsonl"


def _load_rank_progress(progress_dir: Path, benchmark: str, rank: int) -> dict | None:
    path = _rank_progress_path(progress_dir, benchmark, rank)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _format_rank_progress(output_dir: Path, benchmark: str, world_size: int) -> str:
    progress_dir = output_dir / "_progress"
    parts: list[str] = []
    for rank in range(world_size):
        payload = _load_rank_progress(progress_dir, benchmark, rank)
        if not payload:
            parts.append(f"rank{rank}=missing")
            continue
        done = int(payload.get("done", 0) or 0)
        total = int(payload.get("total", 0) or 0)
        status = str(payload.get("status", "unknown") or "unknown")
        last_sample_id = payload.get("last_sample_id")
        piece = f"rank{rank}={done}/{total}"
        if status:
            piece += f"({status})"
        if last_sample_id not in (None, ""):
            piece += f"[last={last_sample_id}]"
        parts.append(piece)
    return ", ".join(parts)


def _write_rank_progress(
    progress_dir: Path,
    benchmark: str,
    rank: int,
    *,
    done: int,
    total: int,
    correct: int,
    parser_correct: int,
    has_answer: int,
    semantic_judge_used: int,
    semantic_judge_rescued: int,
    semantic_judge_invalid: int,
    status: str,
    last_sample_id: str = "",
) -> None:
    progress_dir.mkdir(exist_ok=True)
    payload = {
        "benchmark": benchmark,
        "rank": rank,
        "done": done,
        "total": total,
        "correct": correct,
        "parser_correct": parser_correct,
        "has_answer": has_answer,
        "semantic_judge_used": semantic_judge_used,
        "semantic_judge_rescued": semantic_judge_rescued,
        "semantic_judge_invalid": semantic_judge_invalid,
        "status": status,
        "last_sample_id": last_sample_id,
        "updated_at": time.time(),
    }
    _rank_progress_path(progress_dir, benchmark, rank).write_text(
        json.dumps(payload, default=_json_default)
    )


def _missing_rank_artifacts(output_dir: Path, benchmark: str, world_size: int):
    """Return missing per-rank shard/stat filenames for a benchmark."""
    missing_stats = []
    missing_predictions = []
    for r in range(world_size):
        stats_name = f"_stats_{benchmark}_rank{r}.json"
        pred_name = f"predictions_{benchmark}_rank{r}.jsonl"
        if not (output_dir / stats_name).exists():
            missing_stats.append(stats_name)
        if not (output_dir / pred_name).exists():
            missing_predictions.append(pred_name)
    return missing_stats, missing_predictions


def _wait_for_all_ranks(
    flag_dir: Path,
    benchmark: str,
    world_size: int,
    timeout_sec: float = 7200,
    output_dir: Path | None = None,
    poll_interval_sec: float = 5.0,
):
    """Block until a benchmark has enough per-rank artifacts to aggregate."""
    deadline = time.time() + max(0.0, float(timeout_sec))
    progress_report_every_sec = max(30.0, float(poll_interval_sec))
    last_progress_report_at = time.time()
    expected = {
        _rank_done_flag_path(flag_dir, benchmark, rank).name
        for rank in range(world_size)
    }
    glob_pattern = f"{benchmark}_rank*.done"

    while True:
        seen = {p.name for p in flag_dir.glob(glob_pattern)}
        if expected.issubset(seen):
            return "flags"

        if output_dir is not None:
            missing_stats, missing_predictions = _missing_rank_artifacts(
                output_dir, benchmark, world_size
            )
            if not missing_stats and not missing_predictions:
                return "artifacts"

        now = time.time()
        if output_dir is not None and now - last_progress_report_at >= progress_report_every_sec:
            print(
                f"[eval][rank0] still waiting for {benchmark}: "
                f"{_format_rank_progress(output_dir, benchmark, world_size)}"
            )
            last_progress_report_at = now

        if now >= deadline:
            break
        time.sleep(max(0.0, float(poll_interval_sec)))

    missing = sorted(expected - {p.name for p in flag_dir.glob(glob_pattern)})
    if output_dir is not None:
        missing_stats, missing_predictions = _missing_rank_artifacts(
            output_dir, benchmark, world_size
        )
        if not missing_stats and not missing_predictions:
            return "artifacts"
        raise TimeoutError(
            f"Timed out waiting for ranks: {missing}; "
            f"missing_stats={missing_stats}; missing_predictions={missing_predictions}; "
            f"progress={_format_rank_progress(output_dir, benchmark, world_size)}"
        )
    raise TimeoutError(f"Timed out waiting for ranks: {missing}")


def _requested_eval_recipe(benchmark: str) -> dict:
    recipe = {
        "version": 4,
        "benchmark": benchmark,
        "prompt_recipe": getattr(args, "prompt_recipe", OFFICIAL_THYME_PROMPT_RECIPE),
        "prompt_source": (
            "VLMEvalKit inference.py: model.build_prompt when "
            "model.use_custom_prompt(dataset), else dataset.build_prompt"
        ),
        "use_ats": bool(getattr(args, "use_ats", False)),
        "max_pixels": getattr(args, "max_pixels", DEFAULT_MAX_PIXELS),
        "scoring": (
            "explicit_final_answer_phrase_or_extract_characters_regex + "
            "get_dimension_rating"
            if _is_mme_realworld(benchmark)
            else "mcq_option_letter_extraction"
            if _is_mcq_benchmark(benchmark)
            else "legacy simple exact-normalized summary"
        ),
        "sampling": {
            "start_index": max(0, int(getattr(args, "start_index", 0))),
            "max_examples": int(getattr(args, "max_examples", 0) or 0),
            "sample_fraction": float(getattr(args, "sample_fraction", 0.0) or 0.0),
            "sample_seed": int(getattr(args, "sample_seed", 0)),
            "sample_indices_file": str(getattr(args, "sample_indices_file", "") or ""),
        },
        "semantic_judge": {
            "enabled": bool(getattr(args, "semantic_judge", False)),
            "base_url": str(getattr(args, "judge_base_url", "") or ""),
            "model": str(getattr(args, "judge_model", "") or ""),
            "temperature": float(getattr(args, "judge_temperature", 0.1)),
            "max_tokens": int(getattr(args, "judge_max_tokens", 256)),
            "attempts": int(getattr(args, "judge_attempts", 3)),
        },
    }

    if _is_gcep_bench(benchmark):
        # v5: gcep protocol extension. Non-gcep benchmarks keep the exact v4
        # recipe so existing focus-5 eval_*.json idempotent skips are unaffected.
        from gcep_bench.scorers import get_scorer

        scorer = get_scorer(benchmark)
        recipe["version"] = 5
        recipe["prompt_source"] = "dataset.build_prompt (official benchmark prompt; model mixin bypassed)"
        recipe["scoring"] = f"gcep_bench:{scorer.PROTOCOL_NAME}"
        recipe["generation_assert"] = {"top_k": 1, "top_p": 0.001, "temperature": 0.01}
        recipe["gcep"] = {
            "protocol_name": scorer.PROTOCOL_NAME,
            "scoring_layer": scorer.SCORING_LAYER,
            "dataset_size": GCEP_DATASET_SIZES[benchmark],
        }
        if benchmark in GCEP_SHORT_ANSWER_SUFFIX_BENCHMARKS:
            recipe["gcep"]["prompt_suffix"] = "vlm-evalkit-standard: Answer the question using a single word or phrase."
    return recipe


def _recipe_matches(existing: dict | None, requested: dict) -> bool:
    if not isinstance(existing, dict):
        return False
    return existing == requested


def _cleanup_benchmark_outputs(output_dir: Path, benchmark: str, world_size: int):
    for path in [
        output_dir / f"eval_{benchmark}.json",
        output_dir / f"eval_{benchmark}_rating.json",
        output_dir / f"eval_{benchmark}_detailed.jsonl",
        output_dir / f"predictions_{benchmark}.jsonl",
        output_dir / f"sample_{benchmark}.json",
        # Optional per-sample artifacts (absent for benchmarks that do not emit them)
        output_dir / f"tool_traces_{benchmark}.jsonl",
        output_dir / f"scores_{benchmark}.jsonl",
        output_dir / f"metrics_{benchmark}.json",
        output_dir / f"failures_{benchmark}.jsonl",
        output_dir / f"coverage_{benchmark}.json",
        output_dir / f"judge_audit_{benchmark}.jsonl",
    ]:
        path.unlink(missing_ok=True)

    flag_dir = output_dir / "_flags"
    progress_dir = output_dir / "_progress"
    for r in range(world_size):
        (output_dir / f"_stats_{benchmark}_rank{r}.json").unlink(missing_ok=True)
        (output_dir / f"predictions_{benchmark}_rank{r}.jsonl").unlink(missing_ok=True)
        _partial_rank_predictions_path(output_dir, benchmark, r).unlink(missing_ok=True)
        (output_dir / f"tool_traces_{benchmark}_rank{r}.jsonl").unlink(missing_ok=True)
        (output_dir / f"tool_traces_{benchmark}_rank{r}.partial.jsonl").unlink(missing_ok=True)
        _rank_done_flag_path(flag_dir, benchmark, r).unlink(missing_ok=True)
        _rank_progress_path(progress_dir, benchmark, r).unlink(missing_ok=True)


def _tool_use_checkpoint_label(model_path: str) -> str:
    explicit = str(getattr(args, "tool_use_checkpoint_label", "") or "").strip()
    if explicit:
        return explicit
    tag = str(getattr(args, "tag", "") or "").strip()
    if tag in TOOL_USE_REPORT_CHECKPOINT_LABELS:
        return TOOL_USE_REPORT_CHECKPOINT_LABELS[tag]
    model_name = Path(model_path).name.strip()
    if model_name == "Thyme-SFT":
        return "Thyme-SFT 7B (SFT model baseline)"
    return tag or model_name or str(model_path)


def _tool_use_log_path(output_dir: Path) -> Path:
    raw = str(getattr(args, "tool_use_log_path", "") or "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()
    return (output_dir / "run.log").resolve()


def _tool_use_report_meta(model_path: str, output_dir: Path, benchmark: str) -> dict | None:
    if benchmark not in PHASE3_FOCUS_BENCHMARKS:
        return None
    checkpoint_label = _tool_use_checkpoint_label(model_path)
    if not checkpoint_label:
        return None
    return {
        "checkpoint_label": checkpoint_label,
        "log_path": str(_tool_use_log_path(output_dir)),
        "tag": str(getattr(args, "tag", "") or ""),
        "model_path": str(Path(model_path).resolve()),
    }


def _refresh_overall_tool_use_report(output_dir: Path, benchmarks: list[str]):
    if not getattr(args, "refresh_tool_use_report", True):
        return
    if not any(bench in PHASE3_FOCUS_BENCHMARKS for bench in benchmarks):
        return

    log_path = _tool_use_log_path(output_dir)
    if not log_path.exists():
        print(
            f"[eval] WARNING: expected tool-use log at {log_path}, but it does not exist yet. "
            "overall_tool_use.md refresh will proceed, but the current run will only be picked up "
            "once that log path exists. Use --tool-use-log-path to override the location if needed."
        )

    if not TOOL_USE_REPORT_SCRIPT.exists():
        print(
            f"[eval] WARNING: tool-use report script not found at {TOOL_USE_REPORT_SCRIPT}; "
            "skipping overall_tool_use.md refresh."
        )
        return

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    try:
        completed = subprocess.run(
            [sys.executable, str(TOOL_USE_REPORT_SCRIPT)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
    except subprocess.CalledProcessError as exc:
        print("[eval] WARNING: failed to refresh overall_tool_use.md")
        if exc.stdout:
            print(exc.stdout.strip())
        if exc.stderr:
            print(exc.stderr.strip())


def _build_eval_message(model, ds, sample, benchmark: str):
    prompt_recipe = getattr(args, "prompt_recipe", OFFICIAL_THYME_PROMPT_RECIPE)
    if _is_gcep4_bench(benchmark):
        # Dedicated serializers (PerceptionBench multi-image
        # interleaving at <|image_N|> positions, HallusionBench text-only rows
        # without fake images). Never the Thyme model prompt mixin.
        from gcep_bench.serializers import build_messages as gcep4_build_messages
        return gcep4_build_messages(benchmark, sample)
    if _is_gcep_bench(benchmark):
        # GCEP benchmarks always use the dataset class's official prompt
        # (TreeBench OCR no-options rule / verbatim VQA questions), never the
        # Thyme model prompt mixin.
        msgs = ds.build_prompt(sample)
        if benchmark in GCEP_SHORT_ANSWER_SUFFIX_BENCHMARKS:
            # VLMEvalKit-standard short-answer suffix for DocVQA-family
            # benchmarks (recorded in prompt_hash and
            # protocol manifest). Scorer is untouched.
            for m in reversed(msgs):
                if m.get("type") == "text":
                    m["value"] += GCEP_SHORT_ANSWER_SUFFIX
                    break
        return msgs
    if prompt_recipe == OFFICIAL_THYME_PROMPT_RECIPE:
        # Match VLMEvalKit's inference.py prompt dispatch exactly.
        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(benchmark):
            assert hasattr(model, "build_prompt")
            return model.build_prompt(sample, dataset=benchmark)
        return ds.build_prompt(sample)
    if prompt_recipe == DATASET_PROMPT_RECIPE:
        return ds.build_prompt(sample)
    raise ValueError(f"Unsupported prompt recipe: {prompt_recipe}")


def _load_sample_indices(path: Path, benchmark: str, n: int) -> list[int]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        indices = payload
    elif isinstance(payload, dict):
        if "indices" in payload:
            indices = payload["indices"]
        elif benchmark in payload:
            indices = payload[benchmark]
        elif "benchmarks" in payload and benchmark in payload["benchmarks"]:
            indices = payload["benchmarks"][benchmark]
        else:
            raise ValueError(
                f"{path} does not contain indices for benchmark {benchmark!r}"
            )
    else:
        raise ValueError(f"Unsupported sample indices format in {path}")

    parsed = [int(i) for i in indices]
    bad = [i for i in parsed if i < 0 or i >= n]
    if bad:
        raise ValueError(f"{path} contains out-of-range indices, e.g. {bad[:5]}")
    return sorted(set(parsed))


def _select_eval_indices(benchmark: str, n: int) -> tuple[list[int], dict]:
    start_index = max(0, int(getattr(args, "start_index", 0)))
    max_examples = int(getattr(args, "max_examples", 0) or 0)
    selected_indices = list(range(start_index, n))
    if max_examples > 0:
        selected_indices = selected_indices[:max_examples]

    sample_indices_file = str(getattr(args, "sample_indices_file", "") or "")
    sample_fraction = float(getattr(args, "sample_fraction", 0.0) or 0.0)
    sample_seed = int(getattr(args, "sample_seed", 0))

    method = "sequential"
    if sample_indices_file:
        selected_indices = _load_sample_indices(Path(sample_indices_file), benchmark, n)
        method = "indices_file"
    elif sample_fraction > 0:
        if sample_fraction > 1:
            raise ValueError("--sample-fraction must be in [0, 1]")
        k = max(1, round(len(selected_indices) * sample_fraction))
        seed_material = f"{sample_seed}:{benchmark}:{n}:{start_index}:{max_examples}:{sample_fraction}"
        seed_int = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed_int)
        selected_indices = sorted(rng.sample(selected_indices, k))
        method = "random_fraction"

    fingerprint = hashlib.sha1(
        ",".join(str(i) for i in selected_indices).encode("utf-8")
    ).hexdigest()
    sampling_info = {
        "method": method,
        "dataset_size": n,
        "selected_count": len(selected_indices),
        "start_index": start_index,
        "max_examples": max_examples,
        "sample_fraction": sample_fraction,
        "sample_seed": sample_seed,
        "sample_indices_file": sample_indices_file,
        "indices_sha1": fingerprint,
        "indices": selected_indices,
    }
    return selected_indices, sampling_info


def _is_mme_realworld(benchmark: str) -> bool:
    return benchmark in {"MME-RealWorld", "MME-RealWorld-CN", "MME-RealWorld-Lite"}


def _is_mcq_benchmark(benchmark: str) -> bool:
    return benchmark in {"VStarBench", "MMStar", "HRBench4K", "HRBench8K"}


# Additional benchmark adapters.
# Scoring authority is the gcep_bench package; these benchmarks always use the
# DATASET's official build_prompt (never the Thyme model prompt mixin) so that
# benchmark-specific prompt rules (e.g. TreeBench OCR no-options rule) hold.
GCEP_BENCHMARKS = (
    "TreeBench",
    "ZoomBench",
    "VisualProbe_Easy",
    "VisualProbe_Medium",
    "VisualProbe_Hard",
    "ReasonMapPlus",
    "ChartQAPro",
    "InfographicVQA_val",
    # Additional scorers: PerceptionBench / HallusionBench /
    # MathVerse Vision-Only / VisuLogic.
    "PerceptionBench",
    "HallusionBench_GCEP",
    "MathVerseVO",
    "VisuLogic_GCEP",
)

# These benchmarks use multi-image/text-only serializers, per-sample tool
# traces, run manifests and detailed failure classification.
GCEP4_BENCHMARKS = frozenset({
    "PerceptionBench", "HallusionBench_GCEP", "MathVerseVO", "VisuLogic_GCEP",
})

# Expected FULL dataset sizes. Used by the rank-0
# coverage gate: a full run whose prediction count differs stops the pipeline.
GCEP_DATASET_SIZES = {
    "TreeBench": 405,
    "ZoomBench": 845,
    "VisualProbe_Easy": 141,
    "VisualProbe_Medium": 268,
    "VisualProbe_Hard": 106,
    "ReasonMapPlus": 1448,
    "ChartQAPro": 1948,
    "InfographicVQA_val": 2801,
    "PerceptionBench": 3000,
    "HallusionBench_GCEP": 1129,
    "MathVerseVO": 788,
    "VisuLogic_GCEP": 1000,
}


def _is_gcep_bench(benchmark: str) -> bool:
    return benchmark in GCEP_BENCHMARKS


def _is_gcep4_bench(benchmark: str) -> bool:
    return benchmark in GCEP4_BENCHMARKS


# DocVQA-family benchmarks get the VLMEvalKit-standard short-answer suffix
# (verbatim from vlmeval ImageVQADataset.build_prompt). All other gcep
# benchmarks keep verbatim questions.
GCEP_SHORT_ANSWER_SUFFIX_BENCHMARKS = frozenset({"ChartQAPro", "InfographicVQA_val"})
GCEP_SHORT_ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."


def _score_gcep_progress(benchmark: str, sample, pred: str) -> tuple[bool, dict]:
    """In-loop progress signal for gcep benchmarks (NOT the authoritative score).

    Deterministic scorers run judge-free here; judge-backed benchmarks
    (VisualProbe) are deferred to the rank-0 post-hoc aggregation, which is the
    only authoritative scoring path.
    """
    from gcep_bench.scorers import get_scorer

    gt = str(sample.get("answer", sample.get("answers", ""))).strip()
    scorer = get_scorer(benchmark)
    if getattr(scorer, "REQUIRES_JUDGE", False) and (
            benchmark.startswith("VisualProbe")
            or getattr(scorer, "DEFER_WITHOUT_JUDGE", False)):
        # Judge-backed scorers that cannot run judge-free defer to the rank-0
        # post-hoc pass (VisualProbe, PerceptionBench, HallusionBench,
        # MathVerseVO). Scorers with a judge-free deterministic stage
        # (ZoomBench, VisuLogic_GCEP) still give an in-loop progress signal.
        return False, {
            "gt": gt,
            "extracted": "",
            "official_extracted": "",
            "extraction_source": "",
            "score_method": "gcep_deferred_to_posthoc",
        }
    sample_dict = sample.to_dict() if hasattr(sample, "to_dict") else dict(sample)
    try:
        from gcep_bench.common import strip_think_tags
        # Scorer view: drop thinking blocks / a lone </think> prefix so thinking
        # subject models don't push their whole CoT into the answer extraction.
        score, meta = scorer.score_sample(sample_dict, strip_think_tags(pred or ""),
                                          judge=None, benchmark=benchmark)
    except Exception as exc:  # progress signal must never kill the eval loop
        return False, {
            "gt": gt,
            "extracted": "",
            "official_extracted": "",
            "extraction_source": "",
            "score_method": f"gcep_progress_error:{type(exc).__name__}",
        }
    meta = meta or {}
    return bool(score is not None and score >= 1.0), {
        "gt": gt,
        "extracted": str(meta.get("extracted", meta.get("extracted_prediction", "")))[:200],
        "official_extracted": "",
        "extraction_source": str(meta.get("score_method", "")),
        "score_method": f"gcep:{scorer.PROTOCOL_NAME}",
        "gcep_score": score,
    }


def _mcq_choices_from_sample(sample) -> list[str]:
    """Collect available option letters from a VLMEvalKit MCQ row."""
    choices: list[str] = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if letter not in sample:
            break
        value = sample.get(letter)
        if value is None:
            break
        text = str(value).strip()
        if not text or text.lower() == "nan":
            break
        choices.append(letter)
    return choices or list("ABCD")


def _extract_mcq_choice(pred: str, choices: list[str]) -> tuple[str, str]:
    """Extract a single MCQ letter from a free-form model response."""
    pred = pred or ""
    if not pred.strip():
        return "", "empty"

    choice_set = {c.upper() for c in choices}
    for line in reversed(pred.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        for pattern, source in (
            (r"^([A-Z])\s*[\.\):\-]", "leading_option_letter"),
            (r"(?:answer|option|choice)\s*(?:is|:)\s*\**\s*([A-Z])\b", "answer_phrase"),
            (r"\b([A-Z])\s*[\.\):]\s*\S", "inline_option_letter"),
            (r"^([A-Z])\b", "line_start_letter"),
        ):
            match = re.search(pattern, line, flags=re.I)
            if not match:
                continue
            letter = match.group(1).upper()
            if letter in choice_set:
                return letter, source

    # VLMEvalKit-style fallback: unique option letter near the response tail.
    tail = pred[-512:] if len(pred) > 512 else pred
    tail_mod = re.sub(r"[.()[\],:;!*#{}]", " ", tail)
    splits = [tok.strip().upper() for tok in tail_mod.split() if tok.strip()]
    hits = [tok for tok in splits if tok in choice_set]
    if len(set(hits)) == 1:
        return hits[-1], "tail_unique_option_letter"
    return "", "unparsed"


def _extract_mme_realworld_choice(pred: str) -> tuple[str, dict]:
    """Extract the final multiple-choice answer for MME-RealWorld.

    VLMEvalKit's `extract_characters_regex` is a good fallback, but it can be
    confused by long reasoning traces that enumerate options (A/B/C/D) before
    the model states a final answer near the end.  We therefore first look for
    an explicit final-answer phrase in the tail of the response and only fall
    back to the official regex when no such phrase exists.
    """
    from vlmeval.dataset.utils.multiple_choice import extract_characters_regex

    pred = pred or ""
    official_extracted = extract_characters_regex(pred)
    tail = pred[-1024:] if len(pred) > 1024 else pred
    tail_for_match = tail.replace("*", "")
    matches: list[tuple[int, str, str]] = []
    for source, pattern in _MME_REALWORLD_FINAL_PATTERNS:
        for match in pattern.finditer(tail_for_match):
            matches.append((match.start(), match.group(1).upper(), source))

    if matches:
        _, extracted, extraction_source = max(matches, key=lambda item: item[0])
        return extracted, {
            "official_extracted": official_extracted,
            "extraction_source": extraction_source,
        }

    return official_extracted, {
        "official_extracted": official_extracted,
        "extraction_source": "extract_characters_regex",
    }


def _extract_first_image_path(message) -> str:
    """Return the first local image path embedded in a VLMEvalKit prompt."""
    for item in message or []:
        if isinstance(item, dict) and item.get("type") == "image":
            value = str(item.get("value", "")).strip()
            if value:
                return value
    return ""


def _mcq_option_map(sample) -> dict[str, str]:
    """Return {letter: option_text} for any MCQ-style benchmark row."""
    option_map: dict[str, str] = {}
    for letter in _mcq_choices_from_sample(sample):
        text = str(sample.get(letter, "")).strip()
        if text and text.lower() != "nan":
            option_map[letter] = text
    return option_map


def _question_for_semantic_judge(benchmark: str, sample) -> str:
    """Build the judge question text, appending MCQ options when present."""
    question = str(sample.get("question", "")).strip()
    if _is_mcq_benchmark(benchmark) or _is_mme_realworld(benchmark):
        option_map = _mcq_option_map(sample)
        if option_map:
            option_lines = "\n".join(f"{letter}. {text}" for letter, text in option_map.items())
            question = f"{question}\n\nOptions:\n{option_lines}"
    return question


def _reference_answer_for_semantic_judge(sample) -> str:
    """Map a letter-labeled MCQ answer to its text so semantic judges can reason."""
    gt = str(sample.get("answer", "")).strip()
    option_map = _mcq_option_map(sample)
    if gt.upper() in option_map:
        return f"Option {gt.upper()}: {option_map[gt.upper()]}"
    return gt


def _build_semantic_judge_client(rank: int) -> JudgeClient | None:
    """Create the remote semantic-judge client when enabled."""
    if not getattr(args, "semantic_judge", False):
        return None

    cfg = RemoteVLMConfig(
        base_url=str(getattr(args, "judge_base_url", "") or ""),
        model=str(getattr(args, "judge_model", "") or ""),
        api_key=str(getattr(args, "judge_api_key", "EMPTY") or "EMPTY"),
        timeout_sec=float(getattr(args, "judge_timeout_sec", 120.0)),
        max_retries=int(getattr(args, "judge_http_retries", 2)),
        retry_backoff_sec=float(getattr(args, "judge_retry_backoff_sec", 2.0)),
    )
    client = JudgeClient(cfg)
    if rank == 0:
        served_models = [entry.get("id", "") for entry in client.healthcheck().get("data", [])]
        print(
            f"[eval][rank0] semantic judge enabled: base_url={client.remote_client.base_url} "
            f"model={cfg.model} served_models={served_models}"
        )
        if cfg.model not in served_models:
            raise RuntimeError(
                f"semantic judge model {cfg.model!r} not found in remote /models: {served_models}"
            )
    return client


def _apply_semantic_judge(
    benchmark: str,
    sample,
    pred: str,
    parser_correct: bool,
    message,
    judge_client: JudgeClient | None,
    *,
    judge_temperature: float,
    judge_max_tokens: int,
    judge_attempts: int,
) -> tuple[bool, dict]:
    """Rescue parser-failed items with the configured remote outcome judge."""
    meta = {
        "semantic_judge_used": False,
        "semantic_judge_rescued": False,
        "semantic_judge_valid": False,
        "semantic_judge_score": None,
        "semantic_judge_error": None,
        "semantic_judge_raw_text": "",
        "semantic_judge_attempts": 0,
        "semantic_judge_reference_answer": "",
    }
    if parser_correct or judge_client is None:
        return parser_correct, meta
    pred = (pred or "").strip()
    if not pred:
        return parser_correct, meta

    image_path = _extract_first_image_path(message)
    if not image_path:
        meta["semantic_judge_error"] = "missing_image_path_for_semantic_judge"
        return parser_correct, meta

    question = _question_for_semantic_judge(benchmark, sample)
    reference_answer = _reference_answer_for_semantic_judge(sample)
    meta["semantic_judge_reference_answer"] = reference_answer

    result = None
    for attempt in range(1, max(1, judge_attempts) + 1):
        meta["semantic_judge_used"] = True
        meta["semantic_judge_attempts"] = attempt
        try:
            result = score_vqa_outcome(
                judge_client,
                question=question,
                candidate_answer=pred,
                reference_answer=reference_answer,
                image_path=image_path,
                temperature=judge_temperature,
                max_tokens=judge_max_tokens,
                # Disable thinking for bounded, parseable judge responses.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as exc:
            # Remote judges occasionally drop long-running HTTP connections.
            # Treat that sample as an unrescued parser failure instead of killing
            # the whole distributed eval.
            meta["semantic_judge_valid"] = False
            meta["semantic_judge_score"] = None
            meta["semantic_judge_error"] = f"{type(exc).__name__}: {exc}"
            meta["semantic_judge_raw_text"] = ""
            continue
        meta["semantic_judge_valid"] = bool(result.valid)
        meta["semantic_judge_score"] = result.score
        meta["semantic_judge_error"] = result.error
        meta["semantic_judge_raw_text"] = result.raw_text
        if result.valid:
            rescued = bool(result.score and result.score >= 1.0)
            meta["semantic_judge_rescued"] = rescued
            return rescued, meta

    return parser_correct, meta


def _score_prediction(benchmark: str, sample, pred: str) -> tuple[bool, dict]:
    """Return a lightweight correctness signal for progress and summaries.

    MME-RealWorld has an official non-LLM extraction rule in VLMEvalKit; use it
    here so `eval_*.json` matches the benchmark's intended multiple-choice
    scoring rather than the old whole-string exact match.
    """
    gt = str(sample.get("answer", "")).strip()
    pred = pred or ""

    if _is_mme_realworld(benchmark):
        extracted, extraction_info = _extract_mme_realworld_choice(pred)
        return extracted == gt, {
            "gt": gt,
            "extracted": extracted,
            "official_extracted": extraction_info["official_extracted"],
            "extraction_source": extraction_info["extraction_source"],
            "score_method": (
                "mme_realworld_explicit_final_answer_phrase_or_"
                "extract_characters_regex"
            ),
        }

    if _is_mcq_benchmark(benchmark):
        choices = _mcq_choices_from_sample(sample)
        extracted, extraction_source = _extract_mcq_choice(pred, choices)
        return extracted.upper() == gt.upper(), {
            "gt": gt,
            "extracted": extracted,
            "official_extracted": "",
            "extraction_source": extraction_source,
            "score_method": "mcq_option_letter_extraction",
        }

    if _is_gcep_bench(benchmark):
        return _score_gcep_progress(benchmark, sample, pred)

    # Simple answer matching for legacy quick checks. Benchmark-specific
    # summaries should be added here instead of relying on whole-string equality.
    from phase3_join_gt_labels import answers_match

    return answers_match(pred, gt), {
        "gt": gt,
        "extracted": pred,
        "score_method": "simple_exact_normalized",
    }


def _summarize_mme_realworld(ds, prediction_rows: list[dict]) -> dict:
    """Mirror VLMEvalKit's MME-RealWorld rating structure from JSONL rows."""
    pred_by_id = {str(row["id"]): row for row in prediction_rows}
    tasks = ["Reasoning", "Perception"]
    subtasks = [
        "Monitoring",
        "Autonomous_Driving",
        "OCR with Complex Context",
        "Diagram and Table",
        "Remote Sensing",
    ]
    results: dict[str, dict] = {"Overall": {}}
    for task in tasks:
        results[task] = {subtask: {} for subtask in subtasks}

    total = correct = failed_extract = 0
    detailed = []

    for _, sample in ds.data.iterrows():
        sid = str(sample["index"])
        row = pred_by_id.get(sid)
        if row is None:
            continue

        gt = str(sample.get("answer", "")).strip()
        pred = row.get("pred", "") or ""
        extracted = str(row.get("extracted", "") or "")
        score = int(bool(row.get("correct", extracted == gt)))
        failed_extract += int(extracted == "")
        total += 1
        correct += score

        category = str(sample["category"])
        task, subtask = category.split("/", 1)
        leaf = str(sample["l2-category"]).lower()
        if "attribute" in leaf:
            leaf = leaf.split("/")[0] + "/attribute"

        bucket = results[task][subtask].setdefault(leaf, {"true": 0, "false": 0})
        bucket["true"] += score
        bucket["false"] += 1 - score

        detailed.append({
            "id": sid,
            "gt": gt,
            "pred": pred,
            "extracted": extracted,
            "score": score,
            "parser_correct": bool(row.get("parser_correct", extracted == gt)),
            "semantic_judge_used": bool(row.get("semantic_judge_used", False)),
            "semantic_judge_rescued": bool(row.get("semantic_judge_rescued", False)),
            "category": category,
            "l2_category": str(sample["l2-category"]),
        })

    for task, task_values in list(results.items()):
        if task == "Overall":
            continue
        task_correct = task_total = 0
        for subtask, subtask_values in task_values.items():
            subtask_correct = subtask_total = 0
            for category, counts in list(subtask_values.items()):
                cat_total = counts["true"] + counts["false"]
                cat_acc = counts["true"] / cat_total if cat_total else 0.0
                subtask_values[category] = cat_acc
                subtask_correct += counts["true"]
                subtask_total += cat_total
            subtask_values["Avg"] = subtask_correct / subtask_total if subtask_total else 0.0
            task_correct += subtask_correct
            task_total += subtask_total
        task_values["Avg"] = task_correct / task_total if task_total else 0.0

    results["Overall"] = correct / total if total else 0.0
    return {
        "accuracy": correct / max(1, total),
        "correct": correct,
        "total": total,
        "failed_extract": failed_extract,
        "rating": results,
        "detailed": detailed,
    }


def run_vlmeval(model_path: str, benchmark: str, output_dir: Path, verbose: bool = False):
    """Run VLMEvalKit evaluation for one benchmark.  Returns result dict or None.

    Behavior under torchrun (WORLD_SIZE > 1): each rank processes
    `dataset[rank::world_size]` and writes its partial predictions, then rank 0
    aggregates. Non-zero ranks return None.
    """
    assert THYME_INFER_ROOT, "source thyme-infer/activate_thyme.sh first"

    rank, world_size = _dp_env()
    is_dp = world_size > 1
    judge_client = _build_semantic_judge_client(rank)

    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"eval_{benchmark}.json"
    requested_recipe = _requested_eval_recipe(benchmark)

    # Idempotent skip on rank 0; in DP mode let other ranks short-circuit too
    if result_file.exists():
        existing = json.loads(result_file.read_text())
        if _recipe_matches(existing.get("eval_recipe"), requested_recipe):
            if rank == 0:
                print(f"[eval] {benchmark}: already done, loading {result_file}")
                return existing
            return None
        if not getattr(args, "overwrite", False):
            raise RuntimeError(
                f"{result_file} exists but was produced with a different or missing eval_recipe. "
                "Use --overwrite or a fresh --output-dir for the official-aligned recipe."
            )
        print(f"[eval][rank{rank}/{world_size}] overwriting stale/mismatched outputs for {benchmark}")
        _cleanup_benchmark_outputs(output_dir, benchmark, world_size)

    import json as _json

    from vlmeval.dataset import build_dataset
    from vlmeval.vlm import Thyme, ThymeInternVL
    try:
        from vlmeval.vlm import ThymeQwen3VL  # Metis（Qwen3-VL）；thyme env 下无此类时保持 None
    except Exception:  # ImportError in old envs (transformers<4.57)
        ThymeQwen3VL = None
    try:
        from vlmeval.vlm import ThymeQwen3VLMoe  # Qwen3-VL MoE（Qwen3-VL-30B-A3B-*）
    except Exception:  # ImportError in old envs (transformers<4.57)
        ThymeQwen3VLMoe = None

    cfg_path = Path(THYME_INFER_ROOT) / "configs" / "thyme_local.json"
    with open(cfg_path) as f:
        cfg = _json.load(f)

    # Resolve which model class + config entry to use.
    # Default behavior is unchanged for Qwen2.5-VL (Thyme-RL-local).
    # InternVL3.5 (architectures contains InternVLChatModel) auto-selects
    # Thyme-InternVL-local. Qwen3-VL (Metis; Qwen3VLForConditionalGeneration)
    # auto-selects Thyme-Qwen3VL-local. --model-class CLI overrides auto-detection.
    model_class_override = getattr(args, "model_class", "auto")
    if model_class_override == "thyme_internvl":
        cfg_key, model_cls = "Thyme-InternVL-local", ThymeInternVL
    elif model_class_override == "thyme_qwen3vl":
        assert ThymeQwen3VL is not None, "ThymeQwen3VL unavailable (needs transformers>=4.57 env)"
        cfg_key, model_cls = "Thyme-Qwen3VL-local", ThymeQwen3VL
    elif model_class_override == "thyme_qwen3vl_moe":
        assert ThymeQwen3VLMoe is not None, "ThymeQwen3VLMoe unavailable (needs transformers>=4.57 env)"
        cfg_key, model_cls = "Thyme-Qwen3VL-local", ThymeQwen3VLMoe
    elif model_class_override == "thyme":
        cfg_key, model_cls = "Thyme-RL-local", Thyme
    else:
        # auto: read model config.json architectures.
        arch_path = Path(model_path) / "config.json"
        cfg_key, model_cls = "Thyme-RL-local", Thyme
        if arch_path.exists():
            try:
                arch_cfg = _json.loads(arch_path.read_text())
                archs = arch_cfg.get("architectures", []) or []
                if any("InternVLChatModel" in a for a in archs):
                    cfg_key, model_cls = "Thyme-InternVL-local", ThymeInternVL
                elif any("Qwen3VLMoe" in a for a in archs):
                    assert ThymeQwen3VLMoe is not None, "Qwen3-VL MoE ckpt detected but transformers<4.57 env"
                    cfg_key, model_cls = "Thyme-Qwen3VL-local", ThymeQwen3VLMoe
                elif any("Qwen3VL" in a for a in archs):
                    assert ThymeQwen3VL is not None, "Metis ckpt detected but transformers<4.57 env"
                    cfg_key, model_cls = "Thyme-Qwen3VL-local", ThymeQwen3VL
            except Exception as exc:
                if rank == 0:
                    print(f"[eval] WARN: could not read {arch_path}: {exc}; defaulting to Thyme (Qwen).")
        else:
            # No config.json at the path (HF repo id etc.): fall back to Thyme.
            if rank == 0:
                print(f"[eval] no config.json at {arch_path}; defaulting to Thyme (Qwen).")

    if cfg_key not in cfg["model"]:
        raise RuntimeError(
            f"Model class {model_class_override!r} resolved to config key "
            f"{cfg_key!r} which is not present in {cfg_path}."
        )

    # Use the resolved config but override model_path
    model_kwargs = {
        k: v for k, v in cfg["model"][cfg_key].items() if k != "class"
    }
    # Resolve local paths to absolute. HF's from_pretrained otherwise treats
    # multi-segment relatives like "thyme-infer/outputs/.../final_model" as
    # an HF repo id and raises HFValidationError ("must be 'repo' or 'org/repo'").
    mp = Path(model_path)
    if mp.exists():
        model_path = str(mp.resolve())
    model_kwargs["model_path"] = model_path
    if getattr(args, "max_pixels", 0) and args.max_pixels > 0:
        model_kwargs["max_pixels"] = args.max_pixels
    # Generation budget override (additive; 0 = keep the model class default 2048).
    # Needed for *thinking* subject models, whose CoT otherwise consumes the whole
    # budget and truncates the final answer mid-sentence (observed on
    # Qwen3-VL-30B-A3B-Thinking: VisuLogic responses cut at 8-10k chars -> 3.5%).
    if getattr(args, "max_new_tokens", 0) and args.max_new_tokens > 0:
        model_kwargs["max_new_tokens"] = args.max_new_tokens
    # thyme_local.json historically has verbose=true for interactive debugging.
    # Batch evals must override it, otherwise every prompt / code block / response
    # is printed into run.log and long sweeps can hit disk quota.
    if model_kwargs.get("verbose") and not getattr(args, "verbose", False) and rank == 0:
        print(f"[eval] overriding {cfg_key} verbose=False; pass --verbose to keep model debug logs")
    model_kwargs["verbose"] = bool(getattr(args, "verbose", False))

    # In DP mode, the top-of-file shim has already pinned this rank to its
    # single GPU via CUDA_VISIBLE_DEVICES (set BEFORE any torch import). So
    # `device_map='auto'` inside Thyme/ThymeInternVL will load the full model
    # onto that one GPU rather than sharding across all 8.

    print(f"[eval][rank{rank}/{world_size}] Loading model from {model_path} "
          f"(class={model_cls.__name__}, cfg_key={cfg_key}, "
          f"max_new_tokens={model_kwargs.get('max_new_tokens', 'class-default(2048)')})")
    load_start = time.time()
    model = model_cls(**model_kwargs)
    # Prove the budget actually reached the generation config (not just model_kwargs).
    _eff = getattr(model, "generate_kwargs", {}) or {}
    print(f"[eval][rank{rank}/{world_size}] Model loaded in {time.time()-load_start:.1f}s "
          f"(effective generate max_new_tokens={_eff.get('max_new_tokens', 'n/a')})")

    # Official Thyme/VLMEvalKit evaluation uses the Thyme model config directly.
    # ATS is available only as an explicit ablation.
    if getattr(args, "use_ats", False):
        from thyme_ats_patch import apply_ats
        model = apply_ats(model, log=getattr(args, "ats_log", False))
        print(f"[eval][rank{rank}/{world_size}] ATS enabled (text temp=1.0, code greedy)")

    ds = build_dataset(benchmark)
    if hasattr(model, "set_dump_image"):
        model.set_dump_image(ds.dump_image)
    n = len(ds)

    if _is_gcep_bench(benchmark):
        # Every main evaluation run must assert the configured sampling
        # config. Missing or mismatched fields stop the run (fail-closed).
        genk = getattr(model, "generate_kwargs", {}) or {}
        got = {k: genk.get(k) for k in ("top_k", "top_p", "temperature")}
        want = {"top_k": 1, "top_p": 0.001, "temperature": 0.01}
        if got != want:
            raise RuntimeError(
                f"[eval] gcep main track requires generation config {want}, "
                f"but resolved {got} (generate_kwargs present: {bool(genk)}); aborting."
            )
        if rank == 0:
            print(f"[eval] gcep generation-config assert PASS: {got}")

    if _is_gcep4_bench(benchmark) and rank == 0:
        # Run manifest and resolved config. Written on first
        # launch of a run; on resume every identity field is re-verified and ANY
        # mismatch aborts (no mixed runs). The datasets/benchmarks sections are
        # append-only and merged across the per-benchmark launches of a run.
        from gcep_bench import run_manifest as gcep_run_manifest

        tsv = Path(os.environ["LMUData"]) / f"{benchmark}.tsv"
        ds_entry = {benchmark: {"tsv_sha256": hashlib.sha256(tsv.read_bytes()).hexdigest(),
                                "n_rows": n}}
        gen_cfg = {"top_k": 1, "top_p": 0.001, "temperature": 0.01,
                   "prompt_recipe": str(getattr(args, "prompt_recipe", "")),
                   "use_ats": bool(getattr(args, "use_ats", False)),
                   "max_pixels": int(getattr(args, "max_pixels", 0) or 0)}
        model_id = str(getattr(args, "tag", "") or Path(model_path).name)
        manifest_path = output_dir / "run_manifest.json"
        current = gcep_run_manifest.build_run_manifest(
            run_id=output_dir.name, output_dir=str(output_dir),
            model_path=str(model_path), model_id=model_id,
            benchmarks=[benchmark], generation_config=gen_cfg,
            dataset_manifest_entries=ds_entry)
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            problems = gcep_run_manifest.verify_resume(
                existing, current,
                ignore_prefixes={"datasets", "benchmarks"})
            if problems:
                raise SystemExit(
                    "[eval] gcep4 resume verification FAILED — this output dir "
                    "belongs to a different configuration; create a new run dir "
                    "instead of mixing runs:\n  - " + "\n  - ".join(problems[:20]))
            existing.setdefault("datasets", {}).update(ds_entry)
            existing["benchmarks"] = sorted(set(existing.get("benchmarks", [])) | {benchmark})
            gcep_run_manifest.write_run_artifacts(existing, str(output_dir))
            print(f"[eval] gcep4 run manifest verified + updated ({benchmark})")
        else:
            gcep_run_manifest.write_run_artifacts(current, str(output_dir))
            print(f"[eval] gcep4 run manifest written ({benchmark})")

    selected_indices, sampling_info = _select_eval_indices(benchmark, n)
    if rank == 0:
        sample_file = output_dir / f"sample_{benchmark}.json"
        sample_file.write_text(json.dumps(sampling_info, indent=2, default=_json_default))

    # Shard selected dataset indices across ranks
    my_indices = selected_indices[rank::world_size]
    print(
        f"[eval][rank{rank}/{world_size}] Running {benchmark} "
        f"({len(my_indices)}/{len(selected_indices)} selected, dataset={n}, "
        f"sampling={sampling_info['method']}, sha1={sampling_info['indices_sha1'][:12]})"
    )

    correct = 0
    total = 0
    n_has_answer = 0
    parser_correct_total = 0
    semantic_judge_used = 0
    semantic_judge_rescued = 0
    semantic_judge_invalid = 0
    last_sample_id = ""
    capture_model_output = bool(getattr(args, "capture_model_output", True)) and not bool(
        getattr(args, "verbose", False)
    )
    prediction_max_chars = int(getattr(args, "prediction_max_chars", 20000) or 0)

    # Per-rank progress prefix to keep parallel logs readable
    log_prefix = f"[eval][rank{rank}]"
    flag_dir = output_dir / "_flags"
    progress_dir = output_dir / "_progress"
    shard_file = output_dir / f"predictions_{benchmark}_rank{rank}.jsonl"
    partial_shard_file = _partial_rank_predictions_path(output_dir, benchmark, rank)
    stats_file = output_dir / f"_stats_{benchmark}_rank{rank}.json"
    trace_shard_file = output_dir / f"tool_traces_{benchmark}_rank{rank}.jsonl"
    partial_trace_file = output_dir / f"tool_traces_{benchmark}_rank{rank}.partial.jsonl"
    flag_dir.mkdir(exist_ok=True)
    progress_dir.mkdir(exist_ok=True)

    gcep4 = _is_gcep4_bench(benchmark)
    gcep4_shard_complete = False
    resume_rows: list[dict] = []
    if gcep4:
        # Sample-level resume: an interrupted benchmark
        # continues from the rows already present in this rank's partial shard;
        # completed sample IDs are never regenerated. A rank whose shard was
        # already finalized skips generation entirely.
        gcep4_shard_complete = shard_file.exists() and stats_file.exists()
        if not gcep4_shard_complete:
            shard_file.unlink(missing_ok=True)
            stats_file.unlink(missing_ok=True)
            trace_shard_file.unlink(missing_ok=True)
            pred_rows, trace_rows = [], []
            if partial_shard_file.exists():
                pred_rows = [json.loads(l) for l in partial_shard_file.read_text().splitlines() if l.strip()]
            if partial_trace_file.exists():
                trace_rows = [json.loads(l) for l in partial_trace_file.read_text().splitlines() if l.strip()]
            if pred_rows or trace_rows:
                # keep only rows present in BOTH partials (crash-desync guard)
                trace_ids = {str(t.get("sample_id")) for t in trace_rows}
                keep = [r for r in pred_rows
                        if str(r.get("sample_id", r.get("id"))) in trace_ids]
                keep_ids = {str(r.get("sample_id", r.get("id"))) for r in keep}
                trace_keep = [t for t in trace_rows if str(t.get("sample_id")) in keep_ids]
                partial_shard_file.write_text("".join(json.dumps(r, default=_json_default) + "\n" for r in keep))
                partial_trace_file.write_text("".join(json.dumps(t, default=_json_default) + "\n" for t in trace_keep))
                resume_rows = keep
                print(f"[eval][rank{rank}] gcep4 resume: {len(resume_rows)} completed "
                      f"rows kept, only missing sample IDs will be generated")
    else:
        shard_file.unlink(missing_ok=True)
        partial_shard_file.unlink(missing_ok=True)
        stats_file.unlink(missing_ok=True)
    _rank_done_flag_path(flag_dir, benchmark, rank).unlink(missing_ok=True)
    if getattr(args, "dump_trajectories", False):
        os.environ["THYME_TRAJECTORY_DUMP_PATH"] = str(
            output_dir / f"trajectories_{benchmark}_rank{rank}.jsonl"
        )

    if gcep4 and gcep4_shard_complete:
        # This rank finalized this benchmark in a previous launch: keep shards,
        # mark done, and let rank 0 proceed straight to merge/aggregation.
        prev_stats = json.loads(stats_file.read_text())
        _rank_done_flag_path(flag_dir, benchmark, rank).touch()
        _write_rank_progress(
            progress_dir, benchmark, rank,
            done=prev_stats["total"], total=prev_stats["total"],
            correct=prev_stats["correct"],
            parser_correct=prev_stats.get("parser_correct", 0),
            has_answer=prev_stats.get("n_has_answer", 0),
            semantic_judge_used=0, semantic_judge_rescued=0,
            semantic_judge_invalid=0, status="done", last_sample_id="")
        print(f"[eval][rank{rank}] gcep4 resume: shard already finalized, skipping generation")
        if rank != 0:
            return None
    else:
        # Seed counters from resumed rows so merged stats reflect the full shard.
        for row in resume_rows:
            total += 1
            correct += int(bool(row.get("correct")))
            parser_correct_total += int(bool(row.get("parser_correct")))
            n_has_answer += int(bool(row.get("has_answer")))
        resume_ids = {str(r.get("sample_id", r.get("id"))) for r in resume_rows}
        _write_rank_progress(
            progress_dir,
            benchmark,
            rank,
            done=total,
            total=len(my_indices),
            correct=correct,
            parser_correct=parser_correct_total,
            has_answer=n_has_answer,
            semantic_judge_used=0,
            semantic_judge_rescued=0,
            semantic_judge_invalid=0,
            status="running",
        )

        trace_out = (open(partial_trace_file, "a", encoding="utf-8")
                     if gcep4 else None)
        with open(partial_shard_file, "a" if gcep4 else "w", encoding="utf-8") as shard_out:
            for local_i, i in enumerate(my_indices):
                sample = ds.data.iloc[i]
                sid = sample["index"]
                if gcep4 and str(sid) in resume_ids:
                    continue  # completed row: never regenerate
                last_sample_id = str(sid)
                gt = str(sample.get("answer", ""))
                msg = _build_eval_message(model, ds, sample, benchmark)
                os.environ["THYME_EPISODE_ID"] = f"eval_{benchmark}_{sid}"

                error = None
                try:
                    pred = _generate_model_prediction(
                        model, msg, benchmark, capture_output=capture_model_output
                    )
                except Exception as e:
                    print(f"{log_prefix} ERROR on {benchmark}/{sid}: {e}")
                    error = repr(e)
                    pred = ""

                tool_use_info = _extract_tool_use_features(model)

                has_answer = pred.strip() != ""
                n_has_answer += int(has_answer)

                parser_correct, score_info = _score_prediction(benchmark, sample, pred)
                if _is_gcep_bench(benchmark):
                    # gcep benches never use the vqa_outcome rescue judge (it sends
                    # the image to the judge, changing the measurement object);
                    # judge-backed gcep scoring happens post-hoc on rank 0.
                    is_correct, semantic_info = parser_correct, {
                        "semantic_judge_used": False,
                        "semantic_judge_rescued": False,
                        "semantic_judge_valid": False,
                        "semantic_judge_score": None,
                        "semantic_judge_error": None,
                        "semantic_judge_raw_text": "",
                        "semantic_judge_attempts": 0,
                        "semantic_judge_reference_answer": "",
                    }
                else:
                    is_correct, semantic_info = _apply_semantic_judge(
                        benchmark,
                        sample,
                        pred,
                        parser_correct,
                        msg,
                        judge_client,
                        judge_temperature=float(getattr(args, "judge_temperature", 0.1)),
                        judge_max_tokens=int(getattr(args, "judge_max_tokens", 256)),
                        judge_attempts=int(getattr(args, "judge_attempts", 3)),
                    )
                parser_correct_total += int(parser_correct)
                correct += int(is_correct)
                total += 1
                semantic_judge_used += int(semantic_info["semantic_judge_used"])
                semantic_judge_rescued += int(semantic_info["semantic_judge_rescued"])
                semantic_judge_invalid += int(
                    semantic_info["semantic_judge_used"] and not semantic_info["semantic_judge_valid"]
                )

                if local_i % 25 == 0 or local_i < 3:
                    print(f"{log_prefix} {local_i}/{len(my_indices)} {benchmark}: "
                          f"acc={correct/max(1,total)*100:.1f}%")

                pred_for_artifact, pred_truncated = _truncate_for_artifact(
                    str(pred), prediction_max_chars
                )
                prediction_row = {
                    "id": sid,
                    "gt": score_info["gt"],
                    "pred": pred_for_artifact,
                    "pred_full_chars": len(str(pred)),
                    "pred_truncated": pred_truncated,
                    "correct": is_correct,
                    "parser_correct": parser_correct,
                    "has_answer": has_answer,
                    "error": error,
                    "extracted": score_info["extracted"],
                    "official_extracted": score_info.get("official_extracted", ""),
                    "extraction_source": score_info.get("extraction_source", ""),
                    "score_method": score_info["score_method"],
                    **semantic_info,
                    **tool_use_info,
                }
                if "gcep_score" in score_info:
                    prediction_row["gcep_score"] = score_info["gcep_score"]
                if _is_gcep_bench(benchmark):
                    # Immutable prediction artifact fields.
                    prediction_row.update({
                        "schema_version": "1",
                        "sample_id": str(sid),
                        "run_id": output_dir.name,
                        "model_id": str(getattr(args, "tag", "") or Path(model_path).name),
                        "checkpoint_path": str(model_path),
                        "benchmark_id": benchmark,
                        "raw_response": pred_for_artifact,
                        "prompt_hash": hashlib.sha256(
                            json.dumps(msg, default=str, ensure_ascii=False).encode("utf-8")
                        ).hexdigest()[:16],
                        "generation_config": requested_recipe.get("generation_assert", {}),
                        "status": "SUCCESS" if error is None else "FAILED",
                    })
                    if _is_gcep4_bench(benchmark):
                        prediction_row["n_images"] = sum(
                            1 for m in msg if isinstance(m, dict) and m.get("type") == "image")
                shard_out.write(json.dumps(prediction_row, default=_json_default) + "\n")
                shard_out.flush()
                if trace_out is not None:
                    # Per-sample per-call tool trace. Base-image
                    # binding is structural (sandbox binds the first input image).
                    from gcep_bench.tool_trace import build_tool_trace
                    img_vals = [m.get("value") for m in msg
                                if isinstance(m, dict) and m.get("type") == "image"]
                    trace = build_tool_trace(model, image_paths=img_vals)
                    trace_row = {
                        "schema_version": "1",
                        "benchmark": benchmark,
                        "sample_id": str(sid),
                        "run_id": output_dir.name,
                        "checkpoint_path": str(model_path),
                        "final_answer": score_info.get("extracted", ""),
                        "prediction_status": "SUCCESS" if error is None else "FAILED",
                        "n_images": len(img_vals),
                        **trace,
                    }
                    trace_out.write(json.dumps(trace_row, default=_json_default) + "\n")
                    trace_out.flush()
                _write_rank_progress(
                    progress_dir,
                    benchmark,
                    rank,
                    done=total,
                    total=len(my_indices),
                    correct=correct,
                    parser_correct=parser_correct_total,
                    has_answer=n_has_answer,
                    semantic_judge_used=semantic_judge_used,
                    semantic_judge_rescued=semantic_judge_rescued,
                    semantic_judge_invalid=semantic_judge_invalid,
                    status="running",
                    last_sample_id=last_sample_id,
                )

        partial_shard_file.replace(shard_file)
        if trace_out is not None:
            trace_out.close()
            partial_trace_file.replace(trace_shard_file)
        stats_file.write_text(json.dumps({
            "rank": rank, "correct": correct, "total": total,
            "n_has_answer": n_has_answer,
            "parser_correct": parser_correct_total,
            "semantic_judge_used": semantic_judge_used,
            "semantic_judge_rescued": semantic_judge_rescued,
            "semantic_judge_invalid": semantic_judge_invalid,
        }, default=_json_default))
        # Benchmark-specific done flag avoids cross-benchmark cleanup races.
        _rank_done_flag_path(flag_dir, benchmark, rank).touch()
        _write_rank_progress(
            progress_dir,
            benchmark,
            rank,
            done=total,
            total=len(my_indices),
            correct=correct,
            parser_correct=parser_correct_total,
            has_answer=n_has_answer,
            semantic_judge_used=semantic_judge_used,
            semantic_judge_rescued=semantic_judge_rescued,
            semantic_judge_invalid=semantic_judge_invalid,
            status="done",
            last_sample_id=last_sample_id,
        )

    if rank != 0:
        return None

    # Rank 0: aggregate
    if is_dp:
        timeout_sec = int(getattr(args, "rank_wait_timeout_sec", 0) or 0) or 7200
        print(f"{log_prefix} Waiting for all {world_size} ranks to finish "
              f"(timeout={timeout_sec}s) ...")
        completion_mode = _wait_for_all_ranks(
            flag_dir,
            benchmark,
            world_size,
            timeout_sec=timeout_sec,
            output_dir=output_dir,
        )
        if completion_mode == "flags":
            print(f"{log_prefix} All ranks done, aggregating ...")
        else:
            print(
                f"{log_prefix} All per-rank shards exist for {benchmark}; "
                "aggregating without done flags ..."
            )

    total_correct = total_n = total_has_answer = 0
    total_parser_correct = 0
    total_semantic_judge_used = 0
    total_semantic_judge_rescued = 0
    total_semantic_judge_invalid = 0
    for r in range(world_size):
        s = json.loads((output_dir / f"_stats_{benchmark}_rank{r}.json").read_text())
        total_correct += s["correct"]
        total_n += s["total"]
        total_has_answer += s["n_has_answer"]
        total_parser_correct += s.get("parser_correct", 0)
        total_semantic_judge_used += s.get("semantic_judge_used", 0)
        total_semantic_judge_rescued += s.get("semantic_judge_rescued", 0)
        total_semantic_judge_invalid += s.get("semantic_judge_invalid", 0)

    accuracy = total_correct / max(1, total_n)
    result = {
        "benchmark": benchmark,
        "total": total_n,
        "correct": total_correct,
        "accuracy": accuracy,
        "traditional_correct": total_parser_correct,
        "traditional_accuracy": total_parser_correct / max(1, total_n),
        "has_answer_rate": total_has_answer / max(1, total_n),
        "semantic_judge_enabled": bool(getattr(args, "semantic_judge", False)),
        "semantic_judge_used": total_semantic_judge_used,
        "semantic_judge_rescued": total_semantic_judge_rescued,
        "semantic_judge_invalid": total_semantic_judge_invalid,
        "world_size": world_size,
        "eval_recipe": requested_recipe,
        "sampling": {
            k: v for k, v in sampling_info.items()
            if k != "indices"
        },
    }
    tool_use_meta = _tool_use_report_meta(model_path, output_dir, benchmark)
    if tool_use_meta:
        result[TOOL_USE_REPORT_META_KEY] = tool_use_meta
    result_file.write_text(json.dumps(result, indent=2, default=_json_default))

    # Concatenate all rank shards into a single predictions JSONL
    pred_file = output_dir / f"predictions_{benchmark}.jsonl"
    all_predictions = []
    with open(pred_file, "w") as fout:
        for r in range(world_size):
            shard = output_dir / f"predictions_{benchmark}_rank{r}.jsonl"
            with open(shard) as fin:
                for line in fin:
                    fout.write(line)
                    if line.strip():
                        all_predictions.append(json.loads(line))

    # Concatenate per-rank tool trace shards into a single JSONL
    if _is_gcep4_bench(benchmark):
        trace_file = output_dir / f"tool_traces_{benchmark}.jsonl"
        with open(trace_file, "w") as fout:
            for r in range(world_size):
                shard = output_dir / f"tool_traces_{benchmark}_rank{r}.jsonl"
                if shard.exists():
                    with open(shard) as fin:
                        for line in fin:
                            fout.write(line)

    # Concatenate per-rank trajectory dumps into a single trajectories JSONL
    if getattr(args, "dump_trajectories", False):
        traj_file = output_dir / f"trajectories_{benchmark}.jsonl"
        with open(traj_file, "w") as fout:
            for r in range(world_size):
                traj_shard = output_dir / f"trajectories_{benchmark}_rank{r}.jsonl"
                if traj_shard.exists():
                    with open(traj_shard) as fin:
                        for line in fin:
                            fout.write(line)

    if _is_mme_realworld(benchmark):
        mme_summary = _summarize_mme_realworld(ds, all_predictions)
        result.update({
            "accuracy": mme_summary["accuracy"],
            "correct": mme_summary["correct"],
            "total": mme_summary["total"],
            "failed_extract": mme_summary["failed_extract"],
            "rating": mme_summary["rating"],
        })
        result_file.write_text(json.dumps(result, indent=2, default=_json_default))
        rating_file = output_dir / f"eval_{benchmark}_rating.json"
        rating_file.write_text(json.dumps(mme_summary["rating"], indent=2, default=_json_default))
        detailed_file = output_dir / f"eval_{benchmark}_detailed.jsonl"
        with open(detailed_file, "w") as f:
            for row in mme_summary["detailed"]:
                f.write(json.dumps(row, default=_json_default) + "\n")

    if _is_gcep_bench(benchmark):
        result = _aggregate_gcep(
            benchmark, ds, all_predictions, output_dir, result, model_path,
            sampling_info=sampling_info,
        )
        result_file.write_text(json.dumps(result, indent=2, default=_json_default))
        accuracy = float(result.get("accuracy", accuracy))

    # Cleanup per-rank intermediate files (keep dir for next benchmark)
    for r in range(world_size):
        (output_dir / f"_stats_{benchmark}_rank{r}.json").unlink(missing_ok=True)
        (output_dir / f"predictions_{benchmark}_rank{r}.jsonl").unlink(missing_ok=True)
        (output_dir / f"trajectories_{benchmark}_rank{r}.jsonl").unlink(missing_ok=True)
        _rank_done_flag_path(flag_dir, benchmark, r).unlink(missing_ok=True)

    print(f"{log_prefix} {benchmark}: accuracy={accuracy*100:.1f}% "
          f"({total_correct}/{total_n}, world_size={world_size})")
    return result


_GCEP_HEADLINE_KEYS = {
    "TreeBench": "answer_accuracy",
    "ZoomBench": "global_view_accuracy",
    "VisualProbe_Easy": "avg_at_1",
    "VisualProbe_Medium": "avg_at_1",
    "VisualProbe_Hard": "avg_at_1",
    "ReasonMapPlus": "acc",
    "ChartQAPro": "overall",
    "InfographicVQA_val": "anls",
    "PerceptionBench": "accuracy",
    "HallusionBench_GCEP": "overall",
    "MathVerseVO": "accuracy",
    "VisuLogic_GCEP": "accuracy",
}


def _aggregate_gcep(benchmark: str, ds, all_predictions: list[dict], output_dir: Path,
                    result: dict, model_path: str, *, sampling_info: dict) -> dict:
    """Authoritative post-hoc scoring for gcep benchmarks.

    Runs the gcep_bench scorer over the merged immutable predictions, writes the
    evaluation artifacts into the run dir, and returns an updated eval result.
    Any UNSCORED sample marks the run INCOMPLETE (fail-closed, never default 0).
    """
    from gcep_bench import manifests as gcep_manifests
    from gcep_bench.common import strip_think_tags
    from gcep_bench.scorers import get_scorer

    scorer = get_scorer(benchmark)
    samples = {str(row["index"]): row.to_dict() for _, row in ds.data.iterrows()}

    judge = None
    if getattr(scorer, "REQUIRES_JUDGE", False):
        from gcep_bench.judge import GcepJudgeConfig, GcepTextJudge

        cfg = GcepJudgeConfig()
        if str(getattr(args, "judge_base_url", "") or ""):
            cfg.base_url = str(args.judge_base_url)
        if str(getattr(args, "judge_model", "") or ""):
            cfg.model = str(args.judge_model)
        if str(getattr(args, "judge_api_key", "") or ""):
            cfg.api_key = str(args.judge_api_key)
        judge = GcepTextJudge(cfg, cache_path=str(output_dir / "judge_cache.jsonl"))
        served = [m.get("id", "") for m in judge.client.healthcheck().get("data", [])]
        print(f"[eval] gcep judge: base_url={judge.client.base_url} model={cfg.model} served={served}")
        if cfg.model not in served:
            raise RuntimeError(f"gcep judge model {cfg.model!r} not served: {served}")

    rows, audit = [], []
    pred_by_sid = {str(p.get("sample_id", p.get("id", ""))): p for p in all_predictions}
    for p in all_predictions:
        sid = str(p.get("sample_id", p.get("id", "")))
        sample = samples.get(sid)
        if sample is None:
            rows.append({"sample_id": sid, "correct": None,
                         "meta": {"status": "UNSCORED", "error": "sample not in dataset"}})
            continue
        if benchmark in GCEP4_BENCHMARKS and str(p.get("status", "SUCCESS")) != "SUCCESS":
            # Prediction-side infra failure (crash/timeout): recovery queue,
            # never sent to the judge, never scored 0.
            rows.append({"sample_id": sid, "correct": None,
                         "meta": {"status": "UNSCORED",
                                  "score_method": "pred_infra_error",
                                  "error": p.get("error")}})
            continue
        correct, meta = scorer.score_sample(
            sample, strip_think_tags(p.get("pred", "") or ""), judge=judge, benchmark=benchmark)
        rows.append({"sample_id": sid, "correct": correct, "meta": meta})
        jr = (meta or {}).get("judge_raw")
        if jr:
            audit.append({"sample_id": sid, **jr})

    # Attach the failure-status taxonomy to every score row and
    # persist the failures queue. Infra/judge failures never convert to 0.
    if benchmark in GCEP4_BENCHMARKS:
        from gcep_bench.failure_classify import classify_failure
        failure_counts: dict[str, int] = {}
        failure_rows = []
        for r in rows:
            p = pred_by_sid.get(r["sample_id"], {})
            fc = classify_failure(
                prediction_status=str(p.get("status", "SUCCESS")),
                prediction_error=p.get("error"),
                score=r["correct"], meta=r["meta"])
            r["failure_class"] = fc
            if fc is not None:
                failure_counts[fc] = failure_counts.get(fc, 0) + 1
                failure_rows.append({"sample_id": r["sample_id"], "failure_class": fc,
                                     "correct": r["correct"],
                                     "score_method": (r["meta"] or {}).get("score_method", ""),
                                     "error": (r["meta"] or {}).get("judge_raw", {}).get("error")
                                              or p.get("error")})
        gcep_manifests.append_jsonl(str(output_dir / f"failures_{benchmark}.jsonl"), failure_rows)
        n_infra_open = sum(failure_counts.get(c, 0) for c in
                           ("PRED_INFRA_ERROR", "TOOL_INFRA_ERROR", "JUDGE_INFRA_ERROR"))
    else:
        failure_counts = {}
        n_infra_open = 0

    metrics = scorer.aggregate(rows)
    metrics["protocol_name"] = scorer.PROTOCOL_NAME
    metrics["scoring_layer"] = scorer.SCORING_LAYER
    n_unscored = sum(1 for r in rows if r["correct"] is None)
    metrics["n_unscored"] = n_unscored
    run_status = "INCOMPLETE" if n_unscored else "COMPLETE"
    metrics["run_status"] = run_status
    if failure_counts:
        metrics["failure_counts"] = failure_counts
        metrics["n_unresolved_infra"] = n_infra_open

    gcep_manifests.append_jsonl(str(output_dir / f"scores_{benchmark}.jsonl"), rows)
    gcep_manifests.write_json(str(output_dir / f"metrics_{benchmark}.json"), metrics)
    if audit:
        gcep_manifests.append_jsonl(str(output_dir / f"judge_audit_{benchmark}.jsonl"), audit)

    sel_positions = sampling_info.get("indices") or []
    if sel_positions:
        selected_ids = [str(ds.data.iloc[i]["index"]) for i in sel_positions]
    else:  # fallback: full dataset
        selected_ids = list(samples.keys())
    coverage = gcep_manifests.check_coverage(
        [{"sample_id": r["sample_id"],
          "status": "SCORED" if r["correct"] is not None else "UNSCORED"} for r in rows],
        selected_ids,
        allow_unscored=True,
    )
    if n_unscored:
        coverage["ok"] = False
    gcep_manifests.write_json(str(output_dir / f"coverage_{benchmark}.json"), coverage)

    from gcep_bench.datasets.builders import PINS as GCEP_PINS

    protocol = gcep_manifests.build_protocol_manifest(
        run_id=output_dir.name,
        model_id=str(getattr(args, "tag", "") or Path(model_path).name),
        checkpoint_path=str(model_path),
        benchmark_id=benchmark,
        dataset_revision=str(GCEP_PINS.get(benchmark, {}).get("revision", "")),
        protocol_name=scorer.PROTOCOL_NAME,
        scoring_layer=scorer.SCORING_LAYER,
        generation_config=result.get("eval_recipe", {}).get("generation_assert", {}),
        prompt_hash="",
        judge=({"base_url": judge.config.base_url, "model": judge.config.model,
                "temperature": 0.0, "enable_thinking": False} if judge else {}),
    )
    gcep_manifests.write_json(str(output_dir / f"protocol_manifest_{benchmark}.json"), protocol)

    headline_key = _GCEP_HEADLINE_KEYS[benchmark]
    headline = metrics.get(headline_key)
    print(f"[eval] gcep {benchmark}: {headline_key}={headline} "
          f"(n={metrics.get('n')}, unscored={n_unscored}, status={run_status})")
    if run_status != "COMPLETE":
        print(f"[eval] WARNING: {benchmark} has {n_unscored} UNSCORED samples; "
              "run marked INCOMPLETE (fail-closed, no default-0).")

    result.update({
        "accuracy": headline if isinstance(headline, (int, float)) else result.get("accuracy"),
        "gcep_headline_key": headline_key,
        "gcep_metrics": metrics,
        "run_status": run_status,
        "coverage_ok": coverage["ok"],
    })
    return result


def compare_sft_ckpt_vs_gt_only_opd(
    sft_ckpt_dir: Path,
    gt_only_opd_dir: Path,
    benchmarks: list[str],
):
    """Print a SFT_CKPT vs gt_only_OPD comparison table."""
    print("\n" + "=" * 60)
    print(" Checkpoint comparison: SFT_CKPT vs gt_only_OPD")
    print("=" * 60)
    print(f"{'Benchmark':<22} {'SFT_CKPT':>12} {'gt_only_OPD':>12} {'Delta':>10}")
    print("-" * 60)

    for bench in benchmarks:
        sft_ckpt_file = sft_ckpt_dir / f"eval_{bench}.json"
        gt_only_opd_file = gt_only_opd_dir / f"eval_{bench}.json"

        sft_ckpt_acc = None
        gt_only_opd_acc = None

        if sft_ckpt_file.exists():
            r = json.loads(sft_ckpt_file.read_text())
            sft_ckpt_acc = r.get("accuracy")
        if gt_only_opd_file.exists():
            r = json.loads(gt_only_opd_file.read_text())
            gt_only_opd_acc = r.get("accuracy")

        sft_ckpt_str = f"{sft_ckpt_acc*100:.1f}%" if sft_ckpt_acc is not None else "N/A"
        gt_only_opd_str = (
            f"{gt_only_opd_acc*100:.1f}%" if gt_only_opd_acc is not None else "N/A"
        )

        if sft_ckpt_acc is not None and gt_only_opd_acc is not None:
            delta = gt_only_opd_acc - sft_ckpt_acc
            delta_str = f"{'+' if delta >= 0 else ''}{delta*100:.1f}pp"
        else:
            delta_str = "N/A"

        print(f"{bench:<22} {sft_ckpt_str:>12} {gt_only_opd_str:>12} {delta_str:>10}")

    print("=" * 60)

    # A comparison table does not certify end-to-end reproducibility.


def main(args):
    rank, _ = _dp_env()

    if args.compare:
        if rank == 0:
            compare_sft_ckpt_vs_gt_only_opd(
                Path(args.sft_ckpt_dir),
                Path(args.gt_only_opd_dir),
                args.benchmarks or BENCHMARK_LIST,
            )
        return

    if not args.model_path:
        sys.exit("--model-path required for evaluation")
    if not args.output_dir:
        sys.exit("--output-dir required for evaluation")

    model_path = args.model_path
    output_dir = Path(args.output_dir)
    benchmarks = args.benchmarks or BENCHMARK_LIST

    for bench in benchmarks:
        run_vlmeval(model_path, bench, output_dir, verbose=args.verbose)

    if rank == 0:
        _refresh_overall_tool_use_report(output_dir, benchmarks)
        print(f"\n[eval] All done. Results in {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="",
                   help="Model checkpoint path (not needed with --compare)")
    p.add_argument("--output-dir", default="",
                   help="Directory for eval results")
    p.add_argument("--benchmarks", nargs="+", default=None,
                   help=("Benchmarks to evaluate (default: "
                         f"{' '.join(BENCHMARK_LIST)})"))
    p.add_argument("--tag", default="",
                   help="Label for this eval run (for logging)")
    p.add_argument("--compare", action="store_true",
                   help="Just print SFT_CKPT vs gt_only_OPD comparison table (no new inference)")
    p.add_argument("--sft-ckpt-dir", default="",
                   help="SFT_CKPT eval directory (for --compare)")
    p.add_argument("--gt-only-opd-dir", default="",
                   help="gt_only_OPD eval directory (for --compare)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-capture-model-output", dest="capture_model_output",
                   action="store_false", default=True,
                   help=("Do not suppress stdout/stderr emitted inside Thyme.generate. "
                         "Default captures it to keep long eval logs quota-safe."))
    p.add_argument("--prediction-max-chars", type=int, default=20000,
                   help=("Maximum stored characters for each prediction artifact. "
                         "Scoring still uses the full prediction before truncation; "
                         "0 disables truncation."))
    p.add_argument("--max-examples", type=int, default=0,
                   help="Evaluate only this many selected examples per benchmark "
                        "(0 means all). Useful for smoke diagnostics.")
    p.add_argument("--start-index", type=int, default=0,
                   help="Start from this zero-based dataset row before optional "
                        "--max-examples slicing.")
    p.add_argument("--sample-fraction", type=float, default=0.0,
                   help=("Randomly evaluate this fraction of the selected dataset "
                         "after start/max slicing. 0 disables random sampling; "
                         "0.1 means a fixed random 10%% validation subset."))
    p.add_argument("--sample-seed", type=int, default=20260602,
                   help="Seed for --sample-fraction, included in eval_recipe metadata.")
    p.add_argument("--sample-indices-file", default="",
                   help=("Optional JSON list/dict of zero-based dataset row indices. "
                         "Overrides --sample-fraction when set."))
    p.add_argument("--prompt-recipe", choices=[OFFICIAL_THYME_PROMPT_RECIPE, DATASET_PROMPT_RECIPE],
                   default=OFFICIAL_THYME_PROMPT_RECIPE,
                   help="Prompt construction recipe. Default matches official Thyme/VLMEvalKit inference.")
    p.add_argument("--use-ats", dest="use_ats", action="store_true", default=False,
                   help="Unavailable in this release: the optional ATS implementation is not bundled.")
    p.add_argument("--no-ats", dest="use_ats", action="store_false",
                   help="Disable ATS; this is the default official-aligned setting.")
    p.add_argument("--ats-log", action="store_true", default=False,
                   help="Log [ATS] segment N -> ... on every generate call.")
    p.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS,
                   help="Per-image max_pixels override for Thyme/Qwen-VL eval tokenization. "
                        "Default 0 leaves the official Thyme config unchanged.")
    p.add_argument("--max-new-tokens", type=int, default=0,
                   help="Generation budget override for the subject model. "
                        "Default 0 keeps the model class default (2048); raise it for "
                        "thinking models so the CoT does not truncate the final answer.")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Overwrite existing eval outputs when their eval_recipe metadata differs.")
    p.add_argument("--dump-trajectories", action="store_true", default=False,
                   help="Dump full conversation_history (code blocks + sandbox_output) per sample "
                        "to trajectories_{benchmark}.jsonl via THYME_TRAJECTORY_DUMP_PATH. "
                        "Per-rank files are concatenated by rank 0 after aggregation.")
    p.add_argument("--rank-wait-timeout-sec", type=int, default=0,
                   help="Rank-0 wait for all DP ranks before aggregation (default: 7200). "
                        "Use a larger value for slow benchmarks like HRBench4K.")
    p.add_argument("--semantic-judge", action="store_true", default=False,
                   help="Use the configured remote judge to rescue parser-failed answers.")
    p.add_argument("--judge-base-url", default=os.environ.get("REMOTE_VLM_BASE_URL", ""),
                   help="Remote judge base URL, with or without /v1.")
    p.add_argument("--judge-api-key", default=os.environ.get("REMOTE_VLM_API_KEY", "EMPTY"),
                   help="Remote judge API key.")
    p.add_argument("--judge-model", default=os.environ.get("REMOTE_VLM_MODEL", os.environ.get("JUDGE_MODEL", "")),
                   help="Served model name for the remote semantic judge.")
    p.add_argument("--judge-temperature", type=float, default=0.1,
                   help="Sampling temperature for the semantic judge (aligns with Thyme agent_rm).")
    p.add_argument("--judge-max-tokens", type=int, default=256,
                   help="Max completion tokens for the semantic judge.")
    p.add_argument("--judge-attempts", type=int, default=3,
                   help="Logical retry count for invalid semantic-judge outputs.")
    p.add_argument("--judge-timeout-sec", type=float, default=120.0,
                   help="HTTP timeout for remote semantic-judge calls.")
    p.add_argument("--judge-http-retries", type=int, default=2,
                   help="HTTP retry count for remote semantic-judge calls.")
    p.add_argument("--judge-retry-backoff-sec", type=float, default=2.0,
                   help="HTTP retry backoff for remote semantic-judge calls.")
    p.add_argument("--tool-use-checkpoint-label", default="",
                   help=("Display label used by overall_tool_use.md. "
                         "Defaults to a friendly label for known tags, else --tag or model dir name."))
    p.add_argument("--tool-use-log-path", default="",
                   help=("Combined eval log path used for tool-use parsing. "
                         "Default: <output-dir>/run.log"))
    p.add_argument("--refresh-tool-use-report", dest="refresh_tool_use_report",
                   action="store_true", default=False,
                   help="Optional: requires the separately supplied build_overall_tool_use_report.py helper.")
    p.add_argument("--no-refresh-tool-use-report", dest="refresh_tool_use_report",
                   action="store_false",
                   help="Disable automatic overall_tool_use.md refresh.")
    p.add_argument("--model-class", dest="model_class",
                   choices=["auto", "thyme", "thyme_internvl", "thyme_qwen3vl", "thyme_qwen3vl_moe"],
                   default="auto",
                   help="Model class to instantiate. 'auto' reads model config.json "
                        "architectures: InternVLChatModel -> ThymeInternVL, Qwen3VLMoe* -> "
                        "ThymeQwen3VLMoe (Qwen3-VL MoE), Qwen3VL* -> "
                        "ThymeQwen3VL (Metis), else Thyme (Qwen). "
                        "Qwen path is the default and unchanged.")
    args = p.parse_args()
    if args.use_ats:
        p.error("--use-ats is unavailable: thyme_ats_patch.py is not included in this release.")
    main(args)
