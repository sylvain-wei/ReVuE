# Modified for anonymous review: release paths, configuration, and documentation.
"""Run provenance and resolved evaluation configuration.

Per eval run (one output_dir) we write:
- ``run_manifest.json``     — identity & provenance of the run;
- ``resolved_config.json``  — the fully resolved configuration that must stay
  identical across the six checkpoints except for allowlisted fields.

Resume rule: when ``run_manifest.json`` already exists in the output
dir, the current environment is verified against it; ANY mismatch on a
non-volatile field aborts the launch — create a new run dir instead of mixing.

Cross-checkpoint rule: ``diff_resolved_configs`` compares two resolved
configs and returns non-allowlisted differences; the queue controller refuses to
advance when any are found.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import manifests

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "thyme-infer" / "scripts"

# Fields allowed to differ between checkpoints. Everything else must match.
CROSS_CKPT_ALLOWLIST = {
    "checkpoint.name",
    "checkpoint.path",
    "checkpoint.signature",
    "model_id",
    "output_dir",
    "run_id",
}

# Manifest keys that may legitimately change between launches of the SAME run
# (never compared during resume verification).
VOLATILE_KEYS = {"created_at", "benchmarks_completed", "provenance_amendments"}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_signature(ckpt_dir: str) -> dict:
    """Signature of a checkpoint dir: config+index+tokenizer shas + shard sizes.

    Full-weight hashing (6 x 16.6GB) is intentionally not part of the per-launch
    signature; shard byte sizes + index sha catch path/mix-up swaps.
    """
    d = Path(ckpt_dir)
    sig = {"config_sha256": _sha_file(d / "config.json"),
           "safetensors_index_sha256": _sha_file(d / "model.safetensors.index.json"),
           "tokenizer_sha256": _sha_file(d / "tokenizer.json")}
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    shards = sorted(set(idx.get("weight_map", {}).values()))
    sig["shard_sizes"] = {s: (d / s).stat().st_size for s in shards}
    sig["total_bytes"] = sum(sig["shard_sizes"].values())
    return sig


def git_state() -> dict:
    """Fingerprint released source files without requiring or inspecting Git."""
    scopes = ["thyme-infer/scripts", "thyme-infer/Thyme", "thyme-infer/configs", "src"]
    source_hashes = {}
    for scope in scopes:
        for path in sorted((REPO_ROOT / scope).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".json", ".txt"}:
                if not {"__pycache__", "temp_processed_images", "sandbox_scratch", "outputs", "checkpoints", ".git"}.intersection(path.parts) and not path.is_symlink():
                    source_hashes[str(path.relative_to(REPO_ROOT))] = _sha_file(path)
    encoded = json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    return {"kind": "source_archive", "scopes": scopes,
            "file_count": len(source_hashes),
            "source_sha256": hashlib.sha256(encoded).hexdigest()}


def code_state() -> dict:
    files = {"phase3_eval_opd.py": SCRIPTS_DIR / "phase3_eval_opd.py"}
    state = {}
    for name, p in files.items():
        state[name] = _sha_file(p)
    gb = SCRIPTS_DIR / "gcep_bench"
    pkg = {}
    for p in sorted(gb.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".txt"):
            pkg[str(p.relative_to(SCRIPTS_DIR))] = _sha_file(p)
    state["gcep_bench"] = pkg
    return state


def judge_config_state() -> dict:
    """Judge endpoint configuration. NEVER stores the API key — env var name only."""
    return {
        "base_url": os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"),
        "model": os.environ.get("JUDGE_MODEL", "qwen3.5-397b-judge"),
        "api_key_env": "JUDGE_API_KEY",
        "api_key_present": bool(os.environ.get("JUDGE_API_KEY")),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "enable_thinking": False,
        "timeout_sec": 120.0,
        "max_transport_retries": 5,
        "max_parse_retries": 2,
    }


def build_run_manifest(*, run_id: str, output_dir: str, model_path: str, model_id: str,
                       benchmarks: list, generation_config: dict,
                       dataset_manifest_entries: dict) -> dict:
    return {
        "schema_version": "1",
        "created_at": manifests.utc_now_iso(),
        "run_id": run_id,
        "output_dir": output_dir,
        "model_id": model_id,
        "checkpoint": {
            "name": model_id,
            "path": model_path,
            "signature": checkpoint_signature(model_path),
        },
        "benchmarks": list(benchmarks),
        "benchmarks_completed": [],
        "generation_config": generation_config,
        "judge": judge_config_state(),
        "git": git_state(),
        "code": code_state(),
        "datasets": dataset_manifest_entries,  # {bench: {tsv_sha256, revision, n_rows}}
    }


def build_resolved_config(manifest: dict) -> dict:
    """Flatten the parts of the manifest that must be uniform across ckpts."""
    return {
        "schema_version": manifest["schema_version"],
        "run_id": manifest["run_id"],
        "output_dir": manifest["output_dir"],
        "model_id": manifest["model_id"],
        "checkpoint": manifest["checkpoint"],
        "benchmarks": manifest["benchmarks"],
        "generation_config": manifest["generation_config"],
        "judge": manifest["judge"],
        "git": manifest["git"],
        "code": manifest["code"],
        "datasets": manifest["datasets"],
    }


def write_run_artifacts(manifest: dict, output_dir: str) -> None:
    out = Path(output_dir)
    manifests.write_json(str(out / "run_manifest.json"), manifest)
    manifests.write_json(str(out / "resolved_config.json"), build_resolved_config(manifest))


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key + "."))
        else:
            flat[key] = v
    return flat


def verify_resume(existing: dict, current: dict,
                  ignore_prefixes: set[str] | None = None) -> list[str]:
    """Compare an on-disk run manifest with the freshly built one.

    Returns a list of human-readable mismatches (empty == safe to resume).
    Volatile keys and ``ignore_prefixes`` top-level sections (e.g. datasets,
    benchmarks — append-only sections merged by the caller) are ignored.
    """
    ignore = set(ignore_prefixes or set()) | {"benchmarks_completed"}
    a, b = _flatten(existing), _flatten(current)
    problems = []
    for key in sorted(set(a) | set(b)):
        if key.split(".")[0] in VOLATILE_KEYS or key.split(".")[0] in ignore:
            continue
        va, vb = a.get(key, "<absent>"), b.get(key, "<absent>")
        if va != vb:
            problems.append(f"{key}: on-disk={str(va)[:80]} vs current={str(vb)[:80]}")
    return problems


def diff_resolved_configs(a: dict, b: dict,
                          allowlist: set[str] | None = None) -> list[str]:
    """Non-allowlisted differences between two resolved configs (cross-ckpt gate)."""
    allow = allowlist or CROSS_CKPT_ALLOWLIST
    fa, fb = _flatten(a), _flatten(b)
    problems = []
    for key in sorted(set(fa) | set(fb)):
        top_allowed = any(key == pat or key.startswith(pat + ".") for pat in allow)
        if top_allowed:
            continue
        if key.split(".")[0] in VOLATILE_KEYS:
            continue
        va, vb = fa.get(key, "<absent>"), fb.get(key, "<absent>")
        if va != vb:
            problems.append(f"{key}: {str(va)[:60]} != {str(vb)[:60]}")
    return problems
