#!/usr/bin/env python3
"""Dependency-free source/package checks; does not run model training or inference."""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCOPES = ("src", "tools", "thyme-infer")
IGNORED = {
    ".git", "__pycache__", ".venv", "venv", "outputs", "checkpoints",
    "temp_processed_images", "sandbox_scratch", "node_modules",
}
ENTRYPOINTS = (
    "thyme-infer/activate_thyme.sh",
    "thyme-infer/scripts/launch_gcep.sh",
    "thyme-infer/scripts/launch_gcep_internvl.sh",
    "thyme-infer/scripts/run_11bench_eval_suite.sh",
    "thyme-infer/scripts/phase3_eval_opd.py",
    "thyme-infer/scripts/internvl35_sft_stage1.sh",
    "thyme-infer/scripts/internvl35_sft_stage2.sh",
    "thyme-infer/scripts/run_baseline_scriptgeom_full_monitored.sh",
    "thyme-infer/scripts/run_baseline_scriptgeom_full_monitored_internvl.sh",
    "thyme-infer/scripts/launch_judge_qwen35_397b_vllm.sh",
    "thyme-infer/scripts/judge_multiimage_json_smoke.py",
    "tools/prepare_thyme_internvl_data.py",
    "tools/audit_thyme_sft_labels.py",
    "tools/verify_eval_data.py",
    "tools/probe_judge.py",
    "docs/RUNNING.md",
    "docs/ENVIRONMENT.md",
    "docs/DATA.md",
    "docs/EVAL_DATA.md",
    "docs/EXPERIMENTS.md",
    "docs/JUDGE_SERVICE.md",
    "requirements.txt",
    "THIRD_PARTY_NOTICES.md",
    "thyme-infer/Thyme/LICENSE",
    "thyme-infer/Thyme/eval/VLMEvalKit/LICENSE",
)


def paths(scopes: tuple[str, ...], suffix: str) -> list[Path]:
    return sorted(
        p for scope in scopes for p in (ROOT / scope).rglob(f"*{suffix}")
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & IGNORED)
        and p.relative_to(ROOT).parts[:2] != ("thyme-infer", "data")
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_sources() -> dict[str, int]:
    python_files = paths(SOURCE_SCOPES, ".py")
    shell_files = paths(SOURCE_SCOPES, ".sh")
    for path in python_files:
        # Parse against the supported Python 3.10 grammar on every CI runtime.
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
    for path in shell_files:
        subprocess.run(["bash", "-n", str(path)], check=True)
    for entry in ENTRYPOINTS:
        require((ROOT / entry).is_file(), f"Missing documented entry: {entry}")
    return {"python_files": len(python_files), "shell_files": len(shell_files), "entry_paths": len(ENTRYPOINTS)}


def check_metadata() -> dict[str, int]:
    scopes = ("data_prep", "licenses", "thyme-infer/configs", "thyme-infer/Thyme", "website")
    json_files = paths(scopes, ".json")
    jsonl_files = paths(scopes, ".jsonl")
    records = 0
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    for path in jsonl_files:
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{index}: {exc}") from exc
            records += 1
    with (ROOT / "data_prep/tsv_hash.tsv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames == ["file", "rows", "bytes", "sha256"], "Invalid reference TSV header")
        rows = list(reader)
    require(bool(rows), "Empty reference TSV")
    require(len({row["file"] for row in rows}) == len(rows), "Duplicate reference TSV entries")
    for row in rows:
        require(int(row["rows"]) > 0 and int(row["bytes"]) > 0, "Invalid reference TSV size")
        require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None, "Invalid reference TSV hash")
    manifests = []
    for name in ("gcep4_dataset_manifest.jsonl", "gcep6_dataset_manifest.jsonl"):
        manifests.extend(json.loads(line) for line in (ROOT / "data_prep" / name).read_text().splitlines() if line.strip())
    by_file = {Path(entry["tsv_path"]).name: entry for entry in manifests}
    for row in rows:
        entry = by_file[row["file"]]
        require(entry["tsv_sha256"] == row["sha256"], f"Conflicting reference hash: {row['file']}")
        require(int(entry["n_rows"]) == int(row["rows"]), f"Conflicting reference row count: {row['file']}")
    return {"json_files": len(json_files), "jsonl_files": len(jsonl_files), "jsonl_records": records, "reference_tsv_rows": len(rows)}


def check_licenses() -> dict[str, int]:
    sources = json.loads((ROOT / "licenses/SOURCES.json").read_text())
    checked = set()
    for component in sources["components"]:
        for item in component.get("license_files", []):
            path = ROOT / item["local_path"]
            require(path.is_file(), f"Missing license: {item['local_path']}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            require(digest == item["sha256"], f"Changed third-party license text: {item['local_path']}")
            checked.add(item["local_path"])
    require(bool(checked), "No third-party licenses checked")
    return {"license_components": len(sources["components"]), "hashed_license_files": len(checked)}


def check_source_provenance() -> dict[str, int]:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "thyme-infer/scripts"))
    module = importlib.import_module("gcep_bench.run_manifest")
    require(module.REPO_ROOT == ROOT, "Evaluation provenance resolved the wrong repository root")
    state = module.git_state()
    require(state["file_count"] > 0, "Empty runtime source fingerprint")
    require(re.fullmatch(r"[0-9a-f]{64}", state["source_sha256"]) is not None, "Invalid source fingerprint")
    require(bool(module.code_state()["gcep_bench"]), "Empty scorer provenance")
    return {"runtime_fingerprint_files": state["file_count"]}


def check_documentation_links() -> dict[str, int]:
    checked = 0
    files = list((ROOT / "docs").rglob("*.md")) + [
        ROOT / "THIRD_PARTY_NOTICES.md", ROOT / "VALIDATION.md", ROOT / "licenses/README.md",
    ]
    for path in files:
        for target in re.findall(r"\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")):
            parsed = urlsplit(target.strip("<>"))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            local = (path.parent / unquote(parsed.path)).resolve()
            require(local.is_relative_to(ROOT), f"Documentation link leaves the repository: {path}: {target}")
            require(local.exists(), f"Missing documentation target: {path}: {target}")
            checked += 1
    return {"local_documentation_links": checked}


def main() -> int:
    if sys.version_info < (3, 10):
        print("Python 3.10 or newer is required for the repository checks.", file=sys.stderr)
        return 2
    counts = {}
    for check in (check_sources, check_metadata, check_licenses, check_source_provenance, check_documentation_links):
        counts.update(check())
    print(json.dumps({"status": "passed", "checks": counts}, indent=2))
    print("Static checks only: no GPU, model inference, live judge, or numerical reproduction was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
