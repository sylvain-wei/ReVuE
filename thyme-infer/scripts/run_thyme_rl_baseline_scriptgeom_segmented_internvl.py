#!/usr/bin/env python3
# Modified for anonymous review: release paths, configuration, and documentation.
"""InternVL3.5 variant of the script-aligned segmented Thyme RL baseline runner.

Differences vs run_thyme_rl_baseline_scriptgeom_segmented.py (vqa_norm=0 baseline):
  1. SFT_CKPT is supplied through INTERNVL_SFT_CKPT
  2. build_train_cmd passes `--model_type internvl3_5` right after `--model`
     (swift registers the type: swift/llm/model/constant.py + model/internlm.py)
  3. `--max_pixels` removed (Qwen-specific); InternVL uses input_size=448 /
     max_num=12 via the swift internvl2_5 template.

Everything else (GRPO geometry, reward funcs, stop words, eval recipe) is kept
byte-identical to the vqa_norm=0 baseline.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
import re
import subprocess
import sys
from pathlib import Path

THYME_INFER_ROOT = Path(__file__).resolve().parents[1]
THYME_REPO_ROOT = THYME_INFER_ROOT / "Thyme"
SCRIPTS = THYME_INFER_ROOT / "scripts"
PHASE3_EVAL = SCRIPTS / "phase3_eval_opd.py"
INDICES_FILE = (
    THYME_INFER_ROOT
    / "configs"
    / "phase3_periodic_validation_vstar_hrbench4k_indices.json"
)
SFT_CKPT = Path(os.environ.get(
    "INTERNVL_SFT_CKPT",
    str(THYME_INFER_ROOT / "checkpoints" / "InternVL3_5-4B-Thyme-SFT-ckpt288"),
))
RL_DATA_DIR = Path(os.environ.get("RL_DATA_DIR", str(THYME_INFER_ROOT / "data" / "thyme_rl_train" / "data")))

BENCHMARKS = ["VStarBench", "HRBench4K"]
CKPT_RE = re.compile(r"checkpoint-(\d+)$")


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


NUM_GPUS = env_int("SCRIPTGEOM_NUM_GPUS", 8)
GRAD_ACCUM = env_int("SCRIPTGEOM_GRAD_ACCUM", 20)
NUM_GENERATIONS = env_int("SCRIPTGEOM_NUM_GENERATIONS", 4)
BETA = env_float("SCRIPTGEOM_BETA", 0.01)
LR = env_float("SCRIPTGEOM_LR", 5e-7)
REP_PENALTY = env_float("SCRIPTGEOM_REP_PENALTY", 1.05)
TEMPERATURE = env_float("SCRIPTGEOM_TEMPERATURE", 1.0)
# InternVL does not use --max_pixels (Qwen-specific); kept only for config-print
# compatibility with the monitored runner.
MAX_PIXELS = env_int("SCRIPTGEOM_MAX_PIXELS", 0)
EVAL_MAX_PIXELS = env_int("SCRIPTGEOM_EVAL_MAX_PIXELS", 0)
VLLM_GPU_UTIL = env_float("SCRIPTGEOM_VLLM_GPU_UTIL", 0.40)
MAX_LEN = env_int("SCRIPTGEOM_MAX_LEN", 10240)
MAX_COMPLETION = env_int("SCRIPTGEOM_MAX_COMPLETION", 3196)
SAVE_STEPS = env_int("SCRIPTGEOM_SAVE_STEPS", 50)
NUM_TRAIN_EPOCHS = env_int("SCRIPTGEOM_NUM_TRAIN_EPOCHS", 2)
PREDICTION_MAX_CHARS = env_int("SCRIPTGEOM_EVAL_PREDICTION_MAX_CHARS", 4000)


def latest_checkpoint(out_dir: Path) -> tuple[int, Path] | None:
    best: tuple[int, Path] | None = None
    for d in out_dir.rglob("checkpoint-*"):
        if not d.is_dir():
            continue
        m = CKPT_RE.search(d.name)
        if not m or not (d / "model.safetensors.index.json").exists():
            continue
        step = int(m.group(1))
        if best is None or step > best[0]:
            best = (step, d)
    return best


def build_train_cmd(out_dir: Path, max_steps: int, resume_ckpt: Path | None) -> list[str]:
    cmd = [
        "deepspeed",
        f"--num_gpus={NUM_GPUS}",
        "scripts/rlhf_ds.py",
        "--rlhf_type",
        "grpo",
        "--model",
        str(SFT_CKPT),
        "--model_type",
        "internvl3_5",
        "--external_plugins",
        "./examples/train/grpo/plugin/agent_rm.py",
        "--reward_funcs",
        "fmt_orm",
        "vqa_orm",
        "cst_orm",
        "--use_vllm",
        "true",
        "--vllm_mode",
        "colocate",
        "--vllm_max_model_len",
        str(MAX_LEN),
        "--vllm_tensor_parallel_size",
        "1",
        "--vllm_limit_mm_per_prompt",
        '{"image": 6, "video": 0}',
        "--vllm_device",
        "auto",
        "--vllm_gpu_memory_utilization",
        str(VLLM_GPU_UTIL),
        "--train_type",
        "full",
        "--torch_dtype",
        "bfloat16",
        "--dataset",
        str(RL_DATA_DIR),
        "--dataset_shuffle",
        "true",
        "--train_dataloader_shuffle",
        "true",
        "--max_length",
        str(MAX_LEN),
        "--max_completion_length",
        str(MAX_COMPLETION),
        "--freeze_aligner",
        "false",
        "--stop_words",
        r"\<\|im_end\|\>",
        r"\</code\>",
        r"\</answer\>",
        r"\<code\>",
        "--num_train_epochs",
        str(NUM_TRAIN_EPOCHS),
        "--max_steps",
        str(max_steps),
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "1",
        "--padding_side",
        "left",
        "--learning_rate",
        str(LR),
        "--lr_scheduler_type",
        "cosine_with_min_lr",
        "--lr_scheduler_kwargs",
        '{"min_lr_rate": 0.1, "num_cycles": 0.5}',
        "--gradient_accumulation_steps",
        str(GRAD_ACCUM),
        "--save_strategy",
        "steps",
        "--eval_strategy",
        "no",
        "--split_dataset_ratio",
        "0",
        "--eval_steps",
        "20000",
        "--save_steps",
        str(SAVE_STEPS),
        "--save_total_limit",
        "100000",
        "--logging_steps",
        "1",
        "--disable_tqdm",
        "true",
        "--output_dir",
        str(out_dir),
        "--warmup_ratio",
        "0.03",
        "--dataloader_num_workers",
        "8",
        "--num_generations",
        str(NUM_GENERATIONS),
        "--temperature",
        str(TEMPERATURE),
        "--beta",
        str(BETA),
        "--top_p",
        "0.9",
        "--top_k",
        "50",
        "--repetition_penalty",
        str(REP_PENALTY),
        "--deepspeed",
        "zero3",
        "--O3",
        "true",
        "--log_completions",
        "true",
        "--report_to",
        "tensorboard",
        "--async_generate",
        "false",
        "--num_iterations",
        "1",
        "--overlong_filter",
        "true",
        "--offload_optimizer",
        "true",
        "--offload_model",
        "true",
        "--gc_collect_after_offload",
        "true",
        "--attn_impl",
        "flash_attn",
        "--vllm_enforce_eager",
        "true",
    ]
    if resume_ckpt is not None:
        cmd += ["--resume_from_checkpoint", str(resume_ckpt)]
    return cmd


def build_eval_cmd(ckpt: Path, eval_dir: Path, step: int) -> list[str]:
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={NUM_GPUS}",
        str(PHASE3_EVAL),
        "--model-path",
        str(ckpt),
        "--output-dir",
        str(eval_dir),
        "--benchmarks",
        *BENCHMARKS,
        "--sample-indices-file",
        str(INDICES_FILE),
        "--prompt-recipe",
        "official_thyme_vlmevalkit",
        "--semantic-judge",
        "--tag",
        f"scriptgeom_step{step}",
        "--rank-wait-timeout-sec",
        "43200",
        "--prediction-max-chars",
        str(PREDICTION_MAX_CHARS),
    ]
    if EVAL_MAX_PIXELS > 0:
        cmd += ["--max-pixels", str(EVAL_MAX_PIXELS)]
    return cmd


def read_accuracy(eval_dir: Path, benchmark: str) -> float | None:
    f = eval_dir / f"eval_{benchmark}.json"
    if not f.exists():
        return None
    try:
        return float(json.loads(f.read_text()).get("accuracy"))
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=50)
    p.add_argument("--wandb-run-id", default="internvl35_thyme_grpo_smoke")
    p.add_argument("--wandb-project", default="anonymous-reproduction")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = (not args.no_wandb
                 and os.environ.get("WANDB_DISABLED", "true").lower() in {"false", "0", "no"}
                 and os.environ.get("WANDB_MODE", "disabled") != "disabled")
    if not use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
    eval_run_id = f"{args.wandb_run_id}_eval"

    def log_eval_accuracy(step: int, accs: dict[str, float | None]) -> None:
        if not use_wandb:
            return
        import wandb

        env_run_id = os.environ.pop("WANDB_RUN_ID", None)
        try:
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=eval_run_id,
                name=eval_run_id,
                group=args.wandb_run_id,
                resume="allow",
            )
            wandb.define_metric("eval/step")
            wandb.define_metric("eval/*", step_metric="eval/step")
            log_dict = {"eval/step": step}
            for b, acc in accs.items():
                if acc is not None:
                    log_dict[f"eval/{b}_acc"] = acc
            run.log(log_dict)
            run.finish()
        except Exception as exc:
            print(f"[scriptgeom] WARNING: wandb eval log failed at step {step}: {exc}", flush=True)
        finally:
            if env_run_id is not None:
                os.environ["WANDB_RUN_ID"] = env_run_id

    print("[scriptgeom] resolved config:", flush=True)
    for key, value in [
        ("NUM_GPUS", NUM_GPUS),
        ("GRAD_ACCUM", GRAD_ACCUM),
        ("NUM_GENERATIONS", NUM_GENERATIONS),
        ("BETA", BETA),
        ("MAX_PIXELS", MAX_PIXELS),
        ("EVAL_MAX_PIXELS", EVAL_MAX_PIXELS),
        ("VLLM_GPU_UTIL", VLLM_GPU_UTIL),
        ("SAVE_STEPS", SAVE_STEPS),
        ("NUM_TRAIN_EPOCHS", NUM_TRAIN_EPOCHS),
    ]:
        print(f"[scriptgeom]   {key}={value}", flush=True)
    print(f"[scriptgeom] out_dir={out_dir}", flush=True)

    start_done = latest_checkpoint(out_dir)
    done_step = start_done[0] if start_done else 0

    target = args.eval_every
    while target <= args.total_steps + args.eval_every - 1:
        seg_target = min(target, args.total_steps)
        if seg_target <= done_step:
            target += args.eval_every
            continue

        prev = latest_checkpoint(out_dir)
        resume_ckpt = prev[1] if prev else None
        print(
            f"\n[scriptgeom] === train to step {seg_target} "
            f"(resume from {resume_ckpt}) ===",
            flush=True,
        )
        env = os.environ.copy()
        env["WANDB_RUN_ID"] = f"{args.wandb_run_id}_seg{seg_target}"
        env["WANDB_NAME"] = f"{args.wandb_run_id}_seg{seg_target}"
        env["WANDB_RUN_GROUP"] = args.wandb_run_id
        env["WANDB_RESUME"] = "allow"

        rc = subprocess.run(
            build_train_cmd(out_dir, seg_target, resume_ckpt),
            cwd=str(THYME_REPO_ROOT),
            env=env,
        ).returncode
        if rc != 0:
            print(f"[scriptgeom] FATAL: training segment exited {rc}", flush=True)
            sys.exit(rc)

        cur = latest_checkpoint(out_dir)
        if cur is None:
            print("[scriptgeom] FATAL: no checkpoint found after training", flush=True)
            sys.exit(5)
        step, ckpt = cur
        eval_dir = out_dir / f"eval_step_{step}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        print(f"[scriptgeom] === eval step {step}: {ckpt} ===", flush=True)
        log_path = eval_dir / "eval.log"
        with open(log_path, "w", encoding="utf-8") as lf:
            erc = subprocess.run(
                build_eval_cmd(ckpt, eval_dir, step),
                cwd=str(THYME_REPO_ROOT),
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
            ).returncode
        accs = {b: read_accuracy(eval_dir, b) for b in BENCHMARKS}
        print(f"[scriptgeom] step {step}: accs={accs} eval_rc={erc}", flush=True)
        log_eval_accuracy(step, accs)

        done_step = step
        target += args.eval_every

    print("\n[scriptgeom] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
