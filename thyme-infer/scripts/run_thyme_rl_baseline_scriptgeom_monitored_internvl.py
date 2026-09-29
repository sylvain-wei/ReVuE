#!/usr/bin/env python3
# Modified for anonymous review: release paths, configuration, and documentation.
"""InternVL3.5 variant of the full-horizon script-aligned Thyme RL runner.

Identical orchestration to run_thyme_rl_baseline_scriptgeom_monitored.py; only
the shared constants / command builders are imported from the InternVL segmented
variant instead.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_thyme_rl_baseline_scriptgeom_segmented_internvl import (
    BETA,
    EVAL_MAX_PIXELS,
    GRAD_ACCUM,
    MAX_PIXELS,
    NUM_GENERATIONS,
    NUM_GPUS,
    NUM_TRAIN_EPOCHS,
    SAVE_STEPS,
    THYME_REPO_ROOT,
    build_eval_cmd,
    build_train_cmd,
    latest_checkpoint,
    read_accuracy,
)


DEFAULT_FULL_MAX_STEPS = int(os.environ.get("SCRIPTGEOM_FULL_MAX_STEPS", "2760"))
POLL_SEC = float(os.environ.get("SCRIPTGEOM_MONITOR_POLL_SEC", "30"))
STOP_GRACE_SEC = float(os.environ.get("SCRIPTGEOM_STOP_GRACE_SEC", "120"))


def sanitize_wandb_service_env(env: dict[str, str]) -> dict[str, str]:
    """Start each segment with a fresh W&B core service."""
    sanitized = env.copy()
    sanitized.pop("WANDB_SERVICE", None)
    return sanitized


def checkpoint_ready(ckpt: Path, expected_step: int) -> bool:
    required = [
        ckpt / "trainer_state.json",
        ckpt / "model.safetensors.index.json",
        ckpt / "config.json",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        state = json.loads((ckpt / "trainer_state.json").read_text())
    except Exception:
        return False
    return int(state.get("global_step", -1)) >= expected_step


def checkpoint_for_step(out_dir: Path, step: int) -> Path | None:
    for ckpt in out_dir.rglob(f"checkpoint-{step}"):
        if ckpt.is_dir() and checkpoint_ready(ckpt, step):
            return ckpt
    return None


def eval_targets(full_max_steps: int, eval_every: int) -> list[int]:
    targets = list(range(eval_every, full_max_steps + 1, eval_every))
    if not targets or targets[-1] != full_max_steps:
        targets.append(full_max_steps)
    return targets


def terminate_process_group(proc: subprocess.Popen) -> int:
    if proc.poll() is not None:
        return int(proc.returncode)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return int(proc.wait())
    deadline = time.time() + STOP_GRACE_SEC
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            return int(rc)
        time.sleep(2)
    print(
        f"[scriptgeom-monitor] training did not stop after {STOP_GRACE_SEC}s; sending SIGKILL",
        flush=True,
    )
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return int(proc.wait())


def run_training_until_checkpoint(
    *,
    out_dir: Path,
    target_step: int,
    full_max_steps: int,
    resume_ckpt: Path | None,
    env: dict[str, str],
    final_target: bool,
) -> tuple[int, Path]:
    existing = checkpoint_for_step(out_dir, target_step)
    if existing is not None:
        print(f"[scriptgeom-monitor] checkpoint-{target_step} already exists: {existing}", flush=True)
        return target_step, existing

    print(
        f"\n[scriptgeom-monitor] === train toward checkpoint-{target_step} "
        f"(full_horizon={full_max_steps}, resume={resume_ckpt}) ===",
        flush=True,
    )
    cmd = build_train_cmd(out_dir, full_max_steps, resume_ckpt)
    train_log_path = out_dir / f"train_step_{target_step}.log"
    env = sanitize_wandb_service_env(env)
    env.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("TQDM_DISABLE", "1")
    print(f"[scriptgeom-monitor] train log: {train_log_path}", flush=True)
    with open(train_log_path, "a", encoding="utf-8") as train_log:
        train_log.write(
            f"\n[scriptgeom-monitor] === train toward checkpoint-{target_step} "
            f"(full_horizon={full_max_steps}, resume={resume_ckpt}) ===\n"
        )
        train_log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(THYME_REPO_ROOT),
            env=env,
            stdout=train_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        if final_target:
            rc = proc.wait()
            if rc != 0:
                print(f"[scriptgeom-monitor] FATAL: final training exited {rc}", flush=True)
                sys.exit(rc)
            ckpt = checkpoint_for_step(out_dir, target_step)
            if ckpt is None:
                latest = latest_checkpoint(out_dir)
                if latest is None:
                    print("[scriptgeom-monitor] FATAL: no checkpoint found after final training", flush=True)
                    sys.exit(5)
                print(
                    f"[scriptgeom-monitor] final checkpoint-{target_step} not found; using latest {latest[1]}",
                    flush=True,
                )
                return latest
            return target_step, ckpt

        while True:
            ckpt = checkpoint_for_step(out_dir, target_step)
            if ckpt is not None:
                print(f"[scriptgeom-monitor] checkpoint-{target_step} is ready: {ckpt}", flush=True)
                rc = terminate_process_group(proc)
                print(f"[scriptgeom-monitor] training paused after checkpoint-{target_step}; rc={rc}", flush=True)
                return target_step, ckpt

            rc = proc.poll()
            if rc is not None:
                latest = latest_checkpoint(out_dir)
                print(
                    f"[scriptgeom-monitor] FATAL: training exited {rc} before checkpoint-{target_step}; "
                    f"latest={latest}; train_log={train_log_path}",
                    flush=True,
                )
                sys.exit(rc if rc != 0 else 6)

            time.sleep(POLL_SEC)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--full-max-steps", type=int, default=DEFAULT_FULL_MAX_STEPS)
    p.add_argument("--wandb-run-id", default="internvl35_thyme_grpo_baseline")
    p.add_argument("--wandb-project", default="anonymous-reproduction")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = eval_targets(args.full_max_steps, args.eval_every)
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
            for benchmark, acc in accs.items():
                if acc is not None:
                    log_dict[f"eval/{benchmark}_acc"] = acc
            run.log(log_dict)
            run.finish()
        except Exception as exc:
            print(f"[scriptgeom-monitor] WARNING: wandb eval log failed at step {step}: {exc}", flush=True)
        finally:
            if env_run_id is not None:
                os.environ["WANDB_RUN_ID"] = env_run_id

    print("[scriptgeom-monitor] resolved config:", flush=True)
    for key, value in [
        ("NUM_GPUS", NUM_GPUS),
        ("GRAD_ACCUM", GRAD_ACCUM),
        ("NUM_GENERATIONS", NUM_GENERATIONS),
        ("BETA", BETA),
        ("MAX_PIXELS", MAX_PIXELS),
        ("EVAL_MAX_PIXELS", EVAL_MAX_PIXELS),
        ("SAVE_STEPS", SAVE_STEPS),
        ("NUM_TRAIN_EPOCHS", NUM_TRAIN_EPOCHS),
        ("FULL_MAX_STEPS", args.full_max_steps),
        ("EVAL_EVERY", args.eval_every),
    ]:
        print(f"[scriptgeom-monitor]   {key}={value}", flush=True)
    print(f"[scriptgeom-monitor] out_dir={out_dir}", flush=True)
    print(f"[scriptgeom-monitor] eval_targets={targets[:8]}...{targets[-3:]}", flush=True)

    if args.dry_run:
        cmd = build_train_cmd(out_dir, args.full_max_steps, None)
        print("[scriptgeom-monitor] dry-run train command:")
        print(" ".join(cmd))
        return

    for target in targets:
        eval_dir = out_dir / f"eval_step_{target}"
        done_eval = all((eval_dir / f"eval_{benchmark}.json").exists() for benchmark in ["VStarBench", "HRBench4K"])
        if done_eval:
            print(f"[scriptgeom-monitor] eval_step_{target} already exists; skipping", flush=True)
            continue

        prev = latest_checkpoint(out_dir)
        resume_ckpt = prev[1] if prev else None
        env = os.environ.copy()
        env["WANDB_RUN_ID"] = f"{args.wandb_run_id}_seg{target}"
        env["WANDB_NAME"] = f"{args.wandb_run_id}_seg{target}"
        env["WANDB_RUN_GROUP"] = args.wandb_run_id
        env["WANDB_RESUME"] = "allow"

        step, ckpt = run_training_until_checkpoint(
            out_dir=out_dir,
            target_step=target,
            full_max_steps=args.full_max_steps,
            resume_ckpt=resume_ckpt,
            env=env,
            final_target=(target == targets[-1]),
        )

        eval_dir.mkdir(parents=True, exist_ok=True)
        print(f"[scriptgeom-monitor] === eval step {step}: {ckpt} ===", flush=True)
        log_path = eval_dir / "eval.log"
        with open(log_path, "w", encoding="utf-8") as lf:
            erc = subprocess.run(
                build_eval_cmd(ckpt, eval_dir, step),
                cwd=str(THYME_REPO_ROOT),
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
            ).returncode
        accs = {benchmark: read_accuracy(eval_dir, benchmark) for benchmark in ["VStarBench", "HRBench4K"]}
        print(f"[scriptgeom-monitor] step {step}: accs={accs} eval_rc={erc}", flush=True)
        log_eval_accuracy(step, accs)

        if erc != 0:
            print(f"[scriptgeom-monitor] FATAL: eval step {step} exited {erc}", flush=True)
            sys.exit(erc)

    print("\n[scriptgeom-monitor] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
