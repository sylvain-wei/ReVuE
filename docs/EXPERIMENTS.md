# Main-method model resources and training recipes

This document covers the Qwen2.5-VL-7B and InternVL3.5-4B training and evaluation
paths in this release. Model weights must be provided by the user. The recorded
checkpoint numbers below are resource identifiers; they do not establish that a
fresh run will produce byte-identical weights or the paper's scores.

## Student and frozen teacher resources

| Resource | Qwen2.5-VL-7B | InternVL3.5-4B |
| --- | --- | --- |
| Cold-start student | `Kwai-Keye/Thyme-SFT`, based on `Qwen/Qwen2.5-VL-7B-Instruct` | `OpenGVLab/InternVL3_5-4B-Instruct`, revision `a3fd3158be4027880d91ccee80ea8089559e0896`, followed by two-stage SFT |
| SFT preparation | Public Thyme-SFT checkpoint | Thyme-SFT data revision `8a65e8065475f8a1f95f290b0478f1d12ccf8e68`; stage 1 checkpoint 7728, stage 2 checkpoint 288 |
| Frozen teacher | Locally reproduced GRPO expert, checkpoint 2150 | Locally reproduced GRPO expert, checkpoint 2750 |
| Recorded ReVuE evaluation checkpoint | 1075 | 350 |

An evaluation checkpoint is distinct from the training horizon. The available
records identify InternVL checkpoint 350 but do not establish that run's full
training horizon. The supplied main-method launcher defaults to 1075 steps;
the InternVL example in [the running guide](RUNNING.md#4-main-method-training)
retains that default. Setting a 350-step horizon would also change the
learning-rate schedule before checkpoint 350.

The two teachers are local GRPO reproductions. The public `Kwai-Keye/Thyme-RL`
model is not identified as the teacher used in these runs and must not be treated
as an interchangeable checkpoint. Complete teacher weight hashes were not
included in the supplied resource records.

## Preparation and training entrypoints

Run commands from the package root after completing `DATA.md`, `ENVIRONMENT.md`
and `JUDGE_SERVICE.md`.

1. Prepare the Thyme RL pool and, for InternVL, the two SFT parquet datasets.
2. Build the InternVL cold-start checkpoint using
   `thyme-infer/scripts/internvl35_sft_stage1.sh` and
   `thyme-infer/scripts/internvl35_sft_stage2.sh`.
3. Build the local GRPO teacher using
   `thyme-infer/scripts/run_baseline_scriptgeom_full_monitored.sh` or its
   `_internvl.sh` counterpart. These scripts are retained because the main
   method requires the corresponding frozen teacher.
4. Use `thyme-infer/scripts/launch_gcep.sh` or `launch_gcep_internvl.sh` with
   `--mode strong_opd_gcep_combined_v2` and an explicit `--teacher-ckpt`.
5. Evaluate the chosen checkpoint using
   `thyme-infer/scripts/run_11bench_eval_suite.sh`.

### InternVL cold start

After converting the SFT dataset, run the two stages in order:

```bash
MODEL="/path/to/InternVL3_5-4B-Instruct" \
DATA_DIR="/path/to/thyme_sft_internvl/stage1" \
OUTPUT_DIR="/path/to/internvl-stage1-output" \
bash thyme-infer/scripts/internvl35_sft_stage1.sh

STAGE1_CKPT="/path/to/stage1-final-checkpoint" \
DATA_DIR="/path/to/thyme_sft_internvl/stage2" \
OUTPUT_DIR="/path/to/internvl-stage2-output" \
bash thyme-infer/scripts/internvl35_sft_stage2.sh
```

Use the actual final checkpoint directory written by stage 1 as `STAGE1_CKPT`.
The recorded run selected checkpoints 7728 and 288 for the two stages; a new
run's directory name should be read from its own outputs.

### Frozen teacher preparation

With the reward judge and periodic evaluation data configured, the Qwen teacher
recipe starts from Thyme-SFT:

```bash
export SFT_CKPT="/path/to/Thyme-SFT"
export RL_DATA_DIR="/path/to/thyme_rl_train/data"
bash thyme-infer/scripts/run_baseline_scriptgeom_full_monitored.sh \
  "/path/to/qwen-teacher-training-output"
```

The InternVL teacher recipe starts from the stage 2 cold-start checkpoint:

```bash
export INTERNVL_SFT_CKPT="/path/to/internvl-stage2-final-checkpoint"
export RL_DATA_DIR="/path/to/thyme_rl_train/data"
bash thyme-infer/scripts/run_baseline_scriptgeom_full_monitored_internvl.sh \
  "/path/to/internvl-teacher-training-output"
```

Run teacher preparation in a fresh shell configured for the baseline recipe,
without inherited OPD overrides. Both monitor scripts use a 2760-step schedule
and resume existing output directories. The recorded frozen teachers were
selected at Qwen step 2150 and InternVL step 2750. Use explicit checkpoint
directories in the [ReVuE launch commands](RUNNING.md#4-main-method-training).

### Main-method settings

The supplied main-method configuration uses 8 GPUs, gradient accumulation 20,
8 generations, beta 0, seed 42, learning rate 1e-6, teacher support size 32,
and impact fraction 0.2. High- and low-impact groups receive equal total weight.
The teacher recipe uses 4 generations, beta 0.01, learning rate 5e-7 and a
2760-step schedule. Qwen uses `max_pixels=2408448`; InternVL uses its dynamic
image tiling path. Explicit runtime settings and resource versions should be
recorded when executing these recipes.

## Release scope

The supported paper-method entry is `strong_opd_gcep_combined_v2`. The shared
training implementation retains internal modes and controls used by the original
code, but this release does not supply standalone ablation, analysis, plotting,
or experimental-result archives. Generic framework modules and benchmark
scorers needed by the training and evaluation imports are retained.

Vision-OPD, V-Zero and VAD matched-teacher modules and evidence-bank builders are
not present in the provided files. Their launch modes are rejected explicitly;
this package must not be described as containing complete implementations of
those comparison methods.
