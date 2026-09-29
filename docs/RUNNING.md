# Training and evaluation guide

Research code accompanying the technical report **On-Policy Visual Evidence Distillation**.
The full method is named `strong_opd_gcep_combined_v2` in the implementation.
All commands below start at the repository root (the directory containing `requirements.txt`). Replace `/path/to/...` values
with your own resources. No author-hosted service or tracking account is required.

This package contains the main-method training and evaluation code for the Qwen
and InternVL lines, with their required SFT, frozen-teacher, data-preparation and
judge-service support. It does not contain experiment-result archives, standalone
ablations, analysis or plotting scripts. Model weights and datasets must be
provided separately. The resource and environment requirements below must be
satisfied before running an experiment.

## 1. Code map

- Training entry: `thyme-infer/scripts/launch_gcep.sh`.
- InternVL entry: `thyme-infer/scripts/launch_gcep_internvl.sh`.
- Training integration: `thyme-infer/Thyme/swift/trainers/rlhf_trainer/grpo_trainer.py`.
- Critic, certificate validation, anchor/reflection construction, token impact and
  reverse-KL: `thyme-infer/Thyme/swift/trainers/rlhf_trainer/gcep/`.
- Training reward: `thyme-infer/Thyme/examples/train/grpo/plugin/agent_rm.py`.
- Evaluation entry: `thyme-infer/scripts/run_11bench_eval_suite.sh` and
  `thyme-infer/scripts/phase3_eval_opd.py`.
- Benchmark builders and scorers: `thyme-infer/scripts/gcep_bench/`.
- InternVL SFT entries: `thyme-infer/scripts/internvl35_sft_stage1.sh` and
  `thyme-infer/scripts/internvl35_sft_stage2.sh`.
- InternVL data conversion: `tools/prepare_thyme_internvl_data.py`.
- Judge launch: `thyme-infer/scripts/launch_judge_qwen35_397b_vllm.sh`.
- Detailed setup: [environment](ENVIRONMENT.md), [training data](DATA.md),
  [evaluation data](EVAL_DATA.md), [judge service](JUDGE_SERVICE.md), and
  [model resources and recipes](EXPERIMENTS.md).

## 2. Environment and logging

The supplied configuration targets Linux, Python 3.10, CUDA 12.6 and PyTorch 2.7.
The root requirements define a release target based on the supplied experiment environment. The fork's
NumPy and TRL constraints have been aligned with those recorded versions;
a fresh GPU installation of the revised package has not yet been validated.

A reference installation sequence is:

```bash
conda create -n revue python=3.10
export ENV_NAME="revue"
# Set CONDA_DIR if your conda installation is not at $HOME/miniconda3.
source thyme-infer/activate_thyme.sh
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install packaging psutil ninja setuptools wheel
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -r requirements.txt
python -m pip install -e ./thyme-infer/Thyme
python -m pip install -e ./thyme-infer/Thyme/eval/VLMEvalKit
python -m pip check
```

FlashAttention requires a matching CUDA compiler/toolchain. These installation
commands are a reference recipe, not a claim of successful installation on every
GPU system. Keep the resulting package versions and installation diagnostics when
validating your environment.

W&B is disabled in the supplied launch configuration. Training reports use local
outputs/TensorBoard. Do not publish generated logs without checking them for your
own paths, service URLs, sample content and credentials.

The paper's training setup uses 8 NVIDIA H20 GPUs (96 GB each); the critic is
served on a separate machine with 8 H20 GPUs. Evaluation scripts default to
8 workers. The critic service and its memory requirements are additional to the
student training process.

## 3. External resources

Prepare these resources and use explicit local paths:

- Qwen student: the Thyme-SFT checkpoint for Qwen2.5-VL-7B.
- Qwen teacher: the frozen reproduced Thyme-RL expert selected at step 2150.
- InternVL cold-start student: the two-stage InternVL3.5-4B SFT checkpoint.
- InternVL teacher: the corresponding frozen RL expert selected at step 2750.
- Training data: the Thyme RL dataset in the Hugging Face datasets layout expected
  by the training runner, including images, questions and solutions.
- InternVL SFT data: the `stage1-*.parquet` and `stage2-*.parquet` files consumed by
  the two SFT scripts. The conversion program and recorded source revision,
  filtering rules and sample counts are described in `docs/DATA.md`.

The public Thyme project is https://github.com/yfzhang114/Thyme.
Exact experiment checkpoint hashes, training-data revisions and reconstruction
recipes must accompany any full reproduction claim. An internal checkpoint step
number alone does not identify a downloadable artifact.

### Judge service

Use a service under your control, with the required model loaded and an
OpenAI-compatible API. The reported critic is Qwen3.5-397B-A17B-FP8; its serving
version and launch configuration must be matched to the experiment records.
Set every client alias explicitly:

```bash
export REWARD_API_ADDRESS="localhost"
export QWEN_API_PORT="8000"
export REWARD_API_KEY="EMPTY"
export JUDGE_MODEL_NAME="your-served-model-alias"
export GCEP_JUDGE_BASE_URL="http://localhost:8000/v1"
export GCEP_JUDGE_API_KEY="EMPTY"
export GCEP_JUDGE_MODEL="your-served-model-alias"
export REMOTE_VLM_BASE_URL="http://localhost:8000/v1"
export REMOTE_VLM_API_KEY="EMPTY"
export REMOTE_VLM_MODEL="your-served-model-alias"
export JUDGE_BASE_URL="http://localhost:8000/v1"
export JUDGE_API_KEY="EMPTY"
export JUDGE_MODEL="your-served-model-alias"
```

Replace `EMPTY` when authentication is required. Check the served model alias and
perform a JSON-schema request before training; endpoint/model errors must not be
mistaken for valid critic feedback. Record critic failures and fallback counts.

### Benchmark data

`LMUData` defaults to `thyme-infer/data/lmudata`. Public registered VLMEvalKit
benchmarks can download their data through their dataset classes. Custom names
such as `VisualProbe_Easy`, `HallusionBench_GCEP`, `MathVerseVO` and
`VisuLogic_GCEP` require local TSV preparation; they are not all auto-downloaded.

The included builder entry accepts one benchmark at a time:

```bash
python thyme-infer/scripts/gcep_bench/datasets/builders.py --bench treebench
python thyme-infer/scripts/gcep_bench/datasets/builders.py --bench visualprobe
python thyme-infer/scripts/gcep_bench/datasets/builders.py --bench chartqapro
python thyme-infer/scripts/gcep_bench/datasets/builders.py --bench infographicvqa
python thyme-infer/scripts/gcep_bench/datasets/builders.py --bench mathverse_vo
python thyme-infer/scripts/gcep_bench/datasets/builders.py --bench visulogic_gcep
```

The HallusionBench builder additionally requires an explicit source directory
containing the pinned `HallusionBench.json` and any missing images in
`recovered_images/`. The source JSON must have SHA-256
`ca6e0fb677dd56e5fb0e2700d06c58d01f8a6ae66e0f1773b2ac48c0a095d051`.
The image-recovery procedure is not included and must be supplied before a full
reproduction:

```bash
export HALLUSIONBENCH_ROOT="/path/to/verified/HallusionBench"
python thyme-infer/scripts/gcep_bench/datasets/builders.py --bench hallusionbench
```

Builder assertions check sample counts and some pinned sources. Sources still
identified by a moving `main` revision need to be fixed to the experiment revision.
Record final TSV/image hashes, splits and row counts. Respect each dataset's use
and redistribution terms; the code licenses do not grant rights to all images.

## 4. Main-method training

```bash
source thyme-infer/activate_thyme.sh
export SFT_CKPT="/path/to/student-checkpoint"
export RL_DATA_DIR="/path/to/training-dataset"
bash thyme-infer/scripts/launch_gcep.sh \
  --mode strong_opd_gcep_combined_v2 \
  --teacher-ckpt "/path/to/teacher-checkpoint" \
  --lr 1e-6 --max-steps 1075 \
  "/path/to/training-output"
```

Configure the judge variables above first. For the InternVL line:

```bash
export INTERNVL_SFT_CKPT="/path/to/internvl-sft-checkpoint"
export RL_DATA_DIR="/path/to/training-dataset"
bash thyme-infer/scripts/launch_gcep_internvl.sh \
  --mode strong_opd_gcep_combined_v2 \
  --teacher-ckpt "/path/to/internvl-teacher-checkpoint" \
  --lr 1e-6 \
  "/path/to/internvl-training-output"
```

The recorded ReVuE evaluation checkpoints are 1075 (Qwen) and 350 (InternVL).
The InternVL command leaves `--max-steps` unset, preserving the supplied
launcher's default 1075-step schedule. Checkpoint 350 identifies an evaluated
snapshot; it does not establish the full training horizon of the reported run.
That horizon is not established by the available resource records. Changing it
also changes the learning-rate schedule before the selected checkpoint.
The monitor can resume from an existing output directory, so use a fresh directory
for an independent run.

The supplied main-method settings include 8 generations, gradient accumulation 20,
learning rate 1e-6, beta 0, seed 42, teacher top-32 support and impact fraction 0.2.
High- and low-impact groups each receive total weight 0.5. Refer to the source
configuration and paper for the complete settings. Baseline modes whose dependency
modules are absent are rejected explicitly rather than presented as runnable.

## 5. Evaluation

```bash
source thyme-infer/activate_thyme.sh
bash thyme-infer/scripts/run_11bench_eval_suite.sh \
  "/path/to/evaluation-checkpoint" "/path/to/evaluation-output"
```

The suite covers HRBench4K, HRBench8K, VStarBench, TreeBench, VisualProbe
(Easy/Medium/Hard), MathVista_MINI, MathVerseVO, VisuLogic_GCEP,
HallusionBench_GCEP, ChartQAPro and InfographicVQA_val. VisualProbe's three
subsets count as one benchmark family. The suite runs the subsets separately.
A subset can be selected by appending its exact script name.

For a small evaluation check after providing the required data and model:

```bash
SMOKE_N=2 bash thyme-infer/scripts/run_11bench_eval_suite.sh \
  "/path/to/evaluation-checkpoint" "/path/to/smoke-output" HRBench4K
```

Inspect both generation and scoring results. Judge-assisted benchmark scoring
requires the configured service. Protocol changes from upstream scorers are
identified in their source headers; these adapted scores must not be silently
presented as identical to unrelated official leaderboard protocols.

## 6. Execution boundary and validation status

The image-tool executor runs generated Python. Its filtering and timeout logic
are not an operating-system security boundary. Run experiments in an isolated
container or restricted account with only the required data available; do not
mount personal credentials or unrelated writable directories.

The source package uses explicit local resource paths,
disables W&B, and includes release-path corrections and third-party notices. Existing system paths and intentional
sandbox image-path conventions are retained. Public third-party attribution is
retained as required by the corresponding licenses.

Local validation covers Python and shell syntax, JSON parsing, release-path
checks and a scan for private paths and operational records. It does not establish
successful dependency installation, GPU compatibility, end-to-end training,
benchmark reproduction or agreement with the paper's numbers. Final validation
requires the actual model/data resources in a fresh environment.

## 7. Licenses and attribution

Original ReVuE code and original code modifications are licensed under the
[Apache License 2.0](../LICENSE); see [NOTICE](../NOTICE) for scope and attribution.
Third-party portions retain their respective terms. See
[third-party notices](../THIRD_PARTY_NOTICES.md), [license copies](../licenses/),
the [Thyme license](../thyme-infer/Thyme/LICENSE), and the
[VLMEvalKit license](../thyme-infer/Thyme/eval/VLMEvalKit/LICENSE).
Modified vendored files retain their historical release-modification notices.
The code license does not apply to the paper, figures, photographs, branding,
fonts, datasets, or model weights; these remain subject to their own terms where provided.
