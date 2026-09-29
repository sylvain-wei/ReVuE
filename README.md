<p align="center">
  <img src="assets/hero.jpg" alt="ReVuE — On-Policy Visual Evidence Distillation" width="100%">
</p>

<h1 align="center">On-Policy Visual Evidence Distillation</h1>

<p align="center">
  Shaohang Wei<sup>1‡*</sup>,
  Feifan Song<sup>1</sup>,
  Guangyue Peng<sup>1</sup>,
  Wenhao Yu<sup>3</sup>,
  Wei Li<sup>1</sup>,
  Wen Luo<sup>1</sup>,<br>
  Yang Xu<sup>4</sup>,
  Yufan Shen<sup>2</sup>,
  Luke Mao<sup>2</sup>,
  Yang Du<sup>2</sup>,
  Asher Qin<sup>2</sup>,
  Houfeng Wang<sup>1†</sup>
</p>

<p align="center">
  <sup>1</sup>Peking University &nbsp; <sup>2</sup>Tencent &nbsp; <sup>3</sup>CUHK &nbsp; <sup>4</sup>Nanjing University
</p>

<p align="center">
  <sup>‡</sup> Project leader &nbsp; <sup>†</sup> Corresponding authors<br>
  <sup>*</sup> Work done during the internship at Tencent.<br>
  Correspondence: <a href="mailto:shaohang@stu.pku.edu.cn">shaohang@stu.pku.edu.cn</a>
</p>

<p align="center">
  <a href="website/assets/documents/revue-paper.pdf"><img src="https://img.shields.io/badge/Paper-Technical_Report-B31B1B?style=flat-square" alt="Paper: Technical report"></a>
  <a href="https://sylvain-wei.github.io/ReVuE/"><img src="https://img.shields.io/badge/Website-Project_Page-275EE8?style=flat-square" alt="Project website"></a>
  <a href="#license-and-acknowledgments"><img src="https://img.shields.io/badge/License-Pending-lightgrey?style=flat-square" alt="License selection pending"></a>
  <a href="docs/ENVIRONMENT.md"><img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10 target environment"></a>
  <a href="docs/ENVIRONMENT.md"><img src="https://img.shields.io/badge/CUDA-12.6-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA 12.6 target environment"></a>
  <a href="https://github.com/sylvain-wei/ReVuE/actions/workflows/static-checks.yml"><img src="https://github.com/sylvain-wei/ReVuE/actions/workflows/static-checks.yml/badge.svg" alt="Static checks workflow"></a>
</p>

<p align="center">
  <a href="#from-visual-evidence-to-better-supervision">Overview</a> &nbsp;·&nbsp;
  <a href="#two-islands-not-one">Token-level case</a> &nbsp;·&nbsp;
  <a href="#main-results">Results</a> &nbsp;·&nbsp;
  <a href="#key-findings">Findings</a> &nbsp;·&nbsp;
  <a href="#getting-started">Getting started</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

## From visual evidence to better supervision

**ReVuE (Reflection on Visual Evidence)** teaches visual agents to acquire the right evidence, read it correctly, and use it to answer the question. It compares student-generated attempts, diagnoses the first **Acquire → Read → Ground** failure, and supplies the resulting reflection to the teacher during training. Token-level distillation gives greater weight to positions where reflection changes the teacher's predictions most.

<p align="center">
  <img src="assets/overview.gif" alt="Animated ReVuE overview: HRBench 8K results, followed by Acquire, Read, Ground, and error attribution" width="100%">
</p>

The overview pairs HRBench 8K results with an evidence-use example: crop the blue car, read its plate, and map the plate number to the answer. The student retains its original interaction history; reflection guides the teacher during training.

<sub>[View the complete figure](website/assets/figures/plate-evidence-chain.svg) · [Interactive animation](https://sylvain-wei.github.io/ReVuE/) · [Figure PDF](website/assets/figures/plate-evidence-chain.pdf)</sub>

## Two islands, not one

<p align="center">
  <img src="assets/two-islands.gif" alt="Token-by-token student trajectory: two islands are misread as one; reflection changes teacher support at the intermediate mistake" width="100%">
</p>

The crop shows **two islands**, but the student describes one and answers **Saint Lucia** instead of **Saint Kitts and Nevis**. Reflection lowers teacher support for the intermediate mistake—`island`, `one`, and the first `Lucia`—while the final `Lucia` changes little under the fixed erroneous prefix.

Blue and orange show increased and decreased teacher support. Both evaluations score **the same student trajectory**; the animation reveals its tokens and score changes.

<sub>[View the complete figure](website/assets/figures/two-islands.svg) · [Figure PDF](website/assets/figures/two-islands.pdf)</sub>

## Main results

Accuracy (%), higher is better. **Bold** marks the best OPD result in each model family, including ties. Category averages use benchmark sample counts as weights.

### Overview

| Model / Method | Perception | Math | General |
| :-- | --: | --: | --: |
| **Off-the-Shelf Models** |  |  |  |
| GPT-4o | 50.28 | 41.22 | 53.28 |
| Gemini3.1FL | 43.89 | 62.63 | 63.57 |
| Qwen2.5-32B | 63.89 | 51.90 | 59.29 |
| Qwen3-30B-T | 63.26 | 56.71 | 64.45 |
| InternVL-38B | 54.34 | 48.89 | 55.71 |
| **Qwen2.5-VL-7B** |  |  |  |
| Base Model | 57.77 | 46.34 | 50.53 |
| Cold-start | 60.72 | 46.38 | 52.68 |
| RL Expert | 63.89 | 47.56 | 53.60 |
| RFT | 63.12 | 46.70 | 53.07 |
| Vanilla OPD | 62.61 | 47.67 | 53.28 |
| GT-Privileged | 62.26 | 47.49 | 53.23 |
| Vision-OPD | 59.98 | 45.48 | 52.08 |
| V-Zero | 62.08 | 46.99 | 52.79 |
| VAD | 62.08 | 45.62 | 53.65 |
| **ReVuE (Ours)** | **65.01** | **48.49** | **54.14** |
| **InternVL3.5-4B-Instruct** |  |  |  |
| Base Model | 49.32 | 42.00 | 47.78 |
| Cold-start | 55.66 | 47.45 | 52.79 |
| RL Expert | 58.20 | 47.60 | 54.38 |
| RFT | 58.22 | 46.81 | 54.26 |
| Vanilla OPD | 58.02 | 46.34 | 53.48 |
| GT-Privileged | 57.36 | 46.09 | 53.91 |
| Vision-OPD | 53.04 | 46.02 | 54.14 |
| V-Zero | 54.45 | 47.56 | 53.99 |
| VAD | 53.78 | 47.45 | 54.61 |
| **ReVuE (Ours)** | **59.72** | **49.17** | **55.13** |

<details>
<summary><b>Perception — all benchmark results</b></summary>

| Model / Method | HRBench 4K | HRBench 8K | V\*Bench | TreeBench | VisualProbe | Wtd. Avg. |
| :-- | --: | --: | --: | --: | --: | --: |
| **Off-the-Shelf Models** |  |  |  |  |  |  |
| GPT-4o | 61.00 | 54.00 | 61.78 | 49.88 | 23.88 | 50.28 |
| Gemini3.1FL | 46.00 | 43.00 | 64.92 | 51.85 | 27.96 | 43.89 |
| Qwen2.5-32B | 75.13 | 69.25 | 78.01 | 48.40 | 45.05 | 63.89 |
| Qwen3-30B-T | 77.13 | 71.38 | 80.10 | 45.43 | 36.89 | 63.26 |
| InternVL-38B | 71.50 | 62.13 | 65.97 | 41.98 | 20.97 | 54.34 |
| **Qwen2.5-VL-7B** |  |  |  |  |  |  |
| Base Model | 69.00 | 63.50 | 75.39 | 37.04 | 41.17 | 57.77 |
| Cold-start | 74.38 | 66.62 | 80.10 | 37.28 | 41.56 | 60.72 |
| RL Expert | 75.50 | 71.75 | 83.77 | 40.25 | 44.85 | 63.89 |
| RFT | 75.38 | 72.00 | 81.20 | 40.00 | 41.75 | 63.12 |
| Vanilla OPD | 75.40 | 70.50 | 81.20 | 38.02 | 42.91 | 62.61 |
| GT-Privileged | 73.62 | 70.50 | 79.58 | 39.75 | 43.10 | 62.26 |
| Vision-OPD | 73.00 | 65.62 | 76.44 | 39.01 | 41.36 | 59.98 |
| V-Zero | 75.62 | 69.50 | 78.01 | 37.28 | 43.10 | 62.08 |
| VAD | 74.75 | 67.75 | 76.96 | **41.98** | 43.89 | 62.08 |
| **ReVuE (Ours)** | **77.10** | **74.00** | **82.20** | 41.12 | **44.66** | **65.01** |
| **InternVL3.5-4B-Instruct** |  |  |  |  |  |  |
| Base Model | 62.00 | 55.00 | 68.59 | 40.49 | 20.58 | 49.32 |
| Cold-start | 69.50 | 63.25 | 66.49 | 40.99 | 29.90 | 55.66 |
| RL Expert | 72.62 | 64.62 | 71.73 | 40.74 | 34.56 | 58.20 |
| RFT | 71.25 | 66.00 | 71.73 | 40.25 | 35.00 | 58.22 |
| Vanilla OPD | 71.13 | 65.87 | **73.30** | 40.25 | 33.79 | 58.02 |
| GT-Privileged | 71.25 | 64.88 | 70.16 | 39.75 | 33.20 | 57.36 |
| Vision-OPD | 66.25 | 60.00 | 65.97 | 37.78 | 28.93 | 53.04 |
| V-Zero | 68.13 | 61.75 | 67.02 | 40.25 | 28.35 | 54.45 |
| VAD | 67.50 | 61.00 | 67.02 | 39.01 | 27.96 | 53.78 |
| **ReVuE (Ours)** | **73.25** | **67.37** | **73.30** | **41.48** | **36.12** | **59.72** |

</details>

<details>
<summary><b>Math — all benchmark results</b></summary>

| Model / Method | MathVista | MathVerse | VisuLogic | Wtd. Avg. |
| :-- | --: | --: | --: | --: |
| **Off-the-Shelf Models** |  |  |  |  |
| GPT-4o | 58.83 | 39.21 | 25.20 | 41.22 |
| Gemini3.1FL | 80.80 | 77.92 | 32.40 | 62.63 |
| Qwen2.5-32B | 77.00 | 53.05 | 25.90 | 51.90 |
| Qwen3-30B-T | 80.20 | 66.12 | 25.80 | 56.71 |
| InternVL-38B | 70.90 | 48.48 | 27.20 | 48.89 |
| **Qwen2.5-VL-7B** |  |  |  |  |
| Base Model | 69.10 | 44.04 | 25.40 | 46.34 |
| Cold-start | 68.00 | 44.29 | 26.40 | 46.38 |
| RL Expert | 70.60 | 45.43 | 26.20 | 47.56 |
| RFT | 69.50 | 45.81 | 24.60 | 46.70 |
| Vanilla OPD | 70.60 | 46.07 | 26.00 | 47.67 |
| GT-Privileged | 71.20 | 45.69 | 25.20 | 47.49 |
| Vision-OPD | 67.70 | 42.39 | 25.70 | 45.48 |
| V-Zero | 69.80 | 43.91 | **26.60** | 46.99 |
| VAD | 69.00 | 45.56 | 22.30 | 45.62 |
| **ReVuE (Ours)** | **71.60** | **47.08** | 26.50 | **48.49** |
| **InternVL3.5-4B-Instruct** |  |  |  |  |
| Base Model | 68.50 | 28.93 | 25.80 | 42.00 |
| Cold-start | 69.30 | 47.21 | 25.80 | 47.45 |
| RL Expert | 69.30 | 46.32 | 26.90 | 47.60 |
| RFT | 70.00 | 45.94 | 24.30 | 46.81 |
| Vanilla OPD | 68.40 | 43.40 | 26.60 | 46.34 |
| GT-Privileged | 71.30 | 39.34 | 26.20 | 46.09 |
| Vision-OPD | 66.60 | 44.54 | 26.60 | 46.02 |
| V-Zero | 68.80 | 46.95 | 26.80 | 47.56 |
| VAD | 68.30 | 46.07 | 27.70 | 47.45 |
| **ReVuE (Ours)** | **71.60** | **47.46** | **28.10** | **49.17** |

</details>

<details>
<summary><b>General — all benchmark results</b></summary>

| Model / Method | HallusionBench | ChartQA-Pro | InfographicVQA | Wtd. Avg. |
| :-- | --: | --: | --: | --: |
| **Off-the-Shelf Models** |  |  |  |  |
| GPT-4o | 51.37 | 28.67 | 71.17 | 53.28 |
| Gemini3.1FL | 59.92 | 37.03 | 83.50 | 63.57 |
| Qwen2.5-32B | 50.74 | 30.02 | 83.10 | 59.29 |
| Qwen3-30B-T | 61.82 | 35.46 | 85.68 | 64.45 |
| InternVL-38B | 50.74 | 26.98 | 77.70 | 55.71 |
| **Qwen2.5-VL-7B** |  |  |  |  |
| Base Model | 40.91 | 20.79 | 75.10 | 50.53 |
| Cold-start | 41.73 | 21.12 | 79.05 | 52.68 |
| RL Expert | 43.83 | 21.69 | 79.74 | 53.60 |
| RFT | 42.33 | 21.18 | 79.57 | 53.07 |
| Vanilla OPD | 42.88 | 21.69 | 79.44 | 53.28 |
| GT-Privileged | 43.45 | 20.84 | 79.70 | 53.23 |
| Vision-OPD | 39.89 | 20.79 | 78.75 | 52.08 |
| V-Zero | 42.16 | 21.08 | 79.12 | 52.79 |
| VAD | 44.24 | 21.73 | 79.65 | 53.65 |
| **ReVuE (Ours)** | **45.12** | **21.81** | **80.27** | **54.14** |
| **InternVL3.5-4B-Instruct** |  |  |  |  |
| Base Model | 40.04 | 26.01 | 66.04 | 47.78 |
| Cold-start | 48.50 | 26.24 | 72.98 | 52.79 |
| RL Expert | 46.92 | 30.59 | 73.93 | 54.38 |
| RFT | 47.78 | 29.39 | 74.17 | 54.26 |
| Vanilla OPD | 46.85 | 30.46 | 72.16 | 53.48 |
| GT-Privileged | 49.29 | 29.90 | 72.48 | 53.91 |
| Vision-OPD | 44.57 | 30.46 | **74.47** | 54.14 |
| V-Zero | 49.03 | 28.65 | 73.61 | 53.99 |
| VAD | 48.05 | 31.44 | 73.36 | 54.61 |
| **ReVuE (Ours)** | **49.38** | **32.27** | 73.35 | **55.13** |

</details>

MathVista uses the Mini split; MathVerse uses the vision-only split. Gemini3.1FL denotes Gemini-3.1-Flash-Lite; Qwen2.5-32B, Qwen3-30B-T, and InternVL-38B denote Qwen2.5-VL-32B, Qwen3-VL-30B-A3B-Thinking, and InternVL3.5-38B-Instruct.

## Accuracy, response length, and tool use

<p align="center">
  <img src="assets/efficiency.png" alt="Four panels: A HRBench 8K accuracy versus response length, B VStarBench accuracy versus response length, C HRBench 8K tool-use and conditional accuracy, D VStarBench tool-use and conditional accuracy" width="100%">
</p>

**A–B:** Accuracy versus mean response length on HRBench 8K and V\* Bench. **C–D:** Tool-use rate and answer accuracy on samples with and without tool use, for the same two benchmarks. All four panels use Qwen2.5-VL-7B.

<sub>[Vector figure](website/assets/figures/efficiency.svg) · [Figure PDF](website/assets/figures/efficiency.pdf)</sub>

| Benchmark | Method | Accuracy (%) | Mean response tokens | Tool-use rate (%) |
| :-- | :-- | --: | --: | --: |
| HRBench 8K | Vanilla OPD | 70.50 | 456 | 74.2 |
| HRBench 8K | ReVuE (Ours) | 74.00 | 382 | 54.0 |
| V\* Bench | Vanilla OPD | 81.20 | 515 | 95.8 |
| V\* Bench | ReVuE (Ours) | 82.20 | 401 | 74.3 |

## Key findings

- **Category gains in both model families.** Across 11 benchmarks, ReVuE leads all evaluated OPD methods in the three weighted category averages for both Qwen2.5-VL-7B and InternVL3.5-4B-Instruct. Perception rises from **62.61% to 65.01%** for Qwen and from **58.02% to 59.72%** for InternVL over Vanilla OPD (Table 1).
- **Higher accuracy with shorter responses and less frequent tool use.** The four panels show these gains over Vanilla OPD on HRBench 8K and V\* Bench. On HRBench 8K, ReVuE also exceeds its RL Expert teacher: **74.00% vs. 71.75%** accuracy with **382 vs. 458** mean response tokens.
- **Useful supervision can appear before the final answer.** In the two-island case, reflection lowers teacher support for the intermediate one-island judgment even though the final-answer token changes little under the fixed erroneous prefix. Both scores refer to the same student trajectory.

## Getting started

The release includes ReVuE training and evaluation for **Qwen2.5-VL-7B** and **InternVL3.5-4B**, together with data preparation, the InternVL SFT stages, frozen-teacher recipes, and judge-service integration. Start with the [complete running guide](docs/RUNNING.md).

| Step | Guide |
| :-- | :-- |
| Install the training environment | [Environment and dependencies](docs/ENVIRONMENT.md) |
| Prepare student, frozen teacher, and training data | [Model resources and recipes](docs/EXPERIMENTS.md) · [Training data](docs/DATA.md) |
| Start and verify the critic / judge | [Judge service](docs/JUDGE_SERVICE.md) |
| Prepare evaluation benchmarks | [Evaluation data](docs/EVAL_DATA.md) |
| Check release scope and provenance | [Validation](VALIDATION.md) · [Source provenance](docs/PROVENANCE.md) |

### Installation

The target environment is **Linux, Python 3.10, CUDA 12.6, and PyTorch 2.7**. The main training recipe uses **8 GPUs**; the reported training setup used 8 × H20 (96 GB), with a separate judge service. Model checkpoints and benchmark datasets are supplied separately.

Clone the repository, then install the dependencies:

```bash
git clone https://github.com/sylvain-wei/ReVuE.git
cd ReVuE

conda create -n revue python=3.10 -y
export ENV_NAME="revue"
source thyme-infer/activate_thyme.sh

python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install packaging psutil ninja setuptools wheel
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -r requirements.txt
python -m pip install -e ./thyme-infer/Thyme
python -m pip install -e ./thyme-infer/Thyme/eval/VLMEvalKit
python -m pip check
```

FlashAttention needs a compatible CUDA build toolchain. Follow the [environment guide](docs/ENVIRONMENT.md) for the recorded package versions and installation constraints.

### Prepare resources and the judge

1. Prepare the RL dataset and local image paths using the [data guide](docs/DATA.md). For InternVL, run the two SFT stages documented in [EXPERIMENTS.md](docs/EXPERIMENTS.md).
2. Supply the cold-start student and its matched frozen RL teacher. The report used locally reproduced teachers; a public Thyme-RL checkpoint is not a verified substitute.
3. Configure the OpenAI-compatible judge and the reward, reflection, and scoring endpoint variables in [JUDGE_SERVICE.md](docs/JUDGE_SERVICE.md). Run its schema and two-image smoke checks before training.

### Train ReVuE

**Qwen2.5-VL-7B**

```bash
export ENV_NAME="revue"
source thyme-infer/activate_thyme.sh
export SFT_CKPT="/path/to/student-checkpoint"
export RL_DATA_DIR="/path/to/training-dataset"

bash thyme-infer/scripts/launch_gcep.sh \
  --mode strong_opd_gcep_combined_v2 \
  --teacher-ckpt "/path/to/teacher-checkpoint" \
  --lr 1e-6 --max-steps 1075 \
  "/path/to/training-output"
```

**InternVL3.5-4B**

```bash
export ENV_NAME="revue"
source thyme-infer/activate_thyme.sh
export INTERNVL_SFT_CKPT="/path/to/internvl-sft-checkpoint"
export RL_DATA_DIR="/path/to/training-dataset"

bash thyme-infer/scripts/launch_gcep_internvl.sh \
  --mode strong_opd_gcep_combined_v2 \
  --teacher-ckpt "/path/to/internvl-teacher-checkpoint" \
  --lr 1e-6 \
  "/path/to/internvl-training-output"
```

The InternVL command retains the launcher’s default 1,075-step schedule. The report evaluated checkpoint 350; that checkpoint number does not establish the original run’s total training horizon.

The main configuration uses 8 generations, learning rate `1e-6`, seed `42`, teacher support size `32`, and high-impact fraction `0.2`. High- and low-impact groups receive equal total loss weight. See the [full recipes](docs/EXPERIMENTS.md) for SFT, teacher training, and checkpoint details.

### Evaluate on 11 benchmarks

Prepare the datasets and judge using [EVAL_DATA.md](docs/EVAL_DATA.md) and [JUDGE_SERVICE.md](docs/JUDGE_SERVICE.md), then run:

```bash
export ENV_NAME="revue"
source thyme-infer/activate_thyme.sh
bash thyme-infer/scripts/run_11bench_eval_suite.sh \
  "/path/to/evaluation-checkpoint" \
  "/path/to/evaluation-output"
```

The suite covers **HRBench 4K, HRBench 8K, V\* Bench, TreeBench, VisualProbe, MathVista, MathVerse, VisuLogic, HallusionBench, ChartQA-Pro, and InfographicVQA**. VisualProbe's Easy / Medium / Hard splits count as one benchmark. The [running guide](docs/RUNNING.md#5-evaluation) also covers benchmark subsets and smoke runs.

### Code map

```text
src/m_rlsd/                        Shared ReVuE helpers
thyme-infer/scripts/               Training, teacher, SFT, and evaluation entrypoints
thyme-infer/scripts/gcep_bench/    Dataset builders and benchmark scorers
thyme-infer/Thyme/                Modified training framework and VLMEvalKit
data_prep/                       Data preparation notes and resources
tools/                          Data conversion, verification, and static checks
docs/                           Complete environment and execution guides
website/                        Interactive project website
assets/                         README animations and ReVuE branding
```

### Checks and release scope

```bash
python tools/check_repository.py
```

The **Static checks** workflow checks source syntax, release entrypoints, metadata, and retained license integrity. The [validation record](VALIDATION.md) distinguishes current checks from the earlier mocked release tests. Fresh Linux/CUDA installation, GPU training, live-judge inference, and reproduction of the reported scores remain unverified.

This release supplies the main method; standalone comparison-method implementations, ablation runners, experiment archives, model weights, and benchmark datasets are not bundled. The image-tool executor runs generated Python; use the isolated environment described in [the execution guide](docs/RUNNING.md#6-execution-boundary-and-validation-status).

## License and acknowledgments

A general open-source license for original ReVuE code is pending selection. Existing third-party licenses remain in force; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/).

ReVuE builds on [Thyme](https://github.com/Kwai-Keye/Thyme), [ms-swift](https://github.com/modelscope/ms-swift), and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). We thank the benchmark and framework authors whose work makes these experiments possible.

Banner photograph by [Jasper Wilde on Unsplash](https://unsplash.com/photos/boys-blue-eyes-Sk3fZLg-zTc). Branding and figure-source details are in [assets/README.md](assets/README.md).

## Citation

```bibtex
@misc{revue,
  author = {Shaohang Wei and Feifan Song and Guangyue Peng and
            Wenhao Yu and Wei Li and Wen Luo and
            Yang Xu and Yufan Shen and Luke Mao and
            Yang Du and Asher Qin and Houfeng Wang},
  title  = {On-Policy Visual Evidence Distillation},
  year   = {2026},
  note   = {Technical report}
}
```
