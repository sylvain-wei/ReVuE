# Third-party notices and source attribution

This package contains third-party frameworks, adapted evaluation code, and copied prompt templates. Their authors retain their copyrights and their respective licenses apply to those portions. Names and public upstream links below identify third-party projects and are retained for attribution.

Any statement about original ReVuE code does not replace or restrict these third-party licenses. This notice grants no new license for original ReVuE code. Dataset and model-weight terms are separate from code licenses; this archive does not redistribute the benchmark datasets or model weights.

The original framework LICENSE files and source-level copyright notices are retained. Additional license texts are reproduced unchanged under `licenses/`. [licenses/SOURCES.json](licenses/SOURCES.json) records official source URLs, local scope, recorded source revisions, license-verification revisions, and file hashes.

## Vendored frameworks

| Component and upstream attribution | Local scope | License and version information |
| --- | --- | --- |
| [Thyme — Kwai-Keye/Thyme](https://github.com/Kwai-Keye/Thyme), the public release fork of [yfzhang114/Thyme](https://github.com/yfzhang114/Thyme) | `thyme-infer/Thyme/`, including Thyme training, templates, sandbox, and agent evaluation support | Apache-2.0; full text retained at [Thyme/LICENSE](thyme-infer/Thyme/LICENSE). Original vendoring commit was not recorded in the supplied archive. |
| [ms-swift — ModelScope / Alibaba](https://github.com/modelscope/ms-swift) | `thyme-infer/Thyme/swift/` and framework packaging | Apache-2.0; the existing [Thyme/LICENSE](thyme-infer/Thyme/LICENSE) supplies the full terms. Retained source headers credit Alibaba, Inc. and its affiliates. Original vendoring commit was not recorded; the included version string is `3.5.0.dev0`, which is not a commit identifier. |
| [VLMEvalKit — VLMEvalKit Authors / OpenCompass](https://github.com/open-compass/VLMEvalKit) | `thyme-infer/Thyme/eval/VLMEvalKit/` | Apache-2.0; full text and `Copyright 2023 VLMEvalKit Authors` retained at [VLMEvalKit/LICENSE](thyme-infer/Thyme/eval/VLMEvalKit/LICENSE). Original vendoring commit was not recorded. |

These are trimmed, locally modified distributions. They include ReVuE integration, local training and evaluation adaptations, and release configuration changes. They are not represented as pristine upstream checkouts. Existing individual-file notices also apply.

## Copied prompts and adapted benchmark scorers

Paths in this table are relative to `thyme-infer/scripts/gcep_bench/`. The references are source-code revisions, not Hugging Face dataset revisions.

| Upstream project | Supplied material and upstream source | Recorded source revision | License copy |
| --- | --- | --- | --- |
| [HallusionBench](https://github.com/tianyi-lab/HallusionBench), Fuxiao Liu | `assets/hallusionbench_judge_prompt.txt` and `scorers/hallusionbench.py`; adapted from `utils.py` | `744007c232c292942c7f80eb61edb2465482da31` | [BSD-3-Clause](licenses/HallusionBench/LICENSE.md) |
| [MathVerse](https://github.com/ZrrSkywalker/MathVerse), Renrui Zhang | `assets/mathverse_demo_prompt_extract.txt`, `assets/mathverse_demo_prompt_score.txt`, and `scorers/mathverse_vo.py`; `evaluation/prompts.py` and `evaluation/score_answer_s2.py` | `937b090597aeafb8e82b35d310a4bc5b9e2ea29d` | [MIT](licenses/MathVerse/LICENSE) |
| [PerceptionBench](https://github.com/MoonshotAI/PerceptionBench), MoonshotAI | `assets/perceptionbench_judge_prompt.txt`, `scorers/perceptionbench.py`; `eval/judge_prompt.txt` and `eval/eval.py` | `ba032c06` (prefix recorded by adapter) | [Apache-2.0](licenses/PerceptionBench/LICENSE) |
| [Mini-o3](https://github.com/Mini-o3/Mini-o3) | Prompt text and adapted scoring in `scorers/visualprobe.py`; `verl/utils/reward_score/general_qa_tool.py` | `2c5a0dedb5279eff2c0e6049aac05de97bf7a2b3` | [Apache-2.0](licenses/Mini-o3/LICENSE) |
| [VisuLogic-Eval](https://github.com/VisuLogic-Benchmark/VisuLogic-Eval) | Extraction messages and adapted extraction code in `scorers/visulogic_gcep.py`; `evaluation/eval_model.py` | `00fba6dd` (prefix recorded by adapter) | [Apache-2.0](licenses/VisuLogic/LICENSE) |
| [Zooming without Zooming / ZoomBench](https://github.com/inclusionAI/Zooming-without-Zooming), inclusionAI | Judge prompt and extraction protocol in `scorers/zoombench.py`; adapter identifies `judge_qwenlm.py` | `fdc0ba1` (prefix recorded by adapter) | [Apache-2.0](licenses/ZoomBench/LICENSE) |
| [ChartQAPro](https://github.com/vis-nlp/ChartQAPro), vis-nlp | Adapted helpers in `scorers/chartqapro.py`; `evaluate_predictions.py` | Original port commit not recorded; adapter only names `main` | [MIT](licenses/ChartQAPro/LICENSE) |
| [ReasonMap / ReasonMap-Plus](https://github.com/fscdc/ReasonMap), copyright holder 冯思程 | Adapted extraction and weighted aggregation in `scorers/reasonmap_plus.py`; `main_plus.py` and `cal_metrics.py` | Original port commit not recorded | [MIT](licenses/ReasonMap/LICENSE) |

The adapters document changes to judge models, extraction, parsing, retries, and aggregation in their module docstrings. These adaptations do not imply endorsement by upstream authors or equivalence to every upstream evaluation setting. The unused MathVerse extraction template is also covered because it remains distributed in the archive.

## Additional third-party portions retained inside the frameworks

The following are identified by explicit copied/borrowed-code statements or copyright headers in files shipped in this archive. This list is not a list of Python packages merely imported at runtime. Original copying revisions are not recorded; the manifest distinguishes the official license snapshot inspected from the unknown original code revision.

| Upstream source | Supplied portion | Applicable notice or license |
| --- | --- | --- |
| [Microsoft LoRA](https://github.com/microsoft/LoRA) | `Thyme/swift/tuners/lora.py` and `lora_layers.py` retain Microsoft copyright and MIT headers | [MIT](licenses/LoRA/LICENSE.md) |
| [Donut — NAVER Corp.](https://github.com/clovaai/donut) | `VLMEvalKit/vlmeval/dataset/utils/ccocr_evaluator/kie_evaluator.py` | [MIT](licenses/Donut/LICENSE); the existing NAVER copyright header is retained |
| [VQA — Aishwarya Agrawal](https://github.com/GT-Vision-Lab/VQA) | `VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py`, which identifies adoption from the VQA evaluation code | [Original BSD-style two-clause text and additional disclaimer](licenses/VQA/license.txt) |
| [OpenAI prm800k](https://github.com/openai/prm800k) and [Hendrycks MATH](https://github.com/hendrycks/math) | `Thyme/swift/plugin/math_normalize.py` names both sources | [prm800k MIT](licenses/prm800k/LICENSE), [MATH MIT](licenses/MATH/LICENSE) |
| [TableVQA-Bench — NAVER Cloud Corp.](https://github.com/naver-ai/tablevqabench) | `VLMEvalKit/vlmeval/dataset/utils/tablevqabench.py` | [MIT](licenses/TableVQABench/LICENSE) and [upstream NOTICE](licenses/TableVQABench/NOTICE) |
| [allennlp-semparse — Allen Institute for AI](https://github.com/allenai/allennlp-semparse) | The same TableVQA evaluation module explicitly identifies this source | Apache-2.0; full terms in [VLMEvalKit/LICENSE](thyme-infer/Thyme/eval/VLMEvalKit/LICENSE), attribution also retained in the TableVQA [NOTICE](licenses/TableVQABench/NOTICE) |
| [Hugging Face TRL](https://github.com/huggingface/trl) | `Thyme/swift/trainers/rlhf_trainer/grpo_trainer.py` identifies borrowed implementation | Apache-2.0; full terms in [Thyme/LICENSE](thyme-infer/Thyme/LICENSE); source statement retained |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | `Thyme/swift/trainers/{utils,mixin,trainers}.py`, `swift/tuners/longlora/llama.py`, and identified optimizer portions | Apache-2.0; full terms in [Thyme/LICENSE](thyme-infer/Thyme/LICENSE); source statements retained |
| [GaLore](https://github.com/jiaweizzhao/GaLore) | `Thyme/swift/trainers/optimizers/galore/` | Apache-2.0; full terms in [Thyme/LICENSE](thyme-infer/Thyme/LICENSE); source statements retained |
| [LongLoRA](https://github.com/dvlab-research/LongLoRA) | `Thyme/swift/tuners/longlora/` | Apache-2.0; full terms in [Thyme/LICENSE](thyme-infer/Thyme/LICENSE); source statements retained |
| [Sentence Transformers](https://github.com/UKPLab/sentence-transformers), Nils Reimers and contributors | Borrowed loss implementation identified in `Thyme/swift/plugin/loss.py` | Apache-2.0; full terms in [Thyme/LICENSE](thyme-infer/Thyme/LICENSE) |
| [MMEngine](https://github.com/open-mmlab/mmengine), [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), and [XTuner](https://github.com/InternLM/xtuner) | Explicitly identified borrowed portions of `Thyme/swift/trainers/sequence_parallel/ulysses.py` | Apache-2.0; full terms in [Thyme/LICENSE](thyme-infer/Thyme/LICENSE); source statements retained |

Here `Thyme/` abbreviates `thyme-infer/Thyme/`, and `VLMEvalKit/` abbreviates its `eval/VLMEvalKit/` subdirectory.

`Thyme/swift/plugin/math_equal.py` and `VLMEvalKit/vlmeval/dataset/utils/bmmr_grade.py` already contain Apache-2.0 and full MIT notices crediting NVIDIA, Microsoft, OpenAI, and Dan Hendrycks. These notices remain in those files. IBM copyright and Apache-2.0 notices in the shipped TEDS/OCR utilities also remain in place.

Hugging Face copyright headers in the shipped tuner and import utilities are also retained. Those headers alone do not identify an exact originating repository or revision, so this notice does not infer a separate source attribution merely from the presence of PEFT imports.

## Provenance limits

The original archive did not provide exact vendoring commits for the three framework trees or the ChartQAPro and ReasonMap ports. Current official license snapshots were verified and hashed; their commits must not be interpreted as the versions used for the paper's runs. The manifest retains this distinction explicitly. These additions do not verify every historical upstream modification or establish dataset/model redistribution rights.
