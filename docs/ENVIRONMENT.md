# Environment and installation

The server supplement reports the following experiment environment. This is a record supplied
by the experiment host, not a clean-install validation performed for this merged package.

| Component | Reported version |
|---|---|
| Python / pip | 3.10.20 / 26.0.1 |
| CUDA toolkit / torch / torchvision | 12.6 / 2.7.0+cu126 / 0.22.0+cu126 |
| Hardware | 8 × NVIDIA H20, 96 GB per GPU; driver 535.247.01 |
| transformers / vllm / trl | 4.52.4 / 0.9.2 / 0.19.1 |
| numpy / deepspeed / flash-attn | 2.2.6 / 0.19.2 / 2.8.3 |
| accelerate / peft / datasets | 1.13.0 / 0.15.2 / 3.3.2 |
| qwen-vl-utils / pillow / pandas | 0.0.14 / 12.2.0 / 2.3.3 |
| av / decord / timm | 17.0.1 / 0.6.0 / 1.0.27 |
| openai / httpx / einops | 1.90.0 / 0.28.1 / 0.8.2 |
| sentencepiece / math-verify / timeout-decorator | 0.2.1 / 0.9.0 / 0.5.0 |
| OpenCV | opencv-python-headless 4.13.0.92; opencv-python absent |
| tensorboard / triton | 2.21.0 / 3.7.1 |
| Forks installed as editables | ms-swift 3.5.0.dev0; VLMEvalKit 0.1.0 |

The reported host environment had unresolved `pip check` findings: ms-swift declared
`numpy<2` and `trl<0.18`; torch declared `triton==3.3.0` while the host had 3.7.1;
VLMEvalKit declared `opencv-python` while only its headless distribution was installed;
and an installed cupy package lacked `cuda-pathfinder`. Successful recorded runs do not
establish that these dependency conflicts are harmless on every code path.

## Release dependency decisions

`thyme-infer/Thyme/setup.py` reads `requirements.txt`, which includes
`requirements/framework.txt`. Therefore that framework metadata **does participate** in both
ordinary and editable installation. The merged release retains the source-package metadata change
to `numpy==2.2.6` and `trl==0.19.1`, consistent with the supplied experiment targets.
This removes those two declared conflicts; it does not prove runtime compatibility.

The root requirements pin `datasets==3.3.2` and `tensorboard==2.21.0` using the supplied record.
They select `opencv-python==4.13.0.92` to satisfy the existing VLMEvalKit metadata with one
OpenCV distribution. This differs from the host's headless packaging and needs a clean-server
check. Do not install both OpenCV distributions in the same environment. W&B is not a release
requirement, and release entrypoints force its online logging off.

Do not force the recorded `triton==3.7.1` into this target: it conflicts with torch's declared
requirement. Let the resolver select a compatible version and record it during server
validation. The supplement did not establish which change introduced the host's newer Triton.
`latex2sympy2_extended` and some transitive/fork dependencies remain unpinned because a complete,
consistent lockfile and their exact experiment versions were not supplied.

## Installation target (Linux, Python 3.10, CUDA 12.6)

Use a new environment. From the package root:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install packaging ninja wheel setuptools
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -r requirements.txt
python -m pip install -e ./thyme-infer/Thyme
python -m pip install -e ./thyme-infer/Thyme/eval/VLMEvalKit
python -m pip check
```

FlashAttention is deliberately absent from the root requirement list: its source build needs
a working torch and matching CUDA toolchain first. A matching prebuilt wheel may be used when
available. Installation may also require platform build libraries; this sequence has not been
executed in a fresh GPU environment for the merged release.

The server reports a resolution-only dry run of 182 packages **excluding FlashAttention**.
It did not validate the merged root requirements plus both editable installs, and it was not
an installation or a training/evaluation run. Before claiming reproducibility, record the
full resolved environment, require a clean `pip check`, import both forks, and run a small
training and evaluation job with the intended checkpoints and datasets.

## Judge serving environment

The Qwen3.5-397B judge uses a separate environment and separate hardware. The training
`vllm==0.9.2`/`transformers==4.52.4` target is not a serving environment for that model.
See `JUDGE_SERVICE.md`; the exact tested serving build still needs confirmation from the
experiment host. Missing or invalid critic responses fall back to clean-teacher OPD at the
group level, so a running training process alone is not proof that the critic was active.
