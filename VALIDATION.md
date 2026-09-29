# Validation status

## Repository checks

Run the dependency-free checks from the repository root with Python 3.10 or newer:

```bash
python tools/check_repository.py
```

The **Static checks** GitHub Actions workflow runs the same command on Python
3.10 and 3.12. It checks Python 3.10 syntax, shell syntax, JSON/JSONL parsing,
documented entry paths, reference-data metadata consistency, third-party
license hashes, local documentation links, and the evaluation code's source
fingerprint. It does not install the training dependencies or execute a model.

The local run on 2026-09-29 passed using Python 3.12.14:

| Check | Result |
| --- | --- |
| Python sources | 685 files parsed with Python 3.10 grammar |
| Shell sources | 9 files passed `bash -n` |
| Documented entry paths | 25 present |
| JSON and JSONL | 39 JSON files; 2 JSONL files with 9 records |
| Reference evaluation metadata | 9 TSV entries agree with their source manifests |
| Third-party license inventory | 26 components; 17 distinct hashed license files verified |
| Evaluation source fingerprint | Repository root resolves correctly; 742 source files fingerprinted |
| Local documentation links | 46 targets present |

All 684 Python files and 9 shell files copied from the curated source package
remain byte-identical. The added Python file is the repository checker.
The command-line help for `tools/probe_judge.py` and `tools/verify_eval_data.py`
also ran locally without loading model dependencies or calling a service.

Python 3.12 reports one existing `SyntaxWarning` for an invalid escape sequence
in the vendored `seephys.py` scorer; the file still parses. The source is retained
unchanged. The check command rejects Python versions older than 3.10.

## Execution scope

A fresh Linux/CUDA installation, GPU training, model inference, live judge
request, dataset download, and numerical benchmark reproduction have not been
performed for this repository. The configured package versions and the supplied
recipes still require validation with the intended checkpoints and datasets.
See [environment details](docs/ENVIRONMENT.md), [model resources](docs/EXPERIMENTS.md),
[evaluation data](docs/EVAL_DATA.md), and [judge setup](docs/JUDGE_SERVICE.md).

The inherited source package documents earlier isolated mock checks of launch
failure propagation, judge request structure, data verification, and converter
guards. That historical record is preserved in
[the original validation note](docs/archive/code-review-VALIDATION.md).
Those checks are not represented as newly run model tests. The accompanying
[archived manifest](docs/archive/code-review-MANIFEST.sha256) describes that
earlier source package, before this repository's documentation and visual assets.

The image-tool executor runs model-generated Python. Its filters and timeout
are not an operating-system security boundary. Run experiments in an isolated
container or restricted account with only the required data available.
