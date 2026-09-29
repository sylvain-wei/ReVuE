# Upstream provenance

The release contains vendored source snapshots with research-specific modifications.
The repository records the released source and its research-specific changes.
[The original source-package manifest](archive/code-review-MANIFEST.sha256)
fingerprints the earlier review archive, before the public repository documentation
and presentation assets were added; it is not a manifest of this repository.
A public upstream checkout alone does not reproduce the research modifications.

## Thyme

| Item | Value |
|---|---|
| Public upstream | <https://github.com/yfzhang114/Thyme> |
| Packaged location | `thyme-infer/Thyme` |
| Exact upstream base | **UNKNOWN:** no authoritative upstream base revision is recorded in this release. |
| Research modifications | GCEP/ReVuE trainer integration, teacher loading, sandbox integration, and evaluation support. |

## ms-swift

| Item | Value |
|---|---|
| Public upstream | <https://github.com/modelscope/ms-swift> |
| Packaged location | `thyme-infer/Thyme/swift` |
| Version string | `swift/version.py`: `3.5.0.dev0` |
| Exact upstream base | **UNKNOWN:** the vendored source has no independent recorded upstream revision. |
| Dependency guidance | See `ENVIRONMENT.md`; version metadata alone does not identify the full vendored source. |

## VLMEvalKit

| Item | Value |
|---|---|
| Public upstream | <https://github.com/open-compass/VLMEvalKit> |
| Packaged location | `thyme-infer/Thyme/eval/VLMEvalKit` |
| Version string | `vlmeval/__init__.py`: `0.2rc1`; installation metadata: `0.1.0` |
| Exact upstream base | **UNKNOWN.** The supplied source documentation reports a candidate content match to `d71e9470c5e5bc815529a90bfc712043d23641be`; this is **NOT_VERIFIED** and is not an authoritative pin. |
| Research modifications | Model registry integration and the `vlmeval/vlm/thyme/` evaluation wrapper. |

## Evaluation scorers

| Component | Location | Source status |
|---|---|---|
| ChartQAPro | `thyme-infer/scripts/gcep_bench/scorers/chartqapro.py` | Ported evaluation code; exact upstream revision **UNKNOWN**. |
| ReasonMap / ReasonMap-Plus | `thyme-infer/scripts/gcep_bench/scorers/reasonmap_plus.py` | The dataset manifest identifies public source `fscdc/ReasonMap`; the scorer has no independently recorded version. Dataset revisions do not establish scorer provenance. |
| Other benchmark scorers | `thyme-infer/scripts/gcep_bench/scorers/` and `vlmeval/dataset/utils/` | Packaged implementations and benchmark-specific prompts; see `gcep_bench/assets/` and retained source notices. |

Third-party copyright notices and license texts remain part of the release.
Missing source revisions are reported as unknown rather than replaced with a
current upstream commit.
