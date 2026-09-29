# Evaluation data preparation (11 benchmark families)

`LMUData` is the VLMEvalKit data root (`activate_thyme.sh` sets it to
`<package>/thyme-infer/data/lmudata`). Everything below assumes that variable.

**Read this first:** the custom TSVs used in the paper are **not** auto-downloaded by
VLMEvalKit. Only the registry datasets (MathVista_MINI, VStarBench, HRBench4K/8K,
and TreeBench) come from VLMEvalKit's built-in URLs. The custom benchmark TSVs are built from
the upstream sources below by `gcep_bench/datasets/builders.py`, which is shipped in this package
and asserts the exact row counts listed below.

```bash
export LMUData="<package>/thyme-infer/data/lmudata"
# Run from the package root. Prepare only the custom datasets used by the main suite:
for bench in treebench visualprobe chartqapro infographicvqa mathverse_vo visulogic_gcep; do
  PYTHONPATH="$PWD/thyme-infer/scripts:$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
    python -m gcep_bench.datasets.builders --bench "$bench"
done
# HallusionBench requires the source JSON and the recovery images described below:
export HALLUSIONBENCH_ROOT="<directory-containing-HallusionBench.json-and-recovered_images>"
PYTHONPATH="$PWD/thyme-infer/scripts:$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m gcep_bench.datasets.builders --bench hallusionbench
# Strict comparison with the recorded hashes (may differ after relocation; see below):
python tools/verify_eval_data.py --lmudata "$LMUData" --manifest-dir data_prep \
  --benchmarks TreeBench,VisualProbe_Easy,VisualProbe_Medium,VisualProbe_Hard,ChartQAPro,InfographicVQA_val,MathVerseVO,VisuLogic_GCEP,HallusionBench_GCEP \
  --json /tmp/local_eval_data_report.json --write-hash-table /tmp/local_tsv_hash.tsv
```

The builder writes `$LMUData/<BENCH>.tsv`, `$LMUData/images/<BENCH>/…`, and appends to
`$LMUData/gcep4_dataset_manifest.jsonl` or `gcep6_dataset_manifest.jsonl` (revision + row count + SHA-256). The manifests from the
supplied runs are shipped in `data_prep/gcep4_dataset_manifest.jsonl` and
`data_prep/gcep6_dataset_manifest.jsonl`; their recorded hash table is `data_prep/tsv_hash.tsv`.
These reference files retain only the latest supplied entry for each custom dataset used
by the main suite. Source revisions, row counts and hashes of those entries are unchanged.

## 1. Per-benchmark pins (server-reported checks against the paper-run tree)

| Benchmark | Source of truth | Rows | TSV SHA-256 (first 12) | Status on the paper-run tree |
|---|---|---|---|---|
| MathVista_MINI | VLMEvalKit registry (`image_vqa.py`) | 1000 | — | registry |
| VStarBench | VLMEvalKit registry | 191 | — | registry |
| HRBench4K | VLMEvalKit registry (`*_local.tsv` variant, >1 GB) | 800 | — | registry |
| HRBench8K | VLMEvalKit registry (`*_local.tsv`) | 800 | — | registry |
| TreeBench | HF `HaochenWang/TreeBench` @ `a4eb437ab71d`, split `train` | 405 | `035296025786` | **OK** (rows+hash+images) |
| VisualProbe Easy / Medium / Hard | HF `Mini-o3/VisualProbe_*` @ `0e665a57946d` / `25d5ff5bce48` / `bbe9b7a6c41d`, split `validation` | 141 / 268 / 106 | `b36da358e7ad` / `eea0484580cf` / `645706d587ae` | **OK** |
| MathVerseVO (vision-only) | HF `AI4Math/MathVerse` @ `3bc86196678b`, split `testmini` | 788 | `8d96f0c0347b` | **OK** |
| VisuLogic_GCEP | HF `VisuLogic/VisuLogic` @ `3e483f8cca0b`, split `test` | 1000 | `a6a533653a91` | **OK** |
| HallusionBench_GCEP | GH `tianyi-lab/HallusionBench` @ `744007c232c2`, split `test` | 1129 | `871855fed66e` | rows+hash+951 declared images **OK**; 178 rows declare no image (see Section 2) |
| ChartQAPro | HF `ahmed-masry/ChartQAPro`, split `test` | 1948 | `666b3c58cd54` | **OK** |
| InfographicVQA_val | VLMEvalKit mirror `InfoVQA_VAL.tsv` (`md5:2342e9c2`) | 2801 | `0d2ed8efdf3e` | **OK** |

Notes that matter:
- **Row counts must be counted with a TSV parser, not `wc -l`.** ChartQAPro, MathVerseVO,
  VisuLogic and InfographicVQA carry quoted fields with embedded newlines; line counts over-count
  them (e.g. ChartQAPro 3739 lines vs 1948 rows). The shipped `tsv_hash.tsv` uses parsed rows.
- `HallusionBench_GCEP` (suffixed) exists precisely because VLMEvalKit's md5 registry knows a
  dataset called `HallusionBench`; if you ship a file with that exact name, the registry replaces
  it at load time with the official TSV. Use the suffixed `HallusionBench_GCEP` identifier.
- The main suite has 13 dataset identifiers, corresponding to 11 benchmark families after
  grouping the three VisualProbe subsets. MME-RealWorld-Lite is not in its default list.

## 2. HallusionBench_GCEP: how the images were obtained

The supplied reconstruction record uses a source checkout, a public image mirror and nine
additional recovery images. It describes the following assembly; this release has not
downloaded or independently rebuilt that tree:

1. **Source checkout** — `$HALLUSIONBENCH_ROOT` at commit `744007c232c2` (the commit the
   manifest pins). Provides the questions, categories and the on-disk directory layout
   (`examples/<set>/…`).
2. **Image mirror** — the public HF mirror `rayguan/HallusionBench` @ `2067dfdc`, downloaded and
   expanded into `$LMUData/images/HallusionBench/{VS,VD}/{chart,table,video,illusion,...}/…`.
3. **Recovered images** — 9 files that neither the source nor the mirror provided
   (`6_1, 7_1, 7_2, 8_1, 8_2, 9_1, 9_2, 10_1, 10_2`, PNG) were recovered from the
   `mm-eval/HallusionBench` @ `b91d8e36` copy and stored in
   `$HALLUSIONBENCH_ROOT/recovered_images/` (an input you must supply).
4. **TSV build** — `gcep_bench.datasets.builders` writes `HallusionBench_GCEP.tsv` with absolute
   `image_path` values resolved against the layout above, then records the row count and SHA-256
   in the manifest (1129 rows, `871855fed66e…`).

Server-reported verification: **all 951 declared image paths exist**. The remaining 178 rows
(all `category=VS`, `visual_input=0`) declare an **empty** image list in the TSV — i.e. they are
part of the recorded dataset as the authors used it; whether the original evaluation resolved them
from another location is `NOT_VERIFIED` (the evaluation produced all 1129 predictions, so the
loader accepted them).

## 3. Verification report supplied by the server

The server's broader verification report marks all nine custom datasets retained here as
matching their recorded row counts and SHA-256 hashes, with every declared image present.
The 178 HallusionBench rows with empty image lists still require the qualification above.
Registry datasets without a supplied hash are identified separately in the table.

These checks have not been repeated against the full datasets in the merged release. The local
verifier uses parsed TSV rows and permits large fields; it fails on missing manifests or malformed
image-path lists. The release's default manifests cover the nine custom inputs relevant
to the main suite.

## 4. Evaluation entry contract (what the data must satisfy)

```bash
torchrun --nproc_per_node=8 thyme-infer/scripts/phase3_eval_opd.py \
    --model-path <checkpoint> --benchmarks <Bench1,Bench2,...> --out-dir <eval_dir> \
    [--judge-base-url $REMOTE_VLM_BASE_URL --judge-api-key $REMOTE_VLM_API_KEY --judge-model $REMOTE_VLM_MODEL]
```

Per benchmark it writes `eval_<BENCH>.json` (aggregate: `total`, `correct`, `accuracy`,
`run_status`, semantic-judge counters), `metrics_<BENCH>.json` (scorer aggregate incl.
`n_unscored`), `coverage_<BENCH>.json`, `scores_<BENCH>.jsonl` (per-sample `sample_id`/
`correct`), and `predictions_<BENCH>.jsonl`. Two considerations when writing custom
aggregation:

- a `correct = null` row means the sample could not be scored; do not coerce it to 0. `scores_*`
  is the per-sample evidence. Paired analyses must check completion status and explicitly
  account for any unscored samples before comparing runs;
- `coverage_*.json` can report `unscored_ids: []` even when `scores_*` contains nulls — check
  `metrics_*/eval_*` `n_unscored` as well before pooling arms.

## 5. Relocation and incomplete recovery inputs

The builders write absolute image paths into TSVs. Rebuilding under a different root normally
changes the TSV byte hash; the supplied hashes identify the original build bytes. The verifier
reports this mismatch rather than rewriting reference evidence. Inspect row counts, pinned
source revisions and image availability separately; a hash mismatch alone after relocation does
not establish changed questions or images. A normalized content hash has not been supplied.

The nine HallusionBench recovery images and a deterministic source-to-file mapping are still
not included. The source names above identify where the server says they came from; they are
not an executable recovery recipe. Supplying the exact images to `HALLUSIONBENCH_ROOT` remains
a prerequisite for rebuilding the paper dataset. Do not substitute the stale unsuffixed TSV.

TreeBench can auto-download through the vendored VLMEvalKit registry. The explicit builder is
recommended when reconstructing the recorded revision. Avoid `--bench all`: it also builds
historical datasets outside the main suite and requires their separate inputs.

Some auxiliary image sources, including the current InfographicVQA image fallback, are loaded
without a dataset revision in the builder. Their exact source snapshots must still be recorded
by the experiment host before claiming a fully pinned data reconstruction.
