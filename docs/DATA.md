# Training data: sources, schema, preprocessing, and the InternVL conversion program

Two datasets are involved. The RL training pool is a **verbatim** public download; the InternVL  
cold start is trained on a **converted** public dataset, and the conversion program is shipped  
here (`tools/prepare_thyme_internvl_data.py`) because it is part of the scientific record.

---

## 1. RL training pool (both lines)

| Item          | Value                                                                                                                                                                                                                                              |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Loaded from   | a local parquet directory `<package>/thyme-infer/data/thyme_rl_train/data`, passed as `--dataset <dir>` (constants: `run_thyme_rl_baseline_scriptgeom_segmented.py:38,116-117`; InternVL variant `:40,122-123`)                                    |
| Public origin | HF dataset `Kwai-Keye/Thyme-RL`, revision `4fd9aa21abe45bb180287fbb769c3616e6597c28`                                                                                                                                                               |
| Layout        | 98 shards, `train-00000-of-00098.parquet` …                                                                                                                                                                                                        |
| Size          | **55,173 rows**, 48,081,907,189 bytes (44.77 GiB)                                                                                                                                                                                                  |
| Columns       | `messages` (list of role/content), `images` (list, base64 strings), `solution`, `question`                                                                                                                                                         |
| Conversion    | **none** — the files are a plain download; the HF download sidecars record a per-shard etag that equals the content SHA-256, so all 98 shard hashes are recoverable from `data/thyme_rl_train/.cache/huggingface/download/data/*.parquet.metadata` |
| Sampled hash  | `train-00000-of-00098.parquet` = `6fc2eaf2de320954ce5a7eeff2a082fbf22f15a28521fb5001a42bebc538f942`                                                                                                                                                |

Why a repo patch exists: the fork's dataset loader treats a local folder as a standard swift  
dataset (`swift/llm/dataset/loader.py:242`), and the preprocessor needed three adjustments for  
this data (`swift/llm/dataset/preprocessor/core.py:326-405`): cast the nested image list to  
`large_string`, convert the `messages` struct-of-arrays to a list-of-structs, and keep `solution`  
as the auxiliary column `__#solution`. There is **no** `thyme_rl_train` entry in the swift  
dataset registry — the directory path is the interface.

Periodic evaluation uses the retained benchmark indices in  
`thyme-infer/configs/phase3_periodic_validation_vstar_hrbench4k_indices.json`.  
The supplied training launchers do not require a separate validation parquet builder.

## 2. InternVL cold-start SFT data (converted)

| Stage   | Output                                                        | Shards | Rows        | Source splits                               |
| ------- | ------------------------------------------------------------- | ------ | ----------- | ------------------------------------------- |
| stage 1 | `thyme-infer/data/thyme_sft_internvl/stage1/stage1-*.parquet` | 167    | **332,996** | `wo_thinking_thyme_single_round` + `2round` |
| stage 2 | `thyme-infer/data/thyme_sft_internvl/stage2/stage2-*.parquet` | 7      | **12,309**  | `computation`                               |

- Source: HF dataset `Kwai-Keye/Thyme-SFT`, revision `8a65e8065475f8a1f95f290b0478f1d12ccf8e68`,  
  downloaded as `<package>/thyme-infer/data/thyme_sft_raw/data/<split>-*.parquet`  
  (218 shards, 345,973 rows: 181,277 single-round / 152,354 two-round / 12,342 computation).
- Row-count check for a shard: `python -c 'import sys, pyarrow.parquet as pq; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' /path/to/shard.parquet`.
- Sampled hashes: `stage1-00000.parquet` = `0c3a25206dd1004bee25cf181a5cac189952b597af3f898e1ba599ef3e0dbb13`,  
  `stage2-00000.parquet` = `97b71e60f912dea02d33e8ace8def03f88ef1c845f1b26d583bd571a1c4633ab`.  
  (Full-set hashing of ~100 GB was not performed; only the first shard of each stage was hashed.)

### 2.1 The conversion program (shipped)

`tools/prepare_thyme_internvl_data.py` contains the supplied conversion logic, with release  
changes for explicit paths and input/output checks. The output counts below are server-reported;  
this merged release has not reconverted the source dataset.

- **Routing**: stage 1 = `["wo_thinking_thyme_single_round", "2round"]`, stage 2 = `["computation"]`  
  with identical filters in both stages.
- **Validation filters** (applied per row): non-empty `response`; the number of images  
  must equal the number of `<image>` tags; `<code>`/`<sandbox_output>`/`<answer>`/`<think>` tag  
  pairs must be balanced; `'<answer>'` must be present; `len(question + response) <= max_chars`.
- **Why HF `datasets` and not plain pyarrow**: the script writes with  
  `datasets.Dataset.from_list(..., features=Features({...})).to_parquet(...)`  so that  
  swift's `ResponsePreprocessor` can load the nested base64 image list; a plain pyarrow parquet  
  triggers `ArrowNotImplementedError` on that nested column.
- **Sharding**: `--shard-rows` (default 2000) → `stage1-%05d.parquet` / `stage2-%05d.parquet`.
- **CLI**: required `--raw-root` and `--out-root`; optional `--max-chars` (default 60000),  
  `--limit-per-split` (0 = no limit), and `--shard-rows` (default 2000). Use a new or empty output  
  directory. Missing source splits or a stage with no valid rows cause an error. A failed run  
  can leave partial output; rerun into a new directory after correcting the input.
- **Recorded statistics** are retained in `data_prep/prepare_stats.json` (paths anonymized): per-split  
  `seen` / `kept` / `dropped` breakdown (e.g. 181,277 seen → 180,652 kept with 625 dropped for  
  unbalanced `<code>`; 152,354 → 152,344), and the final row/shard counts.
- Companion audit tool `tools/audit_thyme_sft_labels.py` (checks the swift-side answer masking;  
  the supplement reports `OVERALL_PASS` on both stages). Run it with  
  `python tools/audit_thyme_sft_labels.py --stage-dir <stage-dir> --model <local-InternVL-checkpoint>`.  
  The release requires a positive sample count and returns nonzero for failed checks. It checks  
  placeholder counts and sampled labels; it does not prove image-content identity or absence  
  of every form of future-image leakage.

The exact experiment-time conversion arguments were not supplied. The packaged defaults,  
filtering logic and recorded statistics describe the available reconstruction recipe.

## 3. Reconstructing training data from public sources

| Reproducible      | How                                                                                                                                                                                                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RL pool           | `huggingface-cli download Kwai-Keye/Thyme-RL --repo-type dataset --revision 4fd9aa21abe45bb180287fbb769c3616e6597c28 --local-dir /path/to/thyme_rl_train` and point `RL_DATA_DIR` at its `data/` directory                                                                                  |
| InternVL SFT data | download `Kwai-Keye/Thyme-SFT` at revision `8a65e8065475f8a1f95f290b0478f1d12ccf8e68`, then `python tools/prepare_thyme_internvl_data.py --raw-root <raw>/data --out-root <package>/thyme-infer/data/thyme_sft_internvl`; compare `prepare_stats.json` (shipped) and the first-shard hashes |

Not shipped: the parquet files themselves (≈150 GB across the four datasets). The packages need  
image bytes for the vision tower, which is why the RL pool keeps base64 images inside the parquet  
rather than paths.

The converter retains an entire stage in memory before writing shards; the supplied datasets  
are large, so sufficient host RAM is required. Byte-level Parquet hashes can also depend on  
writer-library versions. The recorded first-shard hashes are reference evidence, not a guarantee  
that a newly converted file is byte-identical under a different pyarrow version.
