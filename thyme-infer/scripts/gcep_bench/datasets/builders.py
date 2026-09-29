# Modified for anonymous review: release paths, configuration, and documentation.
"""Build lmudata TSV + image files for the 6 gcep benchmarks from pinned sources.

Usage:
    python -m gcep_bench.datasets.builders --bench treebench|zoombench|visualprobe|reasonmap_plus|chartqapro|infographicvqa|all

Outputs (per benchmark):
    $LMUData/<BENCH>.tsv            (image_path-only TSV; VLMEvalKit uses it directly)
    $LMUData/images/<BENCH>/...     (decoded images)
    $LMUData/gcep6_dataset_manifest.jsonl  (appended; revision + row count + sha256)

Assertions: exact row counts (405/845/141+268+106/1448/1948/2801); any mismatch
aborts the build. Nothing here imports torch.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import io
import json
import os
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from gcep_bench import manifests  # noqa: E402

LMUDATA = Path(os.environ.get("LMUData", str(Path(__file__).resolve().parents[3] / "data" / "lmudata")))
MANIFEST_PATH = LMUDATA / "gcep6_dataset_manifest.jsonl"

PINS = {
    "TreeBench": {"source": "hf://datasets/HaochenWang/TreeBench", "revision": "a4eb437ab71d14f6ab1a4cf05213d0f72f831b71", "n": 405, "split": "train"},
    "ZoomBench": {"source": "hf://datasets/inclusionAI/ZoomBench", "revision": "b788097e57d30510c6877824833234a73bf80d25", "n": 845, "split": "test"},
    "VisualProbe_Easy": {"source": "hf://datasets/Mini-o3/VisualProbe_Easy", "revision": "0e665a57946d4473deb9975a13f8ddba973b8026", "n": 141, "split": "validation"},
    "VisualProbe_Medium": {"source": "hf://datasets/Mini-o3/VisualProbe_Medium", "revision": "25d5ff5bce48147be34fd2655617ab17dc55b3fa", "n": 268, "split": "validation"},
    "VisualProbe_Hard": {"source": "hf://datasets/Mini-o3/VisualProbe_Hard", "revision": "bbe9b7a6c41d49844f45e51916ec0a60d1f0e378", "n": 106, "split": "validation"},
    "ReasonMapPlus": {"source": "hf://datasets/FSCCS/ReasonMap-Plus", "revision": "main", "n": 1448, "split": "test"},
    "ChartQAPro": {"source": "hf://datasets/ahmed-masry/ChartQAPro", "revision": "main", "n": 1948, "split": "test"},
    "InfographicVQA_val": {"source": "https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv", "revision": "md5:2342e9c225222f0ef4dec545ebb126fe", "n": 2801, "split": "val"},
}


def _img_dir(bench: str) -> Path:
    d = LMUDATA / "images" / bench
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_pil(img, path: Path) -> str:
    if not path.exists():
        img.save(path)
    return str(path)


def _write_tsv(bench: str, fieldnames: list[str], rows: list[dict]) -> str:
    path = LMUDATA / f"{bench}.tsv"
    tmp = path.with_suffix(".tsv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return str(path)


def _record_manifest(bench: str, tsv_path: str, n_rows: int, extra: dict | None = None) -> None:
    pin = PINS[bench]
    manifests.append_jsonl(str(MANIFEST_PATH), [{
        "schema_version": "1",
        "created_at": manifests.utc_now_iso(),
        "benchmark_id": bench,
        "source": pin["source"],
        "revision": pin["revision"],
        "split": pin["split"],
        "n_rows": n_rows,
        "tsv_path": tsv_path,
        "tsv_sha256": manifests.sha256_file(tsv_path),
        **(extra or {}),
    }])


def _assert_n(bench: str, got: int) -> None:
    want = PINS[bench]["n"]
    assert got == want, f"{bench}: expected {want} rows, got {got}"


# ---------------------------------------------------------------- TreeBench
def build_treebench() -> None:
    from datasets import load_dataset
    bench = "TreeBench"
    ds = load_dataset("HaochenWang/TreeBench", revision=PINS[bench]["revision"], split="train")
    _assert_n(bench, len(ds))
    img_dir = _img_dir(bench)
    letters = list("ABCDEFGHIJK")
    fields = ["index", "question", "multi-choice options", "answer", "category",
              "l2-category", "target_instances", "image_path"] + letters
    rows = []
    for r in ds:
        idx = str(r["index"])
        img_path = img_dir / f"{idx}.jpg"
        if not img_path.exists():
            raw = base64.b64decode(r["image"])
            img_path.write_bytes(raw)
        row = {
            "index": idx,
            "question": r["question"],
            "multi-choice options": r["multi-choice options"],
            "answer": str(r["answer"]).strip().upper(),
            "category": r["category"],
            "l2-category": r.get("l2-category", r["category"]),
            "target_instances": r.get("target_instances", ""),
            "image_path": str(img_path),
        }
        for L in letters:
            v = str(r.get(L, "") or "")
            row[L] = "" if v.lower() in ("none", "nan") else v
        assert row["answer"] in list("ABCDEF"), f"bad answer {row['answer']!r} at {idx}"
        rows.append(row)
    tsv = _write_tsv(bench, fields, rows)
    _record_manifest(bench, tsv, len(rows))
    print(f"[{bench}] {len(rows)} rows -> {tsv}")


# ---------------------------------------------------------------- ZoomBench
def build_zoombench() -> None:
    import glob
    import pandas as pd
    from huggingface_hub import snapshot_download
    from PIL import Image
    bench = "ZoomBench"
    snap = snapshot_download("inclusionAI/ZoomBench", revision=PINS[bench]["revision"], repo_type="dataset")
    pq = sorted(glob.glob(os.path.join(snap, "**", "*.parquet"), recursive=True))
    assert len(pq) == 1, f"expected 1 parquet, got {pq}"
    df = pd.read_parquet(pq[0])
    _assert_n(bench, len(df))
    assert set(df["question_type"].unique()) == {"mcq", "blank"}
    img_dir = _img_dir(bench)
    fields = ["index", "question", "answer", "question_type", "bbox", "image_path", "crop_image_path"]
    rows = []
    for _, r in df.iterrows():
        idx = str(r["id"])
        img_path = img_dir / f"{idx}.png"
        if not img_path.exists():
            Image.open(io.BytesIO(r["image"]["bytes"])).save(img_path)
        # oracle crop saved for the OPTIONAL regional control run only; it is
        # referenced by a separate column and never enters the model prompt.
        crop_path = img_dir / f"crop_{idx}.png"
        if not crop_path.exists():
            Image.open(io.BytesIO(r["crop_image"]["bytes"])).save(crop_path)
        rows.append({
            "index": idx,
            "question": str(r["query"]),
            "answer": str(r["response"]).strip(),
            "question_type": r["question_type"],
            "bbox": str(r["bbox"]),
            "image_path": str(img_path),
            "crop_image_path": str(crop_path),
        })
    n_mcq = sum(1 for x in rows if x["question_type"] == "mcq")
    assert n_mcq == 621 and len(rows) - n_mcq == 224, f"mcq/blank split drift: {n_mcq}/{len(rows)-n_mcq}"
    tsv = _write_tsv(bench, fields, rows)
    _record_manifest(bench, tsv, len(rows), {"n_mcq": n_mcq, "n_blank": len(rows) - n_mcq})
    print(f"[{bench}] {len(rows)} rows (mcq={n_mcq}) -> {tsv}")


# ---------------------------------------------------------------- VisualProbe
def _build_visualprobe_one(bench: str) -> None:
    from huggingface_hub import snapshot_download
    repo = PINS[bench]["source"].split("/")[-1]
    snap = snapshot_download(f"Mini-o3/{repo}", revision=PINS[bench]["revision"], repo_type="dataset")
    # Repo layout: val.json (records) + data/*.jpg (images). No parquet.
    val_json = Path(snap) / "val.json"
    assert val_json.exists(), f"val.json missing in {snap}"
    records = json.loads(val_json.read_text())
    _assert_n(bench, len(records))
    img_dir = _img_dir(bench)
    import shutil
    fields = ["index", "question", "answer", "data_source", "image_path"]
    rows = []
    for r in records:
        doc_id = str(r["doc_id"])
        imgs = r["images"]
        if isinstance(imgs, str):
            imgs = ast.literal_eval(imgs)
        assert len(imgs) == 1, f"{doc_id}: expected 1 image, got {imgs}"
        src = Path(snap) / str(imgs[0])
        # images field looks like 'VisualProbe_Easy/data/xxx.jpg'; fall back to data/
        if not src.exists():
            cand = list(Path(snap).glob(f"**/{Path(str(imgs[0])).name}"))
            assert cand, f"image not found for {doc_id} under {snap}"
            src = cand[0]
        dst = img_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        question = str(r["problem"])
        if question.startswith("<image>"):
            question = question[len("<image>"):].lstrip("\n")
        rows.append({
            "index": doc_id,
            "question": question,
            "answer": str(r["solution"]).strip(),
            "data_source": str(r.get("data_source", "")),
            "image_path": str(dst),
        })
    tsv = _write_tsv(bench, fields, rows)
    _record_manifest(bench, tsv, len(rows))
    print(f"[{bench}] {len(rows)} rows -> {tsv}")


def build_visualprobe() -> None:
    for bench in ("VisualProbe_Easy", "VisualProbe_Medium", "VisualProbe_Hard"):
        _build_visualprobe_one(bench)


# ---------------------------------------------------------------- ReasonMapPlus
REASONMAP_REPO = "fscdc/ReasonMap"


def _github_main_commit(repo: str) -> str:
    import urllib.request
    with urllib.request.urlopen(f"https://api.github.com/repos/{repo}/commits/main", timeout=30) as r:
        return json.loads(r.read().decode())["sha"]


def build_reasonmap_plus() -> None:
    from datasets import load_dataset
    bench = "ReasonMapPlus"
    ds = load_dataset("FSCCS/ReasonMap-Plus", split="test")
    _assert_n(bench, len(ds))
    # The HF parquet carries only metadata; map PNGs live in the GitHub repo.
    commit = _github_main_commit(REASONMAP_REPO)
    img_dir = _img_dir(bench)
    maps_local: dict[str, Path] = {}
    fields = ["index", "question", "answer", "type", "difficulty_city",
              "country", "city", "image_path"]
    rows = []
    for i, r in enumerate(ds):
        fig = str(r["figure"])  # e.g. './maps/portugal/lisboa.png'
        rel = fig.lstrip("./")
        if rel not in maps_local:
            dst = img_dir / f"{r['country']}_{r['city']}.png"
            if not dst.exists():
                import urllib.request
                url = f"https://raw.githubusercontent.com/{REASONMAP_REPO}/{commit}/{rel}"
                urllib.request.urlretrieve(url, dst)
            maps_local[rel] = dst
        rows.append({
            "index": str(i),
            "question": str(r["question"]),
            "answer": str(r["answer"]).strip(),
            "type": str(r["type"]),
            "difficulty_city": str(r["difficulty_city"]).lower(),
            "country": str(r["country"]),
            "city": str(r["city"]),
            "image_path": str(maps_local[rel]),
        })
    tsv = _write_tsv(bench, fields, rows)
    _record_manifest(bench, tsv, len(rows),
                     {"maps_source": f"github:{REASONMAP_REPO}", "maps_commit": commit,
                      "n_maps": len(maps_local)})
    print(f"[{bench}] {len(rows)} rows, {len(maps_local)} maps -> {tsv}")


# ---------------------------------------------------------------- ChartQAPro
def build_chartqapro() -> None:
    from datasets import load_dataset
    bench = "ChartQAPro"
    ds = load_dataset("ahmed-masry/ChartQAPro", split="test")
    _assert_n(bench, len(ds))
    img_dir = _img_dir(bench)
    fields = ["index", "question", "answer", "question_type", "year_flags", "image_path"]

    def _as_list(v):
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        s = str(v)
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [s]

    rows = []
    from PIL import Image
    for i, r in enumerate(ds):
        questions = _as_list(r["Question"])
        answers = _as_list(r["Answer"])
        years = _as_list(r["Year"])
        idx = str(i)
        img_path = img_dir / f"{idx}.png"
        if not img_path.exists():
            img = r["image"]
            if isinstance(img, (bytes, bytearray)):
                img = Image.open(io.BytesIO(img))
            elif isinstance(img, dict) and "bytes" in img:
                img = Image.open(io.BytesIO(img["bytes"]))
            img.save(img_path)
        rows.append({
            "index": idx,
            # multi-turn conversational context preserved verbatim, newline-joined
            "question": "\n".join(q.strip() for q in questions),
            "answer": answers[-1].strip().strip(".").strip("\n"),
            "question_type": str(r["Question Type"]),
            "year_flags": "|".join(y.strip() for y in years),
            "image_path": str(img_path),
        })
    tsv = _write_tsv(bench, fields, rows)
    _record_manifest(bench, tsv, len(rows))
    print(f"[{bench}] {len(rows)} rows -> {tsv}")


# ---------------------------------------------------------------- InfographicVQA
def build_infographicvqa() -> None:
    import pandas as pd
    bench = "InfographicVQA_val"
    url = PINS[bench]["source"]
    raw_path = LMUDATA / "_downloads" / "InfoVQA_VAL.tsv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        import urllib.request
        print(f"[{bench}] downloading {url} ...")
        urllib.request.urlretrieve(url, raw_path)
    # Integrity gate: the opencompass cert is flaky, so the official MD5 is the
    # Verify the pinned source checksum.
    import hashlib
    got_md5 = hashlib.md5(raw_path.read_bytes()).hexdigest()
    want_md5 = PINS[bench]["revision"].split("md5:")[-1]
    assert got_md5 == want_md5, f"{bench}: source md5 {got_md5} != pinned {want_md5}"
    df = pd.read_csv(raw_path, sep="\t")
    _assert_n(bench, len(df))
    assert "question" in df.columns and "answer" in df.columns, f"unexpected columns: {list(df.columns)}"

    # GT/order/answers: pinned opencompass TSV (md5-verified above).
    # Images: mm-eval/InfographicVQA validation split (embedded media), matched
    # by exact question text; 2801/2801 coverage required.
    from datasets import load_dataset
    img_ds = load_dataset("mm-eval/InfographicVQA", split="validation")
    assert len(img_ds) == PINS[bench]["n"], f"image source rows {len(img_ds)} != {PINS[bench]['n']}"
    q2media: dict[str, object] = {}
    for r in img_ds:
        msgs = r["messages"]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        q = msgs[0]["question"] if isinstance(msgs, (list, tuple)) else None
        assert q, f"bad messages row: {str(msgs)[:120]}"
        q2media.setdefault(q.strip(), r["media"])
    img_dir = _img_dir(bench)
    fields = ["index", "question", "answers", "image_path"]
    rows = []
    n_img_base64 = n_img_mmeval = 0
    from vlmeval.smp import decode_base64_to_image_file  # reuse VLMEvalKit decoder
    for _, r in df.iterrows():
        idx = str(r["index"])
        img_path = img_dir / f"{idx}.png"
        q = str(r["question"]).strip()
        if not img_path.exists():
            img_val = str(r.get("image", ""))
            if len(img_val) > 500:  # embedded base64 in the TSV (500 rows)
                decode_base64_to_image_file(r["image"], str(img_path))
                n_img_base64 += 1
            else:
                media = q2media.get(q)
                assert media is not None, f"no image-source match for question: {q[:80]!r}"
                if isinstance(media, (list, tuple)):
                    assert len(media) == 1, f"multi-image row for {q[:60]!r}: {len(media)}"
                    media = media[0]
                media.save(img_path)
                n_img_mmeval += 1
        ans = r["answer"]
        if isinstance(ans, str) and ans.startswith("["):
            try:
                parsed = ast.literal_eval(ans)
                answers = [str(a) for a in parsed] if isinstance(parsed, (list, tuple)) else [ans]
            except Exception:
                answers = [ans]
        else:
            answers = [str(ans)]
        rows.append({
            "index": idx,
            "question": str(r["question"]),
            "answers": "|".join(a.strip() for a in answers),
            "image_path": str(img_path),
        })
    tsv = _write_tsv(bench, fields, rows)
    _record_manifest(bench, tsv, len(rows), {
        "source_md5": got_md5,
        "image_source": "hf://datasets/mm-eval/InfographicVQA:validation (matched by question text)",
        "n_images_from_tsv_base64": n_img_base64,
        "n_images_from_mmeval": n_img_mmeval,
    })
    print(f"[{bench}] {len(rows)} rows (imgs: {n_img_base64} tsv + {n_img_mmeval} mmeval) -> {tsv}")


BUILDERS = {
    "treebench": build_treebench,
    "zoombench": build_zoombench,
    "visualprobe": build_visualprobe,
    "reasonmap_plus": build_reasonmap_plus,
    "chartqapro": build_chartqapro,
    "infographicvqa": build_infographicvqa,
}


# ================================================== Additional datasets
# Additional benchmark builders.
# These builders use a separate manifest with the same TSV and image format.
# Source counts and checksums are asserted; any mismatch aborts the build.

GCEP4_BENCHMARKS = ("PerceptionBench", "HallusionBench_GCEP", "MathVerseVO", "VisuLogic_GCEP")
GCEP4_MANIFEST_PATH = LMUDATA / "gcep4_dataset_manifest.jsonl"

HALLUSIONBENCH_ROOT = Path(os.environ["HALLUSIONBENCH_ROOT"]).expanduser() if os.environ.get("HALLUSIONBENCH_ROOT") else None

GCEP4_PINS = {
    "PerceptionBench": {
        "source": "hf://datasets/moonshotai/PerceptionBench",
        "revision": "6ba8c3135c7675ad6a5c141536a86b9460c70960",
        "n": 3000, "split": "test",
        "jsonl_sha256": "f84afb46c3c150b572481ca34b351d6afb1c31b98fef0c0c46d7659333573c47",
        "jsonl_bytes": 1626804092,
    },
    # registry/TSV name suffixed _GCEP: VLMEvalKit's image_yorn registry owns
    # "HallusionBench" and would md5-clobber a same-named local TSV.
    "HallusionBench_GCEP": {
        "source": "gh://tianyi-lab/HallusionBench",
        "revision": "744007c232c292942c7f80eb61edb2465482da31",
        "n": 1129, "split": "test",
        "json_sha256": "ca6e0fb677dd56e5fb0e2700d06c58d01f8a6ae66e0f1773b2ac48c0a095d051",
        "image_mirror": "hf://datasets/rayguan/HallusionBench@2067dfdce5abc7efdbb0927f4721ae282f0b6c51",
        "image_recover": "hf://datasets/mm-eval/HallusionBench@b91d8e365f9bcf78fc6c2d05d06d0068008d41e8",
    },
    "MathVerseVO": {
        "source": "hf://datasets/AI4Math/MathVerse",
        "revision": "3bc86196678bad115a923d2851c6821dbe235939",
        "n": 788, "split": "testmini",
    },
    "VisuLogic_GCEP": {
        "source": "hf://datasets/VisuLogic/VisuLogic",
        "revision": "3e483f8cca0bc766ecc5f70299f2b8a34b5035ca",
        "n": 1000, "split": "test",
    },
}
PINS.update(GCEP4_PINS)

# Official VisuLogic COT_PROMPT, verbatim from
# third_party/VisuLogic-Eval/models/prompts.py @00fba6dd (models/prompts.py:1).
VISULOGIC_COT_PROMPT = ("Solve the complex visual logical reasoning problem through "
                        "step-by-step reasoning. Think about the reasoning process first "
                        "and answer the question following this format: Answer: \\boxed{$LETTER}.")


def _record_manifest_gcep4(bench: str, tsv_path: str, n_rows: int, extra: dict | None = None) -> None:
    pin = GCEP4_PINS[bench]
    manifests.append_jsonl(str(GCEP4_MANIFEST_PATH), [{
        "schema_version": "1",
        "created_at": manifests.utc_now_iso(),
        "benchmark_id": bench,
        "source": pin["source"],
        "revision": pin["revision"],
        "split": pin["split"],
        "n_rows": n_rows,
        "tsv_path": tsv_path,
        "tsv_sha256": manifests.sha256_file(tsv_path),
        **(extra or {}),
    }])


# ---------------------------------------------------------- PerceptionBench
def build_perceptionbench() -> None:
    bench = "PerceptionBench"
    pin = GCEP4_PINS[bench]
    from huggingface_hub import snapshot_download
    snap = snapshot_download("moonshotai/PerceptionBench", repo_type="dataset",
                             revision=pin["revision"], allow_patterns=["PerceptionBench.jsonl"])
    src = Path(snap) / "PerceptionBench.jsonl"
    assert src.stat().st_size == pin["jsonl_bytes"], f"jsonl size mismatch: {src.stat().st_size}"
    assert manifests.sha256_file(str(src)) == pin["jsonl_sha256"], "jsonl sha256 mismatch"

    img_dir = _img_dir(bench)
    rows, idxs = [], set()
    imgdist: dict[int, int] = {}
    slots = 0
    na = 0
    cats: dict[str, int] = {}
    ph_re = re.compile(r"<\|image_(\d+)\|>")
    with open(src, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            idx = int(r["index"])
            idxs.add(idx)
            imgs = r["image"] or []
            assert isinstance(imgs, list) and imgs, f"row {idx}: empty image list"
            paths = []
            for k, data_uri in enumerate(imgs, start=1):
                m = re.match(r"data:image/(jpeg|jpg|png|webp);base64,(.*)", data_uri, re.DOTALL)
                assert m, f"row {idx} image {k}: unexpected data-uri prefix"
                ext = {"jpeg": "jpg", "jpg": "jpg"}.get(m.group(1), m.group(1))
                p = img_dir / f"{idx}_{k}.{ext}"
                if not p.exists():
                    p.write_bytes(base64.b64decode(m.group(2)))
                paths.append(str(p))
            phs = {int(x) for x in ph_re.findall(r["problem"])}
            assert not phs or max(phs) <= len(imgs), f"row {idx}: placeholder out of range"
            assert not phs or phs == set(range(1, len(imgs) + 1)), f"row {idx}: unreferenced images"
            imgdist[len(imgs)] = imgdist.get(len(imgs), 0) + 1
            slots += len(imgs)
            na += int(r["source_bmk"] == "NA")
            cats[r["error_category"]] = cats.get(r["error_category"], 0) + 1
            rows.append({
                "index": idx,
                "question": r["problem"],          # verbatim, keeps <|image_N|> placeholders
                "answer": str(r["answer"]),
                "error_category": r["error_category"],
                "source_bmk": r["source_bmk"],
                "source_idx": str(r.get("source_idx", "")),
                "n_images": len(imgs),
                "image_path": ";".join(paths),
                # NOTE: `hint` is deliberately dropped (the official evaluator ignores it;
                # it must never reach the model).
            })
    _assert_n(bench, len(rows))
    assert len(idxs) == 3000 and min(idxs) == 0 and max(idxs) == 2999
    assert na == 1200 and len(rows) - na == 1800
    expect_dist = {1: 2652, 2: 245, 3: 34, 4: 41, 5: 20, 6: 3, 8: 5}
    assert imgdist == expect_dist, f"image count distribution mismatch: {imgdist}"
    assert slots == 3566
    assert sum(v for k, v in imgdist.items() if k >= 2) == 348
    assert len(cats) == 10, f"expected 10 categories, got {len(cats)}"
    rows.sort(key=lambda x: x["index"])
    fields = ["index", "question", "answer", "error_category", "source_bmk",
              "source_idx", "n_images", "image_path"]
    tsv = _write_tsv(bench, fields, rows)
    _record_manifest_gcep4(bench, tsv, len(rows), {
        "jsonl_sha256": pin["jsonl_sha256"],
        "image_count_distribution": imgdist,
        "total_image_slots": slots,
        "multi_image_samples": 348,
        "categories": cats,
    })
    print(f"[{bench}] {len(rows)} rows ({slots} images) -> {tsv}")


# ---------------------------------------------------------- HallusionBench
def build_hallusionbench() -> None:
    bench = "HallusionBench_GCEP"
    pin = GCEP4_PINS[bench]
    if HALLUSIONBENCH_ROOT is None:
        raise RuntimeError("Set HALLUSIONBENCH_ROOT to the directory containing the pinned "
                           "HallusionBench.json and, when needed, recovered_images/. "
                           "See README for the unresolved recovery-material requirement.")
    src_json = HALLUSIONBENCH_ROOT / "HallusionBench.json"
    if not src_json.is_file():
        raise FileNotFoundError(f"Missing pinned HallusionBench metadata: {src_json}")
    assert manifests.sha256_file(str(src_json)) == pin["json_sha256"], "HallusionBench.json sha mismatch"
    data = json.loads(src_json.read_text())
    _assert_n(bench, len(data))

    img_root = _img_dir("HallusionBench")  # shared image dir (already populated)
    recovered = HALLUSIONBENCH_ROOT / "recovered_images"
    missing_src: list[str] = []
    rows = []
    cnt = {"VD": 0, "VS": 0, "vi0": 0, "vi1": 0, "vi2": 0}
    fig_files: set[str] = set()
    for i, r in enumerate(data):
        vi = str(r["visual_input"])
        cat = r["category"]
        cnt[cat] += 1
        cnt[f"vi{vi}"] += 1
        if vi == "0":
            assert r["filename"] in (None, "None", ""), f"row {i}: text-only row with filename"
            # "[]" literal: VLMEvalKit toliststr() parses it to an empty list
            # (empty string becomes pandas NaN and crashes toliststr at load).
            img_path = "[]"
        else:
            rel = r["filename"][2:] if r["filename"].startswith("./") else r["filename"]
            fig_files.add(rel)
            dst = img_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                src_img = _hb_mirror_root() / "data" / rel
                if src_img.exists():
                    dst.write_bytes(src_img.read_bytes())
                else:
                    rec = recovered / rel.split("/")[-1]
                    assert rec.exists(), (f"No verified source for image {rel}. Supply the matching original "
                                          f"recovery image under {recovered}; this release does not "
                                          "include the recovery procedure.")
                    dst.write_bytes(rec.read_bytes())
                    missing_src.append(rel)
            img_path = str(dst)
        rows.append({
            "index": i,
            "question": r["question"],
            "answer": str(r["gt_answer"]),
            "gt_answer_details": r["gt_answer_details"],
            "category": cat,
            "subcategory": r["subcategory"],
            "set_id": str(r["set_id"]),
            "figure_id": str(r["figure_id"]),
            "question_id": str(r["question_id"]),
            "visual_input": vi,
            "sample_note": r.get("sample_note", ""),
            "image_path": img_path,
        })
    assert cnt["VD"] == 591 and cnt["VS"] == 538
    assert cnt["vi0"] == 178 and cnt["vi1"] == 447 and cnt["vi2"] == 504
    assert len(fig_files) == 346, f"expected 346 physical figures, got {len(fig_files)}"
    q_groups = {(r["category"], r["subcategory"], r["set_id"], r["question_id"]) for r in rows}
    assert len(q_groups) == 455, f"expected 455 question groups, got {len(q_groups)}"
    f_groups = {(r["category"], r["subcategory"], r["set_id"], r["figure_id"]) for r in rows
                if not (r["category"] == "VS" and r["figure_id"] == "0")}
    assert len(f_groups) == 346, f"expected 346 figure groups, got {len(f_groups)}"
    vi1_figs = {r["image_path"] for r in rows if r["visual_input"] == "1"}
    vi2_figs = {r["image_path"] for r in rows if r["visual_input"] == "2"}
    assert len(vi1_figs) == 165 and len(vi2_figs) == 181
    # every referenced image must be decodable
    from PIL import Image
    for p in fig_files:
        with Image.open(img_root / p) as im:
            im.verify()
    fields = ["index", "question", "answer", "gt_answer_details", "category",
              "subcategory", "set_id", "figure_id", "question_id", "visual_input",
              "sample_note", "image_path"]
    tsv = _write_tsv(bench, fields, rows)
    _record_manifest_gcep4(bench, tsv, len(rows), {
        "json_sha256": pin["json_sha256"],
        "n_figures": 346,
        "n_question_groups": 455,
        "n_images_recovered_from_mmeval": len(set(missing_src)),
        "recovered_images": sorted(set(missing_src)),
    })
    print(f"[{bench}] {len(rows)} rows (346 figures, {len(set(missing_src))} recovered) -> {tsv}")


def _hb_mirror_root() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download("rayguan/HallusionBench", repo_type="dataset",
                                  revision="2067dfdce5abc7efdbb0927f4721ae282f0b6c51"))


# ------------------------------------------------------------ MathVerse VO
def build_mathverse_vo() -> None:
    from datasets import load_dataset
    bench = "MathVerseVO"
    pin = GCEP4_PINS[bench]
    ds = load_dataset("AI4Math/MathVerse", "testmini", revision=pin["revision"], split="testmini")
    assert len(ds) == 3940, f"testmini size drift: {len(ds)}"
    vo = [r for r in ds if r["problem_version"] == "Vision Only"]
    _assert_n(bench, len(vo))
    assert len({r["sample_index"] for r in vo}) == 788
    assert len({r["problem_index"] for r in vo}) == 788
    img_dir = _img_dir(bench)
    rows = []
    qt: dict[str, int] = {}
    for i, r in enumerate(vo):
        q = r["query_wo"]
        assert q and q.strip(), f"row {i}: empty query_wo"
        p = img_dir / f"{r['sample_index']}.png"
        if not p.exists():
            r["image"].convert("RGB").save(p)
        qt[r["question_type"]] = qt.get(r["question_type"], 0) + 1
        meta = r["metadata"] or {}
        rows.append({
            "index": i,
            "sample_index": str(r["sample_index"]),
            "problem_index": str(r["problem_index"]),
            "question": q,                               # model input
            "question_for_eval": r.get("question_for_eval", ""),  # judge context only
            "answer": str(r["answer"]),
            "question_type": r["question_type"],
            "subject": str(meta.get("subject", "")),
            "subfield": str(meta.get("subfield", "")),
            "image_path": str(p),
        })
    assert qt.get("multi-choice") == 436 and qt.get("free-form") == 352, f"question_type drift: {qt}"
    fields = ["index", "sample_index", "problem_index", "question", "question_for_eval",
              "answer", "question_type", "subject", "subfield", "image_path"]
    tsv = _write_tsv(bench, fields, rows)
    _record_manifest_gcep4(bench, tsv, len(rows), {"question_type": qt})
    print(f"[{bench}] {len(rows)} rows -> {tsv}")


# ------------------------------------------------------------ VisuLogic
def build_visulogic_gcep() -> None:
    import zipfile
    bench = "VisuLogic_GCEP"
    pin = GCEP4_PINS[bench]
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download("VisuLogic/VisuLogic", repo_type="dataset",
                                  revision=pin["revision"]))
    objs = []
    with open(snap / "data.jsonl", encoding="utf-8") as f:  # robust: no wc -l
        for line in f:
            line = line.strip()
            if line:
                objs.append(json.loads(line))
    _assert_n(bench, len(objs))
    ids = [o["id"] for o in objs]
    assert len(set(ids)) == 1000 and min(ids) == "00000" and max(ids) == "00999"
    labels: dict[str, int] = {}
    cats: dict[str, int] = {}
    img_dir = _img_dir(bench)
    with zipfile.ZipFile(snap / "images.zip") as z:
        assert z.testzip() is None, "images.zip corrupt"
        names = set(z.namelist())
        rows = []
        for i, o in enumerate(objs):
            assert o["label"] in list("ABCD"), f"row {i}: bad label {o['label']!r}"
            labels[o["label"]] = labels.get(o["label"], 0) + 1
            cats[o["tag"]] = cats.get(o["tag"], 0) + 1
            member = o["image_path"]  # images/00000.png
            assert member in names, f"missing zip member {member}"
            dst = img_dir / f"{o['id']}.png"
            if not dst.exists():
                dst.write_bytes(z.read(member))
            rows.append({
                "index": i,
                "question": o["question"] + "\n" + VISULOGIC_COT_PROMPT,  # official COT_PROMPT baked
                "answer": o["label"],
                "category": o["tag"],
                "image_path": str(dst),
            })
    assert labels == {"A": 231, "B": 267, "C": 252, "D": 250}, f"label drift: {labels}"
    expect_cats = {"Quantitative Reasoning": 353, "Spatial Reasoning": 231,
                   "Positional Reasoning": 136, "Attribute Reasoning": 82,
                   "Stylistic Reasoning": 90, "Other": 108}
    assert cats == expect_cats, f"category drift: {cats}"
    fields = ["index", "question", "answer", "category", "image_path"]
    tsv = _write_tsv(bench, fields, rows)
    _record_manifest_gcep4(bench, tsv, len(rows), {"labels": labels, "categories": cats})
    print(f"[{bench}] {len(rows)} rows -> {tsv}")


GCEP4_BUILDERS = {
    "perceptionbench": build_perceptionbench,
    "hallusionbench": build_hallusionbench,
    "mathverse_vo": build_mathverse_vo,
    "visulogic_gcep": build_visulogic_gcep,
}
BUILDERS.update(GCEP4_BUILDERS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=list(BUILDERS) + ["all"])
    args = ap.parse_args()
    names = list(BUILDERS) if args.bench == "all" else [args.bench]
    for name in names:
        BUILDERS[name]()


if __name__ == "__main__":
    main()
