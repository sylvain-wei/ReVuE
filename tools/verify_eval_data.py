#!/usr/bin/env python3
"""Verify (and optionally re-hash) an LMUData evaluation tree against the pinned manifests.

Why this exists: the artifact needs a reviewer-checkable statement of *which* evaluation data
backed the paper. The manifests (`data_prep/gcep{4,6}_dataset_manifest.jsonl`) record, per
benchmark, the upstream source + revision + split, the expected row count, the TSV basename and
the TSV SHA-256. This tool checks a real `$LMUData` tree against them.

Implementation detail that matters: several of these TSVs contain **quoted fields with embedded
newlines** (ChartQAPro, MathVerseVO, VisuLogic, …), so `wc -l` **over-counts** rows. Rows are
counted with the `csv` module here, never by lines.

Checks per benchmark:
  * TSV present at the recorded basename under $LMUData
  * row count (csv-parsed) == manifest `n_rows`
  * SHA-256 (bytes) == manifest `tsv_sha256`
  * every `image_path` referenced by the TSV exists on disk (counts missing files)
  * reports the legacy `HallusionBench.tsv` clobber hazard

Usage:
    python3 verify_eval_data.py --lmudata <LMUData> [--manifest-dir <dir>] [--json out.json]
    python3 verify_eval_data.py --lmudata <LMUData> --write-hash-table /tmp/local_tsv_hash.tsv
"""
# Modified for anonymous review: portable manifest path and strict malformed-input handling.
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import sys
from collections import OrderedDict


def sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def tsv_stats(path: str):
    """(rows_excluding_header, columns, declared_image_paths, rows_without_image, sha256).

    `image_path` is a plain path in most TSVs but a **stringified list** in a few (the
    HallusionBench_* files), and some rows legitimately declare an empty list. Both forms are
    parsed here; `rows_without_image` counts the empty-list rows so that they are not reported
    as missing files.
    """
    # some TSVs carry very long single-line fields (multi-line questions, long options)
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:                                    # pragma: no cover - platform dependent
        csv.field_size_limit(2 ** 31 - 1)
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        try:
            cols = next(rd)
        except StopIteration:
            return 0, [], [], 0, sha256(path)
        idx = cols.index("image_path") if "image_path" in cols else None
        n = n_noimg = 0
        imgs: list[str] = []
        for row in rd:
            n += 1
            if idx is None or idx >= len(row) or not row[idx].strip():
                n_noimg += 1
                continue
            raw = row[idx].strip()
            if raw.startswith("["):
                try:
                    vals = ast.literal_eval(raw)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(f"Malformed image_path list in {path}, data row {n}") from exc
                if not isinstance(vals, list) or any(not isinstance(v, str) for v in vals):
                    raise ValueError(f"image_path must be a string list in {path}, data row {n}")
                if not vals:
                    n_noimg += 1
                    continue
                imgs.extend(str(v) for v in vals)
            else:
                imgs.append(raw)
    return n, cols, imgs, n_noimg, sha256(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmudata", default=os.environ.get("LMUData", ""))
    ap.add_argument("--manifest-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "data_prep"))
    ap.add_argument("--benchmarks", default="", help="Comma-separated benchmark IDs; default: all recorded entries")
    ap.add_argument("--json", default="")
    ap.add_argument("--write-hash-table", default="")
    args = ap.parse_args()
    if not args.lmudata:
        print("need --lmudata (or $LMUData)")
        return 2

    entries: "OrderedDict[str, list]" = OrderedDict()
    for name in ("gcep4_dataset_manifest.jsonl", "gcep6_dataset_manifest.jsonl"):
        p = os.path.join(args.manifest_dir, name)
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    entries.setdefault(d["benchmark_id"], []).append(d)

    if not entries:
        ap.error("No benchmark entries found in manifest-dir")
    if args.benchmarks:
        selected = list(dict.fromkeys(x.strip() for x in args.benchmarks.split(",") if x.strip()))
        missing = set(selected) - entries.keys()
        if not selected or missing:
            ap.error(f"Empty selection or unknown benchmark IDs: {sorted(missing)}")
        entries = OrderedDict((key, entries[key]) for key in selected)

    rows, hash_rows, ok = [], [], True
    for bench, variants in entries.items():
        d = variants[-1]                      # the last recorded build is the one to check
        tsv = os.path.join(args.lmudata, os.path.basename(d["tsv_path"]))
        rec = {"benchmark": bench, "tsv": os.path.basename(d["tsv_path"]),
               "expected_rows": d["n_rows"], "source": d.get("source", ""),
               "revision": str(d.get("revision", "")), "split": d.get("split", "")}
        if not os.path.exists(tsv):
            rec["status"] = "TSV_MISSING"
            ok = False
        else:
            n, cols, imgs, n_noimg, h = tsv_stats(tsv)
            missing = sum(1 for p in imgs if not os.path.exists(p if os.path.isabs(p) else os.path.join(args.lmudata, p)))
            rec.update(rows=n, columns=cols[:8], sha256=h,
                       rows_match=(n == d["n_rows"]),
                       sha_match=(h == str(d.get("tsv_sha256"))),
                       image_refs=len(imgs), rows_without_image=n_noimg,
                       images_missing=missing)
            rec["status"] = ("OK" if (n == d["n_rows"] and h == str(d.get("tsv_sha256"))
                                      and missing == 0) else "MISMATCH")
            if rec["status"] != "OK":
                ok = False
            hash_rows.append([os.path.basename(tsv), n, os.path.getsize(tsv), h])
        rows.append(rec)

    legacy = os.path.join(args.lmudata, "HallusionBench.tsv")
    warn = ("legacy HallusionBench.tsv is present; VLMEvalKit's md5 registry replaces it at load "
            "time -- always use the suffixed HallusionBench_GCEP") if os.path.exists(legacy) else ""

    out = {"lmudata": os.path.abspath(args.lmudata), "n_benchmarks": len(rows), "all_ok": ok,
           "legacy_warning": warn, "benchmarks": rows}
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
    if args.write_hash_table:
        if os.path.abspath(args.write_hash_table) == os.path.abspath(os.path.join(args.manifest_dir, "tsv_hash.tsv")):
            ap.error("Write local hashes to a new file; preserve the recorded reference hash table")
        with open(args.write_hash_table, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["file", "rows", "bytes", "sha256"])
            w.writerows(hash_rows)
        print(f"wrote {args.write_hash_table} ({len(hash_rows)} benchmarks)")

    for r in rows:
        print(f"{r['status']:12s} {r['benchmark']:22s} rows={r.get('rows')}/{r['expected_rows']} "
              f"img_refs={r.get('image_refs')} (no-image rows={r.get('rows_without_image')}) "
              f"missing={r.get('images_missing')}")
    if warn:
        print("WARNING:", warn)
    print("ALL OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
