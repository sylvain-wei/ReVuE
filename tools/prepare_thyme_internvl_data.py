#!/usr/bin/env python3
# Modified for anonymous review: explicit paths and fail-fast input/output checks.
"""Prepare Thyme-SFT data for InternVL3.5 two-stage SFT: route + validate + shard.
Stage1=single_round+2round, Stage2=computation. Keep source image values unchanged. Output written via HF datasets (Features metadata) so swift's
ResponsePreprocessor loads it without the ArrowNotImplementedError that plain
pyarrow parquet triggers on the nested base64 image list. Structural gate only:
#images==<image> count, balanced tags, non-empty, has <answer>, length guard."""
import argparse, glob, json, os
import pyarrow.parquet as pq
from datasets import Dataset, Features, Value, Sequence
STAGE1 = ["wo_thinking_thyme_single_round", "2round"]
STAGE2 = ["computation"]
PAIRS = [("<code>","</code>"),("<sandbox_output>","</sandbox_output>"),("<answer>","</answer>"),("<think>","</think>")]
FEATURES = Features({"image": Sequence(Value("string")), "question": Value("string"), "response": Value("string")})

def n_images(row):
	img = row.get("image")
	if img is None:
		return 0
	return len(img) if isinstance(img, list) else 1

def validate(row, max_chars):
	q = str(row.get("question") or "")
	r = str(row.get("response") or "")
	both = q + r
	if not r.strip():
		return "empty_response"
	if n_images(row) != both.count("<image>"):
		return "img_mismatch"
	for a, b in PAIRS:
		if both.count(a) != both.count(b):
			return "unbalanced_" + a.strip("<>")
	if "<answer>" not in r:
		return "no_answer"
	if len(both) > max_chars:
		return "too_long"
	return None

def collect_split(raw_root, split, stats, max_chars, limit):
	files = sorted(glob.glob(os.path.join(raw_root, split + "-*.parquet")))
	recs, kept, seen, dropped = [], 0, 0, {}
	for f in files:
		for row in pq.read_table(f).to_pylist():
			seen += 1
			reason = validate(row, max_chars)
			if reason is None:
				img = row.get("image") or []
				recs.append({"image": list(img) if isinstance(img, list) else [img], "question": row.get("question") or "", "response": row.get("response") or ""})
				kept += 1
			else:
				dropped[reason] = dropped.get(reason, 0) + 1
			if limit and kept >= limit:
				break
		if limit and kept >= limit:
			break
	stats["splits"][split] = {"seen": seen, "kept": kept, "dropped": dropped}
	return recs

def write_stage(name, splits, raw_root, out_root, stats, max_chars, limit, shard_rows):
	out_dir = os.path.join(out_root, name)
	os.makedirs(out_dir, exist_ok=True)
	recs = []
	for sp in splits:
		recs.extend(collect_split(raw_root, sp, stats, max_chars, limit))
	if not recs:
		raise RuntimeError(f"No valid rows remain for {name}; no complete dataset was built")
	n = len(recs)
	shard = 0
	for start in range(0, n, shard_rows):
		chunk = recs[start:start + shard_rows]
		ds = Dataset.from_list(chunk, features=FEATURES)
		ds.to_parquet(os.path.join(out_dir, "%s-%05d.parquet" % (name, shard)))
		shard += 1
	stats["stages"][name] = {"rows": n, "shards": shard, "dir": out_dir}
	print("STAGE %s: %d rows, %d shards -> %s" % (name, n, shard, out_dir))
	return n

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--raw-root", required=True, help="Directory containing the source parquet shards")
	ap.add_argument("--out-root", required=True, help="New or empty output directory")
	ap.add_argument("--max-chars", type=int, default=60000)
	ap.add_argument("--limit-per-split", type=int, default=0)
	ap.add_argument("--shard-rows", type=int, default=2000)
	args = ap.parse_args()
	if args.max_chars <= 0 or args.shard_rows <= 0 or args.limit_per_split < 0:
		ap.error("max-chars and shard-rows must be positive; limit-per-split must be nonnegative")
	for split in STAGE1 + STAGE2:
		if not glob.glob(os.path.join(args.raw_root, split + "-*.parquet")):
			ap.error(f"No source shards found for {split} under {args.raw_root}")
	if os.path.exists(args.out_root) and (not os.path.isdir(args.out_root) or os.listdir(args.out_root)):
		ap.error("out-root must be new or empty; use a new directory to avoid mixing shards")
	os.makedirs(args.out_root, exist_ok=True)
	stats = {"stages": {}, "splits": {}, "args": vars(args)}
	write_stage("stage1", STAGE1, args.raw_root, args.out_root, stats, args.max_chars, args.limit_per_split, args.shard_rows)
	write_stage("stage2", STAGE2, args.raw_root, args.out_root, stats, args.max_chars, args.limit_per_split, args.shard_rows)
	with open(os.path.join(args.out_root, "prepare_stats.json"), "w") as fh:
		json.dump(stats, fh, indent=2)
	print(json.dumps(stats["splits"], indent=2))

if __name__ == "__main__":
	main()
