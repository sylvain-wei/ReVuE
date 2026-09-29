#!/usr/bin/env python3
# Modified for anonymous review: explicit model path and meaningful audit exit status.
"""Independent label-audit tool for InternVL3.5-4B Thyme-style SFT data.

Verifies (1) static data integrity, (2) swift training-path label masking
(loss on labels!=-100), (3) temporal image ordering / no future-image leakage.
CPU-only, load_model=False. Does NOT touch GPU, training scripts, or swift code.
"""
import os
import sys
import glob
import json
import argparse
import random
import base64
from io import BytesIO
from collections import Counter

SEED = 42

PROTOCOL_TAGS = [
	("<code>", "</code>"),
	("<sandbox_output>", "</sandbox_output>"),
	("<answer>", "</answer>"),
]

import glob as _glob
import pyarrow.parquet as _pq


class ShardedDataset:
	"""Lazy row access across many parquet shards (avoids pandas nested-list concat bug)."""

	def __init__(self, stage_dir):
		self.files = sorted(_glob.glob(stage_dir.rstrip("/") + "/*.parquet"))
		assert self.files, "no parquet shards found"
		self._counts = []
		for f in self.files:
			self._counts.append(_pq.ParquetFile(f).metadata.num_rows)
		self._cum = []
		acc = 0
		for c in self._counts:
			acc += c
			self._cum.append(acc)
		self._total = acc
		self._cache_shard = -1
		self._cache_rows = None

	def __len__(self):
		return self._total

	def _shard_of(self, idx):
		lo = 0
		for si, cend in enumerate(self._cum):
			if idx < cend:
				return si, idx - lo
			lo = cend
		raise IndexError(idx)

	def _load_shard(self, si):
		if self._cache_shard != si:
			pf = _pq.ParquetFile(self.files[si])
			rows = []
			# small batch_size avoids pyarrow "Nested data conversions not
			# implemented for chunked array outputs" on the base64 image list column.
			for batch in pf.iter_batches(batch_size=64):
				rows.extend(batch.to_pylist())
			self._cache_rows = rows
			self._cache_shard = si
		return self._cache_rows

	def row(self, idx):
		si, off = self._shard_of(idx)
		return self._load_shard(si)[off]

	def iter_rows(self, limit=0):
		n = self._total if limit <= 0 else min(limit, self._total)
		yielded = 0
		for si in range(len(self.files)):
			if yielded >= n:
				break
			rows = self._load_shard(si)
			for r in rows:
				if yielded >= n:
					break
				yield r
				yielded += 1



# ---------- helpers ----------
def b64_to_image(b64str):
	"""Decode a base64 string into a PIL image; return (ok, mode_size_or_err)."""
	try:
		from PIL import Image
		raw = base64.b64decode(b64str)
		img = Image.open(BytesIO(raw))
		img.load()
		return True, f"{img.mode}{img.size}"
	except Exception as e:
		return False, repr(e)[:120]


def tags_balanced(text):
	"""Check every protocol open tag has a matching close and counts match."""
	for open_t, close_t in PROTOCOL_TAGS:
		if text.count(open_t) != text.count(close_t):
			return False, f"{open_t}={text.count(open_t)} {close_t}={text.count(close_t)}"
	return True, ""


def image_positions(text):
	"""Return list of char offsets of every <image> placeholder in text order."""
	pos = []
	start = 0
	while True:
		i = text.find("<image>", start)
		if i < 0:
			break
		pos.append(i)
		start = i + len("<image>")
	return pos


# ---------- static check ----------
def static_check(ds, max_static):
	n = len(ds) if max_static <= 0 else min(max_static, len(ds))
	stats = {
		"rows_checked": n,
		"image_decode_fail": 0,
		"image_count_mismatch": 0,
		"tags_unbalanced": 0,
		"empty_response": 0,
		"missing_answer": 0,
	}
	failures = []
	for idx, r in enumerate(ds.iter_rows(n)):
		imgs = list(r["image"]) if r["image"] is not None else []
		q = r["question"] or ""
		resp = r["response"] or ""
		full = q + "\n" + resp
		# image count vs <image> placeholders (question is where originals live)
		n_img_ph = full.count("<image>")
		if n_img_ph != len(imgs):
			# processed images in response also carry <image>; expected total == len(imgs)
			stats["image_count_mismatch"] += 1
			if len(failures) < 40:
				failures.append({"idx": idx, "why": "image_count", "n_ph": n_img_ph, "n_img": len(imgs)})
		# decode each image
		for j, b in enumerate(imgs):
			ok, info = b64_to_image(b)
			if not ok:
				stats["image_decode_fail"] += 1
				if len(failures) < 40:
					failures.append({"idx": idx, "why": "decode", "img_j": j, "err": info})
				break
		# tags balanced over response
		ok, why = tags_balanced(resp)
		if not ok:
			stats["tags_unbalanced"] += 1
			if len(failures) < 40:
				failures.append({"idx": idx, "why": "tags", "detail": why})
		if not resp.strip():
			stats["empty_response"] += 1
		if "<answer>" not in resp:
			stats["missing_answer"] += 1
			if len(failures) < 40:
				failures.append({"idx": idx, "why": "no_answer"})
	return stats, failures


# ---------- swift template ----------
def load_template(model_path):
	"""Load the InternVL3.5 swift template with NO weights (CPU, tokenizer/processor only)."""
	from swift.llm import get_model_tokenizer, get_template
	model, processor = get_model_tokenizer(model_path, load_model=False)
	assert model is None, "load_model=False must yield model None"
	tmpl_type = processor.model_meta.template or "internvl2_5"
	template = get_template(tmpl_type, processor)
	template.set_mode("train")
	tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
	return template, tok, tmpl_type


def row_to_inputs(r):
	"""Replicate ResponsePreprocessor minimally: question->user, response->assistant, image->images."""
	msgs = [
		{"role": "user", "content": r["question"] or ""},
		{"role": "assistant", "content": r["response"] or ""},
	]
	images = list(r["image"]) if r["image"] is not None else []
	return {"messages": msgs, "images": images}


def classify_round(resp):
	c = resp.count("</sandbox_output>")
	if c == 0:
		return "no_tool"
	if c == 1:
		return "single_round"
	return "multi_round"


def audit_one(template, tok, r):
	"""Encode one row through the training template and verify masking invariants."""
	inp = row_to_inputs(r)
	enc = template.encode(inp)
	iid = enc["input_ids"]
	lab = enc["labels"]
	assert len(iid) == len(lab), "input_ids/labels length mismatch"
	sup_ids = [i for i, l in zip(iid, lab) if l != -100]
	sup_text = tok.decode(sup_ids)
	full_text = tok.decode(iid)
	rtype = classify_round(r["response"] or "")

	checks = {}
	# invariant 1: sandbox_output *content* must not be supervised.
	# The tags themselves are loss_scale 0; the executed-output text between them is masked.
	# Heuristic: supervised text should not contain a full sandbox block payload.
	# We assert no "</sandbox_output>" token region is supervised by checking the tag is absent
	# from supervised text OR only appears with zero surrounding masked mismatch.
	checks["sandbox_tag_not_supervised"] = ("</sandbox_output>" not in sup_text)
	# invariant 2: <image> raw placeholder never supervised (image tokens masked)
	checks["image_placeholder_not_supervised"] = ("<image>" not in sup_text and "<img>" not in sup_text)
	# invariant 3: the user question lives in the MASKED prompt region (not supervised).
	# Structural check: the prompt = all tokens before the first labels!=-100 token.
	# We verify a distinctive question slice appears in that masked prompt region.
	# (A naive "slice not in sup_text" check gives false positives when the model
	# legitimately restates the question inside its own supervised <think> reasoning.)
	q = (r["question"] or "").replace("<image>", "").strip()
	first_sup = next((k for k, l in enumerate(lab) if l != -100), len(iid))
	prompt_text = tok.decode(iid[:first_sup])
	qslice = q[:40]
	checks["question_in_masked_prompt"] = (len(qslice) == 0 or qslice in prompt_text)
	# invariant 4: something IS supervised (non-empty) and <answer> is supervised
	checks["nonempty_supervised"] = (len(sup_ids) > 0)
	checks["answer_supervised"] = ("<answer>" in sup_text)

	# round-specific: for multi_round only content after LAST </sandbox_output> supervised.
	resp = r["response"] or ""
	if rtype == "multi_round":
		last = resp.rfind("</sandbox_output>")
		tail = resp[last + len("</sandbox_output>"):]
		# earlier tool-call reasoning (first <think>) must be masked
		head = resp[:resp.find("</sandbox_output>")] if resp.find("</sandbox_output>") >= 0 else ""
		head_probe = head.replace("<think>", "").strip()[:40]
		checks["multiround_earlyround_masked"] = (len(head_probe) == 0 or head_probe not in sup_text)
		tail_probe = tail.replace("<answer>", "").strip()[:30]
		checks["multiround_finalround_supervised"] = (len(tail_probe) == 0 or tail_probe in sup_text)

	result = {
		"round_type": rtype,
		"n_images": len(inp["images"]),
		"len_input_ids": len(iid),
		"n_supervised": len(sup_ids),
		"n_sandbox_close": resp.count("</sandbox_output>"),
		"checks": checks,
		"all_pass": all(checks.values()),
	}
	return result, sup_text, full_text


# ---------- future-image-leakage ----------
def leakage_one(r):
	"""For multi-round, verify <image> placeholders appear in temporal order and count matches."""
	q = r["question"] or ""
	resp = r["response"] or ""
	concat = q + "\n" + resp
	imgs = list(r["image"]) if r["image"] is not None else []
	q_ph = q.count("<image>")
	resp_ph = resp.count("<image>")
	total_ph = q_ph + resp_ph
	positions = image_positions(concat)
	# temporal order: original image placeholder(s) in question precede response placeholders.
	# Since concat = q + resp, any resp placeholder offset >= len(q); ordering is inherent.
	# The real leakage risk is a count mismatch causing swift to auto-prepend an image.
	count_ok = (total_ph == len(imgs))
	# order_ok: question placeholders come before response placeholders (always true by concat),
	# and there is at least one original image placeholder in the question when images exist.
	order_ok = True
	if len(imgs) > 0 and resp_ph > 0:
		order_ok = (q_ph >= 1)
	passed = count_ok and order_ok
	return {
		"n_images": len(imgs),
		"q_placeholders": q_ph,
		"resp_placeholders": resp_ph,
		"total_placeholders": total_ph,
		"positions": positions[:20],
		"count_match": count_ok,
		"order_ok": order_ok,
		"pass": passed,
	}


# ---------- main ----------
def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--stage-dir", required=True)
	ap.add_argument("--model", required=True, help="Local InternVL tokenizer/processor checkpoint")
	ap.add_argument("--n-audit", type=int, default=50)
	ap.add_argument("--max-static", type=int, default=0)
	args = ap.parse_args()
	if args.n_audit <= 0 or args.max_static < 0:
		ap.error("n-audit must be positive and max-static must be nonnegative")
	if not os.path.isdir(args.model):
		ap.error("model must be an existing local checkpoint directory")

	stage_dir = os.path.abspath(args.stage_dir)
	stage_name = os.path.basename(stage_dir.rstrip("/"))
	ds = ShardedDataset(stage_dir)
	print(f"[load] {len(ds.files)} parquet shard(s) under {stage_dir}")
	print(f"[load] total rows = {len(ds)}")

	# ---- 1. static ----
	print("\n===== 1. STATIC DATA CHECK =====")
	stats, failures = static_check(ds, args.max_static)
	for k, v in stats.items():
		print(f"  {k}: {v}")
	if failures:
		print(f"  first failures (up to 40): {json.dumps(failures[:10])}")
	else:
		print("  no static failures")

	# ---- 2. label audit ----
	print("\n===== 2. LABEL AUDIT (swift train template, labels!=-100) =====")
	template, tok, tmpl_type = load_template(args.model)
	_o3 = getattr(template, "O3", None)
	print(f"  template_type={tmpl_type} class={type(template).__name__} mode={template.mode} O3={_o3}")

	rng = random.Random(SEED)
	n_audit = min(args.n_audit, len(ds))
	sample_idx = rng.sample(range(len(ds)), n_audit)

	audit_results = []
	type_counter = Counter()
	check_fail_counter = Counter()
	examples = {}  # round_type -> (idx, sup_text, full_text)
	for idx in sample_idx:
		r = ds.row(idx)
		try:
			res, sup_text, full_text = audit_one(template, tok, r)
		except Exception as e:
			audit_results.append({"idx": int(idx), "error": repr(e)[:200]})
			check_fail_counter["encode_error"] += 1
			continue
		res["idx"] = int(idx)
		audit_results.append(res)
		type_counter[res["round_type"]] += 1
		for cname, ok in res["checks"].items():
			if not ok:
				check_fail_counter[cname] += 1
		if res["round_type"] not in examples:
			examples[res["round_type"]] = (int(idx), sup_text, full_text)

	print(f"  audited {len(sample_idx)} samples; round-type distribution: {dict(type_counter)}")
	n_pass = sum(1 for a in audit_results if a.get("all_pass"))
	_n_scored = len([a for a in audit_results if "all_pass" in a])
	print(f"  samples passing ALL masking checks: {n_pass}/{_n_scored}")
	if check_fail_counter:
		print(f"  check failures: {dict(check_fail_counter)}")
	else:
		print("  all masking invariants held on every audited sample")

	# ---- print up to 3 full decoded examples (single/multi/computation) ----
	print("\n----- DECODED labels!=-100 EXAMPLES -----")
	order = ["single_round", "multi_round", "no_tool"]
	printed = 0
	for rt in order:
		if rt in examples and printed < 3:
			idx, sup_text, full_text = examples[rt]
			print(f"\n### example round_type={rt} idx={idx} (n_sandbox_close matches)")
			print("--- SUPERVISED (labels!=-100) ---")
			print(sup_text)
			printed += 1

	# ---- 3. leakage ----
	print("\n===== 3. FUTURE-IMAGE-LEAKAGE CHECK (multi-round + all sampled) =====")
	leak_results = []
	leak_fail = 0
	multi_seen = 0
	for idx in sample_idx:
		r = ds.row(idx)
		lk = leakage_one(r)
		lk["idx"] = int(idx)
		lk["round_type"] = classify_round(r["response"] or "")
		leak_results.append(lk)
		if lk["round_type"] == "multi_round":
			multi_seen += 1
		if not lk["pass"]:
			leak_fail += 1
	print(f"  sampled={len(sample_idx)} multi_round={multi_seen} leakage_FAIL={leak_fail}")
	if leak_fail:
		bad = [l for l in leak_results if not l["pass"]][:10]
		print(f"  failing samples: {json.dumps(bad)}")
	else:
		print("  placeholder-count and question-before-response checks passed; image-content provenance is not verified")
	# show a few multi-round leakage rows explicitly
	mr = [l for l in leak_results if l["round_type"] == "multi_round"][:5]
	for l in mr:
		_a, _ni, _qp, _rp, _pos, _pa = l["idx"], l["n_images"], l["q_placeholders"], l["resp_placeholders"], l["positions"], l["pass"]
		print(f"    [multi] idx={_a} n_images={_ni} q_ph={_qp} resp_ph={_rp} positions={_pos} pass={_pa}")

	# ---- write JSON report ----
	static_ok = (
		stats["image_decode_fail"] == 0 and
		stats["image_count_mismatch"] == 0 and
		stats["tags_unbalanced"] == 0 and
		stats["empty_response"] == 0 and
		stats["missing_answer"] == 0
	)
	masking_ok = (len(check_fail_counter) == 0)
	leakage_ok = (leak_fail == 0)
	report = {
		"stage": stage_name,
		"stage_dir": stage_dir,
		"model": os.path.abspath(args.model),
		"template_type": tmpl_type,
		"seed": SEED,
		"total_rows": int(len(ds)),
		"static": {"stats": stats, "failures": failures, "pass": bool(static_ok)},
		"label_audit": {
			"n_audit": int(n_audit),
			"round_type_distribution": dict(type_counter),
			"check_failures": dict(check_fail_counter),
			"pass": bool(masking_ok),
			"per_sample": audit_results,
		},
		"leakage": {
			"n_audit": int(n_audit),
			"multi_round_seen": int(multi_seen),
			"fail": int(leak_fail),
			"pass": bool(leakage_ok),
			"per_sample": leak_results,
		},
		"overall_pass": bool(static_ok and masking_ok and leakage_ok),
		"scope": "Structural and sampled label checks only; image identity and causal visibility are not verified",
	}
	audit_dir = os.path.join(os.path.dirname(stage_dir.rstrip("/")), "audits")
	os.makedirs(audit_dir, exist_ok=True)
	out_path = os.path.join(audit_dir, f"label_audit_{stage_name}.json")
	with open(out_path, "w") as fh:
		json.dump(report, fh, indent=2, ensure_ascii=False)

	print("\n===== SUMMARY =====")
	print(f"  stage={stage_name} rows={len(ds)}")
	print(f"  static_pass={static_ok} masking_pass={masking_ok} leakage_pass={leakage_ok}")
	print(f"  OVERALL_PASS={report['overall_pass']}")
	print(f"  JSON report -> {out_path}")
	return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
	raise SystemExit(main())
