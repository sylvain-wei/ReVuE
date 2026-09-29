#!/usr/bin/env python3
# Modified for anonymous review: correct wire format and full schema/alias validation.
"""Send one alias query and one text-only JSON request to a configured judge.

This checks transport and JSON shape only. It does not validate visual reasoning,
training integration, image capacity, or experimental results. No retry is made.
"""
from __future__ import annotations
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


def load_schema():
    """Read the literal schema without importing the training framework."""
    path = Path(__file__).resolve().parents[1] / "thyme-infer/Thyme/swift/trainers/rlhf_trainer/gcep/judge_client.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "JUDGE_JSON_SCHEMA":
            return ast.literal_eval(node.value)
    raise RuntimeError("Training judge schema is missing")


def validate_shape(value, schema, path="$"):
    """Validate the object/string/boolean/array/enum subset used by the critic schema."""
    expected = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool}
    if expected not in types or not isinstance(value, types[expected]):
        raise ValueError(f"{path}: expected {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: unknown enum value")
    if expected == "object":
        properties = schema.get("properties", {})
        if set(schema.get("required", [])) - value.keys():
            raise ValueError(f"{path}: missing required fields")
        if schema.get("additionalProperties") is False and value.keys() - properties.keys():
            raise ValueError(f"{path}: unexpected fields")
        for key, item in value.items():
            if key in properties:
                validate_shape(item, properties[key], f"{path}.{key}")
    elif expected == "array":
        for index, item in enumerate(value):
            validate_shape(item, schema["items"], f"{path}[{index}]")


def pick(*names, default=""):
    return next((os.environ[n] for n in names if os.environ.get(n)), default)


def http(method, url, key, payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp), time.monotonic() - started
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Judge returned HTTP {exc.code}") from exc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=pick("JUDGE_BASE_URL", "GCEP_JUDGE_BASE_URL", "REMOTE_VLM_BASE_URL"))
    ap.add_argument("--model", default=pick("JUDGE_MODEL", "GCEP_JUDGE_MODEL", "REMOTE_VLM_MODEL"))
    ap.add_argument("--api-key", default=pick("JUDGE_API_KEY", "GCEP_JUDGE_API_KEY", "REMOTE_VLM_API_KEY"))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.base_url or not args.model or not args.api_key or args.max_tokens <= 0:
        ap.error("Set base-url, model, api-key through flags/environment and a positive max-tokens")
    base = args.base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    rec = {"model": args.model, "scope": "text-only transport and JSON schema", "verdict": "check_failed"}
    try:
        schema = load_schema()
        status, listing, elapsed = http("GET", f"{base}/models", args.api_key, timeout=30)
        aliases = [item.get("id") for item in listing.get("data", [])]
        rec["alias_present"] = args.model in aliases
        if not rec["alias_present"]:
            raise ValueError("Configured model alias is absent from /models")
        payload = {"model": args.model, "temperature": 0.0, "max_tokens": args.max_tokens,
                   "messages": [{"role": "user", "content": "Return an inapplicable result matching the supplied JSON schema. No image or rollout evidence is supplied."}],
                   "chat_template_kwargs": {"enable_thinking": False},
                   "response_format": {"type": "json_schema", "json_schema": schema}}
        status, body, elapsed = http("POST", f"{base}/chat/completions", args.api_key, payload)
        choice = body["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("Judge response hit the token limit")
        parsed = json.loads(choice["message"]["content"])
        validate_shape(parsed, schema["schema"])
        rec.update(verdict="ok", schema_valid=True, latency_sec=round(elapsed, 3))
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    text = json.dumps(rec, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0 if rec["verdict"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
