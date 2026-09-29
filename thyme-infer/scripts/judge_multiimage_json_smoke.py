#!/usr/bin/env python3
# Modified for anonymous review: no-thinking requests and strict smoke checks.
"""Check a configured judge alias, two-image JSON output and authentication.
This is a transport smoke check, not a validation of the training critic.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from m_rlsd.inference.remote_vlm_client import RemoteVLMClient, RemoteVLMConfig  # noqa: E402


# --- 最小 JSON schema：足以验证结构化输出通路，不等于最终 GCEP schema ---------
# The training schema is in gcep/judge_client.py; this smoke uses a small image-count schema.
_SMOKE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "applicable": {"type": "boolean"},
        "num_images_seen": {"type": "integer"},
        "left_image_summary": {"type": "string"},
        "right_image_summary": {"type": "string"},
    },
    "required": [
        "applicable",
        "num_images_seen",
        "left_image_summary",
        "right_image_summary",
    ],
    "additionalProperties": False,
}


def _make_synthetic_image(color: tuple[int, int, int], label: str) -> str:
    """生成一张纯色带标签的 PNG，返回 data:image/png;base64 URL。

    仅用于 smoke：不依赖任何外部图片文件即可跑通多图通路。
    """
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            f"[smoke] 需要 Pillow 生成合成图；或用 --image-a/--image-b 指定真实图片。({exc})"
        )
    img = Image.new("RGB", (256, 256), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([16, 16, 240, 240], outline=(255, 255, 255), width=4)
    draw.text((40, 110), label, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _file_to_data_url(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[smoke] 图片不存在: {p}")
    import mimetypes

    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_two_image_message(prompt: str, url_a: str, url_b: str) -> dict[str, Any]:
    """构造一条含两张独立 image item 的 user message。

    刻意与 RemoteVLMClient.build_multimodal_user_message 同格式（image_url/data URL），
    但放两张图——这正是 GCEP 单 interaction 多图的最小复现。
    """
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url_a}},
            {"type": "image_url", "image_url": {"url": url_b}},
        ],
    }


def _run_healthcheck(client: RemoteVLMClient, model: str) -> list[str]:
    listing = client.healthcheck()
    served = [e.get("id", "") for e in listing.get("data", [])]
    print(f"[smoke] base_url={client.base_url}")
    print(f"[smoke] served_models={served}")
    if model not in served:
        raise SystemExit(
            f"[smoke] FAIL: 期望的 alias {model!r} 不在 /v1/models 返回中: {served}"
        )
    print(f"[smoke] OK: /v1/models 含期望 alias {model!r}")
    return served


def _run_multi_image_json(
    client: RemoteVLMClient, url_a: str, url_b: str, max_tokens: int
) -> None:
    prompt = (
        "You are given TWO separate images (left then right). "
        "Look at BOTH. Return ONLY JSON with fields: "
        "applicable (bool), num_images_seen (int, how many distinct images you were given), "
        "left_image_summary (str), right_image_summary (str)."
    )
    message = _build_two_image_message(prompt, url_a, url_b)

    # response_format 通过 extra_body 透传（客户端已支持），温度 0 => 确定性结构化输出。
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "smoke_schema", "schema": _SMOKE_JSON_SCHEMA},
        }
    }
    text, raw = client.generate_text(
        [message],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    usage = raw.get("usage") or {}
    print(f"[smoke] completion_ok=true prompt_tokens={usage.get('prompt_tokens','?')} "
          f"completion_tokens={usage.get('completion_tokens','?')}")
    print(f"[smoke] raw_text={text!r}")

    # 结构化输出必须能解析且满足最小 schema。
    try:
        obj = json.loads(text)
    except Exception as exc:
        raise SystemExit(f"[smoke] FAIL: 返回不是可解析 JSON: {exc}")
    if not isinstance(obj, dict) or set(obj) != set(_SMOKE_JSON_SCHEMA["required"]):
        raise SystemExit("[smoke] FAIL: JSON has missing or unexpected fields")
    if not isinstance(obj["applicable"], bool) or type(obj["num_images_seen"]) is not int:
        raise SystemExit("[smoke] FAIL: incorrect JSON field types")
    if not all(isinstance(obj[key], str) for key in ("left_image_summary", "right_image_summary")):
        raise SystemExit("[smoke] FAIL: image summaries must be strings")
    if obj["num_images_seen"] != 2:
        raise SystemExit("[smoke] FAIL: response does not confirm two images")
    if (raw.get("choices") or [{}])[0].get("finish_reason") == "length":
        raise SystemExit("[smoke] FAIL: response hit the token limit")
    print("[smoke] OK: two-image JSON response passed the smoke schema checks")


def _run_badkey_negative(base_url: str, model: str, timeout: float) -> None:
    """用错误 key 请求，必须失败（鉴权正确性）。"""
    bad = RemoteVLMClient(
        RemoteVLMConfig(
            base_url=base_url,
            model=model,
            api_key="DEFINITELY_WRONG_KEY_smoke",
            timeout_sec=timeout,
            max_retries=0,
        )
    )
    from urllib.error import HTTPError
    # /models may be public; only a completion rejection establishes this check.
    try:
        bad.generate_text([RemoteVLMClient.build_text_user_message("hi")],
                          temperature=0.0, max_tokens=8,
                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    except Exception as exc:
        cause = exc.__cause__
        if isinstance(cause, HTTPError) and cause.code in (401, 403):
            print("[smoke] OK: incorrect API key was rejected with HTTP 401/403")
            return
        raise SystemExit(f"[smoke] FAIL: cannot establish authentication rejection: {type(exc).__name__}")
    raise SystemExit("[smoke] FAIL: completion accepted an incorrect API key")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GCEP Judge multi-image + JSON schema smoke test")
    ap.add_argument("--base-url", required=True, help="judge 服务 base URL（可带或不带 /v1）")
    ap.add_argument("--api-key", required=True, help="judge 服务 Bearer 密钥")
    ap.add_argument("--model", required=True, help="served model alias（如 qwen3.5-397b-judge）")
    ap.add_argument("--image-a", default="", help="第一张图（省略则用合成图）")
    ap.add_argument("--image-b", default="", help="第二张图（省略则用合成图）")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout-sec", type=float, default=120.0)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--healthcheck-only", action="store_true")
    ap.add_argument("--skip-badkey", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_tokens <= 0 or args.timeout_sec <= 0 or args.max_retries < 0:
        raise SystemExit("max-tokens/timeout must be positive; max-retries must be nonnegative")
    client = RemoteVLMClient(
        RemoteVLMConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
        )
    )

    _run_healthcheck(client, args.model)
    if args.healthcheck_only:
        print("[smoke] healthcheck_only=true;跳过多图/JSON 测试。")
        return 0

    url_a = _file_to_data_url(args.image_a) if args.image_a else _make_synthetic_image((200, 40, 40), "LEFT-A")
    url_b = _file_to_data_url(args.image_b) if args.image_b else _make_synthetic_image((40, 80, 200), "RIGHT-B")

    _run_multi_image_json(client, url_a, url_b, args.max_tokens)

    if not args.skip_badkey:
        _run_badkey_negative(args.base_url, args.model, args.timeout_sec)

    print("[smoke] ALL PASS ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
