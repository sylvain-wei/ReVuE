"""InternVL3.5 Thyme-style agentic eval model.

Subclass of `Thyme` that swaps the Qwen2.5-VL model/processor for InternVL3.5
(`InternVLChatModel` + Qwen3 tokenizer), while reusing the Thyme agent loop:
retry x iteration, </code> pause + sandbox execution, <sandbox_output> image
feedback, </answer> stop, fallback simple prompt, post-processing, trajectory
dump.

ADDITIVE only: the Qwen2.5-VL `Thyme` class and its default config entry
`Thyme-RL-local` are untouched. InternVL is selected only when
`phase3_eval_opd.py` resolves `InternVLChatModel` in the model config's
`architectures` (or via `--model-class thyme_internvl`).
"""
from __future__ import annotations

import os
import copy
import logging
import re
import warnings
from typing import Any, List, Optional

import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from .model import Thyme, ensure_image_url
from .utils import sanitize_intermediate_image
from .sandbox import execute_code_in_sandbox
from .utils import (
    generate_prompt_final_qa,
    generate_prompt_simple_qa,
    SPECIAL_STRING_LIST,
    REASONING_SYS_PROMPT,
    SIMPLE_SYS_PROMPT,
)
from transformers import AutoModel, AutoTokenizer


# --- InternVL image preprocessing (from vintern_chat.py, standalone) --------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num: int = 1, max_num: int = 12,
                        image_size: int = 448, use_thumbnail: bool = True):
    """InternVL aspect-ratio tiling. Returns list of PIL tiles."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _load_image_tiles(image_path: str, input_size: int = 448, max_num: int = 12,
                      use_thumbnail: bool = True) -> torch.Tensor:
    """Load an image and return tile tensor [N, 3, H, W]."""
    if image_path.startswith("file://"):
        image_path = image_path[len("file://"):]
    image = Image.open(image_path).convert("RGB")
    tiles = _dynamic_preprocess(
        image, image_size=input_size, max_num=max_num, use_thumbnail=use_thumbnail
    )
    transform = _build_transform(input_size)
    return torch.stack([transform(t) for t in tiles])


# --- ThymeInternVL -----------------------------------------------------------


class ThymeInternVL(Thyme):
    """Thyme agent loop with InternVL3.5 backend.

    Replicates `Thyme.generate_inner_transformers` semantics but with InternVL
    input building: manual internvl2_5 template text, `<image>` placeholders
    expanded to `<img><IMG_CONTEXT>*256*num_patches</img>` in temporal order,
    dynamic_preprocess tiling, and the checkpoint's own
    `InternVLChatModel.generate(input_ids, attention_mask, pixel_values, ...)`
    which internally splices ViT embeddings at <IMG_CONTEXT> positions.

    Latency note: InternVLChatModel.generate injects the image embeddings ONCE
    (into inputs_embeds) and then decodes with the Qwen3 LLM only — there is no
    per-token ViT recompute, so no dummy-pixel patch is required.
    """

    # No video support in the InternVL Thyme path.
    VIDEO_LLM = False

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens: int = 2048,
        top_p: float = 0.001,
        top_k: int = 1,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        max_iterations: int = 5,
        max_retry: int = 5,
        use_custom_prompt: bool = True,
        system_prompt: str | None = "You are a helpful assistant.",
        post_process: bool = True,
        verbose: bool = False,
        input_size: int = 448,
        max_num: int = 12,
        use_thumbnail: bool = True,
        min_num: int = 1,
        **kwargs,
    ):
        # Skip Thyme.__init__ (hardcodes Qwen2_5_VLForConditionalGeneration +
        # AutoProcessor). Jump straight to ThymePromptMixin.__init__ to set up
        # use_custom_prompt + BaseModel state, then load InternVL ourselves.
        from .prompt import ThymePromptMixin
        ThymePromptMixin.__init__(self, use_custom_prompt=use_custom_prompt)

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_retry = max_retry
        self.verbose = verbose
        self.post_process = post_process
        self.fps = None
        self.nframe = None
        self.FRAME_FACTOR = 2

        assert model_path is not None
        self.model_path = model_path
        self.input_size = input_size
        self.max_num = max_num
        self.use_thumbnail = use_thumbnail
        self.min_num = min_num

        # Plan: AutoTokenizer(use_fast=False) + AutoModel(trust_remote_code).
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        # Shim exposing the `processor.tokenizer` API the parent agent loop uses
        # (`batch_decode`, `eos_token_id`).
        class _TokenizerShim:
            def __init__(self, tok):
                self._tok = tok
                self.eos_token_id = tok.eos_token_id

            def batch_decode(self, *a, **kw):
                return self._tok.batch_decode(*a, **kw)

            def __getattr__(self, name):
                return getattr(self._tok, name)

        class _ProcessorShim:
            def __init__(self, tok):
                self.tokenizer = _TokenizerShim(tok)

        self.processor = _ProcessorShim(self.tokenizer)

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        # InternVL forward splices ViT embeddings at <IMG_CONTEXT> token ids.
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.img_context_token_id = self.model.img_context_token_id
        self.num_image_token = int(getattr(self.model, "num_image_token", 256) or 256)
        self._IMG_START = "<img>"
        self._IMG_END = "</img>"
        self._IMG_CONTEXT = "<IMG_CONTEXT>"

        # generate_kwargs — same recipe as the Qwen Thyme class (greedy:
        # no do_sample, so temperature/top_p/top_k are ignored by HF).
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            stop_strings=SPECIAL_STRING_LIST,
            eos_token_id=self.tokenizer.eos_token_id,
            tokenizer=self.tokenizer,
        )

        torch.cuda.empty_cache()

    # ---- content preparation -------------------------------------------------

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None):
        """Same as Thyme._prepare_content, but strips the literal `<image>`
        prefix that generate_prompt_final_qa adds — InternVL renders the image
        placeholder inline from the image content item instead (avoids a
        duplicate `<image>` tag)."""
        user_image_path = self._extract_image_path(inputs)
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video': s['value'],
                    'min_pixels': self.min_pixels,
                    'max_pixels': self.max_pixels,
                }
            elif s['type'] == 'text':
                txt = generate_prompt_final_qa(s['value'], user_image_path)
                if txt.startswith("<image>"):
                    txt = txt[len("<image>"):]
                item = {'type': 'text', 'text': txt}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def _prepare_content_simple(self, inputs: list[dict[str, str]], dataset: str | None = None):
        """Fallback simple prompt; image placeholder rendered inline from the
        image item (generate_prompt_simple_qa has no `<image>` tag)."""
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video': s['value'],
                    'min_pixels': self.min_pixels,
                    'max_pixels': self.max_pixels,
                }
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': generate_prompt_simple_qa(s['value'])}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def _flatten_conversation_text(self, conversation_history: List[dict]) -> str:
        """Build the internvl2_5 prompt text. Image content items render as a
        literal `<image>` placeholder inline at their position (later expanded
        to `<img><IMG_CONTEXT>*256*np</img>` by _expand_image_placeholders)."""
        parts: List[str] = []
        for i, msg in enumerate(conversation_history):
            role = msg["role"]
            body_parts: List[str] = []
            for item in msg["content"]:
                if item.get("type") == "image":
                    body_parts.append("<image>")
                elif item.get("type") == "text":
                    body_parts.append(item["text"])
            body = "".join(body_parts)
            parts.append(f"<|im_start|>{role}\n{body}")
            if i == len(conversation_history) - 1 and role == "assistant":
                parts.append("")  # open assistant turn — generation continues
            else:
                parts.append("<|im_end|>\n")
        return "".join(parts)

    def _expand_image_placeholders(self, text: str, num_patches_list: List[int]) -> str:
        """Replace each literal `<image>` with `<img><IMG_CONTEXT>*256*np</img>`
        consuming num_patches_list in order."""
        out_parts: List[str] = []
        np_iter = iter(num_patches_list)
        last = 0
        for m in re.finditer(r"<image>", text):
            out_parts.append(text[last:m.start()])
            try:
                np = next(np_iter)
            except StopIteration:
                out_parts.append("<image>")
                last = m.end()
                continue
            out_parts.append(
                self._IMG_START
                + self._IMG_CONTEXT * (self.num_image_token * np)
                + self._IMG_END
            )
            last = m.end()
        out_parts.append(text[last:])
        return "".join(out_parts)

    def _collect_image_paths(self, conversation_history: List[dict]) -> List[str]:
        """All image paths in conversation order (user image first, then any
        sandbox feedback images appended as assistant content items)."""
        paths: List[str] = []
        for msg in conversation_history:
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    p = item.get("image", "") or item.get("value", "")
                    if p.startswith("file://"):
                        p = p[len("file://"):]
                    if p:
                        paths.append(p)
        return paths

    def _preprocess_all_images(self, image_paths: List[str]):
        """Load all images, return (pixel_values [N_total_tiles,3,H,W] or None,
        num_patches_list per image)."""
        tile_lists = []
        num_patches_list = []
        for p in image_paths:
            try:
                tiles = _load_image_tiles(
                    p, input_size=self.input_size, max_num=self.max_num,
                    use_thumbnail=self.use_thumbnail,
                )
            except Exception as e:
                self._verbose_print(
                    f"[ThymeInternVL] image load failed for {p}: {e}; using 1-tile dummy"
                )
                tiles = torch.zeros((1, 3, self.input_size, self.input_size), dtype=torch.bfloat16)
            tile_lists.append(tiles)
            num_patches_list.append(tiles.shape[0])
        if not tile_lists:
            return None, []
        return torch.cat(tile_lists, dim=0), num_patches_list

    def _generate_segment(self, conversation_history: List[dict], device):
        """Encode conversation, run one generate call, return (text, new_ids).

        InternVLChatModel.generate splices ViT embeddings into inputs_embeds
        and returns ONLY the newly generated tokens (no input prefix, because
        the underlying language_model.generate is fed inputs_embeds). So we
        decode output[0] directly without slicing off a prompt prefix.
        """
        image_paths = self._collect_image_paths(conversation_history)
        pixel_values, num_patches_list = self._preprocess_all_images(image_paths)
        conv_text = self._flatten_conversation_text(conversation_history)
        conv_text = self._expand_image_placeholders(conv_text, num_patches_list)
        model_inputs = self.tokenizer(conv_text, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                device=device,
                dtype=self.model.dtype if hasattr(self.model, "dtype") else torch.bfloat16,
            )
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                **self.generate_kwargs,
            )
        new_ids = output[0]
        text = self.tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return text, new_ids

    # ---- the agent loop -------------------------------------------------------

    def generate_inner_transformers(self, message, dataset=None):
        user_image_path = self._extract_image_path(message)

        messages: List[dict] = []
        messages.append({"role": "system", "content": [{"type": "text", "text": REASONING_SYS_PROMPT}]})
        messages.append({"role": "user", "content": self._prepare_content(message, dataset=dataset)})

        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        retry_generations = self.max_retry
        has_valid_answer = False
        device = next(self.model.parameters()).device

        while (retry_generations > 0) and (not has_valid_answer):
            conversation_history = copy.deepcopy(messages)
            # No kv_cache across iterations: we rebuild the full context each
            # time (correct + avoids boundary-tokenization mismatch); the prefix
            # re-encode cost is negligible vs 2048-token generation.
            previous_execution_context: dict = {}
            if self.verbose:
                print(f'\033[32m\n--- Generation {self.max_retry - retry_generations + 1} ---\033[0m')

            retry_iterations = self.max_iterations
            while retry_iterations > 0:
                retry_iterations -= 1
                generated_content = []
                if self.verbose:
                    print(f'\033[32m\n--- Iteration {self.max_iterations - retry_iterations} ---\033[0m')

                generated_text_segment, new_ids = self._generate_segment(
                    conversation_history, device
                )

                if "</answer>" in generated_text_segment:
                    generated_content.append({"type": "text", "text": generated_text_segment})

                code_regex = re.compile(
                    r'<code>\s*(?:```\s*)?(?:python\s*)?([\s\S]*?)\s*(?:```\s*)?</code>',
                    re.IGNORECASE,
                )
                code_match = code_regex.search(generated_text_segment)

                if code_match:
                    code_to_execute = code_match.group(1).strip()
                    if self.verbose:
                        print(f"\033[31m--- Found Code Block ---\n{generated_text_segment}\n-------------------------\033[0m")
                    processed_img_paths, captured_stdout, error_msg, current_execution_context = (
                        execute_code_in_sandbox(
                            code_to_execute, user_image_path,
                            previous_execution_context=previous_execution_context,
                        )
                    )
                    previous_execution_context = current_execution_context
                    if not processed_img_paths:
                        self._verbose_print(f"{error_msg}")
                        continue

                    has_valid_images = False
                    generated_content += [
                        {"type": "text", "text": generated_text_segment},
                        {"type": "text", "text": "<sandbox_output>"},
                    ]
                    first_path = processed_img_paths[0]
                    if os.path.exists(first_path):
                        for img_path in processed_img_paths:
                            if os.path.exists(img_path):
                                if not has_valid_images:
                                    has_valid_images = True
                                # Same guard as Thyme(model.py): model-produced
                                # degenerate intermediates crash qwen2_vl.smart_resize.
                                img_path = sanitize_intermediate_image(img_path)
                                generated_content.append({"type": "image", "image": img_path})
                    else:
                        generated_content.append({"type": "text", "text": first_path})

                    if has_valid_images or not os.path.exists(first_path):
                        generated_content.append({"type": "text", "text": "</sandbox_output>"})
                    else:
                        self._verbose_print("skip this generation due to error and adapt the temperature")
                        self.generate_kwargs["temperature"] = 1.0
                        continue
                else:
                    if "</answer>" not in generated_text_segment:
                        self._verbose_print("wo code. wo </answer>")
                        self._verbose_print(generated_text_segment)
                        self.generate_kwargs["temperature"] = 1.0
                        break

                if conversation_history[-1]["role"] == "user":
                    conversation_history.append({"role": "assistant", "content": generated_content})
                elif conversation_history[-1]["role"] == "assistant":
                    conversation_history[-1]["content"] += generated_content

                if "</answer>" in generated_text_segment:
                    has_valid_answer = True
                    self._verbose_print("\033[32m--- Final answer tag found. ---\033[0m")
                    break

                if new_ids.numel() > 0 and new_ids[-1].item() == self.tokenizer.eos_token_id:
                    if self.verbose:
                        print("\033[32m--- Model generated EOS and no further actions (code/answer). Assuming completion. ---\033[0m")
                    break

            if has_valid_answer:
                if self.verbose:
                    print(f'\033[32m\n--- End of processing (max iterations: {self.max_iterations}, actual: {self.max_iterations - retry_iterations + 1}) ---\033[0m')
                break
            else:
                self._verbose_print("Fail to find a valid answer and adapt the temperature")

            retry_generations -= 1
            self.generate_kwargs["temperature"] = 1.0

        # reset generation hyper-param.
        self.generate_kwargs["temperature"] = self.temperature  # 0.01

        # Fallback to simple prompt if all retries failed.
        if not has_valid_answer:
            self._verbose_print(
                f"\033[32m\n --- Fail to find a valid answer after {self.max_retry} retrys. Falling back to simple prompt.---\033[0m"
            )
            saved_stop_strings = self.generate_kwargs.pop("stop_strings", None)

            messages = []
            if self.system_prompt is not None:
                messages.append({"role": "system", "content": [{"type": "text", "text": SIMPLE_SYS_PROMPT}]})
            messages.append({"role": "user", "content": self._prepare_content_simple(message, dataset=dataset)})
            conversation_history = copy.deepcopy(messages)

            generated_text_segment, _ = self._generate_segment(conversation_history, device)

            if saved_stop_strings is not None:
                self.generate_kwargs["stop_strings"] = saved_stop_strings

            answer_match = re.search(r"<answer>(.*?)</answer>", generated_text_segment, re.DOTALL)
            if not answer_match:
                generated_text_segment = "<answer>" + generated_text_segment + "</answer>"
            conversation_history.append(
                {"role": "assistant", "content": [{"type": "text", "text": generated_text_segment}]}
            )

        # ---- final answer extraction + trajectory dump -------------------------
        final_assistant_response = ""
        for msg in reversed(conversation_history):
            if msg["role"] != "assistant":
                continue
            current_content_str = ""
            for item in msg["content"]:
                if item["type"] == "text":
                    current_content_str += item["text"]
            final_assistant_response = current_content_str
            break

        self.last_conversation_history = conversation_history
        self.last_final_assistant_response = final_assistant_response

        _trajectory_dump_path = os.environ.get("THYME_TRAJECTORY_DUMP_PATH")
        if _trajectory_dump_path:
            import json as _json
            import time as _time
            import uuid as _uuid
            _episode_id = os.environ.get("THYME_EPISODE_ID", f"auto_{_uuid.uuid4().hex[:10]}")
            _ans_match = re.search(r"<answer>(.*?)</answer>", final_assistant_response, re.DOTALL)
            _final_answer_raw = _ans_match.group(1).strip() if _ans_match else None
            _record = {
                "episode_id": _episode_id,
                "dataset": dataset,
                "user_image_path": user_image_path,
                "system_prompt": REASONING_SYS_PROMPT,
                "conversation_history": conversation_history,
                "final_assistant_response": final_assistant_response,
                "final_answer_raw": _final_answer_raw,
                "has_valid_answer": has_valid_answer,
                "sampling_mode": getattr(self, "_ats_mode_tag", "vlmeval_default"),
                "timestamp": _time.time(),
            }
            try:
                _dpath = os.path.abspath(_trajectory_dump_path)
                _dpath_dir = os.path.dirname(_dpath)
                if _dpath_dir:
                    os.makedirs(_dpath_dir, exist_ok=True)
                with open(_dpath, "a", encoding="utf-8") as _f:
                    _f.write(_json.dumps(_record, ensure_ascii=False, default=str) + "\n")
            except Exception as _e:
                print(f"[trajectory_dump] WARN: failed to write {_dpath}: {_e}")

        if self.post_process:
            self._verbose_print(
                f'\033[31m--- Final response ---\n{final_assistant_response}\n-------------------------\033[0m'
            )
            answer_match = re.search(r"<answer>(.*?)</answer>", final_assistant_response, re.DOTALL)
            if answer_match:
                final_answer = answer_match.group(1).strip()
            else:
                final_answer = "No answer tag found in the final output."
            match = re.search(r"\\boxed\{(.*?)\}", final_answer)
            if match:
                final_answer = self._extract_box_answer(final_answer)
            if self.verbose:
                print(f'\033[32m{final_answer}\033[0m')
            return final_answer
        else:
            return final_assistant_response

    def generate_inner(self, message, dataset=None):
        return self.generate_inner_transformers(message, dataset=dataset)
