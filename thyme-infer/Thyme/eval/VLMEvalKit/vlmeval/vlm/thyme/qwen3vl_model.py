# Thyme agent loop with Qwen3-VL backend（Metis-8B-ColdStart / Metis-8B-RL）。
# 父类 Thyme 只触通用 API（AutoProcessor.apply_chat_template / processor(text, images) /
# model.generate / tokenizer），Qwen3-VL 与 Qwen2.5-VL 形状一致——仅需替换模型类。
# 父类 __init__ 硬编码 Qwen2_5_VLForConditionalGeneration 且无注入点，故这里显式
# 重放初始化参数段（语义逐项对齐 model.py:43-95），只在 from_pretrained 处分叉。
# 需要 transformers>=4.57（Qwen3VLForConditionalGeneration），在 metis-eval env 运行。
from __future__ import annotations

import torch

from ..base import BaseModel  # noqa: F401  (kept for external imports)
from .model import Thyme
from .utils import SPECIAL_STRING_LIST


class ThymeQwen3VL(Thyme):
    """Thyme agent loop with Qwen3-VL backend (Metis ckpts)."""

    @staticmethod
    def _model_class():
        """Transformers model class to instantiate; overridden by the MoE variant."""
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration

    def __init__(
        self,
        model_path: str,
        min_pixels=None,
        max_pixels=None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        max_iterations=5,
        max_retry=5,
        use_custom_prompt=True,
        system_prompt='You are a helpful assistant.',
        post_process=True,
        verbose=False,
        **kwargs,
    ):
        # 走 MRO 链（Thyme → ThymePromptMixin → BaseModel），与父类 super().__init__ 等价
        super(Thyme, self).__init__(use_custom_prompt=use_custom_prompt)
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
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        assert model_path is not None
        self.model_path = model_path

        from transformers import AutoProcessor
        model_cls = self._model_class()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            stop_strings=SPECIAL_STRING_LIST,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            tokenizer=self.processor.tokenizer,
        )
        # 强制 bf16：Metis-8B-RL 官方 ckpt 是 fp32（33GB），'auto' 会按 fp32 加载
        # （显存×2、吞吐减半）；bf16 推理是标准做法，与 ColdStart（bf16）及既有
        # Qwen/InternVL 评测口径一致。
        self.model = model_cls.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map='auto', attn_implementation='sdpa'
        )
        self.model.eval()
        torch.cuda.empty_cache()


class ThymeQwen3VLMoe(ThymeQwen3VL):
    """Thyme agent loop with Qwen3-VL **MoE** backend (e.g. Qwen3-VL-30B-A3B-*).

    Identical to ThymeQwen3VL except the transformers class (Qwen3VLMoeForConditionalGeneration).
    """

    @staticmethod
    def _model_class():
        from transformers import Qwen3VLMoeForConditionalGeneration
        return Qwen3VLMoeForConditionalGeneration
