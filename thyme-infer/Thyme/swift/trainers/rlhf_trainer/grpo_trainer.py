# Modified for this anonymized research release: portable paths and release cleanup.
# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/trl.
import concurrent.futures
import inspect
import os
import re
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field
from math import ceil
from queue import Queue
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import datetime

import datasets
import torch
import torch.nn as nn
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from packaging import version
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, TrainerCallback
from transformers.trainer import Trainer
from transformers.trainer_utils import seed_worker
from trl import GRPOTrainer as HFGRPOTrainer
from trl.extras.profiling import profiling_decorator
from trl.models import prepare_deepspeed
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_trainer import nanmax, nanmin

from swift.llm import InferRequest, MultiModelKeys, RequestConfig, RowPreprocessor, get_model_arch, to_device
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.plugin import loss_scale_map, multi_turns, orms, rm_plugins
from swift.utils import JsonlWriter, gc_collect, get_device, get_logger, is_vllm_available, is_wandb_available
from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin
from .utils import patch_lora_merge, patch_lora_unmerge, unwrap_model_for_generation
from .vllm_client import VLLMClient
from swift.trainers.sandbox import execute_code_in_sandbox
del HFGRPOTrainer.__init__
del HFGRPOTrainer.log
import json
import random
import string
import pickle
import timeout_decorator    # For more efficient timeout control
logger = get_logger()
if is_wandb_available():
    import wandb

InputsType = List[Dict[str, Union[torch.Tensor, Any]]]
# tuple: (messages, finish_reason)
OutputsType = List[Tuple[List[Dict], str]]
from PIL import Image
INVALID_REWARD_VALUE = -100
def check_aspect_ratio(img_path):
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            aspect_ratio = max(width, height) / min(width, height)
            return aspect_ratio <= 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def _remove_unpickable_values(dictionary):
    if dictionary is None:
        return None
    def is_pickable(obj):
        try:
            pickle.dumps(obj)
            return True
        except Exception:
            return False

    keys_to_remove = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            _remove_unpickable_values(value)
        elif not is_pickable(value):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del dictionary[key]
    return dictionary

def generate_unique_string(existing_strings, length=6):
    # 生成一个唯一的字符串
    while True:
        # 生成一个随机的六位字符串
        new_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        # 检查是否已经存在
        if new_string not in existing_strings:
            existing_strings.add(new_string)
            return new_string

def get_filename_without_extension(file_path):
    # 获取文件名（带后缀）
    file_name_with_extension = os.path.basename(file_path)
    # 分离文件名和后缀
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
    return file_name_without_extension


def _sandbox_b64_fix_enabled() -> bool:
    """GCEP_SANDBOX_B64_IMAGE_FIX=1 时，为沙箱执行物化 base64 图像。

    默认关闭。只有进入沙箱代码执行路径时才会调用该功能。
    """
    return os.getenv('GCEP_SANDBOX_B64_IMAGE_FIX', '0') == '1'


def _materialize_b64_image(image_path, scratch_dir):
    """base64 原图 → 临时真实文件（供 sandbox 用）。

    训练数据 thyme_rl_train 的原图在 parquet 里以 base64 存进 `path` 字段（非真实路径）；
    sandbox 的 ImagePathTransformer 会把模型代码里的 image_path 重写成这个 base64 串，
    模型 `Image.open(base64)` → [Errno 36] File name too long → sandbox 失败、无工具图。
    这里把 base64 物化成临时 PNG 并返回真实路径；非 base64（真实路径 / data URL 之外的形态）
    原样返回。任何失败一律返回原值（不 crash，沿用 sandbox 原有失败语义）。
    """
    try:
        if not isinstance(image_path, str):
            return image_path
        s = image_path.strip()
        b64 = None
        if s.startswith('data:image') and ';base64,' in s:
            b64 = s.split(';base64,', 1)[1]
        elif not s.startswith(('/', 'http://', 'https://')):
            b64 = s  # 裸 base64 候选（真实路径以 / 开头；http/https 不动）
        if not b64:
            return image_path
        import base64 as _b64
        import hashlib as _hl
        raw = _b64.b64decode(b64 + '=' * (-len(b64) % 4))
        # 魔数校验（权威判定）：确认真是图（PNG/JPEG/GIF/BMP/WEBP），否则按真实路径处理
        if not (raw.startswith(b'\x89PNG') or raw.startswith(b'\xff\xd8') or
                raw.startswith(b'GIF8') or raw.startswith(b'BM') or raw[:4] == b'RIFF'):
            return image_path
        os.makedirs(scratch_dir, exist_ok=True)
        fn = os.path.join(scratch_dir, f"orig_{_hl.sha1(raw[:65536]).hexdigest()[:16]}.png")
        if not os.path.exists(fn):
            with open(fn, 'wb') as f:
                f.write(raw)
        return fn
    except Exception:
        return image_path


def has_repeated_content(s: str, threshold: float = 0.3, tail_length: int = 1000) -> bool:
    """
    仅检测字符串末尾指定长度（默认为500）的部分是否存在重复内容。
    这使得检测时间对于长字符串来说是恒定的。

    :param s: 输入字符串
    :param threshold: 重复内容的比例阈值
    :param tail_length: 要分析的字符串末尾的长度
    :return: (是否超过阈值, 替换后的字符串)
    """
    if 'sandbox_output' in s or 'answer' in s or '<code>' in s:
        return False

    log_dir = 'logs/reward_log'
    log_file_path = os.path.join(log_dir, 'rl_max3072_output_zhiweikong_gene8_code_orm.txt')
    # 如果字符串本身就比要分析的长度短，那就分析整个字符串
    if len(s) <= tail_length:
        target_string = s
    else:
        # 否则，只取最后 `tail_length` 个字符进行分析
        target_string = s[-tail_length:]

    # 调用核心函数进行检测
    result = has_repeated_content_optimized(target_string, threshold)

    # 如果检测到了重复内容
    if result:
        repeated_substr, count, repeated_ratio = result

        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)

        # 准备写入文件的日志信息
        log_entry = (
            f"--- [Repeated Content Detected] ---\n"
            f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Threshold: {threshold:.2f}\n"
            f"Detected Ratio: {repeated_ratio:.2f}\n"
            f"Repeated Substring: \"{repeated_substr}\"\n"
            f"Count: {count}\n"
            f"Analyzed Tail ({len(target_string)} chars): \"{target_string}\"\n"
            f"Full Original String ({len(s)} chars): \"{s}\"\n"
            f"-------------------------------------\n\n"
        )
        
        # 将日志信息追加到文件中
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        # 返回True表示检测成功
        return True

    # 如果没有检测到重复内容，返回False
    return False


def has_repeated_content_optimized(s: str, threshold: float = 0.5) -> Optional[Tuple[str, int, float]]:
    """
    在字符串中检测重复内容。

    :param s: 输入字符串
    :param threshold: 重复内容的比例阈值
    :return: 如果找到，则返回一个包含(重复子串, 次数, 重复比例)的元组，否则返回 None
    """
    n = len(s)
    # 对于短字符串，不进行检测
    if n < 50:  
        return None

    # 从长到短遍历所有可能的子串长度
    for length in range(n // 3, 0, -1):
        # 优化：如果一个子串即使重复两次也无法超过阈值，那么更短的子串更不可能
        if (2 * length) / n <= threshold:
            break

        counts = {}
        # 统计当前长度的所有子串的出现次数
        for i in range(n - length + 1):
            current_substr = s[i:i + length]
            counts[current_substr] = counts.get(current_substr, 0) + 1
        
        # 检查是否有子串的重复比例超过阈值
        for substr, count in counts.items():
            if count > 1:
                repeated_ratio = (count * length) / n
                if repeated_ratio > threshold:
                    # 找到了，返回详细信息
                    return substr, count, repeated_ratio
    
    # 遍历结束，没有找到符合条件的重复内容
    return None

class GRPOCallback(TrainerCallback):

    def __init__(self, trainer):
        self.trainer = trainer

    # offload original_modules to cpu, to save memory
    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.queue = self.trainer.train_queue
        train_dataloader = getattr(state, 'train_dataloader', None) or kwargs.get('train_dataloader')
        self.trainer._prefetch(train_dataloader)


@dataclass
class DataCache:
    inputs: List[Dict] = field(default_factory=list)
    outputs: List[Dict] = field(default_factory=list)

REASONING_SYS_PROMPT='''You are a helpful assistant.

Solve the following problem step by step, and optionally write Python code for image manipulation to enhance your reasoning process. The Python code will be executed by an external sandbox, and the processed image or result (wrapped in <sandbox_output></sandbox_output>) can be returned to aid your reasoning and help you arrive at the final answer.

**Reasoning & Image Manipulation (Optional but Encouraged):**
    * You have the capability to write executable Python code to perform image manipulations (e.g., cropping to a Region of Interest (ROI), resizing, rotation, adjusting contrast) or perform calculation for better reasoning.
    * The code will be executed in a secure sandbox, and its output will be provided back to you for further analysis.
    * All Python code snippets **must** be wrapped as follows:
    <code>
    ```python
    # your code.
    ```
    </code>
    * At the end of the code, print the path of the processed image (processed_path) or the result for further processing in a sandbox environment.'''


class GRPOTrainer(RLHFTrainerMixin, SwiftMixin, HFGRPOTrainer):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_model: Optional[List[Union[PreTrainedModel, nn.Module]]] = None,
                 reward_funcs: Optional[List[Union[str, Callable]]] = None,
                 *_args,
                 **kwargs):
        from swift.trainers.rlhf_arguments import GRPOConfig
        args: GRPOConfig = kwargs['args']
        self.existing_strings = set()
        self.args = args
        self.O3 = args.O3
        self.start_time = str(os.environ.get("TIME_STAMP", "0624_14:35:58"))
        logger.warning(f"start_time: {self.start_time}")
        if self.O3:
            logger.warning("You are using O3 paradigm.")
        # for async generate
        self.train_queue = Queue()
        self.eval_queue = Queue()

        self.processing_class = kwargs.get('template').tokenizer

        # for offload model/optimizer
        self.offload_modules = {}
        self.offload_states = {}

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        if reward_funcs:
            for i, reward_func in enumerate(reward_funcs):
                if reward_func in orms:
                    reward_func_class = orms[reward_func]
                    reward_func_args = list(inspect.signature(reward_func_class.__init__).parameters)
                    reward_func_kwargs = {
                        key: getattr(args, key)
                        for key in reward_func_args if key not in ['self', 'args', 'kwargs'] and hasattr(args, key)
                    }
                    if 'tokenizer' in reward_func_args:
                        reward_func_kwargs['tokenizer'] = self.processing_class
                    reward_funcs[i] = reward_func_class(**reward_func_kwargs)
                elif not callable(reward_func):
                    raise ValueError(f'reward_function {reward_func} is not implemented in swift.llm.plugin')

        self.reward_funcs = reward_funcs
        self.reward_func_names = []
        for reward_func in reward_funcs:
            if inspect.isfunction(reward_func):
                reward_func_name = reward_func.__name__
            else:
                reward_func_name = reward_func.__class__.__name__
            self.reward_func_names.append(reward_func_name)

        self.reward_model_plugins = [None] * len(self.reward_funcs)

        if reward_model is not None:
            reward_template = kwargs.pop('reward_template')
            reward_plugins = args.reward_model_plugin
            if reward_plugins is None:
                reward_plugins = ['default'] * len(reward_model)
            assert len(reward_plugins) == len(reward_model), (
                f"The number of 'reward_model_plugin' ({len(reward_plugins)}) does not match "
                f"the number of 'reward_model' ({len(reward_model)}). "
                "Please provide a corresponding 'reward_model_plugin' for each 'reward_model'.")
            for rm, rm_plugin, rm_template in zip(reward_model, reward_plugins, reward_template):
                # Set encoding mode train(see details in Template.encode).
                # Set max_length to None to disable truncation, as the input length has already been truncated earlier.
                rm_template.set_mode('train')
                rm_template.max_length = None
                if rm_plugin not in rm_plugins:
                    raise ValueError(f'rm_plugin {rm_plugin} is not implemented in swift.llm.plugin')
                self.reward_model_plugins.append(rm_plugins[rm_plugin](model=rm, template=rm_template))
                self.reward_funcs.append(rm)
                self.reward_func_names.append(rm.config._name_or_path.split('/')[-1])

        if not self.reward_funcs:
            raise ValueError('You must specify reward_funcs or reward_model')

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(f'Number of reward weights ({len(args.reward_weights)}) must match number of reward '
                                 f'functions ({len(reward_funcs)})')
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        self.multi_turn_func = None
        if self.args.multi_turn_func:
            if isinstance(self.args.multi_turn_func, str):
                assert self.args.multi_turn_func in multi_turns
                multi_turn_func = multi_turns[self.args.multi_turn_func]
                self.multi_turn_func = multi_turn_func
            else:
                self.multi_turn_func = self.args.multi_turn_func

        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.loss_type = args.loss_type
        self.max_completion_length = args.max_completion_length
        self.completion_length_limit_scope = args.completion_length_limit_scope
        model.warnings_issued['estimate_tokens'] = True
        kwargs['data_collator'] = lambda features: features
        self.shuffle_dataset = args.dataset_shuffle

        self.use_vllm = args.use_vllm
        self.async_generate = args.async_generate
        vllm_client = kwargs.pop('vllm_client')  # for external vllm

        super().__init__(model, ref_model, *_args, **kwargs)

        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'completions.jsonl'))
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = {
            'prompt': deque(maxlen=maxlen),
            'completion': deque(maxlen=maxlen),
            'rewards': defaultdict(lambda: deque(maxlen=maxlen)),
        }

        # num_generation check
        num_processes = self.accelerator.num_processes
        self.effective_train_batch_size = effective_batch_size = \
            args.per_device_train_batch_size * num_processes * args.gradient_accumulation_steps
        possible_values = [n_gen for n_gen in range(2, effective_batch_size + 1) if (effective_batch_size) % n_gen == 0]

        if self.num_generations not in possible_values:
            raise ValueError(
                f'The effective train batch size ({num_processes} x {args.per_device_train_batch_size} x '
                f'{args.gradient_accumulation_steps}) must be evenly divisible by the number of generations per '
                f'prompt ({self.num_generations}). Given the current effective train batch size, the valid values for '
                f'the number of generations are: {possible_values}.')
        if self.args.eval_strategy != 'no':
            effective_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen for n_gen in range(2, effective_batch_size + 1) if (effective_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f'The effective eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be '
                    f'evenly divisible by the number of generations per prompt ({self.num_generations}). Given the '
                    'current effective eval batch size, the valid values for the number of generations are: '
                    f'{possible_values}.')

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # ------------------------------------------------------------------
        # RLSD（Reinforcement Learning with Self-Distillation）配置
        # ------------------------------------------------------------------
        # RLSD 用「与学生同权重、但带 privilege 信息」的 teacher 对 rollout 逐 token
        # 打分，再对 GRPO 的 advantage 做 token 级、带 sign(A) 方向反转的乘性重加权：
        #   Δ_t = sg(logP_T(y_t|privilege) − logP_S(y_t))
        #   w_t = clip(exp(sign(A)·Δ_t), 1−ε_w, 1+ε_w)
        #   Â_t = A · ((1−λ) + λ·w_t)
        # λ 在前 RLSD_LAMBDA_DECAY_STEPS 步内从 RLSD_LAMBDA_INIT 线性衰减到 0，
        # 之后 λ=0（纯 GRPO），并短路整个 teacher 前向以零额外开销。
        self.rlsd_enable = bool(int(os.getenv('RLSD_ENABLE', '0')))
        # privilege 信息来源：gt_only | rich_final_obs_correct_only（见 rlsd_privilege.py）
        self.rlsd_privilege_mode = os.getenv('RLSD_PRIVILEGE_MODE', 'gt_only')
        self.rlsd_lambda_init = float(os.getenv('RLSD_LAMBDA_INIT', '0.5'))
        self.rlsd_lambda_decay_steps = int(os.getenv('RLSD_LAMBDA_DECAY_STEPS', '500'))
        self.rlsd_eps_w = float(os.getenv('RLSD_EPS_W', '0.2'))
        # teacher 打分 span：all（全轨迹）| answer（只在 <answer> 段，省算力、信号更稀疏）
        self.rlsd_score_span = os.getenv('RLSD_SCORE_SPAN', 'all')
        self._rlsd_mismatch_count = 0  # token 对齐失败计数（累计，用于 mismatch_ratio 指标）
        self._rlsd_total_count = 0
        if self.rlsd_enable:
            logger.info(
                f'[RLSD] enabled | privilege_mode={self.rlsd_privilege_mode} '
                f'lambda_init={self.rlsd_lambda_init} decay_steps={self.rlsd_lambda_decay_steps} '
                f'eps_w={self.rlsd_eps_w} score_span={self.rlsd_score_span}')

        # ------------------------------------------------------------------
        # GCEP（Group-Contrastive Evidence-Chain Privilege）× OPD 配置
        # ------------------------------------------------------------------
        # 默认 vanilla_grpo：所有 GCEP 路径关闭，训练行为与现状逐字节等价。
        # 非 vanilla 模式：冻结 teacher 延迟加载（首次 teacher 前向时），mixed
        # group 的 privilege plan 在 _generate_and_score_completions 里计算，
        # 重逻辑全部在 gcep/trainer_integration.py（ 统一 mode switch）。
        from .gcep.trainer_integration import GCEPConfig as _GCEPConfig
        self.gcep_config = _GCEPConfig.from_env()
        self._gcep_teacher_loader = None
        self._gcep_teacher = None
        self._gcep_archiver = None
        self._gcep_vqa_col = None
        if self.gcep_config.enabled:
            logger.info(
                f'[GCEP] enabled | mode={self.gcep_config.training_mode} '
                f'teacher_ckpt={self.gcep_config.resolve_teacher_ckpt()} '
                f'lambda_opd={self.gcep_config.lambda_opd} topk={self.gcep_config.topk} '
                f'judge_concurrency={self.gcep_config.judge_concurrency} '
                f'archive={self.gcep_config.archive}')

        self.parameter_groups, self.parameter_groups_no_lora = self.split_batches()
        self.use_fast_infer = self.use_vllm  # whether to use the PT backend
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                                  'Please install vLLM with `pip install vllm -U` to use it.')
            if self.vllm_mode == 'server':
                self.vllm_client: VLLMClient = vllm_client
            elif self.vllm_mode == 'colocate':
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f'vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size '
                        f'({self.accelerator.num_processes}) evenly.')

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration([
                        list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                        for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                    ])

                self.engine = self.prepare_vllm(model)
                # Avoid thread-unsafe modifications of the mode.
                self.engine.default_template = copy(self.template)  # Avoid thread-unsafe modifications of the mode.
        else:
            from swift.llm import PtEngine
            self.engine = PtEngine.from_model_template(self.model, copy(self.template), max_batch_size=0)  # 0: no limit

        self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation
        self.request_config = RequestConfig(
            n=1,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop_words,
        )
        print("-"* 15 + "Stop Words" + "-"* 15)
        print(self.request_config.stop)
        print("-"* 40)
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True)

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle. # noqa
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        if self.async_generate:
            self.add_callback(GRPOCallback(self))

        if self.args.dynamic_sample:
            self.resample_dataset = deepcopy(self.train_dataset)

            def cyclic_iter(iterable):
                while True:
                    for x in iterable:
                        yield x

            self.resample_iterator = cyclic_iter(self.get_resample_dataloader())
        # flag indicating whether the evaluation has started
        self.eval_flag = False
        self.gen_iter_limit = int(os.getenv('GEN_ITER_LIMIT', 4))
        print(f"Current ITERATION LIMIT: {self.gen_iter_limit}")

    @profiling_decorator
    def _prepare_inputs(
            self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = 'train' if self.model.training else 'eval'
        if mode == 'train':
            generate_every = self.args.gradient_accumulation_steps * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                accumulated_local_batch = self._generate_and_score_completions(accumulated_local_batch)
                self._buffered_inputs = accumulated_local_batch  # < this is the change
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(accumulated_local_batch)
        return inputs

    def split_batches(self):
        """Sync weights in batches
        Only split LLM layers for now:
        1. N batches for layers
        2. other, embeds, lm_heads in one batch
        3. multi-modal components in one batch
        """
        model = self.accelerator.unwrap_model(self.model)
        if self.args.move_model_batches is None:
            # All in one
            return [[n for n, p in model.named_parameters() if 'ref_model' not in n]], [None]

        model_arch = get_model_arch(model.model_meta.model_arch)
        non_llm_parameters = []
        llm_embeds = []
        parameters = []
        pattern = r'\.(\d+)\.'

        layer_count = None
        # Get the number of layers in LLM modules
        for name, module in model.named_modules():
            if isinstance(module, ModuleList):
                if model_arch is not None and isinstance(model_arch, MultiModelKeys):
                    llm = model_arch.language_model
                    vision_tower = model_arch.vision_tower
                    if any(vt in name for vt in vision_tower):
                        continue
                    if isinstance(llm, list):
                        llm = llm[0]
                    if name.startswith('base_model'):
                        name = name.replace('base_model.', '')
                    if llm in name:
                        layer_count = len(module)
                else:
                    layer_count = len(module)
        assert layer_count is not None, 'Cannot find ModuleList to split modules.'

        n_layers = ceil(layer_count / self.args.move_model_batches)
        for _ in range(self.args.move_model_batches):
            parameters.append([])

        def replace_lora(name):
            if 'lora_' in name:
                return ''
            else:
                return name.replace('base_layer.', '')

        def remove_lora_and_prefix(names):
            names = set([re.sub(r'^_model\.', '', replace_lora(n)) for n in names])
            return [n for n in names if n]

        def split_llm(name):
            match = re.search(pattern, name)
            if match:
                number = match.group(1)
                group = int(number) // n_layers
                parameters[group].append(name)
            else:
                llm_embeds.append(name)

        for name, parameter in model.named_parameters():
            if 'ref_model' in name:
                continue
            if model_arch is not None and isinstance(model_arch, MultiModelKeys):
                llm = model_arch.language_model
                vision_tower = model_arch.vision_tower
                if any(vt in name for vt in vision_tower):
                    non_llm_parameters.append(name)
                elif isinstance(llm, list):
                    llm = llm[0]
                    if llm in name:
                        split_llm(name)
                    else:
                        non_llm_parameters.append(name)
            else:
                split_llm(name)

        if llm_embeds:
            parameters.append(llm_embeds)
        if non_llm_parameters:
            parameters.append(non_llm_parameters)
        parameters = [p for p in parameters if p]
        parameters_no_lora = [remove_lora_and_prefix(p_list) for p_list in parameters]
        return parameters, parameters_no_lora

    def prepare_vllm(self, model):
        from swift.tuners import Swift
        # from swift.llm.infer.infer_engine import GRPOVllmEngine
        from swift.llm.infer.infer_engine import GRPOVllmEngine_ATS
        if self.vllm_tensor_parallel_size > 1:
            vllm_kwargs = {'distributed_executor_backend': 'external_launcher'}
        else:
            vllm_kwargs = {}

        engine_kwargs = {'seed': self.accelerator.process_index // self.vllm_tensor_parallel_size}

        max_num_seqs = (
            self.args.per_device_train_batch_size * self.vllm_tensor_parallel_size
            * self.args.gradient_accumulation_steps)
        current_device = get_device()
        self.template.O3 = self.O3
        with Swift.grpo_context(model, self.template.processor):
            # engine = GRPOVllmEngine(
            engine = GRPOVllmEngine_ATS(
                model.model_dir,
                model.model_info.torch_dtype,
                model_type=model.model_meta.model_type,
                tensor_parallel_size=self.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                enable_prefix_caching=self.args.vllm_enable_prefix_caching,
                max_num_seqs=max_num_seqs,
                enforce_eager=self.args.vllm_enforce_eager,
                limit_mm_per_prompt=self.args.vllm_limit_mm_per_prompt,
                enable_sleep_mode=self.args.sleep_level > 0,
                use_async_engine=False,
                device=current_device,
                max_model_len=self.args.vllm_max_model_len,
                engine_kwargs=engine_kwargs,
                **vllm_kwargs)
            engine.default_template = self.template
        return engine

    @contextmanager
    def _template_context(self, template):
        # The max_length for prompt and completion has already been restricted, so there is no need for max_length here.
        max_length = template.max_length
        mode = template.mode
        if mode in {'vllm', 'pt', 'lmdeploy'}:
            template.set_mode('train')
        template.max_length = None
        loss_scale = template.loss_scale
        if self.multi_turn_func:
            template.loss_scale = loss_scale_map['default']()
        try:
            yield
        finally:
            template.loss_scale = loss_scale
            template.set_mode(mode)
            template.max_length = max_length

    @profiling_decorator
    def _move_model_to_vllm(self):
        if self.vllm_mode == 'server':
            return super()._move_model_to_vllm()

        from accelerate.utils.other import is_compiled_module

        for i, parameter_group in enumerate(self.parameter_groups):
            parameter_group_no_lora = self.parameter_groups_no_lora[i]
            with unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    gather_parameters=parameter_group) as unwrapped_model:

                if is_compiled_module(unwrapped_model):
                    unwrapped_model = unwrapped_model._orig_mod
                if is_peft_model(unwrapped_model):
                    with patch_lora_merge(unwrapped_model, parameter_group):
                        unwrapped_model.merge_adapter()
                    state_dict = unwrapped_model.state_dict()
                    # Remove base_model and base_layer prefixes
                    state_dict = {
                        k.removeprefix('base_model.model.').replace('.base_layer', ''): v
                        for k, v in state_dict.items()
                    }
                    # Remove values with adapter prefix (example: "_lora")
                    state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                    # When module to save, remove its prefix and discard the original module
                    state_dict = {
                        k.replace('modules_to_save.default.', ''): v
                        for k, v in state_dict.items() if 'original_module' not in k
                    }
                else:
                    state_dict = unwrapped_model.state_dict()
                if parameter_group_no_lora:
                    parameter_group_no_lora = [n.replace('base_model.model.', '') for n in parameter_group_no_lora]
                    state_dict = {k: v for k, v in state_dict.items() if k in parameter_group_no_lora}
                assert len(state_dict) > 0 and all([state.shape != torch.Size([0]) for state in state_dict.values()])
                if self.use_fast_infer:
                    if self.args.async_generate:
                        # before sync weight, we should wait async generate finish
                        self._wait_queue()
                    if self.args.use_vllm:
                        llm_model = self.engine.inner_model
                    else:
                        llm_model = self.engine.engine.engine
                    llm_model.load_weights(state_dict.items())
                    del state_dict
                    gc_collect()
                # Unmerge the adapter to restore the model to its original state.
                # This must be done after loading weights to ensure they correspond to the merged state.
                if is_peft_model(unwrapped_model):
                    with patch_lora_unmerge(unwrapped_model):
                        unwrapped_model.unmerge_adapter()
        if self.use_vllm and self.vllm_mode == 'colocate':
            # since vLLM model weights has been updated, we should reset the prefix cache
            self.engine.engine.reset_prefix_cache()

    def _wait_queue(self):
        while self._queue.empty():
            time.sleep(0.01)

    def _infer(self,
               inputs: Optional[InputsType],
               request_config: RequestConfig,
               is_global_inputs: bool = False) -> OutputsType:
        from swift.llm.infer.protocol import ChatCompletionResponse
        request_config = copy(request_config)
        # keys from InferRequest
        per_device_size = len(inputs)
        if is_global_inputs:
            per_device_size //= self.accelerator.num_processes
        infer_inputs = [{
            k: v
            for k, v in inp.items() if k in ['messages', 'images', 'audios', 'videos', 'tools', 'objects']
        } for inp in inputs] if inputs else []
        if self.vllm_mode == 'server':
            # for server mode, we gather all the inputs and send to remote vllm server in main process
            if is_global_inputs:
                all_inputs = infer_inputs
                all_input_lengths = [per_device_size] + [0] * (self.accelerator.num_processes - 1)
            else:
                all_inputs = gather_object(infer_inputs)
                all_input_lengths = gather_object([len(infer_inputs)])

            if not any(inputs for inputs in all_inputs):
                return []

            if self.accelerator.is_main_process:
                results: List[ChatCompletionResponse] = self._engine_infer(
                    infer_requests=all_inputs, request_config=request_config)
            else:
                results = [None] * len(all_inputs)
            # Broadcast the results from the main process to all processes,
            # ensuring each process receives its corresponding slice.
            if not is_global_inputs:
                results = broadcast_object_list(results, from_process=0)
                start_idx = sum(all_input_lengths[:self.accelerator.process_index])
                end_idx = start_idx + all_input_lengths[self.accelerator.process_index]
                results = results[start_idx:end_idx]
            else:
                results = results if self.accelerator.is_main_process else []
        else:
            # pt / vllm
            if self.vllm_tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                # Note: The input sizes may differ across ranks (e.g., in multi-turn scenarios,
                # the amount of data each rank continues to process may vary).
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                local_input_length = len(inputs)
                all_input_lengths = [None] * self.vllm_tensor_parallel_size
                torch.distributed.all_gather_object(all_input_lengths, local_input_length, group=self.tp_group)
                start_idx = sum(all_input_lengths[:local_rank_in_group])
                end_idx = start_idx + all_input_lengths[local_rank_in_group]

                # orig_size = len(inputs)/
                gathered_inputs = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_inputs, inputs, group=self.tp_group)
                inputs = [p for sublist in gathered_inputs for p in sublist]
            # Set request_config.seed
            # 1. Ensure that the seed for vLLM Engines within each TP (Tensor Parallelism) group is the same;
            #   otherwise, the program may hang.
            # 2. Ensure that the seed for vLLM Engines across different TP groups is different;
            #   otherwise, identical completions will be generated.
            mode = 'train' if self.model.training else 'eval'
            batch_size = (
                self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps if mode == 'train' else self.args.per_device_eval_batch_size)
            # Since the TP (Tensor Parallelism) group gathers the inputs,
            # multiply the batch size by the TP parallel size.
            batch_size *= self.vllm_tensor_parallel_size
            request_config.seed = batch_size * (self.accelerator.process_index // self.vllm_tensor_parallel_size)

            results: List[ChatCompletionResponse] = self._engine_infer(
                infer_requests=inputs, request_config=request_config)

            if self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                results = results[start_idx:end_idx]

        return results

    def _set_inputs_system(self, inputs: InputsType) -> InputsType:
        if all(_input['messages'][0]['role'] == 'system' for _input in inputs):
            return
        for _input in inputs:
            messages = _input['messages']
            if messages[0]['role'] != 'system':
                messages.insert(0, {'role': 'system', 'content': self.template.template_meta.default_system})

    def _infer_single_or_multi_turn(self,
                                    inputs: InputsType,
                                    request_config: RequestConfig,
                                    is_global_inputs: bool = False) -> OutputsType:
        """Perform multi-turn or single-turn inference

        Args:
            inputs: list of input requests
            request_config: Inference configuration parameters
            is_global_inputs:
                A boolean indicating whether the inputs are global. When set to True,
                the returned results in the main process will be a complete list of
                global_outputs, while other processes will return an empty list [].
        Returns:
            List of outputs where each entry contains:
            - List of responses per prompt
            - Each response is a tuple of (message_history, finish_reason)
        """
        self._set_inputs_system(inputs)
        # infer first turn
        results = self._infer(inputs, request_config, is_global_inputs)

        if not self.multi_turn_func:
            # Single-turn: combine completions with messages and retain the finish reason.
            outputs = []
            for i, output in enumerate(results):
                _choices = []
                for choice in output.choices:
                    _input: Dict = deepcopy(inputs[i])
                    InferRequest.remove_response(_input['messages'])
                    _input['messages'].append({'role': 'assistant', 'content': choice.message.content})
                    _choices.append((_input['messages'], choice.finish_reason))
                outputs.append(_choices)
            # flatten 2D list to 1D list
            outputs = [item for sublist in outputs for item in sublist]
        else:
            # Multi-turn: continue to rollout until finished.
            orig_size = len(inputs)
            outputs = [None] * orig_size
            # we remove origin response in first turn
            first_turn = True
            next_turn_inputs = inputs.copy()
            last_turn_results = results
            while True:
                has_local_data = len(next_turn_inputs) > 0
                has_global_data = gather_object([has_local_data])
                if not any(has_global_data):
                    break
                # inputs for current turn
                current_inputs = []
                cnt = 0
                # combine completions from results with messages
                for i, output in enumerate(last_turn_results):
                    for choice in output.choices:
                        current_input = deepcopy(next_turn_inputs[i])
                        messages = current_input['messages']

                        # Determine whether to append a new message or update the last one based on the current state
                        if first_turn or not messages[-1]['content'] or messages[-1]['content'] == '<None>':
                            # If it's the first turn or the last message content is empty(dummy), remove the response
                            InferRequest.remove_response(messages)
                        if messages[-1]['role'] == 'assistant':
                            # If the last message was assistant, concatenate the new content to it
                            messages[-1]['content'] += choice.message.content
                        else:
                            # append a new message from the assistant
                            messages.append({'role': 'assistant', 'content': choice.message.content})

                        if 'index' not in current_input:
                            current_input['index'] = cnt
                        current_input['finish_reason'] = choice.finish_reason
                        cnt += 1
                        current_inputs.append(current_input)

                # Process messages in the multi-turn function
                current_results: List[Dict] = self.multi_turn_func(current_inputs) if has_local_data else []

                # Retain messages that are not yet finished for the next round of rollout
                pending_inputs = []
                for r in current_results:
                    if r['finished'] or r['finish_reason'] == 'length':
                        outputs[r['index']] = (r['messages'], r['finish_reason'])
                    else:
                        if r['messages'][-1]['role'] == 'assistant':
                            # Sometimes, after processing with multi_turn_func,
                            # we want to continue reasoning based on the previous assistant content.
                            # However, _infer will remove the response internally, so we add a dummy response here
                            # to prevent the assistant content from being removed.
                            r['messages'].append({'role': 'assistant', 'content': '<None>'})
                        pending_inputs.append(r)

                current_infer_inputs = pending_inputs if has_local_data else []
                current_results = self._infer(current_infer_inputs, request_config)

                last_turn_results = current_results
                next_turn_inputs = pending_inputs
                first_turn = False

            assert not any([o is None for o in outputs])
        return outputs

    def async_infer(self, all_inputs):
        current_queue = self._queue

        def infer_task():
            try:
                with self.multi_turn_completion_length_context():
                    return self._infer_single_or_multi_turn(all_inputs, self.request_config, is_global_inputs=True)
            except Exception as e:
                logger.error('Inference task failed: %s', str(e))
                raise

        future: Future = self.executor.submit(infer_task)

        # pre-fetch the queue to avoid switching back to eval_queue at the end of training sample sampling

        def done(future):
            try:
                result = future.result()
                current_queue.put(DataCache(all_inputs, result))
            except Exception as e:
                logger.error('Error in async_infer callback: %s', str(e))

        future.add_done_callback(done)

    def _prefetch(self, dataloader: DataLoader):
        inputs = next(iter(dataloader))
        all_inputs = gather_object(inputs)
        outputs = self._infer_single_or_multi_turn(all_inputs, self.request_config, is_global_inputs=True)
        self._queue.put(DataCache(all_inputs, outputs))


    def process_round(self, gen_inputs, gen_outputs, previous_execution_contexts, cur_round=1000):
        if len(gen_inputs) == 0:
            return gen_inputs, previous_execution_contexts
        cache_dir = os.path.join(
            os.getenv("THYME_INFER_ROOT",
                      "thyme-infer"),
            f"outputs/thyme_debug_cache/RL_case_cache_new/{self.start_time}_{self.accelerator.process_index}")
        os.makedirs(cache_dir, exist_ok=True)

        assert len(gen_outputs) == len(gen_inputs)

        for idx in range(len(gen_outputs)):
            # previous_execution_context = previous_execution_contexts[idx]
            if gen_outputs[idx] is not None:
                gen_output = gen_outputs[idx][0][-1]
                assert gen_output["role"] == "assistant", f"Line 937 assert error: {gen_outputs[idx][0]}"
                if gen_inputs[idx]["messages"][-1]["role"] == "assistant":
                    if "Execution timed out after" in gen_inputs[idx]["messages"][-1]["content"]:
                        context = gen_inputs[idx]["messages"][-1]["content"]
                        print(f"Already Timeout: {context}")
                        continue
                    if "</answer>" in gen_inputs[idx]["messages"][-1]["content"]:
                        # context = gen_inputs[idx]["messages"][-1]["content"]
                        # print(f"Already Timeout: {context}")
                        continue
                response = gen_output["content"]

                # 生成侧源头拦截模块（B1-B4，GEN_INTERCEPT=1 启用；默认关=零影响）。
                # 惰性加载一次并缓存；详见 gcep/gen_intercept.py 的机制与校准说明。
                _gi = getattr(self, "_gi_mod", None)
                if _gi is None:
                    try:
                        from .gcep import gen_intercept as _gi
                    except ImportError:
                        from gcep import gen_intercept as _gi
                    self._gi_mod = _gi
                _gi_stats = getattr(self, "_gi_stats", None)
                if _gi_stats is None:
                    _gi_stats = _gi.stats_init()
                    self._gi_stats = _gi_stats

                # Release patch: response-level length guard. Some samples emit a single
                # runaway assistant turn — e.g. a ~790k-char base64 blob or endlessly
                # repeated text — that alone pushes the accumulated sequence past the
                # model max (observed 313420 / 580030 tokens > 131072). Once such content
                # is appended to messages, the NEXT round feeds it into vLLM.generate,
                # which raises and desynchronizes one rank from the Zero3 all-gather,
                # deadlocking all 8 ranks until the 30-min NCCL watchdog aborts them. We
                # truncate the response here, BEFORE it is appended, and force-close the
                # turn with </answer> so process_round's "</answer> in content" short-
                # circuit ends this sample on the next round. This is the correct
                # interception point: my earlier post-process_round char guard ran too
                # late (the runaway turn had already been fed to the engine). Char-count
                # is a zero-cost proxy for tokens; the cap is far above any legitimate
                # completion (~360 tokens) yet well below the pathological blow-ups.
                _o3_max_resp_chars = int(os.getenv('O3_MAX_RESPONSE_CHARS', '40000'))
                if len(response) > _o3_max_resp_chars:
                    if self.accelerator.process_index == 1:
                        print(f"[O3 length guard] truncating runaway response "
                              f"{len(response)} chars -> {_o3_max_resp_chars} + </answer>")
                    response = response[:_o3_max_resp_chars] + "\n</answer>"

                # 生成侧拦截 B3/B4：复读熔断 / 单回合长度熔断。
                # 终止方式 = 截断 + 追加自然的 "</answer>"（使本轮跳过代码执行、下一轮
                # process_round 短路该样本）。**零暴露**：不再写
                # [xxx-terminated] 这类模型没见过的 in-band 标记，reason 挂在样本 dict 上
                # 交给 traj_filter（metadata 优先级高于文本判定）；追加的后缀在
                # _prepare_batch_inputs 里从 labels 抹掉（suffix_span），人工 token 零梯度。
                if _gi.enabled():
                    _acc = 0
                    _acc_text = ""
                    _last_msg = gen_inputs[idx]["messages"][-1] if gen_inputs[idx]["messages"] else {}
                    if _last_msg.get("role") == "assistant":
                        _acc_text = str(_last_msg.get("content", ""))
                        _acc = len(_acc_text)
                    _gi_reason = _gi.check_turn(response, _acc, _acc_text)
                    if _gi_reason:
                        response, _gi_suffix = _gi.terminate(response, _gi_reason, _acc)
                        gen_inputs[idx]["gi_reason"] = _gi_reason
                        gen_inputs[idx]["gi_suffix"] = _gi_suffix
                        _gi_stats[_gi_reason] = _gi_stats.get(_gi_reason, 0) + 1
                        if self.accelerator.process_index == 1:
                            print(f"[gen-intercept] terminate reason={_gi_reason} "
                                  f"acc={_acc} resp={len(response)}")

                if response.strip() in ["</code>", "</answer>", "<|im_end|>"]:
                    response = ""
                if has_repeated_content(response):
                    response = ""
                
                if ( "images" not in gen_inputs[idx] or
                    not gen_inputs[idx]["images"] or
                    not isinstance(gen_inputs[idx]["images"][0], dict) or
                    "path" not in gen_inputs[idx]["images"][0]):
                    
                    raise ValueError(
                        "A training sample is missing its required image path; "
                        "provide an images list with a path entry before rollout."
                    )
                else:
                    image_path = gen_inputs[idx]["images"][0]["path"]
                # ATS-FIX（门控 GCEP_SANDBOX_B64_IMAGE_FIX=1）：base64 原图
                # 物化成临时真实文件，让 sandbox 的 Image.open(image_path) 真的能读到图。
                # 默认关 → 对一切既有 run 逐字节等价。
                if _sandbox_b64_fix_enabled():
                    image_path = _materialize_b64_image(image_path, cache_dir)
                case_id, case_ext = os.path.splitext(os.path.basename(image_path))
                # Release patch: sanitize case_id to avoid OSError [Errno 36] File name
                # too long. The model sometimes produces an image whose `image_path`
                # basename is an entire base64 PNG blob (350+ chars, ending in
                # "SUVORK5CYII="); using that as a cache subdir name blows past the
                # filesystem 255-byte limit at line ~1009 `os.makedirs(case_dir)`,
                # raising OSError and tearing down the whole NCCL group. Cap to 48
                # chars and append a 16-char sha1 suffix so cache dirs stay unique
                # and short.
                if len(case_id) > 64:
                    import hashlib
                    case_id = case_id[:48] + "_" + hashlib.sha1(
                        case_id.encode("utf-8", "ignore")
                    ).hexdigest()[:16]
                # B1：base64 形态 case_id 消毒（GEN_INTERCEPT=1；内部自带 enabled 判定，
                # 关闭时恒等返回）。把 "c29ekge7...SUVORK5CYII=" 之类的名字换成 img_<sha1[:8]>，
                # 从源头去掉 "Sandbox for <base64>: ..." 的回显污染。
                _cid_san = _gi.sanitize_case_id(case_id)
                if _cid_san != case_id:
                    _gi_stats["case_id"] = _gi_stats.get("case_id", 0) + 1
                    case_id = _cid_san
                generate_code_round = False

                if "<code>" in self.request_config.stop and gen_inputs[idx]["messages"][-1]["role"] == "assistant" and  gen_inputs[idx]["messages"][-1]["content"].strip().endswith("<code>"):
                    generate_code_round = True
                # if "images" not in gen_inputs[idx] or not gen_inputs[idx]["images"] or not len(gen_inputs[idx]["images"]) > 10:
                if not len(gen_inputs[idx]["images"]) > 10:
                    if gen_inputs[idx]["messages"][-1]["role"] == "assistant":
                        gen_inputs[idx]["messages"][-1]["content"] += response
                    else:
                        gen_inputs[idx]["messages"].append({"role": "assistant", "content": response})
                else:
                    continue
                # 开始O3的处理流程, 必须在geninputs之后continue
                if "</answer>" in response:
                    continue
                # Case 2: reach code generation.
                # parse current result. Two cases: reach </code> or reach </answer>
                code_regex = re.compile(r'<code>\s*(?:```\s*)?(?:python\s*)?([\s\S]*?)\s*(?:```\s*)?</code>', re.IGNORECASE)
                if generate_code_round:
                    response = "<code>\n" + response
                code_match = code_regex.search(response)


                # execute code and return result.
                if code_match:
                    gen_inputs[idx]["messages"][-1]["content"] += "<sandbox_output>"
                    code_to_execute = code_match.group(1).strip()
                    case_dir = os.path.join(cache_dir, f"{case_id}_{self.accelerator.process_index}_rnd{cur_round}_{idx}")
                    os.makedirs(case_dir, exist_ok=True)
                    
                    # has_valid_images = False
                    processed_img_paths, captured_stdout, error_msg, current_execution_context = execute_code_in_sandbox(
                        code_to_execute, image_path, temp_output_dir=case_dir, item_id=case_id, previous_execution_context=previous_execution_contexts[idx]
                    )
                    # current_execution_context = _remove_unpickable_values()
                    # print(f"\033[31m--- Found Code Block ---\n{code_to_execute}\n-------------------------\033[0m")
                    previous_execution_contexts[idx] = _remove_unpickable_values(current_execution_context)
                    # import pdb; pdb.set_trace()
                    if not processed_img_paths:
                        if error_msg is None:
                            gen_inputs[idx]["messages"][-1]["content"] += f'Sandbox for {case_id}: No output captured.'
                            print("-" * 30)
                            print(f"Corner Case: {processed_img_paths}")
                            print(f"Code: {code_to_execute}")
                            print("-" * 30)
                        else:
                            # Release patch: truncate runaway sandbox error text BEFORE it is
                            # appended. A sandbox failure can echo a huge argument back — e.g.
                            # "[Errno 36] File name too long: '<790k-char base64>'" when the model
                            # passes a base64 image string as a filename. Appending that verbatim
                            # blows the next round's sequence past the model max (observed 416189
                            # tokens > 131072) and NCCL-deadlocks all ranks. This path (sandbox
                            # output) bypasses the response-level guard at line ~962, so it needs
                            # its own cap. Char-count proxy; sandbox text is normally short.
                            _o3_max_sb = int(os.getenv('O3_MAX_SANDBOX_CHARS', '20000'))
                            _emsg = error_msg if isinstance(error_msg, str) else str(error_msg)
                            # B2：回显消毒（GEN_INTERCEPT=1；内部自带 enabled 判定）。
                            # 把 sandbox 注入文本里的 base64 长串换成占位符——保留
                            # 错误类型（学习信号），但不让几 KB 垃圾进入上下文。
                            # 放在长度截断之前，让 blob 检测看到完整串。
                            _emsg_san = _gi.sanitize_echo(_emsg)
                            if _emsg_san != _emsg:
                                _gi_stats["blob"] = _gi_stats.get("blob", 0) + 1
                                _emsg = _emsg_san
                            if len(_emsg) > _o3_max_sb:
                                if self.accelerator.process_index == 1:
                                    print(f"[O3 length guard] truncating sandbox error "
                                          f"{len(_emsg)} chars -> {_o3_max_sb}")
                                _emsg = _emsg[:_o3_max_sb] + " ...[truncated]"
                            gen_inputs[idx]["messages"][-1]["content"] += _emsg
                        # continue
    
                    # has_valid_images = False
                    # generated_content += [
                    #                     {"type": "text", "text": generated_text_segment},
                    #                     {"type": "text", "text": "<sandbox_output>"}
                    #                 ]
                    else:
                        #     import importlib
                        #     importlib.reload(os)
                        # first_path = processed_img_paths[0]
                        first_path = processed_img_paths[0]
                        # gen_inputs[idx]["messages"][-1]["content"] += "<image>"
                        if os.path.exists(first_path):
                            
                            # Iterate through each path in the list
                            valid_aspect_ratio = True
                            for img_path in processed_img_paths:
                                if os.path.exists(img_path):
                                    if not check_aspect_ratio(img_path):
                                        print(f"Invalid image: {img_path}")
                                        valid_aspect_ratio = False
                                        break
                            if valid_aspect_ratio:
                                for img_path in processed_img_paths:
                                    if os.path.exists(img_path):
                                        # if not has_valid_images: # Add text segments only once per sandbox output block
                                        #     has_valid_images = True
                                        if not len(gen_inputs[idx]["images"]) > 10:
                                            gen_inputs[idx]["messages"][-1]["content"] += "<image>"
                                            gen_inputs[idx]["images"].append({"bytes": None, "path": os.path.abspath(img_path)})
                                        else:
                                            break
                            else:
                                gen_inputs[idx]["messages"][-1]["content"] += f"Sandbox for {case_id}: The Generated img has invalid aspect_ratio"

                                    # generated_content.append({"type": "image", "image": img_path})                                
                        else:
                            # generated_content.append({"type": "text", "text": first_path})
                            # Release patch: same length guard as the error_msg path — a
                            # non-image sandbox result string can also be pathologically long.
                            _o3_max_sb = int(os.getenv('O3_MAX_SANDBOX_CHARS', '20000'))
                            _fp = first_path if isinstance(first_path, str) else str(first_path)
                            if len(_fp) > _o3_max_sb:
                                _fp = _fp[:_o3_max_sb] + " ...[truncated]"
                            gen_inputs[idx]["messages"][-1]["content"] += _fp
                    gen_inputs[idx]["messages"][-1]["content"] += "</sandbox_output>"
                else:
                    # wo code. wo </answer>, 那么基本上一直在生成重复内容，直接break这次
                    if "</answer>" not in response and response != "" and not response.strip().endswith("<code>"):
                        response = ''
                        print(f'wo code. wo </answer>: {response}')
        return gen_inputs, previous_execution_contexts


    def _o3_infer(self, inputs: InputsType) -> Tuple[InputsType, OutputsType]:
        # print(f"Input size: {len(inputs)}")
        # process_index = self.infer_rank
        previous_execution_contexts = [{} for i in range(len(inputs))]
        # from swift.llm.infer.protocol import ChatCompletionResponseChoice, ChatMessage, ChatCompletionResponse
        iteration_limit = self.gen_iter_limit
        gen_inputs = deepcopy(inputs)
        # num_prompt_tokens = [0 for i in range(len(inputs))]
        # num_completion_tokens = [0 for i in range(len(inputs))]
        while iteration_limit > 0:
            # if self.accelerator.process_index == 1:
            #     print(f"Start iteration {3 - iteration_limit}.")
            iteration_limit -= 1
            try:
                gen_inputs, gen_outputs = self._fast_infer(gen_inputs)
            except OSError as e:
                gen_outputs = [None] * len(gen_inputs)
                for idx in range(len(gen_outputs)):
                    # 提取role为user和system的部分
                    filtered_messages = [msg for msg in gen_inputs[idx]["messages"] if msg["role"] in ["user", "system"]]

                    # 添加一个role为assistant的message，content为IndexError
                    if (gen_inputs[idx]["messages"][-1]["role"] == "assistant" and not ("This instance triggers" in gen_inputs[idx]["messages"][-1]["content"]) and not ("</answer>" in gen_inputs[idx]["messages"][-1]["content"])) or (gen_inputs[idx]["messages"][-1]["role"] != "assistant"):
                        filtered_messages.append({"role": "assistant", "content": "This instance triggers OSError"})

                        # 进行赋值
                        gen_outputs[idx] = (filtered_messages, "stop")
                    else:
                        gen_outputs[idx] = None
                if self.accelerator.process_index == 1:
                    print(f"OS Error: {e}")
                try:
                    with open("./debug_os_error.jsonl", "a") as fout:
                        fout.write(f"Wrong Case: {e}\n")
                        for i in range(len(gen_inputs)):
                            fout.write(json.dumps(gen_inputs[i]) +"\n")
                            fout.write(json.dumps(gen_outputs[i]) +"\n")
                except Exception:
                    pass
            except torch.OutOfMemoryError as e:
                gen_outputs = [None] * len(gen_inputs)
                for idx in range(len(gen_outputs)):
                    # 提取role为user和system的部分
                    filtered_messages = [msg for msg in gen_inputs[idx]["messages"] if msg["role"] in ["user", "system"]]

                    if (gen_inputs[idx]["messages"][-1]["role"] == "assistant" and not ("This instance triggers" in gen_inputs[idx]["messages"][-1]["content"]) and not ("</answer>" in gen_inputs[idx]["messages"][-1]["content"])) or (gen_inputs[idx]["messages"][-1]["role"] != "assistant"):
                        filtered_messages.append({"role": "assistant", "content": "This instance triggers OutOfMemoryError"})

                        # 进行赋值
                        gen_outputs[idx] = (filtered_messages, "stop")
                    else:
                        gen_outputs[idx] = None
                if self.accelerator.process_index == 1:
                    print("Out of memory!")
                try:
                    with open("./debug_oom.jsonl", "a") as fout:
                        fout.write(f"Wrong Case: {e}\n")
                        for i in range(len(gen_inputs)):
                            fout.write(json.dumps(gen_inputs[i]) +"\n")
                            fout.write(json.dumps(gen_outputs[i]) +"\n")
                except Exception:
                    pass
            except ValueError as e:
                gen_outputs = [None] * len(gen_inputs)
                for idx in range(len(gen_outputs)):
                    # 提取role为user和system的部分
                    filtered_messages = [msg for msg in gen_inputs[idx]["messages"] if msg["role"] in ["user", "system"]]

                    # 添加一个role为assistant的message，content为IndexError
                    if (gen_inputs[idx]["messages"][-1]["role"] == "assistant" and not ("This instance triggers" in gen_inputs[idx]["messages"][-1]["content"]) and not ("</answer>" in gen_inputs[idx]["messages"][-1]["content"])) or (gen_inputs[idx]["messages"][-1]["role"] != "assistant"):
                        filtered_messages.append({"role": "assistant", "content": "This instance triggers ValueError"})

                        # 进行赋值
                        gen_outputs[idx] = (filtered_messages, "stop")
                    else:
                        gen_outputs[idx] = None
                if self.accelerator.process_index == 1:
                    print(f"Value Error: {e}")
                try:
                    with open("./debug_value.jsonl", "a") as fout:
                        fout.write(f"Wrong Case: {e}\n")
                        for i in range(len(gen_inputs)):
                            fout.write(json.dumps(gen_inputs[i]) +"\n")
                            fout.write(json.dumps(gen_outputs[i]) +"\n")
                except Exception:
                    pass
            except IndexError as e:
                gen_outputs = [None] * len(gen_inputs)
                for idx in range(len(gen_outputs)):
                    # 提取role为user和system的部分
                    filtered_messages = [msg for msg in gen_inputs[idx]["messages"] if msg["role"] in ["user", "system"]]

                    # 添加一个role为assistant的message，content为IndexError
                    if (gen_inputs[idx]["messages"][-1]["role"] == "assistant" and not ("This instance triggers" in gen_inputs[idx]["messages"][-1]["content"]) and not ("</answer>" in gen_inputs[idx]["messages"][-1]["content"])) or (gen_inputs[idx]["messages"][-1]["role"] != "assistant"):
                        filtered_messages.append({"role": "assistant", "content": "This instance triggers IndexError"})

                        # 进行赋值
                        gen_outputs[idx] = (filtered_messages, "stop")
                    else:
                        gen_outputs[idx] = None
                if self.accelerator.process_index == 1:
                    print(f"IndexError: {e}")
                try:
                    with open("./debug_idx.jsonl", "a") as fout:
                        fout.write(f"Wrong Case: {e}\n")
                        for i in range(len(gen_inputs)):
                            fout.write(json.dumps(gen_inputs[i]) +"\n")
                            fout.write(json.dumps(gen_outputs[i]) +"\n")
                except Exception:
                    pass
            except timeout_decorator.TimeoutError as e:
                gen_outputs = [None] * len(gen_inputs)
                for idx in range(len(gen_outputs)):
                    # 提取role为user和system的部分
                    filtered_messages = [msg for msg in gen_inputs[idx]["messages"] if msg["role"] in ["user", "system"]]

                    # 添加一个role为assistant的message，content为IndexError
                    if (gen_inputs[idx]["messages"][-1]["role"] == "assistant" and not ("This instance triggers" in gen_inputs[idx]["messages"][-1]["content"]) and not ("</answer>" in gen_inputs[idx]["messages"][-1]["content"])) or (gen_inputs[idx]["messages"][-1]["role"] != "assistant"):
                        filtered_messages.append({"role": "assistant", "content": "This instance triggers TimeoutError"})

                        # 进行赋值
                        gen_outputs[idx] = (filtered_messages, "stop")
                    else:
                        gen_outputs[idx] = None
                if self.accelerator.process_index == 1:
                    print("Out of time!")
                with open(f"./debug_oot.jsonl", "a") as fout:
                    fout.write(f"Wrong Case: {e}\n")
                    for i in range(len(gen_inputs)):
                        fout.write(json.dumps(gen_inputs[i]) +"\n")
                        fout.write(json.dumps(gen_outputs[i]) +"\n")
            except RuntimeError as e:
                gen_outputs = [None] * len(gen_inputs)
                for idx in range(len(gen_outputs)):
                    # 提取role为user和system的部分
                    filtered_messages = [msg for msg in gen_inputs[idx]["messages"] if msg["role"] in ["user", "system"]]

                    # 添加一个role为assistant的message，content为IndexError
                    if (gen_inputs[idx]["messages"][-1]["role"] == "assistant" and not ("This instance triggers" in gen_inputs[idx]["messages"][-1]["content"]) and not ("</answer>" in gen_inputs[idx]["messages"][-1]["content"])) or (gen_inputs[idx]["messages"][-1]["role"] != "assistant"):
                        filtered_messages.append({"role": "assistant", "content": "This instance triggers RuntimeError"})

                        # 进行赋值
                        gen_outputs[idx] = (filtered_messages, "stop")
                    else:
                        gen_outputs[idx] = None
                if self.accelerator.process_index == 1:
                    print(f"Runtime Error: {e}!")
                try:
                    with open(f"./debug_runtime.jsonl", "a") as fout:
                        fout.write(f"Wrong Case: {e}\n")
                        for i in range(len(gen_inputs)):
                            fout.write(json.dumps(gen_inputs[i]) +"\n")
                            fout.write(json.dumps(gen_outputs[i]) +"\n")
                except Exception:
                    pass
            except Exception as e:
                gen_outputs = [None] * len(gen_inputs)
                for idx in range(len(gen_outputs)):
                    # 提取role为user和system的部分
                    filtered_messages = [msg for msg in gen_inputs[idx]["messages"] if msg["role"] in ["user", "system"]]

                    # 添加一个role为assistant的message，content为IndexError
                    if (gen_inputs[idx]["messages"][-1]["role"] == "assistant" and not ("This instance triggers" in gen_inputs[idx]["messages"][-1]["content"]) and not ("</answer>" in gen_inputs[idx]["messages"][-1]["content"])) or (gen_inputs[idx]["messages"][-1]["role"] != "assistant"):
                        filtered_messages.append({"role": "assistant", "content": "This instance triggers Exception"})

                        # 进行赋值
                        gen_outputs[idx] = (filtered_messages, "stop")
                    else:
                        gen_outputs[idx] = None
                if self.accelerator.process_index == 1:
                    print(f"Exception: {e}")
                try:
                    with open(f"./debug_others.jsonl", "a") as fout:
                        fout.write(f"Wrong Case: {e}\n")
                        for i in range(len(gen_inputs)):
                            fout.write(json.dumps(gen_inputs[i]) +"\n")
                            fout.write(json.dumps(gen_outputs[i]) +"\n")
                except Exception:
                    pass


            gen_inputs, previous_execution_contexts = self.process_round(gen_inputs, gen_outputs, previous_execution_contexts, cur_round=self.gen_iter_limit-iteration_limit)

            # Release patch: O3 multi-turn length guard. Some samples fall into a loop of
            # emitting more images/text without ever producing </answer>, growing the
            # accumulated messages to hundreds of thousands of tokens (observed 580030 >
            # 131072). Such a runaway sample stalls one rank's rollout while other ranks
            # proceed to the Zero3 all-gather, so the collective desynchronizes and the
            # NCCL watchdog aborts all 8 ranks after the 30-min timeout. To bound this,
            # once a sample's accumulated content exceeds O3_MAX_TOTAL_CHARS we force-close
            # it with </answer>; the next while-iteration then skips it (see the
            # "</answer> in ...content" short-circuit in process_round), keeping every rank
            # on an identical collective-communication path. Char-count is a zero-cost
            # proxy for token-count and only needs to catch pathological blow-ups; normal
            # completions are ~360 tokens, far below the threshold.
            _o3_max_chars = int(os.getenv('O3_MAX_TOTAL_CHARS', '200000'))
            for _idx in range(len(gen_inputs)):
                try:
                    _msgs = gen_inputs[_idx].get("messages", [])
                    _total_chars = sum(len(str(_m.get("content", ""))) for _m in _msgs)
                    if _total_chars > _o3_max_chars:
                        _last = _msgs[-1] if _msgs else None
                        if _last is not None and _last.get("role") == "assistant" \
                                and "</answer>" not in str(_last.get("content", "")):
                            _last["content"] = str(_last.get("content", "")) + "\n</answer>"
                            if self.accelerator.process_index == 1:
                                print(f"[O3 length guard] force-closed sample {_idx} at "
                                      f"{_total_chars} chars (> {_o3_max_chars})")
                except Exception:
                    pass

        for idx in range(len(gen_outputs)):
            gen_outputs[idx] = (gen_inputs[idx]["messages"], "stop")
            if "</answer>" in gen_outputs[idx][0][-1]["content"] :
                gen_outputs[idx][0][-1]["content"] = gen_outputs[idx][0][-1]["content"].split("</answer>")[0] + "</answer>"
            assert gen_inputs[idx]["messages"][-1]["role"] == "assistant", f"Input: {gen_inputs[idx]}\n\nOutput: {gen_outputs[idx]}"
            gen_inputs[idx]["messages"] = gen_inputs[idx]["messages"][:-1]
        # with open(f"/path/to/Thyme/scripts/rl/debug_{self.accelerator.process_index}.jsonl", "a") as fout:
        #     fout.write(json.dumps({"final input": gen_inputs[-1], "final output": gen_outputs[-1]}) +"\n")
        return gen_inputs, gen_outputs

    def _fast_infer(self, inputs: InputsType) -> Tuple[InputsType, OutputsType]:
        if self.vllm_mode == 'colocate' and self.args.sleep_level > 0:
            if self.args.offload_model:
                self.offload_model()
            if self.args.offload_optimizer:
                self.offload_optimizer()
            if self.args.gc_collect_after_offload:
                gc_collect()
            # Skip the first wake_up to avoid the warning "Executor is not sleeping"
            if self.engine.inner_model_executor.is_sleeping:
                self.engine.engine.wake_up()
        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        if self.async_generate:
            # send this step data to server
            # we gather inputs outside the thread for prevent potential gather deadlock
            all_inputs = gather_object(inputs)
            self.async_infer(all_inputs)
            # cached data from last step
            data_cache = self._queue.get()
            all_inputs = data_cache.inputs
            all_outputs = gather_object(data_cache.outputs)
            process_slice = slice(
                self.accelerator.process_index * len(inputs),
                (self.accelerator.process_index + 1) * len(inputs),
            )
            inputs = all_inputs[process_slice]
            outputs = all_outputs[process_slice]

        else:
            with self.multi_turn_completion_length_context():
                outputs = self._infer_single_or_multi_turn(inputs, self.request_config)

        if self.vllm_mode == 'colocate' and self.args.sleep_level > 0:
            self.engine.engine.sleep(level=self.args.sleep_level)
            if self.args.gc_collect_after_offload:
                gc_collect()
            if self.args.offload_model:
                self.load_model()
            if self.args.offload_optimizer:
                self.load_optimizer()
        # print(f"inputs size: {len(inputs)}")
        # print(f"outputs size: {len(outputs)}")
        return inputs, outputs

    def _generate_completions(self, inputs: InputsType) -> InputsType:
        """Generate completions for given inputs using either fast inference or standard PyTorch inference.

        Args:
            inputs: List of input examples containing conversation messages.

        Returns:
            Modified inputs with generated completions added to the last message
            and truncation flag set in 'is_truncated' field.
        """
        mode = 'train' if self.model.training else 'eval'
        if self.use_fast_infer:
            # inputs, outputs = self._fast_infer(inputs)
            if not self.O3:
                inputs, outputs = self._fast_infer(inputs)
            else:
                inputs, outputs = self._o3_infer(inputs)
        else:
            # pt infer
            is_multimodal = self.model.model_meta.is_multimodal
            if is_multimodal:
                models = self.template.remove_post_encode_hook()
            with unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ), self.multi_turn_completion_length_context():
                outputs = self._infer_single_or_multi_turn(inputs, self.request_config)
                if mode == 'train':
                    # In training mode, ensure the model is returned to train() mode after inference
                    # This is necessary as pt engines set the model to eval mode during generation
                    self.model.train()
            if is_multimodal:
                self.template.register_post_encode_hook(models)

        for i, output in enumerate(outputs):
            inputs[i]['messages'] = output[0]
            inputs[i]['is_truncated'] = output[1] == 'length'

        return inputs

    def _generate_and_score_completions(self, inputs: InputsType) -> InputsType:

        # 生成侧拦截计数每步重置（统计"这一步"的事件数；没启用时不存在该属性）
        if getattr(self, "_gi_stats", None) is not None:
            self._gi_stats = self._gi_mod.stats_init()
        inputs = self._generate_completions(inputs)
        total_rewards_per_func, total_rewards, completions = self._score_completions(inputs)
        mode = 'train' if self.model.training else 'eval'

        if self.args.dynamic_sample and mode == 'train':
            # dynamic sampling for std=0 groups
            inputs, total_rewards, total_rewards_per_func, completions = \
                self._dynamic_sampling(inputs, total_rewards, total_rewards_per_func, completions)

        # GCEP_TRAJ_FILTER（gcep/traj_filter.py）：含无效轨迹的
        # group 整组 reward 清零（GRPO 优势归零）+ 后续整组不调 judge、无效行不进 OPD。
        # 默认关（GCEP_TRAJ_FILTER=1 启用），关闭时逐字节等价（Qwen 零影响）。
        if self.gcep_config.enabled and mode == 'train':
            from .gcep import traj_filter as _trajf
            if _trajf.traj_filter_enabled():
                # 零暴露路径：生成侧拦截的 reason 走 metadata（reasons 非 None 时优先），
                # 文本标记判定作为没有 metadata 原因时的兼容回退。
                _inv_local, _inv_counts = _trajf.detect_invalid(
                    completions, [b.get('gi_reason') for b in inputs])
                _inv_global = list(gather_object(_inv_local))
                _ng = self.num_generations
                _n_gr = len(_inv_global) // _ng
                _contam = [any(_inv_global[g * _ng:(g + 1) * _ng]) for g in range(_n_gr)]
                # 过滤规则：**行级清零**——invalid 行当错误样本
                # （reward=0），组内其余 valid 行保留真实 reward（VQAORM/FMTORM/CSTORM 显示
                # 恢复真实值，口径 A：非熔断行∧答对 / N），不再整组连坐。
                # judge 侧仍按"组内含 invalid → 整组跳过"（valid 行 clean-teacher OPD，不变）。
                # _contam 保留仅供 gcep/traj_contaminated_group_ratio 遥测。
                if any(_inv_global):
                    _keep = torch.tensor(
                        [0.0 if _inv_global[j] else 1.0 for j in range(len(_inv_global))],
                        dtype=total_rewards.dtype, device=total_rewards.device)
                    total_rewards = total_rewards * _keep
                    total_rewards_per_func = total_rewards_per_func * _keep.unsqueeze(-1)
                _off = self.accelerator.process_index * len(inputs)
                for _i, _inp in enumerate(inputs):
                    _g = _off + _i
                    _inp['traj_invalid'] = bool(_inv_global[_g]) if _g < len(_inv_global) else False
                _inv_n = int(sum(_inv_global))
                self._metrics[mode]['gcep/traj_invalid_ratio'].append(
                    _inv_n / max(1, len(_inv_global)))
                self._metrics[mode]['gcep/traj_contaminated_group_ratio'].append(
                    sum(_contam) / max(1, _n_gr))
                self._metrics[mode]['gcep/traj_invalid_incomplete'].append(float(_inv_counts.get('incomplete', 0)))
                self._metrics[mode]['gcep/traj_observation_overflow'].append(float(_inv_counts.get('observation_overflow', 0)))

        # GCEP（仅 train）：mixed group 判定 → Judge → privilege plan（main rank
        # 计算 + broadcast，任一环节失败该组回退 clean）；随后全量
        # rollout 归档（分 rank 写 shard）。vanilla 模式完全不进入。
        if self.gcep_config.enabled and mode == 'train':
            inputs = self._gcep_attach_privilege_plans(inputs, total_rewards, total_rewards_per_func)
            inputs = self._mt_attach_evidence_plans(inputs)
            self._gcep_archive_rollouts(inputs, total_rewards, total_rewards_per_func)

        # Prepare final outputs with advantages and other required fields
        batch_encoded_inputs = self._prepare_batch_inputs(inputs, total_rewards)
        # Log metrics
        messages = [inputs[i]['messages'][:-1] for i in range(len(inputs))]

        self._log_metrics(batch_encoded_inputs, messages, completions, total_rewards, total_rewards_per_func)

        return batch_encoded_inputs

    def fill_invalid_value(self, reward_value):
        # refer to _prepare_batch_inputs
        grouped_reward_value = reward_value.view(-1, self.num_generations)

        # Create a mask for invalid values 
        invalid_mask = (grouped_reward_value == INVALID_REWARD_VALUE)

        # Calculate the sum and count of valid values for each group
        valid_sum = torch.where(invalid_mask, torch.tensor(0.0), grouped_reward_value).sum(dim=1, keepdim=True)
        valid_count = (~invalid_mask).sum(dim=1, keepdim=True)

        # Avoid division by zero by setting valid_mean to 0 where valid_count is 0
        valid_mean = torch.where(valid_count == 0, torch.tensor(0.0), valid_sum / valid_count)

        # Replace invalid values with the mean of valid values
        grouped_reward_value = torch.where(invalid_mask, valid_mean, grouped_reward_value)

        return grouped_reward_value.view(-1), bool(invalid_mask.sum() > 0)

    def _score_completions(self, inputs: InputsType) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Score completions using all reward functions

        Args:
            inputs: List of input examples, each containing a 'messages' list with conversation history

        Returns:
            Tuple containing:
            - rewards_per_func: Tensor of shape (num_examples, num_reward_funcs) with individual rewards
            - total_rewards: Tensor of shape (num_examples,) with weighted sum of rewards
            - completions: List of generated completion strings
        """
        device = self.accelerator.device
        completions = [example['messages'][-1]['content'] for example in inputs]
        rewards_per_func = torch.zeros((len(inputs), len(self.reward_funcs)), device=device)
        reward_name_list = []
        for i, (reward_func, reward_model_plugin) in enumerate(zip(self.reward_funcs, self.reward_model_plugins)):
            # reward model
            reward_name = str(reward_func)
            reward_name_list.append(reward_name.split(" object at")[0].replace("<agent_rm.", "").strip())
            if isinstance(reward_func, nn.Module):
                rewards_per_func[:, i] = reward_model_plugin(inputs=inputs)
            # reward function
            else:
                # Repeat all input columns (but "messages" and "completion") to match the number of generations
                reward_kwargs = RowPreprocessor.rows_to_batched(inputs)
                output_reward_func = reward_func(completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        total_rewards_per_func = gather(rewards_per_func)
        total_rewards_per_func_before = total_rewards_per_func.clone()

        # 获取vqa_orm对应的索引
        # print(f"Reward functions: {reward_name_list}")
        # cst_orm_index = reward_name_list.index("CSTORM")
        # vqa_orm_index = self.reward_funcs.index('vqa_orm')
        # mask = total_rewards_per_func[:, cst_orm_index] <= 0
        # for depend_reward in ["VQAORM"]:
        #     # depend_reward_func = orms[depend_reward]
        #     if depend_reward in reward_name_list:
        #         depend_reward_index = reward_name_list.index(depend_reward)
        #         total_rewards_per_func[mask, depend_reward_index] = 0
        vqa_orm_index = reward_name_list.index("VQAORM")
        # GCEP：把每条 rollout 的 vqa 二值正确性显式存到 input dict。
        # 说明：VQA_NORM=0 且 VQA_WEIGHT=1 时，vqa_orm reward 列即原始 {0,1} 二值
        # correctness（agent_rm.py VQAORM.evaluate：norm=0 时不做 (x-0.5)*2 映射）。
        # 这里统一做 >0.5 显式二值化，对该配置下的 {0,1} 列是恒等映射。
        if self.gcep_config.enabled:
            self._gcep_vqa_col = vqa_orm_index
            _vqa_col = rewards_per_func[:, vqa_orm_index]
            for _i in range(len(inputs)):
                inputs[_i]['vqa_norm_binary'] = int(float(_vqa_col[_i]) > 0.5)
        # vqa_orm_index = self.reward_funcs.index('vqa_orm')
        mask = total_rewards_per_func[:, vqa_orm_index] <= 0
        for depend_reward in ["CODEORM", "CODEPRM", "VQAPRM", "CSTORM", "CODEORMLLM"]:
            # depend_reward_func = orms[depend_reward]
            if depend_reward in reward_name_list:
                depend_reward_index = reward_name_list.index(depend_reward)
                total_rewards_per_func[mask, depend_reward_index] = 0

        cst_orm_index = reward_name_list.index("CSTORM")
        # vqa_orm_index = self.reward_funcs.index('vqa_orm')
        mask = total_rewards_per_func[:, vqa_orm_index] <= 0
        for depend_reward in ["CODEORM", "CODEPRM", "VQAPRM", "CODEORMLLM"]:
            # depend_reward_func = orms[depend_reward]
            if depend_reward in reward_name_list:
                depend_reward_index = reward_name_list.index(depend_reward)
                total_rewards_per_func[mask, depend_reward_index] = 0
        # 创建一个字典来存储所有变量
        data_to_save = {
            'total_rewards_per_func_before': total_rewards_per_func_before,
            'total_rewards_per_func_after': total_rewards_per_func,
            'mask': mask,
            'vqa_orm_index': vqa_orm_index,
            'depend_reward_indices': {depend_reward: self.reward_funcs.index(depend_reward) for depend_reward in ["code_orm", "code_prm", "vqa_prm", "cst_orm"] if depend_reward in self.reward_funcs}
        }
        if (total_rewards_per_func_before - total_rewards_per_func).sum() != 0:
            # Release: redirect hardcoded debug dump to a writable dir + guard.
            try:
                _addmask_dir = os.path.join(
                    os.getenv("THYME_INFER_ROOT",
                              "thyme-infer"),
                    "outputs/thyme_debug_cache/scripts/rl")
                os.makedirs(_addmask_dir, exist_ok=True)
                torch.save(data_to_save, os.path.join(_addmask_dir, 'add_mask.pth'))
            except Exception:
                pass

        total_rewards = (total_rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        return total_rewards_per_func, total_rewards, completions

    def _dynamic_sampling(self, inputs, rewards, rewards_per_func, completions):
        # DAPO https://arxiv.org/abs/2503.14476
        # Replaces samples with zero-reward-variance groups (std=0)
        resample_count = 0
        valid_samples = []
        valid_rewards = []
        valid_rewards_per_func = []
        valid_completions = []

        origin_data = (inputs, rewards, rewards_per_func, completions)

        while resample_count < self.args.max_resample_times:
            grouped_rewards = rewards.view(-1, self.num_generations)
            group_std = grouped_rewards.std(dim=1)

            valid_mask = (group_std > 0).repeat_interleave(self.num_generations)
            all_inputs = gather_object(inputs)
            valid_samples.extend([inp for inp, mask in zip(all_inputs, valid_mask) if mask])
            valid_rewards.append(rewards[valid_mask])
            valid_rewards_per_func.append(rewards_per_func[valid_mask])
            valid_completions.extend(
                [inp['messages'][-1]['content'] for inp, mask in zip(all_inputs, valid_mask) if mask])

            if len(valid_samples) >= self.effective_train_batch_size:
                break

            inputs = next(self.resample_iterator)
            inputs = Trainer._prepare_inputs(self, inputs)
            inputs = self._generate_completions(inputs)
            rewards_per_func, rewards, completions = self._score_completions(inputs)
            resample_count += 1

        if len(valid_samples) >= self.effective_train_batch_size:
            process_slice = slice(
                self.accelerator.process_index * len(inputs),
                (self.accelerator.process_index + 1) * len(inputs),
            )
            inputs = valid_samples[:self.effective_train_batch_size][process_slice]
            rewards = torch.cat(valid_rewards)[:self.effective_train_batch_size]
            rewards_per_func = torch.cat(valid_rewards_per_func)[:self.effective_train_batch_size]
            completions = valid_completions[:self.effective_train_batch_size][process_slice]
        else:
            logger.warning(f'There are still std=0 groups present after {self.args.max_resample_times} retries.')
            inputs, rewards, rewards_per_func, completions = origin_data

        return inputs, rewards, rewards_per_func, completions

    def _prepare_batch_inputs(self, inputs: InputsType, rewards: torch.Tensor) -> List[InputsType]:
        """
        Prepare the final batch inputs with advantages, ref/old_policy logps and other fields for RL training.

        Args:
            inputs (InputsType): List of input samples. Original shape is [gas*bs] where:
                - gas: gradient accumulation steps
                - bs: per-device batch size
            rewards (torch.Tensor): Tensor of rewards corresponding to the inputs.
                Shape should match the total number of samples (gas*bs*num_generations)

        Returns:
            List[InputsType]: A list of prepared batch inputs, organized as [gas][bs]
        """
        # Compute advantages
        grouped_rewards = rewards.view(-1, self.num_generations)
        mean_grouped_rewards = grouped_rewards.mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = grouped_rewards.std(dim=1).repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards)
        if self.args.scale_rewards:
            advantages /= (std_grouped_rewards + 1e-4)

        adv_mean = advantages.mean()
        adv_std = advantages.std()

        # 使用全局统计数据对 advantages 进行标准化
        # advantages = (advantages - adv_mean) / (adv_std + 1e-8)  # 1e-8 防止除以零

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(inputs),
            (self.accelerator.process_index + 1) * len(inputs),
        )
        advantages = advantages[process_slice]

        # RLSD：把本进程的原始 reward 附到每条 rollout 上，供 rich_*_correct_only
        # 模式做「答对才给观测」的过滤（rewards 是全局张量，inputs 是本地切片）。
        if self.rlsd_enable:
            local_rewards = rewards[process_slice]
            for _bi, _inp in enumerate(inputs):
                _inp['reward'] = float(local_rewards[_bi])
        mode = 'train' if self.model.training else 'eval'
        bs = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size
        gas = self.args.gradient_accumulation_steps if mode == 'train' else 1

        assert len(inputs) == bs * gas, f'Expected {bs * gas} inputs, got {len(inputs)}'
        gas_chunks = [inputs[i * bs:(i + 1) * bs] for i in range(gas)]

        ga_batch_encoded_inputs = []
        template = self.template

        # Split advantages by GAS chunks
        advantage_chunks = torch.chunk(advantages, gas)

        for i, (batch, batch_advantages) in enumerate(zip(gas_chunks, advantage_chunks)):
            # Encode and process each batch (size=bs)
            with self._template_context(template):
                batch_encoded_inputs = [template.encode(infer_request) for infer_request in batch]
                batch_encoded_inputs = to_device(template.data_collator(batch_encoded_inputs), self.model.device)

            # Process labels and masks
            labels = batch_encoded_inputs.pop('labels')
            # 零暴露：把生成侧拦截追加的后缀（"\n</answer>"）从
            # labels 抹掉 → 人工 token 零梯度、也不进任何 teacher 目标；input_ids 不变，
            # 故不影响 logits_to_keep 与 teacher/student 的 token 对齐（只动尾部）。
            _gi_suffix_rows = [(j, b.get('gi_suffix')) for j, b in enumerate(batch)
                               if b.get('gi_suffix')]
            if _gi_suffix_rows:
                if getattr(self, '_gi_mod', None) is None:
                    try:
                        from .gcep import gen_intercept as _gi_m
                    except ImportError:
                        from gcep import gen_intercept as _gi_m
                    self._gi_mod = _gi_m
                _gi_stats = getattr(self, '_gi_stats', None)
                _ids = batch_encoded_inputs['input_ids']
                for _j, _suf in _gi_suffix_rows:
                    _span = self._gi_mod.suffix_span(_ids[_j].tolist(), _suf, self.processing_class)
                    if _span is None:
                        if _gi_stats is not None:
                            _gi_stats['suffix_miss'] = _gi_stats.get('suffix_miss', 0) + 1
                        continue
                    labels[_j, _span[0]:] = -100
                    if _gi_stats is not None:
                        _gi_stats['suffix_masked'] = _gi_stats.get('suffix_masked', 0) + 1
            logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
            batch_encoded_inputs.update({
                'completion_mask':
                labels[:, -logits_to_keep:] != -100,
                'truncated_mask':
                torch.tensor([b['is_truncated'] for b in batch], dtype=torch.bool),
                'logits_to_keep':
                logits_to_keep,
                'advantages':
                batch_advantages
            })

            with torch.no_grad():
                batch_encoded_inputs['old_per_token_logps'] = (
                    self._get_per_token_logps(self.model, batch_encoded_inputs) if self.old_policy else None)

                if self.beta == 0.0:
                    ref_per_token_logps = None
                elif self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, batch_encoded_inputs)
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(self.model, batch_encoded_inputs)
                batch_encoded_inputs['ref_per_token_logps'] = ref_per_token_logps

            # ------------------------------------------------------------------
            # RLSD 教师自蒸馏 advantage 重加权（token 级、带 sign(A) 方向反转）
            # ------------------------------------------------------------------
            # 仅当 RLSD 开启且 λ(step)>0 时执行；否则保持原标量 advantages（纯 GRPO）。
            if self.rlsd_enable:
                lam = self._rlsd_lambda()
                if lam > 0.0:
                    batch_encoded_inputs['advantages'] = self._rlsd_teacher_reweight(
                        batch=batch,
                        batch_encoded_inputs=batch_encoded_inputs,
                        scalar_advantages=batch_advantages,
                        lam=lam,
                    )

            # ------------------------------------------------------------------
            # GCEP：冻结 teacher Top32 目标（no_grad 前向后立即 topk
            # reduce 释放全量 logits）。privilege 行 token 对齐失败自动回退
            # clean，绝不 crash。
            # ------------------------------------------------------------------
            if self.gcep_config.enabled:
                from .gcep import trainer_integration as _gcep
                from .gcep import traj_filter as _trajf
                _gcep_teacher = self._gcep_get_teacher()
                if getattr(self.gcep_config, 'is_mt_mode', False):
                    _gcep_plans = [b.get('mt_evidence_plan') for b in batch]
                else:
                    _gcep_plans = [b.get('gcep_privilege_plan') for b in batch]
                _traj_inv = ([bool(b.get('traj_invalid')) for b in batch]
                             if _trajf.traj_filter_enabled() else None)
                batch_encoded_inputs['gcep_teacher'] = _gcep.gcep_build_teacher_targets(
                    self, _gcep_teacher, batch, batch_encoded_inputs, _gcep_plans,
                    self.gcep_config, traj_invalid=_traj_inv)
                # A2v2：impact sidecar 归档上下文（step/chunk/rank/out_dir）
                batch_encoded_inputs['gcep_teacher']['dump_ctx'] = {
                    'step': int(self.state.global_step),
                    'chunk': int(i),
                    'rank': int(self.accelerator.process_index),
                    'out_dir': str(self.args.output_dir),
                }

            ga_batch_encoded_inputs.append(batch_encoded_inputs)

        # V-Zero-MT：全部 chunk 的 teacher 分数就绪后，跨 rank 全局 group z-score
        # 门控写回各 chunk（官方在全局 batch 上算权重的等价实现；仅 train）。
        if (self.gcep_config.enabled and getattr(self.gcep_config, 'is_mt_mode', False)
                and self.gcep_config.mt_kind == 'vzero_mt' and mode == 'train'):
            from .gcep import mt_teacher as _mt
            _mt.assign_vzero_gates(self, ga_batch_encoded_inputs, len(inputs),
                                   self.num_generations, self.gcep_config)

        return ga_batch_encoded_inputs

    # ----------------------------------------------------------------------
    # RLSD 辅助方法
    # ----------------------------------------------------------------------
    def _rlsd_lambda(self) -> float:
        """按 global_step 计算当前 λ（前 decay_steps 步从 init 线性衰减到 0）。"""
        step = int(self.state.global_step)
        if step >= self.rlsd_lambda_decay_steps:
            return 0.0
        frac = 1.0 - step / max(1, self.rlsd_lambda_decay_steps)
        return self.rlsd_lambda_init * frac

    def _rlsd_teacher_reweight(self, batch, batch_encoded_inputs, scalar_advantages, lam):
        """对一个 micro-batch 的标量 advantage 做 RLSD token 级重加权。

        Args:
            batch: 该 chunk 的原始 rollout 列表（含 messages / solution）。
            batch_encoded_inputs: 已 encode 的张量字典（含 input_ids/completion_mask/
                logits_to_keep），completion 段用于对齐 teacher/student。
            scalar_advantages: [bs] 原 GRPO 组内标准化标量 advantage。
            lam: 当前 λ。

        Returns:
            per-token advantages 张量 [bs, logits_to_keep]。
        """
        import time as _time
        from .rlsd_privilege import build_privilege_messages

        t0 = _time.time()
        device = scalar_advantages.device
        logits_to_keep = batch_encoded_inputs['logits_to_keep']
        completion_mask = batch_encoded_inputs['completion_mask']  # [bs, T]
        student_ids = batch_encoded_inputs['input_ids'][:, -logits_to_keep:]  # [bs, T] completion 段
        bs = student_ids.shape[0]

        # 1) student 逐 token logp（专门 no_grad 前向，与 teacher 成对、严格可比）
        with torch.no_grad():
            logp_s = self._get_per_token_logps(self.model, batch_encoded_inputs)  # [bs, T]

        # 2) 构造 privilege-augmented batch，做 teacher no_grad 前向
        priv_batch = []
        for b in batch:
            msgs = b['messages']
            solution = b.get('solution', '')
            reward = b.get('reward', None)
            # solution 可能是 list（batched 列），取标量
            if isinstance(solution, (list, tuple)) and solution:
                solution = solution[0]
            priv_b = dict(b)  # 浅拷贝，只替换 messages
            priv_b['messages'] = build_privilege_messages(
                msgs, str(solution), self.rlsd_privilege_mode, reward=reward)
            priv_batch.append(priv_b)

        with self._template_context(self.template):
            teacher_encoded = [self.template.encode(r) for r in priv_batch]
            teacher_encoded = to_device(self.template.data_collator(teacher_encoded), self.model.device)
        teacher_labels = teacher_encoded.pop('labels')
        teacher_ltk = (teacher_labels.shape[-1] - (torch.ne(teacher_labels, -100).int().argmax(-1))).max().item()
        teacher_encoded['logits_to_keep'] = teacher_ltk
        with torch.no_grad():
            logp_t_full = self._get_per_token_logps(self.model, teacher_encoded)  # [bs, T_t]
        teacher_ids = teacher_encoded['input_ids'][:, -teacher_ltk:]

        # 3) 逐样本对齐 completion token；不齐则该样本 w_t=1 软跳过
        per_token_adv = scalar_advantages.unsqueeze(1).expand(bs, logits_to_keep).clone()  # 默认 = 标量广播
        eps_w = self.rlsd_eps_w
        w_accum, w_count, clip_count = 0.0, 0, 0
        for j in range(bs):
            self._rlsd_total_count += 1
            # teacher completion 段末尾 logits_to_keep 个 token 应与 student 完全一致
            t_ids_j = teacher_ids[j, -logits_to_keep:]
            s_ids_j = student_ids[j]
            if t_ids_j.shape[0] != s_ids_j.shape[0] or not torch.equal(t_ids_j, s_ids_j):
                self._rlsd_mismatch_count += 1
                continue  # 软跳过：保持标量广播 advantage（w_t=1 等价）
            logp_t_j = logp_t_full[j, -logits_to_keep:]  # [T]
            logp_s_j = logp_s[j]                          # [T]
            A_j = scalar_advantages[j]
            sign_a = torch.sign(A_j) if A_j != 0 else torch.tensor(1.0, device=device)
            delta = (logp_t_j - logp_s_j).detach()        # Δ_t
            w = torch.exp(sign_a * delta)                  # (P_T/P_S)^{sign(A)}
            w = torch.clamp(w, 1.0 - eps_w, 1.0 + eps_w)   # trust-region 裁剪
            # 可选：只在 answer 段生效（其余位置 w=1）
            if self.rlsd_score_span == 'answer':
                ans_mask = self._rlsd_answer_mask(s_ids_j)  # [T] bool
                w = torch.where(ans_mask, w, torch.ones_like(w))
            per_token_adv[j] = A_j * ((1.0 - lam) + lam * w)
            # 统计（只在有效 token 上）
            cm = completion_mask[j].bool()
            if cm.any():
                w_accum += float(w[cm].mean()); w_count += 1
                clip_count += int(((w <= 1.0 - eps_w + 1e-6) | (w >= 1.0 + eps_w - 1e-6))[cm].float().mean() > 0)

        # 4) 记录 wandb 指标
        mode = 'train' if self.model.training else 'eval'
        self._metrics[mode]['rlsd/lambda'].append(lam)
        if w_count > 0:
            self._metrics[mode]['rlsd/w_mean'].append(w_accum / w_count)
            self._metrics[mode]['rlsd/w_clip_ratio'].append(clip_count / w_count)
        if self._rlsd_total_count > 0:
            self._metrics[mode]['rlsd/mismatch_ratio'].append(
                self._rlsd_mismatch_count / self._rlsd_total_count)
        self._metrics[mode]['rlsd/teacher_fwd_time'].append(_time.time() - t0)

        return per_token_adv

    def _rlsd_answer_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """返回一个 [T] bool mask，标出 <answer>...</answer> 段内的 token。

        用 token 反解文本后按字符区间近似定位（answer 段通常较短，开销可忽略）。
        解析失败时返回全 True（退化为全轨迹打分）。
        """
        try:
            tok = self.template.tokenizer
            text = tok.decode(ids, skip_special_tokens=False)
            start = text.find('<answer>')
            end = text.find('</answer>')
            if start == -1 or end == -1 or end <= start:
                return torch.ones_like(ids, dtype=torch.bool)
            # 逐 token 累积解码长度，落在 [start, end) 字符区间的 token 记为 answer
            mask = torch.zeros_like(ids, dtype=torch.bool)
            cum = 0
            for k in range(ids.shape[0]):
                piece = tok.decode(ids[k:k + 1], skip_special_tokens=False)
                nxt = cum + len(piece)
                if nxt > start and cum < end + len('</answer>'):
                    mask[k] = True
                cum = nxt
            return mask if mask.any() else torch.ones_like(ids, dtype=torch.bool)
        except Exception:
            return torch.ones_like(ids, dtype=torch.bool)

    # ----------------------------------------------------------------------
    # GCEP 辅助方法（薄 hook；重逻辑全在 gcep/trainer_integration.py）
    # ----------------------------------------------------------------------
    def _gcep_get_teacher(self):
        """延迟加载冻结 teacher（首次使用时）；vanilla 模式绝不会被调用。"""
        if self._gcep_teacher is None:
            from .gcep.trainer_integration import FrozenTeacherLoader
            if self._gcep_teacher_loader is None:
                self._gcep_teacher_loader = FrozenTeacherLoader(
                    ckpt=self.gcep_config.resolve_teacher_ckpt(),
                    device_map=self.gcep_config.teacher_device_map,
                    attn=self.gcep_config.teacher_attn,
                    accelerator=self.accelerator)
            logger.info(f'[GCEP] loading frozen teacher from {self._gcep_teacher_loader.ckpt}')
            self._gcep_teacher = self._gcep_teacher_loader.load()
        return self._gcep_teacher

    def _gcep_attach_privilege_plans(self, inputs, total_rewards, total_rewards_per_func):
        """mixed group → Judge → validate → anchor → certificate（main rank 计算 +
        broadcast），把 per-rollout privilege plan 附到本地 inputs；任何失败整批
        回退 clean（plan=None），绝不 crash。

        GCEP v2：compute_group_privilege_plans 返回 (plans, stats, group_meta)；
        group_type 透传到 inp（归档用）；main rank 把 group 级 judge_records 写
        rollout_archive/judge_outputs/step_XXXXXXX.jsonl sidecar。
        """
        from .gcep import trainer_integration as _gcep
        try:
            all_inputs = gather_object(inputs)
            # GCEP_TRAJ_FILTER 遥测修正：group 分类（MIXED/ALL_CORRECT/
            # ALL_WRONG）必须用**真实正确性**，不能用可能已被 filter 清零的 reward 表。
            # vqa_norm_binary 在 _score_completions 里于 filter 之前由原始 VQA 列写入
            # （见上方 1590-1594），filter 关闭时与 reward 列逐元素恒等（零影响）；
            # 开启时若仍从 reward 表取，污染组（reward 被整组清零）会被误分类成
            # ALL_WRONG，污染 gcep/*_group_ratio 遥测。
            if all('vqa_norm_binary' in inp for inp in all_inputs):
                # 分组规则：组分类时 invalid 行一律当作
                # 错误样本（vqa 记 0）；其余行用真实正确性（vqa_norm_binary 在 filter
                # 之前写入，不受清零影响）。traj_invalid 由上方 traj_filter 段写入本地
                # inputs 后经 gather_object 到 all_inputs。
                vqa_global = [
                    0 if inp.get('traj_invalid') else int(inp.get('vqa_norm_binary', 0))
                    for inp in all_inputs
                ]
            else:
                vqa_col = self._gcep_vqa_col
                if vqa_col is None:
                    vqa_col = self.reward_func_names.index('VQAORM')
                vqa_global = (total_rewards_per_func[:, vqa_col] > 0.5).int().tolist()
            plans, stats, group_meta = _gcep.compute_group_privilege_plans(
                all_inputs, vqa_global, self.num_generations, self.gcep_config,
                is_main=self.accelerator.is_main_process, logger=logger)
            mode = 'train' if self.model.training else 'eval'
            for _k, _v in stats.items():
                self._metrics[mode][f'gcep/{_k}'].append(_v)
            offset = self.accelerator.process_index * len(inputs)
            rollout_gts = (group_meta or {}).get('rollout_group_types') or []
            for i, inp in enumerate(inputs):
                g = offset + i
                inp['gcep_privilege_plan'] = plans[g] if g < len(plans) else None
                if g < len(rollout_gts):
                    inp['gcep_group_type'] = rollout_gts[g]
                inp.setdefault(
                    'vqa_norm_binary', int(vqa_global[g]) if g < len(vqa_global) else 0)
            self._gcep_dump_judge_records(group_meta)
        except Exception as exc:
            logger.warning(f'[GCEP] attach privilege plans failed, fallback all clean: {exc}')
            for inp in inputs:
                inp['gcep_privilege_plan'] = None
        return inputs

    def _mt_attach_evidence_plans(self, inputs):
        """MT baseline：离线证据库逐行查找，挂 inp['mt_evidence_plan']。

        仅 is_mt_mode 生效（其他模式立即返回，零副作用）。查找失败的行 plan=None
        （该行回退 clean teacher， 同款）。bank 未配置时硬报错（MT 模式必需）。
        """
        if not getattr(self.gcep_config, 'is_mt_mode', False):
            return inputs
        try:
            from .gcep import mt_evidence as _mte
            bank = getattr(self, '_mt_bank', None)
            if bank is None:
                bank_dir = self.gcep_config.mt_evidence_bank
                if not bank_dir:
                    raise RuntimeError('[MT] GCEP_MT_EVIDENCE_BANK 未配置（MT 模式必需）')
                bank = _mte.MtEvidenceBank(bank_dir)
                self._mt_bank = bank
                logger.info(f'[MT] evidence bank loaded: {bank_dir} '
                            f'({sum(len(v) for v in bank.by_key.values())} entries)')
            plans = _mte.build_mt_plans(inputs, bank)
            for inp, plan in zip(inputs, plans):
                inp['mt_evidence_plan'] = plan
            mode = 'train' if self.model.training else 'eval'
            st = bank.stats
            for _k in ('hit', 'miss_key', 'multi_image', 'no_image', 'disambiguated'):
                self._metrics[mode][f'mt/bank_{_k}'].append(float(st.get(_k, 0)))
            hit_rate = st['hit'] / max(1, st['lookup'])
            self._metrics[mode]['mt/bank_hit_rate'].append(hit_rate)
        except Exception as exc:
            logger.warning(f'[MT] attach evidence plans failed, fallback all clean: {exc}')
            for inp in inputs:
                inp['mt_evidence_plan'] = None
        return inputs

    def _gcep_dump_judge_records(self, group_meta):
        """把 group 级 judge output 写 sidecar（main rank；失败只告警）。"""
        try:
            records = (group_meta or {}).get('judge_records') or []
            if not records or not self.accelerator.is_main_process:
                return
            import json as _json
            dest_dir = os.path.join(
                str(self.args.output_dir), 'rollout_archive', 'judge_outputs')
            os.makedirs(dest_dir, exist_ok=True)
            step = int(self.state.global_step)
            path = os.path.join(dest_dir, f'step_{step:07d}.jsonl')
            with open(path, 'a', encoding='utf-8') as f:
                for rec in records:
                    rec = dict(rec)
                    rec['global_step'] = step
                    f.write(_json.dumps(rec, ensure_ascii=False) + '\n')
        except Exception as exc:
            logger.warning(f'[GCEP] judge records dump failed (training unaffected): {exc}')

    def _gcep_archive_rollouts(self, inputs, total_rewards, total_rewards_per_func):
        """全量 rollout 归档（强制）；异常只在集成层告警，不影响训练。"""
        from .gcep import trainer_integration as _gcep
        _gcep.archive_rollouts(
            self, inputs, total_rewards, total_rewards_per_func, self.gcep_config,
            logger=logger)

    def _log_metrics(self, inputs, messages, completions, rewards, rewards_per_func):
        """Log training/evaluation metrics"""
        mode = 'train' if self.model.training else 'eval'
        device = self.accelerator.device

        # Calculate completion length metrics
        agg_completion_mask = gather(torch.cat([inp['completion_mask'].sum(1) for inp in inputs]))

        self._metrics[mode]['completions/mean_length'].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]['completions/min_length'].append(agg_completion_mask.float().min().item())
        self._metrics[mode]['completions/max_length'].append(agg_completion_mask.float().max().item())
        # Calculate clip ratio
        agg_truncated_mask = gather(torch.cat([inp['truncated_mask'] for inp in inputs]).to(device))

        term_completion_mask = agg_completion_mask[agg_truncated_mask]
        clipped_completions_ratio = len(term_completion_mask) / len(agg_completion_mask)

        self._metrics[mode]['completions/clipped_ratio'].append(clipped_completions_ratio)

        # 生成侧拦截遥测（GEN_INTERCEPT=1 时非零；默认全 0 不影响既有看板）。
        # 计数在 _generate_and_score_completions 里每步重置，这里聚合当步事件数。
        _gi_stats = getattr(self, "_gi_stats", None)
        if _gi_stats is not None:
            for _k, _v in _gi_stats.items():
                self._metrics[mode][f'gen_intercept/{_k}'].append(float(_v))

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = rewards_per_func[:, i].mean().item()
            self._metrics[mode][f'rewards/{reward_func_name}/mean'].append(mean_rewards)
            std_rewards = rewards_per_func[:, i].std().item()
            self._metrics[mode][f'rewards/{reward_func_name}/std'].append(std_rewards)

        # Log overall reward stats
        grouped_rewards = rewards.view(-1, self.num_generations)
        self._metrics[mode]['reward'].append(grouped_rewards.mean().item())
        self._metrics[mode]['reward_std'].append(grouped_rewards.std(dim=1).mean().item())

        # Log prompt and completion texts
        self._textual_logs['prompt'].extend(self._apply_chat_template_to_messages_list(gather_object(messages)))
        self._textual_logs['completion'].extend(gather_object(completions))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs['rewards'][name].extend(rewards_per_func[:, i].tolist())

    def _apply_chat_template_to_messages_list(self, messages_list: InputsType):
        prompts_text = []
        for messages in messages_list:
            InferRequest.remove_response(messages)
            template_inputs = StdTemplateInputs.from_dict({'messages': messages})
            res_context_list, _, _ = self.template._swift_encode(template_inputs)
            prompts_text.append(''.join(res_context_list))
        return prompts_text

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Compute the per-token log probabilities for the model, return_outputs=True in mini-batch training
        if isinstance(inputs, list):
            assert len(inputs) == 1
            inputs = inputs[0]
        completion_mask = inputs['completion_mask']
        truncated_mask = inputs['truncated_mask']
        # apply the completion_mask to exclude loss and metrics for overlong completions
        if self.args.overlong_filter and any(truncated_mask):
            if all(truncated_mask):
                logger.info('All completions are overlong, loss and KL will be zero')
            truncated_mask = truncated_mask.unsqueeze(-1).expand_as(completion_mask).to(completion_mask.device)
            completion_mask = completion_mask * (~truncated_mask)

        # GCEP：启用时用一次 student 前向同时供 GRPO logp 与 OPD 全量 logits
        # （避免两次 student 前向）；未启用时走原路径（逐字节等价）。
        gcep_on = self.gcep_config.enabled
        student_logits = None
        if gcep_on:
            from .gcep import trainer_integration as _gcep
            student_logits = _gcep.student_completion_logits(self, model, inputs)
            if student_logits is not None:
                per_token_logps = _gcep.logps_from_student_logits(self, student_logits, inputs)
            else:
                # runaway 样本软跳过（与 _get_per_token_logps 的 guard 同语义）
                per_token_logps = self._get_per_token_logps(model, inputs)
        else:
            per_token_logps = self._get_per_token_logps(model, inputs)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs['ref_per_token_logps']
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)

        advantages = inputs['advantages']
        old_per_token_logps = inputs['old_per_token_logps'] if self.old_policy else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        # advantages 可能是 [B]（原生 GRPO 标量）或 [B, T]（RLSD token 级重加权）。
        # 前者需 unsqueeze 广播到所有 token；后者已是 per-token，直接用。
        adv_bcast = advantages.unsqueeze(1) if advantages.dim() == 1 else advantages
        per_token_loss1 = coef_1 * adv_bcast
        per_token_loss2 = coef_2 * adv_bcast
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == 'grpo':
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == 'bnpo':
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == 'dr_grpo':
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        # Log the metrics
        mode = 'train' if self.model.training else 'eval'

        # GCEP：OPD loss 组合（两个 loss 各自独立归一化后再相加；
        # A/B 系纯 OPD：loss = L_OPD，GRPO 项不进入 loss）。
        if gcep_on:
            opd_loss, gcep_metrics = _gcep.gcep_opd_loss(
                student_logits, inputs.get('gcep_teacher'), completion_mask, self.gcep_config,
                per_token_logps=per_token_logps, old_per_token_logps=old_per_token_logps)
            grpo_loss = loss if self.gcep_config.use_grpo_loss else None
            loss = _gcep.combine_losses(grpo_loss, opd_loss, self.gcep_config)
            for _k, _v in gcep_metrics.items():
                self._metrics[mode][f'gcep/{_k}'].append(_v)
            self._metrics[mode]['gcep/total_loss'].append(float(loss.detach()))
            if grpo_loss is not None:
                self._metrics[mode]['gcep/grpo_loss'].append(float(grpo_loss.detach()))

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]['kl'].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (adv_bcast < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (adv_bcast > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]['clip_ratio/low_mean'].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]['clip_ratio/low_min'].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]['clip_ratio/high_mean'].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]['clip_ratio/high_max'].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]['clip_ratio/region_mean'].append(gathered_clip_ratio.nanmean().item())

        return loss

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, inputs):
        from trl.trainer.utils import selective_log_softmax
        logits_to_keep = inputs['logits_to_keep']
        input_ids = inputs['input_ids']
        # Release patch: guard against runaway multi-turn O3 rollouts. Some samples
        # accumulate huge numbers of image tokens across turns (observed 165k tokens),
        # which would materialize a [B, seq_len, vocab~152k] logits tensor and OOM the
        # GPU (~48 GiB single alloc). Normal samples are ~10k tokens, so a 100k
        # threshold never touches them. Skip the forward and return zero logps: the
        # sample then contributes zero gradient (soft-skip), and training continues.
        # NOTE: assumes per_device_train_batch_size == 1 (each chunk is one sample).
        safe_max_len = int(os.getenv('LOGP_MAX_SEQ_LEN', '100000'))
        if input_ids.shape[1] > safe_max_len:
            logger.warning(
                f'[training] Skipping logps forward: seq_len={input_ids.shape[1]} > '
                f'{safe_max_len} (runaway multi-turn sample). Returning zero logps '
                f'to soft-skip this sample and avoid OOM.')
            return torch.zeros((input_ids.shape[0], logits_to_keep),
                               device=input_ids.device,
                               dtype=torch.float32)
        unwrapped_model = self.accelerator.unwrap_model(model)
        if is_peft_model(unwrapped_model):
            parameters = inspect.signature(unwrapped_model.base_model.model.forward).parameters
        else:
            parameters = inspect.signature(unwrapped_model.forward).parameters
        if not unwrapped_model.model_meta.is_multimodal and 'logits_to_keep' in parameters:
            # save memory
            return super()._get_per_token_logps(model, input_ids, inputs['attention_mask'], logits_to_keep)
        inputs = {
            k: v
            for k, v in inputs.items() if k not in [
                'logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps',
                'truncated_mask', 'gcep_teacher'
            ]
        }
        with self._template_context(self.template):
            logits = model(**inputs).logits
        # exclude the last logit: it corresponds to the next token pred
        logits = logits[:, -(logits_to_keep + 1):-1, :]
        logits = logits / self.temperature
        input_ids = input_ids[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    def evaluation_loop(self, dataloader, *args, **kwargs):
        # Wait for the training rollout to complete
        if self.args.async_generate:
            while not self.is_async_generate_train_rollout_done():
                time.sleep(0.1)
        if self._queue.empty() and self.args.async_generate:
            self._prefetch(dataloader)
        metric_key_prefix = kwargs['metric_key_prefix']
        output = super().evaluation_loop(dataloader, *args, **kwargs)
        metrics = {f'{metric_key_prefix}_{key}': sum(val) / len(val) for key, val in self._metrics['eval'].items()}
        output.metrics.update(metrics)
        self.eval_flag = True
        return output

    def training_step(self, model: nn.Module, inputs: InputsType, num_items_in_batch=None) -> torch.Tensor:
        if self.args.async_generate:
            # Wait for the eval rollout to complete
            while not self.is_async_generate_eval_rollout_done():
                time.sleep(0.1)
        return super().training_step(model, inputs, num_items_in_batch)

    def _engine_infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        *,
        use_tqdm: Optional[bool] = False,
    ):
        if self.vllm_mode == 'server':
            self._process_infer_requests_images(infer_requests)
            return self.vllm_client.infer(infer_requests, asdict(request_config), use_tqdm=use_tqdm)
        else:
            # return self.engine.infer(infer_requests, request_config, use_tqdm=use_tqdm)
            return self.engine.infer(infer_requests, request_config, use_tqdm=use_tqdm, process_index=self.accelerator.process_index)

    def _process_infer_requests_images(self, infer_requests: List[InferRequest]):
        # Process image format into a format that session.post can accept
        import base64
        if not any('images' in request for request in infer_requests):
            return
        for request in infer_requests:
            if 'images' not in request:
                continue
            for i, img in enumerate(request['images']):
                if 'bytes' in img and img['bytes']:
                    request['images'][i] = base64.b64encode(img['bytes']).decode('utf-8')
                elif 'path' in img and img['path']:
                    request['images'][i] = img['path']
        return

    @property
    def old_policy(self):
        return self.num_iterations > 1

    @property
    def _queue(self):
        if self.control.should_evaluate:
            return self.eval_queue
        else:
            return self.train_queue

    @torch.no_grad()
    def offload_model(self):
        if len(self.offload_modules) > 0:
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        for name, module in unwrapped_model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                self.offload_modules[name] = module.weight.device
                module.to('cpu')
            elif not hasattr(module, 'device'):
                pass
            elif module.device.type != 'cpu':
                self.offload_modules[name] = module.device
                module.to('cpu')

    @torch.no_grad()
    def load_model(self):
        if len(self.offload_modules) == 0:
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        for name, device in self.offload_modules.items():
            module = unwrapped_model.get_submodule(name)
            if isinstance(module, torch.nn.Embedding):
                module.weight.to(device)
            else:
                module.to(device)
        self.offload_modules.clear()

    @torch.no_grad()
    def offload_optimizer(self):
        if len(self.offload_states) > 0:
            return
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        self.offload_states[key] = value.device
                        state[key] = value.to('cpu', non_blocking=True)

    @torch.no_grad()
    def load_optimizer(self):
        if len(self.offload_states) == 0:
            return
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.offload_states[key], non_blocking=True)
        self.offload_states.clear()

    @contextmanager
    def multi_turn_completion_length_context(self):
        """
        Context manager that temporarily adjusts the engine's max length handling
        for multi-turn generation scenarios.

        Ensures the total sequence length (prompt + completion) never exceeds:
            min(original_max_len, prompt_tokens + max_completion_length)
        """
        if not (self.multi_turn_func and
                self.use_fast_infer) or self.vllm_mode == 'server' or self.completion_length_limit_scope == 'per_round':
            yield
            return

        original_fn = self.engine.set_default_max_tokens
        original_max_len = self.engine.max_model_len

        def set_default_max_tokens(_self, request_config: RequestConfig, inputs: InputsType) -> None:
            # Calculate required context window
            original_max_len = _self.max_model_len or 8192
            if isinstance(inputs, dict):
                inputs = [inputs]
            prompt_tokens = max(_self._get_num_tokens(inp) for inp in inputs)

            if not hasattr(_self, 'set_grpo_max_model_len'):
                # set max model len in first round
                max_len = min(original_max_len, prompt_tokens + request_config.max_tokens)
                _self.max_model_len = max_len
                _self.set_grpo_max_model_len = True
            else:
                if _self.max_model_len <= prompt_tokens:
                    # modify max_model_len > prompt_tokens to avoid crash
                    num_tokens_avoid_crash = 10
                    _self.max_model_len = (prompt_tokens + num_tokens_avoid_crash)
                    request_config.max_tokens = num_tokens_avoid_crash

            original_fn(request_config, inputs)

        try:
            self.engine.set_default_max_tokens = MethodType(set_default_max_tokens, self.engine)
            yield
        finally:
            self.engine.set_default_max_tokens = original_fn
            self.engine.max_model_len = original_max_len
            del self.engine.set_grpo_max_model_len

    def get_resample_dataloader(self) -> DataLoader:
        resample_dataset = self.resample_dataset
        data_collator = self.data_collator
        if isinstance(resample_dataset, datasets.Dataset):
            resample_dataset = self._remove_unused_columns(resample_dataset, description='training')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='training')

        dataloader_params = {
            'batch_size': self._train_batch_size * self.args.gradient_accumulation_steps,
            'collate_fn': data_collator,
            'num_workers': self.args.dataloader_num_workers,
            'pin_memory': self.args.dataloader_pin_memory,
            'persistent_workers': self.args.dataloader_persistent_workers,
        }

        @contextmanager
        def seed_context(self):
            seed = self.args.seed
            self.args.seed = seed + 1
            yield
            self.args.seed = seed

        if not isinstance(resample_dataset, torch.utils.data.IterableDataset):
            with seed_context(self):  # Set a different seed for resampling than the train_dataset.
                dataloader_params['sampler'] = self._get_train_sampler()
            dataloader_params['drop_last'] = self.args.dataloader_drop_last
            dataloader_params['worker_init_fn'] = seed_worker
            dataloader_params['prefetch_factor'] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(resample_dataset, **dataloader_params))

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = 'train' if self.model.training else 'eval'
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == 'eval':
            metrics = {f'eval_{key}': val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse('4.47.0.dev0'):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            table = {
                'step': [str(self.state.global_step)] * len(self._textual_logs['prompt']),
                'prompt': self._textual_logs['prompt'],
                'completion': self._textual_logs['completion'],
                **self._textual_logs['rewards'],
            }
            self.jsonl_writer.append(table)
            if self.args.report_to and 'wandb' in self.args.report_to and wandb.run is not None:
                import pandas as pd
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=['prompt'])
                wandb.log({'completions': wandb.Table(dataframe=df)})

    def is_async_generate_eval_rollout_done(self):
        return not self.eval_flag or not self.eval_queue.empty()

    def is_async_generate_train_rollout_done(self):
        return not self.train_queue.empty()
