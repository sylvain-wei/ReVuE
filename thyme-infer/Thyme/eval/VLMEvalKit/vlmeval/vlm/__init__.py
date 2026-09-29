"""Patched vlmeval/vlm/__init__.py.

Thyme's vendored VLMEvalKit fork is missing many vlm/ submodules that the
original VLMEvalKit __init__.py imports (qwen_vl, qwen2_vl, internvl, etc.).
For Phase 2 we only need the Thyme model class.

Strategy:
1. Mandatorily import Thyme.
2. Try to import every other vlm class; missing modules silently skipped.
3. After the wave of try-imports, scan the names config.py expects to find
   in this namespace and bind any still-missing one to a stub that raises
   only when instantiated. This makes `partial(MissingClass, ...)` at
   module-load succeed, while preserving a clear error if someone tries
   to actually use a missing model.
"""
import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)

from .base import BaseModel  # required


def _safe_import(stmt):
    """Run an import statement; on ModuleNotFoundError just skip silently."""
    try:
        exec(stmt, globals())
    except (ModuleNotFoundError, ImportError):
        pass


# === Mandatory: Thyme ===
from .thyme import Thyme, ThymeInternVL
# Metis（Qwen3-VL）：模块级不 import transformers>=4.57 符号，惰性安全
_safe_import("from .thyme import ThymeQwen3VL")
# Qwen3-VL MoE（例如 Qwen3-VL-30B-A3B-Thinking）
_safe_import("from .thyme import ThymeQwen3VLMoe")

# === Best-effort: every upstream model that ships with this fork ===
_safe_import("from .aria import Aria")
_safe_import("from .hawk_vl import HawkVL")
_safe_import("from .cogvlm import CogVlm, GLM4v")
_safe_import("from .emu import Emu, Emu3_chat, Emu3_gen")
_safe_import("from .eagle_x import Eagle")
_safe_import("from .granite_vision import GraniteVision3")
_safe_import("from .idefics import IDEFICS, IDEFICS2")
_safe_import("from .instructblip import InstructBLIP")
_safe_import("from .kosmos import Kosmos2")
_safe_import("from .llava import (LLaVA, LLaVA_Next, LLaVA_XTuner, LLaVA_Next2, LLaVA_OneVision, LLaVA_OneVision_HF)")
_safe_import("from .vita import VITA, VITAQwen2")
_safe_import("from .long_vita import LongVITA")
_safe_import("from .minicpm_v import MiniCPM_V, MiniCPM_Llama3_V, MiniCPM_V_2_6, MiniCPM_o_2_6")
_safe_import("from .minigpt4 import MiniGPT4")
_safe_import("from .mmalaya import MMAlaya, MMAlaya2")
_safe_import("from .monkey import Monkey, MonkeyChat")
_safe_import("from .moondream import Moondream1, Moondream2")
_safe_import("from .minimonkey import MiniMonkey")
_safe_import("from .mplug_owl2 import mPLUG_Owl2")
_safe_import("from .omnilmm import OmniLMM12B")
_safe_import("from .open_flamingo import OpenFlamingo")
_safe_import("from .pandagpt import PandaGPT")
_safe_import("from .qwen_vl import QwenVL, QwenVLChat")
_safe_import("from .qwen2_vl import Qwen2VLChat, Qwen2VLChatAguvis")
_safe_import("from .transcore_m import TransCoreM")
_safe_import("from .visualglm import VisualGLM")
_safe_import("from .xcomposer import (ShareCaptioner, XComposer, XComposer2, XComposer2_4KHD, XComposer2d5)")
_safe_import("from .yi_vl import Yi_VL")
_safe_import("from .internvl import InternVLChat")
_safe_import("from .deepseek_vl import DeepSeekVL")
_safe_import("from .deepseek_vl2 import DeepSeekVL2")
_safe_import("from .janus import Janus")
_safe_import("from .mgm import Mini_Gemini")
_safe_import("from .bunnyllama3 import BunnyLLama3")
_safe_import("from .vxverse import VXVERSE")
_safe_import("from .gemma import PaliGemma, Gemma3")
_safe_import("from .qh_360vl import QH_360VL")
_safe_import("from .phi3_vision import Phi3Vision, Phi3_5Vision")
_safe_import("from .phi4_multimodal import Phi4Multimodal")
_safe_import("from .wemm import WeMM")
_safe_import("from .cambrian import Cambrian")
_safe_import("from .chameleon import Chameleon")
_safe_import("from .video_llm import (VideoLLaVA, VideoLLaVA_HF, Chatunivi, VideoChatGPT, LLaMAVID, VideoChat2_HD, PLLaVA)")
_safe_import("from .vila import VILA, NVILA")
_safe_import("from .ovis import Ovis, Ovis1_6, Ovis1_6_Plus, Ovis2, OvisU1")
_safe_import("from .mantis import Mantis")
_safe_import("from .mixsense import LLama3Mixsense")
_safe_import("from .parrot import Parrot")
_safe_import("from .omchat import OmChat")
_safe_import("from .rbdash import RBDash")
_safe_import("from .xgen_mm import XGenMM")
_safe_import("from .slime import SliME")
_safe_import("from .mplug_owl3 import mPLUG_Owl3")
_safe_import("from .pixtral import Pixtral")
_safe_import("from .llama_vision import llama_vision")
_safe_import("from .llama4 import llama4")
_safe_import("from .molmo import molmo")
_safe_import("from .points import POINTS, POINTSV15")
_safe_import("from .nvlm import NVLM")
_safe_import("from .vintern_chat import VinternChat")
_safe_import("from .h2ovl_mississippi import H2OVLChat")
_safe_import("from .falcon_vlm import Falcon2VLM")
_safe_import("from .smolvlm import SmolVLM, SmolVLM2")
_safe_import("from .sail_vl import SailVL")
_safe_import("from .valley import Valley2Chat")
_safe_import("from .ross import Ross")
_safe_import("from .ola import Ola")
_safe_import("from .x_vl import X_VL_HF")
_safe_import("from .ursa import UrsaChat")
_safe_import("from .vlm_r1 import VLMR1Chat")
_safe_import("from .aki import AKI")
_safe_import("from .ristretto import Ristretto")
_safe_import("from .vlaa_thinker import VLAAThinkerChat")
_safe_import("from .kimi_vl import KimiVL")
_safe_import("from .wethink_vl import WeThinkVL")
_safe_import("from .flash_vl import FlashVL")
_safe_import("from .oryx import Oryx")
_safe_import("from .treevgr import TreeVGR")
_safe_import("from .glm4_1v import GLM4_1v")
_safe_import("from .varco_vision import VarcoVision")
_safe_import("from .qtunevl import (QTuneVL, QTuneVLChat)")


# === Stub fill-in: any class config.py references but we couldn't import.
# config.py builds a registry like `partial(SomeModel, ...)` at module load,
# so undefined names raise NameError. Bind missing names to a stub class.
def _make_missing_stub(name):
    def _factory(*args, **kwargs):
        raise ImportError(
            f"Model class '{name}' is not bundled in Thyme's vendored "
            f"VLMEvalKit fork. Vendor the missing file from upstream "
            f"open-compass/VLMEvalKit if you need it."
        )
    _factory.__name__ = name
    return _factory


_EXPECTED_CLASSES = [
    # Names config.py references via `partial(<Name>, ...)`. If a class above
    # imported successfully it stays; otherwise we install a missing-stub.
    "AKI", "Aria", "BunnyLLama3", "Cambrian", "Chameleon", "Chatunivi",
    "DeepSeekVL", "DeepSeekVL2", "Eagle", "Emu", "Emu3_chat", "Emu3_gen",
    "Falcon2VLM", "FlashVL", "Gemma3", "GLM4_1v", "GLM4v", "GraniteVision3",
    "H2OVLChat", "IDEFICS", "IDEFICS2", "InstructBLIP", "InternVLChat",
    "Janus", "KimiVL", "Kosmos2", "LLaMAVID", "LLaVA", "LLaVA_Next",
    "LLaVA_Next2", "LLaVA_OneVision", "LLaVA_OneVision_HF", "LLaVA_XTuner",
    "LLama3Mixsense", "LongVITA", "Mantis", "MiniCPM_Llama3_V",
    "MiniCPM_V", "MiniCPM_V_2_6", "MiniCPM_o_2_6", "MiniGPT4", "MiniMonkey",
    "Mini_Gemini", "MMAlaya", "MMAlaya2", "Monkey", "MonkeyChat",
    "Moondream1", "Moondream2", "NVILA", "NVLM", "Ola", "OmChat",
    "OmniLMM12B", "OpenFlamingo", "Oryx", "Ovis", "Ovis1_6", "Ovis1_6_Plus",
    "Ovis2", "OvisU1", "POINTS", "POINTSV15", "PaliGemma", "PandaGPT",
    "Parrot", "PLLaVA", "Phi3Vision", "Phi3_5Vision", "Phi4Multimodal",
    "Pixtral", "QH_360VL", "QTuneVL", "QTuneVLChat", "Qwen2VLChat",
    "Qwen2VLChatAguvis", "QwenVL", "QwenVLChat", "RBDash", "Ristretto",
    "Ross", "SailVL", "ShareCaptioner", "SliME", "SmolVLM", "SmolVLM2",
    "TransCoreM", "UrsaChat", "VarcoVision", "VILA", "VLAAThinkerChat",
    "VLMR1Chat", "VXVERSE", "VideoChat2_HD", "VideoChatGPT", "VideoLLaVA",
    "VideoLLaVA_HF", "VinternChat", "VisualGLM", "VITA", "VITAQwen2",
    "Valley2Chat", "WeMM", "WeThinkVL", "X_VL_HF", "XComposer",
    "XComposer2", "XComposer2_4KHD", "XComposer2d5", "XGenMM", "Yi_VL",
    "HawkVL", "CogVlm", "TreeVGR", "llama_vision", "llama4", "molmo",
    "mPLUG_Owl2", "mPLUG_Owl3",
]
for _name in _EXPECTED_CLASSES:
    if _name not in globals():
        globals()[_name] = _make_missing_stub(_name)
del _name


# === Stub fill-in: any class config.py references but we couldn't import.
# config.py builds a registry like `partial(SomeModel, ...)` at module load,
# so undefined names raise NameError. Bind missing names to a stub class.
def _make_missing_stub(name):
    def _factory(*args, **kwargs):
        raise ImportError(
            f"Model class '{name}' is not bundled in Thyme's vendored "
            f"VLMEvalKit fork. Vendor the missing file from upstream "
            f"open-compass/VLMEvalKit if you need it."
        )
    _factory.__name__ = name
    return _factory


_EXPECTED_CLASSES = [
    # Names config.py references via `partial(<Name>, ...)`. If a class above
    # imported successfully it stays; otherwise we install a missing-stub.
    "AKI", "Aria", "BunnyLLama3", "Cambrian", "Chameleon", "Chatunivi",
    "DeepSeekVL", "DeepSeekVL2", "Eagle", "Emu", "Emu3_chat", "Emu3_gen",
    "Falcon2VLM", "FlashVL", "Gemma3", "GLM4_1v", "GLM4v", "GraniteVision3",
    "H2OVLChat", "IDEFICS", "IDEFICS2", "InstructBLIP", "InternVLChat",
    "Janus", "KimiVL", "Kosmos2", "LLaMAVID", "LLaVA", "LLaVA_Next",
    "LLaVA_Next2", "LLaVA_OneVision", "LLaVA_OneVision_HF", "LLaVA_XTuner",
    "LLama3Mixsense", "LongVITA", "Mantis", "MiniCPM_Llama3_V",
    "MiniCPM_V", "MiniCPM_V_2_6", "MiniCPM_o_2_6", "MiniGPT4", "MiniMonkey",
    "Mini_Gemini", "MMAlaya", "MMAlaya2", "Monkey", "MonkeyChat",
    "Moondream1", "Moondream2", "NVILA", "NVLM", "Ola", "OmChat",
    "OmniLMM12B", "OpenFlamingo", "Oryx", "Ovis", "Ovis1_6", "Ovis1_6_Plus",
    "Ovis2", "OvisU1", "POINTS", "POINTSV15", "PaliGemma", "PandaGPT",
    "Parrot", "PLLaVA", "Phi3Vision", "Phi3_5Vision", "Phi4Multimodal",
    "Pixtral", "QH_360VL", "QTuneVL", "QTuneVLChat", "Qwen2VLChat",
    "Qwen2VLChatAguvis", "QwenVL", "QwenVLChat", "RBDash", "Ristretto",
    "Ross", "SailVL", "ShareCaptioner", "SliME", "SmolVLM", "SmolVLM2",
    "TransCoreM", "UrsaChat", "VarcoVision", "VILA", "VLAAThinkerChat",
    "VLMR1Chat", "VXVERSE", "VideoChat2_HD", "VideoChatGPT", "VideoLLaVA",
    "VideoLLaVA_HF", "VinternChat", "VisualGLM", "VITA", "VITAQwen2",
    "Valley2Chat", "WeMM", "WeThinkVL", "X_VL_HF", "XComposer",
    "XComposer2", "XComposer2_4KHD", "XComposer2d5", "XGenMM", "Yi_VL",
    "HawkVL", "CogVlm", "TreeVGR", "llama_vision", "llama4", "molmo",
    "mPLUG_Owl2", "mPLUG_Owl3",
]
for _name in _EXPECTED_CLASSES:
    if _name not in globals():
        globals()[_name] = _make_missing_stub(_name)
del _name
