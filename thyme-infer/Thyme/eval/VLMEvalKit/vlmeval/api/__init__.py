# Patched __init__.py: Thyme's vendored VLMEvalKit ships an incomplete `api/`
# subdirectory. Several wrappers (gemini, qwen_vl_api, qwen_api, reka,
# sensechat_vision, bailingmm, bluelm_api) referenced by the upstream
# __init__.py are not present in this fork. We wrap each import in
# try/except so vlmeval can still be imported. The Thyme model class only
# needs `GPT4V` (used as the MathVista judge); all other API wrappers are
# irrelevant to Phase 2 inference.

# Always available (files present in the repo)
from .gpt import OpenAIWrapper, GPT4V
from .hf_chat_model import HFChatModel
from .claude import Claude_Wrapper, Claude3V
from .glm_vision import GLMVisionAPI
from .cloudwalk import CWWrapper
from .siliconflow import SiliconFlowAPI, TeleMMAPI
from .hunyuan import HunyuanVision
from .jt_vl_chat import JTVLChatAPI
from .taiyi import TaiyiAPI
from .lmdeploy import LMDeployAPI
from .taichu import TaichuVLAPI, TaichuVLRAPI
from .doubao_vl_api import DoubaoVL
from .mug_u import MUGUAPI
from .kimivl_api import KimiVLAPIWrapper, KimiVLAPI

# Optional API wrappers — files missing in Thyme's vendored fork.
# We make them callable stubs that raise at construction time, so that
# `partial(MissingClass, ...)` at module-load is fine, but actually trying
# to instantiate the missing model raises a clear error.
class _MissingAPIStub:
    def __init__(self, *args, _missing_name="API wrapper", **kwargs):
        raise ImportError(
            f"{_missing_name} is not available in Thyme's vendored "
            f"VLMEvalKit fork. Use a different model or vendor the missing "
            f"file from upstream open-compass/VLMEvalKit."
        )


def _missing(name):
    """Return a callable stub for a missing API class."""
    def _factory(*args, **kwargs):
        raise ImportError(
            f"{name} is not available in Thyme's vendored VLMEvalKit fork."
        )
    return _factory


GeminiWrapper = _missing("GeminiWrapper")
Gemini = _missing("Gemini")
QwenVLWrapper = _missing("QwenVLWrapper")
QwenVLAPI = _missing("QwenVLAPI")
Qwen2VLAPI = _missing("Qwen2VLAPI")
QwenAPI = _missing("QwenAPI")
Reka = _missing("Reka")
SenseChatVisionAPI = _missing("SenseChatVisionAPI")
bailingMMAPI = _missing("bailingMMAPI")
BlueLMWrapper = _missing("BlueLMWrapper")
BlueLM_API = _missing("BlueLM_API")

__all__ = [
    'OpenAIWrapper', 'HFChatModel', 'GeminiWrapper', 'GPT4V', 'Gemini',
    'QwenVLWrapper', 'QwenVLAPI', 'QwenAPI', 'Claude3V', 'Claude_Wrapper',
    'Reka', 'GLMVisionAPI', 'CWWrapper', 'SenseChatVisionAPI', 'HunyuanVision',
    'Qwen2VLAPI', 'BlueLMWrapper', 'BlueLM_API', 'JTVLChatAPI',
    'bailingMMAPI', 'TaiyiAPI', 'TeleMMAPI', 'SiliconFlowAPI', 'LMDeployAPI',
    'TaichuVLAPI', 'TaichuVLRAPI', 'DoubaoVL', "MUGUAPI", 'KimiVLAPIWrapper', 'KimiVLAPI'
]
