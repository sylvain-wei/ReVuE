# Metis（Qwen3-VL-8B）模型注册 —— 仿 internlm.py 的 internvl3_5 先例。
# 惰性设计：本模块在 thyme env（transformers 4.52.4）下 import 也必须零副作用——
# Qwen3VL 类只在 loader 函数体内 import，注册仅登记元数据。
from swift.llm import TemplateType

from ..constant import MLLMModelType
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal, register_model
from ..patcher import patch_output_clone, patch_output_to_input_device


def get_model_tokenizer_metis(*args, **kwargs):
    # 惰性：只有真正加载 Metis 时才要求 transformers>=4.57
    from transformers import Qwen3VLForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen3VLForConditionalGeneration
    model, tokenizer = get_model_tokenizer_multimodal(*args, **kwargs)
    if model is not None:
        # Qwen3-VL 8B tie_word_embeddings=True：embed_tokens 需要 clone/to_input_device 补丁。
        # 结构差异：Qwen2.5-VL 在 model.model.embed_tokens；Qwen3-VL 在 model.model.language_model.embed_tokens。
        base = model.model
        emb = getattr(base, 'embed_tokens', None)
        if emb is None and hasattr(base, 'language_model'):
            emb = base.language_model.embed_tokens
        if emb is not None:
            patch_output_clone(emb)
            patch_output_to_input_device(emb)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.qwen3_vl,
        [
            ModelGroup([
                Model('Accio-Lab/Metis-8B-ColdStart', 'Accio-Lab/Metis-8B-ColdStart'),
                Model('Accio-Lab/Metis-8B-RL', 'Accio-Lab/Metis-8B-RL'),
            ]),
        ],
        # 模板：MetisQwen3VLTemplate（Qwen2_5VLTemplate 子类，_post_encode 直通保 deepstack）
        TemplateType.qwen3_vl,
        get_model_tokenizer_metis,
        architectures=['Qwen3VLForConditionalGeneration'],
        requires=['transformers>=4.57'],
        tags=['vision', 'video'],
    ))
