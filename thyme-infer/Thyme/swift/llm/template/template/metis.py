# Metis（Qwen3-VL-8B）模板 —— Qwen2_5VLTemplate 子类。
# 关键差异（2026-09 勘察结论）：Qwen3-VL 的 visual 前向带 deepstack 分支输出，
# 父类 _post_encode 手工 `model.visual(...)` + masked_scatter 拼 inputs_embeds 会
# 静默丢掉 deepstack → 训练 logp 与真实前向不一致。这里改为直通：把 pixel_values
# 交还模型原生 forward（由 transformers 4.57 的 Qwen3VL 自己处理 deepstack 注入）。
from typing import Any, Dict

from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import findall
from .qwen import Qwen2VLTemplate, Qwen2_5VLTemplate, QwenTemplateMeta
from ..constant import MLLMTemplateType


class MetisQwen3VLTemplate(Qwen2_5VLTemplate):
    # version 用于父类的 v2/v2_5/omni 分支判断（视频 second_per_grid_ts、packing 时
    # 的 flash-attn 补丁模块选择）。Qwen3-VL 均不适用 → 独立取值，自然跳过那些分支。
    version = 'qwen3vl'

    def replace_tag(self, media_type, index, inputs):
        # 父类对 image 调 qwen_vl_utils.fetch_image：其 image_patch_size=14（factor=28）
        # 是 Qwen2.5-VL 遗留，与 Qwen3-VL patch16×merge2=32 不符，且会做一次额外的
        # min/max 像素预缩放。这里保持原图像素，网格化完全交给处理器（与评测侧一致）。
        if media_type == 'image':
            return ['<|vision_start|><|image_pad|><|vision_end|>']
        return super().replace_tag(media_type, index, inputs)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        # 与父类（Qwen2VLTemplate._encode）逐项相同，唯一差异：image_processor 调用
        # 去掉 do_resize=False —— transformers 4.57 的 Qwen2VLImageProcessor(Fast) 在
        # do_resize=False 下对非网格对齐图片 reshape 必崩（4.52 无此问题）；且 Metis
        # ckpt 的忠实路径本来就是处理器内部 smart_resize（size.shortest_edge=65536 /
        # longest_edge=16777216 上/下限），swift 的 max_pixels 预缩放在更前面生效。
        encoded = super(Qwen2VLTemplate, self)._encode(inputs)  # 跳到 Template 基类层
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        images = inputs.images
        videos = inputs.videos
        for media_type in ['images', 'videos']:
            if locals()[media_type]:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(
                        images=images, videos=None, return_tensors='pt')  # ← 不带 do_resize=False
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    media_inputs = processor.image_processor(
                        images=None, videos=videos, return_tensors='pt')
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    return [media_token] * token_len

                input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        # 直通：不构造 inputs_embeds；pixel 字段原样传给模型原生 forward。
        # pre_forward_hook 会从 old_kwargs 回填 input_ids/attention_mask/labels/position_ids，
        # 这里只需带上 pixel 相关字段（且只在存在时携带）。
        out: Dict[str, Any] = {}
        for k in ('pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw'):
            if inputs.get(k) is not None:
                out[k] = inputs[k]
        return out


register_template(QwenTemplateMeta(MLLMTemplateType.qwen3_vl, template_cls=MetisQwen3VLTemplate))
