import os
from typing import List
from PIL import Image

# Qwen2.5-VL image processor (transformers qwen2_vl.smart_resize) hard limits:
# both sides must be >= its 28px factor and max(w,h)/min(w,h) must be <= 200,
# otherwise it raises ValueError and the whole sample becomes UNSCORED.
_IMG_MIN_SIDE = 28
_IMG_MAX_RATIO = 200


def sanitize_intermediate_image(img_path, max_ratio=_IMG_MAX_RATIO, min_side=_IMG_MIN_SIDE):
    """Return a processor-safe path for a *model-produced* intermediate image.

    Model-generated sandbox code can save degenerate images (e.g. a 291x1
    strip from a bad crop), which makes transformers qwen2_vl.smart_resize
    raise and fail-closes the whole sample (UNSCORED -> GCEP4 ABORT).  This
    clamps such images to the processor's accepted geometry before they are
    fed back to the model on the next iteration.

    Benchmark *source* images never pass through here (they are not produced
    by the sandbox), so corrupt originals keep their fail-closed behaviour.
    Any unreadable file is returned untouched so the original error surface
    is never masked.
    """
    try:
        with Image.open(img_path) as im:
            w, h = im.size
            if w >= min_side and h >= min_side and max(w, h) / min(w, h) <= max_ratio:
                return img_path
            # Upscale a degenerate tiny side to the processor's 28px factor,
            # then clamp the aspect ratio to the processor's 200 limit.
            scale = max(min_side / min(w, h), 1.0)
            new_w = max(min_side, round(w * scale))
            new_h = max(min_side, round(h * scale))
            if max(new_w, new_h) / min(new_w, new_h) > max_ratio:
                if new_w >= new_h:
                    new_w = max(min_side, round(new_h * max_ratio))
                else:
                    new_h = max(min_side, round(new_w * max_ratio))
            im = im.convert("RGB").resize((new_w, new_h), Image.BILINEAR)
            base, _ = os.path.splitext(img_path)
            safe_path = f"{base}.safe_{new_w}x{new_h}.png"
            im.save(safe_path, format="PNG")
            return safe_path
    except Exception:
        return img_path


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


SIMPLE_SYS_PROMPT="You are a helpful assistant."

def generate_prompt_simple_qa(user_question):
    # Construct the prompt based on the given requirements
    prompt = f"""
You are an advanced AI assistant specializing in visual question answering (VQA). You don't need to perform any image manipulation or reasoning. Give the answer to the following question directly.
**User's Question:** "{user_question}"
"""
    return prompt

def generate_prompt_final_qa(user_question, user_image_path):
    # Construct the prompt based on the given requirements
    try:
        with Image.open(user_image_path) as img:
            user_image_size = f"{img.width}x{img.height}"
    except Exception as e:
        user_image_size = "Unable to determine (error reading image)"

    prompt = f"""<image>
{user_question}

### User Image Path:** "{user_image_path}"
### User Image Size:** "{user_image_size}"

### **Output Format (strict adherence required):**

<think>Your detailed reasoning process, including any code, should go here.</think>
<answer>Your final answer to the user's question goes here.</answer>
"""
    return prompt

SPECIAL_STRING_LIST=["</code>", "</answer>"]

