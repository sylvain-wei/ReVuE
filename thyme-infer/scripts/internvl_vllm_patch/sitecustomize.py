# Modified for anonymous review: release paths, configuration, and documentation.
"""Runtime patch: InternVL3.5 vision tower under vLLM 0.9.2 + flash_attn 2.8.3.

Problem: vLLM 0.9.2's `MultiHeadAttention` (used by the InternVL vision tower in
`model_executor/models/intern_vit.py`) maps FLASH_ATTN / FLASH_ATTN_VLLM_V1 /
FLEX_ATTENTION -> XFORMERS. This environment ships flash_attn 2.8.3 while the
installed xformers pins `>=2.7.1,<=2.7.4` at module import, so ANY XFORMERS call
dies with `ImportError: Requires Flash-Attention version ...` and InternVL3.5
cannot roll out under vLLM colocate.

Fix (no shared package modified): inject this directory on PYTHONPATH; Python
auto-imports this `sitecustomize` at interpreter startup, installing an import
hook that flips the vision `MultiHeadAttention` backend from the broken
XFORMERS to TORCH_SDPA. `torch.nn.functional.scaled_dot_product_attention`
dispatches to the flash/mem-efficient kernels on H20+bf16, so the vision tower
keeps flash-class speed (zero efficiency cost). The LM decoder is untouched and
still uses vLLM's native FLASH_ATTN (`vllm.vllm_flash_attn`, which works here).

Usage:  export PYTHONPATH=<this_dir>:$PYTHONPATH   before vllm/training launch.
"""
import importlib.abc
import importlib.machinery
import sys

_TARGET = "vllm.attention.layer"


def _apply(module) -> None:
    try:
        backend_enum = module._Backend
        cls = module.MultiHeadAttention
    except AttributeError:
        return
    if getattr(cls, "_internvl_sdpa_patched", False):
        return
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # XFORMERS is unusable in this env (flash_attn 2.8.3 vs xformers pin
        # <=2.7.4); SDPA dispatches to flash kernels on Hopper — equivalent.
        if getattr(self, "attn_backend", None) == backend_enum.XFORMERS:
            self.attn_backend = backend_enum.TORCH_SDPA

    cls.__init__ = __init__
    cls._internvl_sdpa_patched = True


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return None
        orig_loader = spec.loader

        class _Loader(importlib.abc.Loader):
            def create_module(self, s):
                return orig_loader.create_module(s)

            def exec_module(self, module):
                orig_loader.exec_module(module)
                _apply(module)

        spec.loader = _Loader()
        return spec


sys.meta_path.insert(0, _Hook())
