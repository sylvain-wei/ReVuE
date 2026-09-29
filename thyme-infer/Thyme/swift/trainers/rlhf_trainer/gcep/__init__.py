"""GCEP (Group-Contrastive Evidence-Chain Privilege) 组件包。

惰性导出子模块，保证 import 本包时不引入 torch / swift 等重依赖。
"""

__all__ = [
    "judge_client",
    "validator",
    "anchor",
    "privilege",
    "rollout_archive",
    "evidence_state",
    "opd_losses",
    "group_types",
]


def __getattr__(name):
    if name in __all__:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
