# Modified for anonymous review: release paths, configuration, and documentation.
"""GCEP extended-benchmark adapters (TreeBench/ZoomBench/VisualProbe/ReasonMap-Plus/ChartQAPro/InfographicVQA).

Benchmark protocols are defined in the corresponding scorers and dataset builders.
Constraint: this package is imported by phase3_eval_opd.py BEFORE its env-stripping
shim; it must never import torch (directly or transitively).
"""

from . import common, manifests  # noqa: F401
