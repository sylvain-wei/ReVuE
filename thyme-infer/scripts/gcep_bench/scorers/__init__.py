# Modified for anonymous review: release paths, configuration, and documentation.
"""Per-benchmark scorers. Each module exposes:

    PROTOCOL_NAME: str            # scoring protocol identifier
    SCORING_LAYER: str            # OFFICIAL_DETERMINISTIC | ADAPTED_LLM_JUDGE | ...
    score_sample(sample, prediction, **ctx) -> tuple[Optional[float], dict]
        correct in {0.0, 1.0} or None for UNSCORED; meta carries audit fields.
    aggregate(rows) -> dict       # rows: [{"sample":..,"prediction":..,"correct":..,"meta":..}]

Judge-backed scorers receive ``judge=GcepTextJudge`` via ctx (constructed lazily
by the caller only when the benchmark's protocol requires it).
"""

from __future__ import annotations

import importlib

_REGISTRY = {
    "TreeBench": "treebench",
    "ZoomBench": "zoombench",
    "VisualProbe_Easy": "visualprobe",
    "VisualProbe_Medium": "visualprobe",
    "VisualProbe_Hard": "visualprobe",
    "ReasonMapPlus": "reasonmap_plus",
    "ChartQAPro": "chartqapro",
    "InfographicVQA_val": "infographicvqa",
    # Additional benchmark scorers
    "PerceptionBench": "perceptionbench",
    "HallusionBench_GCEP": "hallusionbench",
    "MathVerseVO": "mathverse_vo",
    "VisuLogic_GCEP": "visulogic_gcep",
}

GCEP6_BENCHMARKS = tuple(_REGISTRY.keys())
GCEP4_BENCHMARKS = ("PerceptionBench", "HallusionBench_GCEP", "MathVerseVO", "VisuLogic_GCEP")


def get_scorer(benchmark: str):
    mod_name = _REGISTRY.get(benchmark)
    if mod_name is None:
        raise KeyError(f"no gcep_bench scorer registered for benchmark {benchmark!r}")
    return importlib.import_module(f".{mod_name}", package=__name__)
