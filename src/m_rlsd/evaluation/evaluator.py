"""Evaluation module for benchmark inference results."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvalMetrics:
    """Evaluation metrics for a single episode."""
    episode_id: str
    correct: bool
    exact_match: bool
    answer_extracted: bool
    final_answer: Optional[str] = None
    ground_truth_answer: Optional[str] = None
    benchmark: Optional[str] = None
    num_turns: int = 0
    code_execution_success_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnswerMatcher:
    """Answer matching utilities for different domains."""
    
    @staticmethod
    def exact_match(predicted: str, ground_truth: str) -> bool:
        """Check exact match after normalization."""
        p = AnswerMatcher._normalize(predicted)
        g = AnswerMatcher._normalize(ground_truth)
        return p == g
    
    @staticmethod
    def numeric_match(predicted: str, ground_truth: str, tolerance: float = 0.01) -> bool:
        """Check numeric match with tolerance."""
        try:
            p_num = float(predicted)
            g_num = float(ground_truth)
            rel_error = abs(p_num - g_num) / (abs(g_num) + 1e-10)
            return rel_error <= tolerance
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def substring_match(predicted: str, ground_truth: str) -> bool:
        """Check if ground truth is substring of predicted."""
        p = AnswerMatcher._normalize(predicted)
        g = AnswerMatcher._normalize(ground_truth)
        return g.lower() in p.lower()
    
    @staticmethod
    def case_insensitive_match(predicted: str, ground_truth: str) -> bool:
        """Check case-insensitive match."""
        p = AnswerMatcher._normalize(predicted).lower()
        g = AnswerMatcher._normalize(ground_truth).lower()
        return p == g
    
    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for matching."""
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove common punctuation
        text = text.rstrip('.,!?;:')
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    @staticmethod
    def extract_latex_box(text: str) -> Optional[str]:
        """Extract answer from \\boxed{...} LaTeX format."""
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1).strip()
        return None
    
    @staticmethod
    def extract_numeric(text: str) -> Optional[float]:
        """Extract first numeric value from text."""
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
        return None


class BenchmarkEvaluator:
    """Evaluator for benchmark inference results."""
    
    def __init__(self, benchmark: str = "unknown", strict: bool = False):
        """Initialize evaluator.
        
        Args:
            benchmark: Benchmark name (mathvista, mme, etc.)
            strict: Whether to use strict matching
        """
        self.benchmark = benchmark.lower()
        self.strict = strict
        self.episodes: List[Dict[str, Any]] = []
    
    def evaluate_episode(self, episode_path: str) -> EvalMetrics:
        """Evaluate a single episode.
        
        Args:
            episode_path: Path to episode JSON file
            
        Returns:
            EvalMetrics with evaluation results
        """
        with open(episode_path, 'r') as f:
            episode = json.load(f)
        
        episode_id = episode.get("episode_id", "unknown")
        final_answer = episode.get("final_answer")
        ground_truth = episode.get("ground_truth_answer")
        benchmark = episode.get("benchmark", self.benchmark)
        
        # Check if answer was extracted
        answer_extracted = final_answer is not None and final_answer != ""
        
        # Evaluate correctness
        correct = False
        exact_match = False
        
        if answer_extracted and ground_truth:
            # Try different matching strategies
            if AnswerMatcher.exact_match(final_answer, ground_truth):
                correct = True
                exact_match = True
            elif AnswerMatcher.case_insensitive_match(final_answer, ground_truth):
                correct = True
            elif AnswerMatcher.numeric_match(final_answer, ground_truth):
                correct = True
            elif AnswerMatcher.substring_match(final_answer, ground_truth):
                correct = True if not self.strict else False
        
        # Calculate code execution success rate
        code_execution_success_rate = self._get_code_success_rate(episode)
        
        # Count turns
        num_turns = len(episode.get("events", []))
        
        metrics = EvalMetrics(
            episode_id=episode_id,
            correct=correct,
            exact_match=exact_match,
            answer_extracted=answer_extracted,
            final_answer=final_answer,
            ground_truth_answer=ground_truth,
            benchmark=benchmark,
            num_turns=num_turns,
            code_execution_success_rate=code_execution_success_rate,
        )
        
        self.episodes.append(episode)
        return metrics
    
    def compute_summary(self, detailed: bool = False) -> Dict[str, Any]:
        """Compute summary metrics across all episodes.
        
        Args:
            detailed: Whether to include per-category breakdowns
            
        Returns:
            Dictionary with summary metrics
        """
        if not self.episodes:
            return {
                "total_episodes": 0,
                "accuracy": 0.0,
                "answer_extraction_rate": 0.0,
                "avg_turns": 0.0,
                "avg_code_execution_success_rate": 0.0,
            }
        
        total = len(self.episodes)
        correct_count = 0
        extracted_count = 0
        total_turns = 0
        total_code_success = 0.0
        
        # Aggregate metrics
        for episode in self.episodes:
            final_answer = episode.get("final_answer")
            ground_truth = episode.get("ground_truth_answer")
            
            # Check extraction
            if final_answer is not None and final_answer != "":
                extracted_count += 1
            
            # Check correctness
            if final_answer and ground_truth:
                if (AnswerMatcher.exact_match(final_answer, ground_truth) or
                    AnswerMatcher.case_insensitive_match(final_answer, ground_truth) or
                    AnswerMatcher.numeric_match(final_answer, ground_truth) or
                    AnswerMatcher.substring_match(final_answer, ground_truth)):
                    correct_count += 1
            
            # Count turns
            total_turns += len(episode.get("events", []))
            
            # Aggregate code execution rates
            code_rate = self._get_code_success_rate(episode)
            total_code_success += code_rate
        
        summary = {
            "total_episodes": total,
            "accuracy": correct_count / total if total > 0 else 0.0,
            "answer_extraction_rate": extracted_count / total if total > 0 else 0.0,
            "avg_turns": total_turns / total if total > 0 else 0.0,
            "avg_code_execution_success_rate": total_code_success / total if total > 0 else 0.0,
            "benchmark": self.benchmark,
        }
        
        return summary
    
    @staticmethod
    def _get_code_success_rate(episode: Dict[str, Any]) -> float:
        """Get code execution success rate for an episode."""
        events = episode.get("events", [])
        
        if not events:
            return 0.0
        
        code_executions = 0
        code_successes = 0
        
        for event in events:
            if event.get("role") == "tool":
                # This is a code execution observation
                code_executions += 1
                if event.get("execution_status") or "success" in str(event).lower():
                    code_successes += 1
        
        return code_successes / code_executions if code_executions > 0 else 0.0


def evaluate_episodes_file(
    episodes_file: str,
    benchmark: str = "unknown",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate all episodes from a JSONL file.
    
    Args:
        episodes_file: Path to JSONL file with episodes
        benchmark: Benchmark name
        output_dir: Optional directory to save results
        
    Returns:
        Summary metrics
    """
    evaluator = BenchmarkEvaluator(benchmark=benchmark)
    
    # Load and evaluate all episodes
    with open(episodes_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            episode = json.loads(line)
            # Create temp file for evaluation (BenchmarkEvaluator expects file paths)
            # For simplicity, we'll evaluate inline
            episode_id = episode.get("episode_id", "unknown")
            final_answer = episode.get("final_answer")
            ground_truth = episode.get("ground_truth_answer")
            
            # Check correctness
            correct = False
            if final_answer and ground_truth:
                if (AnswerMatcher.exact_match(final_answer, ground_truth) or
                    AnswerMatcher.case_insensitive_match(final_answer, ground_truth) or
                    AnswerMatcher.numeric_match(final_answer, ground_truth)):
                    correct = True
            
            evaluator.episodes.append(episode)
    
    # Compute summary
    summary = evaluator.compute_summary()
    
    # Save results if output_dir is provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary_file = output_path / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Results saved to {summary_file}")
    
    return summary
