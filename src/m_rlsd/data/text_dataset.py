"""Text OPD dataset and data loading utilities."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset


@dataclass
class TextOPDExample:
    """Single training example for text OPD.

    Schema:
        id: Unique example identifier
        prompt: User question/task
        privileged_context: Ground-truth solution or hint (teacher-only)
        answer: Gold answer (for evaluation)
        solution: Full reference solution string. Optional; populated only
            when the source parquet has a ``solution`` column (Step 2 data).
            For Step 1 (answer-only) data this stays ``None`` / empty string.
    """
    id: str
    prompt: str
    privileged_context: str
    answer: str
    solution: str = ""


class TextOPDDataset(Dataset):
    """PyTorch Dataset for text OPD examples."""
    
    def __init__(self, parquet_path: str):
        """Load from parquet file.

        Two schemas are accepted:

        * Legacy (Phase 1 Step 0): ``id, prompt, privileged_context, answer``
          (privileged_context typically a textual hint or solution).

        * Step 1 (OpenThoughts answer-only): ``id, problem, answer``
          -- no separate ``privileged_context`` column.  The renderer
          consumes ``answer`` directly as the privilege payload.

        * Step 2 (OpenThoughts solution): ``id, problem, solution, answer``
          -- ``solution`` carries the full reference reasoning (which itself
          contains the boxed final answer). The renderer uses ``solution``
          as the privilege payload in solution_only mode; the legacy
          ``privileged_context`` is still synthesized from ``answer`` for
          back-compat with Step 1 callers.

        For the new schemas we synthesize the legacy column names in-memory
        so downstream code (rollout, teacher score, trainer) keeps working
        without changes:

            prompt              <- problem
            privileged_context  <- answer  (Step 1 answer-only privilege)
            solution            <- solution if present else ""
        """
        self.parquet_path = Path(parquet_path)
        self.df = pd.read_parquet(self.parquet_path)

        # Step 1 / Step 2 -> legacy adapter.
        cols = set(self.df.columns)
        if "prompt" not in cols and "problem" in cols:
            self.df = self.df.assign(prompt=self.df["problem"])
        if "privileged_context" not in cols and "answer" in self.df.columns:
            self.df = self.df.assign(privileged_context=self.df["answer"])
        if "solution" not in self.df.columns:
            self.df = self.df.assign(solution="")

        # Validate required columns
        required_cols = {"id", "prompt", "privileged_context", "answer"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Missing columns: {missing}. Got: {sorted(self.df.columns)}"
            )

        # Quick validation: no NaN values
        if self.df[list(required_cols)].isnull().any().any():
            raise ValueError("Found NaN values in required columns")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> TextOPDExample:
        row = self.df.iloc[idx]
        sol = row["solution"] if "solution" in row else ""
        return TextOPDExample(
            id=row["id"],
            prompt=row["prompt"],
            privileged_context=row["privileged_context"],
            answer=row["answer"],
            solution=sol if sol is not None else "",
        )
    
    def validate(self) -> dict:
        """Run data quality checks.
        
        Returns dict with validation results.
        """
        results = {
            "total_examples": len(self),
            "has_nans": self.df[["id", "prompt", "privileged_context", "answer"]].isnull().any().any(),
            "avg_prompt_chars": self.df["prompt"].str.len().mean(),
            "avg_context_chars": self.df["privileged_context"].str.len().mean(),
            "avg_answer_chars": self.df["answer"].str.len().mean(),
        }
        
        # Check for no leakage: privileged_context should NOT appear in prompt
        leakage_count = 0
        for idx in range(min(len(self), 100)):  # Sample first 100
            ex = self[idx]
            # Simple heuristic: check if any word from answer appears in prompt
            answer_words = set(ex.answer.lower().split())
            prompt_words = set(ex.prompt.lower().split())
            if answer_words & prompt_words:
                leakage_count += 1
        results["potential_answer_leakage_sample_count"] = leakage_count
        
        return results


def write_text_opd_parquet(examples: list[TextOPDExample], output_path: str) -> None:
    """Write list of TextOPDExample to parquet file.
    
    Parameters
    ----------
    examples : list[TextOPDExample]
        List of training examples
    output_path : str
        Path to write parquet file
    """
    data = [asdict(ex) for ex in examples]
    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} examples to {output_path}")


def create_toy_dataset(output_path: str, n_examples: int = 10) -> None:
    """Create a tiny toy dataset for smoke testing.
    
    Uses simple math/reasoning examples.
    """
    examples = []
    
    prompts = [
        "What is 2 + 2?",
        "Solve: 5x = 20. What is x?",
        "If a triangle has sides 3, 4, and 5, is it a right triangle?",
        "What is the capital of France?",
        "Simplify: (2^3) * (2^2)",
        "What is 10% of 50?",
        "Is 17 a prime number?",
        "What is the area of a circle with radius 3?",
        "Solve: x^2 - 4 = 0",
        "What is the sum of angles in a triangle?",
    ]
    
    contexts = [
        "The answer can be found by adding 2 and 2 together.",
        "Divide both sides by 5: x = 20/5 = 4",
        "Yes, because 3^2 + 4^2 = 9 + 16 = 25 = 5^2 (Pythagorean theorem)",
        "France is in Western Europe, and its capital is the most visited city in the world.",
        "Use exponent rule: a^m * a^n = a^(m+n), so 2^3 * 2^2 = 2^5",
        "10% means 1/10, so 50 * (1/10) = 5",
        "17 is only divisible by 1 and 17, so yes it's prime.",
        "Area of circle = π * r^2 = π * 3^2 = 9π",
        "Rearrange to (x-2)(x+2) = 0, so x = 2 or x = -2",
        "All triangles have interior angles summing to 180 degrees.",
    ]
    
    answers = [
        "4",
        "4",
        "Yes",
        "Paris",
        "32",
        "5",
        "Yes",
        "9π",
        "2 or -2",
        "180 degrees",
    ]
    
    for i in range(n_examples):
        idx = i % len(prompts)
        examples.append(TextOPDExample(
            id=f"toy_{i:05d}",
            prompt=prompts[idx],
            privileged_context=contexts[idx],
            answer=answers[idx],
        ))
    
    write_text_opd_parquet(examples, output_path)


if __name__ == "__main__":
    # Create toy dataset for testing
    toy_path = "data/toy_text_opd_10.parquet"
    Path("data").mkdir(exist_ok=True)
    create_toy_dataset(toy_path, n_examples=10)
    
    # Load and validate
    dataset = TextOPDDataset(toy_path)
    print(f"\nDataset loaded: {len(dataset)} examples")
    print("\nValidation results:")
    for k, v in dataset.validate().items():
        print(f"  {k}: {v}")
    
    print("\nFirst example:")
    ex = dataset[0]
    print(f"  ID: {ex.id}")
    print(f"  Prompt: {ex.prompt}")
    print(f"  Privileged Context: {ex.privileged_context}")
    print(f"  Answer: {ex.answer}")
