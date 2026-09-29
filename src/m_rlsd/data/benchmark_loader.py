"""Benchmark data loaders for Phase 2 inference.

Supports loading and validating benchmark datasets:
- MME-RealWorld-Lite: Real-world image understanding
- MathVista Mini: Visual math and computation

All loaders inherit from BenchmarkDataset base class.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np


@dataclass
class BenchmarkExample:
    """A single example from a benchmark dataset."""
    example_id: str
    question: str
    image_path: str
    ground_truth_answer: str
    benchmark: str
    metadata: Optional[Dict[str, Any]] = None

    def validate(self) -> Dict[str, bool]:
        """Validate example fields.
        
        Returns:
            Dict with validation results for each field
        """
        checks = {
            "has_id": bool(self.example_id),
            "has_question": bool(self.question),
            "has_image_path": bool(self.image_path),
            "has_answer": bool(self.ground_truth_answer),
            "image_exists": Path(self.image_path).exists() if self.image_path else False,
        }
        return checks


class BenchmarkDataset(ABC):
    """Base class for benchmark datasets."""

    def __init__(self, data_path: Union[str, Path], image_root: Optional[Union[str, Path]] = None):
        """Initialize benchmark dataset.
        
        Args:
            data_path: Path to dataset (parquet, jsonl, json, or directory)
            image_root: Root directory for images (if not absolute paths in data)
        """
        self.data_path = Path(data_path)
        self.image_root = Path(image_root) if image_root else None
        self.examples: List[BenchmarkExample] = []
        self.benchmark_name = self.__class__.__name__

    @abstractmethod
    def load(self) -> None:
        """Load dataset from data_path."""
        pass

    def validate_all(self) -> Dict[str, Any]:
        """Validate all examples in dataset.
        
        Returns:
            Summary of validation results
        """
        if not self.examples:
            return {
                "total_examples": 0,
                "valid_examples": 0,
                "all_valid": False,
                "failed_examples": [],
            }

        valid_count = 0
        failed = []

        for ex in self.examples:
            checks = ex.validate()
            if all(checks.values()):
                valid_count += 1
            else:
                failed.append({
                    "example_id": ex.example_id,
                    "failures": [k for k, v in checks.items() if not v]
                })

        return {
            "total_examples": len(self.examples),
            "valid_examples": valid_count,
            "all_valid": valid_count == len(self.examples),
            "failed_examples": failed,
        }

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> BenchmarkExample:
        """Get example by index."""
        return self.examples[idx]

    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)


class MathVistaDataset(BenchmarkDataset):
    """Loader for MathVista Mini benchmark.
    
    Data format (JSONL or JSON lines):
    ```
    {
        "image": "path/to/image.png",
        "question": "What is the area?",
        "answer": "42"
    }
    ```
    """

    def load(self) -> None:
        """Load MathVista dataset from JSONL or JSON file."""
        if self.data_path.suffix == ".jsonl":
            self._load_jsonl()
        elif self.data_path.suffix == ".json":
            self._load_json()
        else:
            raise ValueError(f"Unsupported format: {self.data_path.suffix}")

    def _load_jsonl(self) -> None:
        """Load from JSONL file."""
        with open(self.data_path, "r") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                example = self._parse_record(data, line_idx)
                if example:
                    self.examples.append(example)

    def _load_json(self) -> None:
        """Load from JSON file (array of records)."""
        with open(self.data_path, "r") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # Assume it's wrapped: {"data": [...]}
            data = data.get("data", [])
        
        if not isinstance(data, list):
            raise ValueError("JSON must contain list of records or {\"data\": [...]}")

        for idx, record in enumerate(data):
            example = self._parse_record(record, idx)
            if example:
                self.examples.append(example)

    def _parse_record(self, record: Dict, idx: int) -> Optional[BenchmarkExample]:
        """Parse a single record into BenchmarkExample."""
        try:
            image_path = record.get("image") or record.get("image_path")
            question = record.get("question") or record.get("text")
            answer = record.get("answer") or record.get("ground_truth")

            if not all([image_path, question, answer]):
                return None

            # Make image path absolute if image_root is set
            if self.image_root and not Path(image_path).is_absolute():
                image_path = str(self.image_root / image_path)

            example_id = record.get("id") or f"mathvista_{idx:06d}"

            return BenchmarkExample(
                example_id=example_id,
                question=question,
                image_path=str(image_path),
                ground_truth_answer=str(answer),
                benchmark="MathVista",
                metadata=record,
            )
        except Exception as e:
            print(f"Warning: Failed to parse MathVista record {idx}: {e}")
            return None


class MMERealWorldDataset(BenchmarkDataset):
    """Loader for MME-RealWorld-Lite benchmark.
    
    Data format (JSONL or JSON lines):
    ```
    {
        "image": "path/to/image.jpg",
        "question": "What objects are in this image?",
        "answer": "..."
    }
    ```
    """

    def load(self) -> None:
        """Load MME-RealWorld dataset from JSONL or JSON file."""
        if self.data_path.suffix == ".jsonl":
            self._load_jsonl()
        elif self.data_path.suffix == ".json":
            self._load_json()
        else:
            raise ValueError(f"Unsupported format: {self.data_path.suffix}")

    def _load_jsonl(self) -> None:
        """Load from JSONL file."""
        with open(self.data_path, "r") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                example = self._parse_record(data, line_idx)
                if example:
                    self.examples.append(example)

    def _load_json(self) -> None:
        """Load from JSON file (array of records)."""
        with open(self.data_path, "r") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # Assume it's wrapped: {"data": [...]}
            data = data.get("data", [])
        
        if not isinstance(data, list):
            raise ValueError("JSON must contain list of records or {\"data\": [...]}")

        for idx, record in enumerate(data):
            example = self._parse_record(record, idx)
            if example:
                self.examples.append(example)

    def _parse_record(self, record: Dict, idx: int) -> Optional[BenchmarkExample]:
        """Parse a single record into BenchmarkExample."""
        try:
            image_path = record.get("image") or record.get("image_path")
            question = record.get("question") or record.get("text")
            answer = record.get("answer") or record.get("ground_truth")

            if not all([image_path, question, answer]):
                return None

            # Make image path absolute if image_root is set
            if self.image_root and not Path(image_path).is_absolute():
                image_path = str(self.image_root / image_path)

            example_id = record.get("id") or f"mme_{idx:06d}"

            return BenchmarkExample(
                example_id=example_id,
                question=question,
                image_path=str(image_path),
                ground_truth_answer=str(answer),
                benchmark="MME-RealWorld",
                metadata=record,
            )
        except Exception as e:
            print(f"Warning: Failed to parse MME record {idx}: {e}")
            return None


class BenchmarkFactory:
    """Factory for creating benchmark dataset loaders."""

    _registry: Dict[str, type] = {
        "mathvista": MathVistaDataset,
        "mme": MMERealWorldDataset,
        "mme-realworld": MMERealWorldDataset,
    }

    @classmethod
    def create(
        cls,
        benchmark_name: str,
        data_path: Union[str, Path],
        image_root: Optional[Union[str, Path]] = None,
    ) -> BenchmarkDataset:
        """Create a benchmark dataset loader.
        
        Args:
            benchmark_name: Name of benchmark ("mathvista", "mme", etc.)
            data_path: Path to dataset
            image_root: Root directory for images
            
        Returns:
            Instantiated BenchmarkDataset subclass
            
        Raises:
            ValueError: If benchmark name not recognized
        """
        name_lower = benchmark_name.lower().replace("-", "").replace("_", "")
        
        for key, loader_cls in cls._registry.items():
            if key in name_lower or name_lower.startswith(key):
                dataset = loader_cls(data_path, image_root)
                dataset.load()
                return dataset

        available = ", ".join(cls._registry.keys())
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}\n"
            f"Available benchmarks: {available}"
        )

    @classmethod
    def register(cls, name: str, loader_class: type) -> None:
        """Register a new benchmark loader.
        
        Args:
            name: Benchmark name
            loader_class: Subclass of BenchmarkDataset
        """
        cls._registry[name.lower()] = loader_class

    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List registered benchmarks."""
        return list(cls._registry.keys())
