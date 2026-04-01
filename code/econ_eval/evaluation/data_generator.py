"""Test case generator for evaluation tasks."""

from __future__ import annotations

import json
import gzip
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TASK23_SOURCE = ARTIFACT_ROOT / "data_derived" / "task2_jel_similarity_side_capped_jaccard05_shared2.jsonl.gz"


def load_json(file_path):
    """Load JSON or JSONL file."""
    path = Path(file_path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        if path.suffixes[-2:] == [".jsonl", ".gz"] or path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


@dataclass
class Task1Case:
    """Test case for Task 1: Causality Verification."""
    case_id: str
    context: str
    treatment: str
    outcome: str
    sign: str
    expected_answer: str  # "Yes" or "No"
    paper_id: str
    is_flipped: bool = False  # Whether this is a negative case with flipped sign


@dataclass
class Task2Case:
    """Test case for Task 2: Sign Prediction."""
    case_id: str
    context: str
    treatment: str
    outcome: str
    expected_sign: str  # "+", "-", "None", "mixed"
    paper_id: str


@dataclass
class Task3Case:
    """Test case for Task 3: Context-Aware Reasoning (similar T/O, different contexts).

    Uses embedding cosine similarity (avg of treatment + outcome >= 0.8)
    to find similar T/O pairs from different papers.
    """
    case_id: str
    treatment: str              # target treatment
    outcome: str                # target outcome
    examples: list[dict]        # List of {treatment, outcome, context, sign}
    test_context: str           # target context
    expected_sign: str          # true sign of target
    paper_ids: list[str]
    avg_similarity: float = 0.0
    sign_differs: bool = False  # True if any example sign != target expected_sign
    context: Optional[str] = None
    sign: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    publication_year: Optional[int] = None
    published_venue: Optional[str] = None
    jel_codes: Optional[str] = None
    paper_url: Optional[str] = None
    variant: Optional[str] = None
    pair_case_type: Optional[str] = None
    matching_rule: Optional[str] = None
    target_side: Optional[str] = None
    example_true_side: Optional[str] = None
    example_false_side: Optional[str] = None
    target_exact_jel_codes: list[str] = field(default_factory=list)
    example_exact_jel_codes: list[str] = field(default_factory=list)
    shared_exact_jel_codes: list[str] = field(default_factory=list)
    shared_exact_jel_count: Optional[int] = None
    target_exact_jel_count: Optional[int] = None
    jel_overlap_ratio: Optional[float] = None
    example_overlap_ratio: Optional[float] = None
    jel_similarity: Optional[float] = None
    union_exact_jel_count: Optional[int] = None
    selection_score: Optional[float] = None
    different_paper: bool = False
    triplet_key: Optional[str] = None


@dataclass
class Task4Case:
    """Test case for Task 4: Noise Robustness (similar T/O, different contexts, reverted signs).

    Same pairing as Task 3, but example signs are intentionally reverted
    to test LLM robustness against misleading information.
    """
    case_id: str
    treatment: str              # target treatment
    outcome: str                # target outcome
    examples: list[dict]        # List of {treatment, outcome, context, sign, original_sign}
    test_context: str           # target context
    expected_sign: str          # true sign of target
    paper_ids: list[str]
    avg_similarity: float = 0.0
    sign_differs: bool = False  # True if any example (reverted) sign != target expected_sign
    context: Optional[str] = None
    sign: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    publication_year: Optional[int] = None
    published_venue: Optional[str] = None
    jel_codes: Optional[str] = None
    paper_url: Optional[str] = None
    variant: Optional[str] = None
    pair_case_type: Optional[str] = None
    matching_rule: Optional[str] = None
    target_side: Optional[str] = None
    example_true_side: Optional[str] = None
    example_false_side: Optional[str] = None
    target_exact_jel_codes: list[str] = field(default_factory=list)
    example_exact_jel_codes: list[str] = field(default_factory=list)
    shared_exact_jel_codes: list[str] = field(default_factory=list)
    shared_exact_jel_count: Optional[int] = None
    target_exact_jel_count: Optional[int] = None
    jel_overlap_ratio: Optional[float] = None
    example_overlap_ratio: Optional[float] = None
    jel_similarity: Optional[float] = None
    union_exact_jel_count: Optional[int] = None
    selection_score: Optional[float] = None
    different_paper: bool = False
    triplet_key: Optional[str] = None


class TestCaseGenerator:
    """Generate test cases from evaluation JSON or JSONL."""

    def __init__(
        self,
        data_path: str,
        step2_path: str = None,  # Deprecated, kept for backward compatibility
        task2_source_path: str | None = None,
        task3_source_path: str | None = None,
        seed: int = 42,
    ):
        """
        Initialize test case generator.

        Args:
            data_path: Path to evaluation JSON or JSONL
            step2_path: Deprecated, ignored
            seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.data = load_json(data_path)
        self.seed = seed
        self.task2_source_path = task2_source_path or str(DEFAULT_TASK23_SOURCE)
        self.task3_source_path = task3_source_path or str(DEFAULT_TASK23_SOURCE)
        random.seed(seed)

        # Extract results - supports both nested JSON and flat JSONL/list format
        if isinstance(self.data, dict):
            self.triplets = self.data.get("results", {})
        elif isinstance(self.data, list):
            # Convert flat list (JSONL style) to nested paper_id grouping
            self.triplets = defaultdict(list)
            for item in self.data:
                p_id = str(item.get("paper_id", "unknown"))
                self.triplets[p_id].append(item)
        else:
            self.triplets = {}

    def _get_context(self, paper_id: str, triplet: dict) -> Optional[str]:
        """Get context for a triplet."""
        # New format: context directly in triplet
        if "context" in triplet and triplet["context"]:
            return triplet["context"]

        # Old format: context in selection.context_selected
        selection = triplet.get("selection", {})
        context_selected = selection.get("context_selected", [])
        if context_selected and len(context_selected) > 0:
            return context_selected[0]

        return None

    def _flip_sign(self, sign: str) -> Optional[str]:
        """Flip a sign for negative test cases."""
        flip_map = {
            "+": "-",
            "-": "+",
            "None": "+",  # None becomes positive
            "mixed": None,  # Can't flip mixed
        }
        return flip_map.get(sign)

    def _get_sign_description(self, sign: str) -> str:
        """Get human-readable description of sign."""
        descriptions = {
            "+": "increase",
            "-": "decrease",
            "None": "have no significant effect on",
            "mixed": "have a mixed effect on",
        }
        return descriptions.get(sign, "affect")

    def generate_task1_cases(self, max_samples: Optional[int] = None) -> list[Task1Case]:
        """
        Generate verification task cases.

        For each triplet:
        - Create a positive case (expected: Yes)
        - Create a negative case with flipped sign (expected: No)

        Args:
            max_samples: Maximum number of cases to generate

        Returns:
            List of Task1Case objects
        """
        cases = []
        case_id = 0

        for paper_id, triplet_list in self.triplets.items():
            for triplet in triplet_list:
                context = self._get_context(paper_id, triplet)
                if not context:
                    continue

                treatment = triplet.get("treatment", "")
                outcome = triplet.get("outcome", "")
                sign = triplet.get("sign", "")

                if not treatment or not outcome or not sign:
                    continue

                # Positive case (true claim)
                cases.append(Task1Case(
                    case_id=f"t1_{case_id}",
                    context=context,
                    treatment=treatment,
                    outcome=outcome,
                    sign=sign,
                    expected_answer="Yes",
                    paper_id=paper_id,
                    is_flipped=False,
                ))
                case_id += 1

                # Negative case (flipped sign)
                flipped_sign = self._flip_sign(sign)
                if flipped_sign:
                    cases.append(Task1Case(
                        case_id=f"t1_{case_id}",
                        context=context,
                        treatment=treatment,
                        outcome=outcome,
                        sign=flipped_sign,
                        expected_answer="No",
                        paper_id=paper_id,
                        is_flipped=True,
                    ))
                    case_id += 1

        # Shuffle and limit
        random.shuffle(cases)
        if max_samples:
            cases = cases[:max_samples]

        return cases

    def generate_task2_cases(self, max_samples: Optional[int] = None) -> list[Task2Case]:
        """
        Generate sign prediction task cases.

        Args:
            max_samples: Maximum number of cases to generate

        Returns:
            List of Task2Case objects
        """
        cases = []
        case_id = 0

        for paper_id, triplet_list in self.triplets.items():
            for triplet in triplet_list:
                context = self._get_context(paper_id, triplet)
                if not context:
                    continue

                treatment = triplet.get("treatment", "")
                outcome = triplet.get("outcome", "")
                sign = triplet.get("sign", "")

                if not treatment or not outcome or not sign:
                    continue

                existing_case_id = triplet.get("case_id")
                cases.append(Task2Case(
                    case_id=existing_case_id or f"t1_{case_id}",
                    context=context,
                    treatment=treatment,
                    outcome=outcome,
                    expected_sign=sign,
                    paper_id=paper_id,
                ))
                case_id += 1

        # Shuffle and limit
        random.shuffle(cases)
        if max_samples:
            cases = cases[:max_samples]

        return cases

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()

    def _load_generated_task_rows(self, task_path: str) -> list[dict]:
        """Load generated JSONL rows for task2/task3."""
        path = Path(task_path)
        if not path.exists():
            raise FileNotFoundError(f"Generated task source not found: {path}")
        rows = load_json(str(path))
        if not isinstance(rows, list):
            raise ValueError(f"Expected JSONL list from {path}, got {type(rows).__name__}")
        return rows

    def _parse_examples(self, row: dict) -> list[dict]:
        """Parse example rows from a generated task record."""
        examples = row.get("examples") or row.get("example_details") or []
        parsed_examples = []
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            parsed_examples.append(
                {
                    "paper_id": ex.get("paper_id"),
                    "title": ex.get("title"),
                    "author": ex.get("author"),
                    "publication_year": ex.get("publication_year"),
                    "published_venue": ex.get("published_venue"),
                    "jel_codes": ex.get("jel_codes"),
                    "paper_url": ex.get("paper_url"),
                    "treatment": ex.get("treatment"),
                    "outcome": ex.get("outcome"),
                    "context": ex.get("context"),
                    "sign": ex.get("sign"),
                    "original_sign": ex.get("original_sign", ex.get("sign")),
                    "example_true_side": ex.get("example_true_side"),
                    "example_false_side": ex.get("example_false_side"),
                    "example_exact_jel_codes": ex.get("example_exact_jel_codes") or [],
                    "shared_exact_jel_codes": ex.get("shared_exact_jel_codes") or [],
                    "shared_exact_jel_count": ex.get("shared_exact_jel_count"),
                    "jel_overlap_ratio": ex.get("jel_overlap_ratio"),
                    "example_overlap_ratio": ex.get("example_overlap_ratio"),
                    "jel_similarity": ex.get("jel_similarity"),
                    "union_exact_jel_count": ex.get("union_exact_jel_count"),
                    "selection_score": ex.get("selection_score"),
                    "triplet_key": ex.get("triplet_key"),
                }
            )
        return parsed_examples

    def generate_task3_cases(
        self,
        max_samples: Optional[int] = None,
        num_examples: int = 1,
    ) -> list[Task3Case]:
        """
        Generate context-aware reasoning cases.

        Uses pre-computed similarity data (task2_input.json or legacy task3_input.json)
        to find similar
        T/O pairs (avg cosine similarity >= 0.8) from different papers.
        Examples have correct signs.

        Args:
            max_samples: Maximum number of cases to generate
            num_examples: Number of examples to include (default: 1)

        Returns:
            List of Task3Case objects
        """
        cases = []
        for row in self._load_generated_task_rows(self.task2_source_path):
            examples = self._parse_examples(row)
            if len(examples) < num_examples:
                continue
            selected = examples[:num_examples]
            target_sign = row.get("expected_sign") or row.get("sign", "")
            sign_differs = any(ex["sign"] != target_sign for ex in selected)
            cases.append(Task3Case(
                case_id=row.get("case_id", ""),
                treatment=row.get("treatment", ""),
                outcome=row.get("outcome", ""),
                examples=selected,
                test_context=row.get("test_context") or row.get("context", ""),
                expected_sign=target_sign,
                paper_ids=row.get("paper_ids") or [ex.get("paper_id") for ex in selected] + [row.get("paper_id")],
                avg_similarity=float(row.get("avg_similarity") or row.get("jel_overlap_ratio") or 0.0),
                sign_differs=sign_differs,
                context=row.get("context") or row.get("test_context"),
                sign=row.get("sign"),
                title=row.get("title"),
                author=row.get("author"),
                publication_year=row.get("publication_year"),
                published_venue=row.get("published_venue"),
                jel_codes=row.get("jel_codes"),
                paper_url=row.get("paper_url"),
                variant=row.get("variant"),
                pair_case_type=row.get("pair_case_type"),
                matching_rule=row.get("matching_rule"),
                target_side=row.get("target_side"),
                example_true_side=row.get("example_true_side"),
                example_false_side=row.get("example_false_side"),
                target_exact_jel_codes=row.get("target_exact_jel_codes") or [],
                example_exact_jel_codes=row.get("example_exact_jel_codes") or [],
                shared_exact_jel_codes=row.get("shared_exact_jel_codes") or [],
                shared_exact_jel_count=row.get("shared_exact_jel_count"),
                target_exact_jel_count=row.get("target_exact_jel_count"),
                jel_overlap_ratio=row.get("jel_overlap_ratio"),
                example_overlap_ratio=row.get("example_overlap_ratio"),
                jel_similarity=row.get("jel_similarity"),
                union_exact_jel_count=row.get("union_exact_jel_count"),
                selection_score=row.get("selection_score"),
                different_paper=bool(row.get("different_paper")),
                triplet_key=row.get("triplet_key"),
            ))

        # Shuffle and limit
        random.shuffle(cases)
        if max_samples:
            cases = cases[:max_samples]

        return cases

    def generate_task4_cases(
        self,
        max_samples: Optional[int] = None,
        num_examples: int = 1,
    ) -> list[Task4Case]:
        """
        Generate noise robustness cases.

        Uses pre-computed similarity data (task3_input.json or legacy task4_input.json)
        to find similar
        T/O pairs (avg cosine similarity >= 0.8) from different papers.
        Example signs are intentionally REVERTED to test robustness.

        Args:
            max_samples: Maximum number of cases to generate
            num_examples: Number of examples to include (default: 1)

        Returns:
            List of Task4Case objects
        """
        cases = []
        for row in self._load_generated_task_rows(self.task3_source_path):
            examples = self._parse_examples(row)
            if len(examples) < num_examples:
                continue
            selected = examples[:num_examples]
            target_sign = row.get("expected_sign") or row.get("sign", "")
            sign_differs = any(ex["sign"] != target_sign for ex in selected)
            cases.append(Task4Case(
                case_id=row.get("case_id", ""),
                treatment=row.get("treatment", ""),
                outcome=row.get("outcome", ""),
                examples=selected,
                test_context=row.get("test_context") or row.get("context", ""),
                expected_sign=target_sign,
                paper_ids=row.get("paper_ids") or [ex.get("paper_id") for ex in selected] + [row.get("paper_id")],
                avg_similarity=float(row.get("avg_similarity") or row.get("jel_overlap_ratio") or 0.0),
                sign_differs=sign_differs,
                context=row.get("context") or row.get("test_context"),
                sign=row.get("sign"),
                title=row.get("title"),
                author=row.get("author"),
                publication_year=row.get("publication_year"),
                published_venue=row.get("published_venue"),
                jel_codes=row.get("jel_codes"),
                paper_url=row.get("paper_url"),
                variant=row.get("variant"),
                pair_case_type=row.get("pair_case_type"),
                matching_rule=row.get("matching_rule"),
                target_side=row.get("target_side"),
                example_true_side=row.get("example_true_side"),
                example_false_side=row.get("example_false_side"),
                target_exact_jel_codes=row.get("target_exact_jel_codes") or [],
                example_exact_jel_codes=row.get("example_exact_jel_codes") or [],
                shared_exact_jel_codes=row.get("shared_exact_jel_codes") or [],
                shared_exact_jel_count=row.get("shared_exact_jel_count"),
                target_exact_jel_count=row.get("target_exact_jel_count"),
                jel_overlap_ratio=row.get("jel_overlap_ratio"),
                example_overlap_ratio=row.get("example_overlap_ratio"),
                jel_similarity=row.get("jel_similarity"),
                union_exact_jel_count=row.get("union_exact_jel_count"),
                selection_score=row.get("selection_score"),
                different_paper=bool(row.get("different_paper")),
                triplet_key=row.get("triplet_key"),
            ))

        # Shuffle and limit
        random.shuffle(cases)
        if max_samples:
            cases = cases[:max_samples]

        return cases

    def generate_all_cases(
        self,
        max_samples_per_task: Optional[int] = None,
        task2_num_examples: int = 1,
        task3_num_examples: int = 1,
    ) -> dict[str, list]:
        """
        Generate test cases for all tasks.

        Args:
            max_samples_per_task: Maximum samples per task
            task2_num_examples: Number of examples for task 2 (default: 1)
            task3_num_examples: Number of examples for task 3 (default: 1)

        Returns:
            Dictionary mapping task name to list of cases
        """
        return {
            "task1": self.generate_task2_cases(max_samples_per_task),
            "task2": self.generate_task3_cases(max_samples_per_task, num_examples=task2_num_examples),
            "task3": self.generate_task4_cases(max_samples_per_task, num_examples=task3_num_examples),
        }

    def get_statistics(self) -> dict:
        """Get statistics about available data."""
        total_triplets = sum(len(v) for v in self.triplets.values())
        total_papers = len(self.triplets)

        # Count triplets with context
        triplets_with_context = 0
        for paper_id, triplet_list in self.triplets.items():
            for triplet in triplet_list:
                if self._get_context(paper_id, triplet):
                    triplets_with_context += 1

        # Sign distribution
        sign_counts = defaultdict(int)
        for triplet_list in self.triplets.values():
            for triplet in triplet_list:
                sign = triplet.get("sign", "unknown")
                sign_counts[sign] += 1

        return {
            "total_papers": total_papers,
            "total_triplets": total_triplets,
            "triplets_with_context": triplets_with_context,
            "sign_distribution": dict(sign_counts),
        }
