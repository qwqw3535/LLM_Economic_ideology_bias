"""Case generators for the released main-results and ICL experiments."""

from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ICL_SOURCE = ARTIFACT_ROOT / "icl_experiment" / "input" / "jel_similarity_shared2.jsonl.gz"


def load_json(path_like: str | Path):
    """Load JSON, JSONL, or JSONL.GZ files."""
    path = Path(path_like)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        if path.suffixes[-2:] == [".jsonl", ".gz"] or path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


@dataclass
class VerificationCase:
    """Compatibility-only verification case, not used in the released artifact."""

    case_id: str
    context: str
    treatment: str
    outcome: str
    sign: str
    expected_answer: str
    paper_id: str
    is_flipped: bool = False


@dataclass
class MainResultsCase:
    """Main-results sign-prediction case."""

    case_id: str
    context: str
    treatment: str
    outcome: str
    expected_sign: str
    paper_id: str


@dataclass
class ICLExperimentCase:
    """ICL experiment case with matched examples."""

    case_id: str
    treatment: str
    outcome: str
    examples: list[dict]
    test_context: str
    expected_sign: str
    paper_ids: list[str]
    avg_similarity: float = 0.0
    sign_differs: bool = False
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
    """Generate released experiment cases from JSONL inputs."""

    def __init__(
        self,
        data_path: str,
        step2_path: str | None = None,
        icl_source_path: str | None = None,
        seed: int = 42,
        **_: object,
    ):
        self.data_path = data_path
        self.data = load_json(data_path)
        self.seed = seed
        self.icl_source_path = icl_source_path or str(DEFAULT_ICL_SOURCE)
        random.seed(seed)

        if isinstance(self.data, dict):
            self.triplets = self.data.get("results", {})
        elif isinstance(self.data, list):
            self.triplets = defaultdict(list)
            for item in self.data:
                paper_id = str(item.get("paper_id", "unknown"))
                self.triplets[paper_id].append(item)
        else:
            self.triplets = {}

    def _get_context(self, triplet: dict) -> Optional[str]:
        if triplet.get("context"):
            return triplet["context"]
        selection = triplet.get("selection", {})
        context_selected = selection.get("context_selected", [])
        if context_selected:
            return context_selected[0]
        return None

    def generate_main_results_cases(self, max_samples: Optional[int] = None) -> list[MainResultsCase]:
        """Generate sign-prediction cases for the main-results experiment."""
        cases: list[MainResultsCase] = []
        case_id = 0

        for paper_id, triplet_list in self.triplets.items():
            for triplet in triplet_list:
                context = self._get_context(triplet)
                treatment = triplet.get("treatment", "")
                outcome = triplet.get("outcome", "")
                sign = triplet.get("sign", "")
                if not context or not treatment or not outcome or not sign:
                    continue
                cases.append(
                    MainResultsCase(
                        case_id=f"main_results_{case_id}",
                        context=context,
                        treatment=treatment,
                        outcome=outcome,
                        expected_sign=sign,
                        paper_id=paper_id,
                    )
                )
                case_id += 1
                if max_samples is not None and len(cases) >= max_samples:
                    return cases

        return cases

    def _load_released_icl_rows(self) -> list[dict]:
        """Load the released ICL source rows."""
        return load_json(self.icl_source_path)

    def generate_icl_experiment_cases(
        self,
        max_samples: Optional[int] = None,
        num_examples: int = 1,
    ) -> list[ICLExperimentCase]:
        """Generate ICL experiment cases from the released matching file."""
        cases: list[ICLExperimentCase] = []
        for row in self._load_released_icl_rows():
            examples = list(row.get("examples") or [])[:num_examples]
            if not examples:
                continue
            cases.append(
                ICLExperimentCase(
                    case_id=row["case_id"],
                    treatment=row["treatment"],
                    outcome=row["outcome"],
                    examples=examples,
                    test_context=row["test_context"],
                    expected_sign=row["expected_sign"],
                    paper_ids=row.get("paper_ids") or [],
                    avg_similarity=float(row.get("avg_similarity") or 0.0),
                    sign_differs=bool(row.get("sign_differs", False)),
                    context=row.get("context"),
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
                    different_paper=bool(row.get("different_paper", False)),
                    triplet_key=row.get("triplet_key"),
                )
            )
            if max_samples is not None and len(cases) >= max_samples:
                return cases
        return cases

    def generate_all_tasks(
        self,
        max_samples_per_task: Optional[int] = None,
        icl_examples: int = 1,
    ) -> dict[str, list]:
        """Generate both released experiments."""
        return {
            "main_results": self.generate_main_results_cases(max_samples_per_task),
            "icl_experiment": self.generate_icl_experiment_cases(max_samples_per_task, num_examples=icl_examples),
        }

    def get_statistics(self) -> dict[str, int]:
        """Return lightweight dataset statistics for logging."""
        return {
            "papers": len(self.triplets),
            "triplets": sum(len(rows) for rows in self.triplets.values()),
            "icl_rows": len(self._load_released_icl_rows()) if Path(self.icl_source_path).exists() else 0,
        }
