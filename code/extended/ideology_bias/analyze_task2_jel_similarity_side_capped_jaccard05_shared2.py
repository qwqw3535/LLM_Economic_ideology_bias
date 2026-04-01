"""Analyze JEL-similarity side_capped shared2 task2 results."""

from __future__ import annotations

from .exact50_analysis_common import print_summary, run_task2_analysis
from .paths import analysis_dataset_paths


def main() -> None:
    _, jsonl_path = analysis_dataset_paths("task2", "jel_similarity_side_capped_jaccard05_shared2")
    print_summary(run_task2_analysis(jsonl_path, "task2_jel_similarity_side_capped_jaccard05_shared2"))


if __name__ == "__main__":
    main()
