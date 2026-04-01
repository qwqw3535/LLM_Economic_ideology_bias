"""Analyze JEL-similarity side_capped shared2 task3 results."""

from __future__ import annotations

from .exact50_analysis_common import print_summary, run_task3_analysis
from .paths import analysis_dataset_paths


def main() -> None:
    _, jsonl_path = analysis_dataset_paths("task3", "jel_similarity_side_capped_jaccard05_shared2")
    print_summary(run_task3_analysis(jsonl_path, "task3_jel_similarity_side_capped_jaccard05_shared2"))


if __name__ == "__main__":
    main()
