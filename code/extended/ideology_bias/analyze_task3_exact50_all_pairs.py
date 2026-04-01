"""Analyze exact50 all_pairs task3 results."""

from __future__ import annotations

from .exact50_analysis_common import print_summary, run_task3_analysis
from .paths import analysis_dataset_paths


def main() -> None:
    _, jsonl_path = analysis_dataset_paths("task3", "exact50_all_pairs")
    print_summary(run_task3_analysis(jsonl_path, "task3_exact50_all_pairs"))


if __name__ == "__main__":
    main()
