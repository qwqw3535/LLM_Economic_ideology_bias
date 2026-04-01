"""Analyze exact50 side_capped task2 results."""

from __future__ import annotations

from .exact50_analysis_common import print_summary, run_task2_analysis
from .paths import analysis_dataset_paths


def main() -> None:
    _, jsonl_path = analysis_dataset_paths("task2", "exact50_side_capped")
    print_summary(run_task2_analysis(jsonl_path, "task2_exact50_side_capped"))


if __name__ == "__main__":
    main()
