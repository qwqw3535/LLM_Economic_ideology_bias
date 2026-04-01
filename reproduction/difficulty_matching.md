# Difficulty Matching Reproduction

## Included Curated Files

- `metadata_curated/classification/difficulty_scores_clean.jsonl`
- `metadata_curated/difficulty_matching/task1_bias_difficulty_matched_report.md`
- `metadata_curated/difficulty_matching/task1_difficulty_theme_matching_summary.csv`
- `metadata_curated/difficulty_matching/task1_difficulty_distribution_before_matching.csv`
- `metadata_curated/difficulty_matching/task1_difficulty_distribution_after_matching.csv`
- `metadata_curated/difficulty_matching/task1_difficulty_matched_accuracy_by_ground_truth_side.csv`
- `metadata_curated/difficulty_matching/task1_difficulty_matched_accuracy_by_model_and_ground_truth_side.csv`
- `metadata_curated/difficulty_matching/task1_difficulty_matched_bias_by_model.csv`

## Main Code

- `code/extended/evaluate_difficulty.py`
- `code/extended/ideology_bias/analyze_task1_bias_difficulty_matched.py`

## Commands

Run difficulty scoring:

```bash
python scripts/run_difficulty_eval.py
```

Build analysis datasets after Task 1 reruns:

```bash
python scripts/run_build_analysis.py
```

Run the matched analysis:

```bash
python scripts/run_difficulty_matching.py
```

## Notes

- Difficulty scoring requires `OPENAI_API_KEY`.
- The matched analysis expects Task 1 results and a task-analysis dataset under `outputs/`.
- The shipped curated outputs are the paper results used for the difficulty-controlled comparison.

