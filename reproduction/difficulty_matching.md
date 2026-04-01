# Difficulty Matching

## Released Outputs

- `difficulty_matching/outputs/difficulty_scores_clean.jsonl`
- `difficulty_matching/outputs/main_results_bias_difficulty_matched_report.md`
- `difficulty_matching/outputs/main_results_difficulty_theme_matching_summary.csv`
- `difficulty_matching/outputs/main_results_difficulty_distribution_before_matching.csv`
- `difficulty_matching/outputs/main_results_difficulty_distribution_after_matching.csv`
- `difficulty_matching/outputs/main_results_difficulty_matched_accuracy_by_ground_truth_side.csv`
- `difficulty_matching/outputs/main_results_difficulty_matched_accuracy_by_model_and_ground_truth_side.csv`
- `difficulty_matching/outputs/main_results_difficulty_matched_bias_by_model.csv`

## Prompt

- `code/prompts/difficulty_scoring.py`

## Main Code

- `code/extended/evaluate_difficulty.py`
- `code/extended/ideology_bias/analyze_main_results_difficulty_matched.py`

## Commands

```bash
python scripts/run_difficulty_scoring.py
python scripts/run_difficulty_matching.py
```
