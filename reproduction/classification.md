# Classification

## Released Outputs

- `classification/outputs/classified_triplets_merged.jsonl.gz`
- `classification/outputs/ideology_sensitive_subset_current.jsonl`
- `classification/outputs/review_sample_balanced126.jsonl`

## Prompt

- `code/prompts/classification_ideology.py`

## Main Code

- `code/extended/classify_triplets.py`
- `code/extended/ideology_classification_common.py`

## Command

```bash
python scripts/run_classification.py --input /path/to/original_econcausal.jsonl
```
