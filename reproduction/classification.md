# Classification Reproduction

## Included Curated Files

- `metadata_curated/classification/causal_triplets_gpt-5-mini_classified_merged.jsonl`
- `metadata_curated/classification/causal_triplets_multillm_ideology_qwen32b.jsonl`
- `metadata_curated/classification/ideology_triplet_subset_current.jsonl`
- `metadata_curated/classification/ideology_triplet_review_sample_balanced126.jsonl`

## Main Code

- `code/extended/classify_triplets.py`
- `code/extended/ideology_classification_common.py`
- `code/extended/ideology_bias/normalize_metadata.py`
- `code/extended/ideology_bias/build_ideology_triplet_subsets.py`

## Typical Workflow

1. Download the original EconCausal source file externally.
2. Run classification over that source file.
3. Normalize metadata from the classification output.
4. Build the ideology-sensitive subset.

## Commands

```bash
python scripts/run_classification.py --input /path/to/original_econcausal.jsonl
python scripts/run_normalize_metadata.py
python scripts/run_build_ideology_subset.py
```

## Notes

- `run_build_ideology_subset.py` expects a Task 1 analysis CSV when rebuilding the subset from scratch.
- The curated subset result shipped in this artifact is `metadata_curated/classification/ideology_triplet_subset_current.jsonl`.
- The ideology review sample shipped in this artifact is `metadata_curated/classification/ideology_triplet_review_sample_balanced126.jsonl`.

