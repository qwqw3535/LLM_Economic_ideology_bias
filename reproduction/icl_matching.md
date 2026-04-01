# ICL Matching Reproduction

## Included Curated Files

- `data_derived/task2_jel_similarity_side_capped_jaccard05_shared2.jsonl`
- `metadata_curated/matching/task23_jel_similarity_side_capped_match_catalog.csv`
- `metadata_curated/matching/task2_analysis_rows_jel_similarity_side_capped_jaccard05_shared2.csv.gz`
- `metadata_curated/matching/task2_analysis_rows_jel_similarity_side_capped_jaccard05_shared2.jsonl.gz`
- `metadata_curated/matching/task2_jel_similarity_side_capped_jaccard05_shared2_report.md`

## Included Matching Code

- `code/econ_eval/evaluation/generate_task23_exact50.py`
- `code/extended/ideology_bias/generate_task23_ideology_subset_shared2.py`
- `code/extended/ideology_bias/generate_task23_ideology_subset_variants.py`

## Matching Logic Used in This Paper

The released Task 2 file follows the JEL-similarity shared2 setup documented in the paper:

- target and example must come from different papers
- matching uses JEL overlap metadata
- shared-code and Jaccard-style constraints are enforced by the generator
- side-capped selection is used for the official released Task 2 variant

## Commands

Full-corpus generator family:

```bash
python scripts/run_generate_icl_matching.py --generator full_exact50
```

Ideology-subset shared2 generator:

```bash
python scripts/run_generate_icl_matching.py --generator ideology_subset_shared2
python scripts/run_generate_icl_matching.py --generator ideology_subset_variants
```

## Notes

- The official Task 2 rerun path in this artifact uses the already released `data_derived/task2_jel_similarity_side_capped_jaccard05_shared2.jsonl`.
- The matching catalog and unmatched-audit files are included to make the released pairing process inspectable without rerunning all generation steps.

