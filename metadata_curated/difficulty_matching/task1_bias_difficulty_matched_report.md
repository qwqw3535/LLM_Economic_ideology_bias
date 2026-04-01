# Task1 Difficulty-Matched Bias Analysis Report

## 1. Motivation

Liberal/conservative accuracy gaps may reflect inherent question difficulty
rather than model ideological priors. This analysis controls for difficulty
by matching liberal and conservative triplets on LLM-evaluated difficulty levels.

## 2. Methodology

- Each ideology-sensitive triplet receives an overall difficulty score (1-5)
  from GPT-5-mini based on domain knowledge, context dependence, ambiguity,
  causal complexity, and evidence sufficiency.
- For each difficulty level, we take min(n_liberal, n_conservative) triplets
  from each side (random sample, seed=42).
- This ensures identical difficulty distributions for both sides.
- Theme-level matching does the same within each JEL policy theme.

## 3. Data Summary

- Difficulty scores available: 2398 triplets
- Labeled triplets in Task1: 2077
- After difficulty matching (overall): lib=882, cons=882
- After theme-level matching: lib=829, cons=829

## 4. How Theme-Level Matching Worked

Within each `jel_policy_theme`, triplets were first split by difficulty level.
At each theme-difficulty cell, the matched count per side is
`min(n_liberal_before, n_conservative_before)`.
Rows with zero on either side are dropped for that cell.

- ambiguous / difficulty level 2: before lib=13, cons=5, matched per side=5 (matched)
- ambiguous / difficulty level 3: before lib=28, cons=12, matched per side=12 (matched)
- ambiguous / difficulty level 4: before lib=88, cons=53, matched per side=53 (matched)
- ambiguous / difficulty level 5: before lib=30, cons=23, matched per side=23 (matched)
- education_policy / difficulty level 1: before lib=3, cons=1, matched per side=1 (matched)
- education_policy / difficulty level 2: before lib=24, cons=15, matched per side=15 (matched)
- education_policy / difficulty level 3: before lib=14, cons=15, matched per side=14 (matched)
- education_policy / difficulty level 4: before lib=38, cons=24, matched per side=24 (matched)
- education_policy / difficulty level 5: before lib=3, cons=0, matched per side=0 (unmatched)
- environment_climate_energy / difficulty level 3: before lib=7, cons=1, matched per side=1 (matched)
- environment_climate_energy / difficulty level 4: before lib=13, cons=8, matched per side=8 (matched)
- environment_climate_energy / difficulty level 5: before lib=5, cons=8, matched per side=5 (matched)
- financial_regulation / difficulty level 3: before lib=16, cons=1, matched per side=1 (matched)
- financial_regulation / difficulty level 4: before lib=127, cons=89, matched per side=89 (matched)
- financial_regulation / difficulty level 5: before lib=36, cons=25, matched per side=25 (matched)
- health_policy / difficulty level 2: before lib=2, cons=1, matched per side=1 (matched)
- health_policy / difficulty level 3: before lib=11, cons=3, matched per side=3 (matched)
- health_policy / difficulty level 4: before lib=88, cons=42, matched per side=42 (matched)
- health_policy / difficulty level 5: before lib=17, cons=12, matched per side=12 (matched)
- immigration / difficulty level 2: before lib=0, cons=3, matched per side=0 (unmatched)
- immigration / difficulty level 3: before lib=3, cons=1, matched per side=1 (matched)
- immigration / difficulty level 4: before lib=12, cons=4, matched per side=4 (matched)
- immigration / difficulty level 5: before lib=3, cons=1, matched per side=1 (matched)
- industrial_policy_development / difficulty level 2: before lib=3, cons=0, matched per side=0 (unmatched)
- industrial_policy_development / difficulty level 3: before lib=5, cons=2, matched per side=2 (matched)
- industrial_policy_development / difficulty level 4: before lib=15, cons=6, matched per side=6 (matched)
- industrial_policy_development / difficulty level 5: before lib=2, cons=0, matched per side=0 (unmatched)
- labor_wages_unions / difficulty level 2: before lib=3, cons=2, matched per side=2 (matched)
- labor_wages_unions / difficulty level 3: before lib=2, cons=3, matched per side=2 (matched)
- labor_wages_unions / difficulty level 4: before lib=28, cons=24, matched per side=24 (matched)
- labor_wages_unions / difficulty level 5: before lib=4, cons=3, matched per side=3 (matched)
- macro_fiscal_state / difficulty level 3: before lib=3, cons=1, matched per side=1 (matched)
- macro_fiscal_state / difficulty level 4: before lib=13, cons=8, matched per side=8 (matched)
- macro_fiscal_state / difficulty level 5: before lib=5, cons=7, matched per side=5 (matched)
- market_regulation_antitrust / difficulty level 3: before lib=3, cons=0, matched per side=0 (unmatched)
- market_regulation_antitrust / difficulty level 4: before lib=9, cons=5, matched per side=5 (matched)
- market_regulation_antitrust / difficulty level 5: before lib=4, cons=1, matched per side=1 (matched)
- other / difficulty level 1: before lib=2, cons=0, matched per side=0 (unmatched)
- other / difficulty level 2: before lib=20, cons=8, matched per side=8 (matched)
- other / difficulty level 3: before lib=50, cons=22, matched per side=22 (matched)
- other / difficulty level 4: before lib=211, cons=221, matched per side=211 (matched)
- other / difficulty level 5: before lib=81, cons=68, matched per side=68 (matched)
- taxation / difficulty level 2: before lib=2, cons=1, matched per side=1 (matched)
- taxation / difficulty level 3: before lib=7, cons=9, matched per side=7 (matched)
- taxation / difficulty level 4: before lib=29, cons=49, matched per side=29 (matched)
- taxation / difficulty level 5: before lib=4, cons=2, matched per side=2 (matched)
- trade_globalization / difficulty level 3: before lib=2, cons=5, matched per side=2 (matched)
- trade_globalization / difficulty level 4: before lib=30, cons=36, matched per side=30 (matched)
- trade_globalization / difficulty level 5: before lib=12, cons=12, matched per side=12 (matched)
- welfare_redistribution / difficulty level 1: before lib=0, cons=1, matched per side=0 (unmatched)
- welfare_redistribution / difficulty level 2: before lib=4, cons=5, matched per side=4 (matched)
- welfare_redistribution / difficulty level 3: before lib=14, cons=12, matched per side=12 (matched)
- welfare_redistribution / difficulty level 4: before lib=45, cons=20, matched per side=20 (matched)
- welfare_redistribution / difficulty level 5: before lib=2, cons=2, matched per side=2 (matched)

## 5. Output Files

### Difficulty Distribution
- `task1_difficulty_distribution_before_matching.csv`
- `task1_difficulty_distribution_after_matching.csv`

### Overall Difficulty-Matched
- `task1_difficulty_matched_accuracy_by_ground_truth_side.csv`
- `task1_difficulty_matched_accuracy_by_model_and_ground_truth_side.csv`
- `task1_difficulty_matched_bias_by_model.csv`

### Theme-Level Difficulty-Matched
- `task1_difficulty_matched_accuracy_by_theme_and_ground_truth_side.csv`
- `task1_difficulty_matched_accuracy_by_model_theme_and_ground_truth_side.csv`
- `task1_difficulty_matched_accuracy_by_family_theme_and_ground_truth_side.csv`
- `task1_difficulty_matched_bias_by_theme.csv`
- `task1_difficulty_matched_bias_by_model_and_theme.csv`

- `task1_difficulty_theme_matching_retention_by_theme_level_and_side.csv`
- `task1_difficulty_theme_matching_summary.csv`

### Diagnostic (per difficulty level, unmatched)
- `task1_accuracy_by_difficulty_and_ground_truth_side.csv`
- `task1_accuracy_by_model_difficulty_and_ground_truth_side.csv`

### Comparison Baseline
- `task1_original_accuracy_by_model_and_ground_truth_side.csv`
