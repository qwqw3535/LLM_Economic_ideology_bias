# Anonymous Reproduction Artifact

This repository contains the anonymous reproduction package for a COLM submission.

It includes four reproducibility components:

1. `classification`: ideology-sensitive subset construction code and curated outputs.
2. `ICL matching`: Task 2 matching code and curated audit/results files.
3. `difficulty matching`: difficulty scoring code plus curated matched-analysis outputs.
4. `evaluation rerun`: Task 1 and Task 2 rerun entrypoints for the paper's 20-model configuration.

The original 10,490-row EconCausal source corpus is not redistributed here. This artifact only ships derived datasets and curated outputs needed for this paper.

## Layout

- `code/`: copied evaluation and analysis code used by the artifact.
- `data_derived/`: paper-specific derived data released with this submission.
- `metadata_curated/`: curated intermediate outputs for classification, metadata normalization, ICL matching, and difficulty matching.
- `results_curated/`: compressed model outputs used in the paper.
- `reproduction/`: step-by-step notes for each public interface.
- `scripts/`: thin wrappers that run the copied code with artifact-local defaults.

## Quick Start

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env
```

Export only the provider keys you need. The copied code has no embedded secret fallbacks.

## Included Derived Data

- `data_derived/task1_ideology_subset_1056.jsonl`
- `data_derived/task2_jel_similarity_side_capped_jaccard05_shared2.jsonl.gz`

## Curated Results

Task outputs are stored as `.json.gz` files to stay within GitHub file-size limits.

- `results_curated/task1_ideology_subset_1056/`
- `results_curated/task2_jel_similarity_side_capped_jaccard05_shared2/`

To inspect one file locally:

```bash
gunzip -kc results_curated/task1_ideology_subset_1056/task1_openai_gpt-5.2_results.json.gz | head
```

## Main Entry Points

- Classification rerun: `python scripts/run_classification.py --input /path/to/original_econcausal.jsonl`
- Metadata normalization: `python scripts/run_normalize_metadata.py`
- Ideology subset build: `python scripts/run_build_ideology_subset.py`
- ICL matching generation: `python scripts/run_generate_icl_matching.py --generator full_exact50`
- Task 1 evaluation rerun: `python scripts/run_task1_eval.py`
- Task 2 evaluation rerun: `python scripts/run_task2_eval.py`
- Analysis dataset build: `python scripts/run_build_analysis.py`
- Difficulty scoring: `python scripts/run_difficulty_eval.py`
- Difficulty-matched Task 1 analysis: `python scripts/run_difficulty_matching.py`

## Original EconCausal Source

See [original_econcausal.md](reproduction/original_econcausal.md). The paper source and public benchmark release are external to this artifact.

## Size Policy

This repository is trimmed for anonymous-review constraints.

- Files larger than `8 MB` were either compressed, omitted from the main tree, or moved into the supplementary zip.
- The supplementary archive is provided separately as a single self-contained zip under `50 MB`.
- Some large curated outputs from the working directory were intentionally excluded from the repository copy to satisfy submission constraints.

## Paper Model Set

The public rerun path documents the 20 paper models:

- OpenAI: `gpt-4o-mini`, `gpt-4o`, `gpt-5-nano`, `gpt-5-mini`, `gpt-5.2`
- Gemini: `gemini-2.5-flash`, `gemini-3-flash-preview`
- Grok: `grok-3-mini`, `grok-3`, `grok-4-1-fast-reasoning`
- Claude: `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-6`
- Qwen: `qwen/qwen3-8b`, `qwen/qwen3-14b`, `qwen/qwen3-32b`
- Llama: `meta-llama/llama-3.2-1b-instruct`, `meta-llama/llama-3.2-3b-instruct`, `meta-llama/llama-3.1-8b-instruct`, `meta-llama/llama-3.3-70b-instruct`

Detailed notes live in [evaluation.md](reproduction/evaluation.md).
