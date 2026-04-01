# Anonymous Reproduction Artifact

This repository contains the anonymous reproduction package for a COLM submission.

The released contents are organized around the paper's four public components:

1. `classification`
2. `main_results`
3. `icl_experiment`
4. `difficulty_matching`

The original 10,490-row EconCausal source corpus is not redistributed here. This artifact only ships derived inputs, prompts, code, and curated outputs used for the paper.

## Layout

- `classification/`: released classification outputs and subset artifacts.
- `main_results/`: released input subset and curated model outputs for the paper's main result.
- `icl_experiment/`: released ICL input file, matching audit files, and curated model outputs.
- `difficulty_matching/`: released difficulty scores and matched-analysis outputs.
- `code/`: minimal code kept for the released artifact.
- `code/prompts/`: prompt files split by paper component.
- `reproduction/`: short reproduction notes for each released component.
- `scripts/`: thin public wrappers.

## Quick Start

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env
```

Export only the provider keys you need. The copied code has no embedded secret fallbacks.

## Main Entry Points

- Classification: `python scripts/run_classification.py --input /path/to/original_econcausal.jsonl`
- Main Results: `python scripts/run_main_results.py`
- ICL Experiment: `python scripts/run_icl_experiment.py`
- ICL example generation: `python scripts/run_generate_icl_experiment.py`
- Difficulty scoring: `python scripts/run_difficulty_scoring.py`
- Difficulty matching: `python scripts/run_difficulty_matching.py`

## Prompt Files

- `code/prompts/classification_ideology.py`
- `code/prompts/main_results.py`
- `code/prompts/icl_experiment.py`
- `code/prompts/difficulty_scoring.py`

## Original EconCausal Source

See [original_econcausal.md](/home/donggyu/econ_causality/anonymous_artifact/reproduction/original_econcausal.md). The source benchmark remains external to this artifact.

## Size Policy

- Files larger than `8 MB` were compressed, omitted, or moved out of the main tree.
- The supplementary archive is provided separately as a single self-contained zip under `50 MB`.
- Some large working files that were not needed for the paper were intentionally excluded.
