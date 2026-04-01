# 📘 Anonymous COLM Submission

This repository is the anonymous companion repository for a COLM submission on ideological bias in LLM-based economic causal reasoning.

It is designed to work both as:

- a paper-introduction repository for reviewers
- a lightweight reproduction package for the main experiments

## ✨ What Is Included

This release is organized around four paper-facing components:

1. `classification`
2. `main_results`
3. `icl_experiment`
4. `difficulty_matching`

The original 10,490-row EconCausal source corpus is intentionally not redistributed here. This repository contains only the derived inputs, prompts, code, and curated outputs needed for the paper.

## 🗂 Repository Map

- `classification/`
  Released classification outputs and subset artifacts.
- `main_results/`
  Released ideology-sensitive subset and curated outputs for the paper's main result.
- `icl_experiment/`
  Released ICL input file, matching files, and curated outputs.
- `difficulty_matching/`
  Released difficulty scores and matched-analysis outputs.
- `code/`
  Minimal code kept for the released artifact.
- `code/prompts/`
  Prompt files split by paper component.
- `reproduction/`
  Short reproduction notes for each released component.
- `scripts/`
  Thin public wrappers for reruns.

## 🚀 Quick Start

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env
```

Export only the provider keys you need. No API keys are embedded in this release.

## 🧪 Main Entry Points

- Classification
  `python scripts/run_classification.py --input /path/to/original_econcausal.jsonl`
- Main Results
  `python scripts/run_main_results.py`
- ICL Experiment
  `python scripts/run_icl_experiment.py`
- ICL example generation
  `python scripts/run_generate_icl_experiment.py`
- Difficulty scoring
  `python scripts/run_difficulty_scoring.py`
- Difficulty matching
  `python scripts/run_difficulty_matching.py`

## 🧠 Prompt Files

- `code/prompts/classification_ideology.py`
- `code/prompts/main_results.py`
- `code/prompts/icl_experiment.py`
- `code/prompts/difficulty_scoring.py`

## 🔗 Original EconCausal Source

The original benchmark stays external to this repository.
See [original_econcausal.md](reproduction/original_econcausal.md) for the source-paper link and acquisition note.

## 📦 Size Notes

- Files larger than `8 MB` were compressed, omitted, or moved out of the main tree.
- A separate supplementary archive is provided as a single self-contained zip under `50 MB`.
- Large intermediate working files not used in the paper were intentionally removed.

## 📄 Reproduction Notes

- [classification.md](reproduction/classification.md)
- [main_results.md](reproduction/main_results.md)
- [icl_experiment.md](reproduction/icl_experiment.md)
- [difficulty_matching.md](reproduction/difficulty_matching.md)

## 🕶 Anonymous Review Context

This repository is prepared for anonymous review. It avoids redistributing the original full benchmark and keeps only the paper-relevant derived artifacts needed to inspect or rerun the released experiments.
