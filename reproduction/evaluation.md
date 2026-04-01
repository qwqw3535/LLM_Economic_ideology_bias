# Evaluation Rerun

## Released Derived Inputs

- Task 1: `data_derived/task1_ideology_subset_1056.jsonl`
- Task 2: `data_derived/task2_jel_similarity_side_capped_jaccard05_shared2.jsonl`

## Released Curated Outputs

- `results_curated/task1_ideology_subset_1056/*.json.gz`
- `results_curated/task2_jel_similarity_side_capped_jaccard05_shared2/*.json.gz`

## Paper Model Configuration

- OpenAI: `gpt-4o-mini`, `gpt-4o`, `gpt-5-nano`, `gpt-5-mini`, `gpt-5.2`
- Gemini: `gemini-2.5-flash`, `gemini-3-flash-preview`
- Grok: `grok-3-mini`, `grok-3`, `grok-4-1-fast-reasoning`
- Claude: `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-6`
- Qwen: `qwen/qwen3-8b`, `qwen/qwen3-14b`, `qwen/qwen3-32b`
- Llama: `meta-llama/llama-3.2-1b-instruct`, `meta-llama/llama-3.2-3b-instruct`, `meta-llama/llama-3.1-8b-instruct`, `meta-llama/llama-3.3-70b-instruct`

## Commands

Task 1:

```bash
python scripts/run_task1_eval.py
```

Task 2:

```bash
python scripts/run_task2_eval.py
```

Both wrappers accept extra `run_evaluation.py` flags, for example:

```bash
python scripts/run_task1_eval.py --models openai claude --max-samples 25
python scripts/run_task2_eval.py --models qwen llama --max-samples 25
```

## Output Locations

- Task 1 reruns: `outputs/evaluation/task1/`
- Task 2 reruns: `outputs/evaluation/task2/`

## Notes

- Task 2 reruns use the released JEL-similarity shared2 source file by default.
- Task 3 is intentionally excluded from this artifact.
- Curated release outputs are compressed for repository size compliance; fresh reruns write normal JSON under `outputs/`.

