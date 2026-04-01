"""Run the ideology-bias pipeline end-to-end with selectable steps."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from extended.ideology_bias.paths import (
        DEFAULT_BOOTSTRAP_METADATA_JSONL,
        DEFAULT_MERGED_CLASSIFICATION_PATH,
        DEFAULT_FULL_METADATA_CANONICAL_CSV,
        DEFAULT_FULL_METADATA_CANONICAL_JSONL,
        FULL_CORPUS_PATH,
        OUTPUT_DIR,
        TASK23_EXACT50_ALL_PAIRS_RESULT_DIR,
        TASK23_EXACT50_SIDE_CAPPED_RESULT_DIR,
        TASK23_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_RESULT_DIR,
        TASK2_EXACT50_ALL_PAIRS_PATH,
        TASK2_EXACT50_SIDE_CAPPED_PATH,
        TASK2_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_PATH,
        TASK3_EXACT50_ALL_PAIRS_PATH,
        TASK3_EXACT50_SIDE_CAPPED_PATH,
        TASK3_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_PATH,
        legacy_classification_output_path,
    )
else:
    from .paths import (
        DEFAULT_BOOTSTRAP_METADATA_JSONL,
        DEFAULT_MERGED_CLASSIFICATION_PATH,
        DEFAULT_FULL_METADATA_CANONICAL_CSV,
        DEFAULT_FULL_METADATA_CANONICAL_JSONL,
        FULL_CORPUS_PATH,
        OUTPUT_DIR,
        TASK23_EXACT50_ALL_PAIRS_RESULT_DIR,
        TASK23_EXACT50_SIDE_CAPPED_RESULT_DIR,
        TASK23_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_RESULT_DIR,
        TASK2_EXACT50_ALL_PAIRS_PATH,
        TASK2_EXACT50_SIDE_CAPPED_PATH,
        TASK2_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_PATH,
        TASK3_EXACT50_ALL_PAIRS_PATH,
        TASK3_EXACT50_SIDE_CAPPED_PATH,
        TASK3_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_PATH,
        legacy_classification_output_path,
    )


STEP_CHOICES = (
    "classify",
    "normalize",
    "build",
    "task1",
    "task1_difficulty_matched",
    "task2",
    "task3",
    "generate_task23_exact50",
    "task2_exact50_side_capped",
    "task3_exact50_side_capped",
    "task2_exact50_all_pairs",
    "task3_exact50_all_pairs",
    "task2_jel_similarity_side_capped_jaccard05_shared2",
    "task3_jel_similarity_side_capped_jaccard05_shared2",
    "annotate_reasoning",
    "analyze_reasoning",
)


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, env=env)


def _publish_outputs(bundle_name: str, patterns: list[str], base_output_dir: Path | None = None) -> None:
    base_output_dir = base_output_dir or OUTPUT_DIR
    bundle_root = base_output_dir / bundle_name
    target_dirs = {
        "analysis_datasets": bundle_root / "analysis_datasets",
        "tables": bundle_root / "tables",
        "figures": bundle_root / "figures",
        "reports": bundle_root / "reports",
        "audits": bundle_root / "audits",
    }
    for path in target_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    source_dirs = {
        "analysis_datasets": base_output_dir / "analysis_datasets",
        "tables": base_output_dir / "tables",
        "figures": base_output_dir / "figures",
        "reports": base_output_dir / "reports",
        "audits": base_output_dir / "audits",
    }
    for label, source_dir in source_dirs.items():
        for pattern in patterns:
            for source_path in source_dir.glob(pattern):
                shutil.copy2(source_path, target_dirs[label] / source_path.name)


def _default_steps(metadata_source: str, with_reasoning: bool) -> list[str]:
    steps = ["normalize", "build", "task1", "task2", "task3"]
    if metadata_source == "full":
        steps.insert(0, "classify")
    if with_reasoning:
        steps.extend(["annotate_reasoning", "analyze_reasoning"])
    return steps


def _resolved_metadata_source(raw_source: str) -> str:
    if raw_source != "auto":
        return raw_source
    if DEFAULT_FULL_METADATA_CANONICAL_JSONL.exists() or DEFAULT_MERGED_CLASSIFICATION_PATH.exists():
        return "full"
    return "bootstrap"


def _normalize_input_path(classification_input: Path, classification_output: Path, steps: list[str]) -> Path:
    if "classify" in steps:
        return classification_output
    if DEFAULT_MERGED_CLASSIFICATION_PATH.exists():
        return DEFAULT_MERGED_CLASSIFICATION_PATH
    if classification_output.exists():
        return classification_output
    return classification_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ideology-bias pipeline with selectable steps.")
    parser.add_argument("--steps", nargs="+", choices=STEP_CHOICES, default=None)
    parser.add_argument("--metadata-source", choices=["bootstrap", "full", "auto"], default="auto")
    parser.add_argument("--task1-theme-source", choices=["vote", "legacy"], default="vote")
    parser.add_argument("--with-reasoning", action="store_true", help="Also run reasoning-frame annotation and analysis.")
    parser.add_argument("--metadata-input", default=None, help="Optional metadata JSONL path for the build step.")
    parser.add_argument("--task1-results-dir", default=None, help="Optional Task1 evaluation results directory override.")
    parser.add_argument("--task23-results-dir", default=None, help="Optional Task2/3 evaluation results directory override.")
    parser.add_argument("--task2-source", default=None, help="Optional Task2 source JSONL override.")
    parser.add_argument("--task3-source", default=None, help="Optional Task3 source JSONL override.")
    parser.add_argument("--dataset-suffix", default="", help="Optional suffix for analysis dataset filenames.")
    parser.add_argument("--output-dir", default=None, help="Optional ideology-bias output directory override.")

    parser.add_argument("--classification-input", default=str(FULL_CORPUS_PATH))
    parser.add_argument("--classification-output", default=None)
    parser.add_argument("--classification-model", default="gpt-5-mini")
    parser.add_argument("--classify-max-parallel", type=int, default=500)
    parser.add_argument("--classify-max-retries", type=int, default=3)
    parser.add_argument("--classify-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--classify-limit", type=int, default=None)

    parser.add_argument("--reasoning-method", choices=["llm", "heuristic"], default="heuristic")
    parser.add_argument("--reasoning-task", choices=["task1", "task2", "task3", "all"], default="task1")
    parser.add_argument("--reasoning-limit", type=int, default=None)
    args = parser.parse_args()

    metadata_source = _resolved_metadata_source(args.metadata_source)
    steps = list(args.steps) if args.steps is not None else _default_steps(args.metadata_source, args.with_reasoning)
    if args.with_reasoning:
        for step in ("annotate_reasoning", "analyze_reasoning"):
            if step not in steps:
                steps.append(step)

    classification_input = Path(args.classification_input)
    classification_output = Path(args.classification_output) if args.classification_output else legacy_classification_output_path(
        classification_input,
        args.classification_model,
    )
    output_dir_override = Path(args.output_dir).resolve() if args.output_dir else None
    child_env = os.environ.copy()
    if output_dir_override is not None:
        child_env["IDEOLOGY_BIAS_OUTPUT_DIR"] = str(output_dir_override)

    if "classify" in steps:
        classify_cmd = [
            sys.executable,
            "extended/classify_triplets.py",
            "--input",
            str(classification_input),
            "--output",
            str(classification_output),
            "--model",
            args.classification_model,
            "--max_parallel",
            str(args.classify_max_parallel),
            "--max_retries",
            str(args.classify_max_retries),
            "--api-key-env",
            args.classify_api_key_env,
        ]
        if args.classify_limit is not None:
            classify_cmd.extend(["--limit", str(args.classify_limit)])
        _run(classify_cmd, env=child_env)

    if "normalize" in steps:
        normalize_cmd = [sys.executable, "-m", "extended.ideology_bias.normalize_metadata"]
        if metadata_source == "bootstrap":
            normalize_cmd.extend(["--source", "legacy"])
        else:
            normalize_input = _normalize_input_path(classification_input, classification_output, steps)
            if not normalize_input.exists():
                raise FileNotFoundError(
                    "No full-corpus classification input found for normalize step. "
                    f"Checked {normalize_input} and {DEFAULT_MERGED_CLASSIFICATION_PATH}. "
                    "Pass --metadata-source bootstrap, run --steps classify first, or provide an existing classified file."
                )
            normalize_cmd.extend(
                [
                    "--input",
                    str(normalize_input),
                    "--source",
                    "legacy",
                    "--output-jsonl",
                    str(DEFAULT_FULL_METADATA_CANONICAL_JSONL),
                    "--output-csv",
                    str(DEFAULT_FULL_METADATA_CANONICAL_CSV),
                ]
            )
        _run(normalize_cmd, env=child_env)

    if "build" in steps:
        build_cmd = [sys.executable, "-m", "extended.ideology_bias.build_analysis_datasets"]
        if args.metadata_input:
            build_cmd.extend(["--metadata-jsonl", args.metadata_input])
        elif metadata_source == "bootstrap":
            build_cmd.extend(["--metadata-jsonl", str(DEFAULT_BOOTSTRAP_METADATA_JSONL)])
        else:
            build_cmd.extend(["--metadata-jsonl", str(DEFAULT_FULL_METADATA_CANONICAL_JSONL)])
        if args.task1_results_dir:
            build_cmd.extend(["--task1-results-dir", args.task1_results_dir])
        if args.task2_source:
            build_cmd.extend(["--task2-source", args.task2_source])
        if args.task3_source:
            build_cmd.extend(["--task3-source", args.task3_source])
        if args.task23_results_dir:
            build_cmd.extend(["--task23-results-dir", args.task23_results_dir])
        if args.dataset_suffix:
            build_cmd.extend(["--dataset-suffix", args.dataset_suffix])
        _run(build_cmd, env=child_env)

    if "task1" in steps:
        task1_module = (
            "extended.ideology_bias.analyze_task1_bias_vote_theme"
            if args.task1_theme_source == "vote"
            else "extended.ideology_bias.analyze_task1_bias"
        )
        _run([sys.executable, "-m", task1_module], env=child_env)
        _run([sys.executable, "-m", "extended.ideology_bias.plot_task1_bias", "--theme-source", args.task1_theme_source], env=child_env)
    if "task1_difficulty_matched" in steps:
        _run([sys.executable, "-m", "extended.ideology_bias.analyze_task1_bias_difficulty_matched"], env=child_env)
    if "task2" in steps:
        _run([sys.executable, "-m", "extended.ideology_bias.analyze_task2_example_sensitivity"], env=child_env)
    if "task3" in steps:
        _run([sys.executable, "-m", "extended.ideology_bias.analyze_task3_noise_robustness"], env=child_env)
    if "generate_task23_exact50" in steps:
        _run([sys.executable, "-m", "econ_eval.evaluation.generate_task23_exact50"], env=child_env)
    if "task2_exact50_side_capped" in steps:
        _run(
            [
                sys.executable,
                "-m",
                "extended.ideology_bias.build_analysis_datasets",
                    "--metadata-jsonl",
                    args.metadata_input or (
                        str(DEFAULT_BOOTSTRAP_METADATA_JSONL)
                    if metadata_source == "bootstrap"
                    else str(DEFAULT_FULL_METADATA_CANONICAL_JSONL)
                ),
                "--task2-source",
                str(TASK2_EXACT50_SIDE_CAPPED_PATH),
                "--task3-source",
                str(TASK3_EXACT50_SIDE_CAPPED_PATH),
                "--task23-results-dir",
                str(TASK23_EXACT50_SIDE_CAPPED_RESULT_DIR),
                "--dataset-suffix",
                "exact50_side_capped",
            ],
            env=child_env,
        )
        _run([sys.executable, "-m", "extended.ideology_bias.analyze_task2_exact50_side_capped"], env=child_env)
    if "task3_exact50_side_capped" in steps:
        _run(
            [
                sys.executable,
                "-m",
                "extended.ideology_bias.build_analysis_datasets",
                    "--metadata-jsonl",
                    args.metadata_input or (
                        str(DEFAULT_BOOTSTRAP_METADATA_JSONL)
                    if metadata_source == "bootstrap"
                    else str(DEFAULT_FULL_METADATA_CANONICAL_JSONL)
                ),
                "--task2-source",
                str(TASK2_EXACT50_SIDE_CAPPED_PATH),
                "--task3-source",
                str(TASK3_EXACT50_SIDE_CAPPED_PATH),
                "--task23-results-dir",
                str(TASK23_EXACT50_SIDE_CAPPED_RESULT_DIR),
                "--dataset-suffix",
                "exact50_side_capped",
            ],
            env=child_env,
        )
        _run([sys.executable, "-m", "extended.ideology_bias.analyze_task3_exact50_side_capped"], env=child_env)
    if "task2_exact50_all_pairs" in steps:
        _run(
            [
                sys.executable,
                "-m",
                "extended.ideology_bias.build_analysis_datasets",
                    "--metadata-jsonl",
                    args.metadata_input or (
                        str(DEFAULT_BOOTSTRAP_METADATA_JSONL)
                    if metadata_source == "bootstrap"
                    else str(DEFAULT_FULL_METADATA_CANONICAL_JSONL)
                ),
                "--task2-source",
                str(TASK2_EXACT50_ALL_PAIRS_PATH),
                "--task3-source",
                str(TASK3_EXACT50_ALL_PAIRS_PATH),
                "--task23-results-dir",
                str(TASK23_EXACT50_ALL_PAIRS_RESULT_DIR),
                "--dataset-suffix",
                "exact50_all_pairs",
            ],
            env=child_env,
        )
        _run([sys.executable, "-m", "extended.ideology_bias.analyze_task2_exact50_all_pairs"], env=child_env)
    if "task3_exact50_all_pairs" in steps:
        _run(
            [
                sys.executable,
                "-m",
                "extended.ideology_bias.build_analysis_datasets",
                    "--metadata-jsonl",
                    args.metadata_input or (
                        str(DEFAULT_BOOTSTRAP_METADATA_JSONL)
                    if metadata_source == "bootstrap"
                    else str(DEFAULT_FULL_METADATA_CANONICAL_JSONL)
                ),
                "--task2-source",
                str(TASK2_EXACT50_ALL_PAIRS_PATH),
                "--task3-source",
                str(TASK3_EXACT50_ALL_PAIRS_PATH),
                "--task23-results-dir",
                str(TASK23_EXACT50_ALL_PAIRS_RESULT_DIR),
                "--dataset-suffix",
                "exact50_all_pairs",
            ],
            env=child_env,
        )
        _run([sys.executable, "-m", "extended.ideology_bias.analyze_task3_exact50_all_pairs"], env=child_env)
    if "task2_jel_similarity_side_capped_jaccard05_shared2" in steps:
        _run(
            [
                sys.executable,
                "-m",
                "extended.ideology_bias.build_analysis_datasets",
                "--metadata-jsonl",
                args.metadata_input or (
                    str(DEFAULT_BOOTSTRAP_METADATA_JSONL)
                    if metadata_source == "bootstrap"
                    else str(DEFAULT_FULL_METADATA_CANONICAL_JSONL)
                ),
                *(
                    ["--task1-results-dir", args.task1_results_dir]
                    if args.task1_results_dir
                    else []
                ),
                "--task2-source",
                args.task2_source or str(TASK2_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_PATH),
                "--task3-source",
                args.task3_source or str(TASK3_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_PATH),
                "--task23-results-dir",
                args.task23_results_dir or str(TASK23_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_RESULT_DIR),
                "--dataset-suffix",
                args.dataset_suffix or "jel_similarity_side_capped_jaccard05_shared2",
                "--skip-task1-output",
            ],
            env=child_env,
        )
        _run([sys.executable, "-m", "extended.ideology_bias.analyze_task2_jel_similarity_side_capped_jaccard05_shared2"], env=child_env)
        _publish_outputs(
            "task23_jel_similarity_side_capped_jaccard05_shared2_analysis",
            [
                "*jel_similarity_side_capped_jaccard05_shared2*",
            ],
            base_output_dir=output_dir_override,
        )
    if "task3_jel_similarity_side_capped_jaccard05_shared2" in steps:
        _run(
            [
                sys.executable,
                "-m",
                "extended.ideology_bias.build_analysis_datasets",
                "--metadata-jsonl",
                args.metadata_input or (
                    str(DEFAULT_BOOTSTRAP_METADATA_JSONL)
                    if metadata_source == "bootstrap"
                    else str(DEFAULT_FULL_METADATA_CANONICAL_JSONL)
                ),
                *(
                    ["--task1-results-dir", args.task1_results_dir]
                    if args.task1_results_dir
                    else []
                ),
                "--task2-source",
                args.task2_source or str(TASK2_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_PATH),
                "--task3-source",
                args.task3_source or str(TASK3_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_PATH),
                "--task23-results-dir",
                args.task23_results_dir or str(TASK23_JEL_SIMILARITY_SIDE_CAPPED_JACCARD05_SHARED2_RESULT_DIR),
                "--dataset-suffix",
                args.dataset_suffix or "jel_similarity_side_capped_jaccard05_shared2",
                "--skip-task1-output",
            ],
            env=child_env,
        )
        _run([sys.executable, "-m", "extended.ideology_bias.analyze_task3_jel_similarity_side_capped_jaccard05_shared2"], env=child_env)
        _publish_outputs(
            "task23_jel_similarity_side_capped_jaccard05_shared2_analysis",
            [
                "*jel_similarity_side_capped_jaccard05_shared2*",
            ],
            base_output_dir=output_dir_override,
        )

    reasoning_input = None
    if "annotate_reasoning" in steps:
        reasoning_cmd = [
            sys.executable,
            "-m",
            "extended.ideology_bias.annotate_reasoning_frames",
            "--task",
            args.reasoning_task,
            "--method",
            args.reasoning_method,
        ]
        if args.reasoning_limit is not None:
            reasoning_cmd.extend(["--limit", str(args.reasoning_limit)])
        _run(reasoning_cmd, env=child_env)
        if args.reasoning_method == "heuristic":
            reasoning_input = "extended/ideology_bias_outputs/metadata/reasoning_frames_heuristic_canonical.jsonl"
        else:
            reasoning_input = "extended/ideology_bias_outputs/metadata/reasoning_frames_canonical.jsonl"

    if "analyze_reasoning" in steps:
        reasoning_cmd = [sys.executable, "-m", "extended.ideology_bias.analyze_reasoning_frames"]
        if reasoning_input:
            reasoning_cmd.extend(["--input", reasoning_input])
        elif args.reasoning_method == "heuristic":
            reasoning_cmd.extend(["--input", "extended/ideology_bias_outputs/metadata/reasoning_frames_heuristic_canonical.jsonl"])
        _run(reasoning_cmd, env=child_env)


if __name__ == "__main__":
    main()
