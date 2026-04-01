"""Generic helpers shared across ideology bias modules."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from .schemas import VALID_SIGNS, normalize_sign

KNOWN_FAMILY_ORDER = ["openai", "claude", "gemini", "grok", "llama", "qwen", "deepseek", "hf_endpoint", "unknown"]

MODEL_ORDER_WITHIN_FAMILY = {
    "claude": [
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ],
    "grok": [
        "grok-3-mini",
        "grok-3",
        "grok-4-1-fast-reasoning",
    ],
    "llama": [
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.2-1b-instruct",
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5.2",
    ],
    "qwen": [
        "qwen/qwen3-8b",
        "qwen/qwen3-14b",
        "qwen/qwen3-32b",
    ],
    "gemini": [
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
    ],
}


def normalize_text(value: object) -> str:
    """Normalize text for deterministic matching."""
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def make_triplet_key(paper_id: object, treatment: object, outcome: object) -> str:
    """Create a stable triplet key."""
    return "|".join(
        [
            str(paper_id).strip(),
            normalize_text(treatment),
            normalize_text(outcome),
        ]
    )


def make_triplet_uid(
    paper_id: object,
    treatment: object,
    outcome: object,
    context: object,
) -> str:
    """Create a stable content hash for a triplet plus context."""
    raw = "\u241f".join(
        [
            str(paper_id).strip(),
            normalize_text(treatment),
            normalize_text(outcome),
            normalize_text(context),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def safe_slug(value: object) -> str:
    """Convert a string into a filename-safe slug."""
    if value is None:
        return "unknown"
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug or "unknown"


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    """Yield parsed JSON objects from a JSONL file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into memory."""
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write dictionaries as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Append dictionaries as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv_rows(path: str | Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write a list of dictionaries as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: set[str] = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_list(value: object) -> list[str]:
    """Normalize a scalar or list into a unique list of strings."""
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    cleaned: list[str] = []
    for item in raw_items:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text.lower() == "nan":
            continue
        cleaned.append(text)
    return list(dict.fromkeys(cleaned))


def parse_json_maybe(value: object, default: object = None) -> object:
    """Parse a stringified JSON payload when possible."""
    if value is None:
        return [] if default is None else default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return [] if default is None else default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [] if default is None else default


def parse_example_details(value: object) -> list[dict]:
    """Parse the raw `example_details` string stored in released ICL source files."""
    parsed = parse_json_maybe(value, default=[])
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def parse_model_meta(family: str, model: str) -> dict:
    """Infer parameter-bucket and tier labels from model names."""
    lowered = model.lower()
    match = re.search(r"(\d+(?:\.\d+)?)b", lowered)
    parameter_billion = float(match.group(1)) if match else None
    if parameter_billion is None:
        parameter_bucket = "unknown"
    elif parameter_billion <= 2:
        parameter_bucket = "<=2B"
    elif parameter_billion <= 10:
        parameter_bucket = "2-10B"
    elif parameter_billion <= 40:
        parameter_bucket = "10-40B"
    else:
        parameter_bucket = ">40B"

    if "nano" in lowered:
        tier = "nano"
    elif "mini" in lowered:
        tier = "mini"
    elif "flash" in lowered:
        tier = "flash"
    elif "pro" in lowered:
        tier = "pro"
    elif "reasoning" in lowered:
        tier = "reasoning"
    else:
        tier = "standard"
    return {
        "family": family,
        "model": model,
        "parameter_billion": parameter_billion,
        "parameter_bucket": parameter_bucket,
        "tier": tier,
    }


def infer_family_from_model(model: object) -> str:
    """Infer family from a raw model name."""
    lowered = str(model or "").lower()
    if lowered.startswith("gpt-"):
        return "openai"
    if lowered.startswith("claude-"):
        return "claude"
    if lowered.startswith("gemini-"):
        return "gemini"
    if lowered.startswith("grok-"):
        return "grok"
    if "llama" in lowered:
        return "llama"
    if "qwen" in lowered:
        return "qwen"
    if "deepseek" in lowered:
        return "deepseek"
    if lowered.startswith("hf_"):
        return "hf_endpoint"
    return "unknown"


def family_sort_key(family: object) -> tuple[int, str]:
    """Return a canonical family ordering key."""
    family_str = str(family or "unknown")
    try:
        return (KNOWN_FAMILY_ORDER.index(family_str), family_str)
    except ValueError:
        return (len(KNOWN_FAMILY_ORDER), family_str)


def model_sort_key(model: object, family: object | None = None) -> tuple[int, int, str]:
    """Return a canonical model ordering key within a family."""
    model_str = str(model or "")
    family_str = str(family or infer_family_from_model(model_str))
    family_models = MODEL_ORDER_WITHIN_FAMILY.get(family_str, [])
    try:
        rank = family_models.index(model_str)
    except ValueError:
        rank = len(family_models)
    return (
        family_sort_key(family_str)[0],
        rank,
        model_str,
    )


def stringify_for_csv(value: object) -> object:
    """Serialize lists/dicts for CSV output while keeping scalars intact."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def load_done_keys(path: str | Path, key_field: str = "triplet_key") -> set[str]:
    """Load completed keys from an append-only JSONL output file."""
    path = Path(path)
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in iter_jsonl(path):
        key = row.get(key_field)
        if key:
            done.add(str(key))
    return done


def ensure_sign(value: object, default: str | None = None) -> str | None:
    """Normalize a raw sign, falling back to a default when needed."""
    normalized = normalize_sign(value)
    return normalized if normalized in VALID_SIGNS else default
