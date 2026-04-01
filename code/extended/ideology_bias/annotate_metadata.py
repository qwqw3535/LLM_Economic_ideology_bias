"""LLM-based ideology metadata extraction for the full triplet corpus."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .llm import OpenAIJsonClient
from .load_sources import load_source_file
from .metadata_prompt_v2 import METADATA_PROMPT_TEMPLATE
from .paths import DEFAULT_FULL_METADATA_RAW_PATH, ensure_output_dirs
from .schemas import METADATA_RESPONSE_SCHEMA, validate_metadata_payload
from .utils import append_jsonl, load_done_keys


logger = logging.getLogger(__name__)


def _build_prompt(row: dict) -> str:
    return METADATA_PROMPT_TEMPLATE.format(
        title=row.get("title") or "",
        context=row.get("context") or "",
        treatment=row.get("treatment") or "",
        outcome=row.get("outcome") or "",
    )


def _annotate_one(client: OpenAIJsonClient, row: dict) -> dict:
    result = {
        "triplet_key": row["triplet_key"],
        "triplet_uid": row.get("triplet_uid"),
        "paper_id": row.get("paper_id"),
        "title": row.get("title"),
        "published_venue": row.get("published_venue"),
        "publication_year": row.get("publication_year"),
        "treatment": row.get("treatment"),
        "outcome": row.get("outcome"),
        "sign": row.get("sign"),
        "context": row.get("context"),
        "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": client.model,
    }
    call = client.call_json_schema(_build_prompt(row), METADATA_RESPONSE_SCHEMA)
    if not call.success:
        result["annotation_error"] = call.error
        return result
    result["metadata_annotation"] = call.payload
    result["validation_errors"] = validate_metadata_payload(call.payload)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate causal triplets with ideology metadata using an LLM.")
    parser.add_argument("--input", required=True, help="Input JSONL path, usually data/causal_triplets.jsonl")
    parser.add_argument("--output", default=str(DEFAULT_FULL_METADATA_RAW_PATH), help="Append-only raw JSONL output path")
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model name")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the OpenAI API key")
    parser.add_argument("--max-workers", type=int, default=16, help="Maximum parallel requests")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per request")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for testing")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    ensure_output_dirs()
    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_source_file(input_path, source_name=input_path.stem)
    done_keys = load_done_keys(output_path)
    pending = [row for row in rows if row["triplet_key"] not in done_keys]
    if args.limit is not None:
        pending = pending[: args.limit]

    logger.info("loaded=%s pending=%s output=%s", len(rows), len(pending), output_path)
    if not pending:
        logger.info("nothing to annotate")
        return

    client = OpenAIJsonClient(
        model=args.model,
        api_key_env=args.api_key_env,
        max_retries=args.max_retries,
    )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_annotate_one, client, row): row["triplet_key"] for row in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="annotate_metadata"):
            row = future.result()
            append_jsonl(output_path, [row])

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "n_total": len(rows),
        "n_done_before": len(done_keys),
        "n_processed_now": len(pending),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

