"""Offline prompt-overlap audit for JSON and JSONL training/evaluation datasets."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array or JSONL file without importing training dependencies."""
    with path.open(encoding="utf-8") as stream:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            records = json.load(stream)
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError(f"{path}: expected a list of JSON objects")
    return records


def prompt_of(record: dict[str, Any]) -> str:
    """Use the user-side prompt only; never compare model-generated answers."""
    prompt = record.get("instruction", record.get("prompt"))
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    messages = record.get("messages")
    if isinstance(messages, list):
        user_parts = [
            message.get("content")
            for message in messages
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ]
        if user_parts:
            return "\n".join(user_parts)
    raise ValueError("record has no non-empty instruction, prompt, or user message")


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def audit(train: list[dict[str, Any]], evaluation: list[dict[str, Any]]) -> dict[str, Any]:
    train_keys = [normalize_prompt(prompt_of(record)) for record in train]
    eval_keys = [normalize_prompt(prompt_of(record)) for record in evaluation]
    if not all(train_keys + eval_keys):
        raise ValueError("empty prompts are not supported")
    train_positions: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(train_keys, 1):
        train_positions[key].append(index)
    overlap_keys = set(train_keys) & set(eval_keys)
    return {
        "train_rows": len(train),
        "evaluation_rows": len(evaluation),
        "train_duplicate_rows": sum(count - 1 for count in Counter(train_keys).values()),
        "evaluation_duplicate_rows": sum(count - 1 for count in Counter(eval_keys).values()),
        "overlap_unique_prompts": len(overlap_keys),
        "overlap_evaluation_rows": sum(key in overlap_keys for key in eval_keys),
        "examples": [
            {"evaluation_row": index, "train_rows": train_positions[key][:10]}
            for index, key in enumerate(eval_keys, 1)
            if key in overlap_keys
        ][:10],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect identical prompts across train/eval JSON or JSONL datasets")
    parser.add_argument("train", type=Path, help="Training dataset (JSON array or JSONL)")
    parser.add_argument("evaluation", type=Path, help="Evaluation dataset (JSON array or JSONL)")
    args = parser.parse_args(argv)
    try:
        result = audit(load_records(args.train), load_records(args.evaluation))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Dataset audit failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["overlap_unique_prompts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
