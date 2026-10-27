"""Read-only structural audit of the four bundled training JSON datasets."""

import argparse
import json
from pathlib import Path

# Filename -> (input/prompt field, required response fields).
SCHEMAS = {
    "code_examples_dataset.json": ("problem", ("solution",)),
    "conversational_dataset.json": ("question", ("response",)),
    "reasoning_chat_dataset.json": ("question", ("reasoning", "answer")),
    "turkish_general_dataset.json": ("instruction", ("output",)),
}


def audit_directory(directory: Path) -> dict:
    """Return counts and field-level issues without exposing dataset text."""
    report = {"records": 0, "files": [], "errors": 0}
    for filename, (prompt_field, response_fields) in SCHEMAS.items():
        entry = {"file": filename, "records": 0, "issues": []}
        report["files"].append(entry)
        try:
            with (directory / filename).open("r", encoding="utf-8") as handle:
                records = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            entry["issues"].append(f"Cannot read valid UTF-8 JSON ({type(exc).__name__})")
            report["errors"] += 1
            continue

        if not isinstance(records, list) or not records:
            entry["issues"].append("Expected a nonempty JSON array")
            report["errors"] += 1
            continue

        entry["records"] = len(records)
        report["records"] += len(records)
        seen = set()
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                entry["issues"].append(f"Record {index}: expected an object")
                continue
            for field in (prompt_field, *response_fields):
                value = record.get(field)
                if not isinstance(value, str) or not value.strip():
                    entry["issues"].append(f"Record {index}: missing/nonempty string required: {field}")
            prompt = record.get(prompt_field)
            if isinstance(prompt, str) and prompt.strip():
                normalized = " ".join(prompt.split()).casefold()
                if normalized in seen:
                    entry["issues"].append(f"Record {index}: duplicate {prompt_field}")
                seen.add(normalized)
            if filename == "turkish_general_dataset.json" and "input" in record and not isinstance(record["input"], str):
                entry["issues"].append(f"Record {index}: input must be a string")
        report["errors"] += len(entry["issues"])
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "training")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary (never dataset content)")
    args = parser.parse_args(argv)
    report = audit_directory(args.directory)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Audited {len(report['files'])} files, {report['records']} records, {report['errors']} issues")
        for entry in report["files"]:
            for issue in entry["issues"]:
                print(f"{entry['file']}: {issue}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
