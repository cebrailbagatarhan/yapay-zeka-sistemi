"""Dependency-free tests for offline train/evaluation leakage auditing."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_dataset_overlap import audit, load_records, main


class DatasetOverlapAuditTests(unittest.TestCase):
    def test_normalizes_turkish_prompt_and_ignores_assistant_answer(self):
        train = [{"instruction": "Merhaba   DÜNYA", "response": "answer one"}]
        evaluation = [{"messages": [
            {"role": "user", "content": "merhaba dünya"},
            {"role": "assistant", "content": "different answer"},
        ]}]
        result = audit(train, evaluation)
        self.assertEqual(result["overlap_unique_prompts"], 1)
        self.assertEqual(result["examples"], [{"evaluation_row": 1, "train_rows": [1]}])
        self.assertNotIn("Merhaba", json.dumps(result))

    def test_duplicate_counts_and_no_overlap(self):
        result = audit([{"prompt": "A"}, {"prompt": "a"}], [{"instruction": "B"}])
        self.assertEqual(result["train_duplicate_rows"], 1)
        self.assertEqual(result["overlap_unique_prompts"], 0)

    def test_load_json_and_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_file = root / "train.json"
            jsonl_file = root / "eval.jsonl"
            json_file.write_text('[{"instruction": "hi"}]', encoding="utf-8")
            jsonl_file.write_text('{"prompt": "hi"}\n\n', encoding="utf-8")
            self.assertEqual(len(load_records(json_file)), 1)
            self.assertEqual(len(load_records(jsonl_file)), 1)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(json_file), str(jsonl_file)]), 1)

    def test_clean_split_exits_successfully(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.json"
            evaluation = root / "eval.json"
            train.write_text('[{"instruction": "first"}]', encoding="utf-8")
            evaluation.write_text('[{"instruction": "second"}]', encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(train), str(evaluation)]), 0)

    def test_invalid_record_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "no non-empty"):
            audit([{"response": "only answer"}], [{"instruction": "question"}])

    def test_bad_json_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"instruction": "not a list"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "list of JSON objects"):
                load_records(path)


if __name__ == "__main__":
    unittest.main()
