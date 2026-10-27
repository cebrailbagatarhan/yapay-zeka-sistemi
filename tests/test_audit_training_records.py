"""Regression tests for the bundled training dataset preflight."""

import json
import tempfile
import unittest
from pathlib import Path
from tools.audit_training_records import SCHEMAS, audit_directory, main


class TrainingRecordAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        for filename, (prompt, answers) in SCHEMAS.items():
            record = {prompt: "sample prompt", **{field: "sample answer" for field in answers}}
            (self.directory / filename).write_text(json.dumps([record]), encoding="utf-8")

    def put(self, filename, data):
        (self.directory / filename).write_text(json.dumps(data), encoding="utf-8")

    def test_valid_datasets(self):
        report = audit_directory(self.directory)
        self.assertEqual((report["records"], report["errors"]), (4, 0))
        self.assertEqual(main(["--directory", str(self.directory), "--json"]), 0)

    def test_missing_file_is_reported_without_content(self):
        (self.directory / "code_examples_dataset.json").unlink()
        report = audit_directory(self.directory)
        self.assertEqual(report["errors"], 1)
        self.assertIn("FileNotFoundError", report["files"][0]["issues"][0])

    def test_malformed_json(self):
        (self.directory / "conversational_dataset.json").write_text("{broken", encoding="utf-8")
        self.assertEqual(audit_directory(self.directory)["errors"], 1)

    def test_empty_array_and_wrong_root(self):
        self.put("code_examples_dataset.json", [])
        self.put("conversational_dataset.json", {})
        self.assertEqual(audit_directory(self.directory)["errors"], 2)

    def test_missing_fields_and_blank_answers(self):
        self.put("reasoning_chat_dataset.json", [{"question": "q", "reasoning": "  "}])
        issues = audit_directory(self.directory)["files"][2]["issues"]
        self.assertEqual(len(issues), 2)
        self.assertTrue(any("answer" in issue for issue in issues))

    def test_non_object_record(self):
        self.put("code_examples_dataset.json", [42])
        self.assertEqual(audit_directory(self.directory)["errors"], 1)

    def test_duplicate_prompt_is_case_and_whitespace_insensitive(self):
        self.put("conversational_dataset.json", [
            {"question": "  HELLO  world ", "response": "one"},
            {"question": "hello world", "response": "two"},
        ])
        self.assertEqual(audit_directory(self.directory)["errors"], 1)

    def test_optional_input_must_be_text(self):
        self.put("turkish_general_dataset.json", [{"instruction": "q", "output": "a", "input": 2}])
        self.assertEqual(audit_directory(self.directory)["errors"], 1)

    def test_nonempty_exit_status(self):
        self.put("code_examples_dataset.json", [None])
        self.assertEqual(main(["--directory", str(self.directory)]), 1)


if __name__ == "__main__":
    unittest.main()
