"""Standard-library tests for the offline preset memory estimator."""

import contextlib
import io
import json
import unittest

from modern_llm.config import PRESET_CONFIGS
from tools.preset_memory import estimate_preset, main


class PresetMemoryTests(unittest.TestCase):
    def test_every_preset_matches_config_parameter_count(self):
        for name, config in PRESET_CONFIGS.items():
            with self.subTest(preset=name):
                row = estimate_preset(name)
                self.assertEqual(row["parameters"], config.estimated_params)
                self.assertEqual(row["weight_bytes"], config.estimated_params * 2)

    def test_fp32_doubles_bf16_weight_bytes(self):
        self.assertEqual(
            estimate_preset("nano", "fp32")["weight_bytes"],
            2 * estimate_preset("nano", "bf16")["weight_bytes"],
        )

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            estimate_preset("missing")
        with self.assertRaises(ValueError):
            estimate_preset("nano", "int4")

    def test_json_cli_reports_weights_only(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--preset", "small", "--json"]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["scope"], "weights_only")
        self.assertEqual([r["preset"] for r in result["estimates"]], ["small"])

    def test_budget_filters_using_exact_bytes_not_rounded_gib(self):
        nano_bytes = estimate_preset("nano")["weight_bytes"]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--max-weight-gib", str(nano_bytes / 1024 ** 3), "--json"]), 0)
        self.assertEqual([r["preset"] for r in json.loads(output.getvalue())["estimates"]], ["nano"])

    def test_tiny_budget_returns_empty_json_list(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--max-weight-gib", "0.00001", "--json"]), 0)
        self.assertEqual(json.loads(output.getvalue())["estimates"], [])

    def test_invalid_budgets_rejected(self):
        for invalid in ("0", "-1", "nan", "inf", "not-a-number"):
            with self.subTest(value=invalid), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["--max-weight-gib", invalid])
                self.assertEqual(raised.exception.code, 2)

    def test_all_presets_are_reported(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--json"]), 0)
        self.assertEqual(
            len(json.loads(output.getvalue())["estimates"]), len(PRESET_CONFIGS)
        )


if __name__ == "__main__":
    unittest.main()
