import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "trace_profile_ab.py"
spec = importlib.util.spec_from_file_location("trace_profile_ab", path)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


class PairedTimingTests(unittest.TestCase):
    def test_known_ratio_and_inverse(self):
        a, b = [100, 200, 400, 800], [80, 160, 320, 640]
        result = ab.paired_result(a, b, 200)
        self.assertAlmostEqual(result["paired_reduction_percent"], 20)
        self.assertAlmostEqual(result["ci95_percent"][0], 20)
        self.assertAlmostEqual(result["ci95_percent"][1], 20)
        self.assertAlmostEqual(
            ab.paired_result(b, a, 200)["paired_reduction_percent"], -25)

    def test_identity(self):
        result = ab.paired_result([10, 20, 30], [10, 20, 30], 200)
        self.assertEqual(result["paired_reduction_percent"], 0)
        self.assertEqual(result["ci95_percent"], [0, 0])

    def test_invalid_inputs(self):
        for a, b in [([1], [1]), ([1, 2], [1]), ([0, 1], [1, 2]),
                     ([float("nan"), 2], [1, 2]),
                     ([float("inf"), 2], [1, 2])]:
            with self.assertRaises(ValueError):
                ab.paired_result(a, b, 200)


class ConfigTests(unittest.TestCase):
    def test_flags(self):
        self.assertEqual(
            ab.config_flags({"b_flag": True, "a_mode": "fast", "c": 3}),
            ["--a_mode=fast", "--b_flag=true", "--c=3"])

    def test_rejects_invalid_names_and_values(self):
        for config in ({"Bad": 1}, {"a=b": 1}, {"ok": [1]}):
            with self.assertRaises(ValueError):
                ab.config_flags(config)


class ParseCorpusTests(unittest.TestCase):
    def test_entries_comments_and_flags(self):
        entries = ab.parse_corpus(
            "# name frame last\n"
            "\n"
            "4D53085B_stream 13 1592 --occlusion_query=fast-alt\n"
            "4D5307E6_stream 875 2671  # Halo 3\n")
        self.assertEqual(entries, [
            {"name": "4D53085B_stream", "frame": 13, "last_command": 1592,
             "args": ["--occlusion_query=fast-alt"]},
            {"name": "4D5307E6_stream", "frame": 875, "last_command": 2671,
             "args": []},
        ])

    def test_incomplete_line(self):
        with self.assertRaises(ValueError):
            ab.parse_corpus("4D53085B_stream 13\n")


if __name__ == "__main__":
    unittest.main()
