from __future__ import annotations

import argparse
import unittest

from agent_gpu_broker.cli import parse_duration


class DurationTests(unittest.TestCase):
    def test_suffixes(self):
        self.assertEqual(parse_duration("30s"), 30)
        self.assertEqual(parse_duration("2m"), 120)
        self.assertEqual(parse_duration("1.5h"), 5400)

    def test_rejects_non_positive(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_duration("0")


if __name__ == "__main__":
    unittest.main()
