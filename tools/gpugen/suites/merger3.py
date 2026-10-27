"""The output merger suite's cases, part 3 of 3 (merger.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merger  # noqa: E402


def build(suite):
    merger.build(suite, 3)
