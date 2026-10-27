"""The hierarchical depth suite's cases, part 2 of 7 (hiz.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hiz  # noqa: E402


def build(suite):
    hiz.build(suite, 2)
