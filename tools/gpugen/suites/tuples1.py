"""The state tuple suite's cases, part 1 of 3 (tuples.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tuples  # noqa: E402


def build(suite):
    tuples.build(suite, 1)
