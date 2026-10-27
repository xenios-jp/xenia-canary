"""Predicated draws with the sample counters reset, part 2 of 2 (vizdet.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vizdet  # noqa: E402


def build(suite):
    vizdet.build(suite, 2)
