"""Predicated draws, part 2 of 2 (vizpred.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vizpred  # noqa: E402


def build(suite):
    vizpred.build(suite, 2)
