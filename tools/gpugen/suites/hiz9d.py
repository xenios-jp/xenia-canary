"""The hierarchical depth suite hiz9's part d (hiz9.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hiz9  # noqa: E402


def build(suite):
    hiz9.build(suite, 'd')
