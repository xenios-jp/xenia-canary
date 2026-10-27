"""The render targets and resolves games use, part 1 of 4 (targets.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import targets  # noqa: E402


def build(suite):
    targets.build(suite, 1)
