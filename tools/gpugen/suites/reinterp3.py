"""The EDRAM reinterpretation suite's cases, part 3 of 3 (reinterp.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reinterp  # noqa: E402


def build(suite):
    reinterp.build(suite, 3)
