"""The hierarchical depth suite hiz8's part b (hiz8.py): what the buffer
holds, how it compares, and what clears leave."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hiz8  # noqa: E402


def build(suite):
    hiz8.build(suite, 'b')
