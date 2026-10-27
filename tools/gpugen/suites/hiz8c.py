"""The hierarchical depth suite hiz8's part c (hiz8.py): where a block
lives, by sample count and hiz_pitch."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hiz8  # noqa: E402


def build(suite):
    hiz8.build(suite, 'c')
