"""The hierarchical depth suite hiz8's part a (hiz8.py): which written blocks
a later draw sees."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hiz8  # noqa: E402


def build(suite):
    hiz8.build(suite, 'a')
