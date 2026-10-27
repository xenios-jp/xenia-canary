"""The hierarchical depth suite hiz8's part d (hiz8.py): Metro 2033's
shadow atlas draws predicated on VIZ queries over hierarchical depth."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hiz8  # noqa: E402


def build(suite):
    hiz8.build(suite, 'd')
