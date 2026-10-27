"""Depth-bias cases through depthcases: polygon offsets on Z and oDepth from
one depth step to past the depth range.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthcases  # noqa: E402


def cases():
    return [(i, g, c) for i, (g, c) in enumerate(depthcases.second())]


def build(suite):
    depthcases.build(suite, cases())
