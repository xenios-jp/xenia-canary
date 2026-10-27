"""Depth-slope cases through depthcases: Z planes whose slopes are exact
binary fractions, read back whole, and polygon offsets wrapping past +-8 or
finer than a depth unit.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthcases  # noqa: E402


def cases():
    return [(i, g, c) for i, (g, c) in enumerate(depthcases.third())]


def build(suite):
    depthcases.build(suite, cases())
