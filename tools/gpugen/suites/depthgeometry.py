"""Depth-geometry cases through depthcases: steep Z planes drawn as the other
split, rotated or back-facing triangles, one triangle twice the size, or
through a shifted viewport, each read back whole.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthcases  # noqa: E402


def cases():
    return [(i, g, c) for i, (g, c) in enumerate(depthcases.fourth())]


def build(suite):
    depthcases.build(suite, cases())
