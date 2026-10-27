"""The stencil2x operation, compare and two-sided cases at all four samples."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthcases  # noqa: E402
import stencil2x  # noqa: E402


def build(suite):
    depthcases.build(suite, stencil2x.cases(), samples=4)
